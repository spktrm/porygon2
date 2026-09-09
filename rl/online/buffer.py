import threading

import jax
import numpy as np
from tqdm import tqdm

from rl.environment.data import NUM_ABILITIES, NUM_ITEMS, NUM_MOVES, NUM_SPECIES
from rl.environment.interfaces import (
    BuilderHistoryOutput,
    BuilderTransition,
    Trajectory,
)
from rl.environment.protos.features_pb2 import PackedSetFeature
from rl.environment.utils import acted_rows, next_tqdm_position


class BuilderTrajectoryStore:
    """Stores builder trajectories for later use by the learner."""

    def __init__(self, max_size: int = 1000, max_reuses: int = 5, name: str = ""):
        self._trajectories: dict[
            int, tuple[BuilderTransition, BuilderHistoryOutput]
        ] = {}
        self._reuses = np.zeros(max_size, dtype=int)
        self._valid = np.zeros(max_size, dtype=bool)

        self._max_size = max_size
        self._max_reuses = max_reuses

        # Both conditions share one lock: add and sample mutate the same
        # arrays/dict, so they must be mutually exclusive — separate locks
        # would only serialize adders against adders and samplers against
        # samplers. Callers never nest the two conditions, and the RLock
        # keeps notify-while-holding-the-sibling-condition legal.
        lock = threading.RLock()
        self._add_cv = threading.Condition(lock)
        self._sample_cv = threading.Condition(lock)

        desc = f"builder_producer-{name}" if name else "builder_producer"
        self._progress = tqdm(desc=desc, smoothing=0.1, position=next_tqdm_position())

    @classmethod
    def is_full(self, limit: int = None) -> bool:
        """Returns True if the store has reached its maximum capacity."""
        if limit is None:
            limit = self._max_size
        return len(self._trajectories) >= limit

    def nbytes(self) -> int:
        """Total host bytes of stored trajectory arrays — RAM diagnostics
        (Learner._log_memory_diagnostics)."""
        with self._sample_cv:
            return sum(
                leaf.nbytes
                for item in self._trajectories.values()
                for leaf in jax.tree.leaves(item)
                if hasattr(leaf, "nbytes")
            )

    def ready_to_sample(self) -> bool:
        """Returns True if there is at least one trajectory that can be sampled."""
        return np.any((self._reuses < self._max_reuses) & self._valid)

    def ready_to_add(self) -> bool:
        """True when there is a free slot OR an over-reused one to evict."""
        return len(self._trajectories) < self._max_size or np.any(
            self._reuses >= self._max_reuses
        )

    def set_max_reuses(self, max_reuses: int):
        """Thread-safe update of the per-trajectory reuse cap. See
        PlayerTrajectoryStore.set_max_reuses — mirrored here so a caller
        reusing one persistent store across phases (main.py) can reapply
        each phase's own config value without reaching into a private
        attribute."""
        with self._add_cv:
            self._max_reuses = int(max_reuses)
            self._add_cv.notify_all()
            self._sample_cv.notify_all()

    def clear(self):
        """Resets the store to empty.

        Used when reusing one persistent store instead of allocating a
        fresh one per lineage — a fresh store meant an actor thread that
        outlived the old one (see main.py's straggler check) could keep
        writing into a store that had already "ended," silently leaking
        trajectories from the wrong model into whatever ran next. Only
        safe to call once every actor thread from the previous phase has
        actually stopped.
        """
        with self._add_cv:
            self._trajectories = {}
            self._reuses = np.zeros(self._max_size, dtype=int)
            self._valid = np.zeros(self._max_size, dtype=bool)

    def add_trajectory(
        self, trajectory: BuilderTransition, history: BuilderHistoryOutput
    ):
        """
        adds a trajectory only if there is capacity
        if not capacity, check if any trajectories have been reused more than max_reuses, if so, remove them and add the new trajectory
        """
        item_to_store = (trajectory, history)

        if len(self._trajectories) < self._max_size:
            current_index = len(self._trajectories)
            self._trajectories[current_index] = item_to_store
            self._reuses[current_index] = 0
            self._valid[current_index] = True
        else:
            available_indices = np.where(self._reuses >= self._max_reuses)[0]
            if len(available_indices) == 0:
                tqdm.write(
                    "Trajectory store is full and no trajectories are available for replacement."
                )
                return
            replace_index = np.random.choice(available_indices)
            self._trajectories[replace_index] = item_to_store
            self._reuses[replace_index] = 0

        self._progress.update(1)

    def sample_trajectory(
        self, increment: bool = True
    ) -> tuple[BuilderTransition, BuilderHistoryOutput]:
        """samples a trajectory uniformly from those with less than max_reuses, and increments its reuse count"""

        valid_indices = (self._reuses < self._max_reuses) & self._valid
        available_indices = np.where(valid_indices)[0]

        sample_index = np.random.choice(available_indices).item()
        if increment:
            self._reuses[sample_index] += 1
        return self._trajectories[sample_index]


def calculate_tracking(old: np.ndarray, new: np.ndarray, tau: float, minlength: int):
    return (1 - tau) * old + tau * np.bincount(new.reshape(-1), minlength=minlength)


class PlayerTrajectoryStore:
    """Stores player trajectories for later use by the learner.

    Unique-chunk sampling under a global reuse cap. An optional first-use
    stream reserves fresh slots across batches and permits early eviction
    of seen chunks.
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_reuses: int = 5,
        need_tracking: bool = False,
        name: str = "",
        fresh_fraction: float = 0.0,
    ):
        if not np.isfinite(fresh_fraction) or not 0 <= fresh_fraction <= 1:
            raise ValueError("fresh_fraction must be finite and in [0, 1]")
        self.fresh_fraction = float(fresh_fraction)
        self._next_id = 1
        self._ids = np.zeros(max_size, dtype=np.uint32)
        self._trajectories: dict[int, Trajectory] = {}
        self._reuses = np.zeros(max_size, dtype=int)
        self._valid = np.zeros(max_size, dtype=bool)

        self._max_size = max_size
        self._max_reuses = max_reuses

        # Single lock behind both conditions — see BuilderTrajectoryStore.
        lock = threading.RLock()
        self._add_cv = threading.Condition(lock)
        self._sample_cv = threading.Condition(lock)

        self._decision_counts = np.zeros(max_size, dtype=np.int64)
        self._reset_decision_accounting()

        desc = f"player_producer-{name}" if name else "player_producer"
        self._progress = tqdm(desc=desc, smoothing=0.1, position=next_tqdm_position())

        self.need_tracking = need_tracking
        if need_tracking:
            self._species_counts = np.zeros(NUM_SPECIES, dtype=np.float32)
            self._item_counts = np.zeros(NUM_ITEMS, dtype=np.float32)
            self._ability_counts = np.zeros(NUM_ABILITIES, dtype=np.float32)
            self._move_counts = np.zeros(NUM_MOVES, dtype=np.float32)
            self._tau = 1e-3

    def is_full(self, limit: int = None) -> bool:
        """Returns True if the store has reached its maximum capacity."""
        if limit is None:
            limit = self._max_size
        return len(self._trajectories) >= limit

    def nbytes(self) -> int:
        """Total host bytes of stored trajectory arrays — RAM diagnostics
        (Learner._log_memory_diagnostics)."""
        with self._sample_cv:
            return sum(
                leaf.nbytes
                for item in self._trajectories.values()
                for leaf in jax.tree.leaves(item)
                if hasattr(leaf, "nbytes")
            )

    def is_min_fill_fraction_reached(self, fraction: float = 0.5) -> bool:
        """Returns True if the store is at least ``fraction`` full.

        Args:
            fraction: Required fill level in [0.0, 1.0].
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0.0, 1.0], got {fraction}")
        return len(self._trajectories) >= int(self._max_size * fraction)

    def _eligible(self):
        return self._valid & (self._reuses < self._max_reuses)

    def _replaceable(self):
        return self._valid & (self._reuses >= self._max_reuses)

    def _fresh_required(self, batch_size):
        return int(
            np.ceil((self.total_samples + batch_size) * self.fresh_fraction)
            - np.ceil(self.total_samples * self.fresh_fraction)
        )

    def ready_to_sample(self, count: int = 1) -> bool:
        """True when at least `count` distinct chunks are eligible."""
        return bool(self._eligible().sum() >= count)

    def ready_to_add(self) -> bool:
        """True when there is a free slot or a replaceable occupant."""
        return len(self._trajectories) < self._max_size or np.any(self._replaceable())

    @property
    def max_reuses(self) -> int:
        return self._max_reuses

    def set_max_reuses(self, max_reuses: int):
        """Thread-safe update of the per-trajectory reuse cap (the replay
        ratio knob). Wakes both waiters: raising the cap can unblock
        samplers, lowering it can unblock adders."""
        with self._add_cv:
            self._max_reuses = int(max_reuses)
            self._add_cv.notify_all()
            self._sample_cv.notify_all()

    def clear(self):
        """Resets the store to empty — see BuilderTrajectoryStore.clear for
        why this exists (one persistent store reused across phase
        transitions, rather than a fresh one per phase)."""
        with self._add_cv:
            self._trajectories = {}
            self._reuses = np.zeros(self._max_size, dtype=int)
            self._valid = np.zeros(self._max_size, dtype=bool)
            self._ids.fill(0)
            self._reset_decision_accounting()
            if self.need_tracking:
                self.reset_usage_counts()

    def reset_usage_counts(self):
        # Called from the learner thread; takes the store lock so it can't
        # interleave with _update_usage_counts running inside add().
        with self._add_cv:
            self._species_counts = np.zeros(NUM_SPECIES, dtype=np.float32)
            self._item_counts = np.zeros(NUM_ITEMS, dtype=np.float32)
            self._ability_counts = np.zeros(NUM_ABILITIES, dtype=np.float32)
            self._move_counts = np.zeros(NUM_MOVES, dtype=np.float32)

    def _update_usage_counts(self, tokens: np.ndarray):
        """Updates EMA usage counts for species, items, abilities, and moves."""
        self._species_counts = calculate_tracking(
            self._species_counts,
            tokens[..., PackedSetFeature.PACKED_SET_FEATURE__SPECIES].reshape(-1),
            self._tau,
            NUM_SPECIES,
        )
        self._item_counts = calculate_tracking(
            self._item_counts,
            tokens[..., PackedSetFeature.PACKED_SET_FEATURE__ITEM].reshape(-1),
            self._tau,
            NUM_ITEMS,
        )
        self._ability_counts = calculate_tracking(
            self._ability_counts,
            tokens[..., PackedSetFeature.PACKED_SET_FEATURE__ABILITY].reshape(-1),
            self._tau,
            NUM_ABILITIES,
        )
        self._move_counts = calculate_tracking(
            self._move_counts,
            np.stack(
                [
                    tokens[..., PackedSetFeature.PACKED_SET_FEATURE__MOVE1],
                    tokens[..., PackedSetFeature.PACKED_SET_FEATURE__MOVE2],
                    tokens[..., PackedSetFeature.PACKED_SET_FEATURE__MOVE3],
                    tokens[..., PackedSetFeature.PACKED_SET_FEATURE__MOVE4],
                ],
                axis=-1,
            ).reshape(-1),
            self._tau,
            NUM_MOVES,
        )

    def add(self, traj: Trajectory):
        """Admit a chunk, preserving unseen occupants in fresh-stream mode."""
        if self._next_id > np.iinfo(np.uint32).max:
            raise OverflowError("replay chunk IDs exhausted; create a new store")
        if self.need_tracking:
            self._update_usage_counts(traj.builder_history.packed_team_member_tokens)

        if len(self._trajectories) < self._max_size:
            current_index = len(self._trajectories)
            self._trajectories[current_index] = traj
            self._reuses[current_index] = 0
            self._valid[current_index] = True
        else:
            available_indices = np.where(self._replaceable())[0]
            if len(available_indices) == 0:
                tqdm.write(
                    "Trajectory store is full and no trajectories are available for replacement."
                )
                return
            if self.fresh_fraction > 0:
                # Retain useful visits regardless of producer speed; take
                # the oldest exhausted occupant.
                replace_index = available_indices[
                    np.argmin(self._ids[available_indices])
                ]
            else:
                replace_index = np.random.choice(available_indices)
            self._evicted_count += 1
            self._evicted_reuses += int(self._reuses[replace_index])
            current_index = replace_index
            self._trajectories[replace_index] = traj
            self._reuses[replace_index] = 0

        self._ids[current_index] = self._next_id
        self._next_id += 1
        self.total_adds += 1
        decisions = int(acted_rows(traj.player_transitions.env_output.done).sum())
        self._decision_counts[current_index] = decisions
        self.total_admitted_decisions += decisions
        self._progress.update(1)

    def sample(self, count: int, increment: bool = True) -> list[Trajectory]:
        """Sample distinct eligible chunks, reserving scheduled first-use slots.

        Each returned trajectory carries its pre-increment reuse count
        (0 = first visit) for the fresh-vs-replayed staleness diagnostics.
        """
        available_indices = np.where(self._eligible())[0]
        if len(available_indices) < count:
            raise ValueError("insufficient eligible chunks for the batch")
        if self.fresh_fraction > 0:
            fresh_indices = available_indices[self._reuses[available_indices] == 0]
            fresh_indices = fresh_indices[np.argsort(self._ids[fresh_indices])]
            replay_indices = available_indices[self._reuses[available_indices] > 0]
            # Fresh slots are a preference when data is available. Falling
            # back to replay lets a full seen buffer exhaust its cap and
            # admit arrivals without discarding useful visits or deadlocking.
            fresh_count = min(
                len(fresh_indices),
                max(self._fresh_required(count), count - len(replay_indices)),
            )
            sample_indices = np.concatenate(
                [
                    fresh_indices[:fresh_count],
                    np.random.choice(
                        replay_indices, size=count - fresh_count, replace=False
                    ),
                ]
            )
            np.random.shuffle(sample_indices)
        else:
            sample_indices = np.random.choice(
                available_indices, size=count, replace=False
            )
        sampled = [
            self._trajectories[i].replace(
                reuse_count=np.array([self._reuses[i]], dtype=np.int32)
            )
            for i in sample_indices
        ]
        if increment:
            # replace=False above guarantees unique indices.
            first_use = self._reuses[sample_indices] == 0
            self.total_fresh_samples += int(first_use.sum())
            self.total_fresh_decisions_sampled += int(
                self._decision_counts[sample_indices][first_use].sum()
            )
            self._reuses[sample_indices] += 1
            self.total_samples += count
            self.total_sampled_decisions += int(
                self._decision_counts[sample_indices].sum()
            )

        return sampled

    def _reset_decision_accounting(self):
        # Cumulative insert/sample counters; the replay controller diffs
        # them per tick to log the realised replay ratio (samples/insert).
        self.total_adds = 0
        self.total_samples = 0
        self._evicted_reuses = 0
        self._evicted_count = 0
        self._decision_counts.fill(0)
        self.total_fresh_samples = 0
        self.total_fresh_decisions_sampled = 0
        self.total_admitted_decisions = 0
        self.total_sampled_decisions = 0
        self.total_processed_decisions = 0
        self.total_applied_decisions = 0
        self.total_processed_updates = 0
        self.total_applied_updates = 0
        self.accounting_start_step = None

    def record_decision_accounting(self, host_logs):
        """Account completed learner calls separately from prefetched samples.

        Counters start at this store's creation/clear, never at the historical
        run origin. The log worker is the sole recorder. Legacy frame counters
        and checkpoint/league scheduling remain untouched.
        """
        decisions = host_logs.get("player_batch_decisions")
        if decisions is None:
            return
        with self._sample_cv:
            if self.accounting_start_step is None:
                self.accounting_start_step = int(host_logs["lifetime_step"]) - 1
            self.total_processed_updates += 1
            self.total_processed_decisions += int(decisions)
            if host_logs["player_update_skipped"] == 0:
                self.total_applied_updates += 1
                self.total_applied_decisions += int(decisions)
            host_logs.update(
                player_accounting_start_lifetime_step=self.accounting_start_step,
                player_decisions_admitted_session=self.total_admitted_decisions,
                player_decisions_sampled_session=self.total_sampled_decisions,
                player_decisions_processed_session=self.total_processed_decisions,
                player_decisions_applied_session=self.total_applied_decisions,
                player_updates_processed_session=self.total_processed_updates,
                player_updates_applied_session=self.total_applied_updates,
                player_replay_fresh_chunks_sampled_session=self.total_fresh_samples,
                player_decisions_first_use_sampled_session=self.total_fresh_decisions_sampled,
                player_replay_chunks_sampled_session=self.total_samples,
            )
            if self.total_admitted_decisions:
                host_logs["player_decision_reuse_session"] = (
                    self.total_applied_decisions / self.total_admitted_decisions
                )
                host_logs["player_updates_per_fresh_decision_session"] = (
                    self.total_applied_updates / self.total_admitted_decisions
                )
            if self.total_samples:
                host_logs["player_replay_fresh_chunk_fraction_session"] = (
                    self.total_fresh_samples / self.total_samples
                )
            if self.total_sampled_decisions:
                host_logs["player_replay_fresh_decision_fraction_session"] = (
                    self.total_fresh_decisions_sampled / self.total_sampled_decisions
                )
            if self._evicted_count:
                host_logs["player_replay_evicted_mean_reuses"] = (
                    self._evicted_reuses / self._evicted_count
                )

    def __len__(self):
        return len(self._trajectories)
