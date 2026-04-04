import threading

import numpy as np
from tqdm import tqdm

from rl.environment.data import NUM_ABILITIES, NUM_ITEMS, NUM_MOVES, NUM_SPECIES
from rl.environment.interfaces import (
    BuilderHistoryOutput,
    BuilderTransition,
    Trajectory,
)
from rl.environment.protos.features_pb2 import PackedSetFeature


class BuilderTrajectoryStore:
    """Stores builder trajectories for later use by the learner."""

    def __init__(self, max_size: int = 1000, max_reuses: int = 5):
        self._trajectories: dict[
            int, tuple[BuilderTransition, BuilderHistoryOutput]
        ] = {}
        self._reuses = np.zeros(max_size, dtype=int)
        self._valid = np.zeros(max_size, dtype=bool)

        self._max_size = max_size
        self._max_reuses = max_reuses

        self._add_cv = threading.Condition()
        self._sample_cv = threading.Condition()

        self._progress = tqdm(desc="builder_producer", smoothing=0.1)

    @classmethod
    def from_trajectories(
        cls,
        trajectories: list[BuilderTransition],
        max_size: int = 1000,
        max_reuses: int = 5,
    ):
        """Initializes the store with a list of trajectories. Primarily for testing."""
        store = cls(max_size=max_size, max_reuses=max_reuses)
        for trajectory in trajectories:
            store.add_trajectory(trajectory)
        return store

    def is_full(self, limit: int = None) -> bool:
        """Returns True if the store has reached its maximum capacity."""
        if limit is None:
            limit = self._max_size
        return len(self._trajectories) >= limit

    def ready_to_sample(self) -> bool:
        """Returns True if there is at least one trajectory that can be sampled."""
        return np.any((self._reuses < self._max_reuses) & self._valid)

    def ready_to_add(self) -> bool:
        """Returns True if there is capacity to add a new trajectory."""
        return len(self._trajectories) < self._max_size or np.any(
            self._reuses >= self._max_reuses
        )

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
                print(
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

    Mirrors the structure of BuilderTrajectoryStore: trajectories are kept
    until they have been sampled at least max_reuses times, after which they
    become eligible for replacement.
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_reuses: int = 5,
        need_tracking: bool = False,
    ):
        self._trajectories: dict[int, Trajectory] = {}
        self._reuses = np.zeros(max_size, dtype=int)
        self._valid = np.zeros(max_size, dtype=bool)

        self._max_size = max_size
        self._max_reuses = max_reuses

        self._add_cv = threading.Condition()
        self._sample_cv = threading.Condition()

        self._progress = tqdm(desc="player_producer", smoothing=0.1)

        # Tracking
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

    def is_min_fill_fraction_reached(self, fraction: float = 0.5) -> bool:
        """Returns True if the store is at least ``fraction`` full.

        Args:
            fraction: Required fill level in [0.0, 1.0].
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0.0, 1.0], got {fraction}")
        return len(self._trajectories) >= int(self._max_size * fraction)

    def ready_to_sample(self, n: int = None) -> bool:
        """Returns True if there is at least one trajectory that can be sampled."""
        if n is None:
            return np.any((self._reuses < self._max_reuses) & self._valid)
        else:
            return np.sum((self._reuses < self._max_reuses) & self._valid) >= n

    def ready_to_add(self) -> bool:
        """Returns True if there is capacity to add a new trajectory."""
        return len(self._trajectories) < self._max_size or np.any(
            self._reuses >= self._max_reuses
        )

    def reset_usage_counts(self):
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
        """Adds a trajectory, replacing the oldest over-used entry if the store is full."""
        if self.need_tracking:
            self._update_usage_counts(traj.builder_history.packed_team_member_tokens)

        # Extract player final reward for builder target computation at train time.
        traj = traj.replace(
            player_final_reward=traj.player_transitions.env_output.win_reward[-1]
        )

        if len(self._trajectories) < self._max_size:
            current_index = len(self._trajectories)
            self._trajectories[current_index] = traj
            self._reuses[current_index] = 0
            self._valid[current_index] = True
        else:
            available_indices = np.where(self._reuses >= self._max_reuses)[0]
            if len(available_indices) == 0:
                print(
                    "Trajectory store is full and no trajectories are available for replacement."
                )
                return
            replace_index = np.random.choice(available_indices)
            self._trajectories[replace_index] = traj
            self._reuses[replace_index] = 0

        self._progress.update(1)

    def sample(self, n: int, increment: bool = True) -> list[Trajectory]:
        """Samples n trajectories uniformly from those with fewer than max_reuses."""
        valid_indices = (self._reuses < self._max_reuses) & self._valid
        available_indices = np.where(valid_indices)[0]

        sample_indices = np.random.choice(available_indices, size=n, replace=False)
        if increment:
            unique_indices, counts = np.unique(sample_indices, return_counts=True)
            for idx, count in zip(unique_indices, counts):
                self._reuses[idx] += count

        return [self._trajectories[i] for i in sample_indices]

    def __len__(self):
        return len(self._trajectories)
