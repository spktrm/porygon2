import threading

import numpy as np
from tqdm import tqdm

from rl.environment.data import (
    CAT_VF_SUPPORT,
    NUM_ABILITIES,
    NUM_ITEMS,
    NUM_MOVES,
    NUM_SPECIES,
)
from rl.environment.interfaces import (
    BuilderHistoryOutput,
    BuilderTargets,
    BuilderTransition,
    PlayerTargets,
    Trajectory,
)
from rl.environment.protos.features_pb2 import PackedSetFeature


def _np_segmented_cumsum(x: np.ndarray, discount: np.ndarray) -> np.ndarray:
    """Backward cumulative sum: result[t] = x[t] + discount[t] * result[t+1].

    Equivalent to jax_segmented_cumsum but runs in NumPy on CPU.
    The discount array encodes episode boundaries: setting discount[t]=0 at a
    terminal step prevents bootstrapping across episodes.
    """
    result = np.empty_like(x)
    result[-1] = x[-1]
    for t in range(x.shape[0] - 2, -1, -1):
        result[t] = x[t] + discount[t] * result[t + 1]
    return result


def compute_player_targets(
    traj: Trajectory,
    td_lambda: float,
    gae_lambda: float,
) -> PlayerTargets:
    """Compute TD(λ) returns and GAE advantages for the player trajectory.

    Called once when the trajectory is added to the replay buffer so that
    these targets do not need to be recomputed on every training step.
    """
    cat_vf_support = CAT_VF_SUPPORT

    player_valid = ~traj.player_transitions.env_output.done  # (T,)
    player_reward = traj.player_transitions.env_output.win_reward.astype(
        np.float32
    )  # (T, 3)
    player_value_probs = np.exp(
        traj.player_transitions.agent_output.actor_output.value_head.log_probs.astype(
            np.float32
        )
    )  # (T, 3)

    player_next_value_probs = np.concatenate(
        [player_value_probs[1:], player_value_probs[-1:]], axis=0
    )
    player_value_target = (
        player_reward + player_next_value_probs * player_valid[..., None]
    )
    player_value_delta = player_value_target - player_value_probs  # (T, 3)
    player_scalar_delta = player_value_delta @ cat_vf_support  # (T,)

    td_lambdas = (td_lambda * player_valid).astype(np.float32)  # (T,)
    gae_lambdas = (gae_lambda * player_valid).astype(np.float32)  # (T,)

    returns = (
        _np_segmented_cumsum(player_value_delta, td_lambdas[..., None])
        + player_value_probs
    )  # (T, 3)
    advantages = _np_segmented_cumsum(player_scalar_delta, gae_lambdas)  # (T,)

    return PlayerTargets(
        returns=returns.astype(np.float32),
        advantages=advantages.astype(np.float32),
    )


def compute_builder_targets(
    traj: Trajectory,
    td_lambda: float,
    gae_lambda: float,
    entropy_normalising_constant: float,
) -> BuilderTargets:
    """Compute TD(λ) returns and GAE advantages for the builder trajectory.

    The builder reward is derived from the player's final win/loss/tie reward,
    so the full Trajectory (containing both builder and player data) is required.
    The entropy temperature, which changes over training, is intentionally *not*
    applied here; raw_ent_advantages must be scaled in train_step.
    """
    cat_vf_support = CAT_VF_SUPPORT
    builder_transitions = traj.builder_transitions

    builder_valid = ~builder_transitions.env_output.done  # (T_b,)
    T_b = builder_valid.shape[0]

    builder_value_probs = np.exp(
        builder_transitions.agent_output.actor_output.value_head.log_probs.astype(
            np.float32
        )
    )  # (T_b, 3)
    builder_log_prob = (
        builder_transitions.agent_output.actor_output.action_head.log_prob.astype(
            np.float32
        )
    )  # (T_b,)
    builder_ent_pred = builder_transitions.agent_output.actor_output.conditional_entropy_head.logits.astype(
        np.float32
    )  # (T_b,)

    # Place the final player reward at the first terminal position of the builder.
    final_reward = traj.player_transitions.env_output.win_reward[-1].astype(
        np.float32
    )  # (3,)
    num_valid_steps = int(builder_valid.sum())
    builder_reward = np.zeros((T_b, 3), dtype=np.float32)
    if num_valid_steps < T_b:
        builder_reward[num_valid_steps] = final_reward
    # If all steps are valid (no terminal), the reward stays zero (matching the
    # behaviour of jax.nn.one_hot with an out-of-range index).

    # Entropy delta: NLL + discounted future entropy - current entropy prediction.
    builder_ent_scaled = builder_ent_pred * entropy_normalising_constant  # (T_b,)
    next_builder_ent_scaled = (
        np.concatenate(
            [builder_ent_scaled[1:], np.zeros_like(builder_ent_scaled[:1])], axis=0
        )
        * builder_valid
    )
    builder_nll = -builder_log_prob  # (T_b,)
    builder_ent_delta = builder_nll + next_builder_ent_scaled - builder_ent_scaled

    # Value computation.
    builder_next_value_probs = np.concatenate(
        [builder_value_probs[1:], builder_value_probs[-1:]], axis=0
    )
    builder_value_target = (
        builder_reward + builder_next_value_probs * builder_valid[..., None]
    )
    builder_value_delta = builder_value_target - builder_value_probs  # (T_b, 3)

    td_lambdas = (td_lambda * builder_valid).astype(np.float32)  # (T_b,)
    gae_lambdas = (gae_lambda * builder_valid).astype(np.float32)  # (T_b,)

    returns = (
        _np_segmented_cumsum(builder_value_delta, td_lambdas[..., None])
        + builder_value_probs
    )  # (T_b, 3)
    win_advantages = (
        _np_segmented_cumsum(builder_value_delta, gae_lambdas[..., None])
        @ cat_vf_support
    )  # (T_b,)
    ent_returns = (
        _np_segmented_cumsum(builder_ent_delta, td_lambdas) + builder_ent_scaled
    ) / entropy_normalising_constant  # (T_b,)
    raw_ent_advantages = _np_segmented_cumsum(builder_ent_delta, gae_lambdas)  # (T_b,)

    return BuilderTargets(
        returns=returns.astype(np.float32),
        win_advantages=win_advantages.astype(np.float32),
        raw_ent_advantages=raw_ent_advantages.astype(np.float32),
        ent_returns=ent_returns.astype(np.float32),
    )


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

    Targets (TD(λ) returns and GAE advantages) are computed once at add time
    so that train_step does not repeat this work on every reuse.
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_reuses: int = 5,
        need_tracking: bool = False,
        player_td_lambda: float = 1.0,
        player_gae_lambda: float = 1.0,
        builder_td_lambda: float = 1.0,
        builder_gae_lambda: float = 1.0,
        builder_entropy_normalising_constant: float = 100.0,
        compute_builder: bool = True,
    ):
        self._trajectories: dict[int, Trajectory] = {}
        self._reuses = np.zeros(max_size, dtype=int)
        self._valid = np.zeros(max_size, dtype=bool)

        self._max_size = max_size
        self._max_reuses = max_reuses

        self._add_cv = threading.Condition()
        self._sample_cv = threading.Condition()

        self._progress = tqdm(desc="player_producer", smoothing=0.1)

        # Target computation parameters.
        self._player_td_lambda = player_td_lambda
        self._player_gae_lambda = player_gae_lambda
        self._builder_td_lambda = builder_td_lambda
        self._builder_gae_lambda = builder_gae_lambda
        self._builder_entropy_normalising_constant = (
            builder_entropy_normalising_constant
        )
        self._compute_builder = compute_builder

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

    def _compute_targets(self, traj: Trajectory) -> Trajectory:
        """Compute and attach TD(λ) returns and GAE advantages to *traj*."""
        player_targets = compute_player_targets(
            traj,
            td_lambda=self._player_td_lambda,
            gae_lambda=self._player_gae_lambda,
        )
        if self._compute_builder:
            builder_targets = compute_builder_targets(
                traj,
                td_lambda=self._builder_td_lambda,
                gae_lambda=self._builder_gae_lambda,
                entropy_normalising_constant=self._builder_entropy_normalising_constant,
            )
        else:
            builder_targets = BuilderTargets()
        return traj.replace(
            player_targets=player_targets,
            builder_targets=builder_targets,
        )

    def add(self, traj: Trajectory):
        """Adds a trajectory, replacing the oldest over-used entry if the store is full.

        Targets (returns and advantages) are computed once here so that
        train_step can reuse the cached values on every sample.
        """
        if self.need_tracking:
            self._update_usage_counts(traj.builder_history.packed_team_member_tokens)

        traj = self._compute_targets(traj)

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
