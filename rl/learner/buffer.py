import collections
import random
import threading
from typing import Callable

import jax
import numpy as np
from tqdm import tqdm

from rl.environment.interfaces import Trajectory
from rl.environment.utils import clip_history
from rl.learner.config import Porygon2LearnerConfig


class ReplayBuffer:
    """A simple, thread-safe FIFO experience replay buffer."""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._buffer = collections.deque[Trajectory](maxlen=capacity)
        self._lock = threading.Lock()
        self._pbar = tqdm(desc="producer", smoothing=0)
        self._total_added = 0

    @property
    def total_added(self):
        """Total number of transitions added to the buffer."""
        return self._total_added

    def is_ready(self, min_size: int):
        return len(self) >= min_size

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, trajectory: Trajectory):
        with self._lock:
            self._buffer.append(trajectory)
            self._pbar.update(1)
            self._total_added += 1

    def sample(self, batch_size: int) -> Trajectory:
        with self._lock:
            if len(self._buffer) < batch_size:
                raise ValueError(
                    f"Not enough transitions in buffer to sample batch of size {batch_size}."
                    f" Buffer size: {len(self._buffer)}"
                )
            batch = random.sample(self._buffer, batch_size)

        stacked_trajectory: Trajectory = jax.tree.map(
            lambda *xs: np.stack(xs, axis=1), *batch
        )

        resolution = 64
        valid = np.bitwise_not(stacked_trajectory.player_transitions.env_output.done)
        num_valid = valid.sum(0).max().item() + 1
        num_valid = int(np.ceil(num_valid / resolution) * resolution)

        clipped_trajectory = Trajectory(
            builder_transitions=stacked_trajectory.builder_transitions,
            player_transitions=jax.tree.map(
                lambda x: x[:num_valid], stacked_trajectory.player_transitions
            ),
            builder_history=stacked_trajectory.builder_history,
            player_history=clip_history(
                stacked_trajectory.player_history, resolution=resolution
            ),
        )

        return jax.device_put(clipped_trajectory)


class ReplayRatioController:
    """Manages a target replay ratio by controlling both learners and actors."""

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        get_num_samples: Callable[[], int],
        learner_config: Porygon2LearnerConfig,
        hyst: float = 0.05,
        alpha: float = 0.1,
    ):
        self.replay_buffer = replay_buffer
        self.get_num_samples = get_num_samples
        self.target_replay_ratio = learner_config.target_replay_ratio
        self.upper_bound = self.target_replay_ratio * (1 + hyst)
        self.lower_bound = self.target_replay_ratio * (1 - hyst)
        self.batch_size = learner_config.batch_size

        # A condition to make the LEARNER wait
        self._learner_can_proceed = threading.Condition()
        # A separate condition to make the ACTORS wait
        self._actor_can_proceed = threading.Condition()

        self._ratio_lock = threading.Lock()
        self._alpha = alpha
        self._ema = None

    def _get_current_ratio(self) -> float:
        num_samples = self.get_num_samples()
        producer_steps = max(1, self.replay_buffer.total_added)
        return num_samples / producer_steps

    def _get_smoothed_ratio(self) -> float:
        """Calculates the current consumer/producer ratio."""
        with self._ratio_lock:
            current_ratio = self._get_current_ratio()
            if self._ema is None:
                self._ema = current_ratio
            else:
                self._ema = self._alpha * current_ratio + (1 - self._alpha) * self._ema
            return self._ema

    def _is_safe_to_train(self) -> bool:
        """Checks if the learner is allowed to proceed"""
        not_enough_learning = self._get_smoothed_ratio() <= self.upper_bound
        buffer_ready = self.replay_buffer.is_ready(self.batch_size)
        return not_enough_learning and buffer_ready

    def _is_safe_to_produce(self) -> bool:
        """Checks if actors are allowed to produce"""
        too_much_learning = self._get_smoothed_ratio() > self.lower_bound
        no_samples = self.get_num_samples() == 0
        return too_much_learning or no_samples

    def learner_wait(self):
        """Called by the learner; blocks until it's safe to train."""
        with self._learner_can_proceed:
            while not self._is_safe_to_train():
                self._learner_can_proceed.wait()

    def actor_wait(self):
        """Called by an actor; blocks until it's safe to produce data."""
        with self._actor_can_proceed:
            while not self._is_safe_to_produce():
                self._actor_can_proceed.wait()

    def signal_learner(self):
        """Called by a producer after adding data."""
        with self._learner_can_proceed:
            self._learner_can_proceed.notify_all()

    def signal_actors(self):
        """Called by the learner after consuming data."""
        with self._actor_can_proceed:
            self._actor_can_proceed.notify_all()
