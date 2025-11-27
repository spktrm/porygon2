import random
import threading
from collections import deque
from typing import NamedTuple

import numpy as np

from rl.environment.data import NUM_SPECIES
from rl.environment.interfaces import BuilderTrajectory


def softmax(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)


class ReplayRatioTokenBucket:
    """Token bucket in TRAJECTORY units.

    - on_trajectories(n): mint target_rr * n tokens
    - can_consume(cost): do we have >= cost tokens?
    - consume(cost): spend tokens
    """

    __slots__ = ("target_rr", "tokens", "capacity")

    def __init__(self, target_rr: float, capacity: float):
        self.target_rr = float(target_rr)
        self.tokens = 0.0
        self.capacity = float(capacity)  # in 'trajectory-tokens'

    def on_trajectories(self, n: int) -> None:
        self.tokens = min(self.capacity, self.tokens + self.target_rr * max(0, int(n)))

    def can_consume(self, cost_traj: int) -> bool:
        return self.tokens >= float(cost_traj)

    def consume(self, cost_traj: int) -> None:
        self.tokens -= float(cost_traj)
        if self.tokens < 0.0:
            self.tokens = 0.0


class DirectRatioLimiter:
    """
    Forces producer/consumer threads to maintain a target replay ratio.

    This controller directly tracks the total number of trajectories
    produced vs. consumed (processed) and uses a threading.Condition
    to throttle the "winning" thread.

    rr = total_consumed / total_produced
    """

    def __init__(
        self, target_rr: float, batch_size: int, warmup_trajectories: int = 1000
    ):
        self.target_rr = float(target_rr)
        self.batch_size = int(batch_size)
        self.warmup_trajectories = int(warmup_trajectories)

        self.total_produced = 0
        self.total_consumed = 0

        # This controller needs to check the replay buffer's length
        # directly to make data + ratio checks atomic.
        self.replay_buffer_len_fn = lambda: 0
        self.cv = threading.Condition()

    def set_replay_buffer_len_fn(self, len_fn):
        """Pass in the ReplayBuffer's len() function."""
        self.replay_buffer_len_fn = len_fn

    def _get_current_rr(self) -> float:
        if self.total_produced == 0:
            return 0.0
        # This is (processed / produced). Can be > 1.0
        return self.total_consumed / self.total_produced

    def wait_for_produce_permission(self):
        """Called by the Actor (Producer)."""
        with self.cv:
            if self.total_produced < self.warmup_trajectories:
                return  # Warmup, always allow

            # THIS IS THE FIX:
            # Wait if the ratio is LOW (e.g., 1.3 < 2.5)
            # This means the producer is "winning" (too far ahead).
            # So, we throttle the producer to let the consumer catch up.
            self.cv.wait_for(lambda: self._get_current_rr() >= self.target_rr)

    def notify_produced(self, n_trajectories: int = 1):
        """Called by the Actor (Producer) after adding to replay."""
        with self.cv:
            self.total_produced += n_trajectories
            # Wake up any waiting consumers
            self.cv.notify_all()

    def wait_for_consume_permission(self):
        """Called by the Learner (Consumer)."""
        with self.cv:
            # We must check for data *and* ratio atomically.

            # 1. Wait if we are starved for data.
            #    This check MUST come first.
            self.cv.wait_for(lambda: self.replay_buffer_len_fn() >= self.batch_size)

            # 2. Wait if the ratio is HIGH (e.g., 2.6 > 2.5)
            #    This means the consumer is "winning" (too far ahead).
            #    So, we throttle the consumer to let the producer catch up.
            self.cv.wait_for(lambda: self._get_current_rr() < self.target_rr)

            # If we pass both checks, we are allowed to consume
            return

    def notify_consumed(self, n_trajectories: int):
        """Called by the Learner (Consumer) after sampling a batch."""
        with self.cv:
            self.total_consumed += n_trajectories
            # Wake up any waiting producers
            self.cv.notify_all()


def calculate_tracking(old: np.ndarray, new: np.ndarray, tau: float, minlength: int):
    return (1 - tau) * old + tau * np.bincount(new.reshape(-1), minlength=minlength)


class PlayerReplayBuffer:
    """Thread-safe uniform replay for [T, ...] trajectories."""

    def __init__(self, capacity: int):
        self._buf: deque[BuilderTrajectory] = deque(maxlen=capacity)
        self._size = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def add(self, traj: BuilderTrajectory):
        with self._lock:

            if self._size == len(self._buf):  # buffer full and maxlen hit
                # deque will evict leftmost; keep _size consistent with len
                pass
            else:
                self._size += 1

            self._buf.append(traj)
            self._not_empty.notify()

    def can_sample(self, n: int) -> bool:
        with self._lock:
            return self._size >= n

    def sample(self, n: int):
        with self._lock:
            while self._size < n:
                self._not_empty.wait()
            # Uniform without replacement
            idxs = random.sample(range(self._size), n)
            out = [self._buf[i] for i in idxs]
            return out

    def __len__(self):
        with self._lock:
            return self._size


class BuilderMetadata(NamedTuple):
    n_sampled: np.ndarray = np.array([0], dtype=np.int32)
    avg_reward: np.ndarray = np.array([0], dtype=np.float32)


class BuilderReplayBuffer:
    def __init__(self, capacity: int):
        self._active_items: dict[int, BuilderTrajectory] = {}
        self._cache: dict[int, BuilderMetadata] = {}

        self._counter = 0
        self._mod = capacity

        self._size = 0

        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

        # Tracking
        self._species_counts = np.zeros(NUM_SPECIES, dtype=np.float32)
        self._tau = 1e-3

    def __len__(self):
        with self._lock:
            return self._size

    def reset_species_counts(self):
        with self._lock:
            self._species_counts = np.zeros(NUM_SPECIES, dtype=np.float32)

    def _get_next_free_id(self):
        candidate_id = self._counter % self._mod
        self._counter += 1
        return candidate_id

    def add(self, item_data: BuilderTrajectory) -> int:
        with self._lock:
            if self._size < self._mod:
                new_id = self._get_next_free_id()
                self._size += 1
            else:
                candidates = list(self._active_items.keys())
                new_id = min(candidates, key=lambda k: self._cache[k].avg_reward.item())

            self._species_counts = calculate_tracking(
                self._species_counts,
                item_data.history.species_tokens.reshape(-1),
                self._tau,
                NUM_SPECIES,
            )

            self._active_items[new_id] = item_data
            self._cache[new_id] = BuilderMetadata()

            self._not_empty.notify()
            return new_id

    def sample_for_player(self):
        with self._lock:
            self._not_empty.wait_for(lambda: len(self._active_items) > 0)

            active_keys = list(self._active_items.keys())

            avg_reward = np.array(
                [self._cache[k].avg_reward.item() for k in active_keys]
            )
            n = np.array([self._cache[k].n_sampled.item() for k in active_keys])
            log_total = np.log(max(np.sum(n), 1))

            scores = avg_reward + np.sqrt(log_total / (n + 1e-5))
            probs = softmax(scores)
            selected_idx = np.random.choice(len(active_keys), p=probs)

            selected_key = active_keys[selected_idx]

            return (
                selected_key,
                self._active_items[selected_key],
                self._cache[selected_key],
            )

    def sample_for_learner(self, n: int):
        with self._lock:
            while self._size < n:
                self._not_empty.wait()
            # Uniform without replacement
            idxs = random.sample(range(self._size), n)
            trajectories = [self._active_items[i] for i in idxs]
            metadatas = [self._cache[i] for i in idxs]
            return trajectories, metadatas

    def update(self, internal_id: int, metadata: BuilderMetadata):
        """
        Updates item by the internal_id retrieved from sample_active.
        """
        with self._lock:
            if internal_id not in self._cache:
                # Item might have been moved to mature buffer by another thread already
                return

            self._cache[internal_id] = metadata
