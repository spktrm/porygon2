import random
import threading
from collections import deque

from rl.environment.interfaces import Trajectory


class ReplayRatioTokenBucket:
    """Token bucket in TRAJECTORY units.

    - on_trajectories(n): mint target_rr * n tokens
    - can_consume(cost): do we have >= cost tokens?
    - consume(cost): spend tokens
    """

    __slots__ = ("target_rr", "tokens", "capacity", "headroom")

    def __init__(self, target_rr: float, capacity: float, headroom: float = 0.05):
        self.target_rr = float(target_rr)
        self.tokens = 0.0
        self.capacity = float(capacity)  # in 'trajectory-tokens'
        self.headroom = float(headroom)  # keeps realized RR slightly under target

    def on_trajectories(self, n: int) -> None:
        self.tokens = min(self.capacity, self.tokens + self.target_rr * max(0, int(n)))

    def can_consume(self, cost_traj: int) -> bool:
        # require a touch more than exact cost to bias under target
        return self.tokens >= (1.0 + self.headroom) * float(cost_traj)

    def consume(self, cost_traj: int) -> None:
        self.tokens -= float(cost_traj)
        if self.tokens < 0.0:
            self.tokens = 0.0


class ReplayBuffer:
    """Thread-safe uniform replay for [T, ...] trajectories."""

    def __init__(self, capacity: int):
        self._buf: deque[Trajectory] = deque(maxlen=capacity)
        self._size = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def add(self, item):
        with self._lock:
            if self._size == len(self._buf):  # buffer full and maxlen hit
                # deque will evict leftmost; keep _size consistent with len
                pass
            else:
                self._size += 1
            self._buf.append(item)
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
