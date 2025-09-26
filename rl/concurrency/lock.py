import heapq
import threading
import time
from dataclasses import dataclass


@dataclass
class _Waiter:
    # 'priority' is the *base* priority (lower is higher).
    priority: int
    seq: int
    thread: int
    enq_time: float
    cancelled: bool = False


class PriorityLock:
    """
    A reentrant, fair, thread-safe priority lock with optional priority aging.

    Priority semantics:
      - Lower numbers mean higher priority: 0 > 1 > 2 ...
      - FIFO order among equal priorities.
      - If aging is enabled, a waiter’s *effective* priority improves over time.

    Methods:
      acquire(priority=0, block=True, timeout=None) -> bool
      release() -> None
      locked() -> bool
      with_priority(priority=0, timeout=None) -> context manager

    Notes:
      - Reentrant: the owner can reacquire without entering the queue.
      - If block=False, returns immediately (True if acquired, False otherwise).
      - Fairness: if higher-priority waiters exist, new callers queue even if free.
      - Aging: every `aging_period` seconds, a waiting thread’s effective priority
        is boosted by `aging_step` (i.e., numerically decreased).
    """

    def __init__(
        self,
        *,
        aging_period: float = 0.0,
        aging_step: int = 1,
        aging_max_boost: int | None = None
    ):
        self._state = threading.Lock()  # guards internal state
        self._cond = threading.Condition(self._state)
        # Heap holds tuples: (effective_priority, seq, id(waiter), waiter)
        self._heap = []
        self._seq = 0
        self._owner = None
        self._count = 0  # reentrancy count

        # Aging configuration
        self._aging_period = float(aging_period) if aging_period else 0.0
        self._aging_step = int(aging_step)
        self._aging_max_boost = (
            None if aging_max_boost is None else int(aging_max_boost)
        )
        self._aging_last_rebuild = 0.0

    # ---------- Public API ----------
    def acquire(self, priority=0, block=True, timeout=None):
        if not block and timeout is not None:
            raise ValueError("timeout must be None when block=False")

        ident = threading.get_ident()
        with self._state:
            # Reentrant fast-path
            if self._owner == ident:
                self._count += 1
                return True

            # Join the heap
            now = time.monotonic()
            w = _Waiter(
                priority=int(priority), seq=self._next_seq(), thread=ident, enq_time=now
            )
            heapq.heappush(self._heap, (self._eff_priority(w, now), w.seq, id(w), w))

            end = None if timeout is None else (now + timeout)
            while True:
                self._prune_cancelled()
                self._rebuild_heap_if_aged()  # apply aging periodically

                # Can we take the lock? Only if we're at the head and it's free.
                if self._is_head(w) and self._count == 0:
                    heapq.heappop(self._heap)  # remove ourselves
                    self._owner = ident
                    self._count = 1
                    return True

                if not block:
                    w.cancelled = True
                    self._cond.notify_all()
                    return False

                # Blocking path with (optional) timeout
                remaining = None if end is None else max(0.0, end - time.monotonic())
                if remaining == 0.0:
                    w.cancelled = True
                    self._cond.notify_all()
                    return False

                self._cond.wait(timeout=remaining)

    def release(self):
        ident = threading.get_ident()
        with self._state:
            if self._owner != ident:
                raise RuntimeError(
                    "PriorityLock.release(): current thread does not own the lock"
                )

            self._count -= 1
            if self._count == 0:
                self._owner = None
                # Aging update before waking helps ensure the true highest effective
                # priority waiter proceeds next.
                self._rebuild_heap_if_aged(force=True)
                self._cond.notify_all()

    def locked(self):
        with self._state:
            return self._count > 0

    def with_priority(self, priority=0, timeout=None):
        lock = self

        class _Guard:
            def __enter__(_self):
                acquired = lock.acquire(priority=priority, timeout=timeout)
                if not acquired:
                    raise TimeoutError("PriorityLock: timed out acquiring lock")
                return lock

            def __exit__(_self, exc_type, exc, tb):
                lock.release()

        return _Guard()

    # ---------- Internal helpers ----------
    def _next_seq(self):
        s = self._seq
        self._seq += 1
        return s

    def _is_head(self, waiter):
        if not self._heap:
            return False
        _, _, _, head = self._heap[0]
        return head is waiter

    def _prune_cancelled(self):
        # Remove cancelled waiters at the top; lazy removal elsewhere.
        while self._heap and self._heap[0][3].cancelled:
            heapq.heappop(self._heap)

    # -- Aging machinery --
    def _eff_priority(self, w: _Waiter, now: float | None = None) -> float:
        """Compute the *effective* priority, applying aging if configured."""
        if self._aging_period <= 0.0:
            return w.priority
        if now is None:
            now = time.monotonic()
        waited = max(0.0, now - w.enq_time)
        boost_steps = int(waited // self._aging_period) * self._aging_step
        if self._aging_max_boost is not None:
            boost_steps = min(boost_steps, self._aging_max_boost)
        # Lower number = higher priority
        return float(w.priority - boost_steps)

    def _rebuild_heap_if_aged(self, force: bool = False):
        """Periodically rebuild the heap using current effective priorities."""
        if self._aging_period <= 0.0:
            return
        now = time.monotonic()
        if not force and (now - self._aging_last_rebuild) < self._aging_period:
            return
        self._aging_last_rebuild = now

        # Also compact away cancelled waiters while rebuilding
        new = []
        for _, _, _, w in self._heap:
            if not w.cancelled:
                new.append((self._eff_priority(w, now), w.seq, id(w), w))
        self._heap = new
        heapq.heapify(self._heap)


class NoOpLock:
    """
    A no-op lock that does nothing.
    Useful for cases where a lock is required but no locking is needed.
    """

    def acquire(self, blocking=True, timeout=-1):
        return True

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
