import collections
import threading
import time


class FairLock:
    def __init__(self):
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._queue = collections.deque()
        self._owner = None
        self._count = 0

    def acquire(self, blocking=True, timeout=-1):
        thread_id = threading.get_ident()
        with self._lock:
            if self._owner == thread_id:
                self._count += 1
                return True

            self._queue.append(thread_id)
            start_time = time.monotonic()

            while self._owner is not None or self._queue[0] != thread_id:
                if not blocking:
                    self._queue.remove(thread_id)
                    return False

                wait_timeout = None
                if timeout >= 0:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        self._queue.remove(thread_id)
                        return False
                    wait_timeout = timeout - elapsed

                if not self._condition.wait(wait_timeout):
                    if thread_id in self._queue:
                        self._queue.remove(thread_id)
                    return False

            self._owner = self._queue.popleft()
            self._count = 1
            return True

    def release(self):
        with self._lock:
            thread_id = threading.get_ident()
            if self._owner != thread_id:
                raise RuntimeError(
                    "Cannot release a lock that is not owned by the current thread."
                )

            self._count -= 1
            if self._count == 0:
                self._owner = None
                # Notify the next waiting thread, if any
                if self._queue:
                    self._condition.notify()

    def __enter__(self):
        # Make the context manager more robust
        if not self.acquire():
            raise RuntimeError("Failed to acquire the lock.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


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
