import threading


class AtomicCounter:
    def __init__(self, initial: int = 0):
        self._n = initial
        self._lock = threading.Lock()

    def inc(self, k: int = 1):
        with self._lock:
            self._n += k
            return self._n

    def value(self) -> int:
        with self._lock:
            return self._n
