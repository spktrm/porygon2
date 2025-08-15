import argparse
import collections
import random
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


class FairLockV2:
    def __init__(self):
        self._state = threading.Lock()  # guards internal state
        self._waiters: collections.deque[threading.Event] = collections.deque()
        self._locked = False  # whether the lock is held
        self._owner = None  # thread id of owner (for assertions)

    def acquire(self, blocking=True, timeout=None):
        me = threading.get_ident()
        # Fast path: uncontended
        with self._state:
            if not self._locked and not self._waiters:
                self._locked = True
                self._owner = me
                return True
            if not blocking:
                return False
            ev = threading.Event()
            self._waiters.append(ev)

        # Wait outside _state
        ok = ev.wait(timeout)
        if not ok:
            # Timed out; remove our event if still queued
            with self._state:
                try:
                    self._waiters.remove(ev)
                except ValueError:
                    pass
            return False

        # We were signaled; take ownership
        with self._state:
            self._locked = True
            self._owner = me
            return True

    def release(self):
        me = threading.get_ident()
        with self._state:
            if not self._locked or self._owner != me:
                raise RuntimeError("lock not held by current thread")
            # Hand off to next waiter if present
            if self._waiters:
                ev = self._waiters.popleft()
                self._owner = None
                self._locked = False
                # Baton pass: wake exactly one waiter
                ev.set()
            else:
                self._owner = None
                self._locked = False

    # Context manager support
    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
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


def jains_fairness(values: list[float]) -> float:
    """Jain’s fairness index: (sum x)^2 / (n * sum x^2). 1.0 is perfectly fair."""
    if not values:
        return 1.0
    s = sum(values)
    sq = sum(v * v for v in values)
    if sq == 0:
        return 1.0
    return (s * s) / (len(values) * sq)


def worker_loop(
    name: str,
    lock,
    work_s: float,
    end_time: float,
    counters: dict[str, int],
    enter_timestamps: dict[str, list[float]],
    spin_delay_s: float,
):
    """Repeatedly acquire the lock, do 'work' for work_s seconds, release."""
    # Small randomized backoff to mix the queue a bit
    rnd = random.Random(hash(name) & 0xFFFFFFFF)

    while time.perf_counter() < end_time:
        # Optional micro backoff before trying to acquire to avoid pathological synchronize
        if spin_delay_s > 0:
            time.sleep(spin_delay_s * rnd.random())

        lock.acquire()
        try:
            # record entry
            counters[name] += 1
            enter_timestamps[name].append(time.perf_counter())
            # emulate work inside the critical section
            time.sleep(work_s)
        finally:
            lock.release()


def run_experiment(
    use_fair_lock: bool,
    short_workers: int,
    short_work_ms: float,
    long_workers: int,
    long_work_ms: float,
    duration_s: float,
    spin_delay_ms: float,
    seed: int | None = None,
):
    random.seed(seed)

    lock = FairLockV2()

    counters = collections.defaultdict(int)  # acquisitions per thread
    enter_timestamps: dict[str, list[float]] = collections.defaultdict(list)

    threads: list[threading.Thread] = []
    end_time = time.perf_counter() + duration_s

    # Launch short workers
    for i in range(short_workers):
        name = f"short-{i}"
        t = threading.Thread(
            target=worker_loop,
            name=name,
            args=(
                name,
                lock,
                short_work_ms / 1000.0,
                end_time,
                counters,
                enter_timestamps,
                spin_delay_ms / 1000.0,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Launch long workers
    for i in range(long_workers):
        name = f"long-{i}"
        t = threading.Thread(
            target=worker_loop,
            name=name,
            args=(
                name,
                lock,
                long_work_ms / 1000.0,
                end_time,
                counters,
                enter_timestamps,
                spin_delay_ms / 1000.0,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Wait for test window to finish
    for t in threads:
        t.join()

    # Aggregate results
    all_names = sorted(
        counters.keys(), key=lambda s: (s.split("-")[0], int(s.split("-")[1]))
    )
    counts = [counters[n] for n in all_names]
    total_entries = sum(counts)
    n_threads = len(all_names)
    equal_share = total_entries / n_threads if n_threads else 0.0

    # Build per-thread metrics
    rows = []
    for n in all_names:
        work_ms = short_work_ms if n.startswith("short-") else long_work_ms
        c = counters[n]
        total_work_ms = c * work_ms
        share_pct = (c / total_entries * 100.0) if total_entries else 0.0
        rows.append((n, work_ms, c, total_work_ms, share_pct))

    # Fairness indices
    fairness_counts = jains_fairness(counts)
    # Another angle: normalize counts by work time to see if “throughput per ms of work” is skewed
    counts_per_ms = []
    for n in all_names:
        work_ms = short_work_ms if n.startswith("short-") else long_work_ms
        counts_per_ms.append(counters[n] / work_ms if work_ms > 0 else 0.0)
    fairness_counts_per_ms = jains_fairness(counts_per_ms)

    # Print summary
    print("=" * 72)
    print(
        "Lock:", "FairLock (FIFO-ish)" if use_fair_lock else "threading.Lock (unfair)"
    )
    print(f"Short workers: {short_workers}  (work {short_work_ms:.1f} ms)")
    print(f"Long workers : {long_workers}   (work {long_work_ms:.1f} ms)")
    print(f"Duration     : {duration_s:.2f} s")
    print(f"Threads      : {n_threads}")
    print("-" * 72)
    print(f"Total entries: {total_entries}")
    print(f"Equal-share target (entries per thread): {equal_share:.2f}")
    print(
        f"Jain fairness (counts)                : {fairness_counts:.4f}  (1.0 = perfectly equal turns)"
    )
    print(
        f"Jain fairness (counts / work_ms)      : {fairness_counts_per_ms:.4f}  (equal per-ms throughput)"
    )
    print("-" * 72)
    print(
        f"{'thread':>10}  {'work_ms':>8}  {'entries':>8}  {'entries*ms':>11}  {'share%':>7}"
    )
    for n, work_ms, c, total_work_ms, share_pct in rows:
        print(
            f"{n:>10}  {work_ms:8.1f}  {c:8d}  {total_work_ms:11.1f}  {share_pct:6.2f}"
        )
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="FairLock test harness")
    parser.add_argument(
        "--fair",
        action="store_true",
        help="Use FairLock (default is Python's threading.Lock)",
    )
    parser.add_argument(
        "--short-workers", type=int, default=8, help="Number of short workers"
    )
    parser.add_argument(
        "--short-work-ms",
        type=float,
        default=2.0,
        help="Work time inside CS for short workers (ms)",
    )
    parser.add_argument(
        "--long-workers", type=int, default=1, help="Number of long workers"
    )
    parser.add_argument(
        "--long-work-ms",
        type=float,
        default=250.0,
        help="Work time inside CS for long workers (ms)",
    )
    parser.add_argument(
        "--duration-s", type=float, default=15.0, help="How long to run the test"
    )
    parser.add_argument(
        "--spin-delay-ms",
        type=float,
        default=0,
        help="Randomized pre-acquire backoff (ms), helps mix queues",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Run once with the chosen lock
    run_experiment(
        use_fair_lock=args.fair,
        short_workers=args.short_workers,
        short_work_ms=args.short_work_ms,
        long_workers=args.long_workers,
        long_work_ms=args.long_work_ms,
        duration_s=args.duration_s,
        spin_delay_ms=args.spin_delay_ms,
        seed=args.seed,
    )

    # Optional: show a side-by-side comparison in a single run
    # Uncomment below to run both locks back-to-back with identical parameters.
    # print("\n\nNow comparing against the other lock with identical parameters...\n")
    # run_experiment(
    #     use_fair_lock=not args.fair,
    #     short_workers=args.short_workers,
    #     short_work_ms=args.short_work_ms,
    #     long_workers=args.long_workers,
    #     long_work_ms=args.long_work_ms,
    #     duration_s=args.duration_s,
    #     spin_delay_ms=args.spin_delay_ms,
    #     seed=args.seed,
    # )


if __name__ == "__main__":
    main()
