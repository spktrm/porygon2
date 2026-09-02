"""Actor-side timing counters, drained into the learner's wandb logs.

Actors log nothing of their own (the eval loop's winrate write is the
only actor-side wandb call); everything else rides the learner's log
queue. This is the one shared sink: every training PlayerActor, its
env and the InferenceServer ``record`` into it from their own threads,
and the learner drains it every ``actor_stats_log_steps`` into ``logs``
as per-name means. Built for the 2026-09-02 actor-step decomposition
(the system rate is actor-bound and no panel said WHERE the actor's
time went) — the baseline the history-carry pass is judged against.
"""

import threading
import time
from contextlib import contextmanager, nullcontext

# The per-iteration timer of PlayerActor.unroll; its count is the step
# count that steps_per_sec reads.
STEP_TOTAL = "actor_time_step_total"
# Timers whose sum is subtracted from STEP_TOTAL to give the unattributed
# remainder. Their counts differ slightly from STEP_TOTAL's (the reset
# receive is a service wait with no step), so the remainder is a mean-
# of-means read, not an exact residual.
STEP_PARTS = (
    "actor_time_service_wait",
    "actor_time_process_state",
    "actor_time_history_clip",
    "actor_time_inference",
)


class ActorStats:
    def __init__(self):
        self._lock = threading.Lock()
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._drained_at = time.perf_counter()

    def record(self, name: str, value: float) -> None:
        with self._lock:
            self._sums[name] = self._sums.get(name, 0.0) + value
            self._counts[name] = self._counts.get(name, 0) + 1

    @contextmanager
    def timed(self, name: str):
        """Records the block's wall time in milliseconds."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - start) * 1e3)

    def drain(self) -> dict[str, float]:
        """Per-name means since the last drain, plus the derived
        ``actor_steps_per_sec`` (aggregate over every actor) and
        ``actor_time_other``. Resets the counters."""
        now = time.perf_counter()
        with self._lock:
            sums = self._sums
            counts = self._counts
            self._sums = {}
            self._counts = {}
            elapsed = now - self._drained_at
            self._drained_at = now
        means = {name: sums[name] / counts[name] for name in sums}
        if STEP_TOTAL in means:
            means["actor_steps_per_sec"] = counts[STEP_TOTAL] / max(elapsed, 1e-6)
            attributed = sum(means.get(part, 0.0) for part in STEP_PARTS)
            means["actor_time_other"] = means[STEP_TOTAL] - attributed
        return means


def timed(stats: "ActorStats | None", name: str):
    """``stats.timed(name)`` when a sink is wired, a no-op otherwise —
    eval actors and offline harnesses carry no sink."""
    if stats is None:
        return nullcontext()
    return stats.timed(name)
