"""Process-wide RAM attribution, and the OOM guard's memory reading.

Free functions: these describe the PROCESS, not the Learner. Added after
session 1786537634's RSS climbed 5.9->17GB (threads 478->775) with no way
to attribute it from wandb alone (CLAUDE.md 1).
"""

import collections
import gc
import json
import logging
import sys
import threading
import time

from rl.online.training.run_state import RunState

logger = logging.getLogger(__name__)


# Known python-thread name prefixes, for the census below — anything
# unrecognized lands in "other".
THREAD_NAME_BUCKETS = (
    "Selfplay-",
    "BuilderActor-",
    "EvalActor-",
    "transfer-",
    "log-",
    "ckpt-",
    "inference-server",
    "ThreadPoolExecutor",
)


def log_memory_diagnostics(run_state: RunState, league, logs: dict) -> None:
    """Process-wide RAM attribution, riding main's periodic wandb logs
    every memory_diag_interval steps.

    Motivated by session 1786537634: RSS climbed 5.9GB -> 17GB while
    the OS thread count grew 478 -> 775 with no obvious owner, and
    none of it was attributable from wandb alone. The bounded-by-
    design consumers (replay buffers, league opponent cache) get
    exact byte counts here; the thread census separates python
    threads (named, bucketed below) from native ones — if
    diag_os_threads far exceeds diag_py_threads, the growth lives in
    native pools (XLA/CUDA/websocket internals), not python code."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    logs["diag_rss_mb"] = int(line.split()[1]) / 1024.0
                elif line.startswith("Threads:"):
                    logs["diag_os_threads"] = int(line.split()[1])
    except Exception:
        pass  # non-Linux — skip, same posture as _available_memory_fraction

    py_threads = threading.enumerate()
    logs["diag_py_threads"] = len(py_threads)
    buckets = dict.fromkeys(THREAD_NAME_BUCKETS, 0)
    buckets["other"] = 0
    for t in py_threads:
        for prefix in THREAD_NAME_BUCKETS:
            if t.name.startswith(prefix):
                buckets[prefix] += 1
                break
        else:
            buckets["other"] += 1
    for prefix, count in buckets.items():
        key = prefix.rstrip("-").lower().replace("-", "_")
        logs[f"diag_py_threads_{key}"] = count

    # Heap census: attributes host RSS the byte-exact counters below
    # (replay buffers, league cache) don't cover — e.g. the ~3GB the
    # 2026-08-18 fork jump left unexplained by thread counts
    # and league cache alone. sys.getsizeof is shallow (a dict/list's
    # own overhead, not its contents), but that's exactly what surfaces
    # a genuine culprit: a huge COUNT of one type (numpy arrays, proto
    # objects, EnvironmentState instances) dominating aggregate bytes.
    # Logged to the console, not wandb — dynamic top-N, not a stable
    # scalar key. gc.get_objects() walks the whole heap, so this rides
    # the same 5000-step cadence as the rest of this function.
    try:
        counts = collections.Counter()
        sizes = collections.Counter()
        for obj in gc.get_objects():
            t = type(obj).__name__
            counts[t] += 1
            sizes[t] += sys.getsizeof(obj)
        top = sizes.most_common(15)
        logger.info(
            "Heap census (top-15 by approx shallow size): %s",
            ", ".join(f"{t}={counts[t]}objs/{sz / 2**20:.1f}MB" for t, sz in top),
        )
    except Exception:
        logger.exception("Heap census failed")

    run_state = run_state
    logs["diag_player_replay_mb"] = run_state.player_replay.nbytes() / 2**20
    logs["diag_builder_replay_mb"] = run_state.builder_replay.nbytes() / 2**20
    try:
        with open("runtime/service_memory.json") as f:
            node_stats = json.load(f)
        # Service writes every 10s; 60s is a generous staleness bound
        # in case the file is left over from a service that's since died.
        if time.time() - node_stats["ts"] < 60:
            for key in (
                "rss_mb",
                "heap_used_mb",
                "num_workers",
                "worker_heap_used_mb",
                "workers_reported",
            ):
                logs[f"diag_node_{key}"] = node_stats[key]
    except Exception:
        pass  # service not up, stats file stale/absent, or race on the rename

    entries, cache_bytes = league.cache_stats()
    logs["diag_league_cache_entries"] = entries
    logs["diag_league_cache_mb"] = cache_bytes / 2**20


def available_memory_fraction() -> float | None:
    """Fraction of total system RAM currently available (reclaimable
    caches counted as available, matching what actually predicts an
    OOM kill), or None if it can't be determined (non-Linux, or
    /proc/meminfo unreadable) — the caller treats None as "skip the
    check", the same defensive posture as this codebase's other
    optional-environment guards (e.g. the matplotlib import)."""
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                key, value = line.split(":", 1)
                meminfo[key] = int(value.strip().split()[0])  # kB
        return meminfo["MemAvailable"] / meminfo["MemTotal"]
    except Exception:
        return None
