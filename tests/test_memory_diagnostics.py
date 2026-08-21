"""log_memory_diagnostics: process-wide RAM attribution logged every
memory_diag_interval steps. A free function over (run_state, league, logs)
since 2026-08-21, so these run against plain stubs — no Learner, no
devices, actors, or service required.
"""

import json
import time
from types import SimpleNamespace

from rl.online.training.diagnostics import log_memory_diagnostics


def make_stub(run_state=None, cache_entries=0, cache_bytes=0):
    return SimpleNamespace(
        run_state=run_state or make_run_state(),
        league=SimpleNamespace(cache_stats=lambda: (cache_entries, cache_bytes)),
    )


def make_run_state(player_bytes=0, builder_bytes=0):
    return SimpleNamespace(
        player_replay=SimpleNamespace(nbytes=lambda: player_bytes),
        builder_replay=SimpleNamespace(nbytes=lambda: builder_bytes),
    )


def diag(stub, logs=None):
    logs = {} if logs is None else logs
    log_memory_diagnostics(stub.run_state, stub.league, logs)
    return logs


def test_core_process_fields_always_present():
    logs = diag(make_stub())
    assert logs["diag_rss_mb"] > 0
    assert logs["diag_os_threads"] > 0
    assert logs["diag_py_threads"] > 0


def test_thread_buckets_sum_to_total():
    logs = diag(make_stub())
    bucket_keys = [k for k in logs if k.startswith("diag_py_threads_")]
    assert bucket_keys, "expected at least the 'other' bucket"
    assert sum(logs[k] for k in bucket_keys) == logs["diag_py_threads"]


def test_replay_and_cache_bytes():
    stub = make_stub(
        run_state=make_run_state(player_bytes=10 * 2**20, builder_bytes=2 * 2**20),
        cache_entries=3,
        cache_bytes=45 * 2**20,
    )
    logs = diag(stub)
    assert logs["diag_player_replay_mb"] == 10
    assert logs["diag_builder_replay_mb"] == 2
    assert logs["diag_league_cache_entries"] == 3
    assert logs["diag_league_cache_mb"] == 45


def test_node_stats_folded_in_when_fresh(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "service_memory.json").write_text(
        json.dumps(
            {
                "rss_mb": 888.0,
                "heap_used_mb": 120.0,
                "num_workers": 6,
                "worker_heap_used_mb": 540.0,
                "workers_reported": 6,
                "ts": time.time(),
            }
        )
    )
    logs = diag(make_stub())
    assert logs["diag_node_rss_mb"] == 888.0
    assert logs["diag_node_heap_used_mb"] == 120.0
    assert logs["diag_node_num_workers"] == 6
    assert logs["diag_node_worker_heap_used_mb"] == 540.0
    assert logs["diag_node_workers_reported"] == 6


def test_stale_node_stats_are_dropped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "service_memory.json").write_text(
        json.dumps(
            {
                "rss_mb": 888.0,
                "heap_used_mb": 120.0,
                "num_workers": 6,
                "ts": time.time() - 120,  # older than the 60s staleness bound
            }
        )
    )
    logs = diag(make_stub())
    assert "diag_node_rss_mb" not in logs


def test_missing_node_stats_file_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no runtime/ dir at all
    logs = diag(make_stub())
    assert "diag_node_rss_mb" not in logs


def test_malformed_node_stats_file_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "service_memory.json").write_text("not json")
    logs = diag(make_stub())
    assert "diag_node_rss_mb" not in logs


def test_heap_census_runs_and_logs_without_raising(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="rl.online.training.diagnostics"):
        diag(make_stub())
    assert any("Heap census" in r.message for r in caplog.records)


def test_heap_census_never_pollutes_the_wandb_logs_dict():
    logs = diag(make_stub())
    # The census reports via logger, not the wandb logs dict — nothing
    # class-name-shaped should leak into a scalar time-series row.
    assert not any(k.startswith("diag_heap") for k in logs)
