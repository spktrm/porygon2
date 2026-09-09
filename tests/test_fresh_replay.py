"""Fresh-stream scheduling, admission liveness and exact decision accounting."""

from types import SimpleNamespace

import numpy as np
import pytest

from rl.environment.interfaces import PlayerEnvOutput, PlayerTransition, Trajectory
from rl.online.buffer import PlayerTrajectoryStore
from rl.online.decisions import count_chunk_decisions
from rl.online.training.workers import wandb_log_worker


def trajectory(done, tag=0):
    # `game_length` carries an admission tag so a sampled chunk can be told
    # apart without the store exposing its slots.
    return Trajectory(
        player_transitions=PlayerTransition(
            env_output=PlayerEnvOutput(done=np.asarray(done, dtype=bool))
        ),
        game_length=np.array([tag], dtype=np.int32),
    )


def tag_of(chunk):
    return chunk.game_length.item()


def fresh_store(size=8, fraction=0.125, cap=8):
    return PlayerTrajectoryStore(max_size=size, max_reuses=cap, fresh_fraction=fraction)


def fill_store(store, count):
    for _ in range(count):
        store.add(trajectory([False, False, True, False], tag=store.total_adds))


@pytest.mark.parametrize(
    "done,expected",
    [
        ([False] * 29 + [True] + [False] * 18, 29),
        ([False] * 64, 63),
        ([False] * 63 + [True], 63),
        ([True, False, False], 0),
        ([False, True, True, False], 1),
        ([], 0),
    ],
)
def test_decision_count_excludes_terminal_padding_and_bootstrap(done, expected):
    assert count_chunk_decisions(done) == expected


def test_overlapping_chunks_count_each_decision_once():
    # 70 actions: row 63 is bootstrap-only in the first chunk and acted
    # row zero in the second. Its terminal padding must never add actions.
    first = np.zeros(64, dtype=bool)
    second = np.zeros(64, dtype=bool)
    second[7] = True
    assert count_chunk_decisions(first) + count_chunk_decisions(second) == 70


def test_fractional_fresh_slots_without_batch_duplicates():
    store = fresh_store()
    fill_store(store, 8)
    initial = store.sample(4)
    assert all(chunk.reuse_count.item() == 0 for chunk in initial)
    oldest_unseen = sorted(
        set(range(store.total_adds)) - {tag_of(chunk) for chunk in initial}
    )
    for expected_fresh in [0, 1, 0, 1]:
        batch = store.sample(4)
        assert len({tag_of(chunk) for chunk in batch}) == 4
        fresh = [chunk for chunk in batch if chunk.reuse_count.item() == 0]
        assert len(fresh) == expected_fresh
        if fresh:
            assert tag_of(fresh[0]) == oldest_unseen.pop(0)
        assert all(chunk.reuse_count.item() < 8 for chunk in batch)


def test_full_seen_buffer_can_admit_fresh_data_without_deadlock():
    store = fresh_store(size=4)
    fill_store(store, 4)
    original_ids = store._ids.copy()
    for visit in range(8):
        assert store.ready_to_sample(4)
        assert not store.ready_to_add()
        assert all(chunk.reuse_count.item() == visit for chunk in store.sample(4))
        np.testing.assert_array_equal(store._ids, original_ids)
    assert not store.ready_to_sample(4)
    assert store.ready_to_add()
    fill_store(store, 4)
    assert store.ready_to_sample(4)
    assert all(chunk.reuse_count.item() == 0 for chunk in store.sample(4))


@pytest.mark.parametrize("cap", [1, 2, 8])
@pytest.mark.parametrize("fraction", [0.125, 0.25, 1.0])
def test_producer_consumer_progress_and_cap_under_fast_arrivals(cap, fraction):
    store = fresh_store(size=8, fraction=fraction, cap=cap)
    fill_store(store, 8)
    for _ in range(40):
        while store.ready_to_add():
            fill_store(store, 1)
        assert store.ready_to_sample(4)
        batch = store.sample(4)
        assert all(chunk.reuse_count.item() < cap for chunk in batch)
        # Producer speed must not consume the retention budget. Every
        # evicted chunk reached the cap, even at an infeasible fresh target.
        if store._evicted_count:
            assert store._evicted_reuses == store._evicted_count * cap


def test_unseen_chunks_cannot_be_evicted_and_dropped_add_not_counted():
    store = fresh_store(size=2)
    fill_store(store, 2)
    original_ids = store._ids.copy()
    assert not store.ready_to_add()
    fill_store(store, 1)
    np.testing.assert_array_equal(store._ids, original_ids)
    assert store.total_admitted_decisions == 4


def test_fresh_stream_under_a_dynamic_cap():
    store = fresh_store(size=2, fraction=0.5)
    fill_store(store, 2)
    store.sample(2)
    assert store.ready_to_sample(2)
    store.set_max_reuses(1)
    assert not store.ready_to_sample(2)
    assert store.ready_to_add()
    fill_store(store, 2)
    assert store.ready_to_sample(2)
    assert all(chunk.reuse_count.item() == 0 for chunk in store.sample(2))


def test_zero_fraction_preserves_uniform_sampler_and_replacement():
    store = fresh_store(size=4, fraction=0)
    fill_store(store, 4)
    np.random.seed(728)
    expected_slots = np.random.choice(np.arange(4), size=2, replace=False)
    np.random.seed(728)
    sampled = store.sample(2)
    # Slots were filled in admission order, so the slot index is the tag.
    assert [tag_of(chunk) for chunk in sampled] == list(expected_slots)
    assert not store.ready_to_add()
    assert store.ready_to_sample(4)


def test_accounting_distinguishes_admission_prefetch_skips_and_clear():
    store = fresh_store(size=2, fraction=0)
    store.add(trajectory([False] * 29 + [True] + [False] * 18))
    store.add(trajectory([False] * 64))
    store.sample(2, increment=False)
    assert store.total_samples == 0
    assert store.total_sampled_decisions == 0
    store.sample(2)
    store.sample(2)  # Prefetched, not yet processed.
    logs = {
        "player_batch_decisions": 92,
        "player_update_skipped": 0,
        "lifetime_step": 1_889_163,
    }
    store.record_decision_accounting(logs)
    assert logs["player_accounting_start_lifetime_step"] == 1_889_162
    assert logs["player_decisions_admitted_session"] == 92
    assert logs["player_decisions_sampled_session"] == 184
    assert logs["player_decisions_processed_session"] == 92
    assert logs["player_decisions_applied_session"] == 92
    assert logs["player_decision_reuse_session"] == 1
    assert logs["player_updates_per_fresh_decision_session"] == 1 / 92
    logs.update(player_update_skipped=1, lifetime_step=1_889_164)
    store.record_decision_accounting(logs)
    assert logs["player_decisions_processed_session"] == 184
    assert logs["player_decisions_applied_session"] == 92
    assert logs["player_updates_processed_session"] == 2
    assert logs["player_updates_applied_session"] == 1
    assert logs["player_replay_fresh_chunk_fraction_session"] == 0.5
    assert logs["player_replay_fresh_decision_fraction_session"] == 0.5
    store.clear()
    assert store.total_admitted_decisions == 0
    assert store.total_sampled_decisions == 0
    assert store.total_processed_updates == 0
    assert store.accounting_start_step is None


def test_log_worker_publishes_accounting_without_replay_controller():
    import queue

    store = fresh_store(size=1)
    fill_store(store, 1)
    store.sample(1)
    logs = queue.Queue()
    logs.put(
        {"player_batch_decisions": 2, "player_update_skipped": 0, "lifetime_step": 9}
    )
    logs.put(None)
    published = []
    run_state = SimpleNamespace(
        player_replay=store, log_q=logs, wandb_run=SimpleNamespace(log=published.append)
    )
    wandb_log_worker(run_state, SimpleNamespace(player_replay_ctrl_enabled=False))
    assert len(published) == 1
    assert published[0]["player_decisions_applied_session"] == 2


@pytest.mark.parametrize("fraction", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_fresh_fraction_rejected(fraction):
    with pytest.raises(ValueError, match="fresh_fraction"):
        fresh_store(fraction=fraction)
