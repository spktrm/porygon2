"""Per-chunk retention feedback: identity, masks, async delay and off control."""

import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.environment.interfaces import Trajectory
from rl.online.buffer import PlayerTrajectoryStore
from rl.online.training.batching import stack_batch
from rl.online.training.replay import chunk_policy_mismatch, consume_replay_feedback


def make_store(mode="protect", size=1, cap=8):
    store = PlayerTrajectoryStore(max_size=size, max_reuses=cap, trajectory_mode=mode)
    for _ in range(size):
        store.add(Trajectory())
    return store


def feedback(trajectory, measured_kl, count=10):
    return (
        trajectory.replay_slot,
        trajectory.replay_id,
        trajectory.reuse_count + 1,
        np.array([measured_kl * count]),
        np.array([count]),
    )


def test_protect_retires_only_stale_chunk_and_preserves_global_ceiling():
    store = make_store(size=2)
    first, second = store.sample(2)
    store.apply_feedback(*feedback(first, 0.06))
    store.apply_feedback(*feedback(second, 0.01))
    assert store.ready_to_add()
    assert not store.ready_to_sample(2)
    for _ in range(7):
        (sampled,) = store.sample(1)
        np.testing.assert_array_equal(sampled.replay_id, second.replay_id)
    assert not store.ready_to_sample()
    assert store.total_samples == 9
    store.set_max_reuses(9)
    (sampled,) = store.sample(1)
    np.testing.assert_array_equal(sampled.replay_id, second.replay_id)
    assert store.feedback_logs()["player_replay_trajectory_retired_total"] == 1


def test_prefetched_visits_remain_counted_and_old_feedback_cannot_reopen():
    store = make_store()
    first = store.sample(1)[0]
    second = store.sample(1)[0]
    store.apply_feedback(*feedback(second, 0.08))
    store.apply_feedback(*feedback(first, 0.001))
    assert store.total_samples == 2
    assert not store.ready_to_sample()
    store.add(Trajectory())
    store.apply_feedback(*feedback(second, 0.09))
    assert store.ready_to_sample()
    logs = store.feedback_logs()
    assert logs["player_replay_feedback_ignored"] == 2
    assert logs["player_replay_evicted_mean_reuses"] == 2


def test_clear_does_not_recycle_feedback_identity():
    store = make_store()
    old = store.sample(1)[0]
    store.clear()
    store.add(Trajectory())
    new = store.sample(1)[0]
    assert old.replay_id != new.replay_id
    store.apply_feedback(*feedback(old, 1.0))
    assert store.ready_to_sample()


@pytest.mark.parametrize(
    "measured_kl,count", [(float("nan"), 10), (float("inf"), 10), (-1.0, 10), (0.9, 0)]
)
def test_invalid_or_empty_feedback_does_not_retire(measured_kl, count):
    store = make_store()
    sampled = store.sample(1)[0]
    store.apply_feedback(*feedback(sampled, measured_kl, count))
    assert store.ready_to_sample()
    assert store.feedback_logs()["player_replay_feedback_ignored"] == 1


def test_observe_changes_neither_eligibility_nor_reuses():
    store = make_store(mode="observe", cap=3)
    first = store.sample(1)[0]
    store.apply_feedback(*feedback(first, 0.2))
    assert not store.ready_to_add()
    assert store.ready_to_sample()
    store.sample(1)
    store.sample(1)
    assert not store.ready_to_sample()
    assert store.feedback_logs()["player_replay_trajectory_threshold_crossings"] == 1
    assert store.feedback_logs()["player_replay_trajectory_retired_total"] == 0


def test_off_and_observe_use_identical_uniform_draws():
    states = []
    for mode in ("off", "observe"):
        store = make_store(mode=mode, size=4, cap=3)
        original = np.random.get_state()
        np.random.seed(73)
        for _ in range(5):
            store.sample(2)
        states.append(store._reuses.copy())
        np.random.set_state(original)
    np.testing.assert_array_equal(*states)
    assert isinstance(make_store(mode="off").sample(1)[0].replay_id, tuple)


def test_identity_survives_batching_and_feedback_is_not_logged():
    store = make_store(size=2)
    sampled = store.sample(2)
    batch = stack_batch(sampled)
    assert batch.replay_slot.shape == (1, 2)
    assert batch.replay_id.dtype == np.uint32
    payload = (
        batch.replay_slot[0],
        batch.replay_id[0],
        batch.reuse_count[0] + 1,
        np.array([0.8, 0.01]),
        np.array([10, 10]),
    )
    logs = {"_player_replay_feedback": payload, "player_update_skipped": 0}
    consume_replay_feedback(store, logs)
    assert "_player_replay_feedback" not in logs
    assert logs["player_replay_trajectory_retired_total"] == 1


def test_skipped_update_does_not_change_retention():
    store = make_store()
    sampled = store.sample(1)[0]
    logs = {
        "_player_replay_feedback": feedback(sampled, 1.0),
        "player_update_skipped": 1,
    }
    consume_replay_feedback(store, logs)
    assert "_player_replay_feedback" not in logs
    assert store.ready_to_sample()


def test_retirement_wakes_blocked_admission():
    store = make_store()
    sampled = store.sample(1)[0]
    waiting = threading.Event()
    admitted = threading.Event()

    def producer():
        with store._add_cv:
            waiting.set()
            if store._add_cv.wait_for(store.ready_to_add, timeout=2):
                admitted.set()

    thread = threading.Thread(target=producer)
    thread.start()
    assert waiting.wait(timeout=2)
    store.apply_feedback(*feedback(sampled, 0.1))
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert admitted.is_set()


def test_chunk_mismatch_respects_masks_and_has_no_gradient():
    ratios = jnp.array([[2.0, 0.5], [1.5, 3.0], [jnp.nan, 5.0]])
    mask = jnp.array([[True, True], [False, True], [False, False]])
    compute = jax.jit(chunk_policy_mismatch)
    totals, counts = compute(ratios, jnp.log(ratios), mask)
    expected = np.array([2 - 1 - np.log(2), 0.5 - 1 - np.log(0.5) + 3 - 1 - np.log(3)])
    np.testing.assert_allclose(totals, expected, rtol=1e-6)
    np.testing.assert_array_equal(counts, [1, 2])
    # Positive control: making the second row eligible changes the first sum.
    changed, _ = compute(ratios, jnp.log(ratios), mask.at[1, 0].set(True))
    assert changed[0] > totals[0]
    gradient = jax.grad(
        lambda values: chunk_policy_mismatch(
            values, jnp.log(values), jnp.ones_like(values, bool)
        )[0].sum()
    )(jnp.full((2, 2), 2.0))
    np.testing.assert_array_equal(gradient, np.zeros((2, 2)))


def test_invalid_configuration_and_malformed_feedback_fail_visibly():
    with pytest.raises(ValueError):
        make_store(mode="priority")
    store = make_store()
    with pytest.raises(ValueError):
        store.apply_feedback([0], [], [1], [0.1], [1])
