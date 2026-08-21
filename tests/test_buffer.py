"""PlayerTrajectoryStore: reuse caps, replacement, and staleness stamping."""

import numpy as np

from rl.environment.interfaces import Trajectory
from rl.online.buffer import PlayerTrajectoryStore


def make_store(max_size=4, max_reuses=2) -> PlayerTrajectoryStore:
    return PlayerTrajectoryStore(max_size=max_size, max_reuses=max_reuses)


def fill(store: PlayerTrajectoryStore, n: int):
    for _ in range(n):
        store.add(Trajectory())


def test_empty_store_not_ready():
    store = make_store()
    assert not store.ready_to_sample()
    assert store.ready_to_add()
    assert len(store) == 0


def test_fill_and_capacity():
    store = make_store(max_size=4)
    fill(store, 4)
    assert len(store) == 4
    assert store.is_full()
    assert store.is_min_fill_fraction_reached(0.5)
    # Full and nothing reused: no slot is replaceable yet.
    assert not store.ready_to_add()


def test_sample_increments_reuse_and_respects_cap():
    store = make_store(max_size=2, max_reuses=2)
    fill(store, 2)
    for _ in range(2):  # each pass burns one reuse on both entries
        store.sample(2)
    assert not store.ready_to_sample()
    # Exhausted entries make room for new trajectories again.
    assert store.ready_to_add()


def test_sample_stamps_pre_increment_reuse_count():
    store = make_store(max_size=1, max_reuses=3)
    fill(store, 1)
    for expected in range(3):
        (traj,) = store.sample(1)
        assert traj.reuse_count == np.array([expected], dtype=np.int32)


def test_replacement_resets_reuse():
    store = make_store(max_size=1, max_reuses=1)
    fill(store, 1)
    store.sample(1)
    assert not store.ready_to_sample()
    fill(store, 1)  # replaces the exhausted slot
    assert store.ready_to_sample()
    assert len(store) == 1


def test_add_when_full_and_nothing_replaceable_is_dropped():
    store = make_store(max_size=1, max_reuses=5)
    fill(store, 2)
    assert len(store) == 1
    # A dropped add is not counted: total_adds tracks realised inserts,
    # which is what the replay-ratio controller diffs against samples.
    assert store.total_adds == 1


def test_ready_to_sample_n():
    store = make_store(max_size=4, max_reuses=1)
    fill(store, 3)
    assert store.ready_to_sample(3)
    assert not store.ready_to_sample(4)


def test_set_max_reuses_reopens_sampling():
    store = make_store(max_size=1, max_reuses=1)
    fill(store, 1)
    store.sample(1)
    assert not store.ready_to_sample()
    store.set_max_reuses(2)
    assert store.ready_to_sample()
    assert store.max_reuses == 2


def test_clear_resets_everything():
    store = make_store(max_size=4)
    fill(store, 3)
    store.sample(2)
    store.clear()
    assert len(store) == 0
    assert store.total_adds == 0
    assert store.total_samples == 0
    assert not store.ready_to_sample()
    assert store.ready_to_add()


def test_counters_track_adds_and_samples():
    store = make_store(max_size=4, max_reuses=10)
    fill(store, 4)
    store.sample(3)
    store.sample(2)
    assert store.total_adds == 4
    assert store.total_samples == 5
