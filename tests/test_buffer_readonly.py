"""sample_readonly contract: reads must be invisible to the owning
store's replay accounting (stage-4 cross-population intake — main's
Q-loss reading an exploiter's buffer must not burn that population's own
reuse budget)."""

import numpy as np

from rl.environment.interfaces import Trajectory
from rl.online.buffer import PlayerTrajectoryStore


def test_sample_readonly_leaves_accounting_untouched():
    store = PlayerTrajectoryStore(max_size=8, max_reuses=2)
    for i in range(4):
        store.add(Trajectory(reuse_count=np.array([i], dtype=np.int32)))

    got = store.sample_readonly(3)
    assert len(got) == 3
    assert store.total_samples == 0
    assert store._reuses[:4].sum() == 0

    # Over-ask returns what exists; an empty store returns nothing.
    assert len(store.sample_readonly(99)) == 4
    assert len(PlayerTrajectoryStore(max_size=2).sample_readonly(1)) == 0

    # Reuse-exhausted entries stay readable — staleness is the reader's
    # (Retrace's) problem, not this store's eligibility filter — and the
    # reads still leave the accounting untouched.
    store._reuses[:4] = 99
    assert len(store.sample_readonly(2)) == 2
    assert store.total_samples == 0
