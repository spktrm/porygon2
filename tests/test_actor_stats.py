"""ActorStats: the actors' shared timing sink (rl/environment/actor_stats.py)."""

from rl.environment.actor_stats import STEP_TOTAL, ActorStats, timed


def test_drain_returns_means_and_resets():
    stats = ActorStats()
    stats.record("actor_time_inference", 2.0)
    stats.record("actor_time_inference", 4.0)
    stats.record(STEP_TOTAL, 10.0)
    stats.record("actor_time_service_wait", 1.0)
    drained = stats.drain()
    assert drained["actor_time_inference"] == 3.0
    # other = step_total - the attributed parts that were recorded.
    assert drained["actor_time_other"] == 10.0 - (1.0 + 3.0)
    assert drained["actor_steps_per_sec"] > 0.0
    assert stats.drain() == {}


def test_timed_records_milliseconds_and_no_sink_is_a_noop():
    stats = ActorStats()
    with stats.timed("phase"):
        pass
    drained = stats.drain()
    assert 0.0 <= drained["phase"] < 100.0
    with timed(None, "phase"):
        pass
