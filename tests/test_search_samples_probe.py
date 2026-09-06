"""The calibration accounting of `rl/offline/search_samples_probe.py`: the
uncentred delta read matches the learner's copy = 0 contract, the
centred read does not (the 2026-09-06 mis-read), the exact code
enumeration weights are a distribution whose expectation is the exact
mixture, and the bootstrap keeps a game's sides together."""

import numpy as np

from rl.offline.search_samples_probe import (
    Transition,
    bootstrap_delta_stats,
    code_grid,
    delta_gain,
    r2_centred,
    resample_games,
)


def _target_with_mean():
    rng = np.random.default_rng(0)
    return rng.normal(0.1, 0.05, 200).astype(np.float32)


def test_uncentred_delta_gain_scores_copy_zero_and_the_target_one():
    target = _target_with_mean()
    assert abs(delta_gain(np.zeros_like(target), target)) < 1e-6
    assert delta_gain(target, target) == 1.0


def test_centred_r2_scores_copy_below_zero_when_the_target_has_a_mean():
    target = _target_with_mean()
    expected = -target.size * target.mean() ** 2 / np.sum((target - target.mean()) ** 2)
    assert r2_centred(0.0, target) < -0.5
    assert np.isclose(r2_centred(0.0, target), expected, atol=1e-5)
    assert r2_centred(target, target) == 1.0
    # A zero-mean target is the case where the two reads agree.
    centred = target - target.mean()
    assert abs(r2_centred(0.0, centred) - delta_gain(0.0, centred)) < 1e-5


def test_code_grid_enumerates_every_joint_code_once():
    grid = np.asarray(code_grid(2, 3))
    assert grid.shape == (9, 2, 3)
    assert (grid.sum(-1) == 1).all()
    joint = {tuple(row.argmax(-1)) for row in grid}
    assert len(joint) == 9


def test_enumeration_weights_are_the_product_prior_and_give_the_exact_mixture():
    grid = np.asarray(code_grid(2, 3))
    probs = np.asarray([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]], np.float32)
    weights = np.prod(np.sum(grid * probs[None], -1), -1)
    assert np.isclose(weights.sum(), 1.0)
    values = np.arange(9, dtype=np.float32)
    exact = sum(
        probs[0, first] * probs[1, second] * values[first * 3 + second]
        for first in range(3)
        for second in range(3)
    )
    assert np.isclose(weights @ values, exact)


def _transition(game, value):
    values = {"root_v": np.float32(0.0), "real_v": np.float32(value)}
    for name in ("post", "prior", "expect", "sample"):
        values[f"{name}_v"] = np.float32(value)
    return Transition(
        values=values,
        outcome=1.0,
        is_switch=False,
        game=game,
        kind=1,
        next_kind=1,
        newly_valid=False,
    )


def test_resample_games_keeps_both_sides_of_a_game_together():
    transitions = [_transition(game, 0.1 * game) for game in range(5) for _ in range(2)]
    rng = np.random.default_rng(0)
    for _ in range(20):
        replicate = resample_games(transitions, rng)
        assert len(replicate) == len(transitions)
        counts = {}
        for transition in replicate:
            counts[transition.game] = counts.get(transition.game, 0) + 1
        assert all(count % 2 == 0 for count in counts.values())


def test_bootstrap_interval_brackets_an_exact_read():
    transitions = [_transition(game, 0.1 + 0.01 * game) for game in range(8)]
    stats = bootstrap_delta_stats(transitions, replicates=50, seed=0)
    assert stats["value_delta_r2_post_lo"] == 1.0
    assert stats["value_delta_r2_post_hi"] == 1.0
    assert stats["copy_delta_r2_centred_hi"] < 0.0
