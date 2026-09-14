"""Potential-channel invariance with current-policy V-trace and chunk boundaries."""

import jax.numpy as jnp
import numpy as np
import pytest

from rl.environment.data import MAX_RATIO_TOKEN
from rl.environment.interfaces import Batch, PlayerEnvOutput, PlayerTransition
from rl.environment.protos.features_pb2 import InfoFeature
from rl.online.config import Porygon2LearnerConfig
from rl.online.training.targets import compute_player_targets

STRENGTH = 0.05
CONFIG = Porygon2LearnerConfig().replace(player_potential_strength=STRENGTH)
LAMBDA = CONFIG.player_lambda
NUM_INFO = max(InfoFeature.values()) + 1


def make_chunk(length, done_row, seed):
    """A (T, 1) chunk: unit potentials, a win on done_row (None: a chunk
    whose last row is bootstrap-only) and padding after it that copies the
    terminal row with done cleared, as player_actor.make_chunk does."""
    rng = np.random.default_rng(seed)
    unit = np.round(rng.uniform(-0.9, 0.9, length) * MAX_RATIO_TOKEN)
    done = np.zeros(length, bool)
    win = np.zeros((length, 3), np.float32)
    if done_row is not None:
        done[done_row] = True
        win[done_row:, 2] = 1.0
        unit[done_row + 1 :] = unit[done_row]
    info = np.zeros((length, 1, NUM_INFO), np.int32)
    info[:, 0, InfoFeature.INFO_FEATURE__STATE_POTENTIAL] = unit
    env = PlayerEnvOutput(
        info=jnp.asarray(info),
        done=jnp.asarray(done[:, None]),
        win_reward=jnp.asarray(win[:, None]),
        action_mask=jnp.ones((length, 1, 4), bool),
    )
    logits = rng.normal(size=(length, 1, 3)).astype(np.float32)
    value_log_probs = jnp.asarray(
        logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    )
    return (
        Batch(player_transitions=PlayerTransition(env_output=env)),
        value_log_probs,
        unit / MAX_RATIO_TOKEN,
        done,
    )


def ratios(length, zero_rows):
    ratio = np.ones((length, 1), np.float32)
    ratio[list(zero_rows)] = 0.0
    return jnp.asarray(ratio)


def mirror(reward, value, done, rho, continuation):
    """targets.scalar_vtrace in numpy, over compute_player_targets' masks."""
    mask = 1.0 - (np.cumsum(done) - done)
    discount = (1.0 - done) * CONFIG.player_gamma * mask
    mask[-1] = float(done[-1])
    rho = np.where(done, 1.0, rho)
    value_next = np.concatenate([value[1:], value[-1:]])
    td = rho * mask * (reward + discount * value_next - value)
    errors = np.zeros_like(td)
    carry = 0.0
    for row in reversed(range(len(td))):
        carry = td[row] + discount[row] * continuation[row] * LAMBDA * carry
        errors[row] = carry
    trace = errors + value
    bootstrap = np.concatenate([trace[1:], value[-1:]])
    advantages = rho * (reward + discount * bootstrap - value) * mask
    return trace * mask, advantages


def run(batch, value_log_probs, ratio, head):
    targets, _ = compute_player_targets(
        batch,
        value_log_probs,
        ratio,
        CONFIG,
        potential_values=jnp.asarray(np.asarray(head, np.float32)[:, None]),
    )
    return (
        np.asarray(targets.potential_advantages)[:, 0],
        np.asarray(targets.potential_returns)[:, 0],
        np.asarray(targets.pg_advantages)[:, 0],
    )


def live_rows(done):
    return (np.cumsum(done) - done == 0) & ~done


@pytest.mark.parametrize("done_row", [9, None])
def test_an_exact_head_makes_the_channel_inert(done_row) -> None:
    length = 13
    batch, value_log_probs, unit, done = make_chunk(length, done_row, seed=1)
    ratio = ratios(length, zero_rows=(3, 6))
    rng = np.random.default_rng(2)
    # Exact on live rows; arbitrary elsewhere, which the forcing must ignore.
    exact = np.where(live_rows(done), -unit, rng.uniform(-3, 3, length))
    advantages, _, total = run(batch, value_log_probs, ratio, exact)
    np.testing.assert_allclose(advantages, 0.0, atol=1e-7)
    plain, _ = compute_player_targets(batch, value_log_probs, ratio, CONFIG)
    np.testing.assert_allclose(total, np.asarray(plain.pg_advantages)[:, 0], atol=1e-7)
    # Control: an unfitted head leaves a real force.
    idle, _, _ = run(batch, value_log_probs, ratio, np.zeros(length))
    assert np.abs(idle).max() > 1e-3


def test_an_unfitted_head_gives_the_closed_form_on_policy() -> None:
    length, done_row = 11, 10
    batch, value_log_probs, unit, done = make_chunk(length, done_row, seed=3)
    ratio = ratios(length, zero_rows=())
    advantages, _, _ = run(batch, value_log_probs, ratio, np.zeros(length))
    psi = STRENGTH * unit * live_rows(done)
    for row in range(done_row):
        future = sum(
            LAMBDA ** (offset - 2) * psi[row + offset]
            for offset in range(2, done_row - row + 1)
        )
        np.testing.assert_allclose(
            advantages[row], -psi[row] + (1 - LAMBDA) * future, atol=1e-6
        )


def test_the_closed_form_breaks_across_a_zero_importance_row() -> None:
    length, done_row = 11, 10
    batch, value_log_probs, unit, done = make_chunk(length, done_row, seed=3)
    ratio = ratios(length, zero_rows=(6,))
    advantages, _, _ = run(batch, value_log_probs, ratio, np.zeros(length))
    psi = STRENGTH * unit * live_rows(done)
    closed = [
        -psi[row]
        + (1 - LAMBDA)
        * sum(
            LAMBDA ** (offset - 2) * psi[row + offset]
            for offset in range(2, done_row - row + 1)
        )
        for row in range(6)
    ]
    assert np.abs(advantages[:6] - np.asarray(closed)).max() > 1e-4


def test_uncentred_equals_centred_with_the_offset_on_the_value() -> None:
    length, done_row = 12, 9
    batch, value_log_probs, unit, done = make_chunk(length, done_row, seed=4)
    ratio = ratios(length, zero_rows=(2, 5))
    rho = np.minimum(1, np.asarray(ratio)[:, 0])
    continuation = rho
    live = live_rows(done)
    head = np.random.default_rng(5).uniform(-1, 1, length)
    advantages, returns, _ = run(batch, value_log_probs, ratio, head)

    mask = 1.0 - (np.cumsum(done) - done)
    discount = (1.0 - done) * mask
    offset = STRENGTH * unit[0] * live
    psi_centred = (STRENGTH * unit - STRENGTH * unit[0]) * live
    shaping = (
        discount * np.concatenate([psi_centred[1:], psi_centred[-1:]]) - psi_centred
    )
    assert abs(shaping.sum()) < 1e-12  # centred rewards sum to zero
    centred_returns, centred_advantages = mirror(
        shaping, STRENGTH * head * live + offset, done, rho, continuation
    )
    np.testing.assert_allclose(advantages, centred_advantages, atol=1e-6)
    np.testing.assert_allclose(
        returns[live], ((centred_returns - offset) / STRENGTH)[live], atol=1e-5
    )
    # The uncentred rewards sum to -eta * Phi(h0) instead: action-independent.
    psi = STRENGTH * unit * live
    uncentred = discount * np.concatenate([psi[1:], psi[-1:]]) - psi
    np.testing.assert_allclose(uncentred.sum(), -STRENGTH * unit[0], atol=1e-12)
    # Control: centred rewards with the plain head are NOT the same learner.
    _, plain_head = mirror(shaping, STRENGTH * head * live, done, rho, continuation)
    assert np.abs(plain_head - advantages).max() > 1e-4


def test_the_done_row_channel_label_is_exactly_zero() -> None:
    length, done_row = 10, 7
    batch, value_log_probs, unit, done = make_chunk(length, done_row, seed=6)
    ratio = ratios(length, zero_rows=())
    ratio = ratio.at[done_row].set(0.5)  # a sampled ratio on a no-decision row
    head = np.random.default_rng(7).uniform(-1, 1, length)  # nonzero on done
    _, returns, _ = run(batch, value_log_probs, ratio, head)
    assert returns[done_row] == 0.0
    # A nonzero terminal potential value contaminates earlier trace residuals.
    live = live_rows(done)
    psi = STRENGTH * unit * live
    mask = 1.0 - (np.cumsum(done) - done)
    shaping = (1.0 - done) * mask * np.concatenate([psi[1:], psi[-1:]]) - psi
    rho = np.minimum(1, np.asarray(ratio)[:, 0])
    unforced = STRENGTH * head * mask
    unforced_returns, _ = mirror(shaping, unforced, done, rho, np.ones(length))
    assert abs(unforced_returns[done_row]) < 1e-7
    assert (
        np.max(np.abs(unforced_returns[:done_row] / STRENGTH - returns[:done_row]))
        > 1e-4
    )


def test_potential_values_need_a_positive_strength() -> None:
    batch, value_log_probs, _, _ = make_chunk(6, 4, seed=8)
    ratio = ratios(6, zero_rows=())
    with pytest.raises(ValueError):
        compute_player_targets(
            batch,
            value_log_probs,
            ratio,
            Porygon2LearnerConfig().replace(player_potential_strength=0.0),
            potential_values=jnp.zeros((6, 1)),
        )
