"""The PBRS potential channel (2026-09-11) on hand-built chunks, against a
numpy mirror of the recursion: the inert contract, the closed form, the
uncentred equivalence and the done row -- each beside the control that
proves it could fail. Threshold-discarded rows (rho 0, raw c 1), padding and
a non-terminal chunk are in the fixtures because the 2026-09-11 review found
claims that held only on the idealised rho == c estimator."""

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


def ratios(length, discarded):
    raw = np.ones((length, 1), np.float32)
    cut = raw.copy()
    cut[list(discarded)] = 0.0
    return jnp.asarray(cut), jnp.asarray(raw)


def mirror(reward, value, done, rho, c):
    """targets.scalar_vtrace in numpy, over compute_player_targets' masks."""
    mask = 1.0 - (np.cumsum(done) - done)
    discount = (1.0 - done) * CONFIG.player_gamma * mask
    value_next = np.concatenate([value[1:], value[-1:]])
    td = rho * mask * (reward + discount * value_next - value)
    errors = np.zeros_like(td)
    carry = 0.0
    for row in reversed(range(len(td))):
        carry = td[row] + discount[row] * c[row] * LAMBDA * carry
        errors[row] = carry
    trace = errors + value
    bootstrap = np.concatenate(
        [LAMBDA * trace[1:] + (1 - LAMBDA) * value[1:], value[-1:]]
    )
    advantages = rho * (reward + discount * bootstrap - value) * mask
    return trace * mask, advantages


def run(batch, value_log_probs, cut, raw, head):
    targets, _ = compute_player_targets(
        batch,
        value_log_probs,
        cut,
        CONFIG,
        isr_raw=raw,
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
    cut, raw = ratios(length, discarded=(3, 6))
    rng = np.random.default_rng(2)
    # Exact on live rows; arbitrary elsewhere, which the forcing must ignore.
    exact = np.where(live_rows(done), -unit, rng.uniform(-3, 3, length))
    advantages, _, total = run(batch, value_log_probs, cut, raw, exact)
    np.testing.assert_allclose(advantages, 0.0, atol=1e-7)
    plain, _ = compute_player_targets(batch, value_log_probs, cut, CONFIG, isr_raw=raw)
    np.testing.assert_allclose(total, np.asarray(plain.pg_advantages)[:, 0], atol=1e-7)
    # Control: an unfitted head leaves a real force.
    idle, _, _ = run(batch, value_log_probs, cut, raw, np.zeros(length))
    assert np.abs(idle).max() > 1e-3


def test_an_unfitted_head_gives_the_closed_form_on_policy() -> None:
    length, done_row = 11, 10
    batch, value_log_probs, unit, done = make_chunk(length, done_row, seed=3)
    cut, raw = ratios(length, discarded=())
    advantages, _, _ = run(batch, value_log_probs, cut, raw, np.zeros(length))
    psi = STRENGTH * unit * live_rows(done)
    for row in range(done_row):
        future = sum(
            LAMBDA ** (k - 1) * psi[row + k] for k in range(1, done_row - row + 1)
        )
        np.testing.assert_allclose(
            advantages[row], -psi[row] + (1 - LAMBDA) * future, atol=1e-6
        )


def test_the_closed_form_breaks_across_a_discarded_row() -> None:
    """Control for the test above: why the inert contract, not the closed
    form, is what the threshold-discard fixtures check."""
    length, done_row = 11, 10
    batch, value_log_probs, unit, done = make_chunk(length, done_row, seed=3)
    cut, raw = ratios(length, discarded=(6,))
    advantages, _, _ = run(batch, value_log_probs, cut, raw, np.zeros(length))
    psi = STRENGTH * unit * live_rows(done)
    closed = [
        -psi[row]
        + (1 - LAMBDA)
        * sum(LAMBDA ** (k - 1) * psi[row + k] for k in range(1, done_row - row + 1))
        for row in range(6)
    ]
    assert np.abs(advantages[:6] - np.asarray(closed)).max() > 1e-4


def test_uncentred_equals_centred_with_the_offset_on_the_value() -> None:
    length, done_row = 12, 9
    batch, value_log_probs, unit, done = make_chunk(length, done_row, seed=4)
    cut, raw = ratios(length, discarded=(2, 5))
    rho, c = np.minimum(1, np.asarray(cut)[:, 0]), np.minimum(1, np.asarray(raw)[:, 0])
    live = live_rows(done)
    head = np.random.default_rng(5).uniform(-1, 1, length)
    advantages, returns, _ = run(batch, value_log_probs, cut, raw, head)

    mask = 1.0 - (np.cumsum(done) - done)
    discount = (1.0 - done) * mask
    offset = STRENGTH * unit[0] * live
    psi_centred = (STRENGTH * unit - STRENGTH * unit[0]) * live
    shaping = (
        discount * np.concatenate([psi_centred[1:], psi_centred[-1:]]) - psi_centred
    )
    assert abs(shaping.sum()) < 1e-12  # centred rewards sum to zero
    centred_returns, centred_advantages = mirror(
        shaping, STRENGTH * head * live + offset, done, rho, c
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
    _, plain_head = mirror(shaping, STRENGTH * head * live, done, rho, c)
    assert np.abs(plain_head - advantages).max() > 1e-4


def test_the_done_row_channel_label_is_exactly_zero() -> None:
    length, done_row = 10, 7
    batch, value_log_probs, unit, done = make_chunk(length, done_row, seed=6)
    cut, raw = ratios(length, discarded=())
    cut = cut.at[done_row].set(0.5)  # a sampled ratio on a no-decision row
    head = np.random.default_rng(7).uniform(-1, 1, length)  # nonzero on done
    _, returns, _ = run(batch, value_log_probs, cut, raw, head)
    assert returns[done_row] == 0.0
    # Control: an unforced value on the done row would leak into its label.
    live = live_rows(done)
    psi = STRENGTH * unit * live
    mask = 1.0 - (np.cumsum(done) - done)
    shaping = (1.0 - done) * mask * np.concatenate([psi[1:], psi[-1:]]) - psi
    rho = np.minimum(1, np.asarray(cut)[:, 0])
    unforced = STRENGTH * head * mask
    unforced_returns, _ = mirror(shaping, unforced, done, rho, np.ones(length))
    assert abs(unforced_returns[done_row]) > 1e-4


def test_potential_values_need_a_positive_strength() -> None:
    batch, value_log_probs, _, _ = make_chunk(6, 4, seed=8)
    cut, raw = ratios(6, discarded=())
    with pytest.raises(ValueError):
        compute_player_targets(
            batch,
            value_log_probs,
            cut,
            Porygon2LearnerConfig(),
            isr_raw=raw,
            potential_values=jnp.zeros((6, 1)),
        )
