"""The consequence losses' contracts (rl/online/training/loss.py), one set
per head: they are different scores with different reference points."""

import jax
import jax.numpy as jnp
import numpy as np

from rl.model.consequence import (
    CONSEQUENCE_GROUP_IDS,
    NUM_CONSEQUENCE_ROWS,
    public_row_alignment,
)
from rl.model.constants import NUM_PUBLIC_SLOTS
from rl.online.training.loss import (
    consequence_energy_loss,
    consequence_mean_loss,
    smoothed_norm,
)

GROUPS = jnp.asarray(CONSEQUENCE_GROUP_IDS)
UNIT = jnp.ones(3)
WIDTH = 8


def _rows(key: int) -> jax.Array:
    return jax.random.normal(jax.random.key(key), (5, NUM_CONSEQUENCE_ROWS, WIDTH))


def _mask() -> jax.Array:
    return jnp.ones((5, NUM_CONSEQUENCE_ROWS), bool)


def test_mean_loss_is_the_pooled_squared_error_over_the_fixed_scale() -> None:
    now, real_next = _rows(0), _rows(1)
    exact = consequence_mean_loss(real_next, now, real_next, _mask(), GROUPS, UNIT)
    np.testing.assert_allclose(exact["loss"], 0.0, atol=1e-6)
    copy = consequence_mean_loss(now, now, real_next, _mask(), GROUPS, UNIT)
    np.testing.assert_allclose(copy["ratio"], copy["copy_ratio"], rtol=1e-6)
    # The scale divides squared: doubling it quarters the loss.
    doubled = consequence_mean_loss(now, now, real_next, _mask(), GROUPS, 2 * UNIT)
    np.testing.assert_allclose(doubled["loss"], copy["loss"] / 4, rtol=1e-6)


def test_a_bigger_true_change_cannot_lower_the_loss() -> None:
    """The defect the fixed scale removes. Hold the prediction's ERROR fixed
    and let the rows change more between steps: the loss must not fall."""
    now = _rows(0)
    error = 0.5 * _rows(2)
    small_next = now + _rows(3)
    big_next = now + 8 * _rows(3)
    losses = [
        float(
            consequence_mean_loss(
                real_next + error, now, real_next, _mask(), GROUPS, UNIT
            )["loss"]
        )
        for real_next in (small_next, big_next)
    ]
    np.testing.assert_allclose(losses[0], losses[1], rtol=1e-5)
    # The control: the ratio this replaced, error over the batch's OWN change,
    # falls 64-fold on the same inputs -- inflating the rows was rewarded.
    live = [
        float(jnp.square(error).sum() / jnp.square(real_next - now).sum())
        for real_next in (small_next, big_next)
    ]
    assert live[1] < live[0] / 50


def test_energy_loss_scores_exact_zero_and_copy_as_the_change_norm() -> None:
    now, real_next = _rows(0), _rows(1)
    exact = consequence_energy_loss(
        jnp.stack((real_next, real_next)), now, real_next, _mask(), GROUPS, UNIT
    )
    np.testing.assert_allclose(exact["loss"], 0.0, atol=2e-3)
    copy = consequence_energy_loss(
        jnp.stack((now, now)), now, real_next, _mask(), GROUPS, UNIT
    )
    np.testing.assert_allclose(copy["ratio"], copy["copy_ratio"], rtol=1e-5)
    np.testing.assert_allclose(copy["ratio"], copy["change_norm"], rtol=1e-5)


def test_energy_loss_has_finite_gradients_where_the_draws_coincide() -> None:
    now, real_next = _rows(0), _rows(1)

    def score(draw, norm):
        samples = jnp.stack((draw, draw))
        skill = norm(samples - real_next[None]).mean(axis=0)
        spread = norm(samples[0] - samples[1])
        return (skill - 0.5 * spread).sum()

    smoothed = jax.grad(score)(now, smoothed_norm)
    assert np.isfinite(np.asarray(smoothed)).all()
    through_loss = jax.grad(
        lambda draw: consequence_energy_loss(
            jnp.stack((draw, draw)), now, real_next, _mask(), GROUPS, UNIT
        )["loss"]
    )(now)
    assert np.isfinite(np.asarray(through_loss)).all()
    assert np.abs(np.asarray(through_loss)).max() > 0
    # The control: the plain norm is 0/0 exactly there, which is the state the
    # sampler is initialised in.
    plain = jax.grad(score)(now, lambda vector: jnp.linalg.norm(vector, axis=-1))
    assert not np.isfinite(np.asarray(plain)).all()


def test_energy_loss_prefers_the_two_modes_to_their_mean() -> None:
    # Half the transitions go +mode, half -mode; nothing observable tells
    # them apart. Draws on the two modes beat the mean predictor, which is
    # what makes the score a reason to be stochastic at all.
    mode = jnp.ones((NUM_CONSEQUENCE_ROWS, WIDTH))
    real_next = jnp.stack((mode, -mode, mode, -mode))
    now = jnp.zeros_like(real_next)
    mask = jnp.ones((4, NUM_CONSEQUENCE_ROWS), bool)
    on_the_modes = jnp.stack(
        (
            jnp.broadcast_to(mode, real_next.shape),
            jnp.broadcast_to(-mode, real_next.shape),
        )
    )
    at_the_mean = jnp.zeros((2,) + real_next.shape)
    modes = consequence_energy_loss(on_the_modes, now, real_next, mask, GROUPS, UNIT)
    mean = consequence_energy_loss(at_the_mean, now, real_next, mask, GROUPS, UNIT)
    assert float(modes["loss"]) < float(mean["loss"])
    np.testing.assert_allclose(mean["spread_over_skill"], 0.0, atol=1e-6)
    assert (np.asarray(modes["spread_over_skill"]) > 0.5).all()
    # Squared error ranks them the other way: its optimum IS the mean, which
    # is why that head is the control and not the sampler.
    assert float(
        consequence_mean_loss(at_the_mean[0], now, real_next, mask, GROUPS, UNIT)[
            "loss"
        ]
    ) < float(
        consequence_mean_loss(on_the_modes[0], now, real_next, mask, GROUPS, UNIT)[
            "loss"
        ]
    )


def test_an_empty_group_is_left_out_of_the_mean() -> None:
    now, real_next = _rows(0), _rows(1)
    mask = _mask().at[:, GROUPS == 1].set(False)
    terms = consequence_mean_loss(now, now, real_next, mask, GROUPS, UNIT)
    assert float(terms["ratio"][1]) == 0.0
    present = np.asarray(terms["ratio"])[[0, 2]]
    np.testing.assert_allclose(terms["loss"], present.mean(), rtol=1e-6)


def test_alignment_follows_identity_through_a_reorder() -> None:
    order_now = jnp.asarray([3, 5, -1, -1, -1, -1, 7, 9, -1, -1, -1, -1])
    # My active switched: identities 3 and 5 trade rows; their side is still.
    order_next = jnp.asarray([5, 3, -1, -1, -1, -1, 7, 9, 11, -1, -1, -1])
    next_index, matched = public_row_alignment(order_now, order_next)
    np.testing.assert_array_equal(next_index[:2], [1, 0])
    np.testing.assert_array_equal(next_index[6:8], [6, 7])
    # Unrevealed rows never match, even though -1 == -1.
    assert not np.asarray(matched)[[2, 3, 8]].any()
    assert np.asarray(matched)[[0, 1, 6, 7]].all()
    # Field rows and the value row are fixed slots.
    np.testing.assert_array_equal(
        next_index[NUM_PUBLIC_SLOTS:], np.arange(NUM_PUBLIC_SLOTS, NUM_CONSEQUENCE_ROWS)
    )
    assert np.asarray(matched)[NUM_PUBLIC_SLOTS:].all()
