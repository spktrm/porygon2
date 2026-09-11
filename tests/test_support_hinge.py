"""The flat support hinge (rl/online/training/loss.py support_hinge_loss):
its contract, its derivative, and the controls that prove each test could
fail."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.environment.data import MOVE_CELL_OFFSET, NUM_ACTION_CELLS
from rl.model.utils import legal_log_policy
from rl.online.training.loss import support_hinge_loss

TAU = 0.01
TAU_MAX_MASS = 0.5
TEMPERATURE = 0.1


def _legal(cells: list[int], rows: int = 1) -> jax.Array:
    legal = np.zeros((rows, NUM_ACTION_CELLS), bool)
    legal[:, cells] = True
    return jnp.asarray(legal)


def _loss(
    legal: jax.Array,
    tau: float = TAU,
    tau_max_mass: float = TAU_MAX_MASS,
    temperature: float = 0.0,
) -> Callable[[jax.Array], jax.Array]:
    def loss(logits: jax.Array) -> jax.Array:
        rows, _, _ = support_hinge_loss(
            legal_log_policy(logits, legal),
            legal,
            tau,
            tau_max_mass,
            temperature=temperature,
        )
        return rows.sum()

    return loss


FOUR = [0, 1, MOVE_CELL_OFFSET, MOVE_CELL_OFFSET + 1]


def test_silent_above_tau_and_positive_on_an_abandoned_cell() -> None:
    legal = _legal(FOUR)
    logits = jnp.zeros((1, NUM_ACTION_CELLS))  # uniform .25 over legal
    rows, active, tau_row = support_hinge_loss(
        legal_log_policy(logits, legal), legal, TAU, TAU_MAX_MASS
    )
    assert float(rows[0]) == 0.0 and float(active[0]) == 0.0
    np.testing.assert_allclose(float(tau_row[0]), TAU)
    # Positive control: one cell driven to ~1e-4.
    starved = logits.at[0, MOVE_CELL_OFFSET + 1].set(-9.0)
    rows, active, _ = support_hinge_loss(
        legal_log_policy(starved, legal), legal, TAU, TAU_MAX_MASS
    )
    assert float(rows[0]) > 0.0
    np.testing.assert_allclose(float(active[0]), 0.25)


def test_coefficient_zero_and_forced_rows_are_exactly_off() -> None:
    # A singleton row has pi = 1 >= tau: no loss, no gradient, no mask.
    legal = _legal([3])
    logits = jnp.linspace(-3.0, 3.0, NUM_ACTION_CELLS)[None]
    assert float(_loss(legal)(logits)) == 0.0
    assert not np.any(np.asarray(jax.grad(_loss(legal))(logits)))


@pytest.mark.parametrize("temperature", [0.0, TEMPERATURE])
def test_illegal_cells_are_never_scored(temperature: float) -> None:
    legal = _legal(FOUR)
    logits = jnp.zeros((1, NUM_ACTION_CELLS)).at[0, MOVE_CELL_OFFSET + 1].set(-9.0)
    loss = _loss(legal, temperature=temperature)
    # Moving an illegal logit by a lot changes nothing; its gradient is 0.
    shifted = logits.at[0, 5].set(40.0)
    assert float(loss(shifted)) == float(loss(logits))
    gradient = np.asarray(jax.grad(loss)(logits))[0]
    assert gradient[5] == 0.0 and gradient[MOVE_CELL_OFFSET + 1] != 0.0


def test_permutation_invariant_over_cells() -> None:
    legal = _legal(FOUR)
    logits = jax.random.normal(jax.random.key(1), (1, NUM_ACTION_CELLS)) * 4.0
    permutation = jax.random.permutation(jax.random.key(2), NUM_ACTION_CELLS)
    base = float(_loss(legal)(logits))
    assert base > 0.0  # some cell is under the line at this spread
    permuted = float(_loss(legal[:, permutation])(logits[:, permutation]))
    np.testing.assert_allclose(permuted, base, rtol=1e-6)


def test_derivative_is_active_fraction_pi_minus_below_over_n_and_zero_sum() -> None:
    legal = _legal(FOUR)
    logits = jnp.asarray([[0.0] * NUM_ACTION_CELLS])
    logits = logits.at[0, FOUR].set([2.0, 1.0, -4.0, -6.0])
    log_policy = legal_log_policy(logits, legal)
    pi = np.asarray(jnp.exp(log_policy))[0]
    below = (pi < TAU) & np.asarray(legal)[0]
    active_fraction = below.sum() / 4
    expected = np.where(np.asarray(legal)[0], active_fraction * pi - below / 4, 0.0)
    gradient = np.asarray(jax.grad(_loss(legal))(logits))[0]
    assert below.sum() == 2  # the control: the formula is exercised
    np.testing.assert_allclose(gradient, expected, atol=1e-6)
    np.testing.assert_allclose(gradient.sum(), 0.0, atol=1e-6)
    assert np.all(np.abs(gradient) <= 1.0)


def test_the_hinge_takes_the_zero_subgradient() -> None:
    # A cell exactly at tau_row is inactive: no loss and no gradient from it.
    legal = _legal(FOUR)
    logits = jnp.zeros((1, NUM_ACTION_CELLS))
    logits = logits.at[0, FOUR].set(jnp.log(jnp.asarray([0.5, 0.3, 0.19, TAU])))
    log_policy = legal_log_policy(logits, legal)
    rows, active, _ = support_hinge_loss(log_policy, legal, TAU, TAU_MAX_MASS)
    # log-softmax rounding can land a hair either side of tau; pin the
    # contract at the exact value by forcing the log-policy itself.
    exact = log_policy.at[0, MOVE_CELL_OFFSET + 1].set(jnp.log(TAU))
    rows, active, _ = support_hinge_loss(exact, legal, TAU, TAU_MAX_MASS)
    assert float(rows[0]) == 0.0 and float(active[0]) == 0.0


def test_feasibility_clamp_binds_on_a_wide_row() -> None:
    # 100 legal cells at tau .01 would ask for the whole simplex; the clamp
    # caps N * tau_row at tau_max_mass.
    wide = _legal(list(range(100)))
    logits = jnp.zeros((1, NUM_ACTION_CELLS))
    _, _, tau_row = support_hinge_loss(
        legal_log_policy(logits, wide), wide, TAU, TAU_MAX_MASS
    )
    np.testing.assert_allclose(float(tau_row[0]) * 100, TAU_MAX_MASS)
    # Positive control: four cells do not bind.
    _, _, tau_row = support_hinge_loss(
        legal_log_policy(logits, _legal(FOUR)), _legal(FOUR), TAU, TAU_MAX_MASS
    )
    np.testing.assert_allclose(float(tau_row[0]), TAU)


@pytest.mark.parametrize("temperature", [0.0, TEMPERATURE])
def test_saturating_row_is_finite_and_bounded(temperature: float) -> None:
    # A row saturated to one cell, logits at +-1e4: the loss is finite and
    # the gradient bounded by 1 with the starved cells lifted.
    legal = _legal(FOUR)
    logits = jnp.zeros((1, NUM_ACTION_CELLS)).at[0, FOUR].set([1e4, -1e4, -1e4, -1e4])
    loss = _loss(legal, temperature=temperature)
    value = float(loss(logits))
    gradient = np.asarray(jax.grad(loss)(logits))[0]
    assert np.isfinite(value) and value > 0.0
    assert np.all(np.isfinite(gradient)) and np.all(np.abs(gradient) <= 1.0)
    assert gradient[0] > 0.0 and all(gradient[cell] < 0.0 for cell in FOUR[1:])


@pytest.mark.parametrize("rows", [1, 3])
def test_batched_rows_are_independent(rows: int) -> None:
    legal = _legal(FOUR, rows)
    logits = jnp.zeros((rows, NUM_ACTION_CELLS)).at[0, MOVE_CELL_OFFSET].set(-9.0)
    row_losses, _, _ = support_hinge_loss(
        legal_log_policy(logits, legal), legal, TAU, TAU_MAX_MASS
    )
    assert float(row_losses[0]) > 0.0
    assert not np.any(np.asarray(row_losses)[1:])


# --- the smooth hinge (temperature > 0, 2026-09-11) --------------------------

# One cell 20% above tau (in the soft band), one starved at tau / 10.
NEAR, STARVED = FOUR[2], FOUR[3]
SOFT_ROW = [0.6, 0.387, 0.012, 0.001]


def _soft_logits() -> jax.Array:
    return (
        jnp.zeros((1, NUM_ACTION_CELLS)).at[0, FOUR].set(jnp.log(jnp.asarray(SOFT_ROW)))
    )


def test_smooth_derivative_is_mean_activation_pi_minus_activation_over_n() -> None:
    """At T > 0 the per-logit derivative is mean(s) * pi_b - s_b / N with
    s = sigmoid(log(tau / pi) / T): bounded by 1 and zero-sum over the legal
    cells, like the hinge's with the indicator smoothed."""
    legal = _legal(FOUR)
    logits = _soft_logits()
    pi = np.asarray(SOFT_ROW, np.float64)
    activation = 1.0 / (1.0 + np.exp(-np.log(TAU / pi) / TEMPERATURE))
    assert 0.05 < activation[2] < 0.95  # the control: the soft band is exercised
    expected = np.zeros(NUM_ACTION_CELLS)
    expected[FOUR] = activation.mean() * pi - activation / 4
    gradient = np.asarray(jax.grad(_loss(legal, temperature=TEMPERATURE))(logits))[0]
    np.testing.assert_allclose(gradient, expected, atol=1e-6)
    np.testing.assert_allclose(gradient.sum(), 0.0, atol=1e-6)
    assert np.all(np.abs(gradient) <= 1.0)


@pytest.mark.parametrize(
    ("temperature", "minimum_excess"), [(TEMPERATURE, 1e-3), (0.01, 0.0)]
)
def test_smooth_loss_exceeds_the_hinge_by_at_most_t_log_two(
    temperature: float, minimum_excess: float
) -> None:
    """T * softplus(x / T) - max(0, x) lies in (0, T log 2] per cell, so the
    row loss (a mean over the four cells) sits above the hinge's and within
    T log 2 of it -- the smooth loss tends to the hinge as T falls. At T = .1
    the cell at 1.2 tau puts the excess well clear of zero; at T = .01 it is
    ~1e-10, below f32 resolution on this loss, so only the bound is pinned."""
    legal = _legal(FOUR)
    logits = _soft_logits()
    hard = float(_loss(legal)(logits))
    smooth = float(_loss(legal, temperature=temperature)(logits))
    assert hard > 0.0
    assert minimum_excess <= smooth - hard <= temperature * np.log(2.0) + 1e-6


def test_smooth_hinge_lifts_a_cell_just_above_tau_that_the_hinge_does_not() -> None:
    """The property that changed, pinned so it cannot change back unnoticed:
    at 1.2 tau the hinge only pushes the cell DOWN (its share of the zero-sum
    term) while the smooth loss lifts it; far below tau the two lifts agree."""
    legal = _legal(FOUR)
    logits = _soft_logits()
    hard = np.asarray(jax.grad(_loss(legal))(logits))[0]
    smooth = np.asarray(jax.grad(_loss(legal, temperature=TEMPERATURE))(logits))[0]
    assert hard[NEAR] > 0.0 and smooth[NEAR] < 0.0
    np.testing.assert_allclose(smooth[STARVED], hard[STARVED], rtol=1e-2)
