"""Contracts of the pairwise entity critic (rl/model/heads.py PairValueHead,
2026-09-12). Every invariance test carries the control that proves it could
fail: the head is exactly zero at init, so each one runs on opened params."""

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from rl.model.heads import PAIR_VALUE_PARTIALS, PairValueHead, masked_softmax
from rl.model.utils import open_zero_init_paths

WIDTH = 32
PER_SIDE = 6
ROWS = 2 * PER_SIDE
# The head's zero-init leaves, by the exact key open_zero_init_paths matches.
ZERO_INIT_KEYS = [
    "cross_query",
    "cross_weight_query",
    "synergy_query",
    "synergy_weight_query",
    "unary",
]


def _head() -> PairValueHead:
    return PairValueHead(ConfigDict(dict(qk_size=WIDTH, unary_hidden=WIDTH)))


def _inputs(key: jax.Array):
    rows = jax.random.normal(key, (ROWS, WIDTH))
    valid = jnp.ones(ROWS, dtype=jnp.bool_)
    alive = jnp.ones(ROWS, dtype=jnp.bool_)
    return rows, valid, alive


def _init():
    head = _head()
    inputs = _inputs(jax.random.key(0))
    params = head.init(jax.random.key(1), *inputs)
    return head, params, inputs


def _opened(params: dict) -> dict:
    return open_zero_init_paths(params, ZERO_INIT_KEYS)


def _swap_sides(rows, valid, alive):
    """The mirror: the two sides' rows and flags exchanged."""
    order = jnp.concatenate((jnp.arange(PER_SIDE, ROWS), jnp.arange(PER_SIDE)))
    return (
        rows[order],
        valid[order],
        alive[order],
    )


# --- init contract ---------------------------------------------------------


def test_value_is_exactly_zero_and_weights_uniform_at_init() -> None:
    head, params, inputs = _init()
    out = head.apply(params, *inputs)
    np.testing.assert_array_equal(np.asarray(out.value), 0.0)
    np.testing.assert_array_equal(np.asarray(out.unary), 0.0)
    np.testing.assert_array_equal(np.asarray(out.cross), 0.0)
    np.testing.assert_array_equal(np.asarray(out.synergy), 0.0)
    np.testing.assert_array_equal(np.asarray(out.partials), 0.0)
    cross_weight = np.asarray(out.cross_weight)
    assert cross_weight.shape == (PER_SIDE, PER_SIDE)
    np.testing.assert_allclose(cross_weight, 1.0 / (PER_SIDE * PER_SIDE), rtol=1e-6)
    synergy_weight = np.asarray(out.synergy_weight)
    assert synergy_weight.shape == (2, PER_SIDE, PER_SIDE)
    np.testing.assert_array_equal(np.diagonal(synergy_weight, axis1=-2, axis2=-1), 0)
    off_diagonal = ~np.eye(PER_SIDE, dtype=bool)
    np.testing.assert_allclose(
        synergy_weight[:, off_diagonal], 1.0 / (PER_SIDE * (PER_SIDE - 1)), rtol=1e-6
    )


def test_queries_and_unary_move_at_step_one_keys_and_weights_at_step_two() -> None:
    """The two-factor stall guard, per pair function: the zero factor's
    gradient is live at step 1, its partner's is exactly 0 for one step and
    unfreezes once the zero factor is nudged. The weight scores are frozen
    one step too: their gradients are proportional to the pair terms, 0 at
    init."""
    head, params, inputs = _init()

    def total(p: dict) -> jax.Array:
        return head.apply(p, *inputs).value

    grads = jax.grad(total)(params)["params"]
    for name in ("cross_query", "synergy_query"):
        assert np.abs(np.asarray(grads[name]["kernel"])).max() > 0, name
    assert np.abs(np.asarray(grads["unary"]["Dense_1"]["kernel"])).max() > 0
    for name in (
        "cross_key",
        "synergy_key",
        "cross_weight_query",
        "cross_weight_key",
        "synergy_weight_query",
        "synergy_weight_key",
    ):
        np.testing.assert_array_equal(np.asarray(grads[name]["kernel"]), 0.0)

    nudged = jax.tree.map(lambda x: x, params)
    nudged["params"]["cross_query"]["kernel"] = (
        nudged["params"]["cross_query"]["kernel"] + 1e-2
    )
    nudged_grads = jax.grad(total)(nudged)["params"]
    assert np.abs(np.asarray(nudged_grads["cross_key"]["kernel"])).max() > 0
    # The cross term is now nonzero, so its weights matter: both weight
    # factors unfreeze together (the query side over live rows).
    assert np.abs(np.asarray(nudged_grads["cross_weight_query"]["kernel"])).max() > 0


# --- structure -------------------------------------------------------------


def test_partials_are_signed_and_sum_to_the_value() -> None:
    head, params, inputs = _init()
    out = head.apply(_opened(params), *inputs)
    partials = np.asarray(out.partials)
    assert partials.shape == (len(PAIR_VALUE_PARTIALS),)
    assert np.abs(partials).min() > 0, "opened params must give every part mass"
    np.testing.assert_allclose(partials.sum(), np.asarray(out.value), rtol=1e-5)
    unary = np.asarray(out.unary)
    cross = np.sum(np.asarray(out.cross_weight) * np.asarray(out.cross))
    synergy = np.sum(
        np.asarray(out.synergy_weight) * np.asarray(out.synergy), axis=(-2, -1)
    )
    expected = np.array(
        [
            unary[:PER_SIDE].sum(),
            -unary[PER_SIDE:].sum(),
            cross,
            synergy[0],
            -synergy[1],
        ]
    )
    np.testing.assert_allclose(partials, expected, rtol=1e-5, atol=1e-6)


def test_swapping_the_sides_negates_the_value_exactly() -> None:
    """The mirror invariant, by construction: one function per relation
    shared across sides. Control: a content change on one row moves V, so
    equality up to sign is not a degenerate constant."""
    head, params, inputs = _init()
    opened = _opened(params)
    out = head.apply(opened, *inputs)
    mirrored = head.apply(opened, *_swap_sides(*inputs))
    assert abs(float(out.value)) > 1e-3
    np.testing.assert_allclose(
        np.asarray(mirrored.value), -np.asarray(out.value), rtol=1e-5, atol=1e-6
    )
    # The cross term transposes and flips, the weights transpose.
    np.testing.assert_allclose(
        np.asarray(mirrored.cross), -np.asarray(out.cross).T, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(mirrored.cross_weight),
        np.asarray(out.cross_weight).T,
        rtol=1e-5,
        atol=1e-6,
    )
    rows, valid, alive = inputs
    control = head.apply(opened, rows.at[3].add(1.0), valid, alive)
    assert not np.allclose(np.asarray(control.value), np.asarray(out.value))


def test_an_invalid_row_contributes_nothing_bit_identically() -> None:
    head, params, inputs = _init()
    opened = _opened(params)
    rows, valid, alive = inputs
    dropped = valid.at[8].set(False)
    base = head.apply(opened, rows, dropped, alive)
    moved = head.apply(opened, rows.at[8].add(5.0), dropped, alive)
    np.testing.assert_array_equal(np.asarray(base.value), np.asarray(moved.value))
    np.testing.assert_array_equal(np.asarray(base.partials), np.asarray(moved.partials))
    assert float(np.asarray(base.unary)[8]) == 0.0
    # Control: the same perturbation on a VALID row moves the value.
    control = head.apply(opened, rows.at[8].add(5.0), valid, alive)
    assert not np.array_equal(np.asarray(control.value), np.asarray(base.value))


def test_a_fainted_row_keeps_its_unary_term_and_no_pair_weight() -> None:
    head, params, inputs = _init()
    opened = _opened(params)
    rows, valid, alive = inputs
    fainted = alive.at[2].set(False)
    out = head.apply(opened, rows, valid, fainted)
    live = head.apply(opened, *inputs)
    np.testing.assert_array_equal(np.asarray(out.unary), np.asarray(live.unary))
    np.testing.assert_array_equal(np.asarray(out.cross_weight)[2], 0.0)
    np.testing.assert_array_equal(np.asarray(out.cross)[2], 0.0)
    np.testing.assert_array_equal(np.asarray(out.synergy_weight)[0, 2], 0.0)
    np.testing.assert_array_equal(np.asarray(out.synergy_weight)[0, :, 2], 0.0)
    # The other pairs' weights renormalise: still a distribution.
    np.testing.assert_allclose(np.asarray(out.cross_weight).sum(), 1.0, rtol=1e-6)
    assert not np.allclose(np.asarray(out.value), np.asarray(live.value))


def test_an_empty_board_is_zero_and_finite() -> None:
    head, params, inputs = _init()
    rows, valid, alive = inputs
    nobody = jnp.zeros(ROWS, dtype=jnp.bool_)
    out = head.apply(_opened(params), rows, nobody, nobody)
    for leaf in jax.tree.leaves(out):
        assert np.isfinite(np.asarray(leaf)).all()
    np.testing.assert_array_equal(np.asarray(out.value), 0.0)
    np.testing.assert_array_equal(np.asarray(out.cross_weight), 0.0)


def test_masked_softmax_is_exact_on_and_off_the_mask() -> None:
    scores = jnp.asarray(np.random.default_rng(0).normal(size=(4, 4)), jnp.float32)
    mask = jnp.asarray(np.random.default_rng(1).uniform(size=(4, 4)) > 0.5)
    weights = np.asarray(masked_softmax(scores, mask))
    np.testing.assert_array_equal(weights[~np.asarray(mask)], 0.0)
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-6)
    on = np.asarray(mask)
    reference = np.exp(np.asarray(scores)[on])
    np.testing.assert_allclose(weights[on], reference / reference.sum(), rtol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(masked_softmax(scores, jnp.zeros_like(mask))), 0.0
    )
