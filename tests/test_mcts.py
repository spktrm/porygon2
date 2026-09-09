"""Bounded PUCT, empirical chance banks, latent interiors and terminal backup."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.model.constants import CLS_ROW, NUM_POLICY_READABLE_ROWS
from rl.model.mcts import mcts_root
from rl.model.search import SearchFns
from rl.model.transition import Candidates


def stub_model(continuation=1.0, terminal=0.0, stochastic=False):
    def encoder(rows, cell):
        return jax.nn.one_hot(cell, 4)

    def prior(rows, action):
        return jnp.zeros((1, 2))

    def imagine(rows, action, chance):
        code = action.argmax()
        level = rows[CLS_ROW, 1]
        root_action = jnp.where(level == 0, code, rows[CLS_ROW, 2])
        if stochastic:
            value = chance[0, 0] - chance[0, 1]
        else:
            value = jnp.where((root_action == 0) & (code == 2), 0.8, -0.8)
            value = jnp.where(level == 0, 0.0, value)
        return rows.at[CLS_ROW].set(jnp.array([value, level + 1, root_action]))

    def generate(rows, rng):
        return Candidates(
            codes=jnp.array([2, 3]),
            occupied=jnp.ones(2, bool),
            log_draw_conditionals=jnp.zeros(2),
            rho_at_codes=jnp.full(2, 0.5),
            support_mask=jnp.ones(4, bool),
            retained_mass=jnp.float32(1),
        )

    return SearchFns(
        encoder,
        prior,
        imagine,
        lambda row: row[0],
        generate,
        lambda rows: jnp.float32(continuation),
        lambda rows: jnp.float32(terminal),
    )


def run_search(fns, *, simulations=256, depth=2, chance_samples=1, legal=None):
    if legal is None:
        legal = jnp.array([True, True, False, False])
    return jax.jit(
        lambda key: mcts_root(
            jnp.zeros((NUM_POLICY_READABLE_ROWS, 3)),
            legal,
            jnp.zeros(4),
            key,
            fns,
            simulations=simulations,
            depth=depth,
            chance_samples=chance_samples,
            max_actions=2,
        )
    )(jax.random.key(11))


def test_mcts_finds_delayed_value_through_latent_actions():
    result = run_search(stub_model())
    assert result.visits.sum() == 256
    assert result.visits[0] > result.visits[1] * 3
    assert result.root.cell_values[0] > 0.7
    assert result.root.cell_values[1] < -0.4
    assert result.model_calls <= 256
    assert result.depth_reached == 2
    np.testing.assert_array_equal(result.visits[2:], 0)
    # A depth-one control cannot see these delayed rewards.
    shallow = run_search(stub_model(), depth=1)
    np.testing.assert_array_equal(shallow.root.cell_values, 0)
    probabilities = jax.nn.softmax(
        jnp.where(jnp.array([True, True, False, False]), result.root.bonus, -jnp.inf)
    )
    np.testing.assert_allclose(
        probabilities, result.visits / result.visits.sum(), atol=1e-6
    )


def test_fractional_terminal_payoff_is_counted_once():
    result = run_search(stub_model(continuation=0.25, terminal=-1))
    # Best descendant return .8: .75*(-1) + .25*.8 = -.55, not +.8.
    assert -0.58 < result.root.cell_values[0] < -0.50
    assert result.root.cell_values[1] < -0.80
    terminal = run_search(stub_model(continuation=0, terminal=0.6))
    np.testing.assert_allclose(terminal.root.cell_values[:2], 0.6, atol=1e-6)
    assert terminal.depth_reached == 1


def test_chance_is_sampled_not_optimised():
    result = run_search(
        stub_model(stochastic=True),
        simulations=1024,
        depth=1,
        chance_samples=64,
        legal=jnp.array([True, False, False, False]),
    )
    assert abs(float(result.root.cell_values[0])) < 0.3
    assert result.model_calls <= 64
    assert result.visits[0] == 1024


def test_overflow_has_zero_model_calls_and_no_policy_change():
    result = run_search(stub_model(), legal=jnp.ones(4, bool))
    assert result.root.legal_truncated
    assert result.model_calls == 0
    np.testing.assert_array_equal(result.root.bonus, 0)
    np.testing.assert_array_equal(result.visits, 0)


def test_invalid_budget_fails():
    with pytest.raises(ValueError):
        run_search(stub_model(), simulations=1)


def test_empty_root_is_inert():
    result = run_search(stub_model(), legal=jnp.zeros(4, bool))
    assert result.model_calls == 0
    np.testing.assert_array_equal(result.root.bonus, 0)
    np.testing.assert_array_equal(result.visits, 0)
    for value in jax.tree.leaves(result):
        assert np.isfinite(value).all()


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("mapping", ["scalar", "map", "vmap", "batch"])
def test_executed_expansions_stay_outside_traversal(depth, mapping):
    calls = []
    generated = []
    original = stub_model()

    def imagine(rows, action, chance):
        jax.debug.callback(lambda value: calls.append(float(value)), rows[CLS_ROW, 0])
        return original.imagine_fn(rows, action, chance)

    def generate(rows, key):
        jax.debug.callback(
            lambda value: generated.append(float(value)), rows[CLS_ROW, 0]
        )
        return original.generate_fn(rows, key)

    fns = original._replace(imagine_fn=imagine, generate_fn=generate)

    def search(rows):
        return mcts_root(
            rows,
            jnp.array([True, True, False, False]),
            jnp.zeros(4),
            jax.random.key(11),
            fns,
            simulations=64,
            depth=depth,
            chance_samples=1,
            max_actions=2,
        )

    rows = jnp.zeros((NUM_POLICY_READABLE_ROWS, 3))
    if mapping == "scalar":
        result = jax.jit(search)(rows)
    elif mapping == "map":
        result = jax.jit(lambda batch: jax.lax.map(search, batch))(rows[None])
    elif mapping == "vmap":
        result = jax.jit(jax.vmap(search))(rows[None])
    else:
        result = jax.jit(jax.vmap(search))(jnp.stack([rows, rows]))
    jax.block_until_ready(result)
    jax.effects_barrier()
    logical = int(np.asarray(result.model_calls).sum())
    assert 0 < logical < 64
    if mapping == "batch":
        # Positive control: multi-root predicates retain masked batching,
        # bounded to one dynamics call per root per simulation.
        assert len(calls) == 128
    else:
        assert len(calls) == logical
    if depth == 1:
        assert not generated


def test_guarded_cond_preserves_mixed_nested_batches_and_captures():
    from rl.model.mcts import _guarded_cond

    def evaluate(condition, values, guarded):
        # Both integer and constant-valued floating captures must be explicit
        # in the custom batching rule, including under nested vmap.
        offset = values.astype(jnp.int32)
        support = values + jnp.float32(0.25)

        def selected(operand):
            return operand * support + offset

        def skipped(operand):
            return operand - support

        if guarded:
            return _guarded_cond(condition, selected, skipped, values)
        return jax.lax.cond(condition, selected, skipped, values)

    conditions = jnp.array([[True], [False]])
    values = jnp.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    results = []
    for guarded in (False, True):
        mapped = jax.vmap(jax.vmap(lambda flag, row: evaluate(flag, row, guarded)))
        results.append(jax.jit(mapped)(conditions, values))
    np.testing.assert_array_equal(*results)


def test_guarded_cond_skips_uniform_false_branch_for_batched_operands():
    from rl.model.mcts import _guarded_cond

    calls = []

    def expensive(values):
        jax.debug.callback(lambda value: calls.append(float(value)), values)
        return values + 1

    mapped = jax.jit(
        jax.vmap(
            lambda value: _guarded_cond(
                False, expensive, lambda operand: operand, value
            )
        )
    )
    result = mapped(jnp.arange(3, dtype=jnp.float32))
    jax.block_until_ready(result)
    jax.effects_barrier()
    np.testing.assert_array_equal(result, np.arange(3))
    assert not calls
