"""Row-scale, actor/learner and gradient contracts for input normalisation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.model.constants import (
    NUM_SEQUENCE_GROUPS,
    NUM_SEQUENCE_ROWS,
    POLICY_READABLE_ROWS,
    SEQUENCE_GROUP_IDS,
)
from rl.model.modules import SequenceInputNormalisation


def inputs(dtype):
    generator = np.random.default_rng(910)
    values = generator.normal(size=(NUM_SEQUENCE_ROWS, 32))
    values *= np.geomspace(0.1, 1000, NUM_SEQUENCE_ROWS)[:, None]
    valid = np.ones(NUM_SEQUENCE_ROWS, bool)
    valid[3] = False
    values[3] = 0
    return (
        jnp.asarray(values, dtype),
        jnp.asarray(valid),
        jnp.asarray(SEQUENCE_GROUP_IDS),
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_unit_rms_dtype_and_zero_padding(dtype):
    sequence, valid, groups = inputs(dtype)
    module = SequenceInputNormalisation(num_groups=NUM_SEQUENCE_GROUPS)
    variables = jax.jit(module.init)(jax.random.key(0), sequence, valid, groups)
    output = jax.jit(module.apply)(variables, sequence, valid, groups)
    assert output.dtype == dtype
    assert variables["params"]["group_scale"].dtype == jnp.float32
    rms = np.sqrt(np.mean(np.square(np.asarray(output, np.float32)), axis=-1))
    np.testing.assert_allclose(rms[np.asarray(valid)], 1, atol=0.008)
    np.testing.assert_array_equal(output[3], 0)
    original_rms = np.sqrt(
        np.mean(np.square(np.asarray(sequence, np.float32)), axis=-1)
    )
    assert original_rms.max() / original_rms[np.asarray(valid)].min() > 1000


def test_group_scales_actor_equivalence_and_privileged_isolation():
    sequence, valid, groups = inputs(jnp.float32)
    module = SequenceInputNormalisation(num_groups=NUM_SEQUENCE_GROUPS)
    variables = jax.jit(module.init)(jax.random.key(0), sequence, valid, groups)
    scale = jnp.arange(NUM_SEQUENCE_GROUPS, dtype=jnp.float32)[:, None] / 10
    variables["params"]["group_scale"] = jnp.broadcast_to(
        scale, (NUM_SEQUENCE_GROUPS, 32)
    )
    apply = jax.jit(module.apply)
    original = apply(variables, sequence, valid, groups)
    actor = apply(
        variables,
        sequence[POLICY_READABLE_ROWS],
        valid[POLICY_READABLE_ROWS],
        groups[POLICY_READABLE_ROWS],
    )
    np.testing.assert_array_equal(actor, original[POLICY_READABLE_ROWS])
    rms = np.sqrt(np.mean(np.square(np.asarray(original)), axis=-1))
    np.testing.assert_allclose(
        rms[np.asarray(valid)],
        (1 + np.asarray(scale[:, 0])[np.asarray(groups)])[np.asarray(valid)],
        atol=0.001,
    )
    privileged = np.setdiff1d(np.arange(NUM_SEQUENCE_ROWS), POLICY_READABLE_ROWS)
    changed = apply(variables, sequence.at[privileged, 0].add(10000), valid, groups)
    np.testing.assert_array_equal(
        changed[POLICY_READABLE_ROWS], original[POLICY_READABLE_ROWS]
    )
    assert float(jnp.max(jnp.abs(changed[privileged] - original[privileged]))) > 0
    public_changed = apply(variables, sequence.at[0, 0].add(100), valid, groups)
    assert float(jnp.max(jnp.abs(public_changed[0] - original[0]))) > 0


def test_live_group_and_directional_input_gradients():
    sequence, valid, groups = inputs(jnp.float32)
    module = SequenceInputNormalisation(num_groups=NUM_SEQUENCE_GROUPS)
    variables = jax.jit(module.init)(jax.random.key(0), sequence, valid, groups)

    def objective(weights, rows):
        output = module.apply(weights, rows, valid, groups)
        return jnp.sum(output * jnp.arange(1, 33, dtype=jnp.float32))

    parameter_gradient, input_gradient = jax.jit(jax.grad(objective, argnums=(0, 1)))(
        variables, sequence
    )
    assert np.all(
        np.linalg.norm(parameter_gradient["params"]["group_scale"], axis=-1) > 0
    )
    assert np.all(
        np.linalg.norm(np.asarray(input_gradient)[np.asarray(valid)], axis=-1) > 0
    )
    np.testing.assert_array_equal(input_gradient[3], 0)


def test_one_group_form_matches_the_full_layout_at_init():
    """The trunk's registers use a one-group instance; at init it is the same
    function as the full-layout bank (RMS 1 per row, zero rows stay zero)."""
    sequence, valid, groups = inputs(jnp.float32)
    full = SequenceInputNormalisation(num_groups=NUM_SEQUENCE_GROUPS)
    one = SequenceInputNormalisation(num_groups=1)
    zeros = jnp.zeros_like(groups)
    full_vars = jax.jit(full.init)(jax.random.key(0), sequence, valid, groups)
    one_vars = jax.jit(one.init)(jax.random.key(0), sequence, valid, zeros)
    assert one_vars["params"]["group_scale"].shape == (1, 32)
    # Two jitted programs: XLA fuses the (1, C) and (G, C) gathers into the
    # multiply differently, so equal to rounding, not bit for bit.
    np.testing.assert_allclose(
        jax.jit(one.apply)(one_vars, sequence, valid, zeros),
        jax.jit(full.apply)(full_vars, sequence, valid, groups),
        rtol=1e-6,
        atol=0,
    )
