"""Internal register workspace must preserve output layout and information sets."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import ConfigDict

from rl.model.constants import (
    NUM_SEQUENCE_ROWS,
    POLICY_READABLE_ROWS,
    SEQUENCE_READ_MASK,
)
from rl.model.trunk import Trunk


def config(num_registers):
    return ConfigDict(
        dict(
            num_blocks=3,
            num_heads=2,
            qk_size=8,
            v_size=8,
            model_size=16,
            hidden_size=32,
            use_bias=False,
            qk_layer_norm=True,
            num_registers=num_registers,
        )
    )


def inputs(dtype):
    sequence = jnp.asarray(
        np.random.default_rng(910).normal(size=(NUM_SEQUENCE_ROWS, 16)), dtype
    )
    valid = jnp.ones(NUM_SEQUENCE_ROWS, dtype=jnp.bool_).at[3].set(False)
    return sequence.at[3].set(0), valid, jnp.asarray(SEQUENCE_READ_MASK)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_registers_are_internal_finite_and_trainable(dtype):
    sequence, valid, mask = inputs(dtype)
    trunk = Trunk(config(4))
    variables = jax.jit(trunk.init)(jax.random.key(0), sequence, valid, mask)
    output = jax.jit(trunk.apply)(variables, sequence, valid, mask)
    assert output.shape == sequence.shape
    assert output.dtype == dtype
    assert np.isfinite(np.asarray(output, np.float32)).all()
    np.testing.assert_array_equal(output[3], 0)
    registers = variables["params"]["register_embeddings"]
    assert registers.shape == (4, 16)
    assert registers.dtype == jnp.float32
    assert np.unique(np.asarray(registers), axis=0).shape[0] == 4

    def objective(weights):
        result = trunk.apply(weights, sequence, valid, mask)
        return jnp.sum(result[POLICY_READABLE_ROWS].astype(jnp.float32) ** 2)

    gradients = jax.jit(jax.grad(objective))(variables)
    assert np.all(
        np.linalg.norm(gradients["params"]["register_embeddings"], axis=-1) > 0
    )
    changed = jax.tree.map(lambda value: value, variables)
    changed["params"]["register_embeddings"] = registers.at[:, 0].add(3)
    altered = jax.jit(trunk.apply)(changed, sequence, valid, mask)
    assert float(jnp.max(jnp.abs(altered - output))) > 0


def test_registers_cannot_relay_privileged_inputs_and_match_actor():
    # Remove TF32 shape-dependent rounding from the actor/learner comparison.
    with jax.default_matmul_precision("highest"):
        sequence, valid, mask = inputs(jnp.float32)
        trunk = Trunk(config(4))
        variables = jax.jit(trunk.init)(jax.random.key(1), sequence, valid, mask)
        apply = jax.jit(trunk.apply)
        original = apply(variables, sequence, valid, mask)
        secret = np.setdiff1d(np.arange(NUM_SEQUENCE_ROWS), POLICY_READABLE_ROWS)
        changed = apply(variables, sequence.at[secret, 0].add(100), valid, mask)
        np.testing.assert_array_equal(
            changed[POLICY_READABLE_ROWS], original[POLICY_READABLE_ROWS]
        )
        assert float(jnp.max(jnp.abs(changed[secret] - original[secret]))) > 0
        public_changed = apply(variables, sequence.at[1, 0].add(100), valid, mask)
        assert (
            float(
                jnp.max(
                    jnp.abs(
                        public_changed[POLICY_READABLE_ROWS]
                        - original[POLICY_READABLE_ROWS]
                    )
                )
            )
            > 0
        )
        actor = apply(
            variables,
            sequence[POLICY_READABLE_ROWS],
            valid[POLICY_READABLE_ROWS],
            mask[np.ix_(POLICY_READABLE_ROWS, POLICY_READABLE_ROWS)],
        )
        np.testing.assert_allclose(
            actor, original[POLICY_READABLE_ROWS], atol=2e-5, rtol=2e-5
        )


def test_zero_registers_is_exact_legacy_path():
    sequence, valid, mask = inputs(jnp.bfloat16)
    control = Trunk(config(0))
    legacy_config = config(0)
    del legacy_config.num_registers
    legacy = Trunk(legacy_config)
    variables = jax.jit(control.init)(jax.random.key(2), sequence, valid, mask)
    assert "register_embeddings" not in variables["params"]
    actual = jax.jit(control.apply)(variables, sequence, valid, mask)
    expected = jax.jit(legacy.apply)(variables, sequence, valid, mask)
    np.testing.assert_array_equal(actual, expected)


def test_checkpoint_merge_preserves_blocks_and_seeds_registers():
    from rl.online.artifact import merge_params

    sequence, valid, mask = inputs(jnp.float32)
    old_trunk = Trunk(config(0))
    new_trunk = Trunk(config(4))
    old = jax.jit(old_trunk.init)(jax.random.key(3), sequence, valid, mask)
    fresh = jax.jit(new_trunk.init)(jax.random.key(4), sequence, valid, mask)
    merged, kept_fresh, dropped = merge_params(fresh, old)
    assert not dropped
    assert set(kept_fresh) == {"/params/register_embeddings", "/params/register_norm"}
    for actual, expected in zip(
        jax.tree.leaves(merged["params"]["blocks"]),
        jax.tree.leaves(old["params"]["blocks"]),
    ):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        merged["params"]["register_embeddings"], fresh["params"]["register_embeddings"]
    )
    output = jax.jit(new_trunk.apply)(merged, sequence, valid, mask)
    assert np.isfinite(np.asarray(output)).all()
