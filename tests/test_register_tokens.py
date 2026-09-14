"""The trunk registers are layout rows, two per tier (2026-09-15): the read
mask governs them like every other row, and the trunk appends nothing.
Each invariance carries its positive control."""

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from rl.model.constants import (
    NUM_SEQUENCE_ROWS,
    POLICY_READABLE_ROWS,
    PRIVATE_REGISTER_ROWS,
    PRIVATE_TIER_ROWS,
    PRIVILEGED_REGISTER_ROWS,
    PUBLIC_REGISTER_ROWS,
    PUBLIC_TIER_ROWS,
    SEQUENCE_READ_MASK,
    VALUE_CLS_ROW,
)
from rl.model.trunk import Trunk


def config() -> ConfigDict:
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
        )
    )


def inputs() -> tuple[jax.Array, jax.Array, jax.Array]:
    sequence = jnp.asarray(
        np.random.default_rng(910).normal(size=(NUM_SEQUENCE_ROWS, 16)), jnp.float32
    )
    valid = jnp.ones(NUM_SEQUENCE_ROWS, dtype=jnp.bool_).at[3].set(False)
    return sequence.at[3].set(0), valid, jnp.asarray(SEQUENCE_READ_MASK)


def _rows(rows: slice) -> np.ndarray:
    return np.arange(*rows.indices(NUM_SEQUENCE_ROWS))


def test_trunk_adds_no_rows_and_registers_sit_in_their_tiers() -> None:
    sequence, valid, mask = inputs()
    trunk = Trunk(config())
    variables = jax.jit(trunk.init)(jax.random.key(0), sequence, valid, mask)
    assert "register_embeddings" not in variables["params"]
    output = jax.jit(trunk.apply)(variables, sequence, valid, mask)
    assert output.shape == sequence.shape
    np.testing.assert_array_equal(output[3], 0)
    assert np.isin(_rows(PUBLIC_REGISTER_ROWS), PUBLIC_TIER_ROWS).all()
    assert np.isin(_rows(PRIVATE_REGISTER_ROWS), PRIVATE_TIER_ROWS).all()
    assert not np.isin(_rows(PRIVILEGED_REGISTER_ROWS), POLICY_READABLE_ROWS).any()


def test_registers_relay_only_within_their_tier_and_match_actor() -> None:
    # Remove TF32 shape-dependent rounding from the actor/learner comparison.
    with jax.default_matmul_precision("highest"):
        sequence, valid, mask = inputs()
        trunk = Trunk(config())
        variables = jax.jit(trunk.init)(jax.random.key(1), sequence, valid, mask)
        apply = jax.jit(trunk.apply)
        original = apply(variables, sequence, valid, mask)
        # A privileged register carrying secret content reaches VALUE_CLS
        # and nothing policy-readable.
        changed = apply(
            variables,
            sequence.at[_rows(PRIVILEGED_REGISTER_ROWS), 0].add(100),
            valid,
            mask,
        )
        np.testing.assert_array_equal(
            changed[POLICY_READABLE_ROWS], original[POLICY_READABLE_ROWS]
        )
        assert not np.allclose(changed[VALUE_CLS_ROW], original[VALUE_CLS_ROW])
        # A private register reaches the private tier and no public row.
        changed = apply(
            variables,
            sequence.at[_rows(PRIVATE_REGISTER_ROWS), 0].add(100),
            valid,
            mask,
        )
        np.testing.assert_array_equal(
            changed[PUBLIC_TIER_ROWS], original[PUBLIC_TIER_ROWS]
        )
        assert not np.allclose(changed[PRIVATE_TIER_ROWS], original[PRIVATE_TIER_ROWS])
        # A public register reaches everything: the control that the
        # registers are live rows, not dead ones.
        changed = apply(
            variables, sequence.at[_rows(PUBLIC_REGISTER_ROWS), 0].add(100), valid, mask
        )
        assert not np.allclose(changed[PUBLIC_TIER_ROWS], original[PUBLIC_TIER_ROWS])
        assert not np.allclose(changed[PRIVATE_TIER_ROWS], original[PRIVATE_TIER_ROWS])
        # The actor's shorter sequence keeps the public and private
        # registers and computes the same rows.
        actor = apply(
            variables,
            sequence[POLICY_READABLE_ROWS],
            valid[POLICY_READABLE_ROWS],
            mask[np.ix_(POLICY_READABLE_ROWS, POLICY_READABLE_ROWS)],
        )
        np.testing.assert_allclose(
            actor, original[POLICY_READABLE_ROWS], atol=2e-5, rtol=2e-5
        )
