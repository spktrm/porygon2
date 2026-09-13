"""Standard GRU equations, mixed-precision carry and attention-only normalisation."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from rl.model.constants import (
    HISTORY_FIELD_STATE_ROWS,
    HISTORY_REGISTER_STATE_ROWS,
    HISTORY_SLOT_STATE_ROWS,
    NUM_HISTORY_STATE_ROWS,
)
from rl.model.history_encoder import HistoryGRUCell, HistorySequenceStep

WIDTH = 8


def test_history_gru_matches_flax_outputs_gradients_and_initialisation():
    reference = nn.GRUCell(features=WIDTH, dtype=jnp.float32)
    actual = HistoryGRUCell(features=WIDTH, dtype=jnp.float32)
    memory = jax.random.normal(jax.random.key(71), (3, WIDTH))
    inputs = jax.random.normal(jax.random.key(72), (3, WIDTH))
    params = jax.jit(reference.init)(jax.random.key(73), memory, inputs)
    actual_params = jax.jit(actual.init)(jax.random.key(73), memory, inputs)
    for expected, found in zip(
        jax.tree.leaves(params), jax.tree.leaves(actual_params), strict=True
    ):
        np.testing.assert_array_equal(found, expected)
    reference_apply = jax.jit(reference.apply)
    actual_apply = jax.jit(actual.apply)
    expected = reference_apply(params, memory, inputs)[0]
    found, write_gate = actual_apply(params, memory, inputs)
    np.testing.assert_allclose(found, expected, atol=1e-6)
    assert ((np.asarray(write_gate) > 0) & (np.asarray(write_gate) < 1)).all()

    def reference_loss(tree, previous, event):
        return jnp.square(reference.apply(tree, previous, event)[0]).sum()

    def actual_loss(tree, previous, event):
        return jnp.square(actual.apply(tree, previous, event)[0]).sum()

    expected_gradients = jax.jit(jax.grad(reference_loss, argnums=(0, 1, 2)))(
        params, memory, inputs
    )
    found_gradients = jax.jit(jax.grad(actual_loss, argnums=(0, 1, 2)))(
        params, memory, inputs
    )
    for expected_gradient, found_gradient in zip(
        jax.tree.leaves(expected_gradients),
        jax.tree.leaves(found_gradients),
        strict=True,
    ):
        np.testing.assert_allclose(found_gradient, expected_gradient, atol=1e-6)


def test_reset_gate_controls_recurrent_candidate_and_update_gate_retains_memory():
    module = HistoryGRUCell(features=WIDTH, dtype=jnp.float32)
    memory = jnp.full((2, WIDTH), 0.4)
    inputs = jnp.zeros_like(memory)
    params = jax.jit(module.init)(jax.random.key(74), memory, inputs)
    params = jax.tree.map(jnp.zeros_like, params)
    params["params"]["hn"]["kernel"] = jnp.eye(WIDTH)
    # The identity recurrent kernel is exact only outside TF32 matmuls.
    apply = jax.jit(jax.default_matmul_precision("highest")(module.apply))

    def with_bias(name, value):
        changed = jax.tree.map(lambda leaf: leaf, params)
        changed["params"][name]["bias"] = jnp.full((WIDTH,), value)
        return changed

    closed, _ = apply(with_bias("ir", -20.0), memory, inputs)
    opened, _ = apply(with_bias("ir", 20.0), memory, inputs)
    np.testing.assert_allclose(closed, 0.5 * memory, atol=1e-6)
    np.testing.assert_allclose(opened, 0.5 * (memory + jnp.tanh(memory)), atol=1e-6)
    assert np.max(np.abs(np.asarray(opened - closed))) > 0.1
    retained, write_gate = apply(with_bias("iz", 20.0), memory, inputs)
    np.testing.assert_allclose(retained, memory, atol=1e-6)
    np.testing.assert_allclose(write_gate, 0, atol=1e-6)
    overwritten, _ = apply(with_bias("iz", -20.0), memory, inputs)
    np.testing.assert_allclose(overwritten, jnp.tanh(0.5 * memory), atol=1e-6)


def test_bf16_projections_keep_small_memory_updates_in_f32():
    module = HistoryGRUCell(features=WIDTH, dtype=jnp.bfloat16)
    memory = jnp.ones((1, WIDTH), jnp.float32)
    inputs = jnp.zeros((1, WIDTH), jnp.bfloat16)
    params = jax.jit(module.init)(jax.random.key(75), memory, inputs)
    params = jax.tree.map(jnp.zeros_like, params)
    params["params"]["iz"]["bias"] = jnp.full((WIDTH,), 8.0)
    updated, write_gate = jax.jit(module.apply)(params, memory, inputs)
    assert updated.dtype == jnp.float32
    assert write_gate.dtype == jnp.float32
    assert (np.asarray(updated) < 1).all()
    np.testing.assert_array_equal(
        updated.astype(jnp.bfloat16).astype(jnp.float32), memory
    )
    np.testing.assert_allclose(updated, jax.nn.sigmoid(jnp.asarray(8.0)), atol=1e-7)


def test_norm_and_identities_do_not_bypass_attention_into_gru():
    cfg = ConfigDict(
        dict(
            entity_size=WIDTH,
            dtype=jnp.float32,
            history_step=dict(num_heads=2, qk_size=4),
        )
    )
    module = HistorySequenceStep(cfg)
    memory = jax.random.normal(jax.random.key(76), (NUM_HISTORY_STATE_ROWS, WIDTH))
    events = jax.random.normal(jax.random.key(77), memory.shape)
    inputs = (events, jnp.ones_like(events), jnp.asarray(True))
    params = jax.jit(module.init)(jax.random.key(78), memory, inputs)
    apply = jax.jit(module.apply)
    muted = jax.tree.map(lambda leaf: leaf, params)
    muted["params"]["attention"]["attn_out"]["kernel"] = jnp.zeros((WIDTH, WIDTH))

    def change_read_branch(tree):
        changed = jax.tree.map(lambda leaf: leaf, tree)
        changed["params"]["group_identity"] += 10
        changed["params"]["register_identity"] += 20
        changed["params"]["input_norm"]["scale"] *= 3
        return changed

    base = apply(muted, memory, inputs)[0]
    changed = apply(change_read_branch(muted), memory, inputs)[0]
    np.testing.assert_array_equal(base, changed)
    live = apply(params, memory, inputs)[0]
    changed_live = apply(change_read_branch(params), memory, inputs)[0]
    assert not np.allclose(live, changed_live)
    changed_events = apply(muted, memory, (events * 3, inputs[1], inputs[2]))[0]
    assert not np.allclose(base, changed_events)


def test_gru_weights_are_separate_by_type_and_shared_within_type():
    cfg = ConfigDict(
        dict(
            entity_size=WIDTH,
            dtype=jnp.float32,
            history_step=dict(num_heads=2, qk_size=4),
        )
    )
    module = HistorySequenceStep(cfg)
    memory = jnp.full((NUM_HISTORY_STATE_ROWS, WIDTH), 0.4)
    events = jnp.full_like(memory, 0.2)
    inputs = (events, jnp.zeros_like(events), jnp.asarray(True))
    params = jax.jit(module.init)(jax.random.key(79), memory, inputs)
    params["params"]["attention"]["attn_out"]["kernel"] = jnp.zeros((WIDTH, WIDTH))
    apply = jax.jit(module.apply)
    baseline = apply(params, memory, inputs)[0]
    groups = (
        ("entity", HISTORY_SLOT_STATE_ROWS),
        ("field", HISTORY_FIELD_STATE_ROWS),
        ("register", HISTORY_REGISTER_STATE_ROWS),
    )
    for token_type, state_rows in groups:
        np.testing.assert_allclose(
            baseline[state_rows],
            jnp.broadcast_to(baseline[state_rows][0], baseline[state_rows].shape),
            atol=1e-6,
        )
        changed = jax.tree.map(lambda leaf: leaf, params)
        changed["params"][f"{token_type}_gru"]["in"]["bias"] += 5
        updated = apply(changed, memory, inputs)[0]
        assert not np.allclose(updated[state_rows], baseline[state_rows])
        for other_type, other_rows in groups:
            if other_type != token_type:
                np.testing.assert_array_equal(updated[other_rows], baseline[other_rows])


def test_identity_changes_only_reach_memory_through_attention_weights():
    """Queries and keys read content plus identity, values read content
    alone, so an identity can steer WHERE a row attends but is never copied
    into memory as a value. Control: with the attention output muted the same
    identity change leaves memory bit-identical, which is what "only through
    the weights" means and what makes the live assertions non-vacuous."""
    cfg = ConfigDict(
        dict(
            entity_size=WIDTH,
            dtype=jnp.float32,
            history_step=dict(num_heads=2, qk_size=4),
        )
    )
    module = HistorySequenceStep(cfg)
    memory = jax.random.normal(jax.random.key(81), (NUM_HISTORY_STATE_ROWS, WIDTH))
    events = jax.random.normal(jax.random.key(82), memory.shape)
    inputs = (events, jnp.ones_like(events), jnp.asarray(True))
    params = jax.jit(module.init)(jax.random.key(83), memory, inputs)
    apply = jax.jit(module.apply)

    def move_identities(tree):
        changed = jax.tree.map(lambda leaf: leaf, tree)
        changed["params"]["group_identity"] += 1.0
        changed["params"]["register_identity"] += 1.0
        return changed

    baseline, (_, baseline_probs, _) = apply(params, memory, inputs)
    moved, (_, moved_probs, _) = apply(move_identities(params), memory, inputs)
    assert not np.allclose(baseline_probs, moved_probs)
    assert not np.allclose(baseline, moved)

    muted = jax.tree.map(lambda leaf: leaf, params)
    muted["params"]["attention"]["attn_out"]["kernel"] = jnp.zeros((WIDTH, WIDTH))
    muted_baseline = apply(muted, memory, inputs)[0]
    muted_moved = apply(move_identities(muted), memory, inputs)[0]
    np.testing.assert_array_equal(muted_baseline, muted_moved)
