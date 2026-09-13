"""History attention primitives and the standalone snapshot stream."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import ConfigDict

from rl.model.history_encoder import (
    NUM_FIELD_ROWS,
    PerSlotHistoryEncoder,
    StepAttention,
)

ENTITY_SIZE = 32
NUM_HEADS = 2
QK_SIZE = 8
HISTORY = 6


NUM_STEPS = 3
NUM_ROWS = 4
ROW_WIDTH = 2 * ENTITY_SIZE + 3


@pytest.fixture(scope="module")
def step_attention() -> tuple[Callable, dict, dict, jax.Array, jax.Array]:
    module = StepAttention(
        num_heads=NUM_HEADS,
        qk_size=QK_SIZE,
        features=ENTITY_SIZE,
        dtype=jnp.float32,
    )
    key_params, key_rows, key_out = jax.random.split(jax.random.key(1), 3)
    rows = jax.random.normal(key_rows, (NUM_STEPS, NUM_ROWS, ROW_WIDTH))
    # steps with 1, 2 and 4 live rows
    row_mask = jnp.arange(NUM_ROWS)[None] < jnp.asarray([1, 2, 4])[:, None]
    params = module.init(key_params, rows, row_mask)
    live_out = jax.random.normal(key_out, (ENTITY_SIZE, ENTITY_SIZE)) * 0.1
    live_params = jax.tree_util.tree_map(lambda leaf: leaf, params)
    live_params["params"]["attn_out"]["kernel"] = live_out
    apply = jax.jit(module.apply)
    return apply, params, live_params, rows, row_mask


def test_attn_out_is_silent_at_init_and_not_after(
    step_attention: tuple[Callable, dict, dict, jax.Array, jax.Array],
) -> None:
    apply, params, live_params, rows, row_mask = step_attention
    out, probs = apply(params, rows, row_mask)
    assert jnp.all(out == 0)
    assert jnp.all(jnp.isfinite(probs))
    live, _ = apply(live_params, rows, row_mask)
    assert jnp.any(live[row_mask] != 0)


def test_attn_out_has_gradient_at_init(
    step_attention: tuple[Callable, dict, dict, jax.Array, jax.Array],
) -> None:
    apply, params, _, rows, row_mask = step_attention
    weights = jax.random.normal(jax.random.key(2), (NUM_STEPS, NUM_ROWS, ENTITY_SIZE))

    def objective(tree: dict) -> jax.Array:
        out, _ = apply(tree, rows, row_mask)
        return (out * weights).sum()

    grads = jax.grad(objective)(params)
    assert jnp.any(grads["params"]["attn_out"]["kernel"] != 0)


def test_padded_row_places_no_mass_and_moves_nothing(
    step_attention: tuple[Callable, dict, dict, jax.Array, jax.Array],
) -> None:
    apply, _, live_params, rows, row_mask = step_attention
    base, probs = apply(live_params, rows, row_mask)
    # no probability lands on a padded key, and live rows sum to one
    assert jnp.all(probs[:, :, :, 2:][1] == 0)
    assert np.allclose(probs[1][:, :2].sum(-1), 1.0, atol=1e-5)
    # perturbing the padded row 2 of step 1 leaves the live rows bit-identical
    bumped = rows.at[1, 2].add(1.0)
    moved, _ = apply(live_params, bumped, row_mask)
    assert jnp.array_equal(moved[1, :2], base[1, :2])
    # control: the same bump on a LIVE row (step 2 has 4 live rows) moves the
    # other rows of its step
    bumped = rows.at[2, 2].add(1.0)
    moved, _ = apply(live_params, bumped, row_mask)
    assert jnp.any(moved[2, :2] != base[2, :2])


def test_one_row_step_is_its_own_value(
    step_attention: tuple[Callable, dict, dict, jax.Array, jax.Array],
) -> None:
    apply, _, live_params, rows, row_mask = step_attention
    _, probs = apply(live_params, rows, row_mask)
    assert jnp.all(probs[0, :, 0, 0] == 1.0)
    assert jnp.all(probs[0, :, 0, 1:] == 0.0)


def test_latest_snapshot_excludes_event_identity_and_carries_forward() -> None:
    from rl.environment.protos.features_pb2 import FieldFeature
    from rl.model.history_encoder import history_carry_from

    cfg = ConfigDict(
        dict(
            entity_size=ENTITY_SIZE,
            dtype=jnp.float32,
            history_step=dict(num_heads=NUM_HEADS, qk_size=QK_SIZE),
        )
    )
    module = PerSlotHistoryEncoder(cfg)
    field = jnp.zeros((1, len(FieldFeature.keys())), jnp.int32)
    field = field.at[0, FieldFeature.FIELD_FEATURE__NUM_RELEVANT].set(1)
    content = jnp.arange(ENTITY_SIZE, dtype=jnp.float32)[None] / ENTITY_SIZE
    inputs = dict(
        history_field=field,
        node_embedding_cache=content,
        node_identity_cache=jnp.ones_like(content),
        field_identities=jnp.ones((NUM_FIELD_ROWS, ENTITY_SIZE)),
        node_content_cache=content,
        edge_embedding_cache=jnp.zeros_like(content),
        edge_slot_ids=jnp.zeros(1, jnp.int32),
        edge_major_args=jnp.zeros(1, jnp.int32),
        field_row_embeddings=jnp.zeros((1, NUM_FIELD_ROWS, ENTITY_SIZE)),
        step_request_count=jnp.ones(1, jnp.int32),
        step_valid=jnp.ones(1, jnp.bool_),
    )
    params = jax.jit(module.init)(jax.random.key(19), **inputs)
    apply = jax.jit(module.apply)
    base = apply(params, **inputs)
    changed_identities = dict(
        inputs,
        node_identity_cache=inputs["node_identity_cache"] + 10,
        field_identities=inputs["field_identities"] + 20,
    )
    moved = apply(params, **changed_identities)
    muted = jax.tree.map(lambda leaf: leaf, params)
    muted["params"]["sequence_step"]["attention"]["attn_out"]["kernel"] = jnp.zeros(
        (ENTITY_SIZE, ENTITY_SIZE)
    )
    muted_base = apply(muted, **inputs)
    muted_moved = apply(muted, **changed_identities)
    for state_name in ("slot_snapshots", "field_snapshots", "register_snapshots"):
        np.testing.assert_array_equal(
            getattr(muted_base, state_name), getattr(muted_moved, state_name)
        )
    np.testing.assert_array_equal(base.node_snapshots[0, 0], content[0])
    np.testing.assert_array_equal(base.node_snapshots, moved.node_snapshots)
    assert not np.allclose(base.slot_snapshots, moved.slot_snapshots)
    changed_content = apply(params, **dict(inputs, node_content_cache=content + 2))
    np.testing.assert_allclose(changed_content.node_snapshots[0, 0], content[0] + 2)
    carried = apply(
        params,
        **dict(inputs, step_valid=jnp.zeros(1, jnp.bool_)),
        carry=history_carry_from(base)
    )
    np.testing.assert_array_equal(carried.node_snapshots[0], base.node_snapshots[-1])
