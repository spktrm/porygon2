"""One 19-row recurrent history sequence with bidirectional memory reads."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import ConfigDict

from rl.environment.protos.features_pb2 import FieldFeature
from rl.model.constants import (
    HISTORY_EVENT_STATE_ROWS,
    HISTORY_FIELD_STATE_ROWS,
    HISTORY_REGISTER_ROWS,
    HISTORY_REGISTER_STATE_ROWS,
    HISTORY_SLOT_STATE_ROWS,
    NUM_ACTIVE_SLOTS,
    NUM_FIELD_ROWS,
    NUM_HISTORY_REGISTERS,
    NUM_HISTORY_STATE_ROWS,
    NUM_PUBLIC_SLOTS,
    OPP_PRIVATE_ROWS,
    POLICY_READABLE_ROWS,
    PRIVATE_TIER_ROWS,
    PUBLIC_TIER_ROWS,
    SEQUENCE_READ_MASK,
    VALUE_CLS_ROW,
)
from rl.model.history_encoder import (
    PerSlotHistoryEncoder,
    history_carry_from,
)

WIDTH = 16
STEPS = 5


def history_config():
    return ConfigDict(
        dict(
            entity_size=WIDTH,
            dtype=jnp.float32,
            history_step=dict(num_heads=2, qk_size=4),
        )
    )


def _recur_call(module, tree, rows, valid, memory, inner):
    """_recur over rows with zero identities, every slot touched; returns
    (states, probs, gates, final_memory, final_inner). f32 matmuls: the
    GPU's default f32 dot runs at TF32, and the associative scan over a
    prefix and over the full window are different fused programs whose
    TF32 rounding differs at ~2e-4 -- the contracts below are exact ones."""
    touched = jnp.ones((rows.shape[0], NUM_PUBLIC_SLOTS), jnp.bool_)
    with jax.default_matmul_precision("float32"):
        return _recur_apply(module, tree, rows, touched, valid, memory, inner)


def _recur_apply(module, tree, rows, touched, valid, memory, inner):
    return module.apply(
        tree,
        rows,
        jnp.zeros_like(rows),
        touched,
        jnp.ones((rows.shape[0], NUM_ACTIVE_SLOTS), jnp.bool_),
        valid,
        memory,
        inner,
        method=PerSlotHistoryEncoder._recur,
    )


@pytest.fixture(scope="module")
def recurrence():
    module = PerSlotHistoryEncoder(history_config())
    rows = jax.random.normal(jax.random.key(35), (STEPS, NUM_HISTORY_STATE_ROWS, WIDTH))
    rows = rows.at[:, HISTORY_REGISTER_STATE_ROWS].set(0)
    valid = jnp.asarray([True, True, False, True, True])
    initial = (
        jax.random.normal(jax.random.key(36), (NUM_HISTORY_STATE_ROWS, WIDTH)) * 0.02
    )
    inner = (
        jax.random.normal(jax.random.key(39), (HISTORY_EVENT_STATE_ROWS.stop, WIDTH))
        * 0.02
    )
    touched = jnp.ones((STEPS, NUM_PUBLIC_SLOTS), jnp.bool_)
    params = jax.jit(
        lambda: module.init(
            jax.random.key(37),
            rows,
            jnp.zeros_like(rows),
            touched,
            jnp.ones((STEPS, NUM_ACTIVE_SLOTS), jnp.bool_),
            valid,
            initial,
            inner,
            method=PerSlotHistoryEncoder._recur,
        )
    )()

    def apply(tree, rows, valid, memory, inner=inner):
        return _recur_call(module, tree, rows, valid, memory, inner)

    return params, jax.jit(apply), rows, valid, initial, inner


@pytest.mark.parametrize(
    "source_row",
    [
        HISTORY_SLOT_STATE_ROWS.start,
        HISTORY_FIELD_STATE_ROWS.start,
        HISTORY_REGISTER_STATE_ROWS.start,
    ],
)
def test_layer_two_memory_is_isolated_and_layer_one_is_read_by_every_row(
    recurrence, source_row
):
    """A row's initial LAYER-2 memory moves only that row -- no attention
    reads it (the isolation that removes the 2026-09-13 loop's chaos) --
    while a row's initial layer-1 memory moves every group, the control
    that the attention still reads across rows."""
    params, apply, rows, valid, initial, inner = recurrence
    states, probs, _, final, _ = apply(params, rows, valid, initial)
    assert probs.shape == (STEPS, 2, NUM_HISTORY_STATE_ROWS, NUM_HISTORY_STATE_ROWS)
    assert states.shape == (STEPS, NUM_HISTORY_STATE_ROWS, WIDTH)
    assert final.dtype == jnp.float32
    np.testing.assert_allclose(probs.sum(-1), 1, atol=1e-6)
    groups = (
        HISTORY_SLOT_STATE_ROWS,
        HISTORY_FIELD_STATE_ROWS,
        HISTORY_REGISTER_STATE_ROWS,
    )
    changed = initial.at[source_row].add(jnp.arange(WIDTH) / WIDTH + 1)
    moved, _, _, _, _ = apply(params, rows, valid, changed)
    row_moved = np.max(np.abs(np.asarray(moved[0] - states[0])), axis=-1) > 1e-5
    expected = np.zeros(NUM_HISTORY_STATE_ROWS, bool)
    expected[source_row] = True
    np.testing.assert_array_equal(row_moved, expected)
    if source_row >= HISTORY_EVENT_STATE_ROWS.stop:
        return
    changed_inner = inner.at[source_row].add(jnp.arange(WIDTH) / WIDTH + 1)
    moved, _, _, _, _ = apply(params, rows, valid, initial, changed_inner)
    row_moved = np.max(np.abs(np.asarray(moved[0] - states[0])), axis=-1) > 1e-5
    for target_rows in groups:
        assert row_moved[target_rows].all()


def test_sequence_carry_matches_full_history_and_padding_holds_exactly(recurrence):
    params, apply, rows, valid, initial, inner = recurrence
    full, _, _, full_final, full_inner = apply(params, rows, valid, initial)
    _, _, _, prefix_final, prefix_inner = apply(params, rows[:2], valid[:2], initial)
    suffix, _, _, suffix_final, suffix_inner = apply(
        params, rows[2:], valid[2:], prefix_final, prefix_inner
    )
    np.testing.assert_allclose(suffix, full[2:], atol=1e-6)
    np.testing.assert_allclose(suffix_final, full_final, atol=1e-6)
    np.testing.assert_allclose(suffix_inner, full_inner, atol=1e-6)
    np.testing.assert_array_equal(full[2], full[1])
    padded, _, _, _, _ = apply(params, rows.at[2].add(100), valid, initial)
    np.testing.assert_array_equal(padded, full)
    idle, _, _, idle_final, idle_inner = apply(
        params, rows, jnp.zeros(STEPS, jnp.bool_), initial
    )
    np.testing.assert_array_equal(idle, jnp.broadcast_to(initial, idle.shape))
    np.testing.assert_array_equal(idle_final, initial)
    np.testing.assert_array_equal(idle_inner, inner)
    future, _, _, _, _ = apply(params, rows.at[-1].add(5), valid, initial)
    np.testing.assert_array_equal(future[:-1], full[:-1])
    assert not np.allclose(future[-1], full[-1])


def test_register_identity_and_shared_attention_have_live_gradients(recurrence):
    params, apply, rows, valid, initial, inner = recurrence

    # Every row of the final state: the registers are read by the trunk
    # alone (queries only), so a slot-only loss would leave
    # `register_identity` untouched by design.
    def loss(tree):
        states, _, _, _, _ = apply(tree, rows, valid, initial)
        return jnp.square(states[-1]).sum()

    gradients = jax.jit(jax.grad(loss))(params)["params"]["sequence_step"]
    assert np.all(
        np.linalg.norm(np.asarray(gradients["register_identity"]), axis=-1) > 0
    )
    for projection in ("query", "key", "value", "out_proj"):
        assert (
            np.linalg.norm(np.asarray(gradients["attention"][projection]["kernel"])) > 0
        )
    cells = {
        **{f"{t}_inner_cell": ("gate", "candidate") for t in ("entity", "field")},
        **{f"{t}_cell": ("gate", "candidate") for t in ("entity", "field", "register")},
    }
    for cell, projections in cells.items():
        for projection in projections:
            assert (
                np.linalg.norm(np.asarray(gradients[cell][projection]["kernel"])) > 0
            ), (cell, projection)


def test_registers_are_queries_only():
    """A register row is read by no one (its column of the attention is
    masked) and its memory reaches no slot; the control is that a slot's
    layer-1 memory reaches the registers."""
    module = PerSlotHistoryEncoder(history_config())
    rows = jax.random.normal(jax.random.key(40), (STEPS, NUM_HISTORY_STATE_ROWS, WIDTH))
    rows = rows.at[:, HISTORY_REGISTER_STATE_ROWS].set(0)
    valid = jnp.ones(STEPS, jnp.bool_)
    initial = jax.random.normal(jax.random.key(41), (NUM_HISTORY_STATE_ROWS, WIDTH))
    inner = jax.random.normal(
        jax.random.key(42), (HISTORY_EVENT_STATE_ROWS.stop, WIDTH)
    )
    touched = jnp.ones((STEPS, NUM_PUBLIC_SLOTS), jnp.bool_)
    params = module.init(
        jax.random.key(43),
        rows,
        jnp.zeros_like(rows),
        touched,
        jnp.ones((STEPS, NUM_ACTIVE_SLOTS), jnp.bool_),
        valid,
        initial,
        inner,
        method=PerSlotHistoryEncoder._recur,
    )
    apply = jax.jit(
        lambda memory, inner: _recur_call(module, params, rows, valid, memory, inner)
    )
    states, probs, _, _, _ = apply(initial, inner)
    np.testing.assert_array_equal(probs[..., HISTORY_REGISTER_STATE_ROWS], 0.0)
    register = HISTORY_REGISTER_STATE_ROWS.start
    moved, _, _, _, _ = apply(initial.at[register].add(3.0), inner)
    np.testing.assert_array_equal(
        moved[:, HISTORY_EVENT_STATE_ROWS], states[:, HISTORY_EVENT_STATE_ROWS]
    )
    moved, _, _, _, _ = apply(initial, inner.at[0].add(3.0))
    assert not np.allclose(
        moved[:, HISTORY_REGISTER_STATE_ROWS], states[:, HISTORY_REGISTER_STATE_ROWS]
    )


@pytest.fixture(scope="module")
def history_case():
    module = PerSlotHistoryEncoder(history_config())
    field = jnp.zeros((STEPS, len(FieldFeature.keys())), jnp.int32)
    field = field.at[:, FieldFeature.FIELD_FEATURE__NUM_RELEVANT].set(1)
    field = field.at[:, FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX0].set(
        jnp.arange(STEPS)
    )
    content = jnp.arange(STEPS * WIDTH, dtype=jnp.float32).reshape(STEPS, WIDTH) / WIDTH
    inputs = dict(
        history_field=field,
        node_embedding_cache=content,
        node_identity_cache=jnp.zeros_like(content),
        active_state_cache=content * 3,
        active_slot_ids=jnp.zeros(STEPS, jnp.int32),
        active_identities=jnp.zeros((NUM_ACTIVE_SLOTS, WIDTH)),
        edge_embedding_cache=content * 2,
        edge_slot_ids=jnp.zeros(STEPS, jnp.int32),
        edge_major_args=jnp.zeros(STEPS, jnp.int32),
        field_row_embeddings=jnp.broadcast_to(
            content[:, None], (STEPS, NUM_FIELD_ROWS, WIDTH)
        ),
        field_identities=jnp.zeros((NUM_FIELD_ROWS, WIDTH)),
        step_request_count=jnp.asarray([2, 4, 0, 8, 12]),
        step_valid=jnp.asarray([True, True, False, True, True]),
    )
    params = jax.jit(module.init)(jax.random.key(38), **inputs)
    apply = jax.jit(module.apply)
    return module, params, apply, inputs


def test_registers_align_to_request_counts_and_enter_carry(history_case):
    module, params, apply, inputs = history_case
    output = apply(params, **inputs)
    initial = params["params"]["initial_memory"][HISTORY_REGISTER_STATE_ROWS]
    requests = jnp.asarray([1, 2, 3, 5, 8, 20])
    aligned = jax.jit(
        lambda result, counts: module.apply(
            params, result, counts, method=PerSlotHistoryEncoder.state_at_requests
        )
    )(output, requests)[3]
    expected = jnp.concatenate(
        (initial[None], output.register_snapshots[jnp.asarray([0, 0, 1, 3, 4])]), axis=0
    )
    np.testing.assert_array_equal(aligned, expected)
    assert not np.allclose(aligned[1], initial)
    carry = history_carry_from(output)
    np.testing.assert_array_equal(carry.register_states, output.final_register_state)
    assert carry.register_states.shape == (NUM_HISTORY_REGISTERS, WIDTH)
    invalid = carry.replace(valid=jnp.asarray(False))
    reset = apply(params, **inputs, carry=invalid)
    np.testing.assert_array_equal(reset.slot_snapshots, output.slot_snapshots)
    np.testing.assert_array_equal(reset.field_snapshots, output.field_snapshots)
    np.testing.assert_array_equal(reset.register_snapshots, output.register_snapshots)
    idle = apply(
        params, **dict(inputs, step_valid=jnp.zeros(STEPS, jnp.bool_)), carry=carry
    )
    np.testing.assert_array_equal(idle.final_register_state, carry.register_states)
    np.testing.assert_array_equal(idle.final_slot_state, carry.slot_states)


def test_field_only_events_update_shared_memory_and_packed_padding_is_inert(
    history_case,
):
    _, params, apply, inputs = history_case
    output = apply(params, **inputs)
    field = (
        inputs["history_field"].at[:, FieldFeature.FIELD_FEATURE__NUM_RELEVANT].set(0)
    )
    field_only = dict(inputs, history_field=field)
    base = apply(params, **field_only)
    assert not np.allclose(
        base.register_snapshots[0],
        params["params"]["initial_memory"][HISTORY_REGISTER_STATE_ROWS],
    )
    padded = apply(
        params,
        **dict(field_only, node_embedding_cache=inputs["node_embedding_cache"] + 100),
    )
    np.testing.assert_array_equal(base.slot_snapshots, padded.slot_snapshots)
    assert not np.allclose(base.slot_snapshots, output.slot_snapshots)
    assert output.step_attention_probs.shape[-2:] == (
        NUM_HISTORY_STATE_ROWS,
        NUM_HISTORY_STATE_ROWS,
    )


def test_history_registers_are_policy_readable_and_cannot_read_private_rows():
    register_rows = np.arange(HISTORY_REGISTER_ROWS.start, HISTORY_REGISTER_ROWS.stop)
    assert len(register_rows) == NUM_HISTORY_REGISTERS
    assert np.isin(register_rows, POLICY_READABLE_ROWS).all()
    secret_rows = np.r_[
        np.arange(OPP_PRIVATE_ROWS.start, OPP_PRIVATE_ROWS.stop), VALUE_CLS_ROW
    ]
    assert not SEQUENCE_READ_MASK[np.ix_(register_rows, secret_rows)].any()
    assert not SEQUENCE_READ_MASK[np.ix_(register_rows, PRIVATE_TIER_ROWS)].any()
    assert SEQUENCE_READ_MASK[np.ix_(POLICY_READABLE_ROWS, register_rows)].all()
    assert SEQUENCE_READ_MASK[np.ix_(register_rows, PUBLIC_TIER_ROWS)].all()
