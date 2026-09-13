"""One 19-row recurrent history sequence with bidirectional memory reads."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import ConfigDict

from rl.environment.protos.features_pb2 import FieldFeature
from rl.model.constants import (
    HISTORY_FIELD_STATE_ROWS,
    HISTORY_REGISTER_ROWS,
    HISTORY_REGISTER_STATE_ROWS,
    HISTORY_SLOT_STATE_ROWS,
    NUM_HISTORY_REGISTERS,
    NUM_HISTORY_STATE_ROWS,
    OPP_PRIVATE_ROWS,
    POLICY_READABLE_ROWS,
    SEQUENCE_READ_MASK,
    VALUE_CLS_ROW,
)
from rl.model.history_encoder import (
    NUM_FIELD_ROWS,
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


@pytest.fixture(scope="module")
def recurrence():
    module = PerSlotHistoryEncoder(history_config())
    rows = jax.random.normal(jax.random.key(35), (STEPS, NUM_HISTORY_STATE_ROWS, WIDTH))
    rows = rows.at[:, HISTORY_REGISTER_STATE_ROWS].set(0)
    valid = jnp.asarray([True, True, False, True, True])
    initial = (
        jax.random.normal(jax.random.key(36), (NUM_HISTORY_STATE_ROWS, WIDTH)) * 0.02
    )
    params = jax.jit(
        lambda: module.init(
            jax.random.key(37),
            rows,
            valid,
            initial,
            method=PerSlotHistoryEncoder._recur,
        )
    )()
    apply = jax.jit(
        lambda tree, inputs, mask, memory: module.apply(
            tree, inputs, mask, memory, method=PerSlotHistoryEncoder._recur
        )
    )
    return params, apply, rows, valid, initial


@pytest.mark.parametrize(
    "source_row",
    [
        HISTORY_SLOT_STATE_ROWS.start,
        HISTORY_FIELD_STATE_ROWS.start,
        HISTORY_REGISTER_STATE_ROWS.start,
    ],
)
def test_every_modality_reads_previous_memory_from_every_other(recurrence, source_row):
    params, apply, rows, valid, initial = recurrence
    states, probs, _, final = apply(params, rows, valid, initial)
    changed = initial.at[source_row].add(jnp.arange(WIDTH) / WIDTH + 1)
    moved, _, _, _ = apply(params, rows, valid, changed)
    assert probs.shape == (STEPS, 2, 19, 19)
    assert states.shape == (STEPS, NUM_HISTORY_STATE_ROWS, WIDTH)
    assert final.dtype == jnp.float32
    for target_rows in (
        HISTORY_SLOT_STATE_ROWS,
        HISTORY_FIELD_STATE_ROWS,
        HISTORY_REGISTER_STATE_ROWS,
    ):
        differences = np.max(
            np.abs(np.asarray(moved[0, target_rows] - states[0, target_rows])), axis=-1
        )
        assert (differences > 1e-5).all()
    np.testing.assert_allclose(probs.sum(-1), 1, atol=1e-6)


def test_sequence_carry_matches_full_history_and_padding_holds_exactly(recurrence):
    params, apply, rows, valid, initial = recurrence
    full, _, _, full_final = apply(params, rows, valid, initial)
    _, _, _, prefix_final = apply(params, rows[:2], valid[:2], initial)
    suffix, _, _, suffix_final = apply(params, rows[2:], valid[2:], prefix_final)
    np.testing.assert_allclose(suffix, full[2:], atol=1e-6)
    np.testing.assert_allclose(suffix_final, full_final, atol=1e-6)
    np.testing.assert_array_equal(full[2], full[1])
    padded, _, _, _ = apply(params, rows.at[2].add(100), valid, initial)
    np.testing.assert_array_equal(padded, full)
    idle, _, _, idle_final = apply(params, rows, jnp.zeros(STEPS, jnp.bool_), initial)
    np.testing.assert_array_equal(idle, jnp.broadcast_to(initial, idle.shape))
    np.testing.assert_array_equal(idle_final, initial)
    future, _, _, _ = apply(params, rows.at[-1].add(5), valid, initial)
    np.testing.assert_array_equal(future[:-1], full[:-1])
    assert not np.allclose(future[-1], full[-1])


def test_register_identity_and_shared_attention_have_live_gradients(recurrence):
    params, apply, rows, valid, initial = recurrence

    def loss(tree):
        states, _, _, _ = apply(tree, rows, valid, initial)
        return jnp.square(states[-1, HISTORY_SLOT_STATE_ROWS]).sum()

    gradients = jax.jit(jax.grad(loss))(params)["params"]["sequence_step"]
    assert np.all(
        np.linalg.norm(np.asarray(gradients["register_identity"]), axis=-1) > 0
    )
    for projection in ("query", "key", "value", "attn_out"):
        assert (
            np.linalg.norm(np.asarray(gradients["attention"][projection]["kernel"])) > 0
        )
    for token_type in ("entity", "field", "register"):
        for projection in ("ir", "hr", "iz", "hz", "in", "hn"):
            assert (
                np.linalg.norm(
                    np.asarray(gradients[f"{token_type}_gru"][projection]["kernel"])
                )
                > 0
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
        node_content_cache=content,
        edge_embedding_cache=content * 2,
        edge_slot_ids=jnp.zeros(STEPS, jnp.int32),
        edge_major_args=jnp.zeros(STEPS, jnp.int32),
        field_row_embeddings=jnp.broadcast_to(
            content[:, None], (STEPS, NUM_FIELD_ROWS, WIDTH)
        ),
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
    assert output.step_attention_probs.shape[-2:] == (19, 19)


def test_history_registers_are_policy_readable_and_cannot_read_private_rows():
    register_rows = np.arange(HISTORY_REGISTER_ROWS.start, HISTORY_REGISTER_ROWS.stop)
    assert len(register_rows) == NUM_HISTORY_REGISTERS
    assert np.isin(register_rows, POLICY_READABLE_ROWS).all()
    secret_rows = np.r_[
        np.arange(OPP_PRIVATE_ROWS.start, OPP_PRIVATE_ROWS.stop), VALUE_CLS_ROW
    ]
    assert not SEQUENCE_READ_MASK[np.ix_(register_rows, secret_rows)].any()
    assert SEQUENCE_READ_MASK[np.ix_(POLICY_READABLE_ROWS, register_rows)].all()
    assert SEQUENCE_READ_MASK[np.ix_(register_rows, POLICY_READABLE_ROWS)].all()
