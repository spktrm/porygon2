"""`PerSlotHistoryEncoder` structural pins, at the recurrence and step-GAT
level so they run without the shared model fixture.

The recurrence (2026-09-02): two gated linear scans, field first. Slot k's
state must read its OWN input, its own past and the field states -- never
another slot's (the gestalt mean was deleted the same day), and the field
never reads a slot. The controls prove each pin can fail: the field input
moves every slot, and a slot's own input moves itself. The scan itself is
pinned against a serial lax.scan of the same recurrence, and a unit no
step writes holds its initial state bit-exactly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import ConfigDict

from rl.model.constants import NUM_PUBLIC_SLOTS
from rl.model.history_encoder import (
    NUM_FIELD_ROWS,
    PerSlotHistoryEncoder,
    StepAttention,
    gated_linear_scan,
)

ENTITY_SIZE = 32
NUM_HEADS = 2
QK_SIZE = 8
HISTORY = 6


@pytest.fixture(scope="module")
def recur():
    cfg = ConfigDict(
        dict(
            entity_size=ENTITY_SIZE,
            dtype=jnp.float32,
            history_step=dict(num_heads=NUM_HEADS, qk_size=QK_SIZE),
        )
    )
    module = PerSlotHistoryEncoder(cfg)
    key_params, key_slots, key_field = jax.random.split(jax.random.key(0), 3)
    slot_inputs = jax.random.normal(
        key_slots, (HISTORY, NUM_PUBLIC_SLOTS, 2 * ENTITY_SIZE)
    )
    field_inputs = jax.random.normal(
        key_field, (HISTORY, NUM_FIELD_ROWS, 2 * ENTITY_SIZE)
    )
    touched = jnp.ones((HISTORY, NUM_PUBLIC_SLOTS), bool)
    step_valid = jnp.ones((HISTORY,), bool)
    params = module.init(
        key_params,
        slot_inputs,
        field_inputs,
        touched,
        step_valid,
        method=PerSlotHistoryEncoder._recur,
    )

    @jax.jit
    def run(slot_inputs, field_inputs, touched=touched, step_valid=step_valid):
        return module.apply(
            params,
            slot_inputs,
            field_inputs,
            touched,
            step_valid,
            method=PerSlotHistoryEncoder._recur,
        )

    return run, slot_inputs, field_inputs


def test_slot_never_reads_another_slot(recur):
    run, slot_inputs, field_inputs = recur
    base_slots, _, _ = run(slot_inputs, field_inputs)
    moved_slots, _, _ = run(slot_inputs.at[:, 3].add(1.0), field_inputs)
    others = np.arange(NUM_PUBLIC_SLOTS) != 3
    np.testing.assert_array_equal(
        np.asarray(base_slots)[:, others], np.asarray(moved_slots)[:, others]
    )
    # Positive control 1: the perturbed slot itself moves, at every step.
    per_step = np.abs(np.asarray(moved_slots - base_slots)[:, 3]).max(axis=-1)
    assert (per_step > 1e-3).all()


def test_field_state_reaches_every_slot(recur):
    """Positive control 2: the shared state the slots DO read. The field
    state after step 0 is an input of every slot from step 1 on."""
    run, slot_inputs, field_inputs = recur
    base_slots, _, _ = run(slot_inputs, field_inputs)
    moved_slots, _, _ = run(slot_inputs, field_inputs.at[0, 0].add(1.0))
    per_slot = np.abs(np.asarray(moved_slots - base_slots)[1:]).max(axis=-1)
    assert (per_slot > 1e-3).all()
    np.testing.assert_array_equal(np.asarray(base_slots)[0], np.asarray(moved_slots)[0])


def test_field_never_reads_slot_states(recur):
    run, slot_inputs, field_inputs = recur
    _, base_field, _ = run(slot_inputs, field_inputs)
    _, moved_field, _ = run(slot_inputs + 1.0, field_inputs)
    np.testing.assert_array_equal(np.asarray(base_field), np.asarray(moved_field))


def test_gated_linear_scan_matches_serial_recurrence():
    key_gate, key_cand, key_write, key_init = jax.random.split(jax.random.key(3), 4)
    shape = (37, 5, 8)
    gate = jax.nn.sigmoid(jax.random.normal(key_gate, shape))
    candidate = jax.random.normal(key_cand, shape)
    write = jax.random.bernoulli(key_write, 0.6, shape[:2])
    initial = jax.random.normal(key_init, shape[1:])

    def serial_step(state, inputs):
        step_gate, step_candidate, step_write = inputs
        effective = step_write[:, None] * step_gate
        state = (1.0 - effective) * state + effective * step_candidate
        return state, state

    _, expected = jax.lax.scan(
        serial_step, initial, (gate, candidate, write.astype(jnp.float32))
    )
    actual = jax.jit(gated_linear_scan)(gate, candidate, write, initial)
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5)


def test_unwritten_units_hold_initial_exactly():
    key_gate, key_cand, key_init = jax.random.split(jax.random.key(4), 3)
    shape = (16, 4, 8)
    gate = jax.nn.sigmoid(jax.random.normal(key_gate, shape))
    candidate = jax.random.normal(key_cand, shape)
    initial = jax.random.normal(key_init, shape[1:])
    write = jnp.ones(shape[:2], bool).at[:, 2].set(False).at[5:, :].set(False)
    states = np.asarray(jax.jit(gated_linear_scan)(gate, candidate, write, initial))
    np.testing.assert_array_equal(states[:, 2], np.broadcast_to(initial[2], (16, 8)))
    # from step 5 on nothing writes: every unit holds its step-4 state
    np.testing.assert_array_equal(states[5:], np.broadcast_to(states[4], (11, 4, 8)))
    # control: a written unit does leave its initial state
    assert np.abs(states[0, 0] - np.asarray(initial[0])).max() > 1e-3


# ---- the step GAT ---------------------------------------------------------
# One attention layer over the rows of a history step (2026-09-02, replacing
# the masked source mean). Pins: the zeros-init output projection makes it
# exactly silent at init yet trainable; a padded row places and receives no
# mass; a 1-row step is exactly its own value.

NUM_STEPS = 3
NUM_ROWS = 4
ROW_WIDTH = 2 * ENTITY_SIZE + 3


@pytest.fixture(scope="module")
def step_attention():
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


def test_attn_out_is_silent_at_init_and_not_after(step_attention):
    apply, params, live_params, rows, row_mask = step_attention
    out, probs = apply(params, rows, row_mask)
    assert jnp.all(out == 0)
    assert jnp.all(jnp.isfinite(probs))
    live, _ = apply(live_params, rows, row_mask)
    assert jnp.any(live[row_mask] != 0)


def test_attn_out_has_gradient_at_init(step_attention):
    apply, params, _, rows, row_mask = step_attention
    weights = jax.random.normal(jax.random.key(2), (NUM_STEPS, NUM_ROWS, ENTITY_SIZE))

    def objective(tree):
        out, _ = apply(tree, rows, row_mask)
        return (out * weights).sum()

    grads = jax.grad(objective)(params)
    assert jnp.any(grads["params"]["attn_out"]["kernel"] != 0)


def test_padded_row_places_no_mass_and_moves_nothing(step_attention):
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


def test_one_row_step_is_its_own_value(step_attention):
    apply, _, live_params, rows, row_mask = step_attention
    _, probs = apply(live_params, rows, row_mask)
    assert jnp.all(probs[0, :, 0, 0] == 1.0)
    assert jnp.all(probs[0, :, 0, 1:] == 0.0)
