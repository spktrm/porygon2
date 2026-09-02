"""`PerSlotHistoryEncoder` structural pins, at the scan-step and step-GAT
level so they run without the shared model fixture.

The gestalt (a mean over the other slots' states fed to every slot's
input) was deleted 2026-09-02: within one history step, slot k's update
must read its OWN state, its own precomputed input and the field states
-- never another slot's state. The two controls prove the pin can fail:
the field carry moves every slot, and a slot's own carry moves itself.
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
)

ENTITY_SIZE = 32
NUM_HEADS = 2
QK_SIZE = 8


@pytest.fixture(scope="module")
def scan_step():
    cfg = ConfigDict(
        dict(
            entity_size=ENTITY_SIZE,
            dtype=jnp.float32,
            history_step=dict(num_heads=NUM_HEADS, qk_size=QK_SIZE),
        )
    )
    module = PerSlotHistoryEncoder(cfg)
    key = jax.random.key(0)
    key_params, key_slots, key_field, key_pre, key_gates = jax.random.split(key, 5)
    h_slots = jax.random.normal(key_slots, (NUM_PUBLIC_SLOTS, ENTITY_SIZE))
    h_field = jax.random.normal(key_field, (NUM_FIELD_ROWS, ENTITY_SIZE))
    slot_pre = tuple(
        jax.random.normal(k, (NUM_PUBLIC_SLOTS, ENTITY_SIZE))
        for k in jax.random.split(key_pre, 3)
    )
    field_gates = tuple(
        jax.random.normal(k, (NUM_FIELD_ROWS, ENTITY_SIZE))
        for k in jax.random.split(key_gates, 3)
    )
    touched = jnp.ones((NUM_PUBLIC_SLOTS,), jnp.float32)
    valid = jnp.asarray(1.0, jnp.float32)
    xs = (slot_pre, field_gates, touched, valid)
    params = module.init(
        key_params, (h_slots, h_field), xs, method=PerSlotHistoryEncoder._scan_step
    )
    step = jax.jit(
        lambda carry: module.apply(
            params, carry, xs, method=PerSlotHistoryEncoder._scan_step
        )[0]
    )
    return step, (h_slots, h_field)


def _perturb_rows(states, rows, scale=1.0):
    bump = jnp.zeros_like(states).at[rows].set(scale)
    return states + bump


def test_slot_update_never_reads_another_slots_state(scan_step):
    step, (h_slots, h_field) = scan_step
    base_slots, _ = step((h_slots, h_field))
    moved_slots, _ = step((_perturb_rows(h_slots, 3), h_field))
    others = np.arange(NUM_PUBLIC_SLOTS) != 3
    np.testing.assert_array_equal(
        np.asarray(base_slots)[others], np.asarray(moved_slots)[others]
    )
    # Positive control 1: the perturbed slot itself moves.
    assert np.abs(np.asarray(moved_slots - base_slots)[3]).max() > 1e-3


def test_field_state_reaches_every_slot(scan_step):
    """Positive control 2: the shared carry the slots DO read. If the pin
    above passed because the step ignored its carry altogether, this
    would fail."""
    step, (h_slots, h_field) = scan_step
    base_slots, _ = step((h_slots, h_field))
    moved_slots, _ = step((h_slots, _perturb_rows(h_field, 0)))
    per_slot = np.abs(np.asarray(moved_slots - base_slots)).max(axis=-1)
    assert (per_slot > 1e-3).all()


def test_field_update_never_reads_slot_states(scan_step):
    step, (h_slots, h_field) = scan_step
    _, base_field = step((h_slots, h_field))
    _, moved_field = step((h_slots + 1.0, h_field))
    np.testing.assert_array_equal(np.asarray(base_field), np.asarray(moved_field))


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
