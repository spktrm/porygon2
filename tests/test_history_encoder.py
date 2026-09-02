"""`PerSlotHistoryEncoder` structural pins, at the scan-step level so they
run without the shared model fixture.

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
from rl.model.history_encoder import NUM_FIELD_ROWS, PerSlotHistoryEncoder

ENTITY_SIZE = 32


@pytest.fixture(scope="module")
def scan_step():
    cfg = ConfigDict(dict(entity_size=ENTITY_SIZE, dtype=jnp.float32))
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
