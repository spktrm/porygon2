"""Contracts of the history recurrence's pieces: the input-gated cell, the
associative scan, and what the step attention does and does not carry."""

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from rl.model.constants import (
    HISTORY_ACTIVE_STATE_ROWS,
    HISTORY_EVENT_STATE_ROWS,
    HISTORY_FIELD_STATE_ROWS,
    HISTORY_REGISTER_STATE_ROWS,
    HISTORY_SLOT_STATE_ROWS,
    NUM_ACTIVE_SLOTS,
    NUM_HISTORY_STATE_ROWS,
    NUM_PUBLIC_SLOTS,
)
from rl.model.history_encoder import (
    GatedLinearCell,
    HistorySequenceStep,
    gated_linear_scan,
)

WIDTH = 8
STEPS = 7
UNITS = 3


def _serial_scan(gate, candidate, write, initial):
    def step(carry, inputs):
        gate_t, candidate_t, write_t = inputs
        rate = write_t[..., None] * gate_t
        carry = (1.0 - rate) * carry + rate * candidate_t
        return carry, carry

    _, states = jax.lax.scan(step, initial, (gate, candidate, write))
    return states


def test_associative_scan_matches_the_serial_recurrence_and_holds_init() -> None:
    keys = jax.random.split(jax.random.key(0), 3)
    gate = jax.nn.sigmoid(jax.random.normal(keys[0], (STEPS, UNITS, WIDTH)))
    candidate = jax.random.normal(keys[1], (STEPS, UNITS, WIDTH))
    initial = jax.random.normal(keys[2], (UNITS, WIDTH))
    write = jnp.ones((STEPS, UNITS), jnp.bool_)
    # Unit 1 is never written; step 3 writes nothing.
    write = write.at[:, 1].set(False).at[3].set(False)
    states = jax.jit(gated_linear_scan)(gate, candidate, write, initial)
    serial = jax.jit(_serial_scan)(gate, candidate, write.astype(jnp.float32), initial)
    np.testing.assert_allclose(states, serial, atol=1e-5)
    np.testing.assert_array_equal(
        states[:, 1], jnp.broadcast_to(initial[1], (STEPS, WIDTH))
    )
    np.testing.assert_array_equal(states[3], states[2])
    assert states.dtype == jnp.float32
    # Control: a written unit does move.
    assert not np.allclose(states[-1, 0], initial[0])


def test_bf16_gate_and_candidate_keep_a_small_update_in_the_f32_carry() -> None:
    cell = GatedLinearCell(WIDTH, dtype=jnp.bfloat16)
    xs = jnp.zeros((STEPS, UNITS, WIDTH), jnp.bfloat16)
    gate, candidate = cell.apply(cell.init(jax.random.key(1), xs), xs)
    assert gate.dtype == jnp.bfloat16 and candidate.dtype == jnp.bfloat16
    # The coefficients the cell emits are bf16; a 1e-3 write of 0 onto a
    # carry of 100 moves it by 0.1, under bf16's ulp at 100 (0.5). The f32
    # carry keeps it.
    gate = jnp.full_like(gate, 1e-3)
    candidate = jnp.zeros_like(candidate)
    states = gated_linear_scan(
        gate,
        candidate,
        jnp.ones((STEPS, UNITS), jnp.bool_),
        jnp.full((UNITS, WIDTH), 100.0),
    )
    assert states.dtype == jnp.float32
    step = float(jnp.abs(states[1] - states[0]).max())
    assert 0.05 < step < 0.2
    np.testing.assert_array_equal(
        states[0].astype(jnp.bfloat16), jnp.full((UNITS, WIDTH), 100.0, jnp.bfloat16)
    )


def _step_module():
    cfg = ConfigDict(
        dict(
            entity_size=WIDTH,
            dtype=jnp.float32,
            history_step=dict(num_heads=2, qk_size=4),
        )
    )
    module = HistorySequenceStep(cfg)
    events = jax.random.normal(
        jax.random.key(77), (STEPS, NUM_HISTORY_STATE_ROWS, WIDTH)
    )
    events = events.at[:, HISTORY_REGISTER_STATE_ROWS].set(0)
    identities = jnp.ones_like(events)
    touched = jnp.ones((STEPS, NUM_PUBLIC_SLOTS), jnp.bool_)
    active_touched = jnp.ones((STEPS, NUM_ACTIVE_SLOTS), jnp.bool_)
    valid = jnp.ones(STEPS, jnp.bool_)
    inner0 = jax.random.normal(
        jax.random.key(78), (HISTORY_EVENT_STATE_ROWS.stop, WIDTH)
    )
    memory0 = jax.random.normal(jax.random.key(79), (NUM_HISTORY_STATE_ROWS, WIDTH))
    arguments = (events, identities, touched, active_touched, valid, inner0, memory0)
    params = jax.jit(module.init)(jax.random.key(80), *arguments)
    jitted = jax.jit(module.apply)

    def apply(tree, *overrides):
        replaced = list(arguments)
        for index, value in overrides:
            replaced[index] = value
        return jitted(tree, *replaced)

    return params, apply, arguments


def _copy(tree):
    return jax.tree.map(lambda leaf: leaf, tree)


def test_norm_and_identities_reach_memory_only_through_attention() -> None:
    params, apply, (events, identities, *_) = _step_module()
    muted = _copy(params)
    muted["params"]["attention"]["out_proj"]["kernel"] = jnp.zeros((WIDTH, WIDTH))

    def change_read_branch(tree):
        changed = _copy(tree)
        changed["params"]["group_identity"] += 10
        changed["params"]["register_identity"] += 20
        changed["params"]["input_norm"]["scale"] *= 3
        return changed

    base = apply(muted)[0]
    np.testing.assert_array_equal(base, apply(change_read_branch(muted))[0])
    # Controls: the same changes move memory through a live out_proj, and
    # the event rows reach memory without it.
    assert not np.allclose(apply(params)[0], apply(change_read_branch(params))[0])
    assert not np.allclose(base, apply(muted, (0, events * 3))[0])


def test_cell_weights_are_separate_by_type_and_shared_within_type() -> None:
    params, apply, (events, identities, *_, inner0, memory0) = _step_module()
    params = _copy(params)
    params["params"]["attention"]["out_proj"]["kernel"] = jnp.zeros((WIDTH, WIDTH))
    uniform = (
        (0, jnp.full_like(events, 0.2).at[:, HISTORY_REGISTER_STATE_ROWS].set(0)),
        (5, jnp.full_like(inner0, 0.4)),
        (6, jnp.full_like(memory0, 0.4)),
    )
    baseline = apply(params, *uniform)[0]
    groups = (
        ("entity", HISTORY_SLOT_STATE_ROWS),
        ("field", HISTORY_FIELD_STATE_ROWS),
        ("active", HISTORY_ACTIVE_STATE_ROWS),
        ("register", HISTORY_REGISTER_STATE_ROWS),
    )
    for token_type, state_rows in groups:
        np.testing.assert_allclose(
            baseline[:, state_rows],
            jnp.broadcast_to(
                baseline[:, state_rows][:, :1], baseline[:, state_rows].shape
            ),
            atol=1e-6,
        )
        changed = _copy(params)
        changed["params"][f"{token_type}_cell"]["candidate"]["bias"] += 5
        updated = apply(changed, *uniform)[0]
        assert not np.allclose(updated[:, state_rows], baseline[:, state_rows])
        for other_type, other_rows in groups:
            if other_type != token_type:
                np.testing.assert_array_equal(
                    updated[:, other_rows], baseline[:, other_rows]
                )


def test_identity_changes_reach_memory_only_through_attention_weights() -> None:
    params, apply, (events, identities, *_) = _step_module()

    def move_identities(tree):
        changed = _copy(tree)
        changed["params"]["group_identity"] += 1.0
        changed["params"]["register_identity"] += 1.0
        return changed

    moved_identities = (1, identities + 2.0)
    baseline, _, baseline_probs, _ = apply(params)
    moved, _, moved_probs, _ = apply(move_identities(params), moved_identities)
    assert not np.allclose(baseline_probs, moved_probs)
    assert not np.allclose(baseline, moved)
    # A zero query fixes the attention weights; the value path stays live,
    # so identity-valued writes would still show.
    fixed = _copy(params)
    fixed["params"]["attention"]["query"]["kernel"] = jnp.zeros_like(
        fixed["params"]["attention"]["query"]["kernel"]
    )
    fixed_baseline, _, fixed_probs, _ = apply(fixed)
    fixed_moved, _, fixed_moved_probs, _ = apply(
        move_identities(fixed), moved_identities
    )
    np.testing.assert_array_equal(fixed_probs, fixed_moved_probs)
    np.testing.assert_array_equal(fixed_baseline, fixed_moved)
