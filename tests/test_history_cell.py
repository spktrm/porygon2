"""Contracts of the stacked recurrence's pieces: the input-gated cell, the
associative scan, and what the step attention does and does not carry."""

import jax
import jax.numpy as jnp
import numpy as np

from rl.model.history_encoder import GatedLinearCell, gated_linear_scan

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
