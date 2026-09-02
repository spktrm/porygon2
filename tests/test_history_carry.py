"""The optional actor-side history carry (2026-09-02).

Contract: a carry with `valid=False` -- whatever its leaves hold -- is the
from-scratch forward bit for bit, so no caller is ever obliged to keep one
(the learner, the offline tools and the standalone server send `()` leaves
and get the same function with no select in the trace). The positive
control is the same garbage under `valid=True`, which must move the policy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.environment.interfaces import HistoryCarry
from rl.environment.protos.features_pb2 import FieldFeature
from rl.model.constants import NUM_PUBLIC_SLOTS
from rl.model.heads import HeadParams

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

NUM_FIELD_ROWS = 3


def _garbage_carry(width: int, valid: bool) -> HistoryCarry:
    key_slots, key_field, key_nodes = jax.random.split(jax.random.key(7), 3)
    return HistoryCarry(
        slot_states=jax.random.normal(key_slots, (NUM_PUBLIC_SLOTS, width)),
        field_states=jax.random.normal(key_field, (NUM_FIELD_ROWS, width)),
        node_snapshots=jax.random.normal(key_nodes, (NUM_PUBLIC_SLOTS, width)),
        valid=jnp.asarray(valid),
    )


def _width(params) -> int:
    return params["params"]["encoder"]["history_encoder"]["initial_slot_state"].shape[
        -1
    ]


def test_invalid_carry_is_the_from_scratch_forward_bit_for_bit(
    real_model_and_trajectory, real_model_apply
):
    network, params, actor_input, actor_output = real_model_and_trajectory
    base = real_model_apply(params, actor_input, actor_output, HeadParams())
    width = _width(params)

    ignored = real_model_apply(
        params,
        actor_input.replace(history_carry=_garbage_carry(width, valid=False)),
        actor_output,
        HeadParams(),
    )
    np.testing.assert_array_equal(
        np.asarray(base.action_head.log_policy, dtype=np.float32),
        np.asarray(ignored.action_head.log_policy, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(base.value_head.log_probs, dtype=np.float32),
        np.asarray(ignored.value_head.log_probs, dtype=np.float32),
    )

    # Control: the same leaves, believed, move the policy.
    used = real_model_apply(
        params,
        actor_input.replace(history_carry=_garbage_carry(width, valid=True)),
        actor_output,
        HeadParams(),
    )
    assert not np.array_equal(
        np.asarray(base.action_head.log_policy, dtype=np.float32),
        np.asarray(used.action_head.log_policy, dtype=np.float32),
    )
    # The forward hands its post-window state back for the next request.
    assert used.history_carry.slot_states.shape[-2:] == (NUM_PUBLIC_SLOTS, width)
    assert used.history_carry.slot_states.dtype == jnp.float32


def test_zero_new_steps_returns_the_carry_itself(real_model_and_trajectory):
    """A window with no valid step: `state_at_requests` falls back to what
    the window started from, i.e. the carry when it is valid and the learned
    h0 when it is not."""
    network, params, actor_input, _ = real_model_and_trajectory
    width = _width(params)
    field = np.asarray(actor_input.history.field).copy()
    field[..., FieldFeature.FIELD_FEATURE__VALID] = 0
    empty_history = actor_input.history.replace(field=jnp.asarray(field))

    @jax.jit
    def encode(carry):
        return network.apply(
            params,
            actor_input.env,
            actor_input.packed_history,
            empty_history,
            carry,
            method=lambda module, *args: module.encoder.encode_history(*args),
        )

    carry = _garbage_carry(width, valid=True)
    slots, field_states, nodes, _ = encode(carry)
    compute_dtype = slots.dtype
    for request_index in range(slots.shape[0]):
        np.testing.assert_array_equal(
            np.asarray(slots[request_index]),
            np.asarray(carry.slot_states.astype(compute_dtype)),
        )
        np.testing.assert_array_equal(
            np.asarray(field_states[request_index]),
            np.asarray(carry.field_states.astype(compute_dtype)),
        )
        np.testing.assert_array_equal(
            np.asarray(nodes[request_index]),
            np.asarray(carry.node_snapshots.astype(compute_dtype)),
        )

    h0_slots = params["params"]["encoder"]["history_encoder"]["initial_slot_state"]
    slots, _, nodes, _ = encode(_garbage_carry(width, valid=False))
    np.testing.assert_array_equal(
        np.asarray(slots[0]),
        np.asarray(jnp.repeat(h0_slots, NUM_PUBLIC_SLOTS, axis=0).astype(slots.dtype)),
    )
    assert not np.asarray(nodes).any()
