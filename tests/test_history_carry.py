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
from rl.model.utils import open_zero_init_paths

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


@pytest.fixture(scope="module")
def carry_params(real_model_and_trajectory):
    """The flat readout is zero-init (every logit exactly 0 at step 0), so
    on fresh params log_policy is uniform whatever the history says and a
    "the carry moves the policy" control passes or fails vacuously. Open
    the readout's zero paths so the pathway is live and the controls bite."""
    return open_zero_init_paths(real_model_and_trajectory[1], ["action_head"])


def _width(params) -> int:
    return params["params"]["encoder"]["history_encoder"]["initial_slot_state"].shape[
        -1
    ]


def test_invalid_carry_is_the_from_scratch_forward_bit_for_bit(
    real_model_and_trajectory, real_model_apply, carry_params
):
    network, _, actor_input, actor_output = real_model_and_trajectory
    params = carry_params
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


def test_zero_new_steps_returns_the_carry_itself(
    real_model_and_trajectory, carry_params
):
    """A window with no valid step: `state_at_requests` falls back to what
    the window started from, i.e. the carry when it is valid and the learned
    h0 when it is not."""
    network, _, actor_input, _ = real_model_and_trajectory
    params = carry_params
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


def _window_at(full_window, request_count: int):
    """What the service sent at request ``request_count``: the full
    window's steps stamped <= that count (edges are stamped with the
    request count they were ingested under, monotone), the packed rows
    those steps reference, and that request's env row."""
    field = np.asarray(full_window.history.field).copy()
    later = field[:, FieldFeature.FIELD_FEATURE__REQUEST_COUNT] > request_count
    later &= field[:, FieldFeature.FIELD_FEATURE__VALID].astype(bool)
    if later.any():
        first_later = int(np.argmax(later))
        row_end = int(
            field[first_later, FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX0]
        )
    else:
        row_end = None
    field[later] = 0

    def cut_rows(cache):
        cache = np.asarray(cache).copy()
        if row_end is not None:
            cache[row_end:] = 0
        return cache

    return full_window.replace(
        env=jax.tree.map(
            lambda x: np.asarray(x)[request_count : request_count + 1], full_window.env
        ),
        history=full_window.history.replace(field=field),
        packed_history=jax.tree.map(cut_rows, full_window.packed_history),
    )


def _squeeze_request(output):
    return jax.tree.map(lambda x: np.asarray(x)[0], output)


def _max_diff(left, right) -> float:
    """Largest gap on either compared readout: log_policy or value log-probs."""
    policy = np.abs(
        np.asarray(left.action_head.log_policy, np.float32)
        - np.asarray(right.action_head.log_policy, np.float32)
    ).max()
    value = np.abs(
        np.asarray(left.value_head.log_probs, np.float32)
        - np.asarray(right.value_head.log_probs, np.float32)
    ).max()
    return float(max(policy, value))


def test_suffix_carry_replays_the_game_within_bf16(
    real_model_and_trajectory, real_model_apply, carry_params
):
    """Test (a): every request of the ex.bin game served from the previous
    request's carry over its suffix alone matches the full-window forward
    on log_policy and value log-probs within the bf16 GEMM leading-dim
    class (tests/test_chunking.py), and the carried f32 states at the end
    match the full scan's. Control: a shifted carry moves the policy.

    The bound is MEASURED, not assumed: the same window tail-clipped to
    the learner's stored width with no carry is pure shape noise (content
    identical, leading dims different), and the carry may not exceed
    that floor or 0.05, whichever is larger. Under the opened readout the
    floor on log_policy reads ~0.14 (2026-09-02: carry worst 0.093
    against it; value log-probs 0.022 against a 0.026 floor) -- the
    chunking test's 0.05 was calibrated on value log-probs alone."""
    from rl.environment.utils import clip_history_suffix, clip_history_windows_tail
    from rl.model.history_encoder import invalid_history_carry
    from rl.online.config import get_learner_config
    from rl.online.player_actor import _last_step_index

    network, _, full_window, full_output = real_model_and_trajectory
    params = carry_params
    width = _width(params)
    num_requests = int(np.asarray(full_window.env.done).shape[0])

    def forward(actor_input, request_count):
        # The session model is train=True: the readout is teacher-forced
        # on the stored action, so hand it the trajectory's own row.
        taken = jax.tree.map(
            lambda x: np.asarray(x)[request_count : request_count + 1], full_output
        )
        return _squeeze_request(
            real_model_apply(params, actor_input, taken, HeadParams())
        )

    def policy_diff(left, right) -> float:
        return float(
            np.abs(
                np.asarray(left.action_head.log_policy, np.float32)
                - np.asarray(right.action_head.log_policy, np.float32)
            ).max()
        )

    def value_diff(left, right) -> float:
        return float(
            np.abs(
                np.asarray(left.value_head.log_probs, np.float32)
                - np.asarray(right.value_head.log_probs, np.float32)
            ).max()
        )

    # The floor: the same window at the learner's stored width (the
    # tail clip every chunk goes through), no carry -- content-identical,
    # shapes differ, and that alone moves the bf16 GEMMs.
    stored_history_length = get_learner_config().player_history_length
    carry = None
    last_step_index = -1
    worst_policy = 0.0
    worst_value = 0.0
    floor_policy = 0.0
    floor_value = 0.0
    for request_count in range(num_requests):
        window = _window_at(full_window, request_count)
        full = forward(
            window.replace(history_carry=invalid_history_carry(width)), request_count
        )
        tail_history, tail_packed = clip_history_windows_tail(
            window.history, window.packed_history, stored_history_length
        )
        tail = forward(
            window.replace(
                history=tail_history,
                packed_history=tail_packed,
                history_carry=invalid_history_carry(width),
            ),
            request_count,
        )
        floor_policy = max(floor_policy, policy_diff(full, tail))
        floor_value = max(floor_value, value_diff(full, tail))
        suffix, _ = clip_history_suffix(window, last_step_index)
        assert suffix is not None
        if carry is None:
            resumed_input = suffix.replace(history_carry=invalid_history_carry(width))
        else:
            resumed_input = suffix.replace(history_carry=carry)
        resumed = forward(resumed_input, request_count)
        worst_policy = max(worst_policy, policy_diff(full, resumed))
        worst_value = max(worst_value, value_diff(full, resumed))
        carry = resumed.history_carry
        last_step_index = _last_step_index(window)
    policy_bound = max(0.05, floor_policy)
    value_bound = max(0.05, floor_value)
    assert worst_policy <= policy_bound, (worst_policy, floor_policy)
    assert worst_value <= value_bound, (worst_value, floor_value)

    final_full = forward(
        window.replace(history_carry=invalid_history_carry(width)), request_count
    )
    np.testing.assert_allclose(
        np.asarray(carry.slot_states),
        np.asarray(final_full.history_carry.slot_states),
        atol=0.05,
    )
    np.testing.assert_allclose(
        np.asarray(carry.field_states),
        np.asarray(final_full.history_carry.field_states),
        atol=0.05,
    )
    np.testing.assert_allclose(
        np.asarray(carry.node_snapshots, np.float32),
        np.asarray(final_full.history_carry.node_snapshots, np.float32),
        atol=0.05,
    )

    # Control: the tolerance can fail -- a shifted carry moves what the
    # test compares.
    shifted = carry.replace(slot_states=np.asarray(carry.slot_states) + 1.0)
    moved = forward(suffix.replace(history_carry=shifted), request_count)
    assert policy_diff(moved, resumed) > policy_bound, (
        policy_diff(moved, resumed),
        policy_bound,
    )


def test_server_mixed_group_matches_single_forwards(
    real_model_and_trajectory, carry_params
):
    """Test (f): one carrying and one non-carrying request in the same
    inference-server group each equal their own single forward within
    bf16 -- the non-carrying one filled with an invalid carry, the
    carrying one resumed."""
    import threading

    from rl.environment.utils import clip_history_suffix
    from rl.model.history_encoder import invalid_history_carry
    from rl.model.utils import ParamsContainer
    from rl.online.inference import InferenceServer, _InferenceRequest
    from rl.online.player_actor import _last_step_index

    network, _, full_window, full_output = real_model_and_trajectory
    params = carry_params
    width = _width(params)
    # The session model is train=True, so it teacher-forces the stored
    # action rather than sampling one (the actor network emits no
    # log_policy to compare on). The server's apply signature is kept; the
    # placeholder output it passes is swapped for one stored row -- the
    # given index only picks which log_prob is reported.
    taken = jax.tree.map(lambda x: np.asarray(x)[21:22], full_output)

    def teacher_forced_apply(params, actor_input, _placeholder, head_params, rngs):
        return network.apply(
            params, actor_input, taken, head_params=head_params, rngs=rngs
        )

    server = InferenceServer(player_apply_fn=teacher_forced_apply)
    container = ParamsContainer(
        step_count=0,
        player_frame_count=0,
        builder_frame_count=0,
        player_params=jax.device_get(params),
        builder_params=None,
    )

    def request(rng_seed, actor_input):
        # A server request's env has no T axis (_run_group adds it);
        # _window_at keeps the [T=1] slice the jitted apply expects.
        return _InferenceRequest(
            container=container,
            rng_key=jax.random.key(rng_seed),
            actor_input=actor_input.replace(
                env=jax.tree.map(lambda x: np.asarray(x)[0], actor_input.env)
            ),
            done=threading.Event(),
        )

    def run(requests):
        server._run_group(requests)
        return [r.output.actor_output for r in requests]

    before = _window_at(full_window, 20)
    window = _window_at(full_window, 21)
    (primed,) = run(
        [request(0, before.replace(history_carry=invalid_history_carry(width)))]
    )
    suffix, suffix_steps = clip_history_suffix(window, _last_step_index(before))
    assert suffix_steps > 0
    carrying = suffix.replace(history_carry=primed.history_carry)
    # The same window with no carry at all -- `()` leaves, as a caller that
    # never carries sends -- so the two share a bucket level, as a real
    # group does by construction (grouping key), and differ ONLY in the
    # carry.
    plain = suffix

    (single_carrying,) = run([request(1, carrying)])
    (single_plain,) = run([request(2, plain)])
    mixed_carrying, mixed_plain = run([request(1, carrying), request(2, plain)])
    # Batch 1 vs batch 2 is a bf16 GEMM leading-dim change, and the noise
    # it makes is content-dependent (0.042 on the plain request, 0.067 on
    # the carrying one, 2026-09-02, opened readout), so the read is
    # structural: each grouped output sits within the shape-noise class
    # of ITS OWN single forward and far from the other's -- a dropped or
    # misrouted carry lands ~2.0 from its single (the plain/carrying
    # separation) and would fail both halves.
    shape_noise = 0.15
    for single, other, mixed in (
        (single_carrying, single_plain, mixed_carrying),
        (single_plain, single_carrying, mixed_plain),
    ):
        assert _max_diff(single, mixed) <= shape_noise, _max_diff(single, mixed)
        assert _max_diff(other, mixed) > 1.0, _max_diff(other, mixed)
    # The fill is the from-scratch forward: the plain request equals the
    # same window sent with an explicit invalid carry ...
    (explicit_invalid,) = run(
        [request(2, suffix.replace(history_carry=invalid_history_carry(width)))]
    )
    np.testing.assert_allclose(
        np.asarray(single_plain.action_head.log_policy, np.float32),
        np.asarray(explicit_invalid.action_head.log_policy, np.float32),
        atol=0.05,
    )
    # ... and the carry is what separates the two requests (control).
    assert _max_diff(single_plain, single_carrying) > 0.05
