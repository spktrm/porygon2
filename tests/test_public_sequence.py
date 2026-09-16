"""The public-only sequence (PUBLIC_SEQUENCE_ROWS) and the per-event path:
the trunk on the 55 public rows alone reproduces the learner's public rows,
and `Encoder.encode_events` assembles every history step's public state
from the scan's own products through the same assembly the request path
uses."""

import jax
import jax.numpy as jnp
import numpy as np
from test_privileged_partition import READ_MASK, WIDTH, _trunk_cfg

from rl.environment.protos.features_pb2 import InfoFeature
from rl.model.constants import (
    HISTORY_ENTITY_ROWS,
    NUM_PUBLIC_SLOTS,
    NUM_SEQUENCE_ROWS,
    PUBLIC_CLS_LOCAL_ROW,
    PUBLIC_CLS_ROW,
    PUBLIC_ROWS,
    PUBLIC_SEQUENCE_ROWS,
    PUBLIC_TIER_ROWS,
    SEQUENCE_READ_MASK,
)
from rl.model.trunk import Trunk


def test_public_sequence_rows_are_the_layout_prefix_plus_public_cls() -> None:
    assert PUBLIC_SEQUENCE_ROWS.tolist() == list(range(len(PUBLIC_TIER_ROWS))) + [
        PUBLIC_CLS_ROW
    ]
    assert PUBLIC_CLS_LOCAL_ROW == len(PUBLIC_TIER_ROWS)
    sub_mask = SEQUENCE_READ_MASK[np.ix_(PUBLIC_SEQUENCE_ROWS, PUBLIC_SEQUENCE_ROWS)]
    assert sub_mask[PUBLIC_CLS_LOCAL_ROW].all()
    assert not sub_mask[:PUBLIC_CLS_LOCAL_ROW, PUBLIC_CLS_LOCAL_ROW].any()
    assert sub_mask[:PUBLIC_CLS_LOCAL_ROW, :PUBLIC_CLS_LOCAL_ROW].all()


def test_trunk_on_the_public_sequence_reproduces_the_full_public_rows() -> None:
    trunk = Trunk(_trunk_cfg())
    full = jax.random.normal(jax.random.key(7), (NUM_SEQUENCE_ROWS, WIDTH))
    valid = jnp.ones(NUM_SEQUENCE_ROWS, bool)
    params = trunk.init(jax.random.key(8), full, valid, READ_MASK)
    sub_mask = READ_MASK[np.ix_(PUBLIC_SEQUENCE_ROWS, PUBLIC_SEQUENCE_ROWS)]

    def both(sequence):
        full_out = trunk.apply(params, sequence, valid, READ_MASK)
        sub_out = trunk.apply(
            params,
            sequence[PUBLIC_SEQUENCE_ROWS],
            valid[PUBLIC_SEQUENCE_ROWS],
            sub_mask,
        )
        return np.asarray(full_out, np.float32), np.asarray(sub_out, np.float32)

    full_out, sub_out = both(full)
    np.testing.assert_allclose(
        sub_out, full_out[PUBLIC_SEQUENCE_ROWS], rtol=1e-3, atol=1e-3
    )
    # Control: a public-row perturbation moves BOTH by the same amount --
    # the agreement is the closed read mask, not a dead trunk.
    full_moved, sub_moved = both(full.at[PUBLIC_TIER_ROWS[0]].add(10.0))
    assert not np.allclose(sub_moved, sub_out)
    np.testing.assert_allclose(
        sub_moved, full_moved[PUBLIC_SEQUENCE_ROWS], rtol=1e-3, atol=1e-3
    )


def test_encode_events_matches_the_request_path_where_the_snapshot_is_current(
    real_model_and_trajectory,
) -> None:
    model, params, actor_input, _ = real_model_and_trajectory
    packed = actor_input.packed_history
    history = actor_input.history
    env = actor_input.env

    def events(module, packed, history):
        return module.encoder.encode_events(packed, history)

    def request_inputs(module, env, packed, history):
        return module.encoder.assembled_sequence(env, packed, history)

    with jax.default_matmul_precision("highest"):
        out = jax.jit(lambda p, a, b: model.apply(p, a, b, method=events))(
            params, packed, history
        )
        assembled, _ = jax.jit(
            lambda p, e, a, b: model.apply(p, e, a, b, method=request_inputs)
        )(params, env, packed, history)

    states = np.asarray(out.states, np.float32)
    inputs = np.asarray(out.inputs, np.float32)
    step_valid = np.asarray(out.step_valid)
    assert states.shape[1:] == (len(PUBLIC_SEQUENCE_ROWS), states.shape[-1])
    assert np.isfinite(states[step_valid]).all()
    row_valid = np.asarray(out.row_valid)
    np.testing.assert_array_equal(
        row_valid[:, :NUM_PUBLIC_SLOTS], np.asarray(out.slot_valid)
    )
    assert row_valid[step_valid, PUBLIC_CLS_LOCAL_ROW].all()

    assembled = np.asarray(assembled, np.float32)
    info = np.asarray(env.info)
    public_team = np.asarray(env.public_team)
    public_cache = np.asarray(packed.public_cache)
    request_count = np.asarray(out.step_request_count)
    row_index = np.asarray(out.node_row_index)
    steps = np.arange(step_valid.shape[0])
    compared = 0
    first_pair = None
    for t in range(info.shape[0]):
        ok = step_valid & (
            request_count <= info[t, InfoFeature.INFO_FEATURE__REQUEST_COUNT]
        )
        if not ok.any():
            continue
        step = int(np.where(ok, steps, -1).max())
        order = info[
            t,
            InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0 : InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11
            + 1,
        ]
        for row, slot in enumerate(order):
            if slot < 0 or slot >= NUM_PUBLIC_SLOTS or row_index[step, slot] < 0:
                continue
            # The cache snapshots a mon on its first touch inside an edge, so
            # only rows whose raw features agree are expected to agree.
            if not np.array_equal(
                public_cache[row_index[step, slot]], public_team[t, row]
            ):
                continue
            np.testing.assert_allclose(
                inputs[step, PUBLIC_ROWS.start + slot],
                assembled[t, PUBLIC_ROWS.start + row],
                rtol=1e-3,
                atol=1e-3,
            )
            np.testing.assert_allclose(
                inputs[step, HISTORY_ENTITY_ROWS.start + slot],
                assembled[t, HISTORY_ENTITY_ROWS.start + row],
                rtol=1e-3,
                atol=1e-3,
            )
            compared += 1
            if first_pair is None:
                first_pair = (step, slot, t, row)
    assert compared > 0
    # Control: the parity is row-specific -- another revealed slot's row at
    # the same step does not reproduce this request row.
    step, slot, t, row = first_pair
    others = [
        other
        for other in range(NUM_PUBLIC_SLOTS)
        if other != slot and row_index[step, other] >= 0
    ]
    assert others
    assert not np.allclose(
        inputs[step, PUBLIC_ROWS.start + others[0]],
        assembled[t, PUBLIC_ROWS.start + row],
        rtol=1e-3,
        atol=1e-3,
    )
