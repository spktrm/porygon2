"""Fresh decision counts approximate, but do not replace, protocol-log rates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.environment.data import CELL_MODALITY_MASK, NUM_ACTION_CELLS
from rl.environment.interfaces import Batch, PlayerEnvOutput, PlayerTransition
from rl.environment.protos.features_pb2 import InfoFeature, RequestType
from rl.environment.protos.service_pb2 import ModalityEnum
from rl.online.training.action_telemetry import voluntary_switch_telemetry
from rl.online.training.telemetry import action_axis_masks

SWITCH_CELL = int(
    np.flatnonzero(CELL_MODALITY_MASK == ModalityEnum.MODALITY_ENUM__SWITCH)[0]
)
MOVE_CELL = int(
    np.flatnonzero(CELL_MODALITY_MASK == ModalityEnum.MODALITY_ENUM__MOVE)[0]
)
WILDCARD_CELL = int(
    np.flatnonzero(CELL_MODALITY_MASK == ModalityEnum.MODALITY_ENUM__WILDCARD)[0]
)
OTHER_CELL = int(
    np.flatnonzero(CELL_MODALITY_MASK == ModalityEnum.MODALITY_ENUM__OTHER)[0]
)
REQUEST_COLUMN = InfoFeature.INFO_FEATURE__REQUEST_TYPE
SWITCH_COUNT = "player_fresh_voluntary_switch_count"
DECISION_COUNT = "player_fresh_move_or_switch_count"
FRACTION = "player_fresh_voluntary_switch_frac"


def make_inputs(actions, *, reuse_count=(0,)):
    actions = np.asarray(actions, dtype=np.int32)
    if actions.ndim == 1:
        actions = actions[:, None]
    info = np.zeros((*actions.shape, len(InfoFeature.keys())), dtype=np.int32)
    info[..., REQUEST_COLUMN] = RequestType.REQUEST_TYPE__MOVE
    legal = np.zeros((*actions.shape, NUM_ACTION_CELLS), dtype=bool)
    legal[..., [SWITCH_CELL, MOVE_CELL, WILDCARD_CELL, OTHER_CELL]] = True
    batch = Batch(
        player_transitions=PlayerTransition(
            env_output=PlayerEnvOutput(
                info=jnp.asarray(info), action_mask=jnp.asarray(legal)
            )
        ),
        reuse_count=jnp.asarray([reuse_count], dtype=jnp.int32),
    )
    return batch, jnp.asarray(actions), jnp.ones(actions.shape, dtype=bool)


def measure(batch, actions, acted_mask):
    axis = action_axis_masks(batch.player_transitions.env_output.action_mask, actions)
    return jax.jit(voluntary_switch_telemetry)(batch, axis, acted_mask)


def with_env(batch, **fields):
    env = batch.player_transitions.env_output.replace(**fields)
    return batch.replace(
        player_transitions=batch.player_transitions.replace(env_output=env)
    )


def test_fresh_counts_exclude_reuse_and_nondecisions():
    taken = [
        SWITCH_CELL,
        MOVE_CELL,
        WILDCARD_CELL,
        OTHER_CELL,
        SWITCH_CELL,
        SWITCH_CELL,
    ]
    batch, actions, acted = make_inputs(np.tile(taken, (2, 1)).T, reuse_count=(0, 3))
    acted = acted.at[-2:].set(False)
    logs = measure(batch, actions, acted)
    assert int(logs[SWITCH_COUNT]) == 1
    assert int(logs[DECISION_COUNT]) == 3
    assert float(logs[FRACTION]) == pytest.approx(1 / 3)

    fresh_batch = batch.replace(reuse_count=jnp.zeros((1, 2), dtype=jnp.int32))
    fresh = measure(fresh_batch, actions, acted)
    assert int(fresh[SWITCH_COUNT]) == 2
    assert int(fresh[DECISION_COUNT]) == 6


@pytest.mark.parametrize(
    "request_type",
    [
        RequestType.REQUEST_TYPE__SWITCH,
        RequestType.REQUEST_TYPE__TEAM,
        RequestType.REQUEST_TYPE___UNSPECIFIED,
    ],
)
def test_forced_switch_preview_and_unspecified_are_excluded(request_type):
    batch, actions, acted = make_inputs([SWITCH_CELL])
    baseline = measure(batch, actions, acted)
    assert int(baseline[SWITCH_COUNT]) == 1
    env = batch.player_transitions.env_output
    changed = with_env(batch, info=env.info.at[..., REQUEST_COLUMN].set(request_type))
    logs = measure(changed, actions, acted)
    assert int(logs[SWITCH_COUNT]) == 0
    assert int(logs[DECISION_COUNT]) == 0
    assert np.isnan(logs[FRACTION])


def test_wait_sentinel_is_excluded_even_with_move_request_token():
    batch, actions, acted = make_inputs([SWITCH_CELL])
    assert int(measure(batch, actions, acted)[DECISION_COUNT]) == 1
    env = batch.player_transitions.env_output
    waiting = with_env(batch, action_mask=jnp.ones_like(env.action_mask))
    logs = measure(waiting, actions, acted)
    assert int(logs[DECISION_COUNT]) == 0
    assert np.isnan(logs[FRACTION])


def test_singleton_moves_without_switch_options_remain_in_denominator():
    batch, actions, acted = make_inputs([SWITCH_CELL, MOVE_CELL, WILDCARD_CELL])
    legal = jax.nn.one_hot(actions, NUM_ACTION_CELLS, dtype=jnp.bool_)
    batch = with_env(batch, action_mask=legal)
    logs = measure(batch, actions, acted)
    assert int(logs[SWITCH_COUNT]) == 1
    assert int(logs[DECISION_COUNT]) == 3
    assert float(logs[FRACTION]) == pytest.approx(1 / 3)


def test_missing_reuse_provenance_does_not_claim_fresh_decisions():
    batch, actions, acted = make_inputs([SWITCH_CELL, MOVE_CELL])
    assert int(measure(batch, actions, acted)[DECISION_COUNT]) == 2
    logs = measure(batch.replace(reuse_count=()), actions, acted)
    assert int(logs[SWITCH_COUNT]) == 0
    assert int(logs[DECISION_COUNT]) == 0
    assert np.isnan(logs[FRACTION])


def test_terminal_bootstrap_and_padding_masks_are_respected():
    batch, actions, acted = make_inputs([SWITCH_CELL, SWITCH_CELL, SWITCH_CELL])
    logs = measure(batch, actions, jnp.zeros_like(acted))
    assert int(logs[DECISION_COUNT]) == 0
    assert np.isnan(logs[FRACTION])
    live = measure(batch, actions, acted.at[1:].set(False))
    assert int(live[SWITCH_COUNT]) == 1
    assert int(live[DECISION_COUNT]) == 1


def test_counters_pool_by_decisions_across_unequal_batches():
    first = measure(*make_inputs([SWITCH_CELL]))
    second = measure(*make_inputs([MOVE_CELL, WILDCARD_CELL, MOVE_CELL]))
    pooled = (first[SWITCH_COUNT] + second[SWITCH_COUNT]) / (
        first[DECISION_COUNT] + second[DECISION_COUNT]
    )
    combined = measure(*make_inputs([SWITCH_CELL, MOVE_CELL, WILDCARD_CELL, MOVE_CELL]))
    assert float(pooled) == pytest.approx(0.25)
    assert float(combined[FRACTION]) == pytest.approx(float(pooled))
    assert float((first[FRACTION] + second[FRACTION]) / 2) != float(pooled)
