"""Step-1 critic telemetry (docs/critic-weakness-analysis.md): the
completed-game side fields ride chunks through stack_batch, and the
pure panel function behaves on hand-built rows — NaN on empty slices,
row 0 excluded from the previous-action split, matched-V counts that
add up."""

import numpy as np
import jax.numpy as jnp
import pytest

from rl.environment.data import FLAT_MODALITY_MASK
from rl.environment.interfaces import Trajectory
from rl.environment.protos.service_pb2 import ModalityEnum
from rl.online.training.batching import _or_empty
from rl.online.training.telemetry import (
    MATCHED_V_EDGES,
    critic_outcome_telemetry,
    masked_mean,
    masked_var,
)

A = int(np.asarray(FLAT_MODALITY_MASK).shape[0])
SWITCH = int(np.argmax(np.asarray(FLAT_MODALITY_MASK) == ModalityEnum.MODALITY_ENUM__SWITCH))
MOVE = int(np.argmax(np.asarray(FLAT_MODALITY_MASK) == ModalityEnum.MODALITY_ENUM__MOVE))


def test_trajectory_side_fields_default_empty_and_or_empty_keeps_sentinel():
    traj = Trajectory()
    assert traj.game_outcome == () and traj.game_length == () and traj.game_step_offset == ()
    assert _or_empty(()) == ()
    arr = np.ones((1, 3), np.float32)
    assert _or_empty(arr) is arr


def test_masked_helpers_nan_on_empty():
    x = jnp.arange(4.0)
    assert jnp.isnan(masked_mean(x, jnp.zeros(4, bool)))
    assert jnp.isnan(masked_var(x, jnp.array([True, False, False, False])))
    assert float(masked_mean(x, jnp.array([True, True, False, False]))) == 0.5


def _rows(T=4, B=2):
    """Chunk 0: voluntary switch at row 1 (both legal), moves elsewhere.
    Chunk 1: forced switch at row 0 (no legal move), moves after."""
    flat = np.zeros((T, B, A), bool)
    flat[..., MOVE] = True
    flat[..., SWITCH] = True
    flat[0, 1, MOVE] = False  # forced switch row
    action = np.full((T, B), MOVE, np.int32)
    action[1, 0] = SWITCH
    action[0, 1] = SWITCH
    q_mask = np.ones((T, B), bool)
    q_mask[-1] = False  # bootstrap-only final row
    v = np.array([[-0.9, 0.5], [-0.1, 0.5], [0.3, 0.7], [0.4, 0.9]], np.float32)
    return flat, action, q_mask, v


def test_critic_outcome_telemetry_counts_and_splits():
    T, B = 4, 2
    flat, action, q_mask, v = _rows(T, B)
    q_all = np.zeros((T, B, A), np.float32)
    q_all[..., SWITCH] = -0.1  # a flat switch offset, as on the collapsed run
    logs = critic_outcome_telemetry(
        game_outcome=jnp.array([[1.0, -1.0]]),
        game_length=jnp.array([[9, 9]]),
        game_step_offset=jnp.array([[0, 6]]),
        v_target=jnp.asarray(v),
        onestep_label=jnp.asarray(v) * 0.5,
        retrace_g=jnp.asarray(v),
        q_taken=jnp.zeros((T, B)),
        q_all=jnp.asarray(q_all),
        flat_action_mask=jnp.asarray(flat),
        action_index=jnp.asarray(action),
        q_mask=jnp.asarray(q_mask),
        value_mask=jnp.ones((T, B), bool),
    )
    logs = {k: float(v) for k, v in logs.items()}
    # support: one voluntary switch row (chunk 0), one forced (chunk 1)
    assert logs["player_q_support_vol_switch_rows"] == 1.0
    assert logs["player_q_support_forced_switch_rows"] == 1.0
    assert logs["player_q_support_chunk_vol_switch_frac"] == 0.5
    # matched-V: every real-choice masked row lands in exactly one bin
    n_vol = sum(logs[f"player_mv_bin{i}_n_vol"] for i in range(len(MATCHED_V_EDGES) - 1))
    n_mv = sum(logs[f"player_mv_bin{i}_n_move"] for i in range(len(MATCHED_V_EDGES) - 1))
    assert n_vol == 1.0
    assert n_mv == 4.0  # chunk 0 rows 0,2 + chunk 1 rows 1,2 (row 0 there has no legal move)
    # the critic's gap is the flat offset on every populated bin
    for i in range(len(MATCHED_V_EDGES) - 1):
        if logs[f"player_mv_bin{i}_n_vol"] + logs[f"player_mv_bin{i}_n_move"] > 0:
            assert logs[f"player_mv_bin{i}_gap_critic"] == pytest.approx(-0.1, abs=1e-6)
        else:
            assert np.isnan(logs[f"player_mv_bin{i}_gap_critic"])
    assert logs["player_mv_pooled_gap_critic"] == pytest.approx(-0.1, abs=1e-6)
    # the voluntary switch is in the won game (+1); the four real-choice
    # move rows split 2 won / 2 lost (mean 0) -> realised gap +1
    assert logs["player_mv_pooled_gap_realised"] == pytest.approx(1.0)
    # phase split: chunk 0 rows 0-2 are early (0..2 of 9), chunk 1 rows are
    # late (6..9 of 9). Each phase holds ONE game's rows here, so its
    # outcome is constant and R² is NaN by the constant-target guard; the
    # all-rows R² spans both outcomes and is finite.
    assert np.isnan(logs["player_v_outcome_r2_early"])
    assert np.isnan(logs["player_v_outcome_r2_late"])
    assert not np.isnan(logs["player_v_outcome_r2_all"])
    # previous-action split excludes row 0; the row after the voluntary
    # switch (chunk 0 row 2) is the only prev_switch row besides chunk 1
    # row 1 (after the forced switch) -> 2 rows, enough for a finite value
    assert not np.isnan(logs["player_v_outcome_bias_prev_switch"])
    # label variance on the empty-ish voluntary slice is NaN, not 0
    assert np.isnan(logs["player_q_label_var_outcome_voluntary"])
    assert not np.isnan(logs["player_q_label_var_onestep_move"])


def test_truncated_game_outcome_nan_drops_outcome_panels_only():
    T, B = 4, 2
    flat, action, q_mask, v = _rows(T, B)
    logs = critic_outcome_telemetry(
        game_outcome=jnp.array([[jnp.nan, jnp.nan]]),
        game_length=jnp.array([[9, 9]]),
        game_step_offset=jnp.array([[0, 6]]),
        v_target=jnp.asarray(v),
        onestep_label=jnp.asarray(v),
        retrace_g=jnp.asarray(v),
        q_taken=jnp.zeros((T, B)),
        q_all=jnp.zeros((T, B, A)),
        flat_action_mask=jnp.asarray(flat),
        action_index=jnp.asarray(action),
        q_mask=jnp.asarray(q_mask),
        value_mask=jnp.ones((T, B), bool),
    )
    assert np.isnan(float(logs["player_v_outcome_r2_all"]))
    assert np.isnan(float(logs["player_mv_pooled_gap_realised"]))
    # one-step panels need no outcome
    assert not np.isnan(float(logs["player_v_onestep_r2"]))
    assert float(logs["player_q_support_vol_switch_rows"]) == 1.0
