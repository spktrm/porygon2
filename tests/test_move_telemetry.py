"""Flat support telemetry against hand-computable rows."""

import jax
import jax.numpy as jnp
import numpy as np

from rl.environment.data import MOVE_CELL_OFFSET, NUM_ACTION_CELLS, NUM_SWITCH_CELLS
from rl.online.training.action_telemetry import legal_support_telemetry

SWITCH_CELLS = jnp.arange(NUM_ACTION_CELLS) < NUM_SWITCH_CELLS


def _rows():
    # Row 0: two switches and two moves, one move at 1e-4. Row 1: forced
    # switch (one legal cell). Row 2: masked out of the policy rows, with
    # an absurd distribution that must not leak into any readout.
    legal = jnp.zeros((3, NUM_ACTION_CELLS), dtype=bool)
    legal = legal.at[0, [0, 1, MOVE_CELL_OFFSET, MOVE_CELL_OFFSET + 1]].set(True)
    legal = legal.at[1, 0].set(True)
    legal = legal.at[2, :].set(True)
    probabilities = jnp.zeros((3, NUM_ACTION_CELLS), dtype=jnp.float32)
    probabilities = probabilities.at[0, [0, 1]].set([0.3, 0.2])
    probabilities = probabilities.at[0, MOVE_CELL_OFFSET].set(0.4999)
    probabilities = probabilities.at[0, MOVE_CELL_OFFSET + 1].set(1e-4)
    probabilities = probabilities.at[1, 0].set(1.0)
    probabilities = probabilities.at[2, :].set(1.0 / NUM_ACTION_CELLS)
    log_policy = jnp.where(legal, jnp.log(jnp.maximum(probabilities, 1e-30)), 0.0)
    policy_mask = jnp.array([True, True, False])
    return log_policy, legal, policy_mask


def test_readouts_match_hand_values():
    log_policy, legal, policy_mask = _rows()
    logs = legal_support_telemetry(log_policy, legal, SWITCH_CELLS, policy_mask)
    logs = {key: float(value) for key, value in logs.items()}
    # Row 0 min 1e-4, row 1 min 1.0 -> mean over the two policy rows.
    np.testing.assert_allclose(logs["player_support_min_prob"], (1e-4 + 1.0) / 2)
    # Row 0 median of [1e-4, .2, .3, .4999] at index (4-1)//2 = 1 -> .2;
    # row 1 median 1.0.
    np.testing.assert_allclose(logs["player_support_median_prob"], (0.2 + 1.0) / 2)
    np.testing.assert_allclose(logs["player_support_legal_count"], (4 + 1) / 2)
    # Row 0: one of four cells is below every line; row 1: none.
    for name in ("p01", "p005", "p001"):
        np.testing.assert_allclose(
            logs[f"player_support_frac_below_{name}"], (0.25 + 0.0) / 2
        )
    # Switch split: row 0 min .2, row 1 min 1.0; nothing below any line.
    np.testing.assert_allclose(logs["player_support_switch_min_prob"], (0.2 + 1.0) / 2)
    np.testing.assert_allclose(logs["player_support_switch_frac_below_p01"], 0.0)
    # Move split: only row 0 has a move cell -- min 1e-4, half its two
    # move cells below the line (the positive control for the split).
    np.testing.assert_allclose(logs["player_support_move_min_prob"], 1e-4)
    np.testing.assert_allclose(logs["player_support_move_frac_below_p001"], 0.5)


def test_permutation_invariant_over_cells():
    log_policy, legal, policy_mask = _rows()
    permutation = jax.random.permutation(jax.random.key(3), NUM_ACTION_CELLS)
    reference = legal_support_telemetry(log_policy, legal, SWITCH_CELLS, policy_mask)
    permuted = legal_support_telemetry(
        log_policy[:, permutation],
        legal[:, permutation],
        SWITCH_CELLS[permutation],
        policy_mask,
    )
    for key, value in reference.items():
        np.testing.assert_allclose(float(permuted[key]), float(value), rtol=1e-6)


def test_lifting_the_abandoned_cell_clears_the_lines():
    # Positive control: the same rows with the 1e-4 cell raised to .011 read
    # nothing below any line.
    log_policy, legal, policy_mask = _rows()
    log_policy = log_policy.at[0, MOVE_CELL_OFFSET + 1].set(jnp.log(0.011))
    logs = legal_support_telemetry(log_policy, legal, SWITCH_CELLS, policy_mask)
    for name in ("p01", "p005", "p001"):
        assert float(logs[f"player_support_frac_below_{name}"]) == 0.0
    np.testing.assert_allclose(float(logs["player_support_move_min_prob"]), 0.011)
