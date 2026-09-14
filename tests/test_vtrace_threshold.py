"""Pruned-policy ratios retained for offline comparisons only."""

import jax
import jax.numpy as jnp
import numpy as np

from rl.model.utils import legal_log_policy
from rl.online.training.targets import (
    thresholded_target_ratio,
    trace_run_length,
)

CELLS = 6


def _rows() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # Three rows, four legal cells each; the taken action's target
    # probability is .40 / .004 / .40 and mu(a) = .25 everywhere.
    legal = jnp.asarray([[True] * 4 + [False] * 2] * 3)
    logits = jnp.log(
        jnp.asarray(
            [
                [0.40, 0.30, 0.20, 0.10, 1.0, 1.0],
                [0.60, 0.004, 0.30, 0.096, 1.0, 1.0],
                [0.40, 0.30, 0.20, 0.10, 1.0, 1.0],
            ]
        )
    )
    target_log_policy = legal_log_policy(logits, legal)
    action_index = jnp.asarray([0, 1, 0])
    behaviour_log_prob = jnp.full((3,), jnp.log(0.25))
    return target_log_policy, behaviour_log_prob, action_index, legal


def test_threshold_zero_is_the_raw_ratio_bit_identical() -> None:
    target_log_policy, behaviour, taken, legal = _rows()
    ratio, ratio_raw, kept, removed = thresholded_target_ratio(
        target_log_policy, behaviour, taken, legal, 0.0
    )
    np.testing.assert_array_equal(np.asarray(ratio), np.asarray(ratio_raw))
    expected = np.exp(np.asarray(target_log_policy)[np.arange(3), np.asarray(taken)])
    np.testing.assert_allclose(np.asarray(ratio_raw), expected / 0.25, rtol=1e-6)
    assert np.all(np.asarray(kept)) and not np.any(np.asarray(removed))


def test_below_the_line_is_discarded_and_above_untouched() -> None:
    target_log_policy, behaviour, taken, legal = _rows()
    ratio, ratio_raw, kept, removed = thresholded_target_ratio(
        target_log_policy, behaviour, taken, legal, 0.005
    )
    ratio, ratio_raw, kept = (np.asarray(x) for x in (ratio, ratio_raw, kept))
    # Row 1's taken action (.004) is discarded: ratio 0, raw ratio intact.
    assert ratio[1] == 0.0 and not kept[1]
    np.testing.assert_allclose(ratio_raw[1], 0.004 / 0.25, rtol=1e-6)
    # Rows 0 and 2 (positive control) are untouched -- nothing removed
    # there, so no renormalisation either.
    np.testing.assert_array_equal(ratio[[0, 2]], ratio_raw[[0, 2]])
    assert kept[0] and kept[2]
    # Row 1's kept actions were renormalised over the .996 that remains.
    np.testing.assert_allclose(np.asarray(removed), [0.0, 0.25, 0.0])


def test_a_row_entirely_below_the_line_keeps_its_ratio() -> None:
    # The reference's degenerate guard: every legal cell under the line.
    legal = jnp.asarray([[True] * 4 + [False] * 2])
    target_log_policy = legal_log_policy(jnp.zeros((1, CELLS)), legal)  # .25 each
    ratio, ratio_raw, kept, _ = thresholded_target_ratio(
        target_log_policy, jnp.log(jnp.asarray([0.25])), jnp.asarray([2]), legal, 0.3
    )
    np.testing.assert_array_equal(np.asarray(ratio), np.asarray(ratio_raw))
    assert bool(kept[0])


def test_trace_run_length_counts_to_the_first_cut() -> None:
    continues = jnp.asarray([[True], [True], [False], [True], [True]])
    runs = np.asarray(trace_run_length(continues))[:, 0]
    np.testing.assert_array_equal(runs, [2.0, 1.0, 0.0, 2.0, 1.0])
