"""DeepNash-style thresholding of a policy (rl/model/utils.py
prune_log_policy): the one definition the thresholded v-trace ratio and
the uniform-KL screen read."""

import jax
import jax.numpy as jnp
import numpy as np

from rl.model.utils import legal_log_policy, prune_log_policy, renormalise_kept

DTYPE_MIN = jnp.finfo(jnp.float32).min


def _row(
    probabilities: list[float] | list[list[float]] | np.ndarray,
    legal: np.ndarray | None = None,
) -> tuple[jax.Array, jax.Array]:
    probabilities = np.asarray(probabilities, np.float32)
    if legal is None:
        legal = probabilities > 0
    logits = np.where(legal, np.log(np.maximum(probabilities, 1e-30)), 0.0)
    return legal_log_policy(jnp.asarray(logits), jnp.asarray(legal)), jnp.asarray(legal)


def test_threshold_zero_is_the_sampling_form_bit_identical() -> None:
    log_policy, legal = _row([0.5, 0.3, 0.0, 0.2])
    pruned = prune_log_policy(log_policy, legal, 0.0)
    expected = jnp.where(legal, log_policy, DTYPE_MIN)
    np.testing.assert_array_equal(np.asarray(pruned), np.asarray(expected))


def test_below_the_line_is_removed_and_the_rest_renormalised() -> None:
    log_policy, legal = _row([0.6, 0.004, 0.0, 0.396])
    pruned = np.asarray(prune_log_policy(log_policy, legal, 0.005))
    assert pruned[1] == DTYPE_MIN and pruned[2] == DTYPE_MIN
    kept = np.exp(pruned[[0, 3]])
    np.testing.assert_allclose(kept, [0.6 / 0.996, 0.396 / 0.996], rtol=1e-6)
    np.testing.assert_allclose(kept.sum(), 1.0, rtol=1e-6)


def test_above_the_line_is_untouched() -> None:
    # Positive control for the previous test: the same row with the small
    # cell lifted just past the threshold keeps every legal cell as-is.
    log_policy, legal = _row([0.6, 0.006, 0.0, 0.394])
    pruned = np.asarray(prune_log_policy(log_policy, legal, 0.005))
    np.testing.assert_array_equal(
        pruned, np.asarray(jnp.where(legal, log_policy, DTYPE_MIN))
    )


def test_a_row_entirely_below_the_line_keeps_its_legal_set() -> None:
    # The degenerate guard of DeepNash's FineTuning._threshold — see the
    # reference note in rl/online/training/targets.py.
    probabilities = np.full(300, 1 / 300, np.float32)
    log_policy, legal = _row(probabilities)
    pruned = np.asarray(prune_log_policy(log_policy, legal, 0.005))
    np.testing.assert_array_equal(
        pruned, np.asarray(jnp.where(legal, log_policy, DTYPE_MIN))
    )


def test_batched_rows_are_independent() -> None:
    log_policy, legal = _row([[0.6, 0.004, 0.0, 0.396], [0.25, 0.25, 0.25, 0.25]])
    pruned = np.asarray(prune_log_policy(log_policy, legal, 0.005))
    assert pruned[0, 1] == DTYPE_MIN
    np.testing.assert_allclose(np.exp(pruned[1]), 0.25, rtol=1e-6)


def test_removed_cell_has_zero_gradient_through_its_own_logit() -> None:
    legal = jnp.asarray([True, True, False, True])
    logits = jnp.log(jnp.asarray([0.6, 0.004, 1.0, 0.396]))

    def kept_log_prob(logits: jax.Array) -> jax.Array:
        return prune_log_policy(legal_log_policy(logits, legal), legal, 0.005)[0]

    gradient = np.asarray(jax.grad(kept_log_prob)(logits))
    # The removed cell's gradient is two cancelling log-sum-exp terms, zero
    # to float32 rounding; the illegal cell's exactly.
    assert abs(gradient[1]) < 1e-6 and gradient[2] == 0.0
    assert abs(gradient[0]) > 0.1 and abs(gradient[3]) > 0.1


def test_renormalise_kept_takes_any_kept_set() -> None:
    log_policy, legal = _row([0.6, 0.004, 0.0, 0.396])
    by_threshold = np.asarray(prune_log_policy(log_policy, legal, 0.005))
    same_set = jnp.asarray([True, False, False, True])
    np.testing.assert_array_equal(
        np.asarray(renormalise_kept(log_policy, legal, same_set)), by_threshold
    )
    # The control: a kept set no threshold can produce (the MOST likely cell
    # removed) gives a different distribution, renormalised over what is left.
    without_the_mode = jnp.asarray([False, True, False, True])
    restricted = np.asarray(renormalise_kept(log_policy, legal, without_the_mode))
    assert restricted[0] == DTYPE_MIN
    np.testing.assert_allclose(
        np.exp(restricted[[1, 3]]), [0.004 / 0.4, 0.396 / 0.4], rtol=1e-5
    )
    assert not np.array_equal(restricted, by_threshold)
