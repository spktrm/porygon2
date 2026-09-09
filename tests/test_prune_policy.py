"""DeepNash-style thresholding of a sampled policy (rl/model/utils.py
prune_log_policy): the one definition both the `thresholded` eval slot
and the learner's v-trace ratio read."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.model.utils import legal_log_policy, prune_log_policy

DTYPE_MIN = jnp.finfo(jnp.float32).min


def _row(probabilities, legal=None):
    probabilities = np.asarray(probabilities, np.float32)
    if legal is None:
        legal = probabilities > 0
    logits = np.where(legal, np.log(np.maximum(probabilities, 1e-30)), 0.0)
    return legal_log_policy(jnp.asarray(logits), jnp.asarray(legal)), jnp.asarray(legal)


def test_threshold_zero_is_the_sampling_form_bit_identical():
    log_policy, legal = _row([0.5, 0.3, 0.0, 0.2])
    pruned = prune_log_policy(log_policy, legal, 0.0)
    expected = jnp.where(legal, log_policy, DTYPE_MIN)
    np.testing.assert_array_equal(np.asarray(pruned), np.asarray(expected))


def test_below_the_line_is_removed_and_the_rest_renormalised():
    log_policy, legal = _row([0.6, 0.004, 0.0, 0.396])
    pruned = np.asarray(prune_log_policy(log_policy, legal, 0.005))
    assert pruned[1] == DTYPE_MIN and pruned[2] == DTYPE_MIN
    kept = np.exp(pruned[[0, 3]])
    np.testing.assert_allclose(kept, [0.6 / 0.996, 0.396 / 0.996], rtol=1e-6)
    np.testing.assert_allclose(kept.sum(), 1.0, rtol=1e-6)


def test_above_the_line_is_untouched():
    # Positive control for the previous test: the same row with the small
    # cell lifted just past the threshold keeps every legal cell as-is.
    log_policy, legal = _row([0.6, 0.006, 0.0, 0.394])
    pruned = np.asarray(prune_log_policy(log_policy, legal, 0.005))
    np.testing.assert_array_equal(
        pruned, np.asarray(jnp.where(legal, log_policy, DTYPE_MIN))
    )


def test_a_row_entirely_below_the_line_keeps_its_legal_set():
    # rnad.py FineTuning._threshold's degenerate guard.
    probabilities = np.full(300, 1 / 300, np.float32)
    log_policy, legal = _row(probabilities)
    pruned = np.asarray(prune_log_policy(log_policy, legal, 0.005))
    np.testing.assert_array_equal(
        pruned, np.asarray(jnp.where(legal, log_policy, DTYPE_MIN))
    )


def test_batched_rows_are_independent():
    log_policy, legal = _row([[0.6, 0.004, 0.0, 0.396], [0.25, 0.25, 0.25, 0.25]])
    pruned = np.asarray(prune_log_policy(log_policy, legal, 0.005))
    assert pruned[0, 1] == DTYPE_MIN
    np.testing.assert_allclose(np.exp(pruned[1]), 0.25, rtol=1e-6)


def test_removed_cell_has_zero_gradient_through_its_own_logit():
    legal = jnp.asarray([True, True, False, True])
    logits = jnp.log(jnp.asarray([0.6, 0.004, 1.0, 0.396]))

    def kept_log_prob(logits):
        return prune_log_policy(legal_log_policy(logits, legal), legal, 0.005)[0]

    gradient = np.asarray(jax.grad(kept_log_prob)(logits))
    # The removed cell's gradient is two cancelling log-sum-exp terms, zero
    # to float32 rounding (measured -2.4e-10); the illegal cell's exactly.
    assert abs(gradient[1]) < 1e-6 and gradient[2] == 0.0
    assert abs(gradient[0]) > 0.1 and abs(gradient[3]) > 0.1


@pytest.mark.gpu
@pytest.mark.slow
def test_real_actor_default_head_params_are_bit_identical(real_model_and_trajectory):
    from rl.model.config import get_player_model_config
    from rl.model.heads import HeadParams
    from rl.model.player_model import get_player_model
    from rl.model.utils import open_zero_init_paths

    _, variables, actor_input, actor_output = real_model_and_trajectory
    variables = open_zero_init_paths(variables, ["action_head", "dynamics_out_proj"])
    actor_input = actor_input.replace(
        env=jax.tree.map(lambda leaf: leaf[:4], actor_input.env)
    )
    actor_output = jax.tree.map(lambda leaf: leaf[:4], actor_output)
    apply_model = jax.jit(
        get_player_model(get_player_model_config(9, train=False)).apply
    )
    rngs = {"sampling": jax.random.key(9)}
    plain = apply_model(variables, actor_input, actor_output, HeadParams(), rngs=rngs)
    explicit = apply_model(
        variables,
        actor_input,
        actor_output,
        HeadParams(prune_threshold=0.0),
        rngs=rngs,
    )
    for expected, actual in zip(
        jax.tree.leaves(plain), jax.tree.leaves(explicit), strict=True
    ):
        np.testing.assert_array_equal(expected, actual)
    # Positive control: a threshold that bites changes the sampled
    # distribution (the stored log_prob is mu's) but no policy metric.
    thresholded = apply_model(
        variables,
        actor_input,
        actor_output,
        HeadParams(prune_threshold=0.1),
        rngs=rngs,
    )
    assert np.any(
        np.asarray(thresholded.action_head.log_prob)
        != np.asarray(plain.action_head.log_prob)
    )
    np.testing.assert_array_equal(
        np.asarray(thresholded.action_head.entropy),
        np.asarray(plain.action_head.entropy),
    )
