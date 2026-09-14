import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.online.training.loss import vtrace_policy_loss


@pytest.mark.parametrize("taken_probability", [0.1, 0.4])
def test_score_gradient_contains_one_clipped_importance_weight(taken_probability):
    behaviour_probability = 0.2
    raw_advantage = 0.8
    logits = jnp.log(jnp.array([taken_probability, 1.0 - taken_probability]))

    def objective(policy_logits):
        log_policy = jax.nn.log_softmax(policy_logits)
        ratio = jnp.exp(log_policy[0]) / behaviour_probability
        weighted_advantage = jnp.minimum(ratio, 1.0) * raw_advantage
        return vtrace_policy_loss(
            log_prob=log_policy[:1],
            advantages=weighted_advantage[None],
            valid=jnp.array([True]),
        )

    gradient = jax.jit(jax.grad(objective))(logits)
    clipped_ratio = min(taken_probability / behaviour_probability, 1.0)
    score_gradient = np.array([1.0 - taken_probability, -(1.0 - taken_probability)])
    expected = -clipped_ratio * raw_advantage * score_gradient
    np.testing.assert_allclose(gradient, expected, rtol=1e-6)
    assert np.max(np.abs(gradient)) > 0.1


def test_advantage_is_detached_but_its_scale_is_preserved():
    log_prob = jnp.log(jnp.array([0.1, 0.6]))
    advantages = jnp.array([0.25, -0.5])
    valid = jnp.ones(2, dtype=bool)

    def objective(probability, advantage):
        return vtrace_policy_loss(
            log_prob=probability, advantages=advantage, valid=valid
        )

    probability_grad, advantage_grad = jax.jit(jax.grad(objective, argnums=(0, 1)))(
        log_prob, advantages
    )
    np.testing.assert_allclose(probability_grad, -advantages / 2.0)
    np.testing.assert_array_equal(advantage_grad, 0.0)
    np.testing.assert_allclose(
        objective(log_prob, 2.0 * advantages), 2.0 * objective(log_prob, advantages)
    )


def test_masked_rows_and_empty_batches_are_inert():
    log_prob = jnp.array([-0.5, jnp.nan, jnp.inf])
    advantages = jnp.array([0.4, jnp.inf, jnp.nan])
    valid = jnp.array([True, False, False])

    def objective(probability, mask):
        return vtrace_policy_loss(
            log_prob=probability, advantages=advantages, valid=mask
        )

    loss, gradient = jax.jit(jax.value_and_grad(objective))(log_prob, valid)
    np.testing.assert_allclose(loss, 0.2)
    np.testing.assert_allclose(gradient, [-0.4, 0.0, 0.0])
    empty_loss, empty_gradient = jax.jit(jax.value_and_grad(objective))(
        log_prob, jnp.zeros(3, dtype=bool)
    )
    np.testing.assert_array_equal(empty_loss, 0.0)
    np.testing.assert_array_equal(empty_gradient, 0.0)


def test_score_loss_promotes_bf16_inputs_to_f32():
    loss = vtrace_policy_loss(
        log_prob=jnp.array([-0.5], dtype=jnp.bfloat16),
        advantages=jnp.array([0.25], dtype=jnp.bfloat16),
        valid=jnp.array([True]),
    )
    assert loss.dtype == jnp.float32
    np.testing.assert_allclose(loss, 0.125)
