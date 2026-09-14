import jax
import jax.numpy as jnp
import numpy as np

from rl.online.training.loss import uniform_kl_rows


def test_uniform_kl_matches_definition_and_ignores_illegal_cells() -> None:
    legal = jnp.array([[True, True, False], [True, True, True]])
    policy = jnp.array([[0.9, 0.1, 0.0], [0.2, 0.3, 0.5]])
    actual = jax.jit(uniform_kl_rows)(jnp.log(policy), legal)
    expected = [
        np.mean(np.log(0.5 / np.array([0.9, 0.1]))),
        np.mean(np.log((1 / 3) / np.array([0.2, 0.3, 0.5]))),
    ]
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    changed = jnp.log(policy).at[0, 2].set(jnp.nan)
    np.testing.assert_array_equal(jax.jit(uniform_kl_rows)(changed, legal), actual)


def test_uniform_empty_and_singleton_rows_are_zero() -> None:
    legal = jnp.array([[True, True, True], [False, False, False], [True, False, False]])
    logits = jnp.where(legal, 0.0, -jnp.inf)
    log_policy = jax.nn.log_softmax(logits, axis=-1)
    actual = jax.jit(uniform_kl_rows)(log_policy, legal)
    np.testing.assert_allclose(actual, 0.0, atol=1e-7)
    assert actual.dtype == jnp.float32


def test_logit_gradient_is_bounded_zero_sum_and_revives_starved_actions() -> None:
    legal = jnp.array([[True, True, False, True], [True, False, False, False]])
    logits = jnp.array([[10000.0, -10000.0, 7.0, 0.0], [4.0, 1.0, 2.0, 3.0]])

    def objective(values):
        log_policy = jax.nn.log_softmax(jnp.where(legal, values, -jnp.inf), axis=-1)
        return uniform_kl_rows(log_policy, legal).sum()

    loss, gradient = jax.jit(jax.value_and_grad(objective))(logits)
    policy = jax.nn.softmax(jnp.where(legal, logits, -jnp.inf), axis=-1)
    uniform = legal / legal.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(gradient, policy - uniform, atol=1e-7)
    np.testing.assert_allclose(gradient.sum(axis=-1), 0.0, atol=1e-7)
    assert np.isfinite(loss)
    assert np.max(np.abs(gradient)) <= 1.0
    assert gradient[0, 1] < -0.3
    np.testing.assert_array_equal(gradient[~legal], 0.0)


def test_zero_coefficient_and_masked_rows_have_no_gradient() -> None:
    legal = jnp.ones((2, 3), dtype=bool)
    valid = jnp.array([True, False])
    logits = jnp.array([[1.0, -2.0, 0.0], [100.0, -100.0, 0.0]])

    def objective(values, coefficient):
        rows = uniform_kl_rows(jax.nn.log_softmax(values), legal)
        return coefficient * jnp.where(valid, rows, 0.0).sum()

    gradient = jax.jit(jax.grad(objective))(logits, 0.005)
    assert np.max(np.abs(gradient[0])) > 0.001
    np.testing.assert_array_equal(gradient[1], 0.0)
    np.testing.assert_array_equal(jax.jit(jax.grad(objective))(logits, 0.0), 0.0)
