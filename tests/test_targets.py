"""v-trace / UPGO return math on hand-checkable inputs."""

import jax.numpy as jnp
import numpy as np

from rl.online.targets import upgo_returns, vtrace


def naive_vtrace(td_errors, discount_t, c_tm1):
    """Reference implementation: plain reverse recursion."""
    T = td_errors.shape[0]
    out = np.zeros_like(td_errors)
    acc = np.zeros_like(td_errors[0])
    for t in reversed(range(T)):
        acc = td_errors[t] + discount_t[t] * c_tm1[t] * acc
        out[t] = acc
    return out


def test_vtrace_matches_naive_recursion():
    rng = np.random.default_rng(0)
    td = rng.normal(size=(12, 4, 3)).astype(np.float32)
    disc = rng.uniform(0.0, 1.0, size=(12, 4, 3)).astype(np.float32)
    c = rng.uniform(0.0, 1.0, size=(12, 4, 3)).astype(np.float32)

    got = np.asarray(vtrace(jnp.asarray(td), jnp.asarray(disc), jnp.asarray(c)))
    want = naive_vtrace(td, disc, c)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_vtrace_zero_trace_is_identity():
    td = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    disc = jnp.ones_like(td)
    got = vtrace(td, disc, jnp.zeros_like(td))
    np.testing.assert_allclose(np.asarray(got), np.asarray(td))


def test_upgo_hand_example():
    # T=4, single trajectory, terminal win at the last step.
    v = jnp.array([0.5, 0.6, 0.2, 0.3])
    r = jnp.array([0.0, 0.0, 0.0, 1.0])
    disc = jnp.array([1.0, 1.0, 1.0, 0.0])

    # q_hat = r + disc * v_next = [0.6, 0.2, 0.3, 1.0]
    # follow = q_hat[t+1] >= v[t+1] -> [False, True, True, (last) False]
    # g3 = 1.0 (terminal); g2 follows -> 1.0; g1 follows -> 1.0;
    # g0 cut -> bootstraps v_next[0] = 0.6
    g, cut = upgo_returns(v, r, disc)
    np.testing.assert_allclose(np.asarray(g), [0.6, 1.0, 1.0, 1.0], atol=1e-6)
    np.testing.assert_array_equal(np.asarray(cut), [True, False, False, True])


def test_upgo_all_better_than_expected_is_monte_carlo():
    # Zero critic + non-negative rewards: lookahead never underperforms,
    # so no truncation before the (always-cut) final step and G is the
    # plain discounted return.
    v = jnp.zeros(4)
    r = jnp.array([0.0, 0.0, 0.0, 1.0])
    disc = jnp.array([0.9, 0.9, 0.9, 0.0])
    g, cut = upgo_returns(v, r, disc)
    np.testing.assert_allclose(np.asarray(g), [0.9**3, 0.9**2, 0.9, 1.0], rtol=1e-6)
    assert not np.asarray(cut)[:-1].any()


def test_upgo_bf16_inputs_upcast_to_f32():
    # Regression for the 2026-08-13 session crash (fixed in 15b6a3f): bf16
    # values with f32 python-scalar-promoted discounts made the scan carry
    # dtype disagree. The recursion must run and return f32.
    v = jnp.array([0.5, 0.25, 0.125], dtype=jnp.bfloat16)
    r = jnp.array([0.0, 0.0, 1.0], dtype=jnp.bfloat16)
    disc = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)
    g, cut = upgo_returns(v, r, disc)
    assert g.dtype == jnp.float32
    assert np.isfinite(np.asarray(g)).all()


def test_upgo_batched_shapes():
    T, B = 6, 3
    rng = np.random.default_rng(1)
    v = jnp.asarray(rng.normal(size=(T, B)).astype(np.float32))
    r = jnp.asarray(rng.normal(size=(T, B)).astype(np.float32))
    disc = jnp.ones((T, B), dtype=jnp.float32)
    g, cut = upgo_returns(v, r, disc)
    assert g.shape == (T, B)
    assert cut.shape == (T, B)
    assert np.isfinite(np.asarray(g)).all()
