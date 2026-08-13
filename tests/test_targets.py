"""v-trace / UPGO return math on hand-checkable inputs, plus the full
target pipeline on the real example trajectory bundled in
rl/environment/ex.bin."""

import jax.numpy as jnp
import numpy as np
import pytest

from rl.online.targets import (
    compute_aux_value_targets,
    compute_player_targets,
    upgo_returns,
    vtrace,
)


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


@pytest.fixture(scope="module")
def ex_target_inputs():
    """Real env outputs from ex.bin (T, B=1), an on-policy isr, and a
    uniform categorical critic — value expectation exactly 0 over the
    [-1, 0, 1] support."""
    from rl.environment.interfaces import Batch, PlayerTransition
    from rl.environment.utils import get_ex_player_step
    from rl.online.config import Porygon2LearnerConfig

    actor_input, _ = get_ex_player_step()
    env = actor_input.env
    batch = Batch(player_transitions=PlayerTransition(env_output=env))
    T, B = env.done.shape
    value_log_probs = jnp.full((T, B, 3), jnp.log(1.0 / 3.0), dtype=jnp.float32)
    isr = jnp.ones((T, B), dtype=jnp.float32)
    return batch, value_log_probs, isr, Porygon2LearnerConfig()


class TestPlayerTargetsOnExTrajectory:
    def test_shapes_and_finiteness(self, ex_target_inputs):
        batch, value_log_probs, isr, config = ex_target_inputs
        T, B = batch.player_transitions.env_output.done.shape
        targets, _ = compute_player_targets(batch, value_log_probs, isr, config)

        assert targets.win_returns.shape == (T, B, 3)
        assert targets.advantages.shape == (T, B)
        assert targets.upgo_advantages.shape == (T, B)
        assert targets.policy_mask.shape == (T, B)
        assert targets.value_mask.shape == (T, B)
        for leaf in (targets.win_returns, targets.advantages, targets.upgo_advantages):
            assert np.isfinite(np.asarray(leaf)).all()

    def test_masks_follow_episode_structure(self, ex_target_inputs):
        batch, value_log_probs, isr, config = ex_target_inputs
        done = np.asarray(batch.player_transitions.env_output.done)
        targets, _ = compute_player_targets(batch, value_log_probs, isr, config)

        # value_mask covers everything up to and including the first done.
        expected = 1 - (np.cumsum(done, axis=0) - done)
        np.testing.assert_array_equal(np.asarray(targets.value_mask), expected.astype(bool))
        # policy_mask is a strict subset: no terminal steps, no forced moves.
        policy_mask = np.asarray(targets.policy_mask)
        assert not (policy_mask & ~np.asarray(targets.value_mask)).any()
        assert not (policy_mask & done).any()
        assert policy_mask.any()  # the example game has real decisions

    def test_value_targets_stay_distributions(self, ex_target_inputs):
        # Bin-space v-trace with a one-hot terminal outcome and gamma=1 must
        # keep each masked target row a probability distribution.
        batch, value_log_probs, isr, config = ex_target_inputs
        targets, _ = compute_player_targets(batch, value_log_probs, isr, config)
        sums = np.asarray(targets.win_returns).sum(-1)
        mask = np.asarray(targets.value_mask)
        np.testing.assert_allclose(sums[mask], 1.0, atol=1e-4)
        np.testing.assert_allclose(sums[~mask], 0.0, atol=1e-6)

    def test_on_policy_diagnostics(self, ex_target_inputs):
        batch, value_log_probs, isr, config = ex_target_inputs
        _, logs = compute_player_targets(batch, value_log_probs, isr, config)
        # isr == 1 everywhere: full effective sample size, nothing clipped.
        np.testing.assert_allclose(float(logs["player_isr_ess"]), 1.0, atol=1e-3)
        np.testing.assert_allclose(float(logs["player_rho_clip_frac"]), 0.0)

    def test_aux_targets_per_lambda(self, ex_target_inputs):
        batch, _, isr, config = ex_target_inputs
        T, B = batch.player_transitions.env_output.done.shape
        K = len(config.player_aux_lambdas)
        aux_log_probs = jnp.full((T, B, K, 3), jnp.log(1.0 / 3.0), dtype=jnp.float32)
        aux = compute_aux_value_targets(batch, aux_log_probs, isr, config)
        assert aux.shape == (T, B, K, 3)
        assert np.isfinite(np.asarray(aux)).all()
        # The lambda=1.0 MC-anchor row must also stay a distribution on
        # masked steps (same argument as the main head).
        done = np.asarray(batch.player_transitions.env_output.done)
        mask = (1 - (np.cumsum(done, axis=0) - done)).astype(bool)
        sums = np.asarray(aux).sum(-1)
        for k in range(K):
            np.testing.assert_allclose(sums[..., k][mask], 1.0, atol=1e-4)
