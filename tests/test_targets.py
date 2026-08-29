"""v-trace return math on hand-checkable inputs, plus the full target
pipeline on the real example trajectory bundled in rl/environment/ex.bin."""

import jax.numpy as jnp
import numpy as np
import pytest

from rl.online.training.targets import (
    compute_player_targets,
    two_hot,
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


def test_vtrace_bf16_inputs_do_not_break_the_scan_carry():
    """Regression for the 2026-08-13 session crash (fixed in 15b6a3f):
    bf16 values with f32 python-scalar-promoted discounts made the scan
    carry dtype disagree. The recursion must run and stay finite. (The
    original guard rode on upgo_returns, deleted 2026-08-21 with the
    single-action PG terms — the constraint is on the recursion, so it
    moved to the one that survives.)"""
    td = jnp.array([0.5, 0.25, 0.125], dtype=jnp.bfloat16)
    disc = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)
    c = jnp.ones(3, dtype=jnp.float32)
    got = vtrace(td, disc, c)
    assert got.dtype == jnp.bfloat16
    assert np.isfinite(np.asarray(got.astype(jnp.float32))).all()


class TestTwoHot:
    def test_bin_centres_are_one_hot(self):
        support = jnp.array([-1.0, 0.0, 1.0])
        got = two_hot(jnp.array([-1.0, 0.0, 1.0]), support)
        np.testing.assert_allclose(
            np.asarray(got), np.eye(3, dtype=np.float32), atol=1e-6
        )

    def test_interpolates_and_clips(self):
        support = jnp.array([-1.0, 0.0, 1.0])
        got = np.asarray(two_hot(jnp.array([0.8, -0.25, 2.0, -3.0]), support))
        np.testing.assert_allclose(got[0], [0.0, 0.2, 0.8], atol=1e-6)
        np.testing.assert_allclose(got[1], [0.25, 0.75, 0.0], atol=1e-6)
        np.testing.assert_allclose(got[2], [0.0, 0.0, 1.0], atol=1e-6)  # clip hi
        np.testing.assert_allclose(got[3], [1.0, 0.0, 0.0], atol=1e-6)  # clip lo
        np.testing.assert_allclose(got.sum(-1), 1.0, atol=1e-6)


@pytest.fixture(scope="module")
def ex_target_inputs():
    """Real env outputs from ex.bin (T, B=1), an on-policy isr, a zero
    Retrace baseline (the flat-at-init advantage head), and a uniform
    categorical critic — value expectation exactly 0 over the [-1, 0, 1]
    support."""
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
        assert targets.policy_mask.shape == (T, B)
        assert targets.value_mask.shape == (T, B)
        assert np.isfinite(np.asarray(targets.win_returns)).all()

    def test_masks_follow_episode_structure(self, ex_target_inputs):
        batch, value_log_probs, isr, config = ex_target_inputs
        done = np.asarray(batch.player_transitions.env_output.done)
        targets, _ = compute_player_targets(batch, value_log_probs, isr, config)

        # value_mask covers everything up to and including the first done.
        expected = 1 - (np.cumsum(done, axis=0) - done)
        np.testing.assert_array_equal(
            np.asarray(targets.value_mask), expected.astype(bool)
        )
        # policy_mask is a strict subset: no terminal steps, no forced moves.
        policy_mask = np.asarray(targets.policy_mask)
        assert not (policy_mask & ~np.asarray(targets.value_mask)).any()
        assert not (policy_mask & done).any()
        assert policy_mask.any()  # the example game has real decisions

    def test_value_targets_stay_distributions(self, ex_target_inputs):
        # The simplex contract (2026-08-26): every masked CE label is a
        # proper two-hot distribution — non-negative, mass exactly 1 —
        # because the recursion runs in scalar space and projects once at
        # the end. The old distribution-space form accumulated signed
        # measures and could leave the simplex.
        batch, value_log_probs, isr, config = ex_target_inputs
        targets, _ = compute_player_targets(batch, value_log_probs, isr, config)
        returns = np.asarray(targets.win_returns)
        mask = np.asarray(targets.value_mask)
        np.testing.assert_allclose(returns.sum(-1)[mask], 1.0, atol=1e-4)
        assert (returns[mask] >= 0.0).all()
        np.testing.assert_allclose(returns.sum(-1)[~mask], 0.0, atol=1e-6)

    def test_on_policy_diagnostics(self, ex_target_inputs):
        batch, value_log_probs, isr, config = ex_target_inputs
        _, logs = compute_player_targets(batch, value_log_probs, isr, config)
        # isr == 1 everywhere: full effective sample size, nothing clipped.
        np.testing.assert_allclose(float(logs["player_isr_ess"]), 1.0, atol=1e-3)
        np.testing.assert_allclose(float(logs["player_rho_clip_frac"]), 0.0)


def _min_batch(done, win_reward, action_mask, action_index):
    """Minimal Batch for compute_player_targets: env rows plus the
    taken-action index."""
    from rl.environment.interfaces import (
        Batch,
        PlayerActorOutput,
        PlayerAgentOutput,
        PlayerEnvOutput,
        PlayerPolicyHeadOutput,
        PlayerTransition,
    )

    return Batch(
        player_transitions=PlayerTransition(
            env_output=PlayerEnvOutput(
                done=done, win_reward=win_reward, action_mask=action_mask
            ),
            agent_output=PlayerAgentOutput(
                actor_output=PlayerActorOutput(
                    action_head=PlayerPolicyHeadOutput(action_index=action_index)
                )
            ),
        )
    )


class TestPolicyAdvantage:
    """The v-trace policy advantage the PPO surrogate reads: hand-checked
    recursion, f32 contract, bootstrap-on-r at the terminal row."""

    def _targets(self, done, win_reward, isr=None, player_lambda=0.8):
        from rl.online.config import Porygon2LearnerConfig

        T, B = done.shape
        batch = _min_batch(
            done, win_reward, jnp.ones((T, B, 2, 2), bool), jnp.zeros((T, B), jnp.int32)
        )
        value_log_probs = jnp.full((T, B, 3), jnp.log(1.0 / 3.0), dtype=jnp.float32)
        if isr is None:
            isr = jnp.ones((T, B), dtype=jnp.float32)
        targets, _ = compute_player_targets(
            batch,
            value_log_probs,
            isr,
            Porygon2LearnerConfig(player_gamma=1.0, player_lambda=player_lambda),
        )
        return targets

    def test_matches_hand_recursion(self):
        """T=3, terminal win on the done row, V=0 everywhere, lambda 0.8:
        td = [0, 0, 1]; v-trace values [0.64, 0.8, 1]; q_bootstrap =
        [0.8*0.8, 0.8*1, 0] and the done row's discount is 0, so
        pg_adv = [0.64, 0.8, 1] — the outcome enters the recursion once
        and decays by lambda per step backward."""
        done = jnp.array([[False], [False], [True]])
        win_reward = jnp.zeros((3, 1, 3), dtype=jnp.float32).at[:, :, 1].set(1.0)
        win_reward = win_reward.at[2, :, :].set(jnp.array([0.0, 0.0, 1.0]))
        targets = self._targets(done, win_reward)
        assert targets.pg_advantages.dtype == jnp.float32
        np.testing.assert_allclose(
            np.asarray(targets.pg_advantages[:, 0]), [0.64, 0.8, 1.0], atol=1e-6
        )

    def test_rho_truncation_attenuates_the_advantage(self):
        """isr > 1 is clipped to 1 (no amplification); isr < 1 scales the
        row's advantage down by exactly rho."""
        done = jnp.array([[False], [False], [True]])
        win_reward = jnp.zeros((3, 1, 3), dtype=jnp.float32).at[:, :, 1].set(1.0)
        win_reward = win_reward.at[2, :, :].set(jnp.array([0.0, 0.0, 1.0]))
        clipped = self._targets(done, win_reward, isr=jnp.full((3, 1), 4.0))
        on_policy = self._targets(done, win_reward)
        np.testing.assert_allclose(
            np.asarray(clipped.pg_advantages), np.asarray(on_policy.pg_advantages)
        )
        half = self._targets(done, win_reward, isr=jnp.full((3, 1), 0.5))
        # The terminal row has no v_t continuation, so its advantage is
        # rho * (r - V) exactly: half the on-policy value.
        assert float(half.pg_advantages[2, 0]) == pytest.approx(0.5, abs=1e-6)


class TestMagnetKl:
    """The differentiated NashPG magnet: full-distribution forward
    KL(pi || pi_reg) over legal cells (targets.reference_kl), zero at the
    reference and insensitive to illegal-cell junk."""

    def test_reference_kl_zero_at_reference_and_positive_off_it(self):
        from rl.online.training.targets import reference_kl

        legal = jnp.asarray([[True, True, True, False]])
        log_pi = jnp.log(jnp.asarray([[0.7, 0.2, 0.1, 1.0]]))
        np.testing.assert_allclose(
            np.asarray(reference_kl(log_pi, log_pi, legal)), 0.0, atol=1e-6
        )
        log_ref = jnp.log(jnp.asarray([[1 / 3, 1 / 3, 1 / 3, 1.0]]))
        kl = float(reference_kl(log_pi, log_ref, legal)[0])
        expected = sum(p * np.log(p / (1 / 3)) for p in (0.7, 0.2, 0.1))
        np.testing.assert_allclose(kl, expected, rtol=1e-5)
        # The illegal cell's junk never leaks in.
        log_ref_junk = log_ref.at[0, 3].set(-50.0)
        np.testing.assert_allclose(
            float(reference_kl(log_pi, log_ref_junk, legal)[0]), expected, rtol=1e-5
        )
