"""v-trace return math on hand-checkable inputs, plus the full target
pipeline on the real example trajectory bundled in rl/environment/ex.bin."""

import jax.numpy as jnp
import numpy as np
import pytest

from rl.model.heads import compose_q
from rl.online.training.targets import (
    compute_player_targets,
    compute_q_onestep_targets,
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


def _q_batch(done, win_reward, action_mask, action_index):
    """Minimal Batch for compute_q_targets: env rows plus the taken-action
    index (the only agent_output field the Retrace path reads)."""
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


class TestQOnestepHandExample:
    """T=3, B=1, terminal win on the last (done) row, V_target = 0.2 on
    every row: y = [0 + V(s1), 0 + r, 0 (done row)] = [0.2, 1, 0]."""

    def test_label_matches_hand_computation(self):
        from rl.online.config import Porygon2LearnerConfig

        T, B, N = 3, 1, 2
        done = jnp.array([[False], [False], [True]])
        win_reward = jnp.zeros((T, B, 3), dtype=jnp.float32)
        win_reward = win_reward.at[:, :, 1].set(1.0)  # scalar 0 rows
        win_reward = win_reward.at[2, :, :].set(jnp.array([0.0, 0.0, 1.0]))  # win
        action_mask = jnp.ones((T, B, N, N), dtype=bool)
        action_index = jnp.zeros((T, B), dtype=jnp.int32)
        batch = _q_batch(done, win_reward, action_mask, action_index)
        y = compute_q_onestep_targets(
            batch, jnp.full((T, B), 0.2), Porygon2LearnerConfig(player_gamma=1.0)
        )
        np.testing.assert_allclose(np.asarray(y[:, 0]), [0.2, 1.0, 0.0], atol=1e-6)

    def test_rows_past_the_first_done_read_zero(self):
        from rl.online.config import Porygon2LearnerConfig

        T, B, N = 4, 1, 2
        done = jnp.array([[False], [True], [False], [False]])
        win_reward = jnp.zeros((T, B, 3), dtype=jnp.float32).at[:, :, 1].set(1.0)
        batch = _q_batch(
            done, win_reward, jnp.ones((T, B, N, N), bool), jnp.zeros((T, B), jnp.int32)
        )
        y = compute_q_onestep_targets(
            batch, jnp.full((T, B), 0.5), Porygon2LearnerConfig(player_gamma=1.0)
        )
        np.testing.assert_allclose(np.asarray(y[:, 0]), [0.0, 0.0, 0.0, 0.0], atol=1e-6)


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
    adv_taken = jnp.zeros((T, B), dtype=jnp.float32)
    return batch, value_log_probs, isr, adv_taken, Porygon2LearnerConfig()


class TestQOnestepOnExTrajectory:
    def test_shapes_and_ranges(self, ex_target_inputs):
        batch, _, isr, _, config = ex_target_inputs
        env = batch.player_transitions.env_output
        T, B = env.done.shape
        flat_mask = np.asarray(env.action_mask).reshape(T, B, -1)
        A = flat_mask.shape[-1]
        full = _q_batch(
            env.done,
            env.win_reward,
            env.action_mask,
            jnp.argmax(jnp.asarray(flat_mask), axis=-1),
        )
        v = jnp.zeros((T, B), dtype=jnp.float32)
        y = compute_q_onestep_targets(full, v, config)
        assert y.shape == (T, B)
        assert np.isfinite(np.asarray(y)).all()
        assert (np.abs(np.asarray(y)) <= 1.0 + 1e-6).all()
        _, q_all = compose_q(
            v,
            jnp.zeros((T, B, A)),
            jnp.full((T, B, A), -np.log(A)),
            jnp.asarray(flat_mask),
        )
        assert q_all.shape == (T, B, A)
        assert np.isfinite(np.asarray(q_all)).all()


class TestPlayerTargetsOnExTrajectory:
    def test_shapes_and_finiteness(self, ex_target_inputs):
        batch, value_log_probs, isr, adv_taken, config = ex_target_inputs
        T, B = batch.player_transitions.env_output.done.shape
        targets, _ = compute_player_targets(
            batch, value_log_probs, isr, adv_taken, config
        )

        assert targets.win_returns.shape == (T, B, 3)
        assert targets.policy_mask.shape == (T, B)
        assert targets.value_mask.shape == (T, B)
        assert np.isfinite(np.asarray(targets.win_returns)).all()

    def test_masks_follow_episode_structure(self, ex_target_inputs):
        batch, value_log_probs, isr, adv_taken, config = ex_target_inputs
        done = np.asarray(batch.player_transitions.env_output.done)
        targets, _ = compute_player_targets(
            batch, value_log_probs, isr, adv_taken, config
        )

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
        batch, value_log_probs, isr, adv_taken, config = ex_target_inputs
        targets, _ = compute_player_targets(
            batch, value_log_probs, isr, adv_taken, config
        )
        returns = np.asarray(targets.win_returns)
        mask = np.asarray(targets.value_mask)
        np.testing.assert_allclose(returns.sum(-1)[mask], 1.0, atol=1e-4)
        assert (returns[mask] >= 0.0).all()
        np.testing.assert_allclose(returns.sum(-1)[~mask], 0.0, atol=1e-6)

    def test_on_policy_diagnostics(self, ex_target_inputs):
        batch, value_log_probs, isr, adv_taken, config = ex_target_inputs
        _, logs = compute_player_targets(batch, value_log_probs, isr, adv_taken, config)
        # isr == 1 everywhere: full effective sample size, nothing clipped.
        np.testing.assert_allclose(float(logs["player_isr_ess"]), 1.0, atol=1e-3)
        np.testing.assert_allclose(float(logs["player_rho_clip_frac"]), 0.0)


class TestRetraceBaseline:
    """Retrace over the composed Q: the value target's baseline is
    Q(s, a) = V(s) + A(s, a), not V(s)."""

    def _expectation(self, targets, support):
        return np.asarray(targets.win_returns @ support)

    def test_zero_baseline_reproduces_the_vtrace_targets(self, ex_target_inputs):
        """The flat-at-init advantage head must leave the estimator exactly
        where it was — checked against a reference implementation of the
        plain scalar v-trace form, not against another call of the same
        function."""
        batch, value_log_probs, isr, adv_taken, config = ex_target_inputs
        support = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32)
        done = batch.player_transitions.env_output.done
        mask = (1 - (jnp.cumsum(done, axis=0) - done)).astype(jnp.float32)
        discount_t = (1 - done).astype(jnp.float32) * config.player_gamma * mask
        rho = jnp.minimum(1.0, isr).astype(jnp.float32)
        v_tm1 = jnp.exp(value_log_probs.astype(jnp.float32)) @ support
        v_t = jnp.concatenate([v_tm1[1:], v_tm1[-1:]], axis=0)
        r_t = (
            batch.player_transitions.env_output.win_reward.astype(jnp.float32) @ support
        )
        td = rho * mask * (r_t + discount_t * v_t - v_tm1)
        scalar = (vtrace(td, discount_t, rho * config.player_lambda) + v_tm1) * mask
        want = two_hot(scalar, support) * mask[..., None]

        got, _ = compute_player_targets(
            batch, value_log_probs, isr, jnp.zeros_like(adv_taken), config
        )
        np.testing.assert_allclose(
            np.asarray(got.win_returns), np.asarray(want), atol=1e-6
        )

    def test_positive_advantage_lowers_the_state_target(self, ex_target_inputs):
        """Positive control — without it the test above passes vacuously.
        Q(s, a) = V + A, so subtracting a larger baseline must lower the
        state's return, and a negative advantage must raise it."""
        batch, value_log_probs, isr, adv_taken, config = ex_target_inputs
        support = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        mask = np.asarray(
            compute_player_targets(batch, value_log_probs, isr, adv_taken, config)[
                0
            ].value_mask
        )

        def mean_target(scalar):
            targets, _ = compute_player_targets(
                batch,
                value_log_probs,
                isr,
                jnp.full_like(adv_taken, scalar),
                config,
            )
            return self._expectation(targets, support)[mask].mean()

        assert mean_target(0.2) < mean_target(0.0) < mean_target(-0.2)

    def test_baseline_does_not_leak_into_the_policy_advantage(self, ex_target_inputs):
        """pg_advantages is the PLAIN v-trace pass: the Retrace baseline
        (adv_taken) moves the VALUE targets only. Positive control below —
        the value targets must move for the same perturbation."""
        batch, value_log_probs, isr, adv_taken, config = ex_target_inputs
        base, _ = compute_player_targets(
            batch, value_log_probs, isr, jnp.zeros_like(adv_taken), config
        )
        shifted, _ = compute_player_targets(
            batch, value_log_probs, isr, jnp.full_like(adv_taken, 0.3), config
        )
        np.testing.assert_array_equal(
            np.asarray(base.pg_advantages), np.asarray(shifted.pg_advantages)
        )
        assert not np.array_equal(
            np.asarray(base.win_returns), np.asarray(shifted.win_returns)
        )

    def test_targets_stay_normalised_under_a_nonzero_baseline(self, ex_target_inputs):
        batch, value_log_probs, isr, adv_taken, config = ex_target_inputs
        targets, _ = compute_player_targets(
            batch, value_log_probs, isr, jnp.full_like(adv_taken, 0.4), config
        )
        sums = np.asarray(targets.win_returns).sum(-1)
        mask = np.asarray(targets.value_mask)
        np.testing.assert_allclose(sums[mask], 1.0, atol=1e-4)


class TestPolicyAdvantage:
    """The v-trace policy advantage the PPO surrogate reads: hand-checked
    recursion, f32 contract, bootstrap-on-r at the terminal row."""

    def _targets(self, done, win_reward, isr=None, player_lambda=0.8):
        from rl.online.config import Porygon2LearnerConfig

        T, B = done.shape
        batch = _q_batch(
            done, win_reward, jnp.ones((T, B, 2, 2), bool), jnp.zeros((T, B), jnp.int32)
        )
        value_log_probs = jnp.full((T, B, 3), jnp.log(1.0 / 3.0), dtype=jnp.float32)
        if isr is None:
            isr = jnp.ones((T, B), dtype=jnp.float32)
        targets, _ = compute_player_targets(
            batch,
            value_log_probs,
            isr,
            jnp.zeros((T, B), jnp.float32),
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


def restoring_force(kl_fn, probs, ref_probs, legal, **kwargs):
    """Restoring force per logit: -dL/dy, so positive = pushes mass UP.
    Differentiates through the legal log-softmax, as the learner does."""
    import jax

    y = jnp.log(jnp.asarray([probs]))
    log_ref = jnp.log(jnp.asarray([ref_probs]))

    def loss(logits):
        log_pi = jax.nn.log_softmax(jnp.where(legal, logits, -1e9), axis=-1)
        return kl_fn(log_pi, log_ref, legal, **kwargs).sum()

    return -np.asarray(jax.grad(loss)(y))[0]


class TestSupportKl:
    """The support-preserving anchor: KL(p_T || pi) over legal cells, where
    p_T ~ pi_reg ** (1/T) is the frozen reference raised to temperature T.

    What these pin is the GRADIENT, not the value — the whole reason the term
    exists is that its per-logit force pi(b) - p_T(b) has no pi prefactor, so
    unlike the magnet it survives on a starved cell."""

    LEGAL = jnp.asarray([[True, True, True, False]])

    def _force(self, kl_fn, probs, ref_probs, **kwargs):
        return restoring_force(kl_fn, probs, ref_probs, self.LEGAL, **kwargs)

    def test_zero_at_reference_only_when_temperature_is_one(self):
        from rl.online.training.targets import support_kl

        legal = self.LEGAL
        log_pi = jnp.log(jnp.asarray([[0.7, 0.2, 0.1, 1.0]]))
        np.testing.assert_allclose(
            np.asarray(support_kl(log_pi, log_pi, legal, 1.0)), 0.0, atol=1e-6
        )
        # POSITIVE CONTROL: raising T must move it off zero, or the assertion
        # above would pass for a function that returns a constant zero.
        assert float(support_kl(log_pi, log_pi, legal, 1.2)[0]) > 1e-4

    def test_gradient_is_pi_minus_p_t(self):
        from rl.online.training.targets import support_kl

        probs, ref = [0.7, 0.2, 0.1, 1.0], [0.5, 0.3, 0.2, 1.0]
        force = self._force(support_kl, probs, ref, temperature=1.2)
        raised = np.asarray(ref[:3]) ** (1 / 1.2)
        p_t = raised / raised.sum()
        np.testing.assert_allclose(force[:3], p_t - np.asarray(probs[:3]), atol=1e-5)

    def test_force_is_zero_sum_over_legal_cells(self):
        from rl.online.training.targets import support_kl

        force = self._force(
            support_kl, [0.7, 0.2, 0.1, 1.0], [0.5, 0.3, 0.2, 1.0], temperature=1.2
        )
        np.testing.assert_allclose(force[:3].sum(), 0.0, atol=1e-5)

    def test_force_survives_a_starved_cell_where_the_magnet_dies(self):
        """The design claim, as a test. Drive one cell's mass toward zero and
        watch the two references diverge: the magnet's pi prefactor takes its
        force to nothing, while this one converges on p_T(b)."""
        from rl.online.training.targets import reference_kl, support_kl

        uniform = [1 / 3, 1 / 3, 1 / 3, 1.0]
        magnet_forces, support_forces = [], []
        for starved in (1e-2, 1e-4, 1e-6):
            rest = (1.0 - starved) / 2
            probs = [rest, rest, starved, 1.0]
            magnet_forces.append(self._force(reference_kl, probs, uniform)[2])
            support_forces.append(
                self._force(support_kl, probs, uniform, temperature=1.2)[2]
            )
        # The magnet's restoring force decays to nothing as the cell starves.
        assert magnet_forces[0] > magnet_forces[1] > magnet_forces[2]
        assert magnet_forces[2] < 1e-4
        # This one converges on p_T(b) ~ 1/3 instead of vanishing.
        assert all(f > 0.3 for f in support_forces)
        assert support_forces[2] > 3000 * magnet_forces[2]

    def test_force_stays_bounded_on_a_numerically_dead_cell(self):
        """Regression for the dx65cpwp failure class: an unbounded per-cell
        force (-eta * log pi) diverged twice. This one is bounded by 1."""
        from rl.online.training.targets import support_kl

        probs = [0.5 - 5e-9, 0.5 - 5e-9, 1e-8, 1.0]
        force = self._force(
            support_kl, probs, [1 / 3, 1 / 3, 1 / 3, 1.0], temperature=1.2
        )
        assert np.all(np.isfinite(force))
        assert np.all(np.abs(force[:3]) <= 1.0)

    def test_illegal_cells_and_empty_rows_are_inert(self):
        from rl.online.training.targets import support_kl

        legal = self.LEGAL
        log_pi = jnp.log(jnp.asarray([[0.7, 0.2, 0.1, 1.0]]))
        log_ref = jnp.log(jnp.asarray([[0.5, 0.3, 0.2, 1.0]]))
        base = float(support_kl(log_pi, log_ref, legal, 1.2)[0])
        junk = log_ref.at[0, 3].set(-50.0)
        np.testing.assert_allclose(
            float(support_kl(log_pi, junk, legal, 1.2)[0]), base, rtol=1e-5
        )
        none_legal = jnp.zeros_like(legal, dtype=bool)
        value = float(support_kl(log_pi, log_ref, none_legal, 1.2)[0])
        assert value == 0.0 and np.isfinite(value)


class TestSupportTilt:
    """Phase 3: p* ~ pi_reg^(1/T) * exp(clip(tilt)) — the tilt is where
    per-cell knowledge (sg(A_target)/tau at the call site) enters the anchor.
    These pin that the tilt reorders the TARGET while every bounding property
    of the untilted anchor survives it."""

    LEGAL = jnp.asarray([[True, True, True, False]])
    UNIFORM = [1 / 3, 1 / 3, 1 / 3, 1.0]

    def test_zero_tilt_is_bitwise_off(self):
        from rl.online.training.targets import support_kl, support_target

        log_ref = jnp.log(jnp.asarray([[0.5, 0.3, 0.2, 1.0]]))
        base = support_target(log_ref, self.LEGAL, 1.2)
        tilted = support_target(log_ref, self.LEGAL, 1.2, jnp.zeros_like(log_ref))
        assert (np.asarray(base) == np.asarray(tilted)).all()
        log_pi = jnp.log(jnp.asarray([[0.7, 0.2, 0.1, 1.0]]))
        untilted_kl = support_kl(log_pi, log_ref, self.LEGAL, 1.2)
        zero_tilt_kl = support_kl(
            log_pi, log_ref, self.LEGAL, 1.2, jnp.zeros_like(log_ref)
        )
        assert (np.asarray(untilted_kl) == np.asarray(zero_tilt_kl)).all()

    def test_gradient_is_pi_minus_tilted_target(self):
        """POSITIVE CONTROL built in: at uniform pi against a uniform
        reference the untilted force is identically zero, so any push here is
        the tilt's — it must lift the preferred cell and drop the others,
        by exactly p* - pi."""
        from rl.online.training.targets import support_kl

        tilt = jnp.asarray([[0.0, 0.0, 1.0, 0.0]])
        force = restoring_force(
            support_kl,
            self.UNIFORM,
            self.UNIFORM,
            self.LEGAL,
            temperature=1.0,
            tilt_logits=tilt,
        )
        weights = np.exp(np.asarray([0.0, 0.0, 1.0]))
        p_star = weights / weights.sum()
        np.testing.assert_allclose(
            force[:3], p_star - np.asarray(self.UNIFORM[:3]), atol=1e-5
        )
        assert force[2] > 0.05 and force[0] < 0
        untilted = restoring_force(
            support_kl, self.UNIFORM, self.UNIFORM, self.LEGAL, temperature=1.0
        )
        np.testing.assert_allclose(untilted[:3], 0.0, atol=1e-6)

    def test_tilted_force_is_zero_sum_over_legal_cells(self):
        from rl.online.training.targets import support_kl

        force = restoring_force(
            support_kl,
            [0.7, 0.2, 0.1, 1.0],
            [0.5, 0.3, 0.2, 1.0],
            self.LEGAL,
            temperature=1.2,
            tilt_logits=jnp.asarray([[0.5, -0.2, 1.0, 0.0]]),
        )
        np.testing.assert_allclose(force[:3].sum(), 0.0, atol=1e-5)

    def test_extreme_tilt_is_clipped_and_force_stays_bounded(self):
        """A wild advantage outlier is capped at the +-3 exponent clip — the
        force it produces must be identical to tilt exactly 3, finite, and
        within the bound of 1 even on a numerically dead cell."""
        from rl.online.training.targets import support_kl

        starved = [0.5 - 5e-9, 0.5 - 5e-9, 1e-8, 1.0]
        wild = restoring_force(
            support_kl,
            starved,
            self.UNIFORM,
            self.LEGAL,
            temperature=1.0,
            tilt_logits=jnp.asarray([[0.0, 0.0, 100.0, 0.0]]),
        )
        capped = restoring_force(
            support_kl,
            starved,
            self.UNIFORM,
            self.LEGAL,
            temperature=1.0,
            tilt_logits=jnp.asarray([[0.0, 0.0, 3.0, 0.0]]),
        )
        np.testing.assert_allclose(wild, capped, atol=1e-6)
        assert np.all(np.isfinite(wild))
        assert np.all(np.abs(wild[:3]) <= 1.0)

    def test_tilt_receives_no_gradient(self):
        """The policy reads the critic through this term; the critic must not
        feel the policy back — d(loss)/d(tilt) is exactly zero."""
        import jax

        from rl.online.training.targets import support_kl

        log_pi = jnp.log(jnp.asarray([[0.7, 0.2, 0.1, 1.0]]))
        log_ref = jnp.log(jnp.asarray([[0.5, 0.3, 0.2, 1.0]]))

        def loss(tilt):
            return support_kl(log_pi, log_ref, self.LEGAL, 1.0, tilt).sum()

        tilt_grad = jax.grad(loss)(jnp.asarray([[0.5, -0.2, 1.0, 0.0]]))
        np.testing.assert_allclose(np.asarray(tilt_grad), 0.0, atol=0.0)


class TestCentreWithinModality:
    """Phase 4: the tilt must carry WITHIN-modality ranking only. The
    modality-level component of raw A is the self-confirming Q^pi view that
    inverted the phase-3 anchor, and centring removes it exactly."""

    LEGAL = jnp.asarray([[True, True, True, True, False]])
    SWITCH = jnp.asarray([[False, False, True, True, False]])

    def test_modality_constant_is_removed_ranking_survives(self):
        from rl.online.training.targets import centre_within_modality

        values = jnp.asarray([[0.1, 0.3, -5.0, -5.4, 9.9]])
        # A uniformly pessimistic view of the whole switch modality must be
        # invisible to the tilt...
        shifted = values + jnp.where(self.SWITCH, -7.0, 0.0)
        base = centre_within_modality(values, self.LEGAL, self.SWITCH)
        moved = centre_within_modality(shifted, self.LEGAL, self.SWITCH)
        np.testing.assert_allclose(np.asarray(base), np.asarray(moved), atol=1e-6)
        # ...while the POSITIVE CONTROL, the within-group ranking, survives:
        # cell 2 sits 0.4 above cell 3 before and after centring.
        assert float(base[0, 2] - base[0, 3]) == pytest.approx(0.4, abs=1e-6)
        assert float(base[0, 1] - base[0, 0]) == pytest.approx(0.2, abs=1e-6)

    def test_zero_sum_per_group_and_illegal_cells_zero(self):
        from rl.online.training.targets import centre_within_modality

        values = jnp.asarray([[0.1, 0.3, -5.0, -5.4, 9.9]])
        centred = np.asarray(centre_within_modality(values, self.LEGAL, self.SWITCH))
        assert centred[0, :2].sum() == pytest.approx(0.0, abs=1e-6)
        assert centred[0, 2:4].sum() == pytest.approx(0.0, abs=1e-6)
        assert centred[0, 4] == 0.0

    def test_composed_target_ignores_modality_level_advantage(self):
        """End-to-end pin of the phase-3 failure: a whole-modality advantage
        shift must leave p* unchanged (to f32 rounding — the two centrings
        subtract different means, so exact bit-equality is not available)."""
        from rl.online.training.targets import (
            centre_within_modality,
            support_target,
        )

        log_ref = jnp.log(jnp.asarray([[0.4, 0.3, 0.2, 0.09, 0.01]]))
        values = jnp.asarray([[0.1, 0.3, -5.0, -5.4, 9.9]])
        shifted = values + jnp.where(self.SWITCH, -7.0, 0.0)
        tau = 0.125

        def target_of(advantages):
            tilt = centre_within_modality(advantages, self.LEGAL, self.SWITCH) / tau
            return np.asarray(support_target(log_ref, self.LEGAL, 1.2, tilt))

        np.testing.assert_allclose(target_of(values), target_of(shifted), atol=1e-5)
