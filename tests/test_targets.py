"""v-trace / UPGO return math on hand-checkable inputs, plus the full
target pipeline on the real example trajectory bundled in
rl/environment/ex.bin."""

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
        # Bin-space v-trace with a one-hot terminal outcome and gamma=1 must
        # keep each masked target row a probability distribution.
        batch, value_log_probs, isr, adv_taken, config = ex_target_inputs
        targets, _ = compute_player_targets(
            batch, value_log_probs, isr, adv_taken, config
        )
        sums = np.asarray(targets.win_returns).sum(-1)
        mask = np.asarray(targets.value_mask)
        np.testing.assert_allclose(sums[mask], 1.0, atol=1e-4)
        np.testing.assert_allclose(sums[~mask], 0.0, atol=1e-6)

    def test_on_policy_diagnostics(self, ex_target_inputs):
        batch, value_log_probs, isr, adv_taken, config = ex_target_inputs
        _, logs = compute_player_targets(batch, value_log_probs, isr, adv_taken, config)
        # isr == 1 everywhere: full effective sample size, nothing clipped.
        np.testing.assert_allclose(float(logs["player_isr_ess"]), 1.0, atol=1e-3)
        np.testing.assert_allclose(float(logs["player_rho_clip_frac"]), 0.0)


class TestAdvantageShift:
    """The signed measure that carries a scalar advantage into the
    distributional value recursion (targets.advantage_shift)."""

    support = jnp.array([-1.0, 0.0, 1.0])

    def _probs(self):
        return jnp.array(
            [
                [1 / 3, 1 / 3, 1 / 3],  # mean 0
                [0.0, 0.02, 0.98],  # near-won
                [0.5, 0.0, 0.5],  # coin flip, mean 0
                [0.9, 0.1, 0.0],  # near-lost
            ]
        )

    def test_zero_advantage_is_exactly_the_identity(self):
        from rl.online.training.targets import advantage_shift

        got = advantage_shift(self._probs(), jnp.zeros(4), self.support)
        np.testing.assert_array_equal(np.asarray(got), np.zeros((4, 3), np.float32))

    def test_zero_total_mass_and_exact_first_moment(self):
        from rl.online.training.targets import advantage_shift

        probs = self._probs()
        adv = jnp.array([0.3, -0.3, 0.45, 0.15])
        shift = advantage_shift(probs, adv, self.support)
        # Adding it must not create or destroy probability mass, so the
        # value targets stay normalised however large the advantage.
        np.testing.assert_allclose(np.asarray(shift).sum(-1), np.zeros(4), atol=1e-6)
        # ...and it must move the mean by exactly the advantage.
        np.testing.assert_allclose(
            np.asarray((probs + shift) @ self.support),
            np.asarray(probs @ self.support + adv),
            atol=1e-6,
        )

    def test_saturates_at_the_support_edge_rather_than_exploding(self):
        from rl.online.training.targets import advantage_shift

        probs = self._probs()
        huge = advantage_shift(probs, jnp.full((4,), 50.0), self.support)
        shifted_mean = np.asarray((probs + huge) @ self.support)
        np.testing.assert_allclose(shifted_mean, np.ones(4), atol=1e-6)

    def test_moves_mass_to_the_adjacent_atom_not_the_far_one(self):
        """The reason for two_hot over a fixed (-1/2, 0, +1/2) direction: a
        slightly worse action at a near-won state means 'more likely a
        draw', not 'more likely a loss'."""
        from rl.online.training.targets import advantage_shift

        near_won = jnp.array([[0.0, 0.02, 0.98]])
        shifted = np.asarray(
            near_won + advantage_shift(near_won, jnp.array([-0.3]), self.support)
        )
        assert shifted[0, 0] == pytest.approx(0.0, abs=1e-6)  # no loss mass added
        assert shifted[0, 1] > 0.02  # the draw atom is where it goes


class TestRetraceBaseline:
    """Retrace over the composed Q: the value target's baseline is
    Q(s, a) = V(s) + A(s, a), not V(s)."""

    def _expectation(self, targets, support):
        return np.asarray(targets.win_returns @ support)

    def test_zero_baseline_reproduces_the_vtrace_targets(self, ex_target_inputs):
        """The flat-at-init advantage head must leave the estimator BITWISE
        where it was — checked against a reference implementation of the
        pre-Retrace v-trace form, not against another call of the same
        function."""
        batch, value_log_probs, isr, adv_taken, config = ex_target_inputs
        done = batch.player_transitions.env_output.done[..., None]
        mask = 1 - (jnp.cumsum(done, axis=0) - done)
        discount_t = (1 - done) * config.player_gamma * mask
        rho = jnp.minimum(1.0, isr)[..., None]
        v_tm1 = jnp.exp(value_log_probs)
        v_t = jnp.concatenate([v_tm1[1:], v_tm1[-1:]], axis=0)
        td = (
            rho
            * mask
            * (
                batch.player_transitions.env_output.win_reward
                + discount_t * v_t
                - v_tm1
            )
        )
        want = (vtrace(td, discount_t, rho * config.player_lambda) + v_tm1) * mask

        got, _ = compute_player_targets(
            batch, value_log_probs, isr, jnp.zeros_like(adv_taken), config
        )
        np.testing.assert_array_equal(np.asarray(got.win_returns), np.asarray(want))

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

    def test_targets_stay_normalised_under_a_nonzero_baseline(self, ex_target_inputs):
        batch, value_log_probs, isr, adv_taken, config = ex_target_inputs
        targets, _ = compute_player_targets(
            batch, value_log_probs, isr, jnp.full_like(adv_taken, 0.4), config
        )
        sums = np.asarray(targets.win_returns).sum(-1)
        mask = np.asarray(targets.value_mask)
        np.testing.assert_allclose(sums[mask], 1.0, atol=1e-4)


class TestRNaDTransform:
    """R-NaD on Retrace (2026-08-22): the reward transform against the
    reference policy enters the Q critic's bootstrap as -eta*KL and the
    policy advantage per legal cell as -eta*(log pi - log pi_reg)."""

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

    @staticmethod
    def _starved_row():
        legal = jnp.ones((1, 4), dtype=bool)
        pi = jnp.asarray([[0.9999 - 2e-4, 1e-4, 1e-4, 1e-4]])
        pi = pi / pi.sum()
        log_ref = jnp.log(jnp.full((1, 4), 0.25))
        return legal, pi, log_ref

    @staticmethod
    def _penalised(pi, log_ref, legal, eta, eta_ent):
        from rl.online.training.targets import reference_penalty

        ref, ent = reference_penalty(jnp.log(pi), log_ref, legal, eta, eta_ent)
        return jnp.where(legal, jnp.zeros_like(ref) - ref - ent, 0.0)

    def test_starved_cell_gets_an_unbounded_upward_push(self):
        """The property the transform is bought for: with A flat, a cell
        at pi ~ 1e-4 against a reference of 0.25 carries +eta*log(2500)
        ~ +1.56 of advantage at eta 0.2 — no pi prefactor anywhere."""
        legal, pi, log_ref = self._starved_row()
        q_reg = self._penalised(pi, log_ref, legal, eta=0.2, eta_ent=0.0)
        adv = q_reg - (pi * q_reg).sum(axis=-1, keepdims=True)
        assert float(adv[0, 1]) > 1.5
        # And the dominant cell's value is marked DOWN by
        # eta*log(pi/pi_ref) ~ 0.28 (its advantage is ~0: the baseline is
        # almost entirely that cell).
        assert float(q_reg[0, 0]) < -0.25
        # eta 0 is the identity.
        np.testing.assert_allclose(
            np.asarray(self._penalised(pi, log_ref, legal, eta=0.0, eta_ent=0.0)),
            0.0,
            atol=1e-7,
        )

    def test_entropy_term_is_the_uniform_reference(self):
        """eta_ent*log pi(a) alone: exactly the reference penalty with
        pi_reg uniform, so it must reproduce that arm cell for cell up to
        the log N constant NeuRD's centring drops."""
        legal, pi, log_ref = self._starved_row()  # log_ref IS uniform here
        ent_only = self._penalised(pi, log_ref, legal, eta=0.0, eta_ent=0.2)
        ref_uniform = self._penalised(pi, log_ref, legal, eta=0.2, eta_ent=0.0)
        offset = np.asarray(ent_only - ref_uniform)
        np.testing.assert_allclose(offset, offset[0, 0], atol=1e-6)

    def test_entropy_term_alone_restores_a_starved_cell(self):
        """Positive control for the term landing at all: with no reference
        pressure, entropy alone must still push the starved cell up, and
        with no pi prefactor the push must GROW as pi(a) shrinks."""
        legal, pi, log_ref = self._starved_row()

        def starved_push(mass):
            probs = jnp.asarray([[1.0 - 3 * mass, mass, mass, mass]])
            q_reg = self._penalised(probs, log_ref, legal, eta=0.0, eta_ent=0.05)
            adv = q_reg - (probs * q_reg).sum(axis=-1, keepdims=True)
            return float(adv[0, 1])

        assert starved_push(1e-2) > 0.0
        assert starved_push(1e-4) > starved_push(1e-2)
        np.testing.assert_allclose(
            np.asarray(self._penalised(pi, log_ref, legal, eta=0.0, eta_ent=0.0)),
            0.0,
            atol=1e-7,
        )
