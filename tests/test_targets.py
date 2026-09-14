"""v-trace return math on hand-checkable inputs, plus the full target
pipeline on the real example trajectory bundled in rl/environment/ex.bin."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.environment.interfaces import Batch, PlayerTargets
from rl.online.config import Porygon2LearnerConfig
from rl.online.training.targets import (
    compute_player_targets,
    scalar_vtrace,
    two_hot,
    vtrace,
)


def naive_vtrace(
    td_errors: np.ndarray, discount_t: np.ndarray, c_tm1: np.ndarray
) -> np.ndarray:
    """Reference implementation: plain reverse recursion."""
    T = td_errors.shape[0]
    out = np.zeros_like(td_errors)
    acc = np.zeros_like(td_errors[0])
    for t in reversed(range(T)):
        acc = td_errors[t] + discount_t[t] * c_tm1[t] * acc
        out[t] = acc
    return out


def test_vtrace_matches_naive_recursion() -> None:
    rng = np.random.default_rng(0)
    td = rng.normal(size=(12, 4, 3)).astype(np.float32)
    disc = rng.uniform(0.0, 1.0, size=(12, 4, 3)).astype(np.float32)
    c = rng.uniform(0.0, 1.0, size=(12, 4, 3)).astype(np.float32)

    got = np.asarray(vtrace(jnp.asarray(td), jnp.asarray(disc), jnp.asarray(c)))
    want = naive_vtrace(td, disc, c)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_vtrace_zero_trace_is_identity() -> None:
    td = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    disc = jnp.ones_like(td)
    got = vtrace(td, disc, jnp.zeros_like(td))
    np.testing.assert_allclose(np.asarray(got), np.asarray(td))


def test_vtrace_bf16_inputs_do_not_break_the_scan_carry() -> None:
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
    def test_bin_centres_are_one_hot(self) -> None:
        support = jnp.array([-1.0, 0.0, 1.0])
        got = two_hot(jnp.array([-1.0, 0.0, 1.0]), support)
        np.testing.assert_allclose(
            np.asarray(got), np.eye(3, dtype=np.float32), atol=1e-6
        )

    def test_interpolates_and_clips(self) -> None:
        support = jnp.array([-1.0, 0.0, 1.0])
        got = np.asarray(two_hot(jnp.array([0.8, -0.25, 2.0, -3.0]), support))
        np.testing.assert_allclose(got[0], [0.0, 0.2, 0.8], atol=1e-6)
        np.testing.assert_allclose(got[1], [0.25, 0.75, 0.0], atol=1e-6)
        np.testing.assert_allclose(got[2], [0.0, 0.0, 1.0], atol=1e-6)  # clip hi
        np.testing.assert_allclose(got[3], [1.0, 0.0, 0.0], atol=1e-6)  # clip lo
        np.testing.assert_allclose(got.sum(-1), 1.0, atol=1e-6)


@pytest.fixture(scope="module")
def ex_target_inputs() -> tuple[Batch, jax.Array, jax.Array, Porygon2LearnerConfig]:
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
    def test_shapes_and_finiteness(
        self,
        ex_target_inputs: tuple[Batch, jax.Array, jax.Array, Porygon2LearnerConfig],
    ) -> None:
        batch, value_log_probs, isr, config = ex_target_inputs
        T, B = batch.player_transitions.env_output.done.shape
        targets, _ = compute_player_targets(batch, value_log_probs, isr, config)

        assert targets.win_returns.shape == (T, B, 3)
        assert targets.policy_mask.shape == (T, B)
        assert targets.value_mask.shape == (T, B)
        assert np.isfinite(np.asarray(targets.win_returns)).all()

    def test_masks_follow_episode_structure(
        self,
        ex_target_inputs: tuple[Batch, jax.Array, jax.Array, Porygon2LearnerConfig],
    ) -> None:
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

    def test_value_targets_stay_distributions(
        self,
        ex_target_inputs: tuple[Batch, jax.Array, jax.Array, Porygon2LearnerConfig],
    ) -> None:
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

    def test_on_policy_diagnostics(
        self,
        ex_target_inputs: tuple[Batch, jax.Array, jax.Array, Porygon2LearnerConfig],
    ) -> None:
        batch, value_log_probs, isr, config = ex_target_inputs
        _, logs = compute_player_targets(batch, value_log_probs, isr, config)
        # isr == 1 everywhere: full effective sample size, nothing clipped.
        np.testing.assert_allclose(float(logs["player_isr_ess"]), 1.0, atol=1e-3)
        np.testing.assert_allclose(float(logs["player_rho_clip_frac"]), 0.0)


def _min_batch(
    done: jax.Array,
    win_reward: jax.Array,
    action_mask: jax.Array,
    action_index: jax.Array,
) -> Batch:
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
    """V-trace continuation and actor importance weights on short episodes."""

    def _targets(
        self,
        done: jax.Array,
        win_reward: jax.Array,
        isr: jax.Array | None = None,
        player_lambda: float = 0.8,
    ) -> PlayerTargets:
        from rl.online.config import Porygon2LearnerConfig

        T, B = done.shape
        batch = _min_batch(
            done, win_reward, jnp.ones((T, B, 4), bool), jnp.zeros((T, B), jnp.int32)
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

    def test_matches_hand_recursion(self) -> None:
        done = jnp.array([[False], [False], [True]])
        win_reward = jnp.zeros((3, 1, 3), dtype=jnp.float32).at[:, :, 1].set(1.0)
        win_reward = win_reward.at[2, :, :].set(jnp.array([0.0, 0.0, 1.0]))
        targets = self._targets(done, win_reward)
        assert targets.pg_advantages.dtype == jnp.float32
        np.testing.assert_allclose(
            np.asarray(targets.pg_advantages[:, 0]), [0.8, 1.0, 1.0], atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(targets.win_returns[:, 0]) @ np.array([-1.0, 0.0, 1.0]),
            [0.64, 0.8, 1.0],
            atol=1e-6,
        )

    def test_zero_lambda_still_uses_next_corrected_value_for_actor(self) -> None:
        done = jnp.array([[False], [False], [True]])
        win_reward = jnp.zeros((3, 1, 3), dtype=jnp.float32).at[:, :, 1].set(1.0)
        win_reward = win_reward.at[2, :, :].set(jnp.array([0.0, 0.0, 1.0]))
        targets = self._targets(done, win_reward, player_lambda=0.0)
        np.testing.assert_allclose(
            np.asarray(targets.pg_advantages[:, 0]), [0.0, 1.0, 1.0], atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(targets.win_returns[:, 0]) @ np.array([-1.0, 0.0, 1.0]),
            [0.0, 0.0, 1.0],
            atol=1e-6,
        )

    def test_rho_truncation_attenuates_the_advantage(self) -> None:
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
        assert float(half.pg_advantages[1, 0]) == pytest.approx(0.5, abs=1e-6)
        np.testing.assert_array_equal(half.win_returns[2, 0], [0.0, 0.0, 1.0])


def test_nonterminal_chunk_tail_only_supplies_its_bootstrap_value() -> None:
    done = jnp.zeros((2, 1), dtype=bool)
    win_reward = jnp.tile(jnp.array([0.0, 1.0, 0.0]), (2, 1, 1))
    batch = _min_batch(
        done,
        win_reward,
        jnp.ones((2, 1, 2), dtype=bool),
        jnp.zeros((2, 1), dtype=jnp.int32),
    )
    value_probs = jnp.array([[[0.0, 1.0, 0.0]], [[0.0, 0.2, 0.8]]])
    targets, _ = compute_player_targets(
        batch,
        jnp.log(value_probs),
        jnp.ones((2, 1)),
        Porygon2LearnerConfig(player_gamma=0.5, player_lambda=0.8),
    )
    np.testing.assert_allclose(targets.pg_advantages[:, 0], [0.4, 0.0], atol=1e-6)
    np.testing.assert_allclose(targets.win_returns[0, 0], [0.0, 0.6, 0.4], atol=1e-6)
    np.testing.assert_array_equal(targets.win_returns[1, 0], 0.0)
    np.testing.assert_array_equal(targets.policy_mask[:, 0], [True, False])
    np.testing.assert_array_equal(targets.value_mask[:, 0], [True, False])


def test_scalar_targets_and_actor_advantages_are_detached_and_f32() -> None:
    def objective(value, ratio):
        returns, advantages = scalar_vtrace(
            jnp.array([0.0, 1.0], dtype=jnp.bfloat16),
            value,
            jnp.array([1.0, 0.0], dtype=jnp.bfloat16),
            jnp.ones(2, dtype=jnp.bfloat16),
            ratio,
            ratio,
            0.8,
        )
        assert returns.dtype == advantages.dtype == jnp.float32
        return returns.sum() + advantages.sum()

    values = jnp.array([0.2, 0.4], dtype=jnp.bfloat16)
    ratios = jnp.array([0.5, 1.0], dtype=jnp.float32)
    value_grad, ratio_grad = jax.jit(jax.grad(objective, argnums=(0, 1)))(
        values, ratios
    )
    np.testing.assert_array_equal(value_grad, 0.0)
    np.testing.assert_array_equal(ratio_grad, 0.0)
    assert float(objective(values + 0.1, ratios)) != float(objective(values, ratios))
    assert float(objective(values, ratios * 0.5)) != float(objective(values, ratios))


class TestMagnetKl:
    """KL(live || detached reference), including collapsed policy tails."""

    def test_reference_kl_direction_and_illegal_cells(self) -> None:
        from rl.online.training.targets import reference_kl

        legal = jnp.asarray([[True, True, True, False]])
        policy = np.asarray([0.7, 0.2, 0.1])
        reference = np.asarray([0.2, 0.3, 0.5])
        log_policy = jnp.log(jnp.asarray([[*policy, 1.0]]))
        log_reference = jnp.log(jnp.asarray([[*reference, 1.0]]))
        np.testing.assert_allclose(
            reference_kl(log_policy, log_policy, legal), 0.0, atol=1e-6
        )
        actual = jax.jit(reference_kl)(log_policy, log_reference, legal)
        expected = np.sum(policy * np.log(policy / reference))
        forward = np.sum(reference * np.log(reference / policy))
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        assert abs(float(actual[0]) - forward) > 0.02
        contaminated = jax.jit(reference_kl)(
            log_policy.at[0, 3].set(jnp.nan),
            log_reference.at[0, 3].set(jnp.inf),
            legal,
        )
        np.testing.assert_array_equal(contaminated, actual)

    def test_logit_gradient_matches_reverse_kl_and_vanishes_near_collapse(self) -> None:
        from rl.online.training.targets import reference_kl

        legal = jnp.array([[True, True, False, True], [True, True, False, True]])
        logits = jnp.array([[10000.0, -10000.0, 7.0, 0.0], [0.0, 1.0, 8.0, -1.0]])
        reference = jnp.array([[0.2, 0.3, 0.0, 0.5], [0.5, 0.2, 0.0, 0.3]])

        def objective(policy_logits):
            log_policy = jax.nn.log_softmax(
                jnp.where(legal, policy_logits, -jnp.inf), axis=-1
            )
            return reference_kl(log_policy, jnp.log(reference), legal).sum()

        loss, gradient = jax.jit(jax.value_and_grad(objective))(logits)
        policy = jax.nn.softmax(jnp.where(legal, logits, -jnp.inf), axis=-1)
        log_policy = jax.nn.log_softmax(jnp.where(legal, logits, -jnp.inf), axis=-1)
        log_ratio = jnp.where(legal, log_policy - jnp.log(reference), 0.0)
        divergence = (policy * log_ratio).sum(axis=-1, keepdims=True)
        expected = policy * (log_ratio - divergence)
        np.testing.assert_allclose(gradient, expected, atol=1e-6)
        np.testing.assert_allclose(gradient.sum(axis=-1), 0.0, atol=1e-6)
        np.testing.assert_array_equal(gradient[~legal], 0.0)
        assert np.isfinite(loss)
        np.testing.assert_array_equal(gradient[0], 0.0)
        assert np.max(np.abs(gradient[1])) > 0.1

    def test_reference_has_no_gradient_but_changes_the_loss(self) -> None:
        from rl.online.training.targets import reference_kl

        legal = jnp.array([[True, True, False]])
        live = jnp.array([[1.0, -2.0, 4.0]])
        reference = jnp.array([[0.0, 1.0, -3.0]])

        def objective(policy_logits, reference_logits):
            return reference_kl(
                jax.nn.log_softmax(policy_logits),
                jax.nn.log_softmax(reference_logits),
                legal,
            ).sum()

        live_gradient, reference_gradient = jax.jit(
            jax.grad(objective, argnums=(0, 1))
        )(live, reference)
        np.testing.assert_array_equal(reference_gradient, 0.0)
        np.testing.assert_allclose(
            live_gradient[0, :2], [0.18070664, -0.18070664], atol=1e-6
        )
        assert float(objective(live, reference)) > float(objective(live, live)) + 0.1

    def test_empty_and_singleton_rows_have_no_loss_or_gradient(self) -> None:
        from rl.online.training.targets import reference_kl

        legal = jnp.array([[False, False, False], [True, False, False]])
        log_policy = jnp.array([[jnp.nan, jnp.inf, -jnp.inf], [-2.0, jnp.nan, jnp.inf]])
        log_reference = jnp.array(
            [[jnp.inf, jnp.nan, -jnp.inf], [-0.2, jnp.inf, jnp.nan]]
        )

        def objective(values):
            return reference_kl(values, log_reference, legal).sum()

        loss, gradient = jax.jit(jax.value_and_grad(objective))(log_policy)
        np.testing.assert_array_equal(loss, 0.0)
        np.testing.assert_array_equal(gradient, 0.0)

    def test_bf16_inputs_are_renormalised_and_accumulated_in_f32(self) -> None:
        from rl.online.training.targets import reference_kl

        legal = jnp.ones((1, 3), dtype=bool)
        log_policy = jnp.log(jnp.array([[0.7, 0.2, 0.1]])).astype(jnp.bfloat16)
        log_reference = jnp.log(jnp.array([[0.2, 0.3, 0.5]])).astype(jnp.bfloat16)
        policy = np.exp(np.asarray(log_policy, dtype=np.float64))
        reference = np.exp(np.asarray(log_reference, dtype=np.float64))
        policy /= policy.sum(axis=-1, keepdims=True)
        reference /= reference.sum(axis=-1, keepdims=True)
        actual = jax.jit(reference_kl)(log_policy, log_reference, legal)
        assert actual.dtype == jnp.float32
        np.testing.assert_allclose(
            actual, np.sum(policy * np.log(policy / reference), axis=-1), rtol=1e-6
        )
