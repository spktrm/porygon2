"""The player's APPO actor: the clipped target ratio, checked against RLlib's
appo_torch_policy.loss algebra by hand, and SPO's quadratic surrogate on it."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.online.training.loss import appo_policy_loss, clipped_target_ratio


def _ratio(learner, behaviour, old, cap=2.0):
    return clipped_target_ratio(
        learner_log_prob=jnp.log(jnp.asarray(learner)),
        behaviour_log_prob=jnp.log(jnp.asarray(behaviour)),
        old_policy_log_prob=jnp.log(jnp.asarray(old)),
        behaviour_ratio_clip=cap,
    )


class TestClippedTargetRatio:
    def test_inside_the_cap_mu_cancels_to_live_over_old(self) -> None:
        got = _ratio([0.3, 0.1], [0.2, 0.25], [0.4, 0.2])
        np.testing.assert_allclose(np.asarray(got), [0.75, 0.5], rtol=1e-6)

    def test_past_the_cap_the_row_is_scaled_by_cap_times_old_over_mu(self) -> None:
        # mu/pi_old = 5 is capped at 2, so the ratio is 2 * pi_live / mu
        # rather than pi_live / pi_old (positive control: they differ).
        got = float(_ratio([0.3], [0.5], [0.1])[0])
        assert got == pytest.approx(2.0 * 0.3 / 0.5, rel=1e-6)
        assert got != pytest.approx(0.3 / 0.1, rel=1e-3)
        uncapped = float(_ratio([0.3], [0.5], [0.1], cap=10.0)[0])
        assert uncapped == pytest.approx(0.3 / 0.1, rel=1e-6)

    def test_only_the_learner_term_carries_gradient(self) -> None:
        def objective(learner, behaviour, old):
            return clipped_target_ratio(
                learner_log_prob=learner,
                behaviour_log_prob=behaviour,
                old_policy_log_prob=old,
                behaviour_ratio_clip=2.0,
            ).sum()

        grads = jax.jit(jax.grad(objective, argnums=(0, 1, 2)))(
            jnp.log(jnp.array([0.3])),
            jnp.log(jnp.array([0.2])),
            jnp.log(jnp.array([0.4])),
        )
        assert float(grads[0][0]) == pytest.approx(0.75, rel=1e-6)
        np.testing.assert_array_equal(grads[1], 0.0)
        np.testing.assert_array_equal(grads[2], 0.0)

    def test_bf16_inputs_are_promoted_to_f32(self) -> None:
        got = clipped_target_ratio(
            learner_log_prob=jnp.array([-1.0], dtype=jnp.bfloat16),
            behaviour_log_prob=jnp.array([-1.0], dtype=jnp.bfloat16),
            old_policy_log_prob=jnp.array([-1.0], dtype=jnp.bfloat16),
            behaviour_ratio_clip=2.0,
        )
        assert got.dtype == jnp.float32
        np.testing.assert_allclose(got, 1.0)


class TestAppoPolicyLoss:
    def _grad_wrt_learner(self, learner, behaviour, old, advantage, clip=0.4):
        def objective(learner_log_prob):
            return appo_policy_loss(
                learner_log_prob=learner_log_prob,
                behaviour_log_prob=jnp.log(jnp.array([behaviour])),
                old_policy_log_prob=jnp.log(jnp.array([old])),
                advantages=jnp.array([advantage]),
                valid=jnp.array([True]),
                clip_ppo=clip,
                behaviour_ratio_clip=2.0,
            )

        return float(jax.jit(jax.grad(objective))(jnp.log(jnp.array([learner])))[0])

    @staticmethod
    def _spo_grad(ratio, advantage, eps):
        # d/dlogpi of -(r A - |A| (r-1)^2 / 2eps) with dr/dlogpi = r.
        return -ratio * (advantage - abs(advantage) * (ratio - 1) / eps)

    def test_at_the_snapshot_inside_the_cap_the_gradient_is_the_score(self) -> None:
        # pi_live == pi_old and mu/pi_old < 2: r == 1, the quadratic is
        # flat, so d(-r A)/dlogpi = -A. This is the loss a snap-every-update
        # config reduces to.
        assert self._grad_wrt_learner(0.3, 0.2, 0.3, 0.5) == pytest.approx(
            -0.5, rel=1e-6
        )

    def test_the_cap_places_the_ratio_below_one_and_spo_pulls_it_up(self) -> None:
        # pi_live == pi_old, mu/pi_old = 5 > 2: r = 2 pi_old/mu = 0.4, below
        # 1 - eps. PPO's min would keep -r A = -0.2; SPO adds the restoring
        # term |A| r (1 - r)/eps = 0.3 toward the band, and with A < 0 that
        # term is the ENTIRE force and still raises pi_live.
        assert self._grad_wrt_learner(0.1, 0.5, 0.1, 0.5) == pytest.approx(
            self._spo_grad(0.4, 0.5, 0.4), rel=1e-6
        )
        assert self._grad_wrt_learner(0.1, 0.5, 0.1, 0.5) == pytest.approx(
            -0.5, rel=1e-6
        )
        assert self._grad_wrt_learner(0.1, 0.5, 0.1, -0.5) == pytest.approx(
            self._spo_grad(0.4, -0.5, 0.4), rel=1e-6
        )
        assert self._grad_wrt_learner(0.1, 0.5, 0.1, -0.5) < 0.0

    def test_past_the_band_the_quadratic_reverses_the_push(self) -> None:
        # pi_live/pi_old = 1.5 with A > 0: past 1 + 0.4, the sign flips and
        # the row pulls back toward the edge where PPO's clip would go flat.
        assert self._grad_wrt_learner(0.3, 0.2, 0.2, 0.5) == pytest.approx(
            self._spo_grad(1.5, 0.5, 0.4), rel=1e-6
        )
        assert self._grad_wrt_learner(0.3, 0.2, 0.2, 0.5) > 0.0
        # Same ratio with A < 0: advantage and restoring term agree.
        assert self._grad_wrt_learner(0.3, 0.2, 0.2, -0.5) == pytest.approx(
            self._spo_grad(1.5, -0.5, 0.4), rel=1e-6
        )
        # Positive control: a wider band puts 1.5 inside it and the A > 0
        # gradient keeps its sign.
        assert self._grad_wrt_learner(0.3, 0.2, 0.2, 0.5, clip=0.6) == pytest.approx(
            self._spo_grad(1.5, 0.5, 0.6), rel=1e-6
        )
        assert self._grad_wrt_learner(0.3, 0.2, 0.2, 0.5, clip=0.6) < 0.0

    def test_masked_rows_and_empty_batches_are_inert(self) -> None:
        learner = jnp.array([-0.5, jnp.nan, jnp.inf])
        behaviour = jnp.array([-0.5, jnp.inf, jnp.nan])
        old = jnp.array([-0.5, jnp.nan, -jnp.inf])
        advantages = jnp.array([0.4, jnp.inf, jnp.nan])

        def objective(learner_log_prob, mask):
            return appo_policy_loss(
                learner_log_prob=learner_log_prob,
                behaviour_log_prob=behaviour,
                old_policy_log_prob=old,
                advantages=advantages,
                valid=mask,
                clip_ppo=0.4,
                behaviour_ratio_clip=2.0,
            )

        valid = jnp.array([True, False, False])
        loss, gradient = jax.jit(jax.value_and_grad(objective))(learner, valid)
        np.testing.assert_allclose(loss, -0.4, rtol=1e-6)
        np.testing.assert_allclose(gradient, [-0.4, 0.0, 0.0], rtol=1e-6)
        empty_loss, empty_gradient = jax.jit(jax.value_and_grad(objective))(
            learner, jnp.zeros(3, dtype=bool)
        )
        np.testing.assert_array_equal(empty_loss, 0.0)
        np.testing.assert_array_equal(empty_gradient, 0.0)

    def test_advantage_is_detached_and_the_loss_is_f32(self) -> None:
        def objective(advantages):
            return appo_policy_loss(
                learner_log_prob=jnp.array([-1.0], dtype=jnp.bfloat16),
                behaviour_log_prob=jnp.array([-1.0], dtype=jnp.bfloat16),
                old_policy_log_prob=jnp.array([-1.0], dtype=jnp.bfloat16),
                advantages=advantages,
                valid=jnp.array([True]),
                clip_ppo=0.4,
                behaviour_ratio_clip=2.0,
            )

        advantages = jnp.array([0.25], dtype=jnp.bfloat16)
        loss, gradient = jax.jit(jax.value_and_grad(objective))(advantages)
        assert loss.dtype == jnp.float32
        np.testing.assert_allclose(loss, -0.25)
        np.testing.assert_array_equal(gradient, 0.0)
