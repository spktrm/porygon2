"""The player policy surrogate: PPO clip semantics, the spo/ppo selector,
and the clip-fraction readout. The differentiated magnet/entropy terms are
covered in test_targets.py (reference_kl) and the train_step smoke."""

import jax
import jax.numpy as jnp
import numpy as np

from rl.online.training.loss import (
    clip_fraction,
    policy_gradient_loss,
    ppo_objective,
    spo_objective,
)


class TestPpoObjective:
    def _grad_wrt_ratio(self, ratio, adv, clip):
        def objective(r):
            return ppo_objective(
                policy_ratios=r, advantages=jnp.asarray(adv), clip_ppo=clip
            ).sum()

        return float(jax.grad(objective)(jnp.asarray(ratio)))

    def test_gradient_is_the_advantage_inside_the_band(self):
        assert self._grad_wrt_ratio(1.0, 0.7, 0.2) == np.float32(0.7)
        assert self._grad_wrt_ratio(1.1, -0.3, 0.2) == np.float32(-0.3)

    def test_gradient_is_zero_outside_the_band_in_the_push_direction(self):
        # A > 0 pushing the ratio further up past 1+eps: clipped, no force.
        assert self._grad_wrt_ratio(1.5, 0.7, 0.2) == 0.0
        # A < 0 pushing the ratio further down past 1-eps: clipped.
        assert self._grad_wrt_ratio(0.5, -0.7, 0.2) == 0.0

    def test_pessimism_keeps_the_corrective_gradient(self):
        # Positive control for the one-sided min: outside the band but with
        # the advantage pointing BACK toward it, the raw term is the lower
        # bound and its gradient survives.
        assert self._grad_wrt_ratio(1.5, -0.7, 0.2) == np.float32(-0.7)
        assert self._grad_wrt_ratio(0.5, 0.7, 0.2) == np.float32(0.7)

    def test_on_policy_value_is_the_advantage(self):
        got = ppo_objective(
            policy_ratios=jnp.ones(3),
            advantages=jnp.asarray([0.5, -0.2, 0.0]),
            clip_ppo=0.2,
        )
        np.testing.assert_allclose(np.asarray(got), [0.5, -0.2, 0.0])


class TestObjectiveSelector:
    ratios = jnp.asarray([0.6, 1.0, 1.4])
    advantages = jnp.asarray([0.3, -0.5, 0.2])
    valid = jnp.ones(3, dtype=bool)

    def _loss(self, objective):
        return float(
            policy_gradient_loss(
                policy_ratios=self.ratios,
                advantages=self.advantages,
                valid=self.valid,
                threshold=0.2,
                objective=objective,
            )
        )

    def test_dispatches_to_each_objective(self):
        want_ppo = -float(
            ppo_objective(
                policy_ratios=self.ratios, advantages=self.advantages, clip_ppo=0.2
            ).mean()
        )
        want_spo = -float(
            spo_objective(
                policy_ratios=self.ratios, advantages=self.advantages, clip_ppo=0.2
            ).mean()
        )
        assert self._loss("ppo") == want_ppo
        assert self._loss("spo") == want_spo
        # Positive control: the two objectives genuinely differ here.
        assert want_ppo != want_spo


def test_clip_fraction_counts_rows_outside_the_band():
    ratios = jnp.asarray([1.0, 1.19, 1.3, 0.7])
    got = clip_fraction(
        policy_ratios=ratios, valid=jnp.ones(4, dtype=bool), clip_ppo=0.2
    )
    assert float(got) == 0.5
