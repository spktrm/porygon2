"""The player policy surrogate: PPO clip semantics, the spo/ppo selector,
and the clip-fraction readout. The differentiated magnet/entropy terms are
covered in test_targets.py (reference_kl) and the train_step smoke."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


class TestFactorisedEntropies:
    """The Oct–Nov-form per-level entropy: bookkeeping identity against the
    joint H (documents exactly what the unit-weight form changes), and the
    two mask-semantics edges."""

    def _setup(self):
        from rl.environment.data import CELL_MODALITY_MASK
        from rl.environment.protos.service_pb2 import ModalityEnum

        flat = np.asarray(CELL_MODALITY_MASK)
        legal = np.zeros(flat.shape[0], dtype=bool)
        move_cells = np.flatnonzero(flat == ModalityEnum.MODALITY_ENUM__MOVE)[:4]
        switch_cells = np.flatnonzero(flat == ModalityEnum.MODALITY_ENUM__SWITCH)[:3]
        legal[move_cells] = True
        legal[switch_cells] = True
        rng = np.random.default_rng(3)
        logits = np.where(legal, rng.normal(size=legal.shape), -1e9)
        log_policy = jnp.asarray(
            jax.nn.log_softmax(jnp.asarray(logits, dtype=jnp.float32))
        )
        return flat, jnp.asarray(legal), log_policy, move_cells, switch_cells

    def test_joint_entropy_decomposition(self):
        from rl.online.training.loss import factorised_entropies

        flat, legal, log_policy, move_cells, switch_cells = self._setup()
        probs = np.where(np.asarray(legal), np.exp(np.asarray(log_policy)), 0.0)
        h_joint = -float((probs[probs > 0] * np.log(probs[probs > 0])).sum())

        h_macro_m, h_micro_move = factorised_entropies(
            log_policy, jnp.asarray(flat[move_cells[0]]), legal
        )
        h_macro_s, h_micro_switch = factorised_entropies(
            log_policy, jnp.asarray(flat[switch_cells[0]]), legal
        )
        # H_macro is taken-independent.
        np.testing.assert_allclose(float(h_macro_m), float(h_macro_s), atol=1e-6)
        p_move = float(probs[move_cells].sum())
        p_switch = float(probs[switch_cells].sum())
        # The helper returns NORMALISED entropies (fraction of each level's
        # own log k — the k-weighting catch); un-normalise to state the
        # bookkeeping identity H_joint = H_macro + sum_m pi_m * H(micro|m),
        # whose pi_m prefactors are exactly what the unit-weight per-level
        # form removes.
        raw_macro = float(h_macro_m) * np.log(2)  # two live modalities
        raw_move = float(h_micro_move) * np.log(len(move_cells))
        raw_switch = float(h_micro_switch) * np.log(len(switch_cells))
        recomposed = raw_macro + p_move * raw_move + p_switch * raw_switch
        np.testing.assert_allclose(recomposed, h_joint, rtol=1e-5)
        # Positive control: with pi_switch < 1 the unit-weight sum genuinely
        # exceeds the joint's weighted one on the switch axis.
        assert p_switch < 1.0
        assert raw_switch > p_switch * raw_switch

    def test_singleton_and_uniform_edges(self):
        from rl.online.training.loss import factorised_entropies

        flat, legal, log_policy, move_cells, switch_cells = self._setup()
        # Singleton taken modality: micro entropy exactly 0.
        single = np.asarray(legal).copy()
        single[switch_cells[1:]] = False
        _, h_micro = factorised_entropies(
            log_policy, jnp.asarray(flat[switch_cells[0]]), jnp.asarray(single)
        )
        np.testing.assert_allclose(float(h_micro), 0.0, atol=1e-6)
        # Uniform within the taken modality: normalised micro entropy = 1
        # regardless of the modality's size — the k-independence the
        # normalisation exists to provide.
        uniform_logits = jnp.where(jnp.asarray(np.asarray(legal)), 0.0, -1e9)
        uniform = jax.nn.log_softmax(uniform_logits.astype(jnp.float32))
        _, h_uni_switch = factorised_entropies(
            uniform, jnp.asarray(flat[switch_cells[0]]), legal
        )
        _, h_uni_move = factorised_entropies(
            uniform, jnp.asarray(flat[move_cells[0]]), legal
        )
        np.testing.assert_allclose(float(h_uni_switch), 1.0, rtol=1e-5)
        np.testing.assert_allclose(float(h_uni_move), 1.0, rtol=1e-5)


class TestUniformKlModalities:
    """The zero-avoiding term on the MODALITY MARGINAL: the WHETHER/WHICH
    split as an algebraic identity, with the row-form failure (sp75c: mass
    bought by flattening the move row) as the thing made unrepresentable."""

    def _legal_and_logits(self, starve_switch=False):
        from rl.environment.data import CELL_MODALITY_MASK
        from rl.environment.protos.service_pb2 import ModalityEnum

        flat = np.asarray(CELL_MODALITY_MASK)
        legal = np.zeros(flat.shape[0], dtype=bool)
        move_cells = np.flatnonzero(flat == ModalityEnum.MODALITY_ENUM__MOVE)[:3]
        switch_cells = np.flatnonzero(flat == ModalityEnum.MODALITY_ENUM__SWITCH)[:2]
        legal[move_cells] = True
        legal[switch_cells] = True
        rng = np.random.default_rng(7)
        logits = rng.normal(size=legal.shape).astype(np.float32)
        if starve_switch:
            logits[switch_cells] = -14.0
        return (
            jnp.asarray(legal),
            jnp.asarray(logits),
            move_cells,
            switch_cells,
        )

    def _loss(self, legal):
        from rl.model.utils import legal_log_policy
        from rl.online.training.loss import uniform_kl_modalities

        def loss(y):
            return uniform_kl_modalities(legal_log_policy(y, legal), legal)

        return loss

    def test_modality_level_gradient_is_pi_m_minus_one_over_m(self):
        from rl.model.utils import legal_log_policy

        legal, logits, move_cells, switch_cells = self._legal_and_logits()
        grad = np.asarray(jax.grad(self._loss(legal))(logits))
        pi = np.asarray(jnp.exp(legal_log_policy(logits, legal)))

        pi_move = pi[move_cells].sum()
        pi_switch = pi[switch_cells].sum()
        np.testing.assert_allclose(grad[move_cells].sum(), pi_move - 0.5, atol=1e-5)
        np.testing.assert_allclose(grad[switch_cells].sum(), pi_switch - 0.5, atol=1e-5)
        assert abs(grad.sum()) < 1e-5, "zero-sum over live modalities"

    def test_starved_modality_still_feels_the_full_pull(self):
        """No pi_m prefactor: at pi_switch ~ 1e-6 the modality-level force is
        the full -(1/M - pi_m), where every pi-prefactored term in the
        bracket is numerically dead. This is the property the term exists
        for."""
        legal, logits, _, switch_cells = self._legal_and_logits(starve_switch=True)
        grad = np.asarray(jax.grad(self._loss(legal))(logits))
        assert grad[switch_cells].sum() == pytest.approx(-0.5, abs=1e-3)

    def test_within_modality_redistribution_is_invariant(self):
        """The loss reads the marginals alone, so swapping two move cells'
        logits -- a pure WHICH-move change -- is exactly invariant. The
        positive control swaps mass ACROSS the modality boundary, which a
        vacuously-flat loss would also ignore."""
        legal, logits, move_cells, switch_cells = self._legal_and_logits()
        loss = self._loss(legal)
        base = float(loss(logits))

        swapped = np.asarray(logits).copy()
        swapped[move_cells[0]], swapped[move_cells[1]] = (
            swapped[move_cells[1]],
            swapped[move_cells[0]],
        )
        np.testing.assert_allclose(float(loss(jnp.asarray(swapped))), base, atol=1e-6)

        crossed = np.asarray(logits).copy()
        crossed[move_cells[0]], crossed[switch_cells[0]] = (
            crossed[switch_cells[0]],
            crossed[move_cells[0]],
        )
        assert float(loss(jnp.asarray(crossed))) != pytest.approx(base, abs=1e-4)

    def test_one_live_modality_is_silent(self):
        """A forced row (only one modality live) contributes -log 1 = 0 and
        zero gradient -- no mask plumbing needed for forced switches."""
        from rl.environment.data import CELL_MODALITY_MASK
        from rl.environment.protos.service_pb2 import ModalityEnum

        flat = np.asarray(CELL_MODALITY_MASK)
        legal = np.zeros(flat.shape[0], dtype=bool)
        switch_cells = np.flatnonzero(flat == ModalityEnum.MODALITY_ENUM__SWITCH)[:3]
        legal[switch_cells] = True
        logits = jnp.asarray(
            np.random.default_rng(11).normal(size=legal.shape).astype(np.float32)
        )
        loss = self._loss(jnp.asarray(legal))
        assert float(loss(logits)) == pytest.approx(0.0, abs=1e-6)
        grad = np.asarray(jax.grad(loss)(logits))
        np.testing.assert_allclose(grad, 0.0, atol=1e-6)
