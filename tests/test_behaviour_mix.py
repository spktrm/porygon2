"""Epsilon-mixed behaviour policy (rl/model/heads.py behaviour_log_policy).

mu = (1 - mix).pi + mix.prior is what the actor SAMPLES and RECORDS on
explore games; pi (log_policy) is untouched. Pins: bitwise identity at
mix == 0 (the learner's default path), the exact mixture at mix > 0, the
collapse-independent floor on a starved cell, and invalid cells staying
unsampleable.
"""

import jax.numpy as jnp
import numpy as np

from rl.model.heads import behaviour_log_policy, calculate_hierarchical_prior


def _case():
    # 12 cells, first 4 moves / rest switches in FLAT_MODALITY_MASK terms
    # is irrelevant here -- use a synthetic valid mask and a collapsed pi.
    valid = jnp.asarray([True] * 6 + [False] * 6)
    logits = jnp.asarray([6.0, 5.0, 4.0, -4.0, -5.0, -6.0] + [0.0] * 6)
    log_pi = jnp.where(valid, logits - jnp.log(jnp.where(valid, jnp.exp(logits), 0).sum()), 0.0)
    prior = jnp.where(valid, 1.0 / 6, 0.0)
    return valid, log_pi, prior


def test_mix_zero_is_bitwise_pi():
    valid, log_pi, prior = _case()
    out = behaviour_log_policy(log_pi, prior, valid, 0.0)
    np.testing.assert_array_equal(np.asarray(out)[:6], np.asarray(log_pi)[:6])
    assert np.all(np.asarray(out)[6:] == np.finfo(np.float32).min)


def test_mix_is_exact_mixture_and_floors_starved_cells():
    valid, log_pi, prior = _case()
    eps = 0.3
    out = np.asarray(behaviour_log_policy(log_pi, prior, valid, eps))
    expect = np.log((1 - eps) * np.exp(np.asarray(log_pi)[:6]) + eps / 6)
    np.testing.assert_allclose(out[:6], expect, rtol=1e-5)
    # The starved cell (pi ~ 6e-6) now carries at least eps * prior.
    assert np.exp(out[5]) >= eps / 6 * 0.999
    assert np.exp(np.asarray(log_pi)[5]) < 1e-4
    # mu still normalises over legal cells.
    np.testing.assert_allclose(np.exp(out[:6]).sum(), 1.0, rtol=1e-5)


def test_hierarchical_prior_gives_switch_modality_its_share():
    """The supply arithmetic in config assumes the prior splits mass
    evenly across valid modalities -- check on a real-shaped mask."""
    from rl.model.heads import FLAT_MODALITY_MASK, NUM_MODALITY_FEATURES

    valid = jnp.ones(FLAT_MODALITY_MASK.shape[0], dtype=bool)
    prior = np.asarray(calculate_hierarchical_prior(valid))
    per_modality = [prior[np.asarray(FLAT_MODALITY_MASK) == m].sum() for m in range(NUM_MODALITY_FEATURES)]
    present = [p for p in per_modality if p > 0]
    np.testing.assert_allclose(present, 1.0 / len(present), rtol=1e-5)
