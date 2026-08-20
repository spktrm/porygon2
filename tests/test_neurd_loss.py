"""NeuRD prefactor identity (rl/online/learner.py, 2026-08-21).

L = -sum_a sg(w(a)) . y(a) on the RAW logits y, with w the advantage
CENTRED over legal cells and zeroed where the logit-gap clip is closed. The
gradient w.r.t. y must be exactly -w on open cells and 0 on clipped ones --
Hennes et al. eq. (10) with no pi prefactor. A log_policy form was tried
first and FAILED this test: once the clip zeroes cells the weights are no
longer zero-sum and the softmax pulls in a pi(b).sum_a w(a) cross-term.
"""

import jax
import jax.numpy as jnp
import numpy as np


def _neurd_loss(logits, legal, adv, beta):
    legal_count = jnp.maximum(legal.sum(axis=-1), 1)
    adv = jnp.where(legal, adv, 0.0)
    adv_c = adv - (adv.sum(axis=-1) / legal_count)[..., None]
    y = jnp.where(legal, logits, 0.0)
    gap = jax.lax.stop_gradient(y - (y.sum(axis=-1) / legal_count)[..., None])
    open_ = legal & jnp.logical_not(
        ((gap > beta) & (adv_c > 0)) | ((gap < -beta) & (adv_c < 0))
    )
    weight = jax.lax.stop_gradient(jnp.where(open_, adv_c, 0.0))
    return -(weight * y).sum(), weight, open_


def test_gradient_is_minus_centred_advantage_on_open_cells():
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(scale=3.0, size=(4, 10)), dtype=jnp.float32)
    legal = jnp.asarray(rng.random((4, 10)) > 0.3)
    legal = legal.at[:, 0].set(True)
    adv = jnp.asarray(rng.normal(scale=0.2, size=(4, 10)), dtype=jnp.float32)
    beta = 1.5

    grad = jax.grad(lambda x: _neurd_loss(x, legal, adv, beta)[0])(logits)
    _, weight, open_ = _neurd_loss(logits, legal, adv, beta)

    # Open cells: exactly -adv_c (no pi factor, no cross-cell term).
    np.testing.assert_allclose(
        np.asarray(grad)[np.asarray(open_)], -np.asarray(weight)[np.asarray(open_)],
        rtol=1e-5, atol=1e-6,
    )
    # Clipped and illegal cells: untouched.
    np.testing.assert_allclose(np.asarray(grad)[~np.asarray(open_)], 0.0, atol=1e-6)
    # Something was actually clipped at scale-3 logits with beta 1.5,
    # otherwise the clip branch went untested.
    assert bool((np.asarray(legal) & ~np.asarray(open_)).any())


def test_no_pi_prefactor_on_a_starved_cell():
    """The whole point: a cell at pi ~ 1e-3 gets the full advantage."""
    logits = jnp.asarray([[5.0, 0.0, 0.0, -2.0]], dtype=jnp.float32)
    legal = jnp.ones((1, 4), dtype=bool)
    adv = jnp.asarray([[0.0, 0.0, 0.0, 0.3]], dtype=jnp.float32)
    grad = jax.grad(lambda x: _neurd_loss(x, legal, adv, beta=10.0)[0])(logits)
    assert np.asarray(grad)[0, 3] < -0.2  # ~ -(0.3 - 0.075), not scaled by pi
