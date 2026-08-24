"""Hierarchical NeuRD gradient identity (rl/online/training/loss.py,
hierarchical_neurd, 2026-08-24).

The policy is pi(a) = softmax_M(y)[m(a)] . softmax_m(z)[a]. The loss is
written against the FREE logits y, z, so d/dy_m = -W_m and d/dz_a = -w(a)
exactly -- open or clipped -- with W_m = sum_{a in m} pi(a|m).adv(a) and
w(a) = adv(a) - W_m(a), each centred over its own legal set and zeroed
where its level's logit-gap clip is closed (Hennes et al. eq. 6 + 10, no
pi prefactor). The previous loss read the composed log-policy, which
picks up a -pi(.).sum_a w(a) cross-term once clipped cells break zero-sum;
the last test pins that difference so it cannot come back.
"""

import jax
import jax.numpy as jnp
import numpy as np

from rl.online.training.loss import hierarchical_neurd

# 8 cells in 3 modalities: 0-3, 4-6, and a singleton 7.
MOD_INDEX = np.array([0, 0, 0, 0, 1, 1, 1, 2])
MODALITY_OH = MOD_INDEX[:, None] == np.arange(3)


def _case(seed=0, rows=4, beta=1.5, scale=3.0):
    rng = np.random.default_rng(seed)
    y = jnp.asarray(rng.normal(scale=scale, size=(rows, 3)), dtype=jnp.float32)
    z = jnp.asarray(rng.normal(scale=scale, size=(rows, 8)), dtype=jnp.float32)
    legal = rng.random((rows, 8)) > 0.3
    legal[:, 0] = True
    legal[:, 4] = True
    legal = jnp.asarray(legal)
    adv = jnp.asarray(rng.normal(scale=0.2, size=(rows, 8)), dtype=jnp.float32)
    return y, z, legal, adv, beta


def _loss(y, z, legal, adv, beta):
    return hierarchical_neurd(
        macro_logits=y,
        micro_logits=z,
        adv=adv,
        legal=legal,
        modality_oh=MODALITY_OH,
        beta=beta,
    )


def test_gradient_is_minus_weight_per_level():
    y, z, legal, adv, beta = _case()
    out = _loss(y, z, legal, adv, beta)
    gy, gz = jax.grad(
        lambda y_, z_: _loss(y_, z_, legal, adv, beta).loss.sum(), argnums=(0, 1)
    )(y, z)
    np.testing.assert_allclose(
        np.asarray(gy), -np.asarray(out.macro_weight), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(gz), -np.asarray(out.micro_weight), rtol=1e-5, atol=1e-6
    )
    # Clipped and illegal cells / modalities: untouched.
    assert not np.asarray(gy)[~np.asarray(out.macro_open)].any()
    assert not np.asarray(gz)[~np.asarray(out.micro_open)].any()
    # Something was actually clipped at both levels, or the clip branch
    # went untested.
    assert bool((np.asarray(out.modality_legal) & ~np.asarray(out.macro_open)).any())
    assert bool((np.asarray(legal) & ~np.asarray(out.micro_open)).any())


def test_level_advantages_are_counterfactual_regrets():
    """W_m = sum_{a in m} pi(a|m).adv(a); w(a) = adv(a) - W_m(a); and the
    modality values recombine to the row's policy-weighted advantage."""
    y, z, legal, adv, beta = _case(seed=1, beta=100.0)
    out = _loss(y, z, legal, adv, beta)
    legal_np, adv_np, z_np = map(np.asarray, (legal, adv, z))
    for r in range(y.shape[0]):
        for m in range(3):
            cells = np.where(legal_np[r] & (MOD_INDEX == m))[0]
            if cells.size == 0:
                assert out.macro_adv[r, m] == 0.0
                continue
            p = np.exp(z_np[r, cells] - z_np[r, cells].max())
            p /= p.sum()
            np.testing.assert_allclose(
                np.asarray(out.macro_adv[r, m]),
                (p * adv_np[r, cells]).sum(),
                rtol=1e-5,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                np.asarray(out.micro_adv[r, cells]),
                adv_np[r, cells] - np.asarray(out.macro_adv[r, m]),
                rtol=1e-5,
                atol=1e-6,
            )
    # A singleton modality has no within-modality contest.
    assert not np.asarray(out.micro_weight)[:, 7].any()


def test_no_pi_prefactor_on_a_starved_modality():
    """A modality at pi_M ~ 1e-4 gets its full centred regret on the macro
    logit, and its cells their full within-modality regret."""
    y = jnp.asarray([[9.0, 0.0, 0.0]], dtype=jnp.float32)  # pi_M(1) ~ 1e-4
    z = jnp.zeros((1, 8), dtype=jnp.float32)
    legal = jnp.asarray([[True, True, False, False, True, True, True, False]])
    adv = jnp.asarray([[0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0]], dtype=jnp.float32)
    out = _loss(y, z, legal, adv, beta=100.0)
    # W_1 = 0.3, centred over the two legal modalities -> +0.15 / -0.15.
    np.testing.assert_allclose(
        np.asarray(out.macro_weight[0]), [-0.15, 0.15, 0.0], atol=1e-6
    )


def test_composed_log_policy_form_has_the_cross_term():
    """The bug being fixed: on the composed log pi, once the clip makes the
    weights non-zero-sum, the free-logit gradient is not -w."""
    y, z, legal, adv, beta = _case(seed=2)
    out = _loss(y, z, legal, adv, beta)
    w_macro, w_micro = out.macro_weight, out.micro_weight
    mod_index = jnp.asarray(MOD_INDEX)

    def composed_log_pi(y_, z_):
        log_pm = jax.nn.log_softmax(jnp.where(out.modality_legal, y_, -1e9), axis=-1)
        zm = jnp.where(
            legal[..., :, None] & jnp.asarray(MODALITY_OH), z_[..., :, None], -1e9
        )
        lse = jax.nn.logsumexp(zm, axis=-2)
        return log_pm[..., mod_index] + z_ - lse[..., mod_index]

    def log_pi_loss(y_, z_):
        # The old form: cell weights on the composed log pi. Fold the
        # macro weights onto their cells so both forms carry the same
        # total weight per modality.
        w_cells = (
            w_micro
            + (
                w_macro
                / jnp.maximum(
                    legal.astype(jnp.float32) @ jnp.asarray(MODALITY_OH, jnp.float32),
                    1.0,
                )
            )[..., mod_index]
        )
        w_cells = jnp.where(legal, w_cells, 0.0)
        return -(w_cells * composed_log_pi(y_, z_)).sum()

    gy = jax.grad(log_pi_loss, argnums=0)(y, z)
    # Non-zero-sum open weights exist in this case (the clip fired) ...
    assert bool((jnp.abs(w_macro.sum(-1)) > 1e-4).any())
    # ... and the composed form's macro gradient is then NOT -W_m.
    assert not np.allclose(np.asarray(gy), -np.asarray(w_macro), atol=1e-4)


def test_logit_decay_gradient_is_centred_logit():
    """d logit_l2 / dy_m = y_c_m and d/dz_a = z_c_a exactly (centring is
    a symmetric idempotent projection), zero on illegal cells and on the
    softmax-invariant direction — so the combined loss's per-cell fixed
    point is |centred logit| = |w| / decay_coef."""
    y, z, legal, adv, beta = _case(seed=3)
    gy, gz = jax.grad(
        lambda y_, z_: _loss(y_, z_, legal, adv, beta).logit_l2.sum(),
        argnums=(0, 1),
    )(y, z)
    out = _loss(y, z, legal, adv, beta)
    np.testing.assert_allclose(
        np.asarray(gy), np.asarray(out.macro_gap), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(gz), np.asarray(out.micro_gap), rtol=1e-5, atol=1e-6
    )
    # Illegal cells: no decay.
    assert not np.asarray(gz)[~np.asarray(legal)].any()
    # A uniform shift of all legal logits (softmax-invariant) is
    # decay-free: the gradient is centred, so it sums to ~0 per level.
    np.testing.assert_allclose(np.asarray(gy).sum(-1), 0.0, atol=1e-5)
    legal_np = np.asarray(legal)
    # Per-modality centring: each modality's gradients sum to ~0.
    for m in range(3):
        sel = legal_np & (MOD_INDEX == m)
        for r in range(z.shape[0]):
            if sel[r].any():
                np.testing.assert_allclose(
                    np.asarray(gz)[r][sel[r]].sum(), 0.0, atol=1e-5
                )
