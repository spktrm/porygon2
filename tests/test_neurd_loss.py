"""Hierarchical NeuRD gradient identity (rl/online/training/loss.py,
hierarchical_neurd, 2026-08-24; centred-logit form 2026-08-26).

The policy is pi(a) = softmax_M(y)[m(a)] . softmax_m(z)[a]. The loss is
written against each level's CENTRED live logits (rnad.py's
`logit_pi - mean_logit`), so d/dy_m = -(W_m - mean_legal(W)) and
d/dz_a = -(w(a) - mean_m(w)) exactly -- open or clipped -- with
W_m = sum_{a in m} pi(a|m).adv(a) and w(a) = adv(a) - W_m(a), each
centred over its own legal set and zeroed where its level's logit-gap
clip is closed (Hennes et al. eq. 6 + 10, no pi prefactor). The update
is therefore zero-sum per level: the softmax-invariant mean direction
receives exactly zero push even when the band clip breaks the weights'
zero-sum -- against the RAW free logits that direction was unopposed
then, which is the gauge drift the dx65cpwp micro runaway rode. The
original loss read the composed log-policy, which picks up a
-pi(.).sum_a w(a) cross-term once clipped cells break zero-sum; the
composed-form test pins that difference so it cannot come back.
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


def test_gradient_is_minus_projected_weight_per_level():
    y, z, legal, adv, beta = _case()
    out = _loss(y, z, legal, adv, beta)
    gy, gz = jax.grad(
        lambda y_, z_: _loss(y_, z_, legal, adv, beta).loss.sum(), argnums=(0, 1)
    )(y, z)
    w_macro = np.asarray(out.macro_weight)
    w_micro = np.asarray(out.micro_weight)
    mod_legal = np.asarray(out.modality_legal)
    legal_np = np.asarray(legal)
    # Expected: -(w - mean_legal(w)) per level, 0 off-legal.
    macro_count = np.maximum(mod_legal.sum(-1, keepdims=True), 1)
    exp_gy = np.where(
        mod_legal, -(w_macro - w_macro.sum(-1, keepdims=True) / macro_count), 0.0
    )
    exp_gz = np.zeros_like(w_micro)
    for m in range(3):
        sel = legal_np & (MOD_INDEX == m)
        for r in range(z.shape[0]):
            if sel[r].any():
                exp_gz[r, sel[r]] = -(w_micro[r, sel[r]] - w_micro[r, sel[r]].mean())
    np.testing.assert_allclose(np.asarray(gy), exp_gy, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(gz), exp_gz, rtol=1e-5, atol=1e-6)
    # Illegal cells / modalities: untouched.
    assert not np.asarray(gy)[~mod_legal].any()
    assert not np.asarray(gz)[~legal_np].any()
    # Something was actually clipped at both levels, or the clip branch
    # went untested.
    assert bool((mod_legal & ~np.asarray(out.macro_open)).any())
    assert bool((legal_np & ~np.asarray(out.micro_open)).any())


def test_gradient_is_zero_sum_even_when_clipped():
    """The gauge fix (2026-08-26): the per-level gradient sums to exactly
    zero over each legal set, so the softmax-invariant mean direction is
    never pushed. Positive control: the clip has fired and the WEIGHTS'
    sums are non-zero — under the previous raw-logit form the gradient
    sum equalled the weight sum, so this test would have failed there."""
    y, z, legal, adv, beta = _case()
    out = _loss(y, z, legal, adv, beta)
    gy, gz = jax.grad(
        lambda y_, z_: _loss(y_, z_, legal, adv, beta).loss.sum(), argnums=(0, 1)
    )(y, z)
    # Control: clipping made the open weights non-zero-sum somewhere.
    w_macro_sums = np.asarray(out.macro_weight).sum(-1)
    assert bool((np.abs(w_macro_sums) > 1e-4).any())
    # Macro: zero-sum per row.
    np.testing.assert_allclose(np.asarray(gy).sum(-1), 0.0, atol=1e-5)
    # Micro: zero-sum per modality per row.
    legal_np = np.asarray(legal)
    gz_np = np.asarray(gz)
    for m in range(3):
        sel = legal_np & (MOD_INDEX == m)
        for r in range(z.shape[0]):
            if sel[r].any():
                np.testing.assert_allclose(gz_np[r][sel[r]].sum(), 0.0, atol=1e-5)


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
