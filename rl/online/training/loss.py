from typing import NamedTuple

import jax
import jax.numpy as jnp

from rl.utils import average


def spo_objective(
    *,
    policy_ratios: jax.Array,
    advantages: jax.Array,
    clip_ppo: float,
):
    """Objective taken from SPO paper: https://arxiv.org/pdf/2401.16025"""
    return policy_ratios * advantages - (
        jnp.abs(advantages) * (1 - policy_ratios) ** 2
    ) / (2 * clip_ppo)


def ppo_objective(
    *,
    policy_ratios: jax.Array,
    advantages: jax.Array,
    clip_ppo: float,
):
    """PPO clipped surrogate (Schulman et al. 2017; restored 2026-08-26
    from the pre-4234016 form): min(r*A, clip(r, 1-eps, 1+eps)*A). The
    min is one-sided pessimism — the gradient is exactly zero once the
    ratio leaves the band IN THE DIRECTION the advantage pushes, and
    untouched when the clip would flatter the objective."""
    l1 = policy_ratios * advantages
    l2 = jnp.clip(policy_ratios, 1.0 - clip_ppo, 1.0 + clip_ppo) * advantages
    return jnp.minimum(l1, l2)


def policy_gradient_loss(
    *,
    policy_ratios: jax.Array,
    advantages: jax.Array,
    valid: jax.Array,
    threshold: float,
    objective: str = "spo",
):
    """Ratio-surrogate PG loss, one selector over the two objectives: the
    builder keeps SPO's smooth quadratic penalty; the player runs NashPG's
    PPO clip (config.player_pg_objective — "spo" is the A/B alternative)."""
    objective_fn = {"spo": spo_objective, "ppo": ppo_objective}[objective]
    pg_loss = objective_fn(
        policy_ratios=policy_ratios,
        advantages=advantages,
        clip_ppo=threshold,
    )
    return -average(pg_loss, valid)


def clip_fraction(
    *,
    policy_ratios: jax.Array,
    valid: jax.Array,
    clip_ppo: float,
):
    """Fraction of valid rows whose ratio sits outside the PPO band."""
    clipped = jnp.abs(policy_ratios - 1) > clip_ppo
    return average(clipped, valid)


def mse_value_loss(*, pred: jax.Array, target: jax.Array, valid: jax.Array):
    mse_loss = jnp.square(pred - target)
    return average(mse_loss, valid)


def approx_forward_kl(*, policy_ratio: jax.Array, log_policy_ratio: jax.Array):
    """
    Calculate the Forward KL approximation.
    """
    return (policy_ratio - 1) - log_policy_ratio


def approx_backward_kl(*, policy_ratio: jax.Array, log_policy_ratio: jax.Array):
    """
    Calculate the Backward KL approximation.
    """
    return policy_ratio * log_policy_ratio - (policy_ratio - 1)


def backward_kl_loss(
    *, policy_ratio: jax.Array, log_policy_ratio: jax.Array, valid: jax.Array
):
    """
    Calculate the Backward KL loss.
    Taken from http://joschu.net/blog/kl-approx.html
    """
    loss = approx_backward_kl(
        policy_ratio=policy_ratio, log_policy_ratio=log_policy_ratio
    )
    return average(loss, valid)


def forward_kl_loss(
    *, policy_ratio: jax.Array, log_policy_ratio: jax.Array, valid: jax.Array
):
    """
    Calculate the Forward KL loss.
    Taken from http://joschu.net/blog/kl-approx.html
    """
    loss = approx_forward_kl(
        policy_ratio=policy_ratio, log_policy_ratio=log_policy_ratio
    )
    return average(loss, valid)


def warmup_scale(step_count, warmup_steps: int):
    """NeuRD warm-up ramp (Step 2, docs/critic-weakness-analysis.md):
    0 -> 1 linearly over the lineage's first `warmup_steps` learner steps,
    1 thereafter; warmup_steps <= 0 disables it. Computed from the train
    state's own step_count INSIDE the jit (a traced leaf of fixed shape),
    so there is no per-step static config and no extra executable —
    the failure class that retired RuntimeScalars on 2026-08-21. Resumed
    lineages carry their step_count, so a warm-up never re-fires on a
    restart; only a fresh launch ramps."""
    if warmup_steps <= 0:
        return jnp.ones((), dtype=jnp.float32)
    return jnp.clip(
        step_count.astype(jnp.float32) / jnp.float32(warmup_steps), 0.0, 1.0
    )


class HierarchicalNeurd(NamedTuple):
    loss: jax.Array  # (...,) per row: -sum_m w_m.y_c_m - sum_a w_a.z_c_a
    macro_weight: jax.Array  # (..., M) open, centred Q(m) - V
    micro_weight: jax.Array  # (..., A) open, centred Q(a) - Q(m)
    macro_open: jax.Array  # (..., M) bool
    micro_open: jax.Array  # (..., A) bool
    macro_adv: jax.Array  # (..., M) Q(m) - V, uncentred, 0 off legal
    micro_adv: jax.Array  # (..., A) Q(a) - Q(m), uncentred, 0 off legal
    modality_legal: jax.Array  # (..., M) bool
    logit_l2: jax.Array  # (...,) 0.5.(sum_m y_c^2 + sum_a z_c^2), carries grad
    macro_gap: jax.Array  # (..., M) stop-grad centred macro logits, 0 off legal
    micro_gap: jax.Array  # (..., A) stop-grad centred micro logits, 0 off legal


def hierarchical_neurd(
    *,
    macro_logits: jax.Array,
    micro_logits: jax.Array,
    adv: jax.Array,
    legal: jax.Array,
    modality_oh: jax.Array,
    beta: float,
) -> HierarchicalNeurd:
    """NeuRD (Hennes et al. 2020, eq. 6 + the eq. 10 logit-gap clip) on
    the FREE logits of a two-level softmax policy

        pi(a) = softmax_M(y)[m(a)] . softmax_{m(a)}(z)[a],

    y (..., M) the macro logits over modalities, z (..., A) the micro
    logits over the flat action grid, modality_oh (A, M) the cell ->
    modality one-hot, adv (..., A) the stop-gradient per-cell advantage
    Q(a) - V (0 on illegal cells).

    Why not the composed log-policy: log pi is normalised, so
    d/dy_m sum_a w_a log pi_a = W_m - pi_M(m).sum_a w_a and
    d/dz_a = w_a - pi(a|m).W_m. That equals NeuRD on y, z only while the
    weights are zero-sum; the clip zeroes cells, and the leftover
    -pi(.).sum w is a push ALONG pi (sharpening toward whatever already
    holds mass when the open weights sum negative). The loss instead
    reads each level's CENTRED live logits (rnad.py's
    `logit_pi - mean_logit`): d/dy_m = -(w_m - mean_legal(w)) — a
    pi-free uniform projection, so no cross-term along pi, and the
    softmax-invariant mean direction gets exactly zero push even when
    the band clips (against raw y, z it was unopposed then).

    Levels: the macro decision's counterfactual value is the modality's
    value under the current micro policy, Q(m) = sum_{a in m} pi(a|m).Q(a),
    so its regret is Q(m) - V; a cell's regret within its modality is
    Q(a) - Q(m). Each is centred uniformly over its own legal set (the
    softmax-invariant direction carries no push) and clipped against its
    level's centred free logit at +-beta: no further OUTWARD push once a
    logit sits beyond the band. pi(a|m) is the within-modality softmax of
    z itself (no division by a starved pi_M(m)), stop-gradient.
    """
    f32 = jnp.float32
    legal = jnp.asarray(legal, dtype=bool)
    modality_oh = jnp.asarray(modality_oh, dtype=bool)
    mod_index = jnp.argmax(modality_oh, axis=-1)  # (A,)
    oh = modality_oh.astype(f32)
    y = jnp.asarray(macro_logits, f32)
    z = jnp.asarray(micro_logits, f32)
    adv = jnp.where(legal, jnp.asarray(adv, f32), 0.0)

    cell_in_modality = legal[..., :, None] & modality_oh  # (..., A, M)
    modality_legal = cell_in_modality.any(axis=-2)  # (..., M)
    count_cells = jnp.maximum(legal.astype(f32) @ oh, 1.0)  # (..., M)
    count_modalities = jnp.maximum(modality_legal.sum(axis=-1), 1).astype(f32)

    # pi(a|m): softmax of the micro logits over the modality's legal cells.
    z_masked = jnp.where(cell_in_modality, z[..., :, None], -1e9)
    lse_m = jax.nn.logsumexp(z_masked, axis=-2)  # (..., M)
    pi_cond = jax.lax.stop_gradient(
        jnp.where(legal, jnp.exp(z - lse_m[..., mod_index]), 0.0)
    )

    macro_adv = (pi_cond * adv) @ oh  # Q(m) - V
    macro_adv = jnp.where(modality_legal, macro_adv, 0.0)
    micro_adv = jnp.where(legal, adv - macro_adv[..., mod_index], 0.0)

    macro_c = macro_adv - (macro_adv.sum(axis=-1) / count_modalities)[..., None]
    macro_c = jnp.where(modality_legal, macro_c, 0.0)
    micro_mean = (micro_adv @ oh) / count_cells
    micro_c = jnp.where(legal, micro_adv - micro_mean[..., mod_index], 0.0)

    y = jnp.where(modality_legal, y, 0.0)
    z = jnp.where(legal, z, 0.0)
    # Centred LIVE (no stop-gradient): the quadratic decay below reads
    # these, so d logit_l2 / dy_m = y_c_m exactly (centring is a
    # symmetric idempotent projection; the softmax-invariant direction
    # carries no decay either). The clip comparisons use the stop-grad
    # copies, as before.
    y_c = jnp.where(
        modality_legal, y - (y.sum(axis=-1) / count_modalities)[..., None], 0.0
    )
    z_c = jnp.where(legal, z - ((z @ oh) / count_cells)[..., mod_index], 0.0)
    y_gap = jax.lax.stop_gradient(y_c)
    z_gap = jax.lax.stop_gradient(z_c)

    def _open(valid, gap, w):
        return valid & jnp.logical_not(
            ((gap > beta) & (w > 0)) | ((gap < -beta) & (w < 0))
        )

    macro_open = _open(modality_legal, y_gap, macro_c)
    micro_open = _open(legal, z_gap, micro_c)
    macro_weight = jax.lax.stop_gradient(jnp.where(macro_open, macro_c, 0.0))
    micro_weight = jax.lax.stop_gradient(jnp.where(micro_open, micro_c, 0.0))

    # The loss reads the CENTRED live logits, not the raw ones —
    # rnad.py's `logit_pi - mean_logit` (get_loss_nerd), verbatim per
    # level. d/dy_m becomes -(w_m - mean_legal(w)): the softmax-invariant
    # mean direction receives exactly zero push, open or clipped. Against
    # the raw logits that direction was unopposed the moment the band
    # zeroed cells (non-zero-sum weights) — free logits could drift
    # together without bound, invisible to pi, the decay and the clip.
    # This is a linear projection, NOT a normalisation: no pi prefactor
    # enters (the log-softmax cross-term argument above is untouched).
    loss = -((macro_weight * y_c).sum(axis=-1) + (micro_weight * z_c).sum(axis=-1))
    # Proximal restoring force (the smooth counterpart of the eq. 10
    # band): the linear NeuRD loss has NO inward force anywhere inside
    # +-beta, so a persistent same-sign advantage integrates into the
    # logits without bound (the 2026-08-25 macro-head grad runaway).
    # 0.5.|centred logit|^2 per level gives d/dy_m = +y_c_m, a
    # mass-INDEPENDENT pull toward the legal-set mean (starved cells are
    # pulled back up as surely as dominant ones are pulled down), with
    # per-cell fixed point gap* = w / decay_coef — bounded rationality
    # with the same geometry as Magnetic Mirror Descent's proximal term.
    logit_l2 = 0.5 * ((y_c**2).sum(axis=-1) + (z_c**2).sum(axis=-1))
    return HierarchicalNeurd(
        loss=loss,
        macro_weight=macro_weight,
        micro_weight=micro_weight,
        macro_open=macro_open,
        micro_open=micro_open,
        macro_adv=macro_adv,
        micro_adv=micro_adv,
        modality_legal=modality_legal,
        logit_l2=logit_l2,
        macro_gap=y_gap,
        micro_gap=z_gap,
    )
