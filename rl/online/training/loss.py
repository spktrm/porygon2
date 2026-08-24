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


def policy_gradient_loss(
    *,
    policy_ratios: jax.Array,
    advantages: jax.Array,
    valid: jax.Array,
    threshold: float,
    objective: str = "spo",
):
    """Ratio-surrogate PG loss. Builder-only since 2026-08-21 — the player
    policy is trained by all-action NeuRD alone (LESSONS.md 3)."""
    objective_fn = {"spo": spo_objective}[objective]
    pg_loss = objective_fn(
        policy_ratios=policy_ratios,
        advantages=advantages,
        clip_ppo=threshold,
    )
    return -average(pg_loss, valid)


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
    loss: jax.Array  # (...,) per row: -sum_m w_m.y_m - sum_a w_a.z_a
    macro_weight: jax.Array  # (..., M) open, centred Q(m) - V
    micro_weight: jax.Array  # (..., A) open, centred Q(a) - Q(m)
    macro_open: jax.Array  # (..., M) bool
    micro_open: jax.Array  # (..., A) bool
    macro_adv: jax.Array  # (..., M) Q(m) - V, uncentred, 0 off legal
    micro_adv: jax.Array  # (..., A) Q(a) - Q(m), uncentred, 0 off legal
    modality_legal: jax.Array  # (..., M) bool


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
    holds mass when the open weights sum negative). Against y and z
    directly, d/dy_m = -w_m and d/dz_a = -w_a exactly, open or clipped.

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
    y_gap = jax.lax.stop_gradient(y - (y.sum(axis=-1) / count_modalities)[..., None])
    z_gap = jax.lax.stop_gradient(z - ((z @ oh) / count_cells)[..., mod_index])

    def _open(valid, gap, w):
        return valid & jnp.logical_not(
            ((gap > beta) & (w > 0)) | ((gap < -beta) & (w < 0))
        )

    macro_open = _open(modality_legal, y_gap, macro_c)
    micro_open = _open(legal, z_gap, micro_c)
    macro_weight = jax.lax.stop_gradient(jnp.where(macro_open, macro_c, 0.0))
    micro_weight = jax.lax.stop_gradient(jnp.where(micro_open, micro_c, 0.0))

    loss = -((macro_weight * y).sum(axis=-1) + (micro_weight * z).sum(axis=-1))
    return HierarchicalNeurd(
        loss=loss,
        macro_weight=macro_weight,
        micro_weight=micro_weight,
        macro_open=macro_open,
        micro_open=micro_open,
        macro_adv=macro_adv,
        micro_adv=micro_adv,
        modality_legal=modality_legal,
    )
