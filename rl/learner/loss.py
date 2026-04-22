from typing import Optional

import jax
import jax.numpy as jnp

from rl.utils import average


def sigreg_loss(
    x: jax.Array,
    rng_key: jax.random.PRNGKey,
    mask: Optional[jax.Array] = None,
    sketch_dim: int = 64,
):
    """
    Forces ECF(x) ~ ECF(Gaussian).
    Matches ALL Moments (Maximum Entropy Cloud).
    Exact implementation of LeJEPA Algorithm 1.
    """
    N, C = x.shape

    if mask is None:
        mask = jnp.ones((N,))

    x_norm = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    x = (x * (C**0.5)) / x_norm.clip(min=1e-6)

    # 1. Sketching
    if C > sketch_dim:
        # Generate random matrix S
        S = jax.random.normal(rng_key, (sketch_dim, C)) / (C**0.5)
        x = x @ S.T  # [N, sketch_dim]
    else:
        sketch_dim = C

    # 2. Centering
    mask_sum = jnp.maximum(mask.sum(axis=0, keepdims=True), 1)
    mean_x = jnp.where(mask[..., None], x, 0.0).sum(axis=0) / mask_sum
    x = jnp.where(mask[..., None], x - mean_x, 0.0)

    # 3. Covariance
    cov_denominator = jnp.maximum(mask_sum - 1.0, 1e-6)
    cov = (x.T @ x) / cov_denominator

    # 4. Target Identity
    target = jnp.eye(sketch_dim)

    # 5. Frobenius Norm
    return jnp.linalg.norm(cov - target, ord="fro")


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
    """Generic PPO clipped surrogate loss"""
    l1 = policy_ratios * advantages
    l2 = jnp.clip(policy_ratios, 1.0 - clip_ppo, 1.0 + clip_ppo) * advantages
    return jnp.minimum(l1, l2)


def hard_neurd_objective(
    mean_centered_logits: jax.Array,
    corrected_advantages: jax.Array,
    threshold: jax.Array,
):
    can_decrease = mean_centered_logits > -threshold
    can_increase = mean_centered_logits < threshold
    force_negative = jnp.minimum(corrected_advantages, 0.0)
    force_positive = jnp.maximum(corrected_advantages, 0.0)
    clipped_force = can_decrease * force_negative + can_increase * force_positive
    return mean_centered_logits * jax.lax.stop_gradient(clipped_force)


def soft_neurd_objective(
    mean_centered_logits: jax.Array,
    corrected_advantages: jax.Array,
    threshold: jax.Array,
    leak: float = 0.0,
    gain: float = 5.0,
):
    corrected_advantages = jax.lax.stop_gradient(corrected_advantages)
    abs_adv = jnp.abs(corrected_advantages)

    # 1. Numerically stable Softmin using logsumexp
    # Equivalent to: -(1/g) * log(exp(-g*a) + exp(-g*b))
    arg1 = -gain * corrected_advantages * mean_centered_logits
    arg2 = -gain * abs_adv * threshold
    loss = -(1.0 / gain) * jax.nn.logsumexp(jnp.stack([arg1, arg2], axis=-1), axis=-1)

    if leak > 0:
        # 2. Numerically stable safe leak using softplus
        # Equivalent to: -(1/g) * log(1 + exp(-g*z))
        sgn_adv = jnp.sign(corrected_advantages)
        z = threshold - sgn_adv * mean_centered_logits
        loss -= (leak * abs_adv / gain) * jax.nn.softplus(-gain * z)

    return loss


def off_policy_neurd_objective(
    *,
    logits: jax.Array,
    policy: jax.Array,
    mask: jax.Array,
    policy_ratios: jax.Array,
    q_values: jax.Array,
    threshold: float,
    advantages_clip: float = 100.0,
):
    """Neurd objective"""

    advantages = q_values - jnp.sum(policy * q_values, axis=-1, keepdims=True)
    corrected_advantages = (advantages * policy_ratios[..., None]).clip(
        -advantages_clip, advantages_clip
    )

    valid_logit_sum = jnp.sum(logits * mask, axis=-1)
    valid_mask_sum = jnp.sum(mask, axis=-1).clip(min=1)
    valid_logit_mean = valid_logit_sum / valid_mask_sum
    mean_centered_logits = logits - valid_logit_mean[..., None]

    threshold = threshold[..., None]

    loss = hard_neurd_objective(
        mean_centered_logits=mean_centered_logits,
        corrected_advantages=corrected_advantages,
        threshold=threshold,
    )

    return jnp.where(mask, loss, 0.0).sum(axis=-1)


def policy_gradient_loss(
    *,
    logits: jax.Array,
    policy: jax.Array,
    policy_ratios: jax.Array,
    q_values: jax.Array,
    valid: jax.Array,
    clip_ppo: float,
    threshold: float,
):
    pg_loss = off_policy_neurd_objective(
        logits=logits,
        policy=policy,
        mask=policy > 0,
        policy_ratios=policy_ratios,
        q_values=q_values,
        threshold=threshold,
    )
    return -average(pg_loss, valid)


def clip_fraction(
    *,
    policy_ratios: jax.Array,
    valid: jax.Array,
    clip_ppo: float,
):
    """Calculate the fraction of clips."""
    clipped = jnp.abs(policy_ratios - 1) > clip_ppo
    return average(clipped, valid)


def mse_value_loss(*, pred: jax.Array, target: jax.Array, valid: jax.Array):
    mse_loss = jnp.square(pred - target)
    return average(mse_loss, valid)


def clipped_value_loss(
    *,
    pred_v: jax.Array,
    target_v: jax.Array,
    old_v: jax.Array,
    valid: jax.Array,
    clip_val: float = 0.2,
):
    loss_unclipped = jnp.square(pred_v - target_v)
    pred_v_clipped = old_v + jnp.clip(pred_v - old_v, -clip_val, clip_val)
    loss_clipped = jnp.square(pred_v_clipped - target_v)
    return average(jnp.maximum(loss_unclipped, loss_clipped), valid)


def ce_value_loss(*, pred_v: jax.Array, target_v: jax.Array, valid: jax.Array):
    mse_loss = -(pred_v * target_v).sum(axis=-1)
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


def power_schedule(
    coef: float, step: int, decay: float, floor: float, ceil: float
) -> jax.Array:
    x = coef / ((step + 1) ** decay)
    return jnp.clip(x, min=floor, max=ceil)
