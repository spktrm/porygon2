import jax
import jax.numpy as jnp

from rl.utils import average


def barlow_twins_loss(
    z_pred: jax.Array, z_target: jax.Array, mask: jax.Array, lambda_param: float = 0.005
) -> jax.Array:
    """
    Computes Barlow Twins loss over valid sequence elements.
    Forces feature dimensions to decorrelate to prevent collapse.
    """
    feature_dim = z_pred.shape[-1]

    # Flatten batch and seq dimensions: (B * T, D)
    z_a = z_pred.reshape(-1, feature_dim)
    z_b = z_target.reshape(-1, feature_dim)
    mask_flat = mask.reshape(-1, 1)  # (B * T, 1)

    # Avoid division by zero
    valid_count = jnp.maximum(mask_flat.sum(), 1.0)

    # 1. Mean center the valid embeddings
    mean_a = jnp.sum(z_a * mask_flat, axis=0, keepdims=True) / valid_count
    mean_b = jnp.sum(z_b * mask_flat, axis=0, keepdims=True) / valid_count

    z_a_centered = (z_a - mean_a) * mask_flat
    z_b_centered = (z_b - mean_b) * mask_flat

    # 2. Compute standard deviation over valid elements
    std_a = jnp.sqrt(
        jnp.sum(z_a_centered**2, axis=0, keepdims=True) / valid_count + 1e-6
    )
    std_b = jnp.sqrt(
        jnp.sum(z_b_centered**2, axis=0, keepdims=True) / valid_count + 1e-6
    )

    # 3. Normalize
    z_a_norm = z_a_centered / std_a
    z_b_norm = z_b_centered / std_b

    # 4. Compute cross-correlation matrix (D x D)
    c = jnp.dot(z_a_norm.T, z_b_norm) / valid_count

    # 5. Compute loss: (1 - diagonal)^2 + lambda * off_diagonal^2
    on_diag = jnp.sum((jnp.diag(c) - 1.0) ** 2)

    off_diag_mask = 1.0 - jnp.eye(feature_dim, dtype=c.dtype)
    off_diag = jnp.sum((c * off_diag_mask) ** 2)

    return on_diag + lambda_param * off_diag


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


def policy_gradient_loss(
    *,
    policy_ratios: jax.Array,
    advantages: jax.Array,
    valid: jax.Array,
    threshold: float,
):
    pg_loss = spo_objective(
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
