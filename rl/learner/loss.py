import jax
import jax.numpy as jnp

from rl.utils import average


def compute_sigreg_loss(
    latent_queries: jax.Array,
    valid_mask: jax.Array,
    key: jax.Array,
    knots: int = 17,
    num_proj: int = 1024,
) -> jax.Array:
    """
    Exact JAX port of LeWM's SIGReg (Epps-Pulley statistic),
    adapted for [T, B, N, D] inputs and masking.
    """
    T, B, N, D = latent_queries.shape

    # 1. Setup the knots and trapezoidal integration weights
    t = jnp.linspace(0, 3, knots, dtype=jnp.float32)
    dt = 3 / (knots - 1)

    weights = jnp.full((knots,), 2 * dt, dtype=jnp.float32)
    weights = weights.at[0].set(dt)
    weights = weights.at[-1].set(dt)

    # Target characteristic function of standard normal
    phi = jnp.exp(-jnp.square(t) / 2.0)
    weights = weights * phi

    # 2. Sketching: Draw random L2-normalized 1D projection vectors
    A = jax.random.normal(key, (D, num_proj))
    A = A / jnp.linalg.norm(A, axis=0, keepdims=True)

    # 3. Compute projections and multiply by knots
    # latent_queries @ A -> [T, B, N, num_proj]
    # Expand dims for knots -> [T, B, N, num_proj, 1] * [knots] -> [T, B, N, num_proj, knots]
    x_t = jnp.expand_dims(latent_queries @ A, axis=-1) * t

    # 4. Masking Preparation
    # Expand mask for broadcasting: [T, B, 1, 1, 1]
    mask_expanded = valid_mask[:, :, None, None, None].astype(x_t.dtype)

    # Total valid tokens per timestep T (shape: [T])
    valid_count_T = valid_mask.sum(axis=1) * N

    # Safe denominator to avoid NaN division on fully padded timesteps (shape: [T, 1, 1])
    safe_valid_count = jnp.maximum(valid_count_T, 1.0)[:, None, None]

    # 5. Compute the masked Empirical Characteristic Function
    # Sum over Batch (axis 1) and Queries (axis 2), divide by valid tokens
    cos_mean = jnp.sum(jnp.cos(x_t) * mask_expanded, axis=(1, 2)) / safe_valid_count
    sin_mean = jnp.sum(jnp.sin(x_t) * mask_expanded, axis=(1, 2)) / safe_valid_count

    # 6. Compute Epps-Pulley error statistic
    err = jnp.square(cos_mean - phi) + jnp.square(sin_mean)

    # Integrate over knots (dot product with weights)
    # err is [T, num_proj, knots], weights is [knots], statistic_T is [T, num_proj]
    statistic_T = err @ weights

    # Scale by sample size (valid elements per timestep) to match convergence rates
    # statistic_T shape: [T, num_proj] * [T, 1] -> [T, num_proj]
    statistic_T = statistic_T * valid_count_T[:, None]

    # 7. Average over valid timesteps and projections
    # Find how many timesteps have at least one valid element
    valid_timesteps = jnp.maximum((valid_count_T > 0).sum(), 1.0)

    # Sum the valid statistics and divide by (valid timesteps * num_proj)
    loss = statistic_T.sum() / (valid_timesteps * num_proj)

    return loss


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
