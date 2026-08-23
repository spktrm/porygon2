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
