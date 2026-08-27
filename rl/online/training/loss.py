import jax
import jax.numpy as jnp

from rl.environment.data import FLAT_MODALITY_MASK, NUM_MODALITY_FEATURES
from rl.utils import average


def factorised_log_probs(
    log_policy: jax.Array,
    log_prob: jax.Array,
    taken_modality: jax.Array,
    legal_mask: jax.Array,
):
    """(macro_log_prob, micro_log_prob) of the taken action, f32.

    By the composition identity (heads.compose_action_grid: log_policy(a)
    = composed_macro[m(a)] + centred micro) the modality-wise logsumexp of
    the log-policy over legal cells IS the composed macro log-softmax, so
    the two levels are exact: macro = marginal[m(a_taken)], micro =
    log_prob - macro. Written once for the factorised policy loss; the
    behaviour side's macro comes stored from the actor
    (PlayerPolicyHeadOutput.macro_log_prob, same construction in
    player_model._modality_log_marginal) and its micro is the same
    subtraction."""
    modality_oh = jax.nn.one_hot(
        jnp.asarray(FLAT_MODALITY_MASK), NUM_MODALITY_FEATURES, dtype=jnp.bool_
    )
    marginal = jax.nn.logsumexp(
        jnp.where(
            legal_mask[..., None] & modality_oh,
            log_policy.astype(jnp.float32)[..., None],
            -1e9,
        ),
        axis=-2,
    )
    macro_log_prob = jnp.take_along_axis(
        marginal, taken_modality[..., None], axis=-1
    ).squeeze(-1)
    micro_log_prob = log_prob.astype(jnp.float32) - macro_log_prob
    return macro_log_prob, micro_log_prob


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
