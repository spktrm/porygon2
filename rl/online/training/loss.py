import jax
import jax.numpy as jnp

from rl.environment.data import CELL_MODALITY_MASK, NUM_MODALITY_FEATURES
from rl.utils import average


def factorised_entropies(
    log_policy: jax.Array,
    taken_modality: jax.Array,
    legal_mask: jax.Array,
):
    """(H_macro, H_micro_taken) per row, f32 — OBSERVERS: the regulariser
    is the plain joint entropy bonus. These stay as the
    collapse instruments the acceptance gates read.

    H_macro is the entropy of the modality marginal over live modalities;
    H_micro_taken the entropy of the conditional within the TAKEN
    modality's legal cells. Both are NORMALISED by their own log(k), so
    each row reads fraction-of-max in [0, 1] and the panels are comparable
    across modalities. Rows with k < 2 are excluded by the caller's masks;
    the guards here only keep the arithmetic finite on excluded rows."""
    modality_oh = jax.nn.one_hot(
        jnp.asarray(CELL_MODALITY_MASK), NUM_MODALITY_FEATURES, dtype=jnp.bool_
    )
    log_policy32 = log_policy.astype(jnp.float32)
    marginal = jax.nn.logsumexp(
        jnp.where(legal_mask[..., None] & modality_oh, log_policy32[..., None], -1e9),
        axis=-2,
    )
    live = marginal > -1e8
    marginal_probs = jnp.where(live, jnp.exp(marginal), 0.0)
    h_macro = -(marginal_probs * jnp.where(live, marginal, 0.0)).sum(axis=-1)
    num_live = live.sum(axis=-1)
    h_macro = h_macro / jnp.log(jnp.maximum(num_live, 2))

    macro_taken = jnp.take_along_axis(marginal, taken_modality[..., None], axis=-1)
    taken_cells = legal_mask & (
        jnp.asarray(CELL_MODALITY_MASK) == taken_modality[..., None]
    )
    log_conditional = log_policy32 - macro_taken
    conditional_probs = jnp.where(taken_cells, jnp.exp(log_conditional), 0.0)
    h_micro_taken = -(
        conditional_probs * jnp.where(taken_cells, log_conditional, 0.0)
    ).sum(axis=-1)
    taken_count = taken_cells.sum(axis=-1)
    h_micro_taken = h_micro_taken / jnp.log(jnp.maximum(taken_count, 2))
    return h_macro, h_micro_taken


def uniform_kl_rows(log_policy: jax.Array, legal_mask: jax.Array) -> jax.Array:
    """KL(uniform over legal actions || policy), per row in f32.

    The logit gradient pi - 1/N stays bounded and zero-sum even when an
    action's probability vanishes. Empty and singleton rows carry no loss.
    """
    legal_count = legal_mask.sum(axis=-1)
    denominator = jnp.maximum(legal_count, 1).astype(jnp.float32)
    legal_log_policy = jnp.where(legal_mask, log_policy.astype(jnp.float32), 0.0)
    divergence = -legal_log_policy.sum(axis=-1) / denominator - jnp.log(denominator)
    return jnp.where(legal_count > 1, divergence, 0.0)


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
    """PPO clipped surrogate (Schulman et al. 2017):
    min(r*A, clip(r, 1-eps, 1+eps)*A). The
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
    """Ratio-surrogate loss: the builder's SPO and the player's APPO share it."""
    objective_fn = {"spo": spo_objective, "ppo": ppo_objective}[objective]
    pg_loss = objective_fn(
        policy_ratios=policy_ratios,
        advantages=advantages,
        clip_ppo=threshold,
    )
    return -average(pg_loss, valid)


def clipped_target_ratio(
    *,
    learner_log_prob: jax.Array,
    behaviour_log_prob: jax.Array,
    old_policy_log_prob: jax.Array,
    behaviour_ratio_clip: float,
) -> jax.Array:
    """IMPACT's surrogate ratio (RLlib appo_torch_policy.loss `logp_ratio`):
    clip(mu/pi_old, 0, c) * pi_live/mu, in f32. Inside the cap the mu
    cancels and this is pi_live/pi_old, so the PPO band is a trust region
    around the target snapshot rather than around each row's own stale
    behaviour policy; past the cap a row whose behaviour policy was more
    than c times likelier than pi_old is scaled down by exactly c*pi_old/mu.
    Only the learner term carries gradient: the other two are batch data
    and a stopped forward."""
    learner_log_prob = learner_log_prob.astype(jnp.float32)
    behaviour_log_prob = jax.lax.stop_gradient(behaviour_log_prob.astype(jnp.float32))
    old_policy_log_prob = jax.lax.stop_gradient(old_policy_log_prob.astype(jnp.float32))
    behaviour_old_ratio = jnp.clip(
        jnp.exp(behaviour_log_prob - old_policy_log_prob), 0.0, behaviour_ratio_clip
    )
    return behaviour_old_ratio * jnp.exp(learner_log_prob - behaviour_log_prob)


def appo_policy_loss(
    *,
    learner_log_prob: jax.Array,
    behaviour_log_prob: jax.Array,
    old_policy_log_prob: jax.Array,
    advantages: jax.Array,
    valid: jax.Array,
    clip_ppo: float,
    behaviour_ratio_clip: float,
) -> jax.Array:
    """The APPO actor loss: PPO's clipped surrogate on the clipped target
    ratio, over stopped V-trace advantages that already carry their own
    truncated rho = min(1, pi_old/mu). Invalid rows are zeroed before the
    ratio is formed so non-finite padding cannot leak through the clip."""
    ratio = clipped_target_ratio(
        learner_log_prob=jnp.where(valid, learner_log_prob, 0.0),
        behaviour_log_prob=jnp.where(valid, behaviour_log_prob, 0.0),
        old_policy_log_prob=jnp.where(valid, old_policy_log_prob, 0.0),
        behaviour_ratio_clip=behaviour_ratio_clip,
    )
    advantages = jax.lax.stop_gradient(advantages.astype(jnp.float32))
    advantages = jnp.where(valid, advantages, 0.0)
    return policy_gradient_loss(
        policy_ratios=ratio,
        advantages=advantages,
        valid=valid,
        threshold=clip_ppo,
        objective="ppo",
    )


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
