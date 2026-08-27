import jax
import jax.numpy as jnp

from rl.environment.data import FLAT_MODALITY_MASK, NUM_MODALITY_FEATURES
from rl.utils import average


def factorised_entropies(
    log_policy: jax.Array,
    taken_modality: jax.Array,
    legal_mask: jax.Array,
):
    """(H_macro, H_micro_taken) per row, f32 — the Oct–Nov 2025 per-level
    entropy form rebuilt on the composed head.

    H_macro is the entropy of the modality marginal over live modalities;
    H_micro_taken the entropy of the conditional within the TAKEN
    modality's legal cells. Consumed as unit-weight masked AVERAGES over
    their own row sets (train_step), which is the whole reason the form
    exists: the joint entropy decomposes as H(macro) + sum_m pi_macro(m) *
    H(micro|m), so its within-switch pressure dies in proportion to switch
    mass — the regulariser defunds the which-axis exactly as the modality
    shrinks — while the per-level form keeps unit weight on WHICH and the
    masked average makes a rare taken-switch row's term inverse-frequency
    amplified. A temperature opposed by live evidence, never a target: per
    axis the equilibrium is pi ∝ exp(A/coef), so real advantages beat it
    wherever they exist.

    Both terms are NORMALISED by their own log(k) (user catch 2026-08-27:
    raw conditional entropy caps at log k, so a move-taken row with 8
    legal cells outweighs a 2-switch row ~3x and the 'unit budget' is
    silently k-weighted). Normalised, each row contributes fraction-of-max
    in [0, 1], the panels are comparable across modalities, and the
    effective per-modality temperature is coef/log k — relatively stronger
    regularisation for SMALL modalities, which is the direction the
    which-axis needs. Rows with k < 2 are excluded by the caller's masks;
    the guards here only keep the arithmetic finite on excluded rows."""
    modality_oh = jax.nn.one_hot(
        jnp.asarray(FLAT_MODALITY_MASK), NUM_MODALITY_FEATURES, dtype=jnp.bool_
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
        jnp.asarray(FLAT_MODALITY_MASK) == taken_modality[..., None]
    )
    log_conditional = log_policy32 - macro_taken
    conditional_probs = jnp.where(taken_cells, jnp.exp(log_conditional), 0.0)
    h_micro_taken = -(
        conditional_probs * jnp.where(taken_cells, log_conditional, 0.0)
    ).sum(axis=-1)
    taken_count = taken_cells.sum(axis=-1)
    h_micro_taken = h_micro_taken / jnp.log(jnp.maximum(taken_count, 2))
    return h_macro, h_micro_taken


def entropy_floor_step(
    log_alpha: jax.Array,
    *,
    target: float,
    entropy_value: jax.Array,
    rows: jax.Array,
    alpha_lr: float,
    alpha_min: float,
    alpha_max: float,
):
    """One dual-ascent step on a per-axis entropy temperature (2026-08-28).

    log alpha moves by alpha_lr * (target - H_norm): below target the
    temperature rises, above it it relaxes — SAC's constraint form
    max E[A] s.t. H >= target, whose equilibrium is still pi ∝ exp(A/alpha)
    per axis, so the controller picks only the temperature and evidence
    keeps deciding WHICH cells hold the mass. Clipped to [alpha_min,
    alpha_max] in log space; FROZEN when the axis had no rows this batch
    (average() reads 0.0 on an empty mask, which would otherwise register
    as maximal entropy deficit — the NaN-EMA lesson's shape, LESSONS 2)."""
    err = target - entropy_value.astype(jnp.float32)
    stepped = jnp.clip(
        log_alpha + alpha_lr * err,
        jnp.log(alpha_min),
        jnp.log(alpha_max),
    )
    return jnp.where(rows > 0, stepped, log_alpha)


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
