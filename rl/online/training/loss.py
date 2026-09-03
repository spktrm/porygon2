import jax
import jax.numpy as jnp

from rl.environment.data import CELL_MODALITY_MASK, NUM_MODALITY_FEATURES
from rl.utils import average


def factorised_entropies(
    log_policy: jax.Array,
    taken_modality: jax.Array,
    legal_mask: jax.Array,
):
    """(H_macro, H_micro_taken) per row, f32 — OBSERVERS since 2026-08-30
    (the per-axis entropy loss terms and their dual temperatures are gone;
    the regulariser is the plain joint entropy bonus). These stay as the
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


def uniform_kl_modalities(log_policy: jax.Array, legal_mask: jax.Array) -> jax.Array:
    """Per-row KL(u || pi_m) over LIVE modalities -- the ZERO-AVOIDING term,
    moved to the MODALITY MARGINAL (2026-08-31).

    The row form (`uniform_kl_rows`, gradient pi_b - 1/k over every legal
    cell) was measured on the sp75b/sp75c matched BR pair to be a ROW
    flattener: its pull separates moves from each other with the same force
    it restores switch mass with, buying WHETHER-to-switch and paying in
    WHICH-move-to-pick (entropy_micro_taken pinned 0.93 against the
    control's 0.84, exploit halved). This form keeps everything the constant
    reference bought and drops the tax:

        KL(u || pi_m) = -(1/M) * sum_m log pi_m   (+ a constant), M = live
        d/d y_b       = pi_b - (1/M) * pi_b / pi_m(b)

    summed over a modality's cells that is exactly **pi_m - 1/M**: bounded,
    zero-sum over modalities, NO pi_m prefactor -- and the per-cell force is
    proportional to the CONDITIONAL pi_b/pi_m, so the term moves mass into a
    modality along the policy's own within-modality ranking and never
    reranks it. The loss depends on the marginals alone: any redistribution
    within a modality is invariant, which is the phase-4 law -- the
    regulariser says WHETHER, never WHICH -- written as an identity rather
    than an aspiration.

    A row with one live modality contributes -log 1 = 0, so forced rows are
    silent without a mask (the caller's row mask still applies). Returned
    per row, f32.
    """
    modality_oh = jax.nn.one_hot(
        jnp.asarray(CELL_MODALITY_MASK), NUM_MODALITY_FEATURES, dtype=jnp.bool_
    )
    log_policy32 = log_policy.astype(jnp.float32)
    marginal = jax.nn.logsumexp(
        jnp.where(legal_mask[..., None] & modality_oh, log_policy32[..., None], -1e9),
        axis=-2,
    )
    live = marginal > -1e8
    num_live = live.sum(axis=-1)
    weights = live / jnp.maximum(num_live, 1)[..., None]
    return -(weights * jnp.where(live, marginal, 0.0)).sum(axis=-1)


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
    """Ratio-surrogate PG loss, one selector over the two objectives:
    both the player and the builder run SPO's smooth quadratic penalty
    (config.player_pg_objective — "ppo" is the A/B alternative)."""
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


def cosine_distance(pred: jax.Array, target: jax.Array) -> jax.Array:
    """1 - cos(pred, target) over the last axis, in f32, in [0, 2]. The
    dynamics head's loss (2026-09-03): scale-free, so a target row that is
    linear in its wire multi-hots is judged on direction alone."""
    pred = pred.astype(jnp.float32)
    target = target.astype(jnp.float32)
    dot = (pred * target).sum(axis=-1)
    norms = jnp.linalg.norm(pred, axis=-1) * jnp.linalg.norm(target, axis=-1)
    return 1.0 - dot / jnp.maximum(norms, 1e-6)
