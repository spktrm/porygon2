import jax
import jax.numpy as jnp

from rl.environment.data import FLAT_MODALITY_MASK, NUM_MODALITY_FEATURES
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


def uniform_kl_rows(log_policy: jax.Array, legal_mask: jax.Array) -> jax.Array:
    """Per-row KL(u || pi) over legal cells -- the ZERO-AVOIDING term.

    Note the DIRECTION. Reverse KL(pi || u) is, up to a constant, just
    negative entropy, i.e. the entropy bonus already in the objective; it
    carries the pi prefactor that CLAUDE.md 4 records as structurally unable
    to hold a floor once a modality starves. This is the other direction:

        KL(u || pi) = -(1/k) * sum_a log pi_a   (+ a constant)
        d/d y_b     = pi_b - 1/k

    Bounded by 1 for ANY pi, zero-sum over legal cells, and with NO pi
    prefactor -- so it still pulls at pi = 1e-6, where both the entropy bonus
    (d/dy_b = -pi_b (log pi_b + H)) and the reverse-KL magnet are numerically
    dead. Those three properties are exactly what every mass-restoring
    mechanism this project has tried lacked, and unlike the four support-anchor
    phases the reference here is the CONSTANT uniform distribution: it cannot
    collapse, cannot be ratcheted by a re-snapping reference, and cannot invert
    on the modality-level sign of Q^pi.

    Why it is back (2026-08-31), having been dropped one commit ago for
    reference alignment: the surrogate's expected force on a cell is also
    pi-prefactored (~pi_b * A_b, the cell is taken on ~pi_b of rows), so
    against the entropy bonus the prefactor CANCELS and the equilibrium is a
    pure temperature, pi ∝ exp(A/alpha) -- mass decaying EXPONENTIALLY in the
    headwind. Here it cancels on one side only, so equilibrium mass is
    ~coef/(k*|A|), decaying merely linearly. That is the difference between a
    floor and a coefficient that has been retuned four times.

    Returned per row and unmasked in the batch sense; the caller applies the
    row mask.
    """
    legal = legal_mask.astype(jnp.float32)
    weights = legal / jnp.maximum(legal.sum(axis=-1, keepdims=True), 1.0)
    return -(weights * log_policy.astype(jnp.float32)).sum(axis=-1)


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
