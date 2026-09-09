"""Paired target/outcome diagnostics on unique, first-use decision rows."""

import jax.numpy as jnp

from rl.environment.data import CAT_VF_SUPPORT
from rl.online.training.targets import compute_player_targets


def paired_advantage_audit(
    batch, public_log_probs, privileged_log_probs, isr, config, axis
):
    """Compare both V-trace estimators with behaviour-outcome residuals.

    Uses each head's own baseline for its MC residual. The rho-weighted
    residual additionally matches the V-trace advantage's outer weight,
    so their difference isolates bootstrap disagreement on the same row.
    Outcomes are observational behaviour returns, not counterfactual Q labels.
    Emit sums/counts, never averages of sparse per-batch means. First visits
    exclude replay duplicates; within-game row correlation still remains.
    """
    if isinstance(batch.game_outcome, tuple) or isinstance(batch.reuse_count, tuple):
        return {}

    estimates = {}
    for name, log_probs in (
        ("public", public_log_probs),
        ("privileged", privileged_log_probs),
    ):
        targets, _ = compute_player_targets(batch, log_probs, isr, config)
        value = jnp.exp(log_probs.astype(jnp.float32)) @ jnp.asarray(
            CAT_VF_SUPPORT, dtype=jnp.float32
        )
        estimates[name] = (targets.pg_advantages, value)

    row_index = jnp.arange(isr.shape[0], dtype=jnp.int32)[:, None]
    remaining = batch.game_length - 1 - (batch.game_step_offset + row_index)
    realised_return = batch.game_outcome * config.player_gamma ** jnp.maximum(
        remaining, 0
    )
    fresh = batch.reuse_count == 0
    valid = (
        targets.policy_mask
        & axis.has_both
        & fresh
        & (remaining >= 1)
        & jnp.isfinite(realised_return)
    )
    rho = jnp.minimum(1.0, isr.astype(jnp.float32))
    signals = {"return": realised_return, "rho": rho}
    for name, (advantage, value) in estimates.items():
        residual = realised_return - value
        signals[f"{name}_td"] = advantage
        signals[f"{name}_mc"] = residual
        signals[f"{name}_rho_mc"] = rho * residual
        signals[f"{name}_value"] = value
        signals[f"{name}_sse"] = residual**2
        signals[f"{name}_td_negative_mc_positive"] = (advantage < 0) & (residual > 0)
        signals[f"{name}_td_positive_mc_negative"] = (advantage > 0) & (residual < 0)
    signals["priv_negative_public_positive"] = (estimates["privileged"][0] < 0) & (
        estimates["public"][0] > 0
    )
    signals["priv_positive_public_negative"] = (estimates["privileged"][0] > 0) & (
        estimates["public"][0] < 0
    )

    logs = {}
    for action, action_mask in (
        ("switch", axis.taken_switch),
        ("stay", ~axis.taken_switch),
    ):
        for horizon, horizon_mask in (
            ("all", remaining >= 1),
            ("1_5", (remaining >= 1) & (remaining <= 5)),
            ("6_15", (remaining >= 6) & (remaining <= 15)),
            ("16_40", (remaining >= 16) & (remaining <= 40)),
            ("41_plus", remaining >= 41),
        ):
            selected = valid & action_mask & horizon_mask
            prefix = f"player_adv_audit_{action}_{horizon}"
            logs[f"{prefix}_count"] = selected.sum()
            for name, values in signals.items():
                logs[f"{prefix}_{name}_sum"] = jnp.where(selected, values, 0.0).sum()
    return logs
