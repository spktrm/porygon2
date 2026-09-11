"""Telemetry over the action axis — readers of the learner's log-policy that
call the loss and target callables (`targets.py` imports `telemetry.py`,
so these cannot live there without a cycle). Observers only: no loss reads
anything here.

`legal_support_telemetry`: how close the policy is to pruning, over each
real decision row's legal cells read as flat complete actions (a move x
target, or one switch, is one cell — no hierarchy): the minimum and median
cell probability, the legal-cell count, the fraction of legal cells below
each of three lines, then the minimum and the fractions split over the
switch and move cells so the switch floor reads on its own. The exposure
instrument for the support hinge (what it must lift) and the v-trace
threshold (what it will discard), and the calibration input for both.

`switch_loss_telemetry`: dL/ds for adding s to every switch logit with the
features held fixed — the direction each actor-loss term pushes switching.

`paired_advantage_audit`: both V-trace estimators against behaviour-outcome
residuals on unique, first-use decision rows.
"""

import jax
import jax.numpy as jnp

from rl.environment.data import CAT_VF_SUPPORT
from rl.online.training.loss import policy_gradient_loss, support_hinge_loss
from rl.online.training.targets import compute_player_targets, reference_kl
from rl.utils import average

SUPPORT_LINES = (("p01", 0.01), ("p005", 0.005), ("p001", 0.001))


def masked_policy(
    log_policy: jax.Array, legal_mask: jax.Array, renormalise: bool = True
) -> jax.Array:
    """f32 probabilities over legal cells: exp(log_policy), illegal cells
    zeroed, and — unless `renormalise` is off — divided so the legal mass
    sums to 1."""
    policy = jnp.where(legal_mask, jnp.exp(log_policy.astype(jnp.float32)), 0.0)
    if renormalise:
        policy = policy / jnp.maximum(policy.sum(axis=-1, keepdims=True), 1e-8)
    return policy


def _fraction_below(policy, cells, line):
    return (cells & (policy < line)).sum(-1) / jnp.maximum(cells.sum(-1), 1)


def _minimum(policy, cells):
    return jnp.where(cells, policy, jnp.inf).min(-1)


def _median(policy, cells):
    ordered = jnp.sort(jnp.where(cells, policy, jnp.inf), axis=-1)
    middle = (jnp.maximum(cells.sum(-1), 1) - 1) // 2
    return jnp.take_along_axis(ordered, middle[..., None], axis=-1)[..., 0]


def legal_support_telemetry(
    log_policy: jax.Array,
    legal_mask: jax.Array,
    switch_cells: jax.Array,
    policy_mask: jax.Array,
) -> dict[str, jax.Array]:
    log_policy = jax.lax.stop_gradient(log_policy.astype(jnp.float32))
    policy = masked_policy(log_policy, legal_mask, renormalise=False)
    logs = {
        "player_support_median_prob": average(_median(policy, legal_mask), policy_mask),
        "player_support_legal_count": average(legal_mask.sum(-1), policy_mask),
    }
    # The split readouts average over the rows that have a cell of that kind.
    kinds = [("", legal_mask, policy_mask)]
    for kind, kind_cells in (
        ("switch_", legal_mask & switch_cells),
        ("move_", legal_mask & jnp.logical_not(switch_cells)),
    ):
        kinds.append((kind, kind_cells, policy_mask & kind_cells.any(-1)))
    for kind, cells, rows in kinds:
        logs[f"player_support_{kind}min_prob"] = average(_minimum(policy, cells), rows)
        for name, line in SUPPORT_LINES:
            logs[f"player_support_{kind}frac_below_{name}"] = average(
                _fraction_below(policy, cells, line), rows
            )
    return logs


def switch_loss_telemetry(
    log_policy,
    reg_log_policy,
    legal_mask,
    switch_cells,
    taken_switch,
    policy_mask,
    choice_mask,
    policy_ratio,
    advantages,
    raw_advantages,
    config,
):
    """dL/ds for adding s to every switch logit, holding features fixed.

    Positive means gradient descent suppresses switch odds. Terms include
    their training coefficients and share the actual policy-row denominator.
    This does not attribute shared-feature updates or Adam momentum. JVPs
    reuse the executable loss definitions, including the SPO/PPO selector.
    """
    log_policy = jax.lax.stop_gradient(log_policy.astype(jnp.float32))
    policy = masked_policy(log_policy, legal_mask)
    switch_mass = (policy * switch_cells).sum(-1)
    log_tangent = jnp.where(
        legal_mask, switch_cells.astype(jnp.float32) - switch_mass[..., None], 0.0
    )
    taken_tangent = taken_switch.astype(jnp.float32) - switch_mass
    policy_ratio = jax.lax.stop_gradient(policy_ratio.astype(jnp.float32))

    def pg_loss(ratios):
        return policy_gradient_loss(
            policy_ratios=ratios,
            advantages=advantages,
            valid=policy_mask,
            threshold=config.player_ppo_clip,
            objective=config.player_pg_objective,
        )

    def entropy_loss(log_probs):
        return average(
            jnp.where(legal_mask, jnp.exp(log_probs) * log_probs, 0.0).sum(-1),
            policy_mask,
        )

    def magnet_loss(log_probs):
        return average(reference_kl(log_probs, reg_log_policy, legal_mask), policy_mask)

    def support_loss(log_probs):
        rows, _, _ = support_hinge_loss(
            log_probs,
            legal_mask,
            config.player_support_tau,
            temperature=config.player_support_temperature,
        )
        return average(rows, policy_mask)

    gradients = {}
    gradients["pg"] = (
        config.player_pg_coef
        * jax.jvp(pg_loss, (policy_ratio,), (policy_ratio * taken_tangent,))[1]
    )
    for name, objective, coefficient in (
        ("entropy", entropy_loss, config.player_ent_coef),
        ("magnet", magnet_loss, config.player_mag_coef),
        ("support", support_loss, config.player_support_hinge_coef),
    ):
        gradients[name] = (
            config.player_pg_coef
            * coefficient
            * jax.jvp(objective, (log_policy,), (log_tangent,))[1]
        )
    logs = {
        f"player_switch_logit_grad_{name}": gradient
        for name, gradient in gradients.items()
    }
    logs["player_switch_logit_grad_actor_total"] = sum(gradients.values())
    logs["player_switch_mass_choice"] = average(switch_mass, choice_mask)
    for name, taken_mask in (
        ("switch", taken_switch),
        ("stay", ~taken_switch),
    ):
        selected = choice_mask & taken_mask
        logs[f"player_choice_{name}_count"] = selected.sum()
        logs[f"player_choice_{name}_adv_raw"] = average(raw_advantages, selected)
        logs[f"player_choice_{name}_adv_normalised"] = average(advantages, selected)
        logs[f"player_switch_logit_grad_pg_taken_{name}"] = (
            config.player_pg_coef
            * jax.jvp(
                pg_loss,
                (policy_ratio,),
                (jnp.where(selected, policy_ratio * taken_tangent, 0.0),),
            )[1]
        )
    return logs


def _sign_disagreement(negative, positive):
    return (negative < 0) & (positive > 0)


def paired_advantage_audit(
    batch, public_log_probs, privileged_log_probs, isr, isr_raw, config, axis
):
    """Compare both V-trace estimators with behaviour-outcome residuals.

    `isr` / `isr_raw` are the learner's own v-trace inputs (rho from the
    thresholded ratio, c from the raw one), so the advantages audited are
    the ones trained on. Uses each head's own baseline for its MC
    residual. The rho-weighted residual additionally matches the V-trace
    advantage's outer weight, so their difference isolates bootstrap
    disagreement on the same row.
    Outcomes are observational behaviour returns, not counterfactual Q
    labels. Sums and counts are emitted, so wandb's window mean is a mean
    over the sparse rows rather than a mean of per-batch means. First
    visits exclude replay duplicates; within-game row correlation remains.
    """
    if isinstance(batch.game_outcome, tuple) or isinstance(batch.reuse_count, tuple):
        return {}

    estimates = {}
    for name, log_probs in (
        ("public", public_log_probs),
        ("privileged", privileged_log_probs),
    ):
        targets, _ = compute_player_targets(
            batch, log_probs, isr, config, isr_raw=isr_raw
        )
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
        signals[f"{name}_td_negative_mc_positive"] = _sign_disagreement(
            advantage, residual
        )
        signals[f"{name}_td_positive_mc_negative"] = _sign_disagreement(
            residual, advantage
        )
    privileged_advantage = estimates["privileged"][0]
    public_advantage = estimates["public"][0]
    signals["priv_negative_public_positive"] = _sign_disagreement(
        privileged_advantage, public_advantage
    )
    signals["priv_positive_public_negative"] = _sign_disagreement(
        public_advantage, privileged_advantage
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
