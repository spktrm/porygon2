"""Directional actor-loss diagnostics; no extra model forward or loss force."""

import jax
import jax.numpy as jnp

from rl.online.training.loss import policy_gradient_loss, uniform_kl_modalities
from rl.online.training.targets import reference_kl
from rl.utils import average


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
    policy = jnp.where(legal_mask, jnp.exp(log_policy), 0.0)
    policy /= jnp.maximum(policy.sum(-1, keepdims=True), 1e-8)
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

    def modality_loss(log_probs):
        return average(uniform_kl_modalities(log_probs, legal_mask), policy_mask)

    gradients = {}
    gradients["pg"] = (
        config.player_pg_coef
        * jax.jvp(pg_loss, (policy_ratio,), (policy_ratio * taken_tangent,))[1]
    )
    for name, objective, coefficient in (
        ("entropy", entropy_loss, config.player_ent_coef),
        ("magnet", magnet_loss, config.player_mag_coef),
        ("modality", modality_loss, config.player_uniform_kl_coef),
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
