"""The jitted learner update.

One function by design: it is jitted with static_argnames=["config"] and
donates the train states, so the whole update has to sit inside a single
traced call. Helpers it calls (targets, losses, telemetry) live beside it.
"""

import logging

import jax
import jax.numpy as jnp
import optax

from rl.environment.data import (
    CAT_VF_SUPPORT,
    PackedSetFeature,
)
from rl.environment.interfaces import Batch, BuilderActorInput, PlayerActorInput
from rl.environment.protos.features_pb2 import (
    FieldFeature,
)
from rl.model.constants import (
    SequenceGroup,
)
from rl.model.heads import HeadParams
from rl.model.history_encoder import major_arg_step_mask
from rl.model.utils import Params
from rl.online.artifact import Porygon2BuilderTrainState, Porygon2PlayerTrainState
from rl.online.config import Porygon2LearnerConfig
from rl.online.training.action_telemetry import (
    legal_support_telemetry,
    masked_policy,
    paired_advantage_audit,
    switch_loss_telemetry,
    voluntary_switch_telemetry,
)
from rl.online.training.loss import (
    appo_policy_loss,
    backward_kl_loss,
    clip_fraction,
    clipped_target_ratio,
    factorised_entropies,
    forward_kl_loss,
    mse_value_loss,
    policy_gradient_loss,
    uniform_kl_rows,
)
from rl.online.training.targets import (
    compute_builder_targets,
    compute_player_targets,
    reference_kl,
    unit_potential,
)
from rl.online.training.telemetry import (
    action_axis_masks,
    applied_delta_telemetry,
    calculate_r2,
    collect_batch_telemetry_data,
    critic_outcome_telemetry,
    head_param_telemetry,
    potential_telemetry,
    promote_map,
    ratio_ess_and_tail,
)
from rl.utils import average

logger = logging.getLogger(__name__)


def trunk_group_l2_logs(pred, value_mask) -> dict[str, jax.Array]:
    """player_trunk_out_row_l2_<group>: the trunk output's mean row L2 per
    SequenceGroup over the valid rows of the valid steps (the per-step sums
    and counts come from trunk.group_row_l2, so each row weighs once)."""
    step = value_mask[..., None]
    l2_sum = jnp.where(step, pred.trunk_out_group_l2_sum, 0).sum(axis=(0, 1))
    rows = jnp.where(step, pred.trunk_out_group_rows, 0).sum(axis=(0, 1))
    mean_l2 = l2_sum / rows.clip(min=1)
    return {
        f"player_trunk_out_row_l2_{group.name.lower()}": mean_l2[int(group)]
        for group in SequenceGroup
    }


def advantage_statistics(
    advantages: jax.Array, policy_mask: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Masked diagnostics in game-return units; never rescale the actor loss."""
    advantages = advantages.astype(jnp.float32)
    count = jnp.maximum(policy_mask.sum().astype(jnp.float32), 1.0)
    mean = jnp.where(policy_mask, advantages, 0.0).sum() / count
    variance = jnp.where(policy_mask, jnp.square(advantages - mean), 0.0).sum() / count
    std = jnp.sqrt(variance)
    return mean, std


def apply_player_gradients(
    player_state: Porygon2PlayerTrainState,
    gradients: Params,
    loss: jax.Array,
    frames: jax.Array,
    reference_rate: float,
    old_policy_snap_steps: int,
) -> tuple[Porygon2PlayerTrainState, jax.Array, jax.Array]:
    """Commit the pre-update-parameter EMA and the old-policy snapshot
    together with accepted optimiser steps.

    The snapshot is IMPACT's target update: on every accepted step whose
    new count is a multiple of `old_policy_snap_steps`, pi_old becomes an
    exact copy of the POST-update live parameters (RLlib syncs its target
    after the iteration's updates, TargetNetworkMixin tau=1). Written as a
    rate-0/1 incremental update rather than a branch so the copy is a
    fresh buffer under donation, like the reference."""
    if not 0.0 <= reference_rate <= 1.0:
        raise ValueError("reference_rate must be between zero and one")
    if old_policy_snap_steps < 1:
        raise ValueError("old_policy_snap_steps must be at least one")
    next_step_count = player_state.step_count + 1
    next_state = player_state.apply_gradients(grads=gradients)
    snap = (next_step_count % old_policy_snap_steps == 0).astype(jnp.float32)
    next_state = next_state.replace(
        step_count=next_step_count,
        frame_count=player_state.frame_count + frames,
        reg_params=optax.incremental_update(
            player_state.params, player_state.reg_params, reference_rate
        ),
        old_policy_params=optax.incremental_update(
            next_state.params, player_state.old_policy_params, snap
        ),
    )
    update_finite = jnp.isfinite(loss) & jnp.isfinite(optax.global_norm(gradients))
    next_state = jax.lax.cond(
        update_finite,
        lambda: next_state,
        lambda: player_state,
    )
    return next_state, update_finite, jnp.where(update_finite, reference_rate, 0.0)


def train_step(
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
    batch: Batch,
    config: Porygon2LearnerConfig,
):
    """Train for a single step.

    Every loss coefficient is a static config field. There used to be a
    RuntimeScalars pytree carrying host-varied coefficients as TRACED leaves
    so a host-side controller could vary them without recompiling — the
    hazard being that config is a jit static_argname, and static scalars
    retained ~5GB of executables per distinct value and OOM-killed run
    1326. Nothing varies them any more (the ramps, the exploiter zeroing
    and every controller that actuated them are gone), so they moved back
    into config. Reintroducing ANY host-varied coefficient — the magnet PI
    controller is the documented candidate — means reintroducing a traced
    pytree for it; never widen the static config with a value that changes
    during a run.
    """
    player_transitions = batch.player_transitions
    player_history = batch.player_history
    player_packed_history = batch.player_packed_history
    builder_transitions = batch.builder_transitions
    builder_history = batch.builder_history

    player_actor_input = PlayerActorInput(
        env=player_transitions.env_output,
        packed_history=player_packed_history,
        history=player_history,
    )

    head_params = HeadParams()

    player_estimate = player_state.apply_fn(
        player_state.params,
        player_actor_input,
        player_transitions.agent_output.actor_output,
        head_params,
    )

    reg_log_policy = player_state.apply_fn(
        player_state.reg_params,
        player_actor_input,
        player_transitions.agent_output.actor_output,
        HeadParams(),
    ).action_head.log_policy

    # pi_old: the V-trace target policy and the surrogate's clip reference.
    # Its values are NOT used -- V-trace bootstraps on the live critic, as
    # RLlib's loss reads model.value_function() beside target_model logits.
    old_policy_log_prob = player_state.apply_fn(
        player_state.old_policy_params,
        player_actor_input,
        player_transitions.agent_output.actor_output,
        HeadParams(),
    ).action_head.log_prob

    player_actor_action_head = player_transitions.agent_output.actor_output.action_head
    player_actor_log_prob = player_actor_action_head.log_prob

    float_dtype = player_actor_log_prob.dtype

    cat_vf_support = jnp.asarray(CAT_VF_SUPPORT, dtype=float_dtype)

    player_valid = jnp.bitwise_not(player_transitions.env_output.done)

    training_logs = {}

    # Directed-message sanity (2026-09-01): fraction of valid history steps
    # carrying an identified SOURCE row (a real major arg). Expect >> 0.5;
    # a collapse here says the src identification broke, not the game.
    history_field = player_history.field
    step_is_valid = history_field[..., FieldFeature.FIELD_FEATURE__VALID] > 0
    step_has_src = jax.vmap(major_arg_step_mask, in_axes=(1, 1), out_axes=1)(
        history_field, player_packed_history.edge_cache
    )
    training_logs["player_history_src_frac"] = jnp.where(
        step_is_valid.sum() > 0,
        (step_has_src & step_is_valid).sum() / step_is_valid.sum().clip(min=1),
        0.0,
    )

    # Already flat: the env mask IS the block-cell vector since 2026-08-31.
    flat_action_mask = player_transitions.env_output.action_mask
    # rho's raw material: pi_old/mu, the ratio V-trace truncates at one.
    old_policy_actor_ratio = jnp.exp(
        old_policy_log_prob.astype(jnp.float32)
        - player_actor_log_prob.astype(jnp.float32)
    )

    if config.player_privileged_targets:
        estimate_value_log_probs = player_estimate.priv_value_head.log_probs
    else:
        estimate_value_log_probs = player_estimate.value_head.log_probs
    potential_values = None
    if config.player_potential_strength > 0:
        potential_values = player_estimate.potential_head.logits
    player_targets, channel_logs = compute_player_targets(
        batch,
        value_log_probs=estimate_value_log_probs,
        isr=old_policy_actor_ratio,
        config=config,
        potential_values=potential_values,
    )
    training_logs.update(channel_logs)
    policy_mask = player_targets.policy_mask
    value_mask = player_targets.value_mask
    axis = action_axis_masks(flat_action_mask, player_actor_action_head.action_index)

    # Fresh-vs-replayed value error: the memorisation gap. A network with
    # healthy plasticity fits fresh and replayed trajectories about equally;
    # replayed error falling while fresh error rises means the buffer is
    # being memorised — the plasticity signature that should gate any
    # shrink-and-perturb, unlike league stagnation which has many causes.
    # Per-group means are NaN in batches with no fresh (or no replayed)
    # member; wandb line plots skip them.
    if not isinstance(batch.reuse_count, tuple):
        target_value = player_estimate.value_head.expectation
        return_value = player_targets.win_returns @ cat_vf_support
        value_sq_err = jnp.square(target_value - return_value)
        vm = value_mask.astype(value_sq_err.dtype)
        per_traj_err = (value_sq_err * vm).sum(axis=0) / (vm.sum(axis=0) + 1e-8)
        fresh = batch.reuse_count[0] == 0
        replayed = ~fresh
        fresh_err = per_traj_err.mean(where=fresh)
        replay_err = per_traj_err.mean(where=replayed)
        training_logs.update(
            {
                "plasticity_fresh_value_err": fresh_err,
                "plasticity_replay_value_err": replay_err,
                "plasticity_value_err_reuse_gap": fresh_err - replay_err,
            }
        )
    pg_advantages = player_targets.pg_advantages.astype(jnp.float32)
    pg_adv_mean, pg_adv_std = advantage_statistics(pg_advantages, policy_mask)
    training_logs["player_pg_adv_mean"] = pg_adv_mean
    training_logs["player_pg_adv_std"] = pg_adv_std

    player_targets = promote_map(player_targets, float_dtype)

    v_target = player_estimate.value_head.expectation.astype(jnp.float32)
    # EMAgnet regularises the policy objective, leaving critic targets in game units.
    # An action was actually taken here — including on forced single-option
    # steps, which policy_mask excludes — but not on terminal rows.
    acted_mask = value_mask & jnp.logical_not(player_transitions.env_output.done)
    training_logs["player_batch_decisions"] = acted_mask.sum(dtype=jnp.int32)
    # One derivation of the switch/move predicates for the whole step —
    # the panels below, critic_outcome_telemetry and the policy-loss
    # telemetry all read THESE, so they cannot drift apart again
    # (telemetry.ActionAxisMasks).
    training_logs.update(
        paired_advantage_audit(
            batch,
            player_estimate.value_head.log_probs,
            player_estimate.priv_value_head.log_probs,
            old_policy_actor_ratio,
            config,
            axis,
        )
    )
    training_logs.update(voluntary_switch_telemetry(batch, axis, acted_mask))
    taken_switch = axis.taken_switch
    has_move = axis.has_move
    voluntary_switch_mask = acted_mask & taken_switch & has_move
    forced_switch_mask = acted_mask & taken_switch & jnp.logical_not(has_move)
    move_mask = acted_mask & jnp.logical_not(taken_switch)
    training_logs.update(
        potential_telemetry(
            unit_potential(player_transitions.env_output),
            player_targets.potential_advantages,
            pg_advantages,
            policy_mask,
            value_mask,
            voluntary_switch_mask,
            move_mask,
        )
    )
    # Realised behaviour frequency on the axis every collapse formed in.
    training_logs["player_taken_switch_frac"] = average(
        taken_switch.astype(jnp.float32), acted_mask
    )
    training_logs["player_taken_voluntary_switch_frac"] = average(
        (taken_switch & has_move).astype(jnp.float32), acted_mask
    )

    # Off-policy attenuation audit, split by the TAKEN modality. isr =
    # pi_old/mu_actor is what v-trace multiplies its TD errors by
    # (targets.py: rho_t = c_t = min(1, isr)). As pi(switch) decays, isr on
    # switch-taken rows falls
    # below 1 and those rows contribute proportionally less.
    #
    # NOTE this is the CORRECT importance correction, not a bug: the
    # readout is effective sample size on switch rows, not bias. A
    # widening gap between the two means the learner is hearing the
    # switch evidence ever more faintly as the collapse deepens, which
    # is a self-reinforcing loop even though every individual update is
    # properly weighted. below1_frac is the cleaner signal than the
    # mean (isr is heavy-tailed on the upside).
    isr_f32 = old_policy_actor_ratio
    # RLlib's mean_IS / var_IS: the clamped mu/pi_old factor the surrogate
    # multiplies in, and how often the cap bit.
    behaviour_old_ratio = jnp.exp(
        player_actor_log_prob.astype(jnp.float32)
        - old_policy_log_prob.astype(jnp.float32)
    )
    training_logs["player_behaviour_old_ratio_mean"] = average(
        jnp.minimum(behaviour_old_ratio, config.player_behaviour_ratio_clip),
        policy_mask,
    )
    training_logs["player_behaviour_old_ratio_clip_frac"] = average(
        behaviour_old_ratio >= config.player_behaviour_ratio_clip, policy_mask
    )
    training_logs["player_isr_switch_voluntary"] = average(
        isr_f32, voluntary_switch_mask
    )
    training_logs["player_isr_switch_forced"] = average(isr_f32, forced_switch_mask)
    training_logs["player_isr_move"] = average(isr_f32, move_mask)
    training_logs["player_isr_below1_switch_voluntary"] = average(
        (isr_f32 < 1.0).astype(jnp.float32), voluntary_switch_mask
    )
    training_logs["player_isr_below1_move"] = average(
        (isr_f32 < 1.0).astype(jnp.float32), move_mask
    )

    if not isinstance(batch.game_outcome, tuple):
        # The realised-outcome instruments, a property of the games, not
        # of any critic.
        training_logs.update(
            critic_outcome_telemetry(
                game_outcome=batch.game_outcome,
                game_length=batch.game_length,
                game_step_offset=batch.game_step_offset,
                v_target=v_target,
                flat_action_mask=flat_action_mask,
                masks=axis,
                acted_mask=acted_mask,
                value_mask=value_mask,
            )
        )
    if not isinstance(batch.reuse_count, tuple):
        fresh_cols = batch.reuse_count[0] == 0
        ~fresh_cols
        vm_fresh = value_mask & fresh_cols[None, :]
        training_logs["player_value_r2_fresh"] = jnp.where(
            vm_fresh.any(),
            calculate_r2(
                value_prediction=player_estimate.value_head.expectation.astype(
                    jnp.float32
                ),
                value_target=(player_targets.win_returns @ cat_vf_support).astype(
                    jnp.float32
                ),
                mask=vm_fresh,
            ),
            0.0,
        )

    def player_loss_fn(params: Params):

        learner_player_pred = player_state.apply_fn(
            params,
            player_actor_input,
            player_transitions.agent_output.actor_output,
            head_params,
        )

        learner_value_head = learner_player_pred.value_head
        learner_action_head = learner_player_pred.action_head
        learner_log_prob = learner_action_head.log_prob

        learner_actor_log_ratio = learner_log_prob - player_actor_log_prob
        learner_actor_ratio = jnp.exp(learner_actor_log_ratio)
        learner_actor_ess, learner_actor_ratio_tail = ratio_ess_and_tail(
            learner_actor_ratio, policy_mask, 2.0
        )

        loss_v_win = average(
            optax.softmax_cross_entropy(
                logits=learner_value_head.logits.astype(jnp.float32),
                labels=player_targets.win_returns.astype(jnp.float32),
            ),
            value_mask,
        )
        # The privileged critic: SAME labels, SAME mask -- the deployable
        # head above stays trained unchanged as the matched control, and
        # the priv-vs-deploy R2 pair is the discriminator the 2026-08-25
        # falsification never had.
        learner_priv_value_head = learner_player_pred.priv_value_head
        loss_v_win_priv = average(
            optax.softmax_cross_entropy(
                logits=learner_priv_value_head.logits.astype(jnp.float32),
                labels=player_targets.win_returns.astype(jnp.float32),
            ),
            value_mask,
        )
        # The public critic: the same labels and mask a third time, over
        # the public tier alone -- what the common-knowledge state is worth
        # under the current policies, with nothing private in reach.
        learner_public_value_head = learner_player_pred.public_value_head
        loss_v_win_public = average(
            optax.softmax_cross_entropy(
                logits=learner_public_value_head.logits.astype(jnp.float32),
                labels=player_targets.win_returns.astype(jnp.float32),
            ),
            value_mask,
        )
        # The PBRS potential channel's head (2026-09-11): regressed on its
        # own channel returns over live nonterminal rows (its value is forced
        # 0 on done rows, so their label carries nothing). Coefficient 1:
        # its input is under stop_gradient, so this is its only gradient --
        # though that gradient still enters the global clip norm.
        loss_potential = 0.0
        potential_logs = {}
        if config.player_potential_strength > 0:
            potential_mask = value_mask & jnp.logical_not(
                player_transitions.env_output.done
            )
            potential_prediction = learner_player_pred.potential_head.logits
            potential_label = player_targets.potential_returns.astype(jnp.float32)
            loss_potential = mse_value_loss(
                pred=potential_prediction, target=potential_label, valid=potential_mask
            )
            potential_logs = {
                "player_loss_potential": loss_potential,
                "player_potential_head_r2": calculate_r2(
                    value_prediction=potential_prediction,
                    value_target=potential_label,
                    mask=potential_mask,
                ),
                # Distance from inert: the head against its EXACT target, -Phi.
                "player_potential_head_fit_r2": calculate_r2(
                    value_prediction=potential_prediction,
                    value_target=-unit_potential(player_transitions.env_output),
                    mask=potential_mask,
                ),
            }
        action_head_entropy = average(learner_action_head.entropy, policy_mask)
        action_head_normalized_entropy = average(
            learner_action_head.normalized_entropy, policy_mask
        )

        loss_actor_forward_kl = forward_kl_loss(
            policy_ratio=learner_actor_ratio,
            log_policy_ratio=learner_actor_log_ratio,
            valid=policy_mask,
        )
        loss_actor_backward_kl = backward_kl_loss(
            policy_ratio=learner_actor_ratio,
            log_policy_ratio=learner_actor_log_ratio,
            valid=policy_mask,
        )
        normalized_modality_entropy = average(
            learner_action_head.normalized_modality_entropy, policy_mask
        )
        # Real-choice rows on the stay/switch axis: both a switch and a
        # a MOVE are legal. This is the slice the collapse forms in, and
        # every policy-modality readout below is scoped to it.
        #
        # STRICT: a switch and a legal MOVE, never a WILDCARD / OTHER /
        # TARGET cell. A stay/switch decision only means something
        # when staying and attacking is actually on offer.
        switch_actions = axis.switch_cells
        switch_choice_mask = policy_mask & axis.has_both

        learner_log_policy = learner_action_head.log_policy
        pi_learner = masked_policy(learner_log_policy, flat_action_mask)

        macro_valid = policy_mask & (axis.num_legal_modalities >= 2)
        micro_valid = policy_mask & (axis.taken_modality_count >= 2)

        loss_pg = appo_policy_loss(
            learner_log_prob=learner_log_prob,
            behaviour_log_prob=player_actor_log_prob,
            old_policy_log_prob=old_policy_log_prob,
            advantages=pg_advantages,
            valid=policy_mask,
            clip_ppo=config.player_ppo_clip,
            behaviour_ratio_clip=config.player_behaviour_ratio_clip,
        )
        surrogate_ratio = clipped_target_ratio(
            learner_log_prob=learner_log_prob,
            behaviour_log_prob=player_actor_log_prob,
            old_policy_log_prob=old_policy_log_prob,
            behaviour_ratio_clip=config.player_behaviour_ratio_clip,
        )
        # Modality entropies observe collapse; only joint entropy enters the loss.
        h_macro_rows, h_micro_rows = factorised_entropies(
            learner_log_policy, axis.taken_modality, flat_action_mask
        )
        entropy_macro = average(h_macro_rows, macro_valid)
        entropy_micro_taken = average(h_micro_rows, micro_valid)
        loss_entropy = -average(
            learner_action_head.entropy.astype(jnp.float32), policy_mask
        )
        magnet_kl_rows = reference_kl(
            learner_log_policy, reg_log_policy, flat_action_mask
        )
        loss_mag = average(magnet_kl_rows, policy_mask)
        loss_uniform_kl = average(
            uniform_kl_rows(learner_log_policy, flat_action_mask), policy_mask
        )

        # Modality decomposition of the two factors any taken-action
        # update is throttled by: pi mass and the observer critic's |A|,
        # over legal switch vs non-switch cells of real-choice rows.
        # Loss-agnostic: prob_ratio falling with absadv_ratio ~ 1 is the
        # starvation signature to watch.
        pg_row = switch_choice_mask[..., None]
        pg_switch_cells = flat_action_mask & switch_actions & pg_row
        pg_move_cells = flat_action_mask & jnp.logical_not(switch_actions) & pg_row

        def modality_ratio(numerator, denominator):
            return numerator / jnp.maximum(denominator, 1e-8)

        policy_prob_switch = average(pi_learner, pg_switch_cells)
        policy_prob_move = average(pi_learner, pg_move_cells)

        pg_logs = dict(
            player_loss_pg=loss_pg,
            player_loss_entropy=loss_entropy,
            player_loss_uniform_kl=loss_uniform_kl,
            player_entropy_macro=entropy_macro,
            player_entropy_micro_taken=entropy_micro_taken,
            player_policy_prob_switch=policy_prob_switch,
            player_policy_prob_move=policy_prob_move,
            player_policy_prob_ratio=modality_ratio(
                policy_prob_switch, policy_prob_move
            ),
            player_ref_kl=loss_mag,
            # Rows outside the PPO band: how much of the batch the trust
            # region is actually holding back, RLlib's clip readout.
            player_ppo_clip_frac=clip_fraction(
                policy_ratios=surrogate_ratio,
                valid=policy_mask,
                clip_ppo=config.player_ppo_clip,
            ),
            player_surrogate_ratio_mean=average(surrogate_ratio, policy_mask),
        )
        pg_logs.update(
            legal_support_telemetry(
                learner_log_policy, flat_action_mask, switch_actions, policy_mask
            )
        )
        pg_logs.update(
            switch_loss_telemetry(
                learner_log_policy,
                reg_log_policy,
                flat_action_mask,
                switch_actions,
                axis.taken_switch,
                policy_mask,
                switch_choice_mask,
                learner_log_prob,
                player_actor_log_prob,
                old_policy_log_prob,
                pg_advantages,
                config,
            )
        )
        loss = (
            config.player_pg_coef
            * (
                loss_pg
                + config.player_ent_coef * loss_entropy
                + config.player_mag_coef * loss_mag
                + config.player_uniform_kl_coef * loss_uniform_kl
            )
            # v: one critic, on the deploy-time information set.
            + config.player_value_head_loss_coef * loss_v_win
            + config.player_priv_value_head_loss_coef * loss_v_win_priv
            + config.player_public_value_head_loss_coef * loss_v_win_public
            # The potential channel's head, unscaled (see above).
            + loss_potential
        )

        return loss, dict(
            **pg_logs,
            player_loss_v_win=loss_v_win,
            player_loss_v_win_priv=loss_v_win_priv,
            player_loss_v_win_public=loss_v_win_public,
            # Trunk over-smoothing (cosine up / participation down = rows
            # converging); the offline per-block twin is
            # rl/offline/trunk_homogeneity.py.
            player_trunk_row_cosine=average(
                learner_player_pred.trunk_row_cosine, value_mask
            ),
            player_trunk_row_participation=average(
                learner_player_pred.trunk_row_participation, value_mask
            ),
            # The trunk's output L2 per SequenceGroup, every valid row of
            # every valid step weighted once (trunk.group_row_l2). Rows
            # enter at RMS 1 = L2 16 at width 256, so this is what the six
            # blocks wrote onto each group; a group sitting at its input
            # norm is one the trunk does not revise.
            **trunk_group_l2_logs(learner_player_pred, value_mask),
            # The history encoder's step GAT and write gate
            # (history_encoder.history_step_stats): per-trajectory scalars
            # broadcast over T, so this is the valid-step-weighted batch mean.
            player_history_step_attn_entropy=average(
                learner_player_pred.history_step_attn_entropy, value_mask
            ),
            player_history_step_attn_to_src=average(
                learner_player_pred.history_step_attn_to_src, value_mask
            ),
            player_history_step_attn_to_src_uniform=average(
                learner_player_pred.history_step_attn_to_src_uniform, value_mask
            ),
            player_history_gate_mean=average(
                learner_player_pred.history_gate_mean, value_mask
            ),
            # Diagnostic only: no coefficient scales it.
            player_loss_kl=loss_actor_backward_kl,
            # Per head entropies (diagnostics only — no longer regularized)
            player_action_entropy=action_head_entropy,
            player_action_normalized_entropy=action_head_normalized_entropy,
            player_normalized_modality_entropy=normalized_modality_entropy,
            player_learner_actor_ratio=average(learner_actor_ratio, policy_mask),
            player_learner_actor_ess=learner_actor_ess,
            player_learner_actor_ratio_tail_gt2=learner_actor_ratio_tail,
            player_learner_actor_forward_kl=loss_actor_forward_kl,
            # Modality-resolved split of the SAME k3 estimator. The
            # global mean is an expectation over the policy, so drift
            # confined to a modality carrying ~0.2 mass is diluted to
            # near nothing — which is why the replay reuse controller
            # (whose set-point is the global mean) cannot fire on a
            # collapsing switch modality even in principle. These two
            # de-average it by the taken modality.
            #
            # LIMITATION: this is still a mu-SAMPLED estimator over
            # taken actions, not a full-support KL, because actors do
            # not persist their full log_policy (player_model.py gates
            # log_policy on cfg.train). An exact full-support modality
            # KL against the actor would need that field stored.
            player_learner_actor_forward_kl_switch=forward_kl_loss(
                policy_ratio=learner_actor_ratio,
                log_policy_ratio=learner_actor_log_ratio,
                valid=policy_mask & taken_switch,
            ),
            player_learner_actor_forward_kl_move=forward_kl_loss(
                policy_ratio=learner_actor_ratio,
                log_policy_ratio=learner_actor_log_ratio,
                valid=policy_mask & jnp.logical_not(taken_switch),
            ),
            player_learner_actor_backward_kl=loss_actor_backward_kl,
            player_value_head_r2=calculate_r2(
                value_prediction=learner_value_head.expectation,
                value_target=player_targets.win_returns @ cat_vf_support,
                mask=value_mask,
            ),
            # THE discriminator for the privileged premise: this pair on one
            # panel. priv < deploy sustained past 30k is this pass's
            # pre-registered abort.
            player_priv_value_head_r2=calculate_r2(
                value_prediction=learner_priv_value_head.expectation,
                value_target=player_targets.win_returns @ cat_vf_support,
                mask=value_mask,
            ),
            player_public_value_head_r2=calculate_r2(
                value_prediction=learner_public_value_head.expectation,
                value_target=player_targets.win_returns @ cat_vf_support,
                mask=value_mask,
            ),
            # Mean absolute priv-minus-deploy expectation gap: what the
            # privileged input is worth, re-measured live.
            player_priv_value_gap=average(
                jnp.abs(
                    learner_priv_value_head.expectation - learner_value_head.expectation
                ),
                value_mask,
            ),
            player_nll_sum=(
                batch.player_transitions.agent_output.actor_output.action_head.log_prob
                * policy_mask
            )
            .sum(axis=0)
            .mean(),
            **potential_logs,
        )

    player_grad_fn = jax.value_and_grad(player_loss_fn, has_aux=True)
    (player_loss_val, player_logs), player_grads = player_grad_fn(player_state.params)

    prev_player_state = player_state
    player_state, player_update_finite, reference_update_rate = apply_player_gradients(
        player_state,
        player_grads,
        player_loss_val,
        player_valid.sum(),
        config.player_reg_ema_rate,
        config.player_old_policy_snap_steps,
    )
    training_logs.update(player_logs)
    training_logs.update(
        dict(
            player_reg_ema_rate=reference_update_rate,
            # Updates since pi_old was last copied (0 = this step copied).
            player_old_policy_age=(
                player_state.step_count % config.player_old_policy_snap_steps
            ),
            player_loss=player_loss_val,
            player_param_norm=optax.global_norm(player_state.params),
            player_gradient_norm=optax.global_norm(player_grads),
            # Head learning readouts: the rms of the pointer kernels
            # against their known init, and per-subtree grad norms
            # (pre-clip).
            **head_param_telemetry(prev_player_state.params, player_grads),
            **applied_delta_telemetry(prev_player_state.params, player_state.params),
            player_win_returns_sum=average(
                player_targets.win_returns.sum(axis=-1), value_mask
            ),
            player_win_returns_min=jnp.min(
                jnp.where(value_mask[..., None], player_targets.win_returns, 1000.0)
            ),
            player_policy_mask_sum=policy_mask.sum(),
            player_value_mask_sum=value_mask.sum(),
            # Which lattice combo this variant was compiled for (static
            # per executable) — the retuning readout for
            # config.player_shape_lattice.
            player_shape_T=float(batch.player_transitions.env_output.done.shape[0]),
            player_shape_H=float(batch.player_history.field.shape[0]),
            player_policy_value_mask_ratio=policy_mask.sum()
            / (value_mask.sum() + 1e-8),
            player_update_skipped=1.0 - player_update_finite.astype(jnp.float32),
        )
    )
    training_logs.update(
        {
            f"player_{k}_gradient_norm": optax.global_norm(player_grads["params"][k])
            for k in player_grads["params"]
        }
    )
    training_logs.update(
        {
            f"player_{k}_gradient_norm": optax.global_norm(
                player_grads["params"]["encoder"][k]
            )
            for k in player_grads["params"]["encoder"]
            if any(substring in k for substring in ("decoder", "encoder"))
        }
    )
    if config.player_potential_strength > 0:
        # The potential head's gradient enters the GLOBAL clip norm
        # (artifact.py clip_by_global_norm): its share is the coupling a
        # stop_gradient reach test cannot see.
        training_logs["player_potential_head_grad_share"] = optax.global_norm(
            player_grads["params"]["potential_head"]
        ) / (optax.global_norm(player_grads) + 1e-12)

    if config.smogon_format != "randombattle":
        builder_actor_input = BuilderActorInput(
            env=builder_transitions.env_output,
            history=builder_history,
        )

        builder_target_pred = builder_state.apply_fn(
            builder_state.target_params,
            builder_actor_input,
            builder_transitions.agent_output.actor_output,
            HeadParams(),
        )

        builder_actor_action_head = (
            builder_transitions.agent_output.actor_output.action_head
        )

        builder_actor_log_prob = builder_actor_action_head.log_prob
        builder_target_log_prob = builder_target_pred.action_head.log_prob
        builder_actor_target_log_ratio = (
            builder_actor_log_prob - builder_target_log_prob
        )
        builder_actor_target_ratio = jnp.exp(builder_actor_target_log_ratio)
        builder_target_actor_ratio = jnp.exp(-builder_actor_target_log_ratio)
        builder_actor_target_clipped_ratio = jnp.clip(
            builder_actor_target_ratio, min=0.0, max=2.0
        )

        builder_valid = jnp.bitwise_not(builder_transitions.env_output.done)
        # Chunked unrolls: compute_builder_targets reads the game outcome
        # off player win_reward[-1], which only the game's TERMINAL chunk
        # carries — a mid-game chunk's builder rows would grade the team
        # against a zero payoff. average()-based builder losses are
        # empty-mask-safe (0, not NaN) if a batch happens to hold no
        # terminal chunk.
        builder_valid = builder_valid & player_transitions.env_output.done.any(axis=0)[
            None, :
        ].astype(jnp.bool_)
        builder_targets = compute_builder_targets(
            batch,
            builder_target_pred,
            builder_target_actor_ratio,
            lambda_=config.builder_lambda,
            entropy_normalising_constant=config.builder_entropy_prediction_normalising_constant,
        )
        builder_returns = promote_map(builder_targets, float_dtype)

        builder_advantages = (
            builder_targets.win_advantages
            + config.builder_entropy_advantage_scale * builder_targets.ent_advantages
        )

        builder_win_return_correction = builder_targets.win_returns.sum(
            axis=-1, keepdims=True
        )
        builder_win_returns = (
            builder_targets.win_returns / builder_win_return_correction
        )

        def builder_loss_fn(params: Params):

            pred = builder_state.apply_fn(
                params,
                builder_actor_input,
                builder_transitions.agent_output.actor_output,
                HeadParams(),
            )

            learner_value_head = pred.value_head
            learner_action_head = pred.action_head
            learner_conditional_entropy_head = pred.conditional_entropy_head
            learner_log_prob = learner_action_head.log_prob

            learner_actor_log_ratio = learner_log_prob - builder_actor_log_prob
            learner_actor_ratio = jnp.exp(learner_actor_log_ratio)

            learner_target_log_ratio = learner_log_prob - builder_target_log_prob
            learner_target_ratio = jnp.exp(learner_target_log_ratio)

            builder_policy_ratio = (
                builder_actor_target_clipped_ratio * learner_actor_ratio
            )

            # Calculate the losses. threshold is REQUIRED (keyword-only,
            # no default): omitting it was a latent TypeError on the first
            # builder train step, unreachable only because randombattle
            # skips this branch entirely.
            loss_pg = policy_gradient_loss(
                policy_ratios=builder_policy_ratio,
                advantages=builder_advantages,
                valid=builder_valid,
                threshold=config.builder_ppo_clip_threshold,
            )

            loss_v = average(
                optax.softmax_cross_entropy(
                    logits=learner_value_head.logits, labels=builder_win_returns
                ),
                builder_valid,
            )

            loss_builder_entropy = -average(learner_action_head.entropy, builder_valid)
            loss_builder_conditional_entropy = mse_value_loss(
                pred=learner_conditional_entropy_head.logits,
                target=builder_targets.ent_returns,
                valid=builder_valid,
            )

            loss_forward_kl = forward_kl_loss(
                policy_ratio=learner_actor_ratio,
                log_policy_ratio=learner_actor_log_ratio,
                valid=builder_valid,
            )
            loss_backward_kl = backward_kl_loss(
                policy_ratio=learner_actor_ratio,
                log_policy_ratio=learner_actor_log_ratio,
                valid=builder_valid,
            )

            human_valid_mask = (
                builder_transitions.env_output.curr_attribute
                != PackedSetFeature.PACKED_SET_FEATURE__HIDDENPOWERTYPE
            ) & (
                builder_transitions.env_output.curr_attribute
                != PackedSetFeature.PACKED_SET_FEATURE__GENDER
            )

            loss_human = average(
                learner_action_head.magnet_kl, valid=builder_valid & human_valid_mask
            )

            loss = (
                config.builder_policy_loss_coef * loss_pg
                + config.builder_value_loss_coef * loss_v
                + config.builder_kl_loss_coef * loss_backward_kl
                + config.builder_human_loss_coef * loss_human
                + config.builder_conditional_entropy_loss_coef
                * loss_builder_conditional_entropy
                + config.builder_entropy_coef * loss_builder_entropy
            )

            return loss, dict(
                builder_loss_pg=loss_pg,
                builder_loss_v=loss_v,
                builder_loss_kl_rl=loss_backward_kl,
                builder_loss_entropy=loss_builder_entropy,
                builder_loss_conditional_entropy=loss_builder_conditional_entropy,
                builder_loss_human=loss_human,
                builder_learner_actor_ratio=average(learner_actor_ratio, builder_valid),
                builder_learner_target_ratio=average(
                    learner_target_ratio, builder_valid
                ),
                builder_learner_actor_approx_kl=loss_forward_kl,
                builder_learner_condtional_entropy_head_mean=average(
                    learner_conditional_entropy_head.logits, builder_valid
                ),
                builder_learner_condtional_entropy_head_std=jnp.std(
                    learner_conditional_entropy_head.logits, where=builder_valid
                ),
                builder_value_function_r2=calculate_r2(
                    value_prediction=learner_value_head.expectation,
                    value_target=builder_returns.win_returns @ cat_vf_support,
                    mask=builder_valid,
                ),
            )

        builder_grad_fn = jax.value_and_grad(builder_loss_fn, has_aux=True)
        (builder_loss_val, builder_logs), builder_grads = builder_grad_fn(
            builder_state.params
        )
        training_logs.update(builder_logs)
        training_logs.update(
            dict(
                builder_loss=builder_loss_val,
                builder_win_return_correction=average(
                    builder_win_return_correction.reshape(builder_valid.shape),
                    builder_valid,
                ),
                builder_nll_sum=(
                    batch.builder_transitions.agent_output.actor_output.action_head.log_prob
                    * builder_valid
                )
                .sum(axis=0)
                .mean(),
                builder_param_norm=optax.global_norm(builder_state.params),
                builder_gradient_norm=optax.global_norm(builder_grads),
                builder_norm_adv_mean=average(builder_advantages, builder_valid),
                builder_norm_adv_std=builder_advantages.std(where=builder_valid),
            )
        )
        prev_builder_state = builder_state
        builder_state = builder_state.apply_gradients(grads=builder_grads)
        builder_state = builder_state.replace(
            step_count=builder_state.step_count + 1,
            frame_count=builder_state.frame_count + builder_valid.sum(),
            target_params=optax.incremental_update(
                builder_state.params,
                builder_state.target_params,
                config.builder_ema_update_rate,
            ),
        )
        # Same non-finite gate as the player update above.
        builder_update_finite = jnp.isfinite(builder_loss_val) & jnp.isfinite(
            optax.global_norm(builder_grads)
        )
        builder_state = jax.tree.map(
            lambda new, old: jnp.where(builder_update_finite, new, old),
            builder_state,
            prev_builder_state,
        )
        training_logs["builder_update_skipped"] = 1.0 - builder_update_finite.astype(
            jnp.float32
        )

    training_logs.update(collect_batch_telemetry_data(batch, config))
    training_logs.update(
        dict(
            player_frame_count=player_state.frame_count,
            builder_frame_count=builder_state.frame_count,
            training_step=player_state.step_count,
        )
    )

    return player_state, builder_state, training_logs


# Module-level, not per-Learner: exactly ONE Learner exists for the life of
# the process, so this is only ever compiled once either way. Kept
# module-level so the compiled fn survives a Learner rebuild in tests and
# scripts rather than paying the compile again.
TRAIN_STEP_JIT = jax.jit(
    train_step,
    static_argnames=["config"],
    donate_argnames=["player_state", "builder_state"],
)
