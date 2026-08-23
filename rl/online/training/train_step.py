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
    FLAT_MODALITY_MASK,
    PackedSetFeature,
)
from rl.environment.interfaces import (
    Batch,
    BuilderActorInput,
    PlayerActorInput,
)
from rl.environment.protos.service_pb2 import ModalityEnum
from rl.model.heads import HeadParams
from rl.model.utils import Params
from rl.online.artifact import (
    Porygon2BuilderTrainState,
    Porygon2PlayerTrainState,
)
from rl.online.config import Porygon2LearnerConfig
from rl.online.training.loss import (
    warmup_scale,
    backward_kl_loss,
    forward_kl_loss,
    mse_value_loss,
    policy_gradient_loss,
)
from rl.online.training.targets import (
    compute_builder_targets,
    compute_player_targets,
    compute_q_targets,
    reference_kl,
    rnad_transformed_q,
)
from rl.online.training.telemetry import (
    critic_outcome_telemetry,
    calculate_r2,
    collect_batch_telemetry_data,
    promote_map,
)
from rl.utils import average

logger = logging.getLogger(__name__)

# Why a snapshot was added to the league. "dominant" is the healthy path
# (the agent beat its own history); "overdue" means only the frame budget

def train_step(
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
    batch: Batch,
    config: Porygon2LearnerConfig,
):
    """Train for a single step.

    Every loss coefficient is a static config field. There used to be a
    RuntimeScalars pytree carrying magnet_coef/neurd_coef as TRACED leaves
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

    player_target_pred = player_state.apply_fn(
        player_state.target_params,
        player_actor_input,
        player_transitions.agent_output.actor_output,
        HeadParams(),
    )

    # R-NaD reference policy: the slow-EMA reg_params' full-support
    # log-policy on the same batch (one extra forward; stop-gradient by
    # construction — reg_params are not the differentiated leaf).
    reg_log_policy = player_state.apply_fn(
        player_state.reg_params,
        player_actor_input,
        player_transitions.agent_output.actor_output,
        HeadParams(),
    ).action_head.log_policy

    player_actor_action_head = player_transitions.agent_output.actor_output.action_head
    player_actor_log_prob = player_actor_action_head.log_prob
    player_target_log_prob = player_target_pred.action_head.log_prob

    float_dtype = player_actor_log_prob.dtype

    cat_vf_support = jnp.asarray(CAT_VF_SUPPORT, dtype=float_dtype)

    player_valid = jnp.bitwise_not(player_transitions.env_output.done)

    training_logs = {}

    target_actor_log_ratio = player_target_log_prob - player_actor_log_prob
    target_actor_ratio = jnp.exp(target_actor_log_ratio)
    # IMPACT clipped-target correction: recenters the surrogate on the
    # slowly-moving fast target so the trust region stays stable across
    # replay reuse instead of resetting to the per-sample behavior policy.
    actor_target_clipped_ratio = jnp.exp(-target_actor_log_ratio).clip(min=0.0, max=2.0)

    # IMPACT-style targets: the fast target network supplies the v-trace
    # reference policy and value/kl bootstraps.
    player_targets, channel_logs = compute_player_targets(
        batch,
        value_log_probs=player_target_pred.value_head.log_probs,
        isr=target_actor_ratio,
        config=config,
    )
    training_logs.update(channel_logs)
    policy_mask = player_targets.policy_mask
    value_mask = player_targets.value_mask
    # Step-2 warm-up (docs/critic-weakness-analysis.md): NeuRD ramps in
    # over the lineage's first player_neurd_warmup_steps so the critic
    # gains coverage under the broad launch behaviour distribution before
    # it steers that distribution; reg_params stays the launch snapshot
    # meanwhile. Both are functions of the traced step_count — no static
    # config variation, no second executable.
    neurd_scale = warmup_scale(player_state.step_count, config.player_neurd_warmup_steps)
    reg_ema_rate = jnp.where(
        neurd_scale >= 1.0, jnp.float32(config.player_reg_ema_rate), jnp.float32(0.0)
    )
    training_logs["player_neurd_coef_effective"] = config.player_neurd_coef * neurd_scale
    training_logs["player_reg_ema_rate_effective"] = reg_ema_rate

    # Fraction of steps where the IMPACT clipped-target correction is
    # saturated at its cap — a second staleness signal alongside the actor
    # KL and ESS diagnostics.
    training_logs["player_impact_clip_frac"] = (
        (actor_target_clipped_ratio >= 2.0).astype(jnp.float32).mean(where=policy_mask)
    )

    # Fresh-vs-replayed value error: the memorisation gap. A network with
    # healthy plasticity fits fresh and replayed trajectories about equally;
    # replayed error falling while fresh error rises means the buffer is
    # being memorised — the plasticity signature that should gate any
    # shrink-and-perturb, unlike league stagnation which has many causes.
    # Per-group means are NaN in batches with no fresh (or no replayed)
    # member; wandb line plots skip them.
    if not isinstance(batch.reuse_count, tuple):
        target_value = jnp.exp(player_target_pred.value_head.log_probs) @ cat_vf_support
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
    player_targets = promote_map(player_targets, float_dtype)

    action_mask = player_transitions.env_output.action_mask
    flat_action_mask = action_mask.reshape(*action_mask.shape[:-2], -1)

    # Two-rung Q critic (docs/q-critic-plan.md): Retrace targets from the
    # fast EMA target's PRIVILEGED all-action Q readout (Q_all — value_all
    # conditioning) and reference policy; the Q_private rung trains by CE
    # against the same labels. Zero policy influence — the heads train,
    # the diagnostics log, nothing reaches the actor loss.
    q_target_probs, q_retrace_g, q_all_target, q_v_exp, q_taken_target = (
        compute_q_targets(
            batch,
            q_logits=player_target_pred.q_logits,
            target_log_policy=player_target_pred.action_head.log_policy,
            isr=target_actor_ratio,
            config=config,
            reg_log_policy=reg_log_policy,
        )
    )
    # Second Retrace pass at the POLICY's lambda (player_pi_lambda): the
    # taken cell of the NeuRD advantage reads this return instead of the
    # critic's Q(s, a_t), while untaken cells keep Q_all — rnad.py's
    # taken-cell-return / critic split without its 1/mu. Same regularised
    # bootstraps, so it is consistent with q_all_target on every other
    # cell. Labels for the Q head stay at player_q_lambda.
    _, q_retrace_g_pi, _, _, _ = compute_q_targets(
        batch,
        q_logits=player_target_pred.q_logits,
        target_log_policy=player_target_pred.action_head.log_policy,
        isr=target_actor_ratio,
        config=config,
        reg_log_policy=reg_log_policy,
        trace_lambda=config.player_pi_lambda,
    )
    # KL(pi_target || pi_reg) per state — the expected per-step penalty
    # the Q critic's bootstrap carries. Drifts up as the policy moves
    # away from the lagged reference and back down as the EMA catches up;
    # a level that keeps climbing is a policy running away from its own
    # past faster than the reference follows.
    training_logs["player_rnad_kl_reg"] = average(
        reference_kl(
            player_target_pred.action_head.log_policy, reg_log_policy, flat_action_mask
        ),
        player_targets.policy_mask,
    )
    # Q(s, a) exists wherever an action was actually taken — including
    # forced single-option steps (policy_mask excludes those; the Q
    # regression must not) — but not on terminal rows.
    q_mask = value_mask & jnp.logical_not(player_transitions.env_output.done)
    # Both readouts estimate the same state value from the same target
    # params AND the same (privileged) information set, so their
    # absolute gap is pure calibration debt of the Q head to the V
    # head (q-critic-plan.md stage-1 acceptance metric) — no
    # information-deficit component since the value_all conditioning.
    training_logs["player_q_ev_gap"] = average(
        jnp.abs(
            q_v_exp - player_target_pred.value_head.expectation.astype(jnp.float32)
        ),
        q_mask,
    )
    # The direct "what does the critic think switching is worth"
    # readout, graded on the POLICY'S information set (Q_private — the
    # question is what a deployable critic believes, so the privileged
    # rung would answer the wrong one): best legal switch's E[Q] minus
    # best legal move's, over states offering both. The number the
    # switch-collapse investigation (Aug 2026) had no way to measure.
    q_private_all_target = jax.nn.softmax(
        player_target_pred.private_q_logits.astype(jnp.float32), axis=-1
    ) @ cat_vf_support.astype(jnp.float32)
    switch_cells = FLAT_MODALITY_MASK == ModalityEnum.MODALITY_ENUM__SWITCH
    move_cells = FLAT_MODALITY_MASK == ModalityEnum.MODALITY_ENUM__MOVE
    valid_switch = flat_action_mask & switch_cells
    valid_move = flat_action_mask & move_cells
    best_switch = jnp.max(
        jnp.where(valid_switch, q_private_all_target, -jnp.inf), axis=-1
    )
    best_move = jnp.max(
        jnp.where(valid_move, q_private_all_target, -jnp.inf), axis=-1
    )
    has_both = valid_switch.any(axis=-1) & valid_move.any(axis=-1)
    training_logs["player_q_switch_move_gap"] = average(
        jnp.where(has_both, best_switch - best_move, 0.0), q_mask & has_both
    )

    # Discriminators for a negative gap: starved switch cells vs a
    # genuine judgement. The CE only trains the taken action's cell,
    # so a collapsing switch_ratio starves voluntary-switch cells of
    # gradient while forced replacements (post-faint, no legal move)
    # keep flowing regardless of policy. Coverage says how bad the
    # starvation is; the conditional Retrace means are the
    # head-independent answer to "do voluntary switches actually lead
    # to worse outcomes than moves from the same kind of state?".
    taken_switch = jnp.take(switch_cells, player_actor_action_head.action_index)
    has_move = valid_move.any(axis=-1)
    q_voluntary_switch_mask = q_mask & taken_switch & has_move
    q_forced_switch_mask = q_mask & taken_switch & jnp.logical_not(has_move)
    q_move_mask = q_mask & jnp.logical_not(taken_switch)
    training_logs["player_q_switch_target_frac"] = average(
        taken_switch.astype(jnp.float32), q_mask
    )
    training_logs["player_q_voluntary_switch_target_frac"] = average(
        (taken_switch & has_move).astype(jnp.float32), q_mask
    )
    training_logs["player_q_target_voluntary_switch"] = average(
        q_retrace_g, q_voluntary_switch_mask
    )

    # Off-policy attenuation audit, split by the TAKEN modality. isr =
    # pi_target/mu_actor is what v-trace and Retrace multiply their TD
    # errors by (targets.py: rho_t = c_t = (1-alpha)*isr +
    # alpha*min(1, isr)). As pi(switch) decays, isr on switch-taken rows falls
    # below 1 and those rows contribute proportionally less.
    #
    # NOTE this is the CORRECT importance correction, not a bug: the
    # readout is effective sample size on switch rows, not bias. A
    # widening gap between the two means the learner is hearing the
    # switch evidence ever more faintly as the collapse deepens, which
    # is a self-reinforcing loop even though every individual update is
    # properly weighted. below1_frac is the cleaner signal than the
    # mean (isr is heavy-tailed on the upside).
    isr_f32 = target_actor_ratio.astype(jnp.float32)
    training_logs["player_isr_switch_voluntary"] = average(
        isr_f32, q_voluntary_switch_mask
    )
    training_logs["player_isr_switch_forced"] = average(
        isr_f32, q_forced_switch_mask
    )
    training_logs["player_isr_move"] = average(isr_f32, q_move_mask)
    training_logs["player_isr_below1_switch_voluntary"] = average(
        (isr_f32 < 1.0).astype(jnp.float32), q_voluntary_switch_mask
    )
    training_logs["player_isr_below1_move"] = average(
        (isr_f32 < 1.0).astype(jnp.float32), q_move_mask
    )
    training_logs["player_q_target_move"] = average(
        q_retrace_g, q_move_mask & has_both
    )

    # Pivotal-state decision panel (2026-08-19). A negative MEAN gap is
    # the expected sign under correct play — switching spends a turn, so
    # most both-modality states correctly favour the move, and the
    # state-averaged statistics above are nearly blind to the collapse
    # failure mode. The signal lives in the tail: the states where the
    # critic actually prefers the switch. Conditioning on the STATE
    # (critic flag) rather than the taken action also dodges the
    # chosen-switch selection bias that muddied the Aug-15 crossover
    # reading (a 2% switch mass keeps only the policy's most confident
    # pivots, so "chosen switches outperform moves" proves little).
    # Collapse signature: pivotal_frac bleeding to ~0 (critic stops
    # flagging any state as switch-worthy) or pi_switch_mass /
    # taken_switch_frac cratering on pivotal states while pivotal_frac
    # holds (policy ignoring the critic's flags).
    pivotal_mask = q_mask & has_both & (best_switch > best_move)
    training_logs["player_q_pivotal_frac"] = average(
        (best_switch > best_move).astype(jnp.float32), q_mask & has_both
    )
    if not isinstance(batch.game_outcome, tuple):
        # Step-1 panels (docs/critic-weakness-analysis.md): the one-step
        # label is the same bootstrap compute_q_targets would use at
        # lambda 0 but on the TARGET V HEAD — r_t + gamma * V_boot(s_{t+1}),
        # V_boot = r on the done row — i.e. the Step-3 candidate label,
        # measured here before anything trains on it.
        _dones = player_transitions.env_output.done.astype(bool)
        _r = (player_transitions.env_output.win_reward @ cat_vf_support).astype(jnp.float32)
        _v_tgt = player_target_pred.value_head.expectation.astype(jnp.float32)
        _v_boot = jnp.where(_dones, _r, _v_tgt)
        _v_next = jnp.concatenate([_v_boot[1:], _v_boot[-1:]], axis=0)
        training_logs.update(
            critic_outcome_telemetry(
                game_outcome=batch.game_outcome,
                game_length=batch.game_length,
                game_step_offset=batch.game_step_offset,
                v_target=_v_tgt,
                onestep_label=_r + config.player_gamma * _v_next,
                retrace_g=q_retrace_g,
                q_taken=q_taken_target,
                q_all=q_all_target,
                flat_action_mask=flat_action_mask,
                action_index=player_actor_action_head.action_index,
                q_mask=q_mask,
                value_mask=value_mask,
            )
        )
    pi_target = (
        jnp.exp(player_target_pred.action_head.log_policy.astype(jnp.float32))
        * flat_action_mask
    )
    pi_target = pi_target / jnp.maximum(pi_target.sum(axis=-1, keepdims=True), 1e-8)
    training_logs["player_q_pivotal_pi_switch_mass"] = average(
        (pi_target * valid_switch).sum(axis=-1), pivotal_mask
    )
    training_logs["player_q_pivotal_taken_switch_frac"] = average(
        taken_switch.astype(jnp.float32), pivotal_mask
    )
    # Within-class return split: same critic-flagged state class,
    # different action — the closest available reading of "are switches
    # better where they matter". Empty slices log 0 (average clips the
    # denominator), so read alongside pivotal_frac/taken_switch_frac.
    training_logs["player_q_pivotal_ret_switch"] = average(
        q_retrace_g, pivotal_mask & taken_switch
    )
    training_logs["player_q_pivotal_ret_stay"] = average(
        q_retrace_g, pivotal_mask & jnp.logical_not(taken_switch)
    )

    # Loss-free critic-quality diagnostics, on permanently. Since
    # 2026-08-21 the policy's ONLY link to returns is NeuRD through
    # Q_all, so these stopped being nice-to-have: action-value spread
    # (is there anything to prefer?) and calibration of the taken-cell
    # readout against its own realised Retrace targets.
    #
    # pi_target computed above (pivotal-state panel). The MEAN
    # undersells the spread by construction when action-value spread
    # concentrates in few high-leverage states (which is how this game
    # works) — the p90 is the honest readout.
    qvar_state = (
        pi_target * jnp.square(q_all_target - q_v_exp[..., None])
    ).sum(axis=-1)
    training_logs["player_q_action_var"] = average(qvar_state, q_mask)
    training_logs["player_q_action_var_p90"] = jnp.nanquantile(
        jnp.where(q_mask, qvar_state, jnp.nan), 0.9
    )
    # π-free counterpart: uniform-over-legal variance of the same
    # Q̄_all means. The π-weighted qvar above is squashed by a
    # collapsed policy regardless of what the critic believes (94%
    # move mass hides any spread on the move↔switch axis), so it
    # can't distinguish "critic is action-flat" from "critic is
    # confidently anti-switch" — opposite remedies (head capacity /
    # supervision vs nothing). Read the pair together: uniform ≫
    # π-weighted means the spread lives on actions the policy has
    # abandoned; both ≈ 0 means the critic genuinely can't tell
    # actions apart.
    n_legal = jnp.maximum(flat_action_mask.sum(axis=-1), 1)
    q_mean_uniform = (
        jnp.where(flat_action_mask, q_all_target, 0.0).sum(axis=-1) / n_legal
    )
    qvar_uniform = (
        jnp.where(
            flat_action_mask,
            jnp.square(q_all_target - q_mean_uniform[..., None]),
            0.0,
        ).sum(axis=-1)
        / n_legal
    )
    training_logs["player_q_action_var_uniform"] = average(
        qvar_uniform, q_mask
    )
    training_logs["player_q_action_var_uniform_p90"] = jnp.nanquantile(
        jnp.where(q_mask, qvar_uniform, jnp.nan), 0.9
    )
    if not isinstance(batch.reuse_count, tuple):
        fresh_cols = batch.reuse_count[0] == 0
        replay_cols = ~fresh_cols

        def q_calibration_r2(cols):
            # NaN (not 0.0) on an empty slice: under replay ratio 8 a
            # batch rarely holds a fresh chunk, and the 0.0 read as a
            # flat-zero panel for a whole run (2026-08-23).
            m = q_mask & cols[None, :]
            return jnp.where(
                m.sum() >= 2,
                calculate_r2(
                    value_prediction=q_taken_target,
                    value_target=q_retrace_g,
                    mask=m,
                ),
                jnp.nan,
            )

        training_logs["player_q_calibration_r2_fresh"] = q_calibration_r2(
            fresh_cols
        )
        training_logs["player_q_calibration_r2_replay"] = q_calibration_r2(
            replay_cols
        )
        vm_fresh = value_mask & fresh_cols[None, :]
        training_logs["player_value_r2_fresh"] = jnp.where(
            vm_fresh.any(),
            calculate_r2(
                value_prediction=player_target_pred.value_head.expectation.astype(
                    jnp.float32
                ),
                value_target=(
                    player_targets.win_returns @ cat_vf_support
                ).astype(jnp.float32),
                mask=vm_fresh,
            ),
            0.0,
        )

    def player_loss_fn(params: Params):

        learner_player_pred = player_state.apply_fn(
            params,
            player_actor_input,
            player_transitions.agent_output.actor_output,
            HeadParams(),
        )

        learner_value_head = learner_player_pred.value_head
        learner_action_head = learner_player_pred.action_head
        learner_log_prob = learner_action_head.log_prob

        learner_actor_log_ratio = learner_log_prob - player_actor_log_prob
        learner_actor_ratio = jnp.exp(learner_actor_log_ratio)

        learner_target_log_ratio = learner_log_prob - player_target_log_prob
        learner_target_ratio = jnp.exp(learner_target_log_ratio)

        # Softmax cross-entropy loss for value head
        loss_v_win = average(
            optax.softmax_cross_entropy(
                logits=learner_value_head.logits,
                labels=player_targets.win_returns,
            ),
            value_mask,
        )

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
        loss_target_forward_kl = forward_kl_loss(
            policy_ratio=learner_target_ratio,
            log_policy_ratio=learner_target_log_ratio,
            valid=policy_mask,
        )
        loss_target_backward_kl = backward_kl_loss(
            policy_ratio=learner_target_ratio,
            log_policy_ratio=learner_target_log_ratio,
            valid=policy_mask,
        )

        normalized_modality_entropy = average(
            learner_action_head.normalized_modality_entropy, policy_mask
        )

        # (player_commit_cov removed 2026-08-14 at the user's request: as
        # a correlation with the advantages the policy itself generated,
        # it largely measured what it was correlated with. With the
        # adaptivity controller also gone, modality collapse has NO
        # automated backstop — watch player_normalized_modality_entropy
        # on the dashboard; 1330 died at 0.08 on that axis.)

        # Counterfactual value ladder: private (deployable information set —
        # no opponent sheet) and public (history-context-only) heads, CE
        # against the SAME win targets as the privileged main head. Each
        # rung learns the best value estimate its information set
        # supports, so the expectation gaps between rungs are per-state
        # value-of-information readouts: |all − private| prices the
        # opponent's hidden team, |private − public| prices the agent's own
        # private information over the public game record.
        private_logits = learner_player_pred.private_value_logits.astype(jnp.float32)
        public_logits = learner_player_pred.public_value_logits.astype(jnp.float32)
        loss_v_private = average(
            optax.softmax_cross_entropy(
                logits=private_logits, labels=player_targets.win_returns
            ),
            value_mask,
        )
        loss_v_public = average(
            optax.softmax_cross_entropy(
                logits=public_logits, labels=player_targets.win_returns
            ),
            value_mask,
        )
        f32_support = cat_vf_support.astype(jnp.float32)
        private_expectation = jax.nn.softmax(private_logits, axis=-1) @ f32_support
        public_expectation = jax.nn.softmax(public_logits, axis=-1) @ f32_support
        all_expectation = learner_value_head.expectation.astype(jnp.float32)
        value_ladder_logs = dict(
            player_loss_v_private=loss_v_private,
            player_loss_v_public=loss_v_public,
            player_value_private_r2=calculate_r2(
                value_prediction=private_expectation,
                value_target=player_targets.win_returns @ cat_vf_support,
                mask=value_mask,
            ),
            player_value_public_r2=calculate_r2(
                value_prediction=public_expectation,
                value_target=player_targets.win_returns @ cat_vf_support,
                mask=value_mask,
            ),
            # Signed means (bias of the richer estimate vs the poorer) and
            # absolute means (the actual information value magnitude).
            player_value_info_gap_opp=average(
                all_expectation - private_expectation, value_mask
            ),
            player_value_info_gap_opp_abs=average(
                jnp.abs(all_expectation - private_expectation), value_mask
            ),
            player_value_info_gap_private=average(
                private_expectation - public_expectation, value_mask
            ),
            player_value_info_gap_private_abs=average(
                jnp.abs(private_expectation - public_expectation), value_mask
            ),
        )

        # Two-rung Q CE (docs/q-critic-plan.md): each rung's taken-action
        # categorical logits against the SAME two-hot Retrace target
        # (labels come from the privileged Q_all recursion; Q_private is
        # its deployable-information sibling, all head params shared).
        def q_taken(q_logits):
            return jnp.take_along_axis(
                q_logits.astype(jnp.float32),
                player_actor_action_head.action_index[..., None, None],
                axis=-2,
            ).squeeze(-2)

        learner_q_logits_taken = q_taken(learner_player_pred.q_logits)
        learner_q_private_logits_taken = q_taken(
            learner_player_pred.private_q_logits
        )
        q_ce_rows = optax.softmax_cross_entropy(
            logits=learner_q_logits_taken, labels=q_target_probs
        )
        loss_q = average(q_ce_rows, q_mask)
        # Optimisation-level support (Step 1): the share of the Q loss each
        # modality actually contributes — the acceptance measure for any
        # row weighting, where sampled-chunk counts are only a replay
        # diagnostic.
        _q_ce_total = jnp.maximum(jnp.sum(q_ce_rows, where=q_mask), 1e-8)
        q_loss_share = {
            f"player_q_loss_share_{name}": jnp.sum(q_ce_rows, where=m) / _q_ce_total
            for name, m in (
                ("move", q_move_mask),
                ("forced", q_forced_switch_mask),
                ("voluntary", q_voluntary_switch_mask),
            )
        }
        loss_q_private = average(
            optax.softmax_cross_entropy(
                logits=learner_q_private_logits_taken, labels=q_target_probs
            ),
            q_mask,
        )

        # Real-choice rows on the stay/switch axis: both a switch and a
        # non-switch are legal. This is the slice the collapse forms in,
        # and every NeuRD decomposition readout below is scoped to it.
        switch_actions = jnp.asarray(
            FLAT_MODALITY_MASK == ModalityEnum.MODALITY_ENUM__SWITCH
        )
        has_switch = (flat_action_mask & switch_actions).any(axis=-1)
        has_other = (
            flat_action_mask & jnp.logical_not(switch_actions)
        ).any(axis=-1)
        switch_choice_mask = policy_mask & has_switch & has_other

        # COMA-style all-action counterfactual policy loss (replaced
        # the stage-2 forward KL, 2026-08-19 — see the config comment
        # and docs/entropy-gradient-pressure.md). Per legal cell,
        # adv(a) = E[Q̄_all(a)] − Σ_a' π(a')·E[Q̄_all(a')]: the COMA
        # counterfactual baseline under the CURRENT policy. The loss
        # −Σ_a π(a)·sg(adv(a)) has per-cell gradient π(a)·adv(a) —
        # the exact all-action policy gradient: zero sampling
        # variance, counterfactual pressure on every real-choice row
        # including untaken actions (the sampled boost advantage
        # structurally carries none for those). Q̄_all is the
        # PRIVILEGED rung (COMA's centralised critic); it enters as
        # stop-gradient scalars only, exactly like the privileged
        # v_head advantages, so the policy stays bitwise invariant
        # to opp_private_team. Weighted by the neurd_coef runtime
        # scalar.
        learner_log_policy = learner_action_head.log_policy
        pi_learner = (
            jnp.exp(learner_log_policy.astype(jnp.float32))
            * flat_action_mask
        )
        pi_learner = pi_learner / jnp.maximum(
            pi_learner.sum(axis=-1, keepdims=True), 1e-8
        )
        # R-NaD reward transform (2026-08-22), applied per legal cell:
        # q'(a) = Q_all(a) - eta*(log pi(a) - log pi_reg(a)). The critic
        # carries the transformed game from the next step on (its
        # bootstrap is the regularised v_exp), so adding the own-step
        # penalty here makes adv the full regularised advantage. The
        # -eta*log pi(a) term is what refills a starved cell: it grows
        # without bound as pi(a) -> 0, with no pi prefactor, and it
        # vanishes only when pi == pi_reg — a moving reference, so the
        # fixed point is the regularised Nash, not a uniform prior.
        taken_one_hot = jax.nn.one_hot(
            player_actor_action_head.action_index, q_all_target.shape[-1], dtype=bool
        )
        q_for_policy = jnp.where(
            taken_one_hot & q_mask[..., None], q_retrace_g_pi[..., None], q_all_target
        )
        q_reg = rnad_transformed_q(
            q_for_policy,
            learner_log_policy,
            reg_log_policy,
            flat_action_mask,
            config.player_rnad_eta,
        )
        v_cf = jax.lax.stop_gradient((pi_learner * q_reg).sum(axis=-1))
        neurd_adv = jax.lax.stop_gradient(
            jnp.where(flat_action_mask, q_reg - v_cf[..., None], 0.0)
        )
        rnad_penalty = jax.lax.stop_gradient(
            jnp.where(flat_action_mask, q_for_policy - q_reg, 0.0)
        )
        # NeuRD prefactor (2026-08-21): the advantage lands on the
        # LOGITS with no pi factor, CENTRED over legal cells so the
        # softmax-invariant mean direction carries no push on its
        # own. Logit-gap clip (NeuRD eq. 10): a cell already
        # beyond +-beta of the row's legal-mean logit gets no
        # further push in the outward direction -- the sum of
        # advantages is not zero-mean in general, so unclipped
        # logits diverge; harmless at the policy level since beta
        # still permits probabilities arbitrarily close to 0/1,
        # NeuRD simply stops contributing outside the band.
        legal_count = jnp.maximum(flat_action_mask.sum(axis=-1), 1)
        adv_centred = neurd_adv - (
            neurd_adv.sum(axis=-1) / legal_count
        )[..., None]
        raw_logits = jnp.where(
            flat_action_mask,
            learner_action_head.logits.astype(jnp.float32),
            0.0,
        )
        logit_gap = jax.lax.stop_gradient(
            raw_logits
            - (raw_logits.sum(axis=-1) / legal_count)[..., None]
        )
        beta = config.player_neurd_logit_clip
        neurd_open = flat_action_mask & jnp.logical_not(
            ((logit_gap > beta) & (adv_centred > 0))
            | ((logit_gap < -beta) & (adv_centred < 0))
        )
        neurd_weight = jax.lax.stop_gradient(
            jnp.where(neurd_open, adv_centred, 0.0)
        )
        # Against the RAW logits: d/dy_b = -w(b) exactly, open
        # or clipped, with no softmax cross-term (the log_policy
        # form only matches while the weights are zero-sum,
        # which the clip breaks).
        loss_neurd = -average(
            (neurd_weight * raw_logits).sum(axis=-1), policy_mask
        )
        neurd_grad_prefactor = neurd_open.astype(jnp.float32)
        # Gradient decomposition for the pi-prefactor question
        # (docs/rare-action-rl-literature.md). The COMA loss
        # -sum_a pi(a).sg(adv(a)) has exact per-logit gradient
        # -pi(b).adv(b) -- the sum_a pi.adv correction term
        # vanishes because the COMA baseline makes it identically
        # zero -- which is NeuRD eq. (6): the counterfactual
        # regret SCALED BY THE ACTION'S OWN PROBABILITY. A starved
        # switch cell therefore gets a restoring force
        # proportional to how starved it already is, so COMA
        # cannot be the restorer on its own.
        #
        # These three pairs decompose the per-cell gradient
        # magnitude pi.|adv| into its two factors, over legal
        # cells of real-choice rows (both a switch and a non-
        # switch legal), so that
        #     grad_ratio ~ prob_ratio x absadv_ratio.
        # prob_ratio << 1 with absadv_ratio ~ 1: the whole
        # suppression is the pi prefactor, and dropping it (NeuRD
        # -- advantage on the LOGITS, no pi) is the fix.
        # absadv_ratio ~ 0: the critic carries no switch belief to
        # amplify, and NeuRD would amplify noise instead. NOTE
        # loss_q supervises only the TAKEN cell, so untaken switch
        # cells are extrapolation from the zero-init head rather
        # than belief -- read absadv_ratio against
        # player_q_switch_target_frac (the supervision coverage)
        # before concluding the critic "means it".
        neurd_row = switch_choice_mask[..., None]
        neurd_switch_cells = flat_action_mask & switch_actions & neurd_row
        neurd_move_cells = (
            flat_action_mask & jnp.logical_not(switch_actions) & neurd_row
        )
        neurd_abs_adv = jnp.abs(neurd_adv)
        # Per-cell |d loss / d logit|: pi.|adv| under COMA,
        # 1{clip open}.|adv| under NeuRD.
        neurd_grad_mag = neurd_grad_prefactor * neurd_abs_adv
        neurd_grad_switch = average(neurd_grad_mag, neurd_switch_cells)
        neurd_grad_move = average(neurd_grad_mag, neurd_move_cells)
        neurd_prob_switch = average(pi_learner, neurd_switch_cells)
        neurd_prob_move = average(pi_learner, neurd_move_cells)
        neurd_absadv_switch = average(neurd_abs_adv, neurd_switch_cells)
        neurd_absadv_move = average(neurd_abs_adv, neurd_move_cells)

        def neurd_ratio(numerator, denominator):
            return numerator / jnp.maximum(denominator, 1e-8)

        # R-NaD penalty -eta*(log pi - log pi_reg) by modality over
        # legal cells of real-choice rows (the sign convention: POSITIVE
        # = the cell is pushed UP relative to its Q, i.e. pi sits below
        # the reference there). penalty_switch climbing while
        # switch_ratio falls is the transform doing its job; both flat
        # at zero is a reference that has caught up (KL 0).
        rnad_penalty_switch = average(-rnad_penalty, neurd_switch_cells)
        rnad_penalty_move = average(-rnad_penalty, neurd_move_cells)

        neurd_logs = dict(
            player_loss_neurd=loss_neurd,
            player_rnad_penalty_switch=rnad_penalty_switch,
            player_rnad_penalty_move=rnad_penalty_move,
            # Per-cell |d loss_neurd / d logit| on legal switch
            # cells vs legal non-switch cells of the same
            # real-choice rows, and the two factors it is the
            # product of. The ratios are the readout: if
            # grad_ratio tracks prob_ratio while absadv_ratio
            # stays near 1, the restoring force is being throttled
            # by the pi prefactor alone.
            player_neurd_grad_switch=neurd_grad_switch,
            player_neurd_grad_move=neurd_grad_move,
            player_neurd_grad_ratio=neurd_ratio(
                neurd_grad_switch, neurd_grad_move
            ),
            player_neurd_prob_switch=neurd_prob_switch,
            player_neurd_prob_move=neurd_prob_move,
            player_neurd_prob_ratio=neurd_ratio(
                neurd_prob_switch, neurd_prob_move
            ),
            player_neurd_absadv_switch=neurd_absadv_switch,
            player_neurd_absadv_move=neurd_absadv_move,
            player_neurd_absadv_ratio=neurd_ratio(
                neurd_absadv_switch, neurd_absadv_move
            ),
            # Net signed gradient mass toward the switch modality
            # per real-choice row (prefactor x adv summed over the
            # row's switch cells): positive = the loss is currently
            # pushing switch logits UP (the critic prefers more
            # switching than the policy carries).
            player_neurd_switch_push=average(
                (neurd_grad_prefactor * neurd_adv * switch_actions).sum(
                    axis=-1
                ),
                switch_choice_mask,
            ),
            player_neurd_adv_std=neurd_adv.std(
                where=flat_action_mask & policy_mask[..., None]
            ),
            # NeuRD clip occupancy: fraction of legal switch / move
            # cells on real-choice rows whose outward push is
            # currently blocked by the logit-gap clip. Under the
            # COMA prefactor nothing is clipped (reads 0).
            player_neurd_clipped_switch=1.0
            - average(
                (neurd_grad_prefactor > 0).astype(jnp.float32),
                neurd_switch_cells,
            ),
            player_neurd_clipped_move=1.0
            - average(
                (neurd_grad_prefactor > 0).astype(jnp.float32),
                neurd_move_cells,
            ),
        )
        q_taken_pred = jax.nn.softmax(
            learner_q_logits_taken, axis=-1
        ) @ cat_vf_support.astype(jnp.float32)
        q_private_taken_pred = jax.nn.softmax(
            learner_q_private_logits_taken, axis=-1
        ) @ cat_vf_support.astype(jnp.float32)

        # Calibration by context, graded on Q_private (the contexts
        # exist to interpret the Q_private switch/move gap, so they
        # must grade the same rung). Forced switches stay data-rich
        # through a switch collapse; voluntary ones starve. Calibrated
        # forced + degraded voluntary = starvation artefact; both
        # calibrated with the gap still negative = the critic means it
        # (0.0 sentinel when a batch has no steps in a context).
        def q_context_r2(context_mask):
            return jnp.where(
                context_mask.any(),
                calculate_r2(
                    value_prediction=q_private_taken_pred,
                    value_target=q_retrace_g,
                    mask=context_mask,
                ),
                0.0,
            )

        q_logs = dict(
            player_loss_q=loss_q,
            player_loss_q_private=loss_q_private,
            player_q_r2=calculate_r2(
                value_prediction=q_taken_pred,
                value_target=q_retrace_g,
                mask=q_mask,
            ),
            player_q_private_r2=calculate_r2(
                value_prediction=q_private_taken_pred,
                value_target=q_retrace_g,
                mask=q_mask,
            ),
            player_q_r2_move=q_context_r2(q_move_mask),
            player_q_r2_switch_forced=q_context_r2(q_forced_switch_mask),
            player_q_r2_switch_voluntary=q_context_r2(q_voluntary_switch_mask),
            **q_loss_share,
        )
        # pg + (v + q) + kl + ent.
        loss = (
            # pg: all-action NeuRD is the ONLY term that moves the action
            # logits toward return — the two below only regularise.
            # neurd_scale: the Step-2 warm-up ramp (1 once warmed up / off).
            config.player_neurd_coef * neurd_scale * loss_neurd
            # v + q: the critic stack. One coefficient for both Q rungs —
            # same estimator family on the same labels, mirroring the
            # value-ladder coef.
            + (
                config.player_value_head_loss_coef * loss_v_win
                + config.player_value_ladder_coef * (loss_v_private + loss_v_public)
                + config.player_q_coef * (loss_q + loss_q_private)
            )
            # kl: trust region against the behaviour policy.
            + config.player_kl_loss_coef * loss_actor_backward_kl
        )

        return loss, dict(
            **q_logs,
            **neurd_logs,
            **value_ladder_logs,
            # Loss values
            player_loss_v_win=loss_v_win,
            player_loss_kl=loss_actor_backward_kl,
            # Per head entropies (diagnostics only — no longer regularized)
            player_action_entropy=action_head_entropy,
            player_action_normalized_entropy=action_head_normalized_entropy,
            player_normalized_modality_entropy=normalized_modality_entropy,
            # Ratios
            player_learner_actor_ratio=average(learner_actor_ratio, policy_mask),
            player_learner_target_ratio=average(learner_target_ratio, policy_mask),
            # KL values
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
            player_learner_target_forward_kl=loss_target_forward_kl,
            player_learner_target_backward_kl=loss_target_backward_kl,
            # Extra stats
            player_value_head_r2=calculate_r2(
                value_prediction=learner_value_head.expectation,
                value_target=player_targets.win_returns @ cat_vf_support,
                mask=value_mask,
            ),
            player_nll_sum=(
                batch.player_transitions.agent_output.actor_output.action_head.log_prob
                * policy_mask
            )
            .sum(axis=0)
            .mean(),
        )

    player_grad_fn = jax.value_and_grad(player_loss_fn, has_aux=True)
    (player_loss_val, player_logs), player_grads = player_grad_fn(player_state.params)

    prev_player_state = player_state
    player_state = player_state.apply_gradients(grads=player_grads)
    player_state = player_state.replace(
        step_count=player_state.step_count + 1,
        frame_count=player_state.frame_count + player_valid.sum(),
        # Frozen at the launch snapshot while the NeuRD warm-up ramps
        # (Step 2): the reference-KL panel then reads drift from launch.
        reg_params=optax.incremental_update(
            player_state.target_params,
            player_state.reg_params,
            reg_ema_rate,
        ),
        target_params=optax.incremental_update(
            player_state.params,
            player_state.target_params,
            config.player_ema_update_rate,
        ),
    )
    # A non-finite loss or gradient must never reach the params or the EMA
    # scalars: one poisoned update is permanent, and the next periodic save
    # then overwrites the last good checkpoint with it. Keep the previous
    # state wholesale and log the skip.
    player_update_finite = jnp.isfinite(player_loss_val) & jnp.isfinite(
        optax.global_norm(player_grads)
    )
    player_state = jax.tree.map(
        lambda new, old: jnp.where(player_update_finite, new, old),
        player_state,
        prev_player_state,
    )

    training_logs.update(player_logs)
    training_logs.update(
        dict(
            player_loss=player_loss_val,
            player_param_norm=optax.global_norm(player_state.params),
            player_gradient_norm=optax.global_norm(player_grads),
            # Mask sums
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
            player_shape_T=float(
                batch.player_transitions.env_output.done.shape[0]
            ),
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

    # --- Builder ---
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

        # Calculate importance sampling ratios for off-policy correction.
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
        builder_valid = builder_valid & player_transitions.env_output.done.any(
            axis=0
        )[None, :].astype(jnp.bool_)
        # Compute builder targets inside train_step (JAX/JIT compatible).
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
                objective=config.builder_policy_objective,
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
                # Ratios
                builder_learner_actor_ratio=average(learner_actor_ratio, builder_valid),
                builder_learner_target_ratio=average(
                    learner_target_ratio, builder_valid
                ),
                # Approx KL values
                builder_learner_actor_approx_kl=loss_forward_kl,
                builder_learner_condtional_entropy_head_mean=average(
                    learner_conditional_entropy_head.logits, builder_valid
                ),
                builder_learner_condtional_entropy_head_std=jnp.std(
                    learner_conditional_entropy_head.logits, where=builder_valid
                ),
                # Extra stats
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
