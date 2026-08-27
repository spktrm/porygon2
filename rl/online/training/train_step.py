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
    FLAT_SRC_GROUP_MASK,
    PackedSetFeature,
)
from rl.environment.interfaces import Batch, BuilderActorInput, PlayerActorInput
from rl.model.heads import HeadParams
from rl.model.utils import Params
from rl.online.artifact import Porygon2BuilderTrainState, Porygon2PlayerTrainState
from rl.online.config import Porygon2LearnerConfig
from rl.online.training.loss import (
    backward_kl_loss,
    clip_fraction,
    forward_kl_loss,
    mse_value_loss,
    policy_gradient_loss,
)
from rl.online.training.targets import (
    compute_builder_targets,
    compute_player_targets,
    compute_q_onestep_targets,
    reference_kl,
)
from rl.online.training.telemetry import (
    action_axis_masks,
    calculate_r2,
    collect_batch_telemetry_data,
    critic_outcome_telemetry,
    head_param_telemetry,
    modality_means,
    promote_map,
    q_fit_telemetry,
)
from rl.utils import average

logger = logging.getLogger(__name__)


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

    player_target_pred = player_state.apply_fn(
        player_state.target_params,
        player_actor_input,
        player_transitions.agent_output.actor_output,
        HeadParams(),
    )

    # NashPG reference policy: the FROZEN reg_params' full-support
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
    # mu/pi_target clipped at 2, telemetry only (player_impact_clip_frac):
    # the IMPACT surrogate it once recentred is gone; the panel still
    # reads how far behaviour has drifted from the fast target.
    actor_target_clipped_ratio = jnp.exp(-target_actor_log_ratio).clip(min=0.0, max=2.0)

    # Retrace baseline (targets.py): the TARGET critic's advantage at the
    # action actually taken. Zero on terminal rows — the reward lands on the
    # terminal OBSERVATION row, where no action was taken, so that row's
    # error must stay r - V. Same condition as q_mask below.
    adv_taken = jnp.take_along_axis(
        player_target_pred.advantage.astype(jnp.float32),
        player_actor_action_head.action_index[..., None],
        axis=-1,
    ).squeeze(-1)
    adv_taken = jnp.where(player_valid, adv_taken, 0.0)

    # IMPACT-style targets: the fast target network supplies the Retrace
    # reference policy and value/kl bootstraps.
    player_targets, channel_logs = compute_player_targets(
        batch,
        value_log_probs=player_target_pred.value_head.log_probs,
        isr=target_actor_ratio,
        adv_taken=adv_taken,
        config=config,
    )
    training_logs.update(channel_logs)
    policy_mask = player_targets.policy_mask
    value_mask = player_targets.value_mask
    # NashPG reference SNAP: reg_params <- target_params every
    # player_reg_snap_steps — their outer-loop rho reset, in place,
    # still three param sets, FROZEN between snaps. (The continuous EMA
    # this replaced never reset, so the KL gap compounded with policy
    # speed: 2wvnlsz3 hit ref_kl 2.07 nats by 98k.) Step 0 snaps
    # trivially (reg = target = init), and a resume at a snap multiple
    # snaps on its first step, repairing any accumulated gap at restart.
    reg_snap = player_state.step_count % config.player_reg_snap_steps == 0
    training_logs["player_reg_snapped"] = reg_snap.astype(jnp.float32)

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
        target_value = player_target_pred.value_head.expectation
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
    action_mask = player_transitions.env_output.action_mask
    flat_action_mask = action_mask.reshape(*action_mask.shape[:-2], -1)

    # NashPG advantage: the plain v-trace pass from targets.py, batch-
    # normalised over the surrogate's own rows with masked mean/std (the
    # reference's update_agent does exactly this per minibatch). f32
    # before promote_map bf16-casts the rest of the targets. The count
    # guard covers an all-masked batch; there is no running statistic
    # here to poison (LESSONS 2 applies to EMAs, not batch stats).
    pg_advantages = player_targets.pg_advantages.astype(jnp.float32)
    pg_adv_count = jnp.maximum(policy_mask.sum().astype(jnp.float32), 1.0)
    pg_adv_mean = jnp.where(policy_mask, pg_advantages, 0.0).sum() / pg_adv_count
    pg_adv_var = (
        jnp.where(policy_mask, jnp.square(pg_advantages - pg_adv_mean), 0.0).sum()
        / pg_adv_count
    )
    pg_adv_std = jnp.sqrt(pg_adv_var)
    pg_adv_norm = jnp.where(
        policy_mask, (pg_advantages - pg_adv_mean) / (pg_adv_std + 1e-8), 0.0
    )
    training_logs["player_pg_adv_mean"] = pg_adv_mean
    training_logs["player_pg_adv_std"] = pg_adv_std

    player_targets = promote_map(player_targets, float_dtype)

    # Residual Q critic (Step 3, docs/critic-weakness-analysis.md,
    # 2026-08-23): Q = sg(V) + A centred under pi, composed in the MODEL
    # (heads.compose_q) and read here — trained by Huber on the taken cell
    # against the TD(0) label r + gamma*V_win_target(s'), plain Q^pi, no
    # trace, no transformed bootstrap. ONE rung since 2026-08-25: the
    # critic and the policy share an information set, so there is no
    # privileged sibling to compare against. The policy no longer reads
    # this stack (2026-08-26): it is the matched-control observer and the
    # Retrace value baseline.
    v_target = player_target_pred.value_head.expectation.astype(jnp.float32)
    # The critics learn the PLAIN game: NashPG carries its reference KL
    # in the POLICY objective and uses no reward transform
    # (arXiv:2510.18183), so no penalty stream enters the labels or
    # bootstraps.
    q_target = player_target_pred.q.astype(jnp.float32)
    adv_target = player_target_pred.advantage.astype(jnp.float32)
    q_label = compute_q_onestep_targets(batch, v_target, config)
    q_taken_target = jnp.take_along_axis(
        q_target, player_actor_action_head.action_index[..., None], axis=-1
    ).squeeze(-1)
    # Q(s, a) exists wherever an action was actually taken — including
    # forced single-option steps (policy_mask excludes those; the Q
    # regression must not) — but not on terminal rows.
    q_mask = value_mask & jnp.logical_not(player_transitions.env_output.done)
    # Legal cells whose UNCLIPPED Q = V + A leaves the reward support —
    # the residual composition has no bin bound, the policy clips; a
    # rising fraction is the head inflating A beyond what V + outcome
    # allow (Step 3 panel, expect ~0).
    training_logs["player_q_saturation_frac"] = average(
        (jnp.abs(q_target) > 1.0).astype(jnp.float32).sum(axis=-1)
        / jnp.maximum(flat_action_mask.sum(axis=-1), 1),
        q_mask,
    )
    # The direct "what does the critic think switching is worth"
    # readout: best legal switch's E[Q] minus best legal move's, over
    # states offering both. The number the switch-collapse investigation
    # (Aug 2026) had no way to measure. Since 2026-08-25 the critic's
    # information set IS the policy's, so the old "grade this on the
    # deployable rung" caveat is structural rather than a choice.
    # One derivation of the switch/move predicates for the whole step —
    # the panels below, critic_outcome_telemetry and the policy-loss
    # telemetry all read THESE, so they cannot drift apart again
    # (telemetry.ActionAxisMasks).
    axis = action_axis_masks(flat_action_mask, player_actor_action_head.action_index)
    valid_switch = axis.valid_switch
    valid_move = axis.valid_move
    best_switch = jnp.max(jnp.where(valid_switch, q_target, -jnp.inf), axis=-1)
    best_move = jnp.max(jnp.where(valid_move, q_target, -jnp.inf), axis=-1)
    has_both = axis.has_both
    training_logs["player_q_switch_move_gap"] = average(
        jnp.where(has_both, best_switch - best_move, 0.0), q_mask & has_both
    )

    # Discriminators for a negative gap: starved switch cells vs a
    # genuine judgement. The Huber loss only trains the taken action's cell,
    # so a collapsing switch_ratio starves voluntary-switch cells of
    # gradient while forced replacements (post-faint, no legal move)
    # keep flowing regardless of policy. Coverage says how bad the
    # starvation is; the conditional one-step-label means are the
    # head-independent answer to "do voluntary switches actually lead
    # to worse outcomes than moves from the same kind of state?".
    taken_switch = axis.taken_switch
    has_move = axis.has_move
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
        q_label, q_voluntary_switch_mask
    )

    # Off-policy attenuation audit, split by the TAKEN modality. isr =
    # pi_target/mu_actor is what v-trace multiplies its TD errors by
    # (targets.py: rho_t = c_t = min(1, isr) — Retrace went 2026-08-23
    # and the alpha blend 2026-08-21). As pi(switch) decays, isr on
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
    isr_f32 = target_actor_ratio.astype(jnp.float32)
    training_logs["player_isr_switch_voluntary"] = average(
        isr_f32, q_voluntary_switch_mask
    )
    training_logs["player_isr_switch_forced"] = average(isr_f32, q_forced_switch_mask)
    training_logs["player_isr_move"] = average(isr_f32, q_move_mask)
    training_logs["player_isr_below1_switch_voluntary"] = average(
        (isr_f32 < 1.0).astype(jnp.float32), q_voluntary_switch_mask
    )
    training_logs["player_isr_below1_move"] = average(
        (isr_f32 < 1.0).astype(jnp.float32), q_move_mask
    )
    # How much action-conditional baseline the advantage head is actually
    # supplying to the Retrace errors above. At zero the value target is
    # bit-identical to the old v-trace one, so a flat ~0 here means the
    # mechanism is inert and should be reverted rather than kept.
    adv_taken_abs = jnp.abs(adv_taken)
    training_logs["player_retrace_baseline_abs"] = average(adv_taken_abs, q_mask)
    training_logs["player_retrace_baseline_switch"] = average(
        adv_taken_abs, q_voluntary_switch_mask
    )
    training_logs["player_retrace_baseline_move"] = average(adv_taken_abs, q_move_mask)
    training_logs["player_q_target_move"] = average(q_label, q_move_mask & has_both)

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
        # Step-1 panels (docs/critic-weakness-analysis.md). Since Step 3
        # the one-step label IS the Q label; the outcome/onestep split
        # of the label-variance panels is kept as the record of why.
        training_logs.update(
            critic_outcome_telemetry(
                game_outcome=batch.game_outcome,
                game_length=batch.game_length,
                game_step_offset=batch.game_step_offset,
                v_target=v_target,
                onestep_label=q_label,
                q_label=q_label,
                q_taken=q_taken_target,
                q_all=q_target,
                flat_action_mask=flat_action_mask,
                masks=axis,
                q_mask=q_mask,
                value_mask=value_mask,
            )
        )
    pi_target = (
        jnp.exp(player_target_pred.action_head.log_policy.astype(jnp.float32))
        * flat_action_mask
    )
    pi_target = pi_target / jnp.maximum(pi_target.sum(axis=-1, keepdims=True), 1e-8)
    # == v_target by the pi-centring in heads.compose_q (kept as the variance baseline).
    q_v_exp = (pi_target * q_target).sum(axis=-1)
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
        q_label, pivotal_mask & taken_switch
    )
    training_logs["player_q_pivotal_ret_stay"] = average(
        q_label, pivotal_mask & jnp.logical_not(taken_switch)
    )

    # Loss-free critic-quality diagnostics, on permanently: action-value
    # spread (is there anything to prefer?) and calibration of the
    # taken-cell readout against its own realised one-step labels.
    #
    # pi_target computed above (pivotal-state panel). The MEAN
    # undersells the spread by construction when action-value spread
    # concentrates in few high-leverage states (which is how this game
    # works) — the p90 is the honest readout.
    qvar_state = (pi_target * jnp.square(q_target - q_v_exp[..., None])).sum(axis=-1)
    training_logs["player_q_action_var"] = average(qvar_state, q_mask)
    training_logs["player_q_action_var_p90"] = jnp.nanquantile(
        jnp.where(q_mask, qvar_state, jnp.nan), 0.9
    )
    # π-free counterpart: uniform-over-legal variance of the same
    # per-cell Q means. The π-weighted qvar above is squashed by a
    # collapsed policy regardless of what the critic believes (94%
    # move mass hides any spread on the move↔switch axis), so it
    # can't distinguish "critic is action-flat" from "critic is
    # confidently anti-switch" — opposite remedies (head capacity /
    # supervision vs nothing). Read the pair together: uniform ≫
    # π-weighted means the spread lives on actions the policy has
    # abandoned; both ≈ 0 means the critic genuinely can't tell
    # actions apart.
    n_legal = jnp.maximum(flat_action_mask.sum(axis=-1), 1)
    q_mean_uniform = jnp.where(flat_action_mask, q_target, 0.0).sum(axis=-1) / n_legal
    qvar_uniform = (
        jnp.where(
            flat_action_mask,
            jnp.square(q_target - q_mean_uniform[..., None]),
            0.0,
        ).sum(axis=-1)
        / n_legal
    )
    training_logs["player_q_action_var_uniform"] = average(qvar_uniform, q_mask)
    training_logs["player_q_action_var_uniform_p90"] = jnp.nanquantile(
        jnp.where(q_mask, qvar_uniform, jnp.nan), 0.9
    )

    # Within- vs between-MODALITY split of the uniform spread (2026-08-24,
    # docs/critic-weakness-analysis.md). The head composes per-cell Q as
    # macro[modality] + gated micro, so the uniform variance is exactly
    # between (what the per-modality macro can carry: switch-vs-move) +
    # within (which move / which reserve — only the pointer micro grid
    # can carry it). 70mhptdc read uniform ≈ p(1-p)·gap² for the whole
    # run, i.e. the critic resolved one bit; this pair says so directly.
    # One rung since 2026-08-25.
    def modality_var_split(q_all):
        cell_mean = modality_means(q_all, flat_action_mask)
        within = jnp.where(flat_action_mask, jnp.square(q_all - cell_mean), 0.0)
        between = jnp.where(
            flat_action_mask, jnp.square(cell_mean - q_mean_uniform[..., None]), 0.0
        )
        return within.sum(axis=-1) / n_legal, between.sum(axis=-1) / n_legal

    q_within, q_between = modality_var_split(q_target)
    training_logs["player_q_action_var_within_modality"] = average(q_within, q_mask)
    training_logs["player_q_action_var_within_modality_p90"] = jnp.nanquantile(
        jnp.where(q_mask, q_within, jnp.nan), 0.9
    )
    training_logs["player_q_action_var_between_modality"] = average(q_between, q_mask)

    # Advantage-axis scale, read off A directly (2026-08-25). Every panel
    # above is a Q spread and therefore V-invariant already; these name the
    # quantity honestly and, crucially, SPLIT IT BY SLOT GROUP — the readout
    # that says whether each group's now-separate parameters are carrying
    # signal. Before the separation the three groups shared one projection
    # and the target group's only parameter was still bitwise zero at 84.9k.
    #
    # There is deliberately no "centring term" panel: A is pi-centred inside
    # the model (heads.compose_q), so E_pi[A] = 0 by construction and the
    # old "is the head leaking state level into A" question is answered
    # structurally rather than measured (tests/test_q_identity.py pins it).
    adv_sq = jnp.square(adv_target)
    training_logs["player_adv_rms"] = average(
        jnp.sqrt(jnp.where(flat_action_mask, adv_sq, 0.0).sum(axis=-1) / n_legal),
        q_mask,
    )
    for gid, gname in enumerate(("move", "switch", "target")):
        in_group = flat_action_mask & (jnp.asarray(FLAT_SRC_GROUP_MASK) == gid)
        cnt = in_group.sum(axis=-1)
        training_logs[f"player_adv_rms_{gname}"] = average(
            jnp.sqrt(
                jnp.where(in_group, adv_sq, 0.0).sum(axis=-1) / jnp.maximum(cnt, 1)
            ),
            q_mask & (cnt > 0),
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
                    value_target=q_label,
                    mask=m,
                ),
                jnp.nan,
            )

        training_logs["player_q_calibration_r2_fresh"] = q_calibration_r2(fresh_cols)
        training_logs["player_q_calibration_r2_replay"] = q_calibration_r2(replay_cols)
        vm_fresh = value_mask & fresh_cols[None, :]
        training_logs["player_value_r2_fresh"] = jnp.where(
            vm_fresh.any(),
            calculate_r2(
                value_prediction=player_target_pred.value_head.expectation.astype(
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
            HeadParams(),
        )

        learner_value_head = learner_player_pred.value_head
        learner_action_head = learner_player_pred.action_head
        learner_log_prob = learner_action_head.log_prob

        learner_actor_log_ratio = learner_log_prob - player_actor_log_prob
        learner_actor_ratio = jnp.exp(learner_actor_log_ratio)

        learner_target_log_ratio = learner_log_prob - player_target_log_prob
        learner_target_ratio = jnp.exp(learner_target_log_ratio)

        loss_v_win = average(
            optax.softmax_cross_entropy(
                logits=learner_value_head.logits.astype(jnp.float32),
                labels=player_targets.win_returns.astype(jnp.float32),
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

        # Residual Q loss: Huber on the taken cell against the one-step
        # label. Q = sg(V) + A is composed in the MODEL (heads.compose_q),
        # so this only reads it.
        #
        # The state route is closed — every state-level degree of freedom
        # sits in V and this loss cannot move it — but WHY changed on
        # 2026-08-25 and the new reason is the thing to protect. It used to
        # hold for free: the composition used the TARGET net's V, which is
        # not a differentiated leaf. It now holds because compose_q
        # stop-gradients the learner's OWN V explicitly. Delete that
        # stop_gradient and this loss silently starts fitting the label
        # through the state route instead of the action axis — which the
        # Step-6 probe in docs/critic-weakness-analysis.md showed it will
        # do, reaching the label-entropy floor while within-state action
        # variance FELL 5x. tests/test_q_identity.py pins it.
        q_taken_pred = jnp.take_along_axis(
            learner_player_pred.q.astype(jnp.float32),
            player_actor_action_head.action_index[..., None],
            axis=-1,
        ).squeeze(-1)
        q_err_rows = optax.huber_loss(q_taken_pred, q_label, delta=1.0)
        loss_q = average(q_err_rows, q_mask)
        q_fit_logs = q_fit_telemetry(
            q_err_rows=q_err_rows,
            q_taken_pred=q_taken_pred,
            q_label=q_label,
            q_mask=q_mask,
            context_masks={
                "move": q_move_mask,
                "forced": q_forced_switch_mask,
                "voluntary": q_voluntary_switch_mask,
            },
        )

        # Real-choice rows on the stay/switch axis: both a switch and a
        # a MOVE are legal. This is the slice the collapse forms in, and
        # every policy-modality readout below is scoped to it.
        #
        # STRICT since 2026-08-25. This previously required a switch and any
        # legal NON-switch, which also admitted WILDCARD / OTHER / TARGET
        # cells — so a row offering {switch, pass} counted as a stay/switch
        # decision here while the identically-described `has_both` in the Q
        # panels excluded it, and CLAUDE.md 3's rule reads the two families
        # against each other. A stay/switch decision only means something
        # when staying and attacking is actually on offer.
        switch_actions = axis.switch_cells
        switch_choice_mask = policy_mask & axis.has_both

        # NashPG policy update (2026-08-26, arXiv:2510.18183): a
        # PPO-clipped surrogate on the taken action's ratio pi/mu against
        # the batch-normalised v-trace advantage, plus a DIFFERENTIATED
        # reverse KL(pi || pi_reg) magnet toward the periodically snapped
        # reference and a differentiated entropy bonus. Every force here
        # runs through the composed log-softmax, whose Jacobian is
        # zero-sum per level — the softmax-invariant mean direction
        # receives no push — and the clip zeroes the surrogate's gradient
        # once the ratio leaves the band in the push direction, so no
        # force persists at a stiff equilibrium. All three terms carry a
        # pi prefactor: there is deliberately no prefactor-free refill
        # force any more (the NashPG bet is that the magnet cycle plus
        # entropy keep pi interior so starvation never starts —
        # switch_ratio through the 13k wire is the acceptance gate).
        learner_log_policy = learner_action_head.log_policy
        pi_learner = jnp.exp(learner_log_policy.astype(jnp.float32)) * flat_action_mask
        pi_learner = pi_learner / jnp.maximum(
            pi_learner.sum(axis=-1, keepdims=True), 1e-8
        )

        loss_pg = policy_gradient_loss(
            policy_ratios=learner_actor_ratio,
            advantages=pg_adv_norm,
            valid=policy_mask,
            threshold=config.player_ppo_clip,
            objective=config.player_pg_objective,
        )
        # Entropy bonus: maximise E[H(pi)] over real-choice rows.
        loss_entropy = -action_head_entropy
        # Magnet: full-distribution KL(pi || pi_reg) per row —
        # differentiated through the learner side (reg_log_policy comes
        # off the frozen reg_params, a constant).
        magnet_kl_rows = reference_kl(
            learner_log_policy, reg_log_policy, flat_action_mask
        )
        loss_mag = average(magnet_kl_rows, policy_mask)
        # The support anchor family (forward KL toward a temperature-raised
        # / advantage-tilted reference, phases 1-4) was REMOVED 2026-08-27:
        # every mass-restoring force either erased within-modality
        # discrimination or taught the mean switch's (losing) value — see
        # the CLAUDE.md removal ledger. The factorised objective below is
        # the replacement: the axes get their own trust regions and their
        # own entropy budgets, and mass is left to follow value.

        # Modality decomposition of the two factors any taken-action
        # update is throttled by: pi mass and the observer critic's |A|,
        # over legal switch vs non-switch cells of real-choice rows.
        # Loss-agnostic, kept across the 2026-08-26 policy-loss transition:
        # prob_ratio falling with absadv_ratio ~ 1 is still the
        # starvation signature to watch, now with nothing but the magnet
        # cycle and entropy to oppose it.
        pg_row = switch_choice_mask[..., None]
        pg_switch_cells = flat_action_mask & switch_actions & pg_row
        pg_move_cells = flat_action_mask & jnp.logical_not(switch_actions) & pg_row
        abs_adv = jnp.abs(adv_target)

        def modality_ratio(numerator, denominator):
            return numerator / jnp.maximum(denominator, 1e-8)

        policy_prob_switch = average(pi_learner, pg_switch_cells)
        policy_prob_move = average(pi_learner, pg_move_cells)
        policy_absadv_switch = average(abs_adv, pg_switch_cells)
        policy_absadv_move = average(abs_adv, pg_move_cells)

        pg_logs = dict(
            player_loss_pg=loss_pg,
            player_loss_entropy=loss_entropy,
            player_ppo_clip_frac=clip_fraction(
                policy_ratios=learner_actor_ratio,
                valid=policy_mask,
                clip_ppo=config.player_ppo_clip,
            ),
            player_policy_prob_switch=policy_prob_switch,
            player_policy_prob_move=policy_prob_move,
            player_policy_prob_ratio=modality_ratio(
                policy_prob_switch, policy_prob_move
            ),
            player_policy_absadv_switch=policy_absadv_switch,
            player_policy_absadv_move=policy_absadv_move,
            player_policy_absadv_ratio=modality_ratio(
                policy_absadv_switch, policy_absadv_move
            ),
            # KL(pi_learner || pi_reg) per state — the magnet loss's own
            # value (name kept across the transition). Sawtooth: drifts
            # up against the FROZEN reference, drops to ~0 at each snap;
            # a level climbing ACROSS snaps is a policy running away
            # faster than the snap period can repair.
            player_ref_kl=loss_mag,
        )

        # Calibration by context. The contexts exist to interpret the
        # switch/move gap, so they grade the same Q — there has been one
        # rung since 2026-08-25. Forced switches stay data-rich
        # through a switch collapse; voluntary ones starve. Calibrated
        # forced + degraded voluntary = starvation artefact; both
        # calibrated with the gap still negative = the critic means it
        # (0.0 sentinel when a batch has no steps in a context).
        def q_context_r2(context_mask):
            return jnp.where(
                context_mask.any(),
                calculate_r2(
                    value_prediction=q_taken_pred,
                    value_target=q_label,
                    mask=context_mask,
                ),
                0.0,
            )

        q_logs = dict(
            player_loss_q=loss_q,
            player_q_r2=calculate_r2(
                value_prediction=q_taken_pred,
                value_target=q_label,
                mask=q_mask,
            ),
            player_q_r2_move=q_context_r2(q_move_mask),
            player_q_r2_switch_forced=q_context_r2(q_forced_switch_mask),
            player_q_r2_switch_voluntary=q_context_r2(q_voluntary_switch_mask),
            **q_fit_logs,
        )
        # pg bracket + (v + q) + kl.
        loss = (
            # pg: the NashPG bracket — the PPO surrogate is the only term
            # that moves the action logits toward return; the entropy
            # bonus and the magnet sit inside the bracket so one
            # coefficient scales improvement and regularisation together.
            config.player_pg_coef
            * (
                loss_pg
                + config.player_ent_coef * loss_entropy
                + config.player_mag_coef * loss_mag
            )
            # v + q: the critic stack — one V on the deploy-time
            # information set, one all-action advantage over it (the
            # policy no longer reads it: matched-control observer and
            # Retrace baseline).
            + (
                config.player_value_head_loss_coef * loss_v_win
                + config.player_q_coef * loss_q
            )
            # kl: trust region against the behaviour policy — the
            # replay-staleness guard alongside the PPO clip.
            + config.player_kl_loss_coef * loss_actor_backward_kl
        )

        return loss, dict(
            **q_logs,
            **pg_logs,
            player_loss_v_win=loss_v_win,
            player_loss_kl=loss_actor_backward_kl,
            # Per head entropies (diagnostics only — no longer regularized)
            player_action_entropy=action_head_entropy,
            player_action_normalized_entropy=action_head_normalized_entropy,
            player_normalized_modality_entropy=normalized_modality_entropy,
            player_learner_actor_ratio=average(learner_actor_ratio, policy_mask),
            player_learner_target_ratio=average(learner_target_ratio, policy_mask),
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
        # Hard-snapped to the target net every player_reg_snap_steps —
        # NashPG's outer-loop reference reset (see reg_snap above).
        reg_params=jax.tree.map(
            lambda t, r: jnp.where(reg_snap, t, r),
            player_state.target_params,
            player_state.reg_params,
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
            # Q-head learning readouts: the three-scalar micro gate, the
            # drift-from-init of the zero-init out layers and the pointer
            # kernels, and per-subtree grad norms (pre-clip). A micro
            # kernel rms sitting at its lecun init (0.0625 at fan-in 256)
            # with a flat gate = the within-modality route never trained.
            **head_param_telemetry(prev_player_state.params, player_grads),
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
        builder_valid = builder_valid & player_transitions.env_output.done.any(axis=0)[
            None, :
        ].astype(jnp.bool_)
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
