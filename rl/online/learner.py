import collections
import dataclasses
import gc
import json
import logging
import os
import pickle
import queue
import random
import sys
import threading
import time
from _thread import LockType
from contextlib import nullcontext
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb.wandb_run
from tqdm import tqdm

import wandb
from rl import checkpoint

from rl.environment.data import (
    CAT_VF_SUPPORT,
    FLAT_MODALITY_MASK,
    STOI,
    PackedSetFeature,
)
from rl.environment.interfaces import (
    Batch,
    BuilderActorInput,
    PlayerActorInput,
    Trajectory,
)
from rl.environment.protos.enums_pb2 import SpeciesEnum
from rl.environment.protos.features_pb2 import (
    EntityRevealedNodeFeature,
    FieldFeature,
)
from rl.environment.protos.service_pb2 import ModalityEnum
from rl.environment.utils import (
    close_tqdm_bar,
    next_tqdm_position,
)
from rl.model.heads import HeadParams, calculate_hierarchical_prior
from rl.model.utils import Params, ParamsContainer
from rl.online.artifact import (
    Porygon2BuilderTrainState,
    Porygon2PlayerTrainState,
    write_checkpoint_components,
)
from rl.online.buffer import BuilderTrajectoryStore, PlayerTrajectoryStore
from rl.online.config import Porygon2LearnerConfig
from rl.online.controllers import PILogController
from rl.online.league import (
    LIVE_KEYS,
    MAIN_KEY,
    League,
    PlayerRef,
)
from rl.online.loss import (
    backward_kl_loss,
    forward_kl_loss,
    mse_value_loss,
    policy_gradient_loss,
)
from rl.online.targets import (
    compute_builder_targets,
    compute_player_targets,
    compute_q_targets,
)
from rl.online.utils import calculate_r2, collect_batch_telemetry_data, promote_map
from rl.utils import average

logger = logging.getLogger(__name__)

# Why a snapshot was added to the league. "dominant" is the healthy path
# (the agent beat its own history); "overdue" means only the frame budget
# expired, which is the plateau signature.
AddReason = Literal["initial", "dominant", "overdue"]

class OOMGuardTriggered(Exception):
    """Raised by Learner._check_oom_guard when available system RAM drops
    below config.oom_guard_min_available_fraction — a self-monitoring
    safety valve, not a leak fix in itself. A full checkpoint has already
    been written by the time this is raised. rl.online.main stops the
    whole process on this (same as a Ctrl-C interrupt) rather than
    continuing in the same process: freeing Python objects doesn't
    guarantee the OS actually reclaims that memory, so the only way to get
    back to a genuinely clean memory state is a fresh process — the user
    (or whatever supervises this box) needs to relaunch, which will resume
    from the checkpoint this exception carries."""

    def __init__(self, checkpoint_path: str):
        super().__init__(checkpoint_path)
        self.checkpoint_path = checkpoint_path


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

    # Exploration-ladder rows (config.explore_game_prob; previously the
    # stage-4 cross-population intake, removed 2026-08-15) train EVERY
    # player loss since 2026-08-17: the heads record the TEMPERED
    # distribution as the behaviour policy, so target_actor_ratio is the
    # exact policy-over-tempered-behaviour ISR that v-trace/IMPACT/Retrace
    # already truncate on — tempered rows are ordinary off-policy rows.
    # own_rows survives only where tempered play would distort a signal
    # that has no correction path: league add cadence (frame_count), the
    # plasticity memorisation gap, the builder losses, and the replay
    # controller's KL set-point.
    own_rows = None
    if not isinstance(batch.explore, tuple):
        own_rows = jnp.logical_not(batch.explore[0].astype(bool))  # (B,)

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
        if own_rows is not None:
            # Tempered-play value errors would contaminate the fresh/replay
            # memorisation gap — grade it on standard-temperature play only.
            fresh = fresh & own_rows
            replayed = replayed & own_rows
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

    # Magnet policy for the KL regularizer: the hierarchical prior — uniform
    # over valid modalities, uniform within each modality — which is the init
    # policy of the hierarchically composed action head, so the KL decomposes
    # into a modality-level KL plus the expected within-modality KLs. The
    # magnet is deliberately stationary — a fixed anchor gives the regularized
    # self-play dynamics a stable fixed point, whereas an EMA magnet chases
    # the policy and degenerates into a short-horizon trust region. The EMA
    # target's only remaining role is the v-trace/IMPACT reference.
    action_mask = player_transitions.env_output.action_mask
    flat_action_mask = action_mask.reshape(*action_mask.shape[:-2], -1)
    magnet_prior = calculate_hierarchical_prior(flat_action_mask).astype(float_dtype)
    magnet_log_policy = jnp.where(
        flat_action_mask, jnp.log(jnp.maximum(magnet_prior, 1e-9)), 0.0
    )

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
        )
    )
    # Q(s, a) exists wherever an action was actually taken — including
    # forced single-option steps (policy_mask excludes those; the Q
    # regression must not) — but not on terminal rows.
    q_mask = value_mask & jnp.logical_not(player_transitions.env_output.done)
    if own_rows is not None:
        training_logs["player_q_explore_frac"] = average(
            jnp.logical_not(own_rows)[None, :].astype(jnp.float32)
            * jnp.ones_like(q_mask, dtype=jnp.float32),
            q_mask,
        )

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
    # alpha*min(1, isr)). The explore ladder records its TEMPERED
    # log_prob, so mu carries MORE switch mass than pi on explore rows
    # — exactly the rows whose evidence would contradict a switch
    # collapse. As pi(switch) decays, isr on switch-taken rows falls
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
    if own_rows is not None:
        # Explore-ladder rows play at flattened temperature, so switches
        # taken there are far closer to randomised interventions than
        # exploit-row switches — the least selection-biased empirical
        # answer to "do voluntary switches lead to better outcomes"
        # available without new machinery.
        explore_cols = jnp.logical_not(own_rows)[None, :]
        training_logs["player_q_explore_ret_vol_switch"] = average(
            q_retrace_g, q_voluntary_switch_mask & explore_cols
        )
        training_logs["player_q_explore_ret_move"] = average(
            q_retrace_g, q_move_mask & has_both & explore_cols
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
        if own_rows is not None:
            # Same standard-temperature filter as the capacity
            # gap: tempered rows would contaminate calibration.
            fresh_cols = fresh_cols & own_rows
            replay_cols = replay_cols & own_rows

        def q_calibration_r2(cols):
            m = q_mask & cols[None, :]
            return jnp.where(
                m.any(),
                calculate_r2(
                    value_prediction=q_taken_target,
                    value_target=q_retrace_g,
                    mask=m,
                ),
                0.0,
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

        # Full-support KL(pi_learner || pi_magnet), exact per state — no
        # importance correction needed since no sampled action is involved.
        learner_log_policy = learner_action_head.log_policy
        magnet_kl = jnp.where(
            flat_action_mask,
            jnp.exp(learner_log_policy) * (learner_log_policy - magnet_log_policy),
            0.0,
        ).sum(axis=-1)
        loss_magnet_kl = average(magnet_kl, valid=policy_mask)

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
        loss_q = average(
            optax.softmax_cross_entropy(
                logits=learner_q_logits_taken, labels=q_target_probs
            ),
            q_mask,
        )
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
        pi_learner = (
            jnp.exp(learner_log_policy.astype(jnp.float32))
            * flat_action_mask
        )
        pi_learner = pi_learner / jnp.maximum(
            pi_learner.sum(axis=-1, keepdims=True), 1e-8
        )
        v_cf = jax.lax.stop_gradient(
            (pi_learner * q_all_target).sum(axis=-1)
        )
        neurd_adv = jax.lax.stop_gradient(
            jnp.where(
                flat_action_mask, q_all_target - v_cf[..., None], 0.0
            )
        )
        # NeuRD prefactor (2026-08-21): the advantage lands on the
        # LOGITS with no pi factor, CENTRED over legal cells so the
        # softmax-invariant mean direction carries no push on its
        # own. Logit-gap clip (NeuRD eq. 10): a cell already
        # beyond +-beta of the row's legal-mean logit gets no
        # further push in the outward direction -- the sum of
        # advantages is not zero-mean in general, so unclipped
        # logits diverge; harmless at the policy level since beta
        # still permits probabilities arbitrarily close to 0/1
        # through the magnet, NeuRD simply stops contributing
        # outside the band.
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

        neurd_logs = dict(
            player_loss_neurd=loss_neurd,
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
        )
        if own_rows is not None:
            # Does the Q head actually fit the explore-actor-generated
            # returns it is digesting? Persistent gap vs the full-mask
            # r2 = the tempered rows are too off-policy to learn from
            # (Retrace cutting every trace) rather than free
            # counterfactuals.
            q_logs["player_q_r2_explore"] = q_context_r2(
                q_mask & jnp.logical_not(own_rows)[None, :]
            )

        # pg + (v + q) + kl + ent.
        loss = (
            # pg: all-action NeuRD is the ONLY term that moves the action
            # logits toward return — the two below only regularise.
            config.player_neurd_coef * loss_neurd
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
            # ent: the magnet KL is per-state entropy regularisation
            # (uniform-over-legal hierarchical prior), and the only force
            # opposing pg.
            + config.player_magnet_kl_coef * loss_magnet_kl
        )

        return loss, dict(
            **q_logs,
            **neurd_logs,
            **value_ladder_logs,
            # Loss values
            player_loss_v_win=loss_v_win,
            player_loss_kl=loss_actor_backward_kl,
            player_loss_magnet_kl=loss_magnet_kl,
            # Per head entropies (diagnostics only — no longer regularized)
            player_action_entropy=action_head_entropy,
            player_action_normalized_entropy=action_head_normalized_entropy,
            player_normalized_modality_entropy=normalized_modality_entropy,
            # Ratios
            player_learner_actor_ratio=average(learner_actor_ratio, policy_mask),
            player_learner_target_ratio=average(learner_target_ratio, policy_mask),
            # KL values
            player_learner_actor_forward_kl=loss_actor_forward_kl,
            # Own-rows-only variant: the replay reuse-cap controller's
            # set-point must not drift with the tempered explore rows now
            # inside policy_mask (see _update_replay_controller).
            # Modality-resolved split of the SAME k3 estimator. The
            # global mean is an expectation over the policy, so drift
            # confined to a modality carrying ~0.2 mass is diluted to
            # near nothing — which is why the replay reuse controller
            # (whose set-point is the _own variant) cannot fire on a
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
            player_learner_actor_forward_kl_own=(
                loss_actor_forward_kl
                if own_rows is None
                else forward_kl_loss(
                    policy_ratio=learner_actor_ratio,
                    log_policy_ratio=learner_actor_log_ratio,
                    valid=policy_mask & own_rows[None, :],
                )
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
        # Own frames only: explore intake rows must not advance the league
        # add cadence (add_player_min_frames counts main's own play).
        frame_count=player_state.frame_count
        + (
            player_valid.sum()
            if own_rows is None
            else (player_valid & own_rows[None, :]).sum()
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
        if own_rows is not None:
            # Explore rows stay out of the builder losses: builder targets
            # are corrected only by the builder's own ISR — there is no
            # correction path for the PLAYER's raised temperature, so a
            # tempered game's outcome grades the team under deliberately
            # noisy piloting.
            builder_valid = builder_valid & own_rows[None, :]

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
_TRAIN_STEP_JIT = jax.jit(
    train_step,
    static_argnames=["config"],
    donate_argnames=["player_state", "builder_state"],
)


def _chunk_required_shape(traj: Trajectory) -> tuple[int, int]:
    """Smallest (chunk_rows, history_rows) this chunk fits LOSSLESSLY.

    T: rows up to and including the done row. Trailing padding rows are
    copies of the terminal step with done zeroed (PlayerActor.make_chunk),
    so trimming them changes nothing any cumsum-done mask or [-1] outcome
    read sees — the row surviving at [-1] carries the same terminal-step
    content. A mid-game chunk has no done row and requires full length.

    H: the stored window is already tail-clipped and REBASED to packed
    row 0 (clip_history_windows_tail at the actor), so keeping every
    valid field step needs only history_rows >= valid steps and
    2 * history_rows >= valid packed rows — under which a re-clip
    degenerates to slicing zero padding (no rebase, nothing dropped).
    """
    done = np.asarray(traj.player_transitions.env_output.done)
    done_rows = done.reshape(done.shape[0], -1).any(axis=-1)
    t_req = int(done_rows.argmax()) + 1 if done_rows.any() else done.shape[0]
    field = np.asarray(traj.player_history.field)
    valid_steps = int(field[:, FieldFeature.FIELD_FEATURE__VALID].sum())
    packed_valid = int(
        (
            np.asarray(traj.player_packed_history.revealed_cache)[
                ..., EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
            ]
            != SpeciesEnum.SPECIES_ENUM___UNSPECIFIED
        ).sum()
    )
    h_req = max(valid_steps, -(-packed_valid // 2), 1)
    return t_req, h_req


def _trim_to_lattice(
    batch: list[Trajectory], lattice: tuple[tuple[int, int], ...]
) -> list[Trajectory]:
    """Slices every chunk's T/H-leading axes down to the first lattice
    combo that fits the batch's content losslessly (see
    _chunk_required_shape). The lattice is a CHAIN ascending in both dims
    whose last entry is the full stored shape, so a fitting combo always
    exists and selecting it is a max + linear scan. Slicing only — no
    padding, no rebase, no data-derived shapes: the set of shapes XLA can
    ever see is exactly the enumerated lattice."""
    if len(lattice) <= 1:
        return batch
    t_req = h_req = 1
    for traj in batch:
        t_c, h_c = _chunk_required_shape(traj)
        t_req = max(t_req, t_c)
        h_req = max(h_req, h_c)
    t_out, h_out = lattice[-1]
    for t_c, h_c in lattice:
        if t_c >= t_req and h_c >= h_req:
            t_out, h_out = t_c, h_c
            break
    if (t_out, h_out) == lattice[-1]:
        return batch
    return [
        traj.replace(
            player_transitions=jax.tree.map(
                lambda x: x[:t_out], traj.player_transitions
            ),
            player_history=jax.tree.map(lambda x: x[:h_out], traj.player_history),
            player_packed_history=jax.tree.map(
                lambda x: x[: 2 * h_out], traj.player_packed_history
            ),
        )
        for traj in batch
    ]


def _stack_batch(
    batch: list[Trajectory],
    rng_key: jax.Array = None,
    lattice: tuple[tuple[int, int], ...] = (),
) -> Batch:
    """Stacks a list of fixed-shape trajectory chunks into a Batch.

    Chunked unrolls (2026-08-16) made every stored trajectory exactly
    (player_chunk_length, player_history_length)-shaped at the actor
    (PlayerActor.unroll), so the geometric shared-bucket machinery that
    used to live here — one clip level per batch, sized by the batch's
    longest game — is gone, and with it the whole family of
    _TRAIN_STEP_JIT shape variants it generated (each a separately
    compiled executable with its own workspace: the first top-bucket
    batch of a session, arriving once games ran long enough, is what
    OOM'd sessions 1786537634 and 1786712180).

    The static shape LATTICE (2026-08-20, config.player_shape_lattice) is
    the bounded successor: batches are trimmed to the first of a fixed,
    enumerated chain of combos that fits their content losslessly, and
    every combo is precompiled at startup (Learner._precompile_lattice) —
    the failure mode above was the surprise LATE compile of a data-derived
    shape, not the existence of a second executable."""
    batch = _trim_to_lattice(batch, tuple(lattice))
    stacked_trajectory: Trajectory = jax.tree.map(
        lambda *xs: np.stack(xs, axis=1), *batch
    )

    return Batch(
        builder_transitions=stacked_trajectory.builder_transitions,
        builder_history=stacked_trajectory.builder_history,
        player_transitions=stacked_trajectory.player_transitions,
        player_packed_history=stacked_trajectory.player_packed_history,
        player_history=stacked_trajectory.player_history,
        reuse_count=(
            ()
            if isinstance(stacked_trajectory.reuse_count, tuple)
            else stacked_trajectory.reuse_count
        ),
        explore=(
            ()
            if isinstance(stacked_trajectory.explore, tuple)
            else stacked_trajectory.explore
        ),
        rng_key=rng_key,
    )


@dataclasses.dataclass
class RunState:
    """The training run's mutable state.

    Kept as a container rather than flattened onto Learner because it
    draws a clean line: everything here is per-lineage and mutable (train
    state, replay stores, controller EMAs, queues, worker threads), while
    Learner owns the process-wide singletons (the League, the gpu_lock,
    the compiled train_step). Was PopulationState until 2026-08-21, when
    the MainExploiter/LeagueExploiter populations were removed and the
    dict-of-one plus its single-inhabitant Literal went with them.
    """

    wandb_run: wandb.wandb_run.Run
    player_replay: PlayerTrajectoryStore
    builder_replay: BuilderTrajectoryStore
    player_state: Porygon2PlayerTrainState | None = None
    builder_state: Porygon2BuilderTrainState | None = None
    created_at_frame: int | None = None
    host_step: int = 0
    device_q: "queue.Queue" = None
    log_q: "queue.Queue" = None
    ckpt_q: "queue.Queue" = None
    # The 3 internal background workers (host_to_device/
    # log/checkpoint) — owned and joined entirely within this file.
    worker_threads: list = dataclasses.field(default_factory=list)
    # The PlayerActor/BuilderActor game-playing threads —
    # constructed and started by main.py (Learner can't import
    # player_actor.py/builder_actor.py without a circular import, since
    # both already import Learner), registered here via
    # Learner.register_actor_threads so a shutdown or reset waits for them
    # too, not just the 3 internal workers.
    actor_threads: list = dataclasses.field(default_factory=list)
    stop_signal: list = dataclasses.field(default_factory=lambda: [False])
    done: bool = False
    # Actor gate: set = the actor threads may play games. Held open for
    # the whole run; kept because the actor threads wait on it between
    # games and shutdown relies on that same wait.
    run_gate: "threading.Event" = None
    replay_pi: PILogController | None = None
    # Fixed at config.player_replay_kl_target — the ExploitabilityController
    # that used to scale it was removed 2026-08-14 (last of the adaptive
    # hyperparameter loops; see rl/online/controllers.py's module docstring).
    replay_kl_target: float = 0.045
    replay_ctrl_kl_sum: float = 0.0
    replay_ctrl_kl_count: int = 0
    replay_ctrl_prev_adds: int = 0
    replay_ctrl_prev_samples: int = 0
    replay_realised_ratio: float = float("nan")
    # Cumulative frames trained since process start. Telemetry only.
    frames_trained_total: int = 0
    # Monotonic train-tick counter over the WHOLE wandb-run
    # lifetime: restored from the checkpoint's host blob, so it never
    # rewinds or resets across a resume. Logged as "lifetime_step" with every
    # metric and set as the run's default x-axis (wandb.define_metric in
    # main.py) — charts read as cumulative training progress instead of
    # the sawtooth/overdraw that _step (log-call count) and the
    # per-attempt counters produce across resumes and re-forks.
    lifetime_step: int = 0
    consumer_progress: object = None
    train_progress: object = None

    def __post_init__(self):
        if self.device_q is None:
            self.device_q = queue.Queue(maxsize=1)
        if self.log_q is None:
            self.log_q = queue.Queue(maxsize=64)
        if self.ckpt_q is None:
            self.ckpt_q = queue.Queue(maxsize=2)
        if self.run_gate is None:
            self.run_gate = threading.Event()


class Learner:
    """Owns the League, the gpu_lock, the compiled train_step and the one
    live RunState. The MainExploiter/LeagueExploiter populations were
    removed 2026-08-21 — see LESSONS.md 9 for the design and why it never
    ran on this box."""

    def __init__(
        self,
        config: Porygon2LearnerConfig,
        league: League,
        player_state: Porygon2PlayerTrainState,
        builder_state: Porygon2BuilderTrainState,
        main_wandb_run: wandb.wandb_run.Run,
        gpu_lock: LockType | None = None,
        debug: bool = False,
        controller_bytes: bytes | None = None,
        spawn_actor_pool: "Callable[[], None] | None" = None,
    ):
        self.config = config
        self.league = league
        self.gpu_lock = gpu_lock or nullcontext()
        self.debug = debug
        # Lets main.py spin up the actor pool once the run state exists
        # (main.py owns PlayerActor/BuilderActor construction — Learner
        # can't import those without a circular import). None is fine for
        # standalone construction (tests, debug scripts): the run just
        # never gets actors, matching the "nothing passed in means don't
        # wire it up" convention elsewhere in this file.
        self._spawn_actor_pool = spawn_actor_pool

        # train_step's config is a static jit arg, so every field is part
        # of the compile cache key. One Learner, constructed once, holds
        # one config for the life of the process — nothing varies it. A
        # value that DOES vary during a run must not live in it at all; it
        # needs its own traced pytree argument, because retained
        # executables per distinct static value OOM-killed run 1326
        # (LESSONS.md 1).

        self._train_step_jit = train_step if debug else _TRAIN_STEP_JIT
        # Shape-lattice fail-fast: every combo compiles at the FIRST batch
        # (_precompile_lattice) so no variant can arrive as a surprise
        # compile mid-run. Process-local by design.
        self._shape_lattice_compiled: bool = False

        self.run_state = self._build_run_state(
            player_state,
            builder_state,
            main_wandb_run,
            controller_bytes=controller_bytes,
        )

        self.done = False

    # --- run-state construction ----------------------------------------------

    def _build_run_state(
        self,
        player_state: Porygon2PlayerTrainState,
        builder_state: Porygon2BuilderTrainState,
        wandb_run: wandb.wandb_run.Run,
        controller_bytes: bytes | None = None,
    ) -> RunState:
        """Builds a fresh RunState around an already-constructed
        player_state/builder_state. Controllers and replay are always
        fresh here; restore_controller_state (below) reinstates their EMAs
        from the checkpoint when there is one."""
        config = self.config
        is_not_randoms = config.smogon_format != "randombattle"
        run_state = RunState(
            wandb_run=wandb_run,
            player_replay=PlayerTrajectoryStore(
                max_size=config.player_replay_buffer_capacity,
                max_reuses=config.player_replay_ratio,
                need_tracking=is_not_randoms,
                name="player",
            ),
            builder_replay=BuilderTrajectoryStore(
                max_size=config.builder_replay_buffer_capacity,
                max_reuses=config.builder_replay_ratio,
                name="builder",
            ),
            player_state=player_state,
            builder_state=builder_state,
            created_at_frame=int(jax.device_get(player_state.frame_count)),
            # Seeded from the state's own (restored) step_count, NOT 0: the
            # league keys snapshots by host_step and get_latest_player picks
            # "newest" as max(key) — a session-local counter restarting at 0
            # made every post-restart add key smaller than the restored
            # league's, so the stale pre-restart ref stayed "latest" forever,
            # frames_passed never reset, and "overdue" fired on every
            # league-management tick (the 2026-08-14 10:15 add storm; also
            # the p_{step:08} snapshot-dir overwrite hazard once the counter
            # caught up).
            host_step=int(jax.device_get(player_state.step_count)),
            replay_pi=PILogController(
                initial_log=float(np.log(config.player_replay_ratio)),
                log_min=float(np.log(config.player_replay_ratio_min)),
                log_max=float(np.log(config.player_replay_ratio_max)),
                kp=config.player_replay_ctrl_kp,
                ki=config.player_replay_ctrl_ki,
            ),
            replay_kl_target=float(config.player_replay_kl_target),
            consumer_progress=tqdm(
                desc="consumer", smoothing=0.1, position=next_tqdm_position()
            ),
            train_progress=tqdm(
                desc="batches", smoothing=0.1, position=next_tqdm_position()
            ),
        )
        run_state.run_gate.set()
        self._restore_controller_state(run_state, controller_bytes)
        self.league.update_live(MAIN_KEY, self._create_params_container(run_state))
        return run_state

    # --- trajectory intake ---------------------------------------------------

    def enqueue_traj(self, traj: Trajectory):
        """Called by actors to push data into the run's
        replay buffer."""
        run_state = self.run_state
        add_cond = run_state.player_replay._add_cv
        with add_cond:
            add_cond.wait_for(lambda: run_state.done or run_state.player_replay.ready_to_add())
            if run_state.done:
                return
            run_state.player_replay.add(traj)

        sample_cond = run_state.player_replay._sample_cv
        with sample_cond:
            sample_cond.notify_all()

    # --- background workers --------------------------------------------------

    def host_to_device_worker(self, run_state: RunState):
        """Background thread to batch data and push to the run's
        own GPU queue."""
        max_burst = 8
        batch_size = self.config.batch_size

        sample_cond = run_state.player_replay._sample_cv
        with sample_cond:
            sample_cond.wait_for(
                lambda: run_state.done
                or run_state.player_replay.is_min_fill_fraction_reached(
                    self.config.replay_buffer_min_fill_fraction
                )
            )

        init_key = jax.random.PRNGKey(random.randint(0, 2**16 - 1))
        while not run_state.done:
            for _ in range(max_burst):
                if run_state.done:
                    break

                sample_cond = run_state.player_replay._sample_cv
                with sample_cond:
                    sample_cond.wait_for(
                        lambda: run_state.done
                        or run_state.player_replay.ready_to_sample(batch_size)
                    )
                    if run_state.done:
                        break
                    batch = run_state.player_replay.sample(batch_size)

                # Normalise the exploration-ladder tag every trajectory
                # carries (explore actors mark theirs explore=True at
                # construction — see PlayerActor; train_step keeps those
                # rows out of the league/builder signals only).
                # Trajectories from before the field was populated stack
                # as False, so the shared train_step jit always sees one
                # pytree structure across batches.
                batch = [
                    t.replace(
                        explore=(
                            np.array([False])
                            if isinstance(t.explore, tuple)
                            else np.asarray(t.explore).reshape(1)
                        )
                    )
                    for t in batch
                ]

                add_cond = run_state.player_replay._add_cv
                with add_cond:
                    add_cond.notify_all()

                run_state.consumer_progress.update(batch_size)

                init_key, batch_key = jax.random.split(init_key)
                stacked = _stack_batch(
                    batch,
                    rng_key=batch_key,
                    lattice=self.config.player_shape_lattice,
                )
                while not run_state.done:
                    try:
                        run_state.device_q.put(stacked, timeout=1.0)
                        break
                    except queue.Full:
                        continue

        logger.info("host_to_device_worker exiting.")

    def _wandb_log_worker(self, run_state: RunState):
        """Background thread: drains log dicts for the run,
        paying the device->host transfer and wandb serialization here so
        the train loop never has to synchronize with the GPU per step. A
        single consumer preserves wandb step ordering. Also hosts the replay-ratio controller, which
        needs exactly the host-side per-step logs this thread already
        produces."""
        while True:
            logs = run_state.log_q.get()
            if logs is None:
                break
            try:
                host_logs = jax.device_get(logs)
                self._update_replay_controller(run_state, host_logs)
                run_state.wandb_run.log(host_logs)
            except Exception:
                logger.exception("wandb logging failed")

    def _checkpoint_writer_worker(self, run_state: RunState):
        """Background thread: does the actual checkpoint disk I/O so the
        training loop never blocks on it. Payloads are already fully
        host-side and pre-serialized by
        the time they're queued (see _handle_periodic_tasks) — this thread
        never touches a live device buffer or mutates self.league
        directly, only writes what it was handed."""
        while True:
            payload = run_state.ckpt_q.get()
            if payload is None:
                break
            try:
                write_checkpoint_components(
                    payload["save_path"],
                    payload["learner_config"],
                    payload["player_components"],
                    payload["builder_components"],
                    payload["league_bytes"],
                    payload["controller_bytes"],
                    step_count=payload["step_count"],
                    frame_count=payload["frame_count"],
                )
            except Exception:
                logger.exception(
                    "Background checkpoint write failed @ "
                    "step %s — the next periodic checkpoint will simply try "
                    "again.",
                    payload.get("step_count"),
                )

    # --- controller state ----------------------------------------------------

    def controller_state_bytes(self, run_state: RunState) -> bytes:
        """Host-side training dynamics for the checkpoint. Every adaptive
        controller this project built has since been removed (LESSONS.md
        §10), so what is left is the monotonic x-axis counter — but the
        section-wise shape is kept: it is what lets a checkpoint written by
        a superseded revision resume without failing."""
        state = {
            # Monotonic per-run x-axis counter (see RunState.
            # lifetime_step) — restored so charts never rewind at a resume.
            "lifetime_step": run_state.lifetime_step,
        }
        return pickle.dumps(state)

    def _restore_controller_state(
        self, run_state: RunState, data: bytes | None
    ) -> None:
        """Counterpart to controller_state_bytes. Missing sections (older
        checkpoints, or a controller since removed) are simply skipped.

        Never fatal: this state only saves a controller some re-warmup, so
        a blob written by a superseded revision must not be able to fail a
        resume."""
        if not data:
            return
        try:
            state = pickle.loads(data)
        except Exception:
            logger.exception("controller state unreadable — starting fresh")
            return
        # Pre-lifetime_step checkpoints fall back to host_step, which is
        # exact: the counter only ever advances with training.
        run_state.lifetime_step = int(state.get("lifetime_step", run_state.host_step))
        # Checkpoints written before the controller removals (entropy_ctrl
        # 2026-08-13; lambda_ctrl/exploit_ctrl 2026-08-14) carry those
        # sections — simply never read, same as any other extra section.

    # No _update_hyper_controllers anymore: the magnet KL coef became a
    # fixed config scalar when the AdaptivityController was removed
    # (2026-08-13), and the advantage lambda went with the
    # LambdaGapController (2026-08-14) — UPGO's per-step cut plus the
    # fixed player_lambda replaced it (see targets.py). The replay
    # reuse-cap controller below is the one remaining per-log-tick loop.

    def _update_replay_controller(self, run_state: RunState, host_logs: dict) -> None:
        """Velocity-form PI loop holding the replayed-batch actor KL at
        run_state.replay_kl_target by adjusting the reuse
        cap."""
        config = self.config
        if not config.player_replay_ctrl_enabled:
            return
        # The _own variant excludes tempered explore rows (which train the
        # policy since 2026-08-17 but would inflate the mean KL and make
        # the controller silently cut the reuse cap).
        kl = host_logs.get("player_learner_actor_forward_kl_own")
        if kl is not None and np.isfinite(kl):
            run_state.replay_ctrl_kl_sum += float(kl)
            run_state.replay_ctrl_kl_count += 1

        if run_state.replay_ctrl_kl_count >= config.player_replay_ctrl_interval:
            kl_mean = run_state.replay_ctrl_kl_sum / run_state.replay_ctrl_kl_count
            run_state.replay_ctrl_kl_sum = 0.0
            run_state.replay_ctrl_kl_count = 0

            err = (run_state.replay_kl_target - kl_mean) / run_state.replay_kl_target
            run_state.replay_pi.step(err)

            cap = int(round(np.exp(run_state.replay_pi.log)))
            if cap != run_state.player_replay.max_reuses:
                run_state.player_replay.set_max_reuses(cap)

            adds = run_state.player_replay.total_adds
            samples = run_state.player_replay.total_samples
            delta_adds = adds - run_state.replay_ctrl_prev_adds
            delta_samples = samples - run_state.replay_ctrl_prev_samples
            run_state.replay_ctrl_prev_adds = adds
            run_state.replay_ctrl_prev_samples = samples
            if delta_adds > 0:
                run_state.replay_realised_ratio = delta_samples / delta_adds

        host_logs["player_replay_max_reuses"] = float(run_state.player_replay.max_reuses)
        host_logs["player_replay_realised_ratio"] = run_state.replay_realised_ratio

    # --- scheduler -----------------------------------------------------------

    def _ready_run_state(self) -> RunState | None:
        """The run state if it is ready to train this tick, else None:
        warm-enough replay buffer, a batch already on device.

        The .empty() peek is race-free for our purposes: this (the train
        loop) is the sole consumer of the device_q, so an observed
        non-empty queue can't be emptied by anyone else before train()
        collects it."""
        run_state = self.run_state
        if (
            run_state is not None
            and run_state.player_state is not None
            and run_state.player_replay.is_min_fill_fraction_reached(
                self.config.replay_buffer_min_fill_fraction
            )
            and not run_state.device_q.empty()
        ):
            return run_state
        return None

    def train(self):
        """Training loop. Each tick: check readiness (_ready_run_state),
        pull one batch from the device_q, train via the compiled train_step
        under the gpu_lock, run the periodic tasks. The actor pool runs
        continuously and independently."""
        for run_state in (self.run_state,):
            self._start_workers(run_state)

        try:
            for _ in range(self.config.num_steps):
                if self.done:
                    break
                run_state = self._ready_run_state()
                if run_state is None:
                    # Nothing has a warm-enough replay buffer yet (e.g. at
                    # process start, before main's own buffer fills), or
                    # the device_q is momentarily
                    # empty — brief wait rather than a busy spin.
                    threading.Event().wait(timeout=0.1)
                    continue

                try:
                    # Never blocks: _ready_run_state only returns after
                    # observing a batch, and this thread
                    # is the sole consumer of every device_q.
                    batch = run_state.device_q.get_nowait()
                except queue.Empty:
                    continue
                with self.gpu_lock:
                    batch = jax.device_put(batch)
                    logs = self._train_step(run_state, batch)

                run_state.host_step += 1
                run_state.lifetime_step += 1
                run_state.frames_trained_total = (
                    int(jax.device_get(run_state.player_state.frame_count))
                    - run_state.created_at_frame
                )
                self._handle_periodic_tasks(run_state, run_state.host_step, logs)

        except KeyboardInterrupt:
            # One synchronous full save so a deliberate restart loses
            # nothing since the last periodic checkpoint.
            logger.info("Keyboard interrupt received. Saving checkpoint...")
            run_state = self.run_state
            try:
                self._write_checkpoint(run_state, synchronous=True)
            except RuntimeError:
                logger.exception(
                    "Skipping interrupt checkpoint: train state was donated "
                    "mid-step. Latest periodic checkpoint is unaffected."
                )
            raise
        except Exception:
            # logger.exception, NOT traceback.print_exc(): the logging
            # handler routes through tqdm.write(), so the traceback prints
            # cleanly above the progress bars — print_exc() wrote raw to
            # stderr and got shredded line-by-line into the concurrent bar
            # redraws (session 1786537634's OOM traceback was near-
            # unreadable in the captured console for exactly this reason).
            logger.exception("Learner training crashed")
            raise
        finally:
            self.done = True
            for run_state in (self.run_state,):
                # strict=False: process is exiting — a straggler here is
                # tolerable (daemon threads die with the process), and
                # raising would mask the real outcome, turning e.g. a
                # clean Ctrl-C into a crash. Resets keep strict=True.
                self._stop_workers(run_state, strict=False)
            tqdm.write("Training Finished.")

    def register_actor_threads(
        self, threads: list[threading.Thread]
    ) -> None:
        """Called by main.py right after it constructs and starts a
        run's PlayerActor/BuilderActor pool (in response to the
        spawn_actor_pool callback, on creation, or after a reset) — Learner
        can't spawn these itself without a circular import. Registering
        them here means a shutdown waits for (and
        straggler-checks) them exactly like the 3 internal workers,
        instead of silently leaving them running against now-stale state."""
        self.run_state.actor_threads.extend(threads)

    def _start_workers(self, run_state: RunState) -> None:
        transfer_thread = threading.Thread(
            target=self.host_to_device_worker,
            args=(run_state,),
            daemon=True,
            name="transfer",
        )
        transfer_thread.start()
        log_thread = threading.Thread(
            target=self._wandb_log_worker,
            args=(run_state,),
            daemon=True,
            name="log",
        )
        log_thread.start()
        ckpt_thread = threading.Thread(
            target=self._checkpoint_writer_worker,
            args=(run_state,),
            daemon=True,
            name="ckpt",
        )
        ckpt_thread.start()
        run_state.worker_threads.extend([transfer_thread, log_thread, ckpt_thread])

    def _stop_workers(
        self, run_state: RunState, strict: bool = True
    ) -> None:
        run_state.done = True
        run_state.stop_signal[0] = True
        # Wake actors idling at the block gate so they observe stop_signal
        # immediately instead of on their next wait() timeout.
        run_state.run_gate.set()
        try:
            run_state.device_q.get_nowait()
        except queue.Empty:
            pass
        for cond in (
            run_state.player_replay._add_cv,
            run_state.player_replay._sample_cv,
            run_state.builder_replay._add_cv,
            run_state.builder_replay._sample_cv,
        ):
            with cond:
                cond.notify_all()

        for t in run_state.worker_threads:
            if t.name.startswith("transfer-"):
                t.join(timeout=10)
        run_state.log_q.put(None)
        for t in run_state.worker_threads:
            if t.name.startswith("log-"):
                t.join(timeout=30)
        run_state.ckpt_q.put(None)
        for t in run_state.worker_threads:
            if t.name.startswith("ckpt-"):
                t.join(timeout=60)
        # External actor threads (main.py's PlayerActor/BuilderActor pool,
        # registered via register_actor_threads): already signalled via
        # run_state.stop_signal[0] above — just wait for them here.
        for t in run_state.actor_threads:
            t.join(timeout=30)

        all_threads = run_state.worker_threads + run_state.actor_threads
        stragglers = [t for t in all_threads if t.is_alive()]
        if stragglers:
            logger.warning(
                "%d worker thread(s) did not stop within "
                "their join timeout: %s — giving a 30s grace period before "
                "treating this as a hung shutdown.",
                len(stragglers),
                [t.name for t in stragglers],
            )
            for t in stragglers:
                t.join(timeout=30)
            stragglers = [t for t in stragglers if t.is_alive()]
        if stragglers:
            if strict:
                raise RuntimeError(
                    f"{len(stragglers)} worker thread(s) never stopped: "
                    f"{[t.name for t in stragglers]}. Refusing to proceed with "
                    "training state still reachable from a live thread."
                )
            # strict=False is the whole-PROCESS shutdown path (train()'s
            # finally, incl. Ctrl-C): the straggler raise exists to stop a
            # rebuild from starting on top of state a leaked
            # thread still holds (the 2026-08-11 RAM/VRAM leak) — at
            # process exit there is no next phase to protect, every
            # thread is a daemon that dies with the process, and raising
            # here would convert a clean Ctrl-C into a "crashed" outcome
            # (an actor blocked on the game-server websocket mid-game is
            # normal at this point, not a leak).
            logger.warning(
                "%d thread(s) still alive at process "
                "shutdown: %s — proceeding; they are daemons and exit "
                "with the process.",
                len(stragglers),
                [t.name for t in stragglers],
            )

        # Return the 4 progress-bar rows to the shared pool
        # (close_tqdm_bar) so the replacement fork reuses the same rows —
        # without this, every rebuild leaked 4 dead rows and
        # pushed all live bars one screen-row further down, unboundedly,
        # for the life of the process. Closing is safe against any update
        # racing in from a straggler: tqdm's close() flips .disable, which
        # every update() checks first.
        for bar in (
            run_state.consumer_progress,
            run_state.train_progress,
            run_state.player_replay._progress,
            run_state.builder_replay._progress,
        ):
            close_tqdm_bar(bar)

    # Known python-thread name prefixes, for the census below — anything
    # unrecognized lands in "other".
    _THREAD_NAME_BUCKETS = (
        "Selfplay-",
        "BuilderActor-",
        "EvalActor-",
        "transfer-",
        "log-",
        "ckpt-",
        "inference-server",
        "ThreadPoolExecutor",
    )

    def _log_memory_diagnostics(self, logs: dict) -> None:
        """Process-wide RAM attribution, riding main's periodic wandb logs
        every memory_diag_interval steps.

        Motivated by session 1786537634: RSS climbed 5.9GB -> 17GB while
        the OS thread count grew 478 -> 775 with no obvious owner, and
        none of it was attributable from wandb alone. The bounded-by-
        design consumers (replay buffers, league opponent cache) get
        exact byte counts here; the thread census separates python
        threads (named, bucketed below) from native ones — if
        diag_os_threads far exceeds diag_py_threads, the growth lives in
        native pools (XLA/CUDA/websocket internals), not python code."""
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        logs["diag_rss_mb"] = int(line.split()[1]) / 1024.0
                    elif line.startswith("Threads:"):
                        logs["diag_os_threads"] = int(line.split()[1])
        except Exception:
            pass  # non-Linux — skip, same posture as _available_memory_fraction

        py_threads = threading.enumerate()
        logs["diag_py_threads"] = len(py_threads)
        buckets = dict.fromkeys(self._THREAD_NAME_BUCKETS, 0)
        buckets["other"] = 0
        for t in py_threads:
            for prefix in self._THREAD_NAME_BUCKETS:
                if t.name.startswith(prefix):
                    buckets[prefix] += 1
                    break
            else:
                buckets["other"] += 1
        for prefix, count in buckets.items():
            key = prefix.rstrip("-").lower().replace("-", "_")
            logs[f"diag_py_threads_{key}"] = count

        # Heap census: attributes host RSS the byte-exact counters below
        # (replay buffers, league cache) don't cover — e.g. the ~3GB the
        # 2026-08-18 fork jump left unexplained by thread counts
        # and league cache alone. sys.getsizeof is shallow (a dict/list's
        # own overhead, not its contents), but that's exactly what surfaces
        # a genuine culprit: a huge COUNT of one type (numpy arrays, proto
        # objects, EnvironmentState instances) dominating aggregate bytes.
        # Logged to the console, not wandb — dynamic top-N, not a stable
        # scalar key. gc.get_objects() walks the whole heap, so this rides
        # the same 5000-step cadence as the rest of this function.
        try:
            counts = collections.Counter()
            sizes = collections.Counter()
            for obj in gc.get_objects():
                t = type(obj).__name__
                counts[t] += 1
                sizes[t] += sys.getsizeof(obj)
            top = sizes.most_common(15)
            logger.info(
                "Heap census (top-15 by approx shallow size): %s",
                ", ".join(f"{t}={counts[t]}objs/{sz / 2**20:.1f}MB" for t, sz in top),
            )
        except Exception:
            logger.exception("Heap census failed")

        run_state = self.run_state
        logs["diag_player_replay_mb"] = run_state.player_replay.nbytes() / 2**20
        logs["diag_builder_replay_mb"] = run_state.builder_replay.nbytes() / 2**20
        try:
            with open("runtime/service_memory.json") as f:
                node_stats = json.load(f)
            # Service writes every 10s; 60s is a generous staleness bound
            # in case the file is left over from a service that's since died.
            if time.time() - node_stats["ts"] < 60:
                for key in (
                    "rss_mb",
                    "heap_used_mb",
                    "num_workers",
                    "worker_heap_used_mb",
                    "workers_reported",
                ):
                    logs[f"diag_node_{key}"] = node_stats[key]
        except Exception:
            pass  # service not up, stats file stale/absent, or race on the rename

        entries, cache_bytes = self.league.cache_stats()
        logs["diag_league_cache_entries"] = entries
        logs["diag_league_cache_mb"] = cache_bytes / 2**20

    def _precompile_lattice(self, run_state: RunState, batch: Batch) -> None:
        """Fail-fast compilation of EVERY lattice combo at the first
        batch, so a shape variant can never arrive as a surprise compile
        mid-run (the exact mechanism that OOM'd the geometric-bucket
        sessions ~20min in: the first top-bucket batch). Each combo is
        exercised through the real jit with a resized copy of the first
        real batch and a COPY of the train states (the jit donates its
        state args; outputs are discarded), so the dispatch cache is
        warm and any compile-time OOM happens at launch, before hours of
        training are at stake. Runs under the caller's gpu_lock.

        Resizing pads the T axis by repeating each chunk's final row —
        the actor's own padding convention, so no all-invalid mask rows
        are fabricated — and pads/slices the H axes with zeros (zero
        history rows are ordinary invalid steps)."""
        lattice = tuple(self.config.player_shape_lattice)
        full = (self.config.player_chunk_length, self.config.player_history_length)
        assert lattice[-1] == full, (lattice, full)
        assert all(
            a[0] <= b[0] and a[1] <= b[1] for a, b in zip(lattice, lattice[1:])
        ), f"player_shape_lattice must be an ascending chain: {lattice}"
        assert len(lattice) <= 4, f"lattice too large (memory risk): {lattice}"
        if len(lattice) <= 1:
            return

        def resize_time(x, target):
            if x.shape[0] == target:
                return x
            if x.shape[0] > target:
                return x[:target]
            pad = jnp.repeat(x[-1:], target - x.shape[0], axis=0)
            return jnp.concatenate([x, pad], axis=0)

        def resize_zeros(x, target):
            if x.shape[0] == target:
                return x
            if x.shape[0] > target:
                return x[:target]
            widths = [(0, target - x.shape[0])] + [(0, 0)] * (x.ndim - 1)
            return jnp.pad(x, widths)

        current = (
            batch.player_transitions.env_output.done.shape[0],
            batch.player_history.field.shape[0],
        )
        for t_c, h_c in lattice:
            if (t_c, h_c) == current:
                continue  # the real call right after this compiles it
            resized = batch.replace(
                player_transitions=jax.tree.map(
                    lambda x: resize_time(x, t_c), batch.player_transitions
                ),
                player_history=jax.tree.map(
                    lambda x: resize_zeros(x, h_c), batch.player_history
                ),
                player_packed_history=jax.tree.map(
                    lambda x: resize_zeros(x, 2 * h_c), batch.player_packed_history
                ),
            )
            logger.info("Precompiling train_step shape combo (%d, %d)…", t_c, h_c)
            start = time.time()

            # True buffer copies with identical avals: the jit donates its
            # state args, so passing the live states would free them, and
            # a leaf-type-changing copy (e.g. jnp.copy on a weak-typed
            # python scalar) would trace as yet another variant.
            def copy_state(tree):
                return jax.tree.map(
                    lambda x: (
                        jnp.array(x, copy=True) if isinstance(x, jax.Array) else x
                    ),
                    tree,
                )

            self._train_step_jit(
                copy_state(run_state.player_state),
                copy_state(run_state.builder_state),
                resized,
                self.config,
            )
            logger.info(
                "Compiled (%d, %d) in %.1fs.", t_c, h_c, time.time() - start
            )

    def _train_step(self, run_state: RunState, batch: Batch) -> dict:
        """Runs the JAX update, rebinding the result onto run_state."""
        if not self._shape_lattice_compiled:
            self._precompile_lattice(run_state, batch)
            self._shape_lattice_compiled = True
        run_state.player_state, run_state.builder_state, logs = self._train_step_jit(
            run_state.player_state,
            run_state.builder_state,
            batch,
            self.config,
        )
        return logs

    def _handle_periodic_tasks(self, run_state: RunState, step: int, logs: dict):
        """Handles logging, progress bars, and checkpointing for run_state."""
        run_state.train_progress.update(1)

        if (
            self.config.smogon_format != "randombattle"
            and step % self.config.save_interval_steps == 0
        ):
            logs.update(self._get_usage_counts(run_state))

        if step % self.config.league_winrate_log_steps == 0:
            logs.update(self._get_league_winrates(run_state))
            logs.update(self._get_league_winrate_heatmap(run_state))

        if (
            self.config.memory_diag_interval > 0
            and step % self.config.memory_diag_interval == 0
        ):
            self._log_memory_diagnostics(logs)

        # The default x-axis for every metric on the run
        # (wandb.define_metric in main.py): monotonic across resumes AND
        # attempt re-forks, unlike host_step/frames.
        logs["lifetime_step"] = run_state.lifetime_step
        run_state.log_q.put(logs)

        # PlayerActor.pull_own_player() reads league.get_live(MAIN_KEY),
        # so this is what makes the actors play the CURRENT policy rather
        # than the one they were started with.
        if step % self.config.main_player_update_steps == 0:
            self.league.update_live(MAIN_KEY, self._create_params_container(run_state))

        if step % self.config.save_interval_steps == 0:
            self._write_checkpoint(run_state)

        if step % self.config.manage_league_interval == 0:
            self._manage_league(run_state, step)

        self._check_oom_guard(run_state, step)

    def _write_checkpoint(self, run_state: RunState, synchronous: bool = False) -> str:
        """Writes the full resumable state: params, target_params,
        opt_state, host counters, the serialized League and the controller
        blob, keyed to the run's own step_count.

        Everything host-side/fast happens synchronously here (device
        pulls, small in-memory serializations); only the actual disk
        write goes to run_state's own background writer, via a payload that's
        already fully host-side (plain dicts/bytes/ints, never a live
        TrainState or the live League object itself). synchronous=True
        (Ctrl-C/OOM-guard path) writes inline instead, since there may be
        no time left for the background writer to run."""
        host_player_state = jax.device_get(run_state.player_state)
        host_builder_state = jax.device_get(run_state.builder_state)
        player_components = dict(
            params=host_player_state.params,
            target_params=host_player_state.target_params,
            opt_state=host_player_state.opt_state,
            scalars=dict(
                step_count=host_player_state.step_count,
                frame_count=host_player_state.frame_count,
            ),
        )
        builder_components = dict(
            params=host_builder_state.params,
            target_params=host_builder_state.target_params,
            opt_state=host_builder_state.opt_state,
            scalars=dict(
                step_count=host_builder_state.step_count,
                frame_count=host_builder_state.frame_count,
            ),
        )
        save_path = os.path.abspath(
            os.path.join(
                f"./ckpts/gen{self.config.generation}",
                f"ckpt_{int(np.asarray(host_player_state.step_count)):08}",
            )
        )
        payload = dict(
            save_path=save_path,
            learner_config=self.config,
            player_components=player_components,
            builder_components=builder_components,
            league_bytes=self.league.serialize(),
            controller_bytes=self.controller_state_bytes(run_state),
            step_count=int(np.asarray(host_player_state.step_count)),
            frame_count=int(np.asarray(host_player_state.frame_count)),
        )
        if synchronous:
            return write_checkpoint_components(
                payload["save_path"],
                payload["learner_config"],
                payload["player_components"],
                payload["builder_components"],
                payload["league_bytes"],
                payload["controller_bytes"],
                step_count=payload["step_count"],
                frame_count=payload["frame_count"],
            )
        run_state.ckpt_q.put(payload)
        return save_path

    def _manage_league(self, run_state: RunState, step: int):
        """Checks whether a new snapshot should be added to the league."""
        reason = self._should_add_new_player(run_state)
        if reason is not None:
            tqdm.write(f"Adding new player to league @ {step} ({reason})")
            self._add_player_to_league(run_state, step, origin="main")
            run_state.player_replay.reset_usage_counts()

    def _available_memory_fraction() -> float | None:
        """Fraction of total system RAM currently available (reclaimable
        caches counted as available, matching what actually predicts an
        OOM kill), or None if it can't be determined (non-Linux, or
        /proc/meminfo unreadable) — the caller treats None as "skip the
        check", the same defensive posture as this codebase's other
        optional-environment guards (e.g. the matplotlib import)."""
        try:
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    key, value = line.split(":", 1)
                    meminfo[key] = int(value.strip().split()[0])  # kB
            return meminfo["MemAvailable"] / meminfo["MemTotal"]
        except Exception:
            return None

    def _check_oom_guard(self, run_state: RunState, step: int) -> None:
        """Self-monitoring safety valve, not a leak fix: if available RAM
        drops below config.oom_guard_min_available_fraction, save a
        full checkpoint now and raise OOMGuardTriggered — better to stop on
        our own terms with a guaranteed-complete checkpoint than let the
        kernel's OOM killer pick an arbitrary moment (possibly mid-write)
        to SIGKILL this process."""
        if (
            not self.config.oom_guard_enabled
            or step % self.config.oom_guard_check_interval != 0
        ):
            return
        available_fraction = self._available_memory_fraction()
        if (
            available_fraction is not None
            and available_fraction < self.config.oom_guard_min_available_fraction
        ):
            logger.warning(
                "Available memory fraction %.3f < oom_guard_min_available_fraction "
                "%.3f @ step %d — saving a checkpoint and "
                "stopping before the kernel OOM-kills this process.",
                available_fraction,
                self.config.oom_guard_min_available_fraction,
                step,
            )
            save_path = self._write_checkpoint(
                self.run_state, synchronous=True
            )
            raise OOMGuardTriggered(save_path)

    # (_measure_exploitability/_update_exploit_controller/_apply_exploit_
    # scale removed 2026-08-14 with the ExploitabilityController — the
    # worst-matchup win-rate signal still exists in _should_add_new_player's
    # "dominant" gate; it just doesn't actuate anything anymore.)

    def _should_add_new_player(self, run_state: RunState) -> AddReason | None:
        """Returns why a snapshot should join the league, or None to skip.
        main only."""
        # Pacing is measured against main's OWN last checkpoint (AlphaStar
        # MainPlayer.ready_to_checkpoint: steps since self._checkpoint_step),
        # not the league's newest entry — a foreign-origin publication
        # would otherwise become "latest" permanently (its offset key wins
        # max()) with a frame count that never advances, firing an overdue
        # add on every league-management tick.
        latest = self.league.get_latest_player(origin="main")
        current = self.league.get_live(MAIN_KEY)

        latest_frames = latest.player_frame_count if latest is not None else 0
        frames_passed = int(current.player_frame_count - latest_frames)

        if frames_passed < self.config.add_player_min_frames:
            return None

        historical_players = [
            v for k, v in self.league.players.items() if k not in LIVE_KEYS
        ]

        if not historical_players:
            if (
                int(run_state.player_state.step_count)
                > self.config.minimum_historical_player_steps
            ):
                return "initial"
            return None

        win_rates = self.league.get_winrate((current, historical_players))

        if win_rates.min() > 0.7:
            return "dominant"
        if frames_passed >= self.config.add_player_max_frames:
            return "overdue"
        return None

    def _create_params_container(self, run_state: RunState) -> ParamsContainer:
        return ParamsContainer(
            player_frame_count=jax.device_get(run_state.player_state.frame_count).item(),
            builder_frame_count=jax.device_get(run_state.builder_state.frame_count).item(),
            step_count=MAIN_KEY,
            player_params=jax.device_get(run_state.player_state.params),
            builder_params=jax.device_get(run_state.builder_state.params),
        )

    def _add_player_to_league(
        self, run_state: RunState, step: int, origin: str = "main"
    ):
        """Persist the current params as an opponent snapshot and register
        a ref. Only the params files are written (no optimiser state); the
        league holds the lightweight ref and materialises the params
        lazily when this player is actually drawn as an opponent."""
        league_step = step
        players_root = f"./ckpts/gen{self.config.generation}/players"
        snapshot_dir = os.path.abspath(f"{players_root}/p_{league_step:08}")
        checkpoint.save_param_snapshot(
            snapshot_dir,
            player_components=dict(
                params=jax.device_get(run_state.player_state.params),
                target_params=jax.device_get(run_state.player_state.target_params),
            ),
            builder_components=dict(
                params=jax.device_get(run_state.builder_state.params),
                target_params=jax.device_get(run_state.builder_state.target_params),
            ),
        )
        self.league.add_player(
            PlayerRef(
                step_count=league_step,
                snapshot_dir=snapshot_dir,
                player_frame_count=jax.device_get(run_state.player_state.frame_count).item(),
                builder_frame_count=jax.device_get(
                    run_state.builder_state.frame_count
                ).item(),
                player_key="params",
                builder_key="params",
                origin=origin,
            )
        )

    def _get_usage_counts(self, run_state: RunState):
        result = {}
        for key, counts in [
            ("species", run_state.player_replay._species_counts),
            ("items", run_state.player_replay._item_counts),
            ("abilities", run_state.player_replay._ability_counts),
            ("moves", run_state.player_replay._move_counts),
        ]:
            names = list(STOI[key])
            table = wandb.Table(columns=[key, "usage"])
            for name, count in zip(names, counts):
                table.add_data(name, count)
            result[f"{key}_usage"] = table
        return result

    def _winrate_tracked_opponents(self) -> list[PlayerRef]:
        """Every historical league member."""
        return [v for k, v in self.league.players.items() if k not in LIVE_KEYS]

    @staticmethod
    def _ref_label(ref: PlayerRef) -> str:
        """Payoff-table label: the snapshot's own step count."""
        return f"{ref.step_count}"

    def _get_league_winrates(self, run_state: RunState):
        current = self.league.get_live(MAIN_KEY)
        others = self._winrate_tracked_opponents()
        if not others:
            return {}
        win_rates = self.league.get_winrate((current, others))
        # Origin-labelled keys ("league_main_v_ME-1834_winrate") still
        # match scripts/wandb_views.py's ^league_main_v_.*_winrate$ panel
        # regex.
        return {
            f"league_main_v_{self._ref_label(others[i])}_winrate": wr
            for i, wr in enumerate(win_rates)
        }

    def _get_league_winrate_heatmap(self, run_state: RunState):
        """Full pairwise win-rate matrix over the whole shared payoff
        table: live main and every historical snapshot (when they
        exist), and every historical snapshot with an origin-labelled
        row — logged through a custom Vega-Lite chart preset
        (jtwin/league-payoff-heatmap-v10, registered once via
        scripts/register_wandb_charts.py) instead of hijacking wandb's
        confusion-matrix preset: proper axis titles (player/opponent, not
        Actual/Predicted), a red/gold/green win-rate colour band per
        cell, and a text label per cell. The colour is a chain of
        condition/value tests on winrate with NO field bound directly to
        the colour channel — every version that bound colour to a table
        field (scale.range, scale.scheme+domain+clamp, a literal
        per-cell hex column with scale: null) rendered as either an
        unrelated colour or one flat colour for every cell in wandb's
        actual custom-chart panel, confirmed via wandb's own GraphQL API
        (spec stored correctly) and a neutral Vega-Lite renderer (spec
        renders correctly outside wandb) — so wandb's Vega2 runtime does
        not honour a field-bound colour channel here. Condition/value
        (no field) is the one pattern proven to render correctly (the
        text mark's black/white choice used exactly this pattern the
        whole time). Interactive (hover shows exact values), no
        matplotlib figure render on the train-loop thread, no image
        upload per log. row_idx/col_idx carry insertion order so the
        chart's ordinal axes sort by league structure rather than
        wandb's default alphabetical sort. A pair that has never actually
        played just shows the table's prior."""
        current = self.league.get_live(MAIN_KEY)
        others = self._winrate_tracked_opponents()
        if not others:
            return {}

        all_players = [current] + others
        labels = ["main (live)"] + [self._ref_label(p) for p in others]
        matrix = np.asarray(self.league.get_winrate((all_players, all_players)))

        table = wandb.Table(
            columns=["row", "row_idx", "col", "col_idx", "winrate"],
            data=[
                [row, i, col, j, float(matrix[i, j])]
                for i, row in enumerate(labels)
                for j, col in enumerate(labels)
            ],
        )
        chart = wandb.plot_table(
            "jtwin/league-payoff-heatmap-v10",
            table,
            fields={
                "row": "row",
                "row_idx": "row_idx",
                "col": "col",
                "col_idx": "col_idx",
                "winrate": "winrate",
            },
            string_fields={
                "title": "league payoff table (row beats column)"
            },
        )
        return {"league_winrate_heatmap": chart}
