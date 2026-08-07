import logging
import os
import pickle
import queue
import random
import threading
import traceback
from _thread import LockType
from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb.wandb_run
from tqdm import tqdm

import wandb
from rl import checkpoint
from rl.environment.data import CAT_VF_SUPPORT, STOI, PackedSetFeature
from rl.environment.interfaces import (
    Batch,
    BuilderActorInput,
    PlayerActorInput,
    Trajectory,
)
from rl.environment.utils import clip_history, clip_packed_history, geometric_bucket
from rl.model.heads import HeadParams, calculate_hierarchical_prior
from rl.model.utils import Params, ParamsContainer
from rl.online.artifact import (
    Porygon2BuilderTrainState,
    Porygon2PlayerTrainState,
    save_train_state,
)
from rl.online.bandit import LambdaBandit, rating_logs
from rl.online.buffer import BuilderTrajectoryStore, PlayerTrajectoryStore
from rl.online.config import Porygon2LearnerConfig
from rl.online.controllers import AdaptivityController, LambdaGapController
from rl.online.league import MAIN_KEY, League, PlayerRef
from rl.online.loss import (
    backward_kl_loss,
    forward_kl_loss,
    mse_value_loss,
    policy_gradient_loss,
)
from rl.online.plasticity import (
    AddReason,
    PlasticityController,
    shrink_and_perturb_player_state,
)
from rl.online.targets import (
    compute_aux_value_targets,
    compute_builder_targets,
    compute_player_targets,
)
from rl.online.utils import calculate_r2, collect_batch_telemetry_data, promote_map
from rl.utils import average

logger = logging.getLogger(__name__)


def train_step(
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
    batch: Batch,
    config: Porygon2LearnerConfig,
    adv_lambda: jax.Array | None = None,
    magnet_coef: jax.Array | None = None,
):
    """Train for a single step.

    ``adv_lambda`` and ``magnet_coef`` are RUNTIME scalars (traced, not
    static) driven by the lambda/entropy controllers — runtime values
    never recompile; see compute_player_targets for the OOM history
    behind this rule.
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
        adv_lambda=adv_lambda,
    )
    training_logs.update(channel_logs)
    policy_mask = player_targets.policy_mask
    value_mask = player_targets.value_mask

    # Per-lambda v-trace distribution targets for the multi-lambda aux
    # value heads, bootstrapped from each lambda's OWN fast-target readout
    # so every row's target is self-consistent.
    player_aux_targets = compute_aux_value_targets(
        batch,
        aux_value_log_probs=jax.nn.log_softmax(
            player_target_pred.aux_value_logits.astype(jnp.float32), axis=-1
        ),
        isr=target_actor_ratio,
        config=config,
    )
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
        fresh_err = per_traj_err.mean(where=fresh)
        replay_err = per_traj_err.mean(where=~fresh)
        training_logs.update(
            {
                "plasticity_fresh_value_err": fresh_err,
                "plasticity_replay_value_err": replay_err,
                "plasticity_value_err_reuse_gap": fresh_err - replay_err,
            }
        )
    player_targets = promote_map(player_targets, float_dtype)

    if config.player_advantage_ema_enabled:
        # Floor on the std divisor: as the policy converges true
        # advantages shrink, and dividing by a vanishing std amplifies
        # value-estimation noise into the actor precisely when the real
        # signal is weakest. Below the floor, normalisation stops
        # rescaling and the gradient is allowed to get small.
        player_advantages = (
            player_targets.advantages - player_state.ema_adv_mean
        ) / jnp.maximum(player_state.ema_adv_std, config.player_adv_std_floor)
    else:
        player_advantages = player_targets.advantages

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

        # IMPACT surrogate: ratio recentered on the fast target via the
        # clipped correction.
        loss_pg = policy_gradient_loss(
            policy_ratios=learner_actor_ratio * actor_target_clipped_ratio,
            advantages=player_advantages,
            valid=policy_mask,
            threshold=config.player_ppo_clip_threshold,
            objective=config.player_policy_objective,
        )

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

        # Commitment sensor for the adaptivity controller: batch
        # CORRELATION between how much probability the policy put on the
        # action it took and the advantage that action earned. Positive
        # means the policy's confidence is being validated (safe to
        # sharpen); near zero or negative means it is confidently
        # choosing actions that are not paying (hold entropy). Drops on a
        # league addition (habitual actions stop working against an
        # unfamiliar opponent) and after a perturbation (preferences
        # scrambled), which is what makes the controller event-responsive
        # without a schedule.
        #
        # Correlation, not covariance: cov = corr * sd(log pi) * sd(adv),
        # and sd(log pi) shrinks as the policy sharpens — i.e. the raw
        # covariance partly measures the entropy the controller itself
        # actuates, coupling sensor to actuator and making the target
        # depend on the current entropy level. Normalising leaves the
        # relationship alone, bounded in [-1, 1], so commit_target reads
        # as "keep the confidence/payoff correlation above X".
        #
        # Blind to actions the policy never takes — hence the entropy
        # floors remain the hard backstop for modality collapse.
        commit_lp = learner_log_prob
        commit_adv = player_advantages
        lp_dev = commit_lp - average(commit_lp, policy_mask)
        adv_dev = commit_adv - average(commit_adv, policy_mask)
        lp_sd = jnp.sqrt(average(jnp.square(lp_dev), policy_mask))
        adv_sd = jnp.sqrt(average(jnp.square(adv_dev), policy_mask))
        commit_cov = average(lp_dev * adv_dev, policy_mask) / jnp.maximum(
            lp_sd * adv_sd, 1e-6
        )

        # Multi-gamma auxiliary value heads (Metamon/AMAGO-style): each
        # row is a categorical value readout for one auxiliary discount,
        # trained by CE against its own v-trace distribution target.
        # Pure representation shaping across horizons — the policy's
        # advantages read ONLY the main gamma=1 v_head; short-gamma
        # advantages are material/tempo-greedy and not policy-invariant,
        # so they never touch the actor loss.
        aux_logits = learner_player_pred.aux_value_logits.astype(jnp.float32)
        loss_v_aux = average(
            optax.softmax_cross_entropy(
                logits=aux_logits, labels=player_aux_targets
            ).mean(axis=-1),
            value_mask,
        )

        aux_expectations = jax.nn.softmax(aux_logits, axis=-1) @ cat_vf_support.astype(
            jnp.float32
        )
        aux_target_expectations = player_aux_targets @ cat_vf_support.astype(
            jnp.float32
        )
        aux_value_r2 = calculate_r2(
            value_prediction=aux_expectations,
            value_target=aux_target_expectations,
            mask=jnp.broadcast_to(value_mask[..., None], aux_logits.shape[:-1]),
        )
        # Sensor for the lambda controller: mean absolute gap between the
        # main head's value and the lambda=1.0 Monte Carlo anchor row —
        # the live per-batch bootstrap-bias estimate. Blind spot: trunk
        # errors shared by both readouts cancel here, which is why the
        # controller keeps a lambda floor.
        mc_row = config.player_aux_lambdas.index(1.0)
        bootstrap_gap = average(
            jnp.abs(
                learner_value_head.expectation.astype(jnp.float32)
                - aux_expectations[..., mc_row]
            ),
            value_mask,
        )

        # Per-row R2, keyed by lambda. The lambda=1.0 (Monte Carlo) row is
        # the calibration anchor: its gap to the main head is a direct
        # bootstrap-bias readout — large/growing gap means the critic is
        # drifting off the data (replay staleness, self-referential
        # low-lambda targets); tiny gap during a strength plateau points
        # at transfer saturation instead.
        aux_row_r2 = {
            f"player_aux_r2_lam{round(lam * 100):03d}": calculate_r2(
                value_prediction=aux_expectations[..., k],
                value_target=aux_target_expectations[..., k],
                mask=value_mask,
            )
            for k, lam in enumerate(config.player_aux_lambdas)
        }

        loss = (
            config.player_policy_loss_coef * loss_pg
            + config.player_value_head_loss_coef * loss_v_win
            + config.player_kl_loss_coef * loss_actor_backward_kl
            + (config.player_magnet_kl_coef if magnet_coef is None else magnet_coef)
            * loss_magnet_kl
            + config.player_aux_value_coef * loss_v_aux
        )

        return loss, dict(
            # Loss values
            player_loss_pg=loss_pg,
            player_loss_v_win=loss_v_win,
            player_loss_v_aux=loss_v_aux,
            player_loss_kl=loss_actor_backward_kl,
            player_loss_magnet_kl=loss_magnet_kl,
            player_aux_value_r2=aux_value_r2,
            player_bootstrap_gap=bootstrap_gap,
            **aux_row_r2,
            # Per head entropies (diagnostics only — no longer regularized)
            player_action_entropy=action_head_entropy,
            player_action_normalized_entropy=action_head_normalized_entropy,
            player_normalized_modality_entropy=normalized_modality_entropy,
            player_commit_cov=commit_cov,
            # Ratios
            player_learner_actor_ratio=average(learner_actor_ratio, policy_mask),
            player_learner_target_ratio=average(learner_target_ratio, policy_mask),
            # KL values
            player_learner_actor_forward_kl=loss_actor_forward_kl,
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

    player_state = player_state.apply_gradients(grads=player_grads)
    player_state = player_state.replace(
        step_count=player_state.step_count + 1,
        frame_count=player_state.frame_count + player_valid.sum(),
        target_params=optax.incremental_update(
            player_state.params,
            player_state.target_params,
            config.player_ema_update_rate,
        ),
        # Advantage stats use their own, much faster EMA rate: they are
        # scalars estimated from ~180 samples per batch (well-averaged at
        # a 100-step time constant), and a slow EMA mis-scales the policy
        # gradient for ~1k steps after every distribution shift — bandit
        # arm switches most visibly.
        ema_adv_mean=optax.incremental_update(
            player_targets.advantages.mean(where=policy_mask),
            player_state.ema_adv_mean,
            config.player_adv_ema_rate,
        ),
        ema_adv_std=optax.incremental_update(
            player_targets.advantages.std(where=policy_mask),
            player_state.ema_adv_std,
            config.player_adv_ema_rate,
        ),
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
            player_policy_value_mask_ratio=policy_mask.sum()
            / (value_mask.sum() + 1e-8),
            player_state_adv_mean=player_state.ema_adv_mean,
            player_state_adv_std=player_state.ema_adv_std,
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

            # Calculate the losses.
            loss_pg = policy_gradient_loss(
                policy_ratios=builder_policy_ratio,
                advantages=builder_advantages,
                valid=builder_valid,
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

    training_logs.update(collect_batch_telemetry_data(batch, config))
    training_logs.update(
        dict(
            player_frame_count=player_state.frame_count,
            builder_frame_count=builder_state.frame_count,
            training_step=player_state.step_count,
        )
    )

    return player_state, builder_state, training_logs


def _stack_and_pad_batch(
    batch: list[Trajectory],
    player_transition_min_length: int = 32,
    player_history_min_length: int = 64,
    rng_key: jax.Array = None,
) -> Batch:
    """Stacks a list of trajectories and pads them to a fixed resolution."""
    stacked_trajectory: Trajectory = jax.tree.map(
        lambda *xs: np.stack(xs, axis=1), *batch
    )

    max_valid = (
        stacked_trajectory.player_transitions.env_output.done.argmax(axis=0)
        .max()
        .item()
    )

    num_valid = num_valid = geometric_bucket(
        max_valid,
        player_transition_min_length,
        stacked_trajectory.player_transitions.env_output.done.shape[0],
    )

    return Batch(
        builder_transitions=stacked_trajectory.builder_transitions,
        builder_history=stacked_trajectory.builder_history,
        player_transitions=jax.tree.map(
            lambda x: x[:num_valid], stacked_trajectory.player_transitions
        ),
        player_packed_history=clip_packed_history(
            stacked_trajectory.player_packed_history,
            min_length=player_history_min_length,
        ),
        player_history=clip_history(
            stacked_trajectory.player_history, min_length=player_history_min_length
        ),
        reuse_count=(
            ()
            if isinstance(stacked_trajectory.reuse_count, tuple)
            else stacked_trajectory.reuse_count
        ),
        rng_key=rng_key,
    )


def _embedding_stats(emb: jax.Array, valid: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Representation-health stats over one batch of trunk embeddings.

    Returns the dormant-unit fraction (ReDo criterion: units whose mean
    |activation| over valid steps is ≤ 0.025× the layer mean) and the
    srank@0.99 fraction (smallest number of singular values holding 99% of
    the spectrum mass, over the feature dim). emb is (T, B, ..., d), valid
    is (T, B); padded rows are zeroed, which leaves the Gram spectrum
    unchanged versus dropping them.
    """
    d = emb.shape[-1]
    lead = valid.reshape(valid.shape + (1,) * (emb.ndim - valid.ndim - 1))
    mask = jnp.broadcast_to(lead, emb.shape[:-1]).reshape(-1).astype(jnp.float32)
    flat = emb.reshape(-1, d).astype(jnp.float32) * mask[:, None]
    denom = mask.sum() + 1e-8

    unit_score = jnp.abs(flat).sum(axis=0) / denom
    dormant_frac = (unit_score <= 0.025 * unit_score.mean()).mean()

    gram = flat.T @ flat / denom
    singular_values = jnp.sqrt(jnp.maximum(jnp.linalg.eigvalsh(gram), 0.0))
    singular_values = jnp.sort(singular_values)[::-1]
    srank = (jnp.cumsum(singular_values) < 0.99 * singular_values.sum()).sum() + 1
    return dormant_frac, srank / d


class Learner:
    def __init__(
        self,
        player_state: Porygon2PlayerTrainState,
        builder_state: Porygon2BuilderTrainState,
        config: Porygon2LearnerConfig,
        wandb_run: wandb.wandb_run.Run,
        league: League,
        gpu_lock: LockType | None = None,
        player_network=None,
        debug: bool = False,
        controller_bytes: bytes | None = None,
    ):
        self.player_state = player_state
        self.builder_state = builder_state
        self.config = config
        self.wandb_run = wandb_run
        self.league = league
        self.gpu_lock = gpu_lock or nullcontext()

        self.plasticity = PlasticityController(
            enabled=config.plasticity_enabled,
            overdue_trigger=config.plasticity_overdue_trigger,
            recovery_winrate=config.plasticity_recovery_winrate,
            cooldown_frames=config.plasticity_cooldown_frames,
        )

        # Advantage-lambda and magnet-coef runtime scalars (traced into
        # train_step — never static config: each static value's recompile
        # retained ~5GB host RAM and OOM-killed 1326). Driven by the
        # controllers below, or by the bandit if it is enabled instead.
        self._current_lambda = float(config.player_adv_lambda)
        self._current_magnet_coef = float(config.player_magnet_kl_coef)

        self.lambda_ctrl: LambdaGapController | None = None
        if config.lambda_ctrl_enabled:
            self.lambda_ctrl = LambdaGapController(
                initial_lambda=config.player_adv_lambda,
                gap_target=config.lambda_ctrl_gap_target,
                kp=config.lambda_ctrl_kp,
                ki=config.lambda_ctrl_ki,
                interval=config.lambda_ctrl_interval,
                lambda_min=config.lambda_ctrl_min,
                lambda_max=config.lambda_ctrl_max,
                sensor_ema=config.lambda_ctrl_sensor_ema,
            )

        self.entropy_ctrl: AdaptivityController | None = None
        if config.entropy_ctrl_enabled:
            self.entropy_ctrl = AdaptivityController(
                baseline_coef=config.player_magnet_kl_coef,
                commit_target=config.adapt_ctrl_commit_target,
                kp=config.adapt_ctrl_kp,
                ki=config.adapt_ctrl_ki,
                interval=config.adapt_ctrl_interval,
                max_scale=config.entropy_ctrl_max_scale,
                min_scale=config.entropy_ctrl_min_scale,
                sensor_ema=config.adapt_ctrl_sensor_ema,
                action_floor=config.entropy_ctrl_floor,
                modality_floor=config.entropy_ctrl_modality_floor,
                floor_gain=config.adapt_ctrl_floor_gain,
                event_bump=config.adapt_ctrl_event_bump,
            )

        # Strength-grounded bandit over adv-lambda arms — kept as the
        # audit tool, off by default in favour of the gap controller.
        self.bandit: LambdaBandit | None = None
        if config.bandit_enabled and len(config.bandit_lambdas) > 1:
            if config.lambda_ctrl_enabled:
                raise ValueError(
                    "bandit_enabled and lambda_ctrl_enabled both set: two "
                    "drivers for the advantage lambda — enable exactly one."
                )
            self.bandit = LambdaBandit(
                arms=config.bandit_lambdas,
                default_arm=config.bandit_default_arm,
                ucb_c=config.bandit_ucb_c,
                discount=config.bandit_discount,
                min_games_per_opponent=config.bandit_min_games_per_opponent,
                min_rated_opponents=config.bandit_min_rated_opponents,
            )
            if league.bandit_state is not None:
                self.bandit.restore(league.bandit_state)
            self._current_lambda = float(self.bandit.arms[self.bandit.current_arm])

        self.done = False
        self.builder_replay = BuilderTrajectoryStore(
            max_size=self.config.builder_replay_buffer_capacity,
            max_reuses=self.config.builder_replay_ratio,
        )

        is_not_randoms = self.config.smogon_format != "randombattle"
        self.player_replay = PlayerTrajectoryStore(
            max_size=self.config.player_replay_buffer_capacity,
            max_reuses=self.config.player_replay_ratio,
            need_tracking=is_not_randoms,
        )

        # Plasticity probe: jitted encoder-only forward measuring trunk
        # representation health (dormant units, spectral rank) on the
        # current train batch every plasticity_probe_interval steps.
        # Requires the network module, which only main.py holds — probe is
        # silently disabled when it isn't passed in.
        self._plasticity_probe_jit = None
        if player_network is not None and config.plasticity_probe_interval > 0:
            self._plasticity_probe_jit = self._make_plasticity_probe(player_network)

        # Replay-ratio PI controller state (see config for the design).
        # Owned by the wandb log worker thread, which already device-syncs
        # every step's logs — the controller reads the actor KL there and
        # drives player_replay.set_max_reuses, so the train loop never pays
        # for it. Velocity form on log(cap): only the previous error and the
        # clamped control value are state.
        self._replay_ctrl_log_cap = float(np.log(self.config.player_replay_ratio))
        self._replay_ctrl_prev_err = 0.0
        self._replay_ctrl_kl_sum = 0.0
        self._replay_ctrl_kl_count = 0
        # Store-counter snapshots for the realised replay ratio
        # (Δsamples/Δinserts per tick) — what the gate actually lets
        # through, versus the cap, which is only what it permits.
        self._replay_ctrl_prev_adds = 0
        self._replay_ctrl_prev_samples = 0
        self._replay_realised_ratio = float("nan")

        # Threading
        self.device_q: queue.Queue[Batch] = queue.Queue(maxsize=1)
        # Log dicts still hold device arrays when enqueued; the log worker
        # pays the device sync so the train loop never blocks on the GPU.
        # Bounded so a stalled wandb client applies backpressure instead of
        # accumulating unbounded device references.
        self._log_q: queue.Queue[dict | None] = queue.Queue(maxsize=64)

        # Progress Bars
        self.consumer_progress = tqdm(desc="consumer", smoothing=0.1)
        self.train_progress = tqdm(desc="batches", smoothing=0.1)

        # JIT Compile
        if debug:
            self._train_step_jit = train_step
        else:
            # Donation requires that no pytree leaf appears twice across the
            # donated states (params/target_params are deep-copied at
            # creation/restore) and that nothing reads a state object after
            # the call is dispatched — all periodic readers run on this
            # thread after the rebind in _train_step.
            self._train_step_jit = jax.jit(
                train_step,
                static_argnames=["config"],
                donate_argnames=["player_state", "builder_state"],
            )

        # Last: the controllers and the plasticity controller must already
        # exist before their checkpointed state is applied.
        self.restore_controller_state(controller_bytes)

    def _make_plasticity_probe(self, network):
        """Builds the jitted plasticity probe: an encoder-only forward on
        the current batch, returning dormant-unit fraction and srank@0.99
        for both trunk embedding streams. These measure plasticity loss
        directly (dead units, representation rank collapse), so a plateau
        can be attributed — or not — to plasticity before shrink-and-
        perturb fires on league stagnation alone."""

        def encoder_only(module, actor_input: PlayerActorInput):
            return module.encoder(
                actor_input.env, actor_input.packed_history, actor_input.history
            )

        encode = jax.vmap(
            lambda params, actor_input: network.apply(
                params, actor_input, method=encoder_only
            ),
            in_axes=(None, 1),
            out_axes=1,
        )

        def probe(params, batch: Batch) -> dict[str, jax.Array]:
            actor_input = PlayerActorInput(
                env=batch.player_transitions.env_output,
                packed_history=batch.player_packed_history,
                history=batch.player_history,
            )
            action_emb, value_emb = encode(params, actor_input)
            dones = batch.player_transitions.env_output.done
            valid = (jnp.cumsum(dones, axis=0) - dones) == 0
            logs = {}
            for name, emb in (("action", action_emb), ("value", value_emb)):
                dormant_frac, srank_frac = _embedding_stats(emb, valid)
                logs[f"plasticity_{name}_emb_dormant_frac"] = dormant_frac
                logs[f"plasticity_{name}_emb_srank_frac"] = srank_frac
            return logs

        return jax.jit(probe)

    def enqueue_traj(self, traj: Trajectory):
        """Called by actors to push data."""
        add_cond = self.player_replay._add_cv
        with add_cond:
            add_cond.wait_for(lambda: self.done or self.player_replay.ready_to_add())
            if self.done:
                return
            self.player_replay.add(traj)

        sample_cond = self.player_replay._sample_cv
        with sample_cond:
            sample_cond.notify_all()

    def host_to_device_worker(self):
        """Background thread to batch data and push to GPU queue."""
        max_burst = 8
        minibatch_size = self.config.batch_size
        batch_size = minibatch_size * self.config.gradient_accumulation_steps

        # Wait until replay buffer is at least replay_buffer_min_fill_fraction full before starting training
        sample_cond = self.player_replay._sample_cv
        with sample_cond:
            sample_cond.wait_for(
                lambda: self.done
                or self.player_replay.is_min_fill_fraction_reached(
                    self.config.replay_buffer_min_fill_fraction
                )
            )

        init_key = jax.random.PRNGKey(random.randint(0, 2**16 - 1))
        while not self.done:
            # Burst processing to minimize lock contention overhead
            for _ in range(max_burst):
                if self.done:
                    break

                sample_cond = self.player_replay._sample_cv
                with sample_cond:
                    sample_cond.wait_for(
                        lambda: self.done
                        or self.player_replay.ready_to_sample(batch_size)
                    )
                    if self.done:
                        break
                    batch = self.player_replay.sample(minibatch_size)

                add_cond = self.player_replay._add_cv
                with add_cond:
                    add_cond.notify_all()

                self.consumer_progress.update(minibatch_size)

                # Process pure data outside lock
                init_key, batch_key = jax.random.split(init_key)
                stacked = _stack_and_pad_batch(batch, rng_key=batch_key)
                # Bounded put that re-checks done: an unbounded put can strand
                # this thread forever if shutdown drains the queue between our
                # done-check and the put.
                while not self.done:
                    try:
                        self.device_q.put(stacked, timeout=1.0)
                        break
                    except queue.Full:
                        continue

        logger.info("host_to_device_worker exiting.")

    def _wandb_log_worker(self):
        """Background thread: drains log dicts, paying the device->host
        transfer and wandb serialization here so the train loop never has to
        synchronize with the GPU per step. A single consumer preserves wandb's
        step ordering. Also hosts the replay-ratio controller, which needs
        exactly the host-side per-step logs this thread already produces."""
        while True:
            logs = self._log_q.get()
            if logs is None:
                break
            try:
                host_logs = jax.device_get(logs)
                self._update_replay_controller(host_logs)
                self._update_hyper_controllers(host_logs)
                self.wandb_run.log(host_logs)
            except Exception:
                logger.exception("wandb logging failed")

    def controller_state_bytes(self) -> bytes:
        """Host-side training dynamics for the checkpoint: controllers and
        plasticity bookkeeping. Not parameters, but resuming without them
        silently resets an in-flight plasticity recovery and re-anneals
        lambda from scratch."""
        state = {"plasticity": self.plasticity.state_dict()}
        if self.lambda_ctrl is not None:
            state["lambda_ctrl"] = self.lambda_ctrl.state_dict()
        if self.entropy_ctrl is not None:
            state["entropy_ctrl"] = self.entropy_ctrl.state_dict()
        return pickle.dumps(state)

    def restore_controller_state(self, data: bytes | None) -> None:
        """Counterpart to controller_state_bytes. Missing sections (older
        checkpoints, or a controller disabled at save time) leave the
        corresponding controller freshly initialised.

        Never fatal: this state only saves a controller some re-warmup,
        so a blob written by a superseded controller revision must not
        be able to fail a resume. Run 1333 died at startup on exactly
        that (KeyError 'cov_ema' restoring EntropyRateController state
        into AdaptivityController) — each section is now isolated and a
        bad one is logged and skipped."""
        if not data:
            return
        try:
            state = pickle.loads(data)
        except Exception:
            logger.exception("controller state unreadable — starting fresh")
            return

        def _restore(name: str, apply):
            section = state.get(name)
            if section is None:
                return
            try:
                apply(section)
            except Exception:
                logger.exception(
                    "controller state for %r incompatible — that controller "
                    "starts fresh",
                    name,
                )

        _restore("plasticity", self.plasticity.load_state_dict)
        if self.lambda_ctrl is not None:
            _restore("lambda_ctrl", self.lambda_ctrl.load_state_dict)
            self._current_lambda = self.lambda_ctrl.value
        if self.entropy_ctrl is not None:
            _restore("entropy_ctrl", self.entropy_ctrl.load_state_dict)
            self._current_magnet_coef = self.entropy_ctrl.value

    def _update_hyper_controllers(self, host_logs: dict) -> None:
        """Lambda and entropy controllers (rl/online/controllers.py).
        Runs on the log worker like the replay controller — the sensors
        are host-side per-step logs, and the actuators are plain floats
        the train thread reads on its next step (GIL-atomic swap)."""
        if self.lambda_ctrl is not None:
            gap = host_logs.get("player_bootstrap_gap")
            host_logs.update(
                self.lambda_ctrl.update(
                    None if gap is None else float(gap),
                    recovering=self.plasticity.recovering,
                )
            )
            self._current_lambda = self.lambda_ctrl.value
        if self.entropy_ctrl is not None:
            # Commitment covariance drives the loop: it falls when the
            # policy's confident choices stop paying — exactly what a new
            # league opponent or a perturbation causes — and recovers as
            # the policy re-adapts. The two entropy axes are passed as
            # hard backstops inside the controller, because the
            # covariance is blind to actions the policy never takes
            # (which is how 1330 lost switching while looking healthy).
            cov = host_logs.get("player_commit_cov")
            action_ent = host_logs.get("player_action_normalized_entropy")
            modal_ent = host_logs.get("player_normalized_modality_entropy")
            host_logs.update(
                self.entropy_ctrl.update(
                    None if cov is None else float(cov),
                    None if action_ent is None else float(action_ent),
                    None if modal_ent is None else float(modal_ent),
                )
            )
            self._current_magnet_coef = self.entropy_ctrl.value

    def _update_replay_controller(self, host_logs: dict) -> None:
        """Velocity-form PI loop holding the replayed-batch actor KL at
        player_replay_kl_target by adjusting the store's reuse cap.

        The KL is averaged over player_replay_ctrl_interval steps per tick
        (the per-batch measurement is noisy; the window is the smoother).
        Working on log(cap) makes the control multiplicative, and clamping
        log(cap) itself gives anti-windup for free: the integral action
        cannot accumulate past the bounds. Adds the current cap to
        host_logs so every wandb step carries it."""
        config = self.config
        if not config.player_replay_ctrl_enabled:
            return
        kl = host_logs.get("player_learner_actor_forward_kl")
        if kl is not None and np.isfinite(kl):
            self._replay_ctrl_kl_sum += float(kl)
            self._replay_ctrl_kl_count += 1

        if self._replay_ctrl_kl_count >= config.player_replay_ctrl_interval:
            kl_mean = self._replay_ctrl_kl_sum / self._replay_ctrl_kl_count
            self._replay_ctrl_kl_sum = 0.0
            self._replay_ctrl_kl_count = 0

            err = (config.player_replay_kl_target - kl_mean) / (
                config.player_replay_kl_target
            )
            self._replay_ctrl_log_cap += (
                config.player_replay_ctrl_kp * (err - self._replay_ctrl_prev_err)
                + config.player_replay_ctrl_ki * err
            )
            self._replay_ctrl_prev_err = err
            self._replay_ctrl_log_cap = float(
                np.clip(
                    self._replay_ctrl_log_cap,
                    np.log(config.player_replay_ratio_min),
                    np.log(config.player_replay_ratio_max),
                )
            )
            cap = int(round(np.exp(self._replay_ctrl_log_cap)))
            if cap != self.player_replay.max_reuses:
                self.player_replay.set_max_reuses(cap)

            adds = self.player_replay.total_adds
            samples = self.player_replay.total_samples
            delta_adds = adds - self._replay_ctrl_prev_adds
            delta_samples = samples - self._replay_ctrl_prev_samples
            self._replay_ctrl_prev_adds = adds
            self._replay_ctrl_prev_samples = samples
            if delta_adds > 0:
                self._replay_realised_ratio = delta_samples / delta_adds

        host_logs["player_replay_max_reuses"] = float(self.player_replay.max_reuses)
        host_logs["player_replay_realised_ratio"] = self._replay_realised_ratio

    def train(self):
        """
        High-level training loop.
        Delegates computation to _execute_model_update and I/O to _handle_periodic_tasks.
        """
        transfer_thread = threading.Thread(
            target=self.host_to_device_worker, daemon=True
        )
        transfer_thread.start()
        log_thread = threading.Thread(target=self._wandb_log_worker, daemon=True)
        log_thread.start()

        # Host-side mirror of player_state.step_count: train_step increments
        # the device counter by exactly one per call, so tracking it here
        # keeps periodic-task scheduling free of per-step device syncs.
        host_step = int(jax.device_get(self.player_state.step_count))

        try:
            for _ in range(self.config.num_steps):

                # 1. Fetch Data (Blocking)
                batch = self.device_q.get()
                with self.gpu_lock:
                    batch = jax.device_put(batch)
                    # 2. Update Model
                    logs = self._train_step(batch)

                    if logs is None:
                        continue  # Skip this step if update failed

                host_step += 1
                # Representation-health probe on the batch just trained on.
                # batch is not donated by the train step, so reusing it here
                # is safe; results are device arrays synced by the log
                # worker like every other metric.
                if (
                    self._plasticity_probe_jit is not None
                    and host_step % self.config.plasticity_probe_interval == 0
                ):
                    with self.gpu_lock:
                        logs.update(
                            self._plasticity_probe_jit(self.player_state.params, batch)
                        )
                self._handle_periodic_tasks(host_step, logs)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Saving checkpoint...")
            try:
                save_train_state(
                    self.wandb_run,
                    self.config,
                    jax.device_get(self.player_state),
                    jax.device_get(self.builder_state),
                    self.league,
                    self.controller_state_bytes(),
                )
            except RuntimeError:
                # An interrupt that lands mid train-step catches the state
                # after its buffers were donated but before the rebind; the
                # pre-step state is unrecoverable then, so skip the save
                # rather than masking the interrupt with a donation error.
                logger.exception(
                    "Skipping interrupt checkpoint: train state was donated "
                    "mid-step. Latest periodic checkpoint is unaffected."
                )
            raise
        except Exception as e:
            logger.error(f"Learner training crashed: {e}")
            traceback.print_exc()
            raise e
        finally:
            self.done = True
            # device_q has maxsize=1; drain the single pending item (if any) to
            # unblock host_to_device_worker if it is blocked on put().
            try:
                self.device_q.get_nowait()
            except queue.Empty:
                pass

            for cond in [
                self.player_replay._add_cv,
                self.player_replay._sample_cv,
                self.builder_replay._add_cv,
                self.builder_replay._sample_cv,
            ]:
                with cond:
                    cond.notify_all()

            transfer_thread.join(timeout=10)

            # Sentinel stops the log worker after it drains pending logs. The
            # worker is a daemon, so a wedged wandb client can't hang exit.
            self._log_q.put(None)
            log_thread.join(timeout=30)
            print("Training Finished.")

    def _train_step(self, batch: Batch) -> dict | None:
        """
        Runs the JAX update, verifies gradients/loss, and updates internal state.
        Returns training logs on success, or None if the step was invalid (e.g. NaN loss).
        """
        # 1. Run JAX Step (thread-safe)
        self.player_state, self.builder_state, logs = self._train_step_jit(
            self.player_state,
            self.builder_state,
            batch,
            self.config,
            np.float32(self._current_lambda),
            np.float32(self._current_magnet_coef),
        )

        return logs

    def _handle_periodic_tasks(self, step: int, logs: dict):
        """Handles logging, progress bars, and checkpointing."""

        # Console Progress
        self.train_progress.update(1)

        if (
            self.config.smogon_format != "randombattle"
            and step % self.config.save_interval_steps == 0
        ):
            logs.update(self._get_usage_counts())

        if step % self.config.league_winrate_log_steps == 0:
            logs.update(self._get_league_winrates())
            logs.update(self.plasticity.logs())

        # Rating window boundary. Bandit enabled: reward the live arm
        # with the BT-rating gain vs the frozen pool and point the
        # runtime lambda at the next arm. Bandit disabled (default —
        # the lambda controller drives instead): still fit and log the
        # BT rating, the only strength-grounded progress telemetry.
        if step % self.config.bandit_window_steps == 0:
            if self.bandit is not None:
                logs.update(self.bandit.update(self.league))
                self._current_lambda = float(self.bandit.arms[self.bandit.current_arm])
                self.league.bandit_state = self.bandit.serialize()
            else:
                logs.update(
                    rating_logs(
                        self.league,
                        self.config.bandit_min_games_per_opponent,
                        self.config.bandit_min_rated_opponents,
                    )
                )

        # Hand off to the log worker; values may still be device arrays and
        # are synced there, off the critical path.
        self._log_q.put(logs)

        # Main Player Update & Checkpoint
        if step % self.config.main_player_update_steps == 0:
            self._update_main_player_in_league()

        if step % self.config.save_interval_steps == 0:
            save_train_state(
                self.wandb_run,
                self.config,
                jax.device_get(self.player_state),
                jax.device_get(self.builder_state),
                self.league,
                self.controller_state_bytes(),
            )

        if step % self.config.manage_league_interval == 0:
            self._manage_league(step)

    def _manage_league(self, step: int):
        """Checks if a new player should be added to the league."""
        reason = self._should_add_new_player()
        if reason is not None:
            print(f"Adding new player to league @ {step} ({reason})")
            self._add_player_to_league(step)
            self.player_replay.reset_usage_counts()
            self.plasticity.on_player_added(reason)
            # A new opponent shifts the state distribution before any
            # batch reflects it: raise diversity pressure now so the
            # policy can adapt instead of defending habitual lines. The
            # commitment covariance decays it back once the new lines
            # start paying.
            if self.entropy_ctrl is not None:
                self.entropy_ctrl.bump()

        self._update_plasticity(step)

    def _should_add_new_player(self) -> AddReason | None:
        """Returns why a snapshot should join the league, or None to skip.

        The reason doubles as a bias-free stagnation signal: "dominant" adds
        mean the main player keeps outgrowing its history, while consecutive
        "overdue" adds mean it has stopped making progress against itself.
        """
        latest = self.league.get_latest_player()
        current = self.league.get_main_player()

        # Calculate frames passed since the last added player
        latest_frames = latest.player_frame_count if latest is not None else 0
        frames_passed = int(current.player_frame_count - latest_frames)

        # Basic gate: minimum frames
        if frames_passed < self.config.add_player_min_frames:
            return None

        historical_players = [
            v for k, v in self.league.players.items() if k != MAIN_KEY
        ]

        # Initial population check
        if not historical_players:
            if (
                int(self.player_state.step_count)
                > self.config.minimum_historical_player_steps
            ):
                return "initial"
            return None

        # Winrate check
        win_rates = self.league.get_winrate((current, historical_players))

        if win_rates.min() > 0.7:
            return "dominant"
        if frames_passed >= self.config.add_player_max_frames:
            return "overdue"
        return None

    def _update_plasticity(self, step: int):
        """Tracks recovery from the last perturbation and fires new ones."""
        frame_count = int(jax.device_get(self.player_state.frame_count))

        if self.plasticity.recovering:
            ref = self.league.players.get(self.plasticity.recovery_ref_step)
            if ref is None:
                # Reference snapshot left the league; nothing to measure
                # recovery against, so unblock the controller.
                self.plasticity.check_recovery(1.0, frame_count)
            else:
                main = self.league.get_main_player()
                winrate = float(self.league.get_winrate((main, ref)).item())
                self.plasticity.check_recovery(winrate, frame_count)

        if self.plasticity.should_perturb(frame_count):
            self._apply_plasticity_update(step, frame_count)

    def _apply_plasticity_update(self, step: int, frame_count: int):
        """Shrink-and-perturb the player params to restore plasticity.

        The pre-perturbation self must be in the league first — both to keep
        its strength as a training signal and to serve as the recovery
        benchmark. The trigger path usually just added it (the overdue add);
        if not, snapshot now.
        """
        latest = self.league.get_latest_player()
        if latest is None or latest.step_count != step:
            self._add_player_to_league(step)
            latest = self.league.get_latest_player()

        print(
            f"Applying shrink-and-perturb plasticity update @ {step} "
            f"(recovery ref: player {latest.step_count})"
        )
        rng = jax.random.PRNGKey(random.randint(0, 2**16 - 1))
        self.player_state = shrink_and_perturb_player_state(
            self.player_state,
            rng,
            default_shrink=self.config.plasticity_default_shrink,
            module_shrink=self.config.plasticity_module_shrink,
        )
        self.plasticity.on_perturbation(latest.step_count, frame_count)
        # Perturbation scrambles the policy's preferences: hold its
        # options open (a larger bump than a league addition) while it
        # re-forms them. Mirrors the lambda controller being pinned at
        # its ceiling for the same reason.
        if self.entropy_ctrl is not None:
            self.entropy_ctrl.bump(self.config.adapt_ctrl_perturb_bump)

    def _update_main_player_in_league(self):
        self.league.update_main_player(self._create_params_container(MAIN_KEY))

    def _create_params_container(self, step_key):
        return ParamsContainer(
            player_frame_count=jax.device_get(self.player_state.frame_count).item(),
            builder_frame_count=jax.device_get(self.builder_state.frame_count).item(),
            step_count=step_key,
            player_params=jax.device_get(self.player_state.params),
            builder_params=jax.device_get(self.builder_state.params),
        )

    def _add_player_to_league(self, step: int):
        """Persist the current params as an opponent snapshot and register a ref.

        Only the params files are written (no optimiser state); the league holds
        the lightweight ref and materialises the params lazily when this player
        is actually drawn as an opponent.
        """
        snapshot_dir = os.path.abspath(
            f"./ckpts/gen{self.config.generation}/players/p_{step:08}"
        )
        checkpoint.save_param_snapshot(
            snapshot_dir,
            player_components=dict(
                params=jax.device_get(self.player_state.params),
                target_params=jax.device_get(self.player_state.target_params),
            ),
            builder_components=dict(
                params=jax.device_get(self.builder_state.params),
                target_params=jax.device_get(self.builder_state.target_params),
            ),
        )
        self.league.add_player(
            PlayerRef(
                step_count=step,
                snapshot_dir=snapshot_dir,
                player_frame_count=jax.device_get(self.player_state.frame_count).item(),
                builder_frame_count=jax.device_get(
                    self.builder_state.frame_count
                ).item(),
                player_key="params",
                builder_key="params",
            )
        )

    def _get_usage_counts(self):
        result = {}

        for key, counts in [
            ("species", self.player_replay._species_counts),
            ("items", self.player_replay._item_counts),
            ("abilities", self.player_replay._ability_counts),
            ("moves", self.player_replay._move_counts),
        ]:
            names = list(STOI[key])
            table = wandb.Table(columns=[key, "usage"])
            for name, count in zip(names, counts):
                table.add_data(name, count)
            result[f"{key}_usage"] = table

        return result

    def _get_league_winrates(self):
        current = self.league.get_main_player()
        others = [v for k, v in self.league.players.items() if k != MAIN_KEY]

        if not others:
            return {}

        win_rates = self.league.get_winrate((current, others))
        return {
            f"league_main_v_{others[i].step_count}_winrate": wr
            for i, wr in enumerate(win_rates)
        }
