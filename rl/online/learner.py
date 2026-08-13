import dataclasses
import logging
import os
import pickle
import queue
import random
import threading
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

# Optional: renders the league win-rate heatmap (_get_league_winrate_heatmap).
# Guarded rather than a hard top-level dependency — this codebase has already
# hit real deployment-sync friction this session (files/config not landing on
# the training box before a restart); crashing the ENTIRE training process
# over a missing plotting library for a supplementary visualization would be
# a disproportionate failure mode. Add matplotlib to requirements.txt and
# `pip install` it on the training box to actually get the heatmap panel.
try:
    import matplotlib

    matplotlib.use("Agg")  # headless — no display server on a training box
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
from rl.environment.data import CAT_VF_SUPPORT, STOI, PackedSetFeature
from rl.environment.interfaces import (
    Batch,
    BuilderActorInput,
    PlayerActorInput,
    Trajectory,
)
from rl.environment.utils import (
    _bucket_level,
    _bucket_value,
    _history_level,
    _packed_history_level,
    clip_history,
    clip_packed_history,
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
from rl.online.bandit import rating_logs
from rl.online.buffer import BuilderTrajectoryStore, PlayerTrajectoryStore
from rl.online.config import Porygon2LearnerConfig
from rl.online.controllers import PILogController
from rl.online.league import (
    LEAGUE_EXPLOITER_KEY,
    LIVE_KEYS,
    MAIN_EXPLOITER_KEY,
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

PopulationName = Literal["main", "main_exploiter", "league_exploiter"]
POPULATION_NAMES: tuple[PopulationName, ...] = (
    "main",
    "main_exploiter",
    "league_exploiter",
)
_LIVE_KEY_BY_POPULATION: dict[PopulationName, int] = {
    "main": MAIN_KEY,
    "main_exploiter": MAIN_EXPLOITER_KEY,
    "league_exploiter": LEAGUE_EXPLOITER_KEY,
}
# Block-sequential rotation order: each time main finishes a training
# window (a routine league addition in _manage_league), the NEXT exploiter
# population in this cycle gets the GPU for one full attempt — trained to
# its own terminal outcome (promotion or frame-budget timeout), exactly
# AlphaStar's ready_to_checkpoint shape — before control returns to main.
# See _select_population for the scheduling model.
_EXPLOITER_ROTATION: tuple[PopulationName, ...] = (
    "main_exploiter",
    "league_exploiter",
)
_FRAME_BUDGET_FIELD = {
    "main_exploiter": "main_exploiter_frame_budget",
    "league_exploiter": "league_exploiter_frame_budget",
}
# Per-population step-count namespace (docs/exploiter-phase-plan.md's
# three-population redesign): each population runs its OWN step counter
# from its own creation/reset point (not main's absolute counter), and all
# three are now permanently visible in the SAME shared League (no more
# restart-merge boundary hiding an exploiter's own step counter from
# main's) — so all three need disjoint PlayerRef.step_count ranges, not
# just promotions vs. main like the old 2-way offset. Sized far above any
# plausible single-population step count (~200k-step target, see
# porygon2-1m-step-target memory).
_STEP_OFFSET = {
    "main": 0,
    "main_exploiter": 100_000_000,
    "league_exploiter": 200_000_000,
}


class ExploiterBudgetExhausted(Exception):
    """No longer used to unwind a phase (there are no phases) — kept only
    as a marker exception type in case external tooling still references
    it. rl.online.learner.Learner now handles an exploiter's non-promotion
    timeout in-process (see _check_exploiter_transitions) by resetting
    that population directly, not by raising."""


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
    upgo_coef: jax.Array | None = None,
    magnet_coef: jax.Array | None = None,
):
    """Train for a single step.

    ``upgo_coef`` and ``magnet_coef`` are RUNTIME scalars (traced, not
    static — runtime values never recompile; static-config scalars
    retained ~5GB of executables per distinct value and OOM-killed run
    1326). upgo_coef is config.player_upgo_coef, zeroed by the caller
    during plasticity recovery (a freshly-perturbed critic cuts UPGO
    returns in the wrong places).
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
        adv_std_divisor = jnp.maximum(
            player_state.ema_adv_std, config.player_adv_std_floor
        )
        player_advantages = (
            player_targets.advantages - player_state.ema_adv_mean
        ) / adv_std_divisor
        # UPGO shares the std divisor (both channels live on the same +-1
        # value scale) but is NOT mean-recentered: its positive skew —
        # extra credit along better-than-expected lines — is the
        # mechanism, not a normalisation artefact to remove.
        player_upgo_advantages = player_targets.upgo_advantages / adv_std_divisor
    else:
        player_advantages = player_targets.advantages
        player_upgo_advantages = player_targets.upgo_advantages

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
        # UPGO PG term (AlphaStar: v-trace loss + UPGO loss, summed) —
        # same surrogate/clipping, the outcome-conditional advantage
        # channel. upgo_coef is the runtime scalar (0 during plasticity
        # recovery).
        loss_upgo = policy_gradient_loss(
            policy_ratios=learner_actor_ratio * actor_target_clipped_ratio,
            advantages=player_upgo_advantages,
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

        # (player_commit_cov removed 2026-08-14 at the user's request: as
        # a correlation with the advantages the policy itself generated,
        # it largely measured what it was correlated with. With the
        # adaptivity controller also gone, modality collapse has NO
        # automated backstop — watch player_normalized_modality_entropy
        # on the dashboard; 1330 died at 0.08 on that axis.)

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
            + (config.player_upgo_coef if upgo_coef is None else upgo_coef) * loss_upgo
            + config.player_value_head_loss_coef * loss_v_win
            + config.player_kl_loss_coef * loss_actor_backward_kl
            + (config.player_magnet_kl_coef if magnet_coef is None else magnet_coef)
            * loss_magnet_kl
            + config.player_aux_value_coef * loss_v_aux
        )

        return loss, dict(
            # Loss values
            player_loss_pg=loss_pg,
            player_loss_upgo=loss_upgo,
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


# Module-level, not per-Learner: exactly ONE Learner exists for the life of
# the process now (three populations sharing it, not one per phase — see
# docs/exploiter-phase-plan.md's three-population redesign), so this is
# already only ever compiled once. Kept module-level anyway (rather than
# built in __init__) since it's still shared across all three populations'
# player_state/builder_state (identical architecture, identical shapes) —
# the scheduler just swaps in whichever PopulationState's state is due to
# train this tick.
_TRAIN_STEP_JIT = jax.jit(
    train_step,
    static_argnames=["config"],
    donate_argnames=["player_state", "builder_state"],
)


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

    # One shared bucket level across player_transitions/history/packed_
    # history instead of three independently-computed geometric_bucket()
    # calls: all three lengths are different views of the same fact (how
    # long this batch's games ran), so bucketing them independently lets
    # XLA see the PRODUCT of each axis's distinct values as new shapes for
    # _TRAIN_STEP_JIT, not just the sum — main's wider opponent/game-length
    # diversity under the three-population redesign turned this from a
    # latent inefficiency into frequent, ongoing recompilation (each
    # permanently retained in JAX's compile cache) that both throttled
    # main's step throughput and was a real contributor to the OOM-guard
    # trip after ~1hr on 2026-08-12. Taking the max of each axis's own
    # required level (rather than letting one field's level drive the
    # others) means no field is ever truncated below what it individually
    # needed — this only ever costs extra padding, never data loss.
    level = max(
        _bucket_level(max_valid, player_transition_min_length),
        _history_level(stacked_trajectory.player_history, player_history_min_length),
        _packed_history_level(
            stacked_trajectory.player_packed_history, player_history_min_length
        ),
    )

    num_valid = _bucket_value(
        level,
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
            level=level,
        ),
        player_history=clip_history(
            stacked_trajectory.player_history,
            min_length=player_history_min_length,
            level=level,
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


def _check_promotion_bar(
    league: League,
    home_key: int,
    candidate_origin_filter: "Callable[[PlayerRef], bool]",
    promote_winrate: float,
    promote_min_games: float,
) -> str | None:
    """Returns a human-readable failure reason, or None if the bar is
    cleared.

    Moved from the former rl/online/promote_exploiter.py (deleted — its
    CLI/manual-promotion path depended entirely on the old per-run
    disposable-checkpoint model this redesign removes; see docs/
    exploiter-phase-plan.md's 2026-08-12 note). home_key generalizes what
    used to be a hardcoded MAIN_KEY: under the three-population design each
    exploiter population has its OWN live identity, and the bar is checked
    against ITS win-rate, not main's.

    Checks win-rate vs. EVERY reliably-measured (>= promote_min_games
    effective games) candidate in the population's own targeting pool
    (candidate_origin_filter — lineage-restricted for main_exploiter,
    unrestricted for league_exploiter, matching get_match()'s own filter)
    — not a small fixed pinned set individually, since neither exploiter
    population pins to one anymore (see player_actor.py). Same
    aggregate-win-rate-over-a-population shape as main's own
    _should_add_new_player "dominant" gate and _measure_exploitability:
    the worst reliably-measured win-rate must still clear the bar, so a
    strategy that crushes most of the pool and loses to one member hasn't
    actually generalized.
    """
    candidates = [
        ref for ref in league.players.values() if candidate_origin_filter(ref)
    ]
    rateable = [
        ref
        for ref in candidates
        if league.games.get((home_key, ref.step_count), 0.0)
        + league.games.get((ref.step_count, home_key), 0.0)
        >= promote_min_games
    ]
    if not rateable:
        return "no reliably-measured opponents yet"
    win_rates = [
        league._win_rate_by_steps(home_key, ref.step_count) for ref in rateable
    ]
    worst_idx = int(np.argmin(win_rates))
    worst_winrate = win_rates[worst_idx]
    if worst_winrate < promote_winrate:
        return (
            f"vs {rateable[worst_idx].step_count}: win-rate {worst_winrate:.3f} "
            f"< {promote_winrate} (worst of {len(rateable)} reliably-measured "
            "opponents)"
        )
    return None


@dataclasses.dataclass
class PopulationState:
    """One live, continuously-training population's mutable state
    (docs/exploiter-phase-plan.md's three-population redesign: MainPlayer,
    MainExploiter, LeagueExploiter). ``main`` always exists from process
    start; the two exploiter populations start with player_state=None and
    are created lazily by Learner._maybe_create_population once there's
    something worth exploiting.

    Deliberately NOT shared across populations: player_state/builder_state
    (genuinely independent search — separate optimizer momentum from
    main's own trajectory), replay buffers (the user's explicit
    requirement; cheap host RAM), and every controller/plasticity object
    (each is pure-Python EMA/PI state over ONE lineage's own training
    signal — sharing would corrupt every EMA the moment two populations'
    gradients diverge).
    """

    name: PopulationName
    live_key: int
    wandb_run: wandb.wandb_run.Run
    player_replay: PlayerTrajectoryStore
    builder_replay: BuilderTrajectoryStore
    plasticity: PlasticityController
    player_state: Porygon2PlayerTrainState | None = None
    builder_state: Porygon2BuilderTrainState | None = None
    created_at_frame: int | None = None
    # For an exploiter population: the main step_count it was forked from
    # (recorded at creation/reset time — see Learner._fork_population).
    fork_step: int | None = None
    host_step: int = 0
    device_q: "queue.Queue" = None
    log_q: "queue.Queue" = None
    ckpt_q: "queue.Queue" = None
    # This population's own 3 internal background workers (host_to_device/
    # log/checkpoint) — owned and joined entirely within this file.
    worker_threads: list = dataclasses.field(default_factory=list)
    # This population's PlayerActor/BuilderActor game-playing threads —
    # constructed and started by main.py (Learner can't import
    # player_actor.py/builder_actor.py without a circular import, since
    # both already import Learner), registered here via
    # Learner.register_actor_threads so a shutdown or reset waits for them
    # too, not just the 3 internal workers.
    actor_threads: list = dataclasses.field(default_factory=list)
    stop_signal: list = dataclasses.field(default_factory=lambda: [False])
    done: bool = False
    # Block-sequential actor gating (Learner._set_active): set = this
    # population's actor threads may play games; cleared = they idle
    # between games (threading.Event wait, threads stay alive — no
    # create/destroy churn). Only the population whose block it is has
    # its gate set, so the full actor budget serves whoever is training.
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
    # Cumulative frames trained AS this population, since its own creation/
    # last reset. Telemetry under the block-sequential scheduler (nothing
    # schedules on it anymore); kept because "how much has this population
    # actually trained" is the first question when judging an attempt.
    frames_trained_total: int = 0
    # frame_count at this population's last terminal outcome that did NOT
    # rebuild it (league_exploiter's continue/perturb fates in
    # _apply_terminal_outcome) — the dwell and frame-budget checks in
    # _check_exploiter_transitions measure from here, so a continued
    # attempt gets a full fresh window instead of instantly re-tripping
    # the timeout on its inherited counter.
    budget_anchor_frames: int = 0
    consumer_progress: object = None
    train_progress: object = None
    plasticity_probe_jit: object = None

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
    """Owns exactly one shared League, one gpu_lock, and one compiled
    train_step — and three PopulationState bundles (main always live; the
    two exploiter populations created lazily). See docs/exploiter-phase-
    plan.md's 2026-08-12 redesign note for the full design rationale."""

    def __init__(
        self,
        config: Porygon2LearnerConfig,
        league: League,
        player_state: Porygon2PlayerTrainState,
        builder_state: Porygon2BuilderTrainState,
        main_wandb_run: wandb.wandb_run.Run,
        main_exploiter_wandb_run: wandb.wandb_run.Run,
        league_exploiter_wandb_run: wandb.wandb_run.Run,
        gpu_lock: LockType | None = None,
        player_network=None,
        debug: bool = False,
        controller_bytes: bytes | None = None,
        spawn_actor_pool: "Callable[[PopulationName], None] | None" = None,
    ):
        self.config = config
        self.league = league
        self.gpu_lock = gpu_lock or nullcontext()
        self.debug = debug
        # Fires the instant a population is created/reset, so main.py's
        # orchestration (which owns PlayerActor/BuilderActor construction —
        # Learner can't import those without a circular import) can spin up
        # that population's actor pool. None is fine for standalone/direct
        # construction (tests, debug scripts): that population just never
        # gets actors, matching today's "nothing passed in means don't
        # wire it up" convention elsewhere in this file.
        self._spawn_actor_pool = spawn_actor_pool

        if not _MATPLOTLIB_AVAILABLE:
            logger.warning(
                "matplotlib not installed — league_winrate_heatmap will not "
                "be logged. `pip install matplotlib` to enable it."
            )

        # train_step's config is a static jit arg. Used to need pinning to
        # a canonical value because the OLD per-phase design constructed a
        # genuinely distinct config.replace(pin_opponent_steps=...) for
        # every exploiter phase, invalidating the jit cache each time (the
        # 1326 failure mode). Under this redesign there is exactly ONE
        # Learner, constructed once, holding ONE config value for the life
        # of the process — nothing ever varies it again, so there's nothing
        # left to protect the jit cache key against. Kept as a separate
        # attribute name (not literally self.config) only so a future
        # per-population config override, if one is ever needed, has an
        # obvious place to plug in without re-deriving this reasoning.
        self._jit_config = config

        self._plasticity_probe_jit = None
        if player_network is not None and config.plasticity_probe_interval > 0:
            self._plasticity_probe_jit = self._make_plasticity_probe(player_network)

        self._train_step_jit = train_step if debug else _TRAIN_STEP_JIT

        # Block-sequential scheduler state (see _select_population): whose
        # block it is right now, and which exploiter population is next in
        # the rotation when main finishes its current window. Always starts
        # at main — an exploiter attempt's state is deliberately not
        # resumable across a restart (see _write_checkpoint), so a restart
        # always re-opens with a main window. Initialised BEFORE the
        # populations dict: _build_population reads self._active to decide
        # whether the new population's actor run_gate starts open.
        self._active: PopulationName = "main"
        self._rotation_idx: int = 0

        self.populations: dict[PopulationName, PopulationState] = {}
        self.populations["main"] = self._build_population(
            "main",
            player_state,
            builder_state,
            main_wandb_run,
            controller_bytes=controller_bytes,
        )
        # Held for lazy construction later — see _maybe_create_population.
        self._pending_wandb_runs: dict[PopulationName, wandb.wandb_run.Run] = {
            "main_exploiter": main_exploiter_wandb_run,
            "league_exploiter": league_exploiter_wandb_run,
        }
        for name in ("main_exploiter", "league_exploiter"):
            wandb_run = self._pending_wandb_runs[name]
            wandb_run.log({"population_created": 0})

        self.done = False

    # --- population construction / lifecycle --------------------------------

    def _build_population(
        self,
        name: PopulationName,
        player_state: Porygon2PlayerTrainState,
        builder_state: Porygon2BuilderTrainState,
        wandb_run: wandb.wandb_run.Run,
        controller_bytes: bytes | None = None,
        fork_step: int | None = None,
    ) -> PopulationState:
        """Builds a fresh PopulationState around an already-constructed
        player_state/builder_state. Controllers/plasticity/replay are
        always fresh here, never shared with or copied from another
        population — restore_controller_state (below) only ever runs for
        cold-start main; a forked exploiter population never receives
        controller_bytes at all (see Learner._fork_population): nothing
        to un-inherit if nothing was ever inherited."""
        config = self.config
        plasticity = PlasticityController(
            enabled=config.plasticity_enabled,
            overdue_trigger=config.plasticity_overdue_trigger,
            recovery_winrate=config.plasticity_recovery_winrate,
            cooldown_frames=config.plasticity_cooldown_frames,
            defer_to_exploiter=(
                config.plasticity_defer_to_exploiter or config.auto_exploiter_enabled
            ),
        )
        is_not_randoms = config.smogon_format != "randombattle"
        pop = PopulationState(
            name=name,
            live_key=_LIVE_KEY_BY_POPULATION[name],
            wandb_run=wandb_run,
            player_replay=PlayerTrajectoryStore(
                max_size=config.player_replay_buffer_capacity,
                max_reuses=config.player_replay_ratio,
                need_tracking=is_not_randoms,
                name=name,
            ),
            builder_replay=BuilderTrajectoryStore(
                max_size=config.builder_replay_buffer_capacity,
                max_reuses=config.builder_replay_ratio,
                name=name,
            ),
            plasticity=plasticity,
            player_state=player_state,
            builder_state=builder_state,
            created_at_frame=int(jax.device_get(player_state.frame_count)),
            fork_step=fork_step,
            replay_pi=PILogController(
                initial_log=float(np.log(config.player_replay_ratio)),
                log_min=float(np.log(config.player_replay_ratio_min)),
                log_max=float(np.log(config.player_replay_ratio_max)),
                kp=config.player_replay_ctrl_kp,
                ki=config.player_replay_ctrl_ki,
            ),
            replay_kl_target=float(config.player_replay_kl_target),
            consumer_progress=tqdm(
                desc=f"consumer-{name}", smoothing=0.1, position=next_tqdm_position()
            ),
            train_progress=tqdm(
                desc=f"batches-{name}", smoothing=0.1, position=next_tqdm_position()
            ),
            plasticity_probe_jit=self._plasticity_probe_jit,
        )
        # Actor gate starts open only for the population whose block it
        # currently is (main at cold start; the just-activated exploiter
        # when _reset_population runs mid-block-transition) — everyone
        # else's actors idle at the gate until _set_active opens it.
        if name == self._active:
            pop.run_gate.set()
        self._restore_controller_state(pop, controller_bytes)
        self.league.update_live(pop.live_key, self._create_params_container(pop))
        return pop

    def _fork_population(self, name: PopulationName) -> tuple:
        """Host round-trip copy of main's CURRENT live params into fresh,
        independent player_state/builder_state — fresh optimizer state
        (never copied from main: this is what makes the search genuinely
        independent, not just biased matchmaking on main's own weights —
        see docs/exploiter-phase-plan.md's rejected-alternatives section),
        step_count/frame_count reset to 0 (this population's OWN age since
        this fork/reset, not main's absolute counter — _STEP_OFFSET is
        exactly what keeps that from colliding with main's or the other
        exploiter population's own from-zero counters in the shared
        League)."""
        main = self.populations["main"]
        host_player = jax.device_get(main.player_state)
        host_builder = jax.device_get(main.builder_state)

        player_params = jax.tree.map(lambda x: np.copy(x), host_player.params)
        player_target_params = jax.tree.map(
            lambda x: np.copy(x), host_player.target_params
        )
        builder_params = jax.tree.map(lambda x: np.copy(x), host_builder.params)
        builder_target_params = jax.tree.map(
            lambda x: np.copy(x), host_builder.target_params
        )

        zero_step = np.zeros_like(host_player.step_count)
        zero_frame = np.zeros_like(host_player.frame_count)
        # .replace() must be called on the HOST copies (host_player/
        # host_builder), not main.player_state/main.builder_state: any
        # field not explicitly overridden below is inherited as-is from
        # whatever .replace() is called on. Calling it on the live device
        # objects silently aliased ema_adv_mean/ema_adv_std (and base
        # TrainState.step) to main's own live, continuously-donated
        # buffers instead of copying them — main's very next train step
        # would donate (free) that shared buffer, and the forked
        # population's own next train step would then try to donate that
        # same already-freed buffer, raising XLA's "Donation requested
        # for invalid buffer".
        player_state = host_player.replace(
            params=player_params,
            target_params=player_target_params,
            opt_state=main.player_state.tx.init(player_params),
            step_count=zero_step,
            frame_count=zero_frame,
        )
        builder_state = host_builder.replace(
            params=builder_params,
            target_params=builder_target_params,
            opt_state=main.builder_state.tx.init(builder_params),
            step_count=np.zeros_like(host_builder.step_count),
            frame_count=np.zeros_like(host_builder.frame_count),
        )
        return jax.device_put(player_state), jax.device_put(builder_state)

    def _maybe_create_population(self, name: PopulationName) -> None:
        """Need-driven creation (docs/exploiter-phase-plan.md): a population
        is born the moment it's empty AND its target pool is non-empty —
        main_exploiter needs >=1 origin=="main" historical snapshot,
        league_exploiter needs >=1 snapshot of any origin. Called from
        _begin_exploiter_block (main-window boundaries), decoupled from
        which population the block actually activates."""
        pop = self.populations.get(name)
        if pop is not None:
            return
        candidates = [
            ref
            for ref in self.league.players.values()
            if name != "main_exploiter" or ref.origin == "main"
        ]
        if not candidates:
            return
        self._reset_population(name, reason="created")

    def _reset_population(self, name: PopulationName, reason: str) -> None:
        """Creates (if new) or resets (if terminal outcome fired — see
        _check_exploiter_transitions) an exploiter population: fresh fork
        from main's current live params, fresh controllers/plasticity/
        replay, fresh actor pool. Never called for "main"."""
        assert name != "main"
        existing = self.populations.get(name)
        if existing is not None:
            # Same teardown as a full process shutdown (_stop_population_
            # workers), scoped to this one population: a population reset
            # is exactly a scaled-down version of what used to be a whole
            # phase boundary — an actor/worker thread that outlives it
            # keeps this population's about-to-be-discarded state
            # reachable, which is why this reuses the identical
            # signal-then-join-then-fail-loudly-on-stragglers logic rather
            # than a separate, easier-to-drift-out-of-sync copy of it.
            self._stop_population_workers(existing)

        player_state, builder_state = self._fork_population(name)
        fork_step = int(
            jax.device_get(self.populations["main"].player_state.step_count)
        )
        wandb_run = (
            self._pending_wandb_runs[name] if existing is None else existing.wandb_run
        )
        pop = self._build_population(
            name, player_state, builder_state, wandb_run, fork_step=fork_step
        )
        self.populations[name] = pop
        self._start_population_workers(pop)
        pop.wandb_run.log(
            {"population_created": 1, "fork_step": fork_step},
            commit=False,
        )
        if self._spawn_actor_pool is not None:
            self._spawn_actor_pool(name)

    # --- plasticity probe / trajectory intake --------------------------------

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

    def enqueue_traj(self, population: PopulationName, traj: Trajectory):
        """Called by actors to push data into their own population's
        replay buffer."""
        pop = self.populations[population]
        add_cond = pop.player_replay._add_cv
        with add_cond:
            add_cond.wait_for(lambda: pop.done or pop.player_replay.ready_to_add())
            if pop.done:
                return
            pop.player_replay.add(traj)

        sample_cond = pop.player_replay._sample_cv
        with sample_cond:
            sample_cond.notify_all()

    # --- per-population background workers -----------------------------------

    def host_to_device_worker(self, pop: PopulationState):
        """Background thread to batch data and push to this population's
        own GPU queue."""
        max_burst = 8
        minibatch_size = self.config.batch_size
        batch_size = minibatch_size * self.config.gradient_accumulation_steps

        sample_cond = pop.player_replay._sample_cv
        with sample_cond:
            sample_cond.wait_for(
                lambda: pop.done
                or pop.player_replay.is_min_fill_fraction_reached(
                    self.config.replay_buffer_min_fill_fraction
                )
            )

        init_key = jax.random.PRNGKey(random.randint(0, 2**16 - 1))
        while not pop.done:
            for _ in range(max_burst):
                if pop.done:
                    break

                sample_cond = pop.player_replay._sample_cv
                with sample_cond:
                    sample_cond.wait_for(
                        lambda: pop.done
                        or pop.player_replay.ready_to_sample(batch_size)
                    )
                    if pop.done:
                        break
                    batch = pop.player_replay.sample(minibatch_size)

                add_cond = pop.player_replay._add_cv
                with add_cond:
                    add_cond.notify_all()

                pop.consumer_progress.update(minibatch_size)

                init_key, batch_key = jax.random.split(init_key)
                stacked = _stack_and_pad_batch(batch, rng_key=batch_key)
                while not pop.done:
                    try:
                        pop.device_q.put(stacked, timeout=1.0)
                        break
                    except queue.Full:
                        continue

        logger.info("host_to_device_worker[%s] exiting.", pop.name)

    def _wandb_log_worker(self, pop: PopulationState):
        """Background thread: drains log dicts for this population,
        paying the device->host transfer and wandb serialization here so
        the train loop never has to synchronize with the GPU per step. A
        single consumer per population preserves that population's own
        wandb step ordering. Also hosts the replay-ratio controller, which
        needs exactly the host-side per-step logs this thread already
        produces."""
        while True:
            logs = pop.log_q.get()
            if logs is None:
                break
            try:
                host_logs = jax.device_get(logs)
                self._update_replay_controller(pop, host_logs)
                pop.wandb_run.log(host_logs)
            except Exception:
                logger.exception("wandb logging failed for population %s", pop.name)

    def _checkpoint_writer_worker(self, pop: PopulationState):
        """Background thread: does the actual checkpoint disk I/O for
        this population (and, every cloud_save_interval_steps, the wandb
        artifact upload for main only) so the training loop never blocks
        on it. Payloads are already fully host-side and pre-serialized by
        the time they're queued (see _handle_periodic_tasks) — this thread
        never touches a live device buffer or mutates self.league
        directly, only writes what it was handed."""
        while True:
            payload = pop.ckpt_q.get()
            if payload is None:
                break
            try:
                save_path = write_checkpoint_components(
                    payload["save_path"],
                    payload["learner_config"],
                    payload["player_components"],
                    payload["builder_components"],
                    payload["league_bytes"],
                    payload["controller_bytes"],
                    step_count=payload["step_count"],
                    frame_count=payload["frame_count"],
                )
                if payload["upload_to_cloud"]:
                    pop.wandb_run.log_artifact(
                        artifact_or_path=save_path,
                        name=f"latest-gen{payload['learner_config'].generation}",
                        type="model",
                    )
            except Exception:
                logger.exception(
                    "Background checkpoint write failed for population %s @ "
                    "step %s — the next periodic checkpoint will simply try "
                    "again.",
                    pop.name,
                    payload.get("step_count"),
                )

    # --- controller state (main only ever restores from a real checkpoint) --

    def controller_state_bytes(self, pop: PopulationState) -> bytes:
        """Host-side training dynamics for the checkpoint — just the
        plasticity bookkeeping now (the adaptive hyperparameter
        controllers were all removed 2026-08-13/-14). Not parameters, but
        resuming without it silently forgets an in-flight plasticity
        recovery, clearing the cooldown that stops a second
        shrink-and-perturb from hitting a convalescing net."""
        state = {"plasticity": pop.plasticity.state_dict()}
        return pickle.dumps(state)

    def _restore_controller_state(
        self, pop: PopulationState, data: bytes | None
    ) -> None:
        """Counterpart to controller_state_bytes. Missing sections (older
        checkpoints, or a controller disabled at save time) leave the
        corresponding controller freshly initialised. Only ever called
        with real data for "main" at cold start — a forked exploiter
        population is always built with controller_bytes=None (see
        _build_population's docstring), so this is a no-op for them.

        Never fatal: this state only saves a controller some re-warmup,
        so a blob written by a superseded controller revision must not be
        able to fail a resume."""
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

        _restore("plasticity", pop.plasticity.load_state_dict)
        # Checkpoints written before the controller removals (entropy_ctrl
        # 2026-08-13; lambda_ctrl/exploit_ctrl 2026-08-14) carry those
        # sections — simply never read, same as any other extra section.

    # No _update_hyper_controllers anymore: the magnet KL coef became a
    # fixed config scalar when the AdaptivityController was removed
    # (2026-08-13), and the advantage lambda went with the
    # LambdaGapController (2026-08-14) — UPGO's per-step cut plus the
    # fixed player_lambda replaced it (see targets.py). The replay
    # reuse-cap controller below is the one remaining per-log-tick loop.

    def _update_replay_controller(self, pop: PopulationState, host_logs: dict) -> None:
        """Velocity-form PI loop holding the replayed-batch actor KL at
        pop.replay_kl_target by adjusting that population's own reuse
        cap."""
        config = self.config
        if not config.player_replay_ctrl_enabled:
            return
        kl = host_logs.get("player_learner_actor_forward_kl")
        if kl is not None and np.isfinite(kl):
            pop.replay_ctrl_kl_sum += float(kl)
            pop.replay_ctrl_kl_count += 1

        if pop.replay_ctrl_kl_count >= config.player_replay_ctrl_interval:
            kl_mean = pop.replay_ctrl_kl_sum / pop.replay_ctrl_kl_count
            pop.replay_ctrl_kl_sum = 0.0
            pop.replay_ctrl_kl_count = 0

            err = (pop.replay_kl_target - kl_mean) / pop.replay_kl_target
            pop.replay_pi.step(err)

            cap = int(round(np.exp(pop.replay_pi.log)))
            if cap != pop.player_replay.max_reuses:
                pop.player_replay.set_max_reuses(cap)

            adds = pop.player_replay.total_adds
            samples = pop.player_replay.total_samples
            delta_adds = adds - pop.replay_ctrl_prev_adds
            delta_samples = samples - pop.replay_ctrl_prev_samples
            pop.replay_ctrl_prev_adds = adds
            pop.replay_ctrl_prev_samples = samples
            if delta_adds > 0:
                pop.replay_realised_ratio = delta_samples / delta_adds

        host_logs["player_replay_max_reuses"] = float(pop.player_replay.max_reuses)
        host_logs["player_replay_realised_ratio"] = pop.replay_realised_ratio

    # --- unified scheduler ---------------------------------------------------

    def _select_population(self) -> PopulationState | None:
        """Block-sequential pick (replacing the old duty-cycle fraction
        scheduler): exactly one population "owns" the GPU at a time
        (self._active). Main owns it by default and trains until a routine
        league addition closes its window (_manage_league →
        _begin_exploiter_block); an exploiter population then owns it for
        one FULL attempt — until its own promotion or frame-budget timeout
        fires (_check_exploiter_transitions → _end_exploiter_block) —
        before main's next window opens. No frame-share fractions: each
        population trains to its own AlphaStar-style terminal condition,
        sequentialising the concurrent league on one GPU instead of
        simulating concurrency by interleaving.

        Main-as-filler: when the active exploiter has no batch ready this
        tick, main trains instead of letting the GPU idle. With actor
        gating (_set_active) every population's pool is main-sized and
        only the active one plays, so filler mostly happens in the
        warm-up stretch right after a block switch (a fresh fork starts
        with an empty replay buffer) and in transient production dips —
        not as a steady tax. During an exploiter block main is therefore
        "mostly frozen", not strictly frozen; note main's own gated-off
        actors aren't refilling its buffer then, so sustained filler
        self-limits at the replay reuse cap (the KL-bounded controller),
        after which the GPU is exclusively the exploiter's.

        The .empty() peek is race-free for our purposes: this (the train
        loop) is the sole consumer of every device_q, so an observed
        non-empty queue can't be emptied by anyone else before train()
        collects it."""

        def pickable(name: PopulationName) -> PopulationState | None:
            pop = self.populations.get(name)
            if (
                pop is not None
                and pop.player_state is not None
                and pop.player_replay.is_min_fill_fraction_reached(
                    self.config.replay_buffer_min_fill_fraction
                )
                and not pop.device_q.empty()
            ):
                return pop
            return None

        active = pickable(self._active)
        if active is not None:
            return active
        if self._active != "main":
            return pickable("main")
        return None

    def train(self):
        """Unified training loop across every live population — no more
        one loop per phase. Each tick: pick a population
        (_select_population), pull one batch from ITS device_q, train via
        the shared compiled train_step under the shared gpu_lock, run its
        own periodic tasks. Actor pools for every live population run
        continuously and independently of whose turn it is to
        gradient-update (see main.py) — this is what makes turn-switching
        free of the fork/recompile cost the old per-phase design paid.
        """
        for pop in self.populations.values():
            self._start_population_workers(pop)

        try:
            for _ in range(self.config.num_steps):
                if self.done:
                    break
                pop = self._select_population()
                if pop is None:
                    # Nothing has a warm-enough replay buffer yet (e.g. at
                    # process start, before main's own buffer fills), or
                    # every ready population's device_q is momentarily
                    # empty — brief wait rather than a busy spin.
                    threading.Event().wait(timeout=0.1)
                    continue

                try:
                    # Never blocks: _select_population only returns a
                    # population it observed a batch for, and this thread
                    # is the sole consumer of every device_q.
                    batch = pop.device_q.get_nowait()
                except queue.Empty:
                    continue
                with self.gpu_lock:
                    batch = jax.device_put(batch)
                    logs = self._train_step(pop, batch)
                    if logs is None:
                        continue

                pop.host_step += 1
                pop.frames_trained_total = (
                    int(jax.device_get(pop.player_state.frame_count))
                    - pop.created_at_frame
                )
                if (
                    pop.plasticity_probe_jit is not None
                    and pop.host_step % self.config.plasticity_probe_interval == 0
                ):
                    with self.gpu_lock:
                        logs.update(
                            pop.plasticity_probe_jit(pop.player_state.params, batch)
                        )
                self._handle_periodic_tasks(pop, pop.host_step, logs)

        except KeyboardInterrupt:
            # Only main persists across a restart — see _write_checkpoint's
            # docstring: an in-progress, not-yet-terminal exploiter
            # population's live state is exactly as disposable here as it
            # already was in the old per-phase design (a non-promoted
            # attempt's state was always scratch, deleted outright); a
            # restart just re-forks it fresh from main's own restored
            # state the next time _maybe_create_population's trigger
            # fires. Only main needs a real resumable checkpoint.
            logger.info("Keyboard interrupt received. Saving main's checkpoint...")
            main_pop = self.populations["main"]
            for pop in [main_pop]:
                try:
                    self._write_checkpoint(pop, synchronous=True)
                except RuntimeError:
                    logger.exception(
                        "Skipping interrupt checkpoint for population %s: "
                        "train state was donated mid-step. Latest periodic "
                        "checkpoint is unaffected.",
                        pop.name,
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
            for pop in self.populations.values():
                # strict=False: process is exiting — a straggler here is
                # tolerable (daemon threads die with the process), and
                # raising would mask the real outcome, turning e.g. a
                # clean Ctrl-C into a crash. Resets keep strict=True.
                self._stop_population_workers(pop, strict=False)
            tqdm.write("Training Finished.")

    def register_actor_threads(
        self, population: PopulationName, threads: list[threading.Thread]
    ) -> None:
        """Called by main.py right after it constructs and starts a
        population's PlayerActor/BuilderActor pool (in response to the
        spawn_actor_pool callback, on creation, or after a reset) — Learner
        can't spawn these itself without a circular import. Registering
        them here means a shutdown or population reset waits for (and
        straggler-checks) them exactly like the 3 internal workers,
        instead of silently leaving them running against now-stale state."""
        self.populations[population].actor_threads.extend(threads)

    def _start_population_workers(self, pop: PopulationState) -> None:
        transfer_thread = threading.Thread(
            target=self.host_to_device_worker,
            args=(pop,),
            daemon=True,
            name=f"transfer-{pop.name}",
        )
        transfer_thread.start()
        log_thread = threading.Thread(
            target=self._wandb_log_worker,
            args=(pop,),
            daemon=True,
            name=f"log-{pop.name}",
        )
        log_thread.start()
        ckpt_thread = threading.Thread(
            target=self._checkpoint_writer_worker,
            args=(pop,),
            daemon=True,
            name=f"ckpt-{pop.name}",
        )
        ckpt_thread.start()
        pop.worker_threads.extend([transfer_thread, log_thread, ckpt_thread])

    def _stop_population_workers(
        self, pop: PopulationState, strict: bool = True
    ) -> None:
        pop.done = True
        pop.stop_signal[0] = True
        # Wake actors idling at the block gate so they observe stop_signal
        # immediately instead of on their next wait() timeout.
        pop.run_gate.set()
        try:
            pop.device_q.get_nowait()
        except queue.Empty:
            pass
        for cond in (
            pop.player_replay._add_cv,
            pop.player_replay._sample_cv,
            pop.builder_replay._add_cv,
            pop.builder_replay._sample_cv,
        ):
            with cond:
                cond.notify_all()

        for t in pop.worker_threads:
            if t.name.startswith("transfer-"):
                t.join(timeout=10)
        pop.log_q.put(None)
        for t in pop.worker_threads:
            if t.name.startswith("log-"):
                t.join(timeout=30)
        pop.ckpt_q.put(None)
        for t in pop.worker_threads:
            if t.name.startswith("ckpt-"):
                t.join(timeout=60)
        # External actor threads (main.py's PlayerActor/BuilderActor pool,
        # registered via register_actor_threads): already signalled via
        # pop.stop_signal[0] above — just wait for them here.
        for t in pop.actor_threads:
            t.join(timeout=30)

        all_threads = pop.worker_threads + pop.actor_threads
        stragglers = [t for t in all_threads if t.is_alive()]
        if stragglers:
            logger.warning(
                "%d worker thread(s) for population %s did not stop within "
                "their join timeout: %s — giving a 30s grace period before "
                "treating this as a hung shutdown.",
                len(stragglers),
                pop.name,
                [t.name for t in stragglers],
            )
            for t in stragglers:
                t.join(timeout=30)
            stragglers = [t for t in stragglers if t.is_alive()]
        if stragglers:
            if strict:
                raise RuntimeError(
                    f"{len(stragglers)} worker thread(s) for population "
                    f"{pop.name} never stopped: {[t.name for t in stragglers]}. "
                    "Refusing to proceed with this population's state still "
                    "reachable from a live thread."
                )
            # strict=False is the whole-PROCESS shutdown path (train()'s
            # finally, incl. Ctrl-C): the straggler raise exists to stop a
            # population RESET from rebuilding on top of state a leaked
            # thread still holds (the 2026-08-11 RAM/VRAM leak) — at
            # process exit there is no next phase to protect, every
            # thread is a daemon that dies with the process, and raising
            # here would convert a clean Ctrl-C into a "crashed" outcome
            # (an actor blocked on the game-server websocket mid-game is
            # normal at this point, not a leak).
            logger.warning(
                "%d thread(s) for population %s still alive at process "
                "shutdown: %s — proceeding; they are daemons and exit "
                "with the process.",
                len(stragglers),
                pop.name,
                [t.name for t in stragglers],
            )

        # Return this population's 4 progress-bar rows to the shared pool
        # (close_tqdm_bar) so the replacement fork reuses the same rows —
        # without this, every exploiter reset leaked 4 dead rows and
        # pushed all live bars one screen-row further down, unboundedly,
        # for the life of the process. Closing is safe against any update
        # racing in from a straggler: tqdm's close() flips .disable, which
        # every update() checks first.
        for bar in (
            pop.consumer_progress,
            pop.train_progress,
            pop.player_replay._progress,
            pop.builder_replay._progress,
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
        every memory_diag_interval steps (main-only, same convention as
        _check_oom_guard: process-wide facts don't need three copies).

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

        for name, pop in self.populations.items():
            logs[f"diag_player_replay_mb_{name}"] = pop.player_replay.nbytes() / 2**20
            logs[f"diag_builder_replay_mb_{name}"] = pop.builder_replay.nbytes() / 2**20
        entries, cache_bytes = self.league.cache_stats()
        logs["diag_league_cache_entries"] = entries
        logs["diag_league_cache_mb"] = cache_bytes / 2**20

    def _train_step(self, pop: PopulationState, batch: Batch) -> dict | None:
        """Runs the JAX update for pop via the shared compiled train_step,
        rebinding the result onto pop (not self — every population takes
        turns through the same compiled closure)."""
        # upgo_coef zeroed during plasticity recovery: a freshly-perturbed
        # critic cuts UPGO returns in the wrong places (the same regime
        # the removed lambda controller handled by forcing lambda to its
        # ceiling). Runtime scalar — flipping it never recompiles.
        upgo_coef = 0.0 if pop.plasticity.recovering else self.config.player_upgo_coef
        pop.player_state, pop.builder_state, logs = self._train_step_jit(
            pop.player_state,
            pop.builder_state,
            batch,
            self._jit_config,
            np.float32(upgo_coef),
            np.float32(self.config.player_magnet_kl_coef),
        )
        return logs

    def _handle_periodic_tasks(self, pop: PopulationState, step: int, logs: dict):
        """Handles logging, progress bars, and checkpointing for pop."""
        pop.train_progress.update(1)

        if (
            self.config.smogon_format != "randombattle"
            and step % self.config.save_interval_steps == 0
        ):
            logs.update(self._get_usage_counts(pop))

        if step % self.config.league_winrate_log_steps == 0:
            # Payoff-table views are main-only: ONE dashboard holds the
            # whole league's pairwise structure (rows labelled by origin,
            # live exploiter populations included), instead of three
            # partial copies of the same shared table. The exploiter
            # populations' attempt progress is visible there too — their
            # live rows vs. main and vs. their own targets.
            if pop.name == "main":
                logs.update(self._get_league_winrates(pop))
                logs.update(self._get_league_winrate_heatmap(pop))
            logs.update(pop.plasticity.logs())

        if pop.name == "main" and step % self.config.bandit_window_steps == 0:
            logs.update(
                rating_logs(
                    self.league,
                    self.config.bandit_min_games_per_opponent,
                    self.config.bandit_min_rated_opponents,
                )
            )

        if (
            pop.name == "main"
            and self.config.memory_diag_interval > 0
            and step % self.config.memory_diag_interval == 0
        ):
            self._log_memory_diagnostics(logs)

        pop.log_q.put(logs)

        # Every population's live entry needs refreshing, not just main's —
        # PlayerActor.pull_own_player() reads league.get_live(pop.live_key)
        # for every population's own actors; leaving an exploiter
        # population's entry pinned at its creation-time snapshot would
        # mean its actors train forever against whatever it looked like
        # the instant it was forked, never picking up its own progress.
        if step % self.config.main_player_update_steps == 0:
            self.league.update_live(pop.live_key, self._create_params_container(pop))

        # Only main gets a full, resumable (params+opt_state) periodic
        # checkpoint — see the KeyboardInterrupt handler in train() for
        # why an exploiter population's in-progress state doesn't need
        # one: it's exactly as disposable as it already was pre-redesign.
        if pop.name == "main" and step % self.config.save_interval_steps == 0:
            self._write_checkpoint(pop)

        if pop.name == "main" and step % self.config.manage_league_interval == 0:
            self._manage_league(pop, step)

        if self.config.auto_exploiter_enabled:
            self._check_exploiter_transitions(pop)

        self._check_oom_guard(pop, step)

    def _write_checkpoint(self, pop: PopulationState, synchronous: bool = False) -> str:
        """Only ever called for "main" — a full, resumable (params+
        opt_state) periodic/interrupt/OOM-guard checkpoint only makes
        sense for the population meant to survive a process restart.
        main_exploiter/league_exploiter's in-progress, not-yet-terminal
        state is exactly as disposable as it already was pre-redesign (a
        non-promoted exploiter attempt's state was always scratch,
        deleted outright); a restart just re-forks them fresh from main's
        own restored state the next time _maybe_create_population's
        trigger fires. Their durable output is what _add_player_to_league
        already writes on promotion/timeout — a params-only snapshot,
        exactly like any other league member.

        Everything host-side/fast happens synchronously here (device
        pulls, small in-memory serializations); only the actual disk
        write goes to pop's own background writer, via a payload that's
        already fully host-side (plain dicts/bytes/ints, never a live
        TrainState or the live League object itself). synchronous=True
        (Ctrl-C/OOM-guard path) writes inline instead, since there may be
        no time left for the background writer to run."""
        host_player_state = jax.device_get(pop.player_state)
        host_builder_state = jax.device_get(pop.builder_state)
        player_components = dict(
            params=host_player_state.params,
            target_params=host_player_state.target_params,
            opt_state=host_player_state.opt_state,
            scalars=dict(
                step_count=host_player_state.step_count,
                frame_count=host_player_state.frame_count,
                ema_adv_mean=host_player_state.ema_adv_mean,
                ema_adv_std=host_player_state.ema_adv_std,
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
        # No pop.name component: every call site (OOM-guard, periodic save,
        # KeyboardInterrupt) only ever passes pop=main, and load_train_state
        # (artifact.py's _ckpt_root/_get_checkpoint_path/load_from_checkpoint)
        # expects the checkpoint directly under ckpt_{step:08}/ with no
        # population subdirectory — see _ckpt_root's docstring. Nesting it
        # under pop.name here (always "main" in practice) silently produced
        # a checkpoint the loader could never find, surfacing as
        # FileNotFoundError on the next resume.
        save_path = os.path.abspath(
            os.path.join(
                f"./ckpts/gen{self.config.generation}",
                f"ckpt_{int(np.asarray(host_player_state.step_count)):08}",
            )
        )
        upload_to_cloud = (
            pop.name == "main"
            and self.config.log_artifacts_online
            and int(np.asarray(host_player_state.step_count))
            % self.config.cloud_save_interval_steps
            == 0
        )
        payload = dict(
            save_path=save_path,
            learner_config=self.config,
            player_components=player_components,
            builder_components=builder_components,
            league_bytes=self.league.serialize(),
            controller_bytes=self.controller_state_bytes(pop),
            step_count=int(np.asarray(host_player_state.step_count)),
            frame_count=int(np.asarray(host_player_state.frame_count)),
            upload_to_cloud=upload_to_cloud,
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
        pop.ckpt_q.put(payload)
        return save_path

    def _manage_league(self, pop: PopulationState, step: int):
        """Checks if a new player should be added to the league. main
        only — exploiter populations' own historical additions happen via
        their promotion/timeout path (_check_exploiter_transitions), not
        this routine stagnation-driven one. A routine addition is also
        what closes main's training window under the block-sequential
        scheduler: the next exploiter in rotation takes the GPU."""
        reason = self._should_add_new_player(pop)
        if reason is not None:
            tqdm.write(f"Adding new player to league @ {step} ({reason})")
            self._add_player_to_league(pop, step, origin="main")
            pop.player_replay.reset_usage_counts()
            pop.plasticity.on_player_added(reason)
            self._begin_exploiter_block()

        self._update_plasticity(pop, step)

    def _set_active(self, name: PopulationName) -> None:
        """Makes `name` the block owner: scheduler preference
        (_select_population) AND actor gating — only the active
        population's actor threads play games (run_gate; checked between
        games in main.py's actor loops), so the full actor budget serves
        whoever is training. Every population's pool is main-sized now;
        gating, not pool sizing, is what divides actor resources."""
        self._active = name
        for pop in self.populations.values():
            if pop.name == name:
                pop.run_gate.set()
            else:
                pop.run_gate.clear()

    def _begin_exploiter_block(self) -> None:
        """Hands the GPU to the next exploiter population in
        _EXPLOITER_ROTATION for one full attempt. Called when main closes
        a training window (a routine league addition — including one hit
        while main is only filling an exploiter's production gaps, hence
        the _active guard: a block never pre-empts a running block).

        Population creation is need-driven here: BOTH exploiter
        populations are created the moment their target pool is non-empty
        (not just the one being activated) so their wandb runs/workers
        exist ahead of their first block; their actors stay gated
        (run_gate) until their own block starts. A rotation slot whose
        population still can't exist (empty target pool) is skipped."""
        if not self.config.auto_exploiter_enabled or self._active != "main":
            return
        for name in _EXPLOITER_ROTATION:
            self._maybe_create_population(name)
        for _ in range(len(_EXPLOITER_ROTATION)):
            name = _EXPLOITER_ROTATION[self._rotation_idx]
            self._rotation_idx = (self._rotation_idx + 1) % len(_EXPLOITER_ROTATION)
            if self.populations.get(name) is not None:
                self._set_active(name)
                tqdm.write(f"Exploiter block begins: {name} owns the GPU.")
                return

    def _check_exploiter_transitions(self, pop: PopulationState) -> None:
        """Terminal-outcome check for the ACTIVE exploiter population, on
        its own train ticks only (pop.host_step is its own counter, so
        the auto_exploiter_check_interval gate advances exactly when this
        population actually trains). Either terminal outcome — promotion
        or frame-budget timeout, AlphaStar's ready_to_checkpoint shape —
        ends the population's block and re-opens main's next window;
        population creation lives in _begin_exploiter_block now, not
        here."""
        name = pop.name
        if name == "main" or name != self._active:
            return
        if pop.host_step % self.config.auto_exploiter_check_interval != 0:
            return

        frame_count = int(jax.device_get(pop.player_state.frame_count))
        # Both windows measure from the last non-rebuilding terminal
        # outcome (0 for a fresh fork), not from fork — a continued
        # league_exploiter would otherwise time out again instantly.
        frames_this_attempt = frame_count - pop.budget_anchor_frames
        if frames_this_attempt < self.config.exploiter_min_dwell_frames:
            return

        failure = _check_promotion_bar(
            self.league,
            pop.live_key,
            (
                (lambda ref: ref.origin == "main")
                if name == "main_exploiter"
                else (lambda ref: True)
            ),
            self.config.exploiter_promote_winrate,
            self.config.exploiter_promote_min_games,
        )
        frame_budget = getattr(self.config, _FRAME_BUDGET_FIELD[name])
        timed_out = frames_this_attempt >= frame_budget

        if failure is None:
            self._add_player_to_league(pop, pop.host_step, origin=name)
            logger.info(
                "Population %s promoted @ own-step %d (fork of main @ " "%s).",
                name,
                pop.host_step,
                pop.fork_step,
            )
            self._apply_terminal_outcome(pop, name, frame_count, "promoted")
        elif timed_out:
            # AlphaStar's checkpoint() publishes a Historical snapshot
            # on EITHER outcome — even a non-promoted attempt is a
            # legitimate, permanent sparring partner for future PFSP
            # matchmaking, not wasted exploration.
            self._add_player_to_league(pop, pop.host_step, origin=name)
            logger.info(
                "Population %s timed out without promotion @ own-step "
                "%d (%s) — added as historical.",
                name,
                pop.host_step,
                failure,
            )
            self._apply_terminal_outcome(pop, name, frame_count, "timeout")
        else:
            return

        self._set_active("main")
        tqdm.write(f"Exploiter block ends: {name} -> main's next window.")

    def _apply_terminal_outcome(
        self,
        pop: PopulationState,
        name: PopulationName,
        frame_count: int,
        reason: str,
    ) -> None:
        """Decides an exploiter population's fate AFTER its snapshot was
        published on a terminal outcome — wires config.main_exploiter_
        reset_to_main and config.exploiter_hard_reset_prob, which
        documented exactly this policy but were never read (both
        populations unconditionally re-forked from main).

        main_exploiter: fresh fork of live main — AlphaStar's fixed rule
        for the role (reset_to_main=False degrades to the continue path).

        league_exploiter, following AlphaStar's LeagueExploiter.
        checkpoint() (25%: reset to original init / otherwise: keep
        training): rolls exploiter_hard_reset_prob for a shrink-and-
        perturb toward a fresh init draw — this codebase's stand-in for
        "original init", since no supervised params exist — and otherwise
        continues training its current weights un-reset. The continue
        path is the one that matters strategically: with every reset
        target being a fork of main, a league exploiter that survives
        checkpoints is the league's only source of opponents that keep
        drifting off main's own policy manifold.

        Both non-rebuilding fates re-anchor budget_anchor_frames so the
        next attempt gets a full dwell/timeout window."""
        if name == "main_exploiter" and self.config.main_exploiter_reset_to_main:
            self._reset_population(name, reason=reason)
            return

        if name == "league_exploiter" and (
            random.random() < self.config.exploiter_hard_reset_prob
        ):
            rng = jax.random.PRNGKey(random.randint(0, 2**16 - 1))
            pop.player_state = shrink_and_perturb_player_state(
                pop.player_state,
                rng,
                default_shrink=self.config.plasticity_default_shrink,
                module_shrink=self.config.plasticity_module_shrink,
            )
            # Actors pull params via the league's live entry — push the
            # perturbed weights now rather than waiting for the next
            # periodic update_live.
            self.league.update_live(pop.live_key, self._create_params_container(pop))
            pop.budget_anchor_frames = frame_count
            pop.wandb_run.log({"population_perturbed": 1}, commit=False)
            logger.info(
                "Population %s (%s): hard reset rolled (p=%.2f) — "
                "shrink-and-perturbed toward fresh init.",
                name,
                reason,
                self.config.exploiter_hard_reset_prob,
            )
            return

        pop.budget_anchor_frames = frame_count
        pop.wandb_run.log({"population_continued": 1}, commit=False)
        logger.info("Population %s (%s): continues training un-reset.", name, reason)

    @staticmethod
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

    def _check_oom_guard(self, pop: PopulationState, step: int) -> None:
        """Self-monitoring safety valve, not a leak fix: if available RAM
        drops below config.oom_guard_min_available_fraction, save main's
        checkpoint now and raise OOMGuardTriggered — better to stop on our
        own terms with a guaranteed-complete checkpoint than let the
        kernel's OOM killer pick an arbitrary moment (possibly mid-write)
        to SIGKILL this process. Only main is saved — see _write_checkpoint's
        docstring for why an exploiter population's in-progress state
        doesn't need a resumable checkpoint at all. Available RAM is a
        process-wide fact, not per-population, so this only actually runs
        once per check interval regardless of which population's tick it's
        attached to (whichever happens to hit the interval boundary first
        checks; that's fine, the reading barely changes tick to tick)."""
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
                "%.3f @ population %s step %d — saving main's checkpoint and "
                "stopping before the kernel OOM-kills this process.",
                available_fraction,
                self.config.oom_guard_min_available_fraction,
                pop.name,
                step,
            )
            save_path = self._write_checkpoint(
                self.populations["main"], synchronous=True
            )
            raise OOMGuardTriggered(save_path)

    # (_measure_exploitability/_update_exploit_controller/_apply_exploit_
    # scale removed 2026-08-14 with the ExploitabilityController — the
    # worst-matchup win-rate signal still exists in _should_add_new_player's
    # "dominant" gate and the league_main_winrate_min auditor; it just
    # doesn't actuate anything anymore.)

    def _should_add_new_player(self, pop: PopulationState) -> AddReason | None:
        """Returns why a snapshot should join the league, or None to skip.
        main only."""
        # Pacing is measured against main's OWN last checkpoint (AlphaStar
        # MainPlayer.ready_to_checkpoint: steps since self._checkpoint_step),
        # not the league's newest entry — an exploiter timeout-publication
        # would otherwise become "latest" permanently (its offset key wins
        # max()) with a frame count that never advances, firing an overdue
        # add on every league-management tick.
        latest = self.league.get_latest_player(origin="main")
        current = self.league.get_live(pop.live_key)

        latest_frames = latest.player_frame_count if latest is not None else 0
        frames_passed = int(current.player_frame_count - latest_frames)

        if frames_passed < self.config.add_player_min_frames:
            return None

        historical_players = [
            v for k, v in self.league.players.items() if k not in LIVE_KEYS
        ]

        if not historical_players:
            if (
                int(pop.player_state.step_count)
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

    def _update_plasticity(self, pop: PopulationState, step: int):
        """Tracks recovery from the last perturbation and fires new ones.
        main only."""
        frame_count = int(jax.device_get(pop.player_state.frame_count))

        if pop.plasticity.recovering:
            ref = self.league.players.get(pop.plasticity.recovery_ref_step)
            if ref is None:
                pop.plasticity.check_recovery(1.0, frame_count)
            else:
                current = self.league.get_live(pop.live_key)
                winrate = float(self.league.get_winrate((current, ref)).item())
                pop.plasticity.check_recovery(winrate, frame_count)

        if pop.plasticity.should_perturb(frame_count):
            self._apply_plasticity_update(pop, step, frame_count)

    def _apply_plasticity_update(
        self, pop: PopulationState, step: int, frame_count: int
    ):
        """Shrink-and-perturb the player params to restore plasticity."""
        latest = self.league.get_latest_player(origin="main")
        if latest is None or latest.step_count != step + _STEP_OFFSET[pop.name]:
            self._add_player_to_league(pop, step, origin="main")
            latest = self.league.get_latest_player(origin="main")

        tqdm.write(
            f"Applying shrink-and-perturb plasticity update @ {step} "
            f"(recovery ref: player {latest.step_count})"
        )
        rng = jax.random.PRNGKey(random.randint(0, 2**16 - 1))
        pop.player_state = shrink_and_perturb_player_state(
            pop.player_state,
            rng,
            default_shrink=self.config.plasticity_default_shrink,
            module_shrink=self.config.plasticity_module_shrink,
        )
        pop.plasticity.on_perturbation(latest.step_count, frame_count)

    def _create_params_container(self, pop: PopulationState) -> ParamsContainer:
        return ParamsContainer(
            player_frame_count=jax.device_get(pop.player_state.frame_count).item(),
            builder_frame_count=jax.device_get(pop.builder_state.frame_count).item(),
            step_count=pop.live_key,
            player_params=jax.device_get(pop.player_state.params),
            builder_params=jax.device_get(pop.builder_state.params),
        )

    def _add_player_to_league(
        self, pop: PopulationState, step: int, origin: PopulationName
    ):
        """Persist the current params as an opponent snapshot and register
        a ref. Only the params files are written (no optimiser state); the
        league holds the lightweight ref and materialises the params
        lazily when this player is actually drawn as an opponent.

        Unifies what used to be two separate near-identical code paths
        (this method, and the former rl/online/promote_exploiter.py's
        write_promoted_snapshot) — both were "write a params snapshot,
        construct a PlayerRef, register it," differing only in which
        directory root and what origin/parent_step to tag. One
        population-aware path now serves main's own routine additions AND
        both exploiter populations' promotions/timeout-publications.
        """
        league_step = step + _STEP_OFFSET[origin]
        if origin == "main":
            players_root = f"./ckpts/gen{self.config.generation}/players"
        else:
            players_root = f"./ckpts/gen{self.config.generation}/exploiters/{origin}"
        snapshot_dir = os.path.abspath(f"{players_root}/p_{league_step:08}")
        checkpoint.save_param_snapshot(
            snapshot_dir,
            player_components=dict(
                params=jax.device_get(pop.player_state.params),
                target_params=jax.device_get(pop.player_state.target_params),
            ),
            builder_components=dict(
                params=jax.device_get(pop.builder_state.params),
                target_params=jax.device_get(pop.builder_state.target_params),
            ),
        )
        self.league.add_player(
            PlayerRef(
                step_count=league_step,
                snapshot_dir=snapshot_dir,
                player_frame_count=jax.device_get(pop.player_state.frame_count).item(),
                builder_frame_count=jax.device_get(
                    pop.builder_state.frame_count
                ).item(),
                player_key="params",
                builder_key="params",
                origin=origin,
                parent_step=pop.fork_step if origin != "main" else None,
            )
        )

    def _get_usage_counts(self, pop: PopulationState):
        result = {}
        for key, counts in [
            ("species", pop.player_replay._species_counts),
            ("items", pop.player_replay._item_counts),
            ("abilities", pop.player_replay._ability_counts),
            ("moves", pop.player_replay._move_counts),
        ]:
            names = list(STOI[key])
            table = wandb.Table(columns=[key, "usage"])
            for name, count in zip(names, counts):
                table.add_data(name, count)
            result[f"{key}_usage"] = table
        return result

    def _winrate_tracked_opponents(self) -> list[PlayerRef]:
        """Every historical league member. Payoff-table logging is
        main-only now (one dashboard holds the one shared table), and
        main draws from the whole population, so there is no
        per-population filtering left to do."""
        return [v for k, v in self.league.players.items() if k not in LIVE_KEYS]

    @staticmethod
    def _ref_label(ref: PlayerRef) -> str:
        """Payoff-table label carrying provenance: main snapshots keep
        their raw step count; exploiter-origin snapshots get an ME-/LE-
        prefix with that population's OWN step count (the _STEP_OFFSET
        namespace un-applied), so a row reads as who produced it — the
        point of labelling is seeing the league's non-transitive
        structure (which exploiters beat which mains) at a glance."""
        prefix = {"main": "", "main_exploiter": "ME-", "league_exploiter": "LE-"}[
            ref.origin
        ]
        return f"{prefix}{ref.step_count - _STEP_OFFSET[ref.origin]}"

    def _get_league_winrates(self, pop: PopulationState):
        current = self.league.get_live(pop.live_key)
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

    def _get_league_winrate_heatmap(self, pop: PopulationState):
        """Full pairwise win-rate matrix over the whole shared payoff
        table: live main, both live exploiter populations (when they
        exist), and every historical snapshot with an origin-labelled
        row — logged as a wandb Image alongside (not replacing) the
        per-opponent line-graph metrics above. Live-vs-live cells come
        from real games too (main_exploiter's live-target branch, main's
        verification branch); a pair that has never actually played just
        shows the table's prior."""
        if not _MATPLOTLIB_AVAILABLE:
            return {}

        current = self.league.get_live(pop.live_key)
        others = self._winrate_tracked_opponents()
        if not others:
            return {}

        live_rows, live_labels = [], []
        for name in _EXPLOITER_ROTATION:
            if name in self.populations:
                live_rows.append(self.league.get_live(_LIVE_KEY_BY_POPULATION[name]))
                live_labels.append(
                    "ME (live)" if name == "main_exploiter" else "LE (live)"
                )

        all_players = [current] + live_rows + others
        labels = ["main (live)"] + live_labels + [self._ref_label(p) for p in others]
        matrix = np.asarray(self.league.get_winrate((all_players, all_players)))

        width = max(0.6 * len(labels) + 2, 5.0)
        height = max(0.6 * len(labels) + 2, 4.5)
        fig, ax = plt.subplots(figsize=(width, height))
        im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("away")
        ax.set_ylabel("home (row beats column)")
        ax.set_title("league payoff table (ME/LE = exploiter origin)", fontsize=12)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7
                )
        fig.colorbar(im, ax=ax, label="win rate")
        fig.tight_layout()

        image = wandb.Image(fig)
        plt.close(fig)
        return {"league_winrate_heatmap": image}
