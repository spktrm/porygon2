import functools
import math
import os
import pickle
import queue
import threading
import traceback
from _thread import LockType
from contextlib import nullcontext
from typing import Callable

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import wandb.wandb_run
from loguru import logger
from tqdm import tqdm

import wandb
from rl.environment.data import MAX_RATIO_TOKEN, STOI
from rl.environment.interfaces import BuilderActorInput, PlayerActorInput, Trajectory
from rl.environment.utils import InfoFeature, clip_history, clip_packed_history
from rl.learner.buffer import DirectRatioLimiter, ReplayBuffer
from rl.learner.config import (
    Porygon2BuilderTrainState,
    Porygon2LearnerConfig,
    Porygon2PlayerTrainState,
    create_train_state,
)
from rl.learner.league import (
    GenerationalPlayer,
    League,
    LeagueExploiter,
    MainExploiter,
    MainPlayer,
    Player,
    PlayerID,
)
from rl.learner.utils import calculate_r2, collect_batch_telemetry_data
from rl.model.builder_model import get_builder_model
from rl.model.config import get_builder_model_config, get_player_model_config
from rl.model.heads import HeadParams
from rl.model.player_model import get_player_model
from rl.model.utils import Params, ParamsContainer, promote_map
from rl.utils import average


def calculate_builder_log_prob(
    *,
    species_log_prob: jax.Array,
    packed_set_log_prob: jax.Array,
):
    return species_log_prob + packed_set_log_prob


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
    """Generic PPO clipped surrogate loss"""
    l1 = policy_ratios * advantages
    l2 = jnp.clip(policy_ratios, 1.0 - clip_ppo, 1.0 + clip_ppo) * advantages
    return jnp.minimum(l1, l2)


def policy_gradient_loss(
    *,
    policy_ratios: jax.Array,
    advantages: jax.Array,
    valid: jax.Array,
    clip_ppo: float,
):
    pg_loss = spo_objective(
        policy_ratios=policy_ratios, advantages=advantages, clip_ppo=clip_ppo
    )
    return -average(pg_loss, valid)


def clip_fraction(
    *,
    policy_ratios: jax.Array,
    valid: jax.Array,
    clip_ppo: float,
):
    """Calculate the fraction of clips."""
    clipped = jnp.abs(policy_ratios - 1) > clip_ppo
    return average(clipped, valid)


def mse_value_loss(*, pred: jax.Array, target: jax.Array, valid: jax.Array):
    mse_loss = jnp.square(pred - target)
    return average(mse_loss, valid)


def clipped_value_loss(
    *,
    pred_v: jax.Array,
    target_v: jax.Array,
    old_v: jax.Array,
    valid: jax.Array,
    clip_val: float = 0.2,
):
    loss_unclipped = jnp.square(pred_v - target_v)
    pred_v_clipped = old_v + jnp.clip(pred_v - old_v, -clip_val, clip_val)
    loss_clipped = jnp.square(pred_v_clipped - target_v)
    return average(jnp.maximum(loss_unclipped, loss_clipped), valid)


def ce_value_loss(*, pred_v: jax.Array, target_v: jax.Array, valid: jax.Array):
    mse_loss = -(pred_v * target_v).sum(axis=-1)
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


def clip_log_ratio(log_ratio: jax.Array, eps: float = 0.2):
    return jnp.clip(log_ratio, a_min=math.log1p(-eps), a_max=math.log1p(eps))


def shift_left_with_zeros(tensor: jax.Array):
    return jnp.concatenate((tensor[1:], jnp.zeros_like(tensor[-1:])), axis=0)


def shift_right_with_zeros(tensor: jax.Array):
    return jnp.concatenate((jnp.zeros_like(tensor[:1]), tensor[:-1]), axis=0)


def postprocess_log_prob(log_prob: jax.Array, min_p: float = 0.03):
    """Postprocess log probabilities to avoid extremely small values."""
    return jnp.where(
        log_prob < math.log(min_p), jnp.finfo(log_prob.dtype).min, log_prob
    )


def scalar_to_two_hot(scalar_target: jax.Array) -> jax.Array:
    """
    Projects a scalar [-1, 1] into a smooth distribution over 3 bins {-1, 0, 1}.
    """
    val = jnp.clip(scalar_target, -1.0, 1.0)
    val_idx = val + 1.0

    lower_idx = jnp.floor(val_idx).astype(jnp.int32)
    upper_idx = lower_idx + 1

    weight = val_idx - lower_idx
    upper_idx = jnp.minimum(upper_idx, 2)
    lower_idx = jnp.minimum(lower_idx, 2)

    probs_lower = jax.nn.one_hot(lower_idx, 3) * (1.0 - weight)[..., None]
    probs_upper = jax.nn.one_hot(upper_idx, 3) * weight[..., None]

    return probs_lower + probs_upper


def calculate_player_shaped_reward(
    batch: Trajectory, config: Porygon2LearnerConfig
) -> jax.Array:
    """Calculate potential-based shaped reward for player transitions."""
    env_output = batch.player_transitions.env_output

    my_hp_t = env_output.info[..., InfoFeature.INFO_FEATURE__MY_HP_COUNT]
    opp_hp_t = env_output.info[..., InfoFeature.INFO_FEATURE__OPP_HP_COUNT]
    my_fainted_t = env_output.info[..., InfoFeature.INFO_FEATURE__MY_FAINTED_COUNT]
    opp_fainted_t = env_output.info[..., InfoFeature.INFO_FEATURE__OPP_FAINTED_COUNT]

    # 1. Calculate Potential (Phi) for the current timestep t
    phi_t = (
        config.shaped_reward_hp_scale * (my_hp_t - opp_hp_t)
        + config.shaped_reward_fainted_scale * (opp_fainted_t - my_fainted_t)
    ) / MAX_RATIO_TOKEN  # Normalize to [-1, 1]

    # 2. Calculate Potential for the next_timestep
    phi_tp1 = jnp.concatenate((phi_t[1:], phi_t[-1:]))

    # 3. Return the difference: Current Potential - Previous Potential
    return shift_right_with_zeros(config.gamma * phi_tp1 - phi_t)


def train_step(
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
    batch: Trajectory,
    config: Porygon2LearnerConfig,
):
    """Train for a single step."""

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
    player_target_pred = promote_map(
        player_state.apply_fn(
            player_state.target_params,
            player_actor_input,
            player_transitions.agent_output.actor_output,
            HeadParams(),
        )
    )

    builder_actor_input = BuilderActorInput(
        env=builder_transitions.env_output,
        history=builder_history,
    )
    builder_target_pred = promote_map(
        builder_state.apply_fn(
            builder_state.target_params,
            builder_actor_input,
            builder_transitions.agent_output.actor_output,
            HeadParams(),
        )
    )

    player_actor_action_head = player_transitions.agent_output.actor_output.action_head

    player_target_value_head = player_target_pred.value_head
    player_target_action_head = player_target_pred.action_head

    player_actor_log_prob = player_actor_action_head.log_prob
    player_target_log_prob = player_target_action_head.log_prob

    builder_actor_conditional_entropy_head = (
        builder_transitions.agent_output.actor_output.conditional_entropy_head
    )
    builder_actor_species_head = (
        builder_transitions.agent_output.actor_output.species_head
    )
    builder_actor_packed_set_head = (
        builder_transitions.agent_output.actor_output.packed_set_head
    )

    builder_target_value_head = builder_target_pred.value_head
    builder_target_species_head = builder_target_pred.species_head
    builder_target_packed_set_head = builder_target_pred.packed_set_head

    builder_actor_log_prob = calculate_builder_log_prob(
        species_log_prob=builder_actor_species_head.log_prob,
        packed_set_log_prob=builder_actor_packed_set_head.log_prob,
    )
    builder_target_log_prob = calculate_builder_log_prob(
        species_log_prob=builder_target_species_head.log_prob,
        packed_set_log_prob=builder_target_packed_set_head.log_prob,
    )

    actor_log_prob = jnp.concatenate(
        (builder_actor_log_prob, player_actor_log_prob), axis=0
    )
    target_log_prob = jnp.concatenate(
        (builder_target_log_prob, player_target_log_prob), axis=0
    )

    player_valid = jnp.bitwise_not(player_transitions.env_output.done)
    builder_valid = jnp.bitwise_not(builder_transitions.env_output.done)
    valid = jnp.concatenate((jnp.ones_like(builder_valid), player_valid), axis=0)

    actor_target_log_ratio = actor_log_prob - target_log_prob
    actor_target_ratio = jnp.exp(actor_target_log_ratio)
    target_actor_ratio = jnp.exp(-actor_target_log_ratio)
    target_actor_ratio = target_actor_ratio.at[6].set(1.0)  # Hack to continue trace

    actor_target_clipped_ratio = jnp.clip(actor_target_ratio, min=0.0, max=2.0)
    builder_actor_target_clipped_ratio = actor_target_clipped_ratio[
        : builder_valid.shape[0]
    ]
    player_actor_target_clipped_ratio = actor_target_clipped_ratio[
        builder_valid.shape[0] :
    ]

    shaped_reward = calculate_player_shaped_reward(batch, config)
    player_reward = (
        player_transitions.env_output.win_reward
        + config.shaped_reward_scale * shaped_reward
    )

    rewards_tm1 = jnp.concatenate(
        (jnp.zeros_like(builder_valid), player_reward), axis=0
    )

    v_tm1 = jnp.concatenate(
        (builder_target_value_head.logits, player_target_value_head.logits), axis=0
    )
    v_t = jnp.concatenate((v_tm1[1:], v_tm1[-1:]))
    rewards = shift_left_with_zeros(rewards_tm1)
    discounts = shift_left_with_zeros(valid).astype(v_t.dtype) * config.gamma

    actor_builder_entropy = -jnp.where(builder_valid, builder_actor_log_prob, 0)
    actor_builder_conditional_entropy = jnp.flip(
        jnp.cumsum(jnp.flip(actor_builder_entropy, axis=0), axis=0), axis=0
    )

    # Ratio taken from IMPACT paper: https://arxiv.org/pdf/1912.00167.pdf
    vtrace_td_error_and_advantage = jax.vmap(
        functools.partial(
            rlax.vtrace_td_error_and_advantage,
            lambda_=config.lambda_,
            clip_rho_threshold=config.clip_rho_threshold,
            clip_pg_rho_threshold=config.clip_pg_rho_threshold,
        ),
        in_axes=1,
        out_axes=1,
    )
    vtrace = vtrace_td_error_and_advantage(
        v_tm1, v_t, rewards, discounts, target_actor_ratio
    )

    target_scalar_v = vtrace.errors + v_tm1
    builder_target_scalar_v = target_scalar_v[: builder_valid.shape[0]]
    player_target_scalar_v = target_scalar_v[builder_valid.shape[0] :]

    adv_mean = average(vtrace.pg_advantage, valid)
    adv_std = vtrace.pg_advantage.std(where=valid)

    # Normalize by the ema mean and std of the advantages.
    norm_advantages = (vtrace.pg_advantage - player_state.target_adv_mean) / (
        player_state.target_adv_std + 1e-8
    )

    builder_norm_advantages = norm_advantages[: builder_valid.shape[0]]
    player_norm_advantages = norm_advantages[builder_valid.shape[0] :]

    action_mask_sum = player_transitions.env_output.action_mask.reshape(
        player_valid.shape + (-1,)
    ).sum(axis=-1)

    player_uniform_prob = 1.0 / (action_mask_sum).clip(min=1)
    player_uniform_log_prob = jnp.log(player_uniform_prob)

    entropy_decay = 1 / ((player_state.step_count + 1) ** 0.3)

    def player_loss_fn(params: Params):

        player_pred = promote_map(
            player_state.apply_fn(
                params,
                player_actor_input,
                player_transitions.agent_output.actor_output,
                HeadParams(),
            )
        )

        learner_value_head = player_pred.value_head
        learner_action_head = player_pred.action_head

        learner_log_prob = learner_action_head.log_prob

        # Calculate the log ratios.
        learner_uniform_log_ratio = learner_log_prob - player_uniform_log_prob
        learner_uniform_ratio = jnp.exp(learner_uniform_log_ratio)

        learner_actor_log_ratio = learner_log_prob - player_actor_log_prob
        learner_actor_ratio = jnp.exp(learner_actor_log_ratio)

        learner_target_log_ratio = learner_log_prob - player_target_log_prob
        learner_target_ratio = jnp.exp(learner_target_log_ratio)

        ratio = player_actor_target_clipped_ratio * learner_actor_ratio

        # Calculate losses.
        loss_pg = policy_gradient_loss(
            policy_ratios=ratio,
            advantages=player_norm_advantages,
            valid=player_valid,
            clip_ppo=config.clip_ppo,
        )

        # Softmax cross-entropy loss for value head
        loss_v = mse_value_loss(
            pred=learner_value_head.logits,
            target=player_target_scalar_v,
            valid=player_valid,
        )

        action_head_entropy = average(learner_action_head.entropy, player_valid)
        loss_entropy = backward_kl_loss(
            policy_ratio=learner_uniform_ratio,
            log_policy_ratio=learner_uniform_log_ratio,
            valid=player_valid,
        )

        learner_actor_approx_kl = forward_kl_loss(
            policy_ratio=learner_actor_ratio,
            log_policy_ratio=learner_actor_log_ratio,
            valid=player_valid,
        )
        learner_target_approx_kl = forward_kl_loss(
            policy_ratio=learner_target_ratio,
            log_policy_ratio=learner_target_log_ratio,
            valid=player_valid,
        )
        loss_kl = backward_kl_loss(
            policy_ratio=learner_actor_ratio,
            log_policy_ratio=learner_actor_log_ratio,
            valid=player_valid,
        )

        loss = (
            config.player_policy_loss_coef * loss_pg
            + config.player_value_loss_coef * loss_v
            + config.player_kl_loss_coef * loss_kl
            + config.player_entropy_loss_coef * entropy_decay * loss_entropy
        )

        return loss, dict(
            # Loss values
            player_loss_pg=loss_pg,
            player_loss_v=loss_v,
            player_loss_entropy=loss_entropy,
            player_loss_kl=loss_kl,
            # Per head entropies
            player_action_entropy=action_head_entropy,
            # Ratios
            player_ratio_clip_fraction=clip_fraction(
                policy_ratios=ratio, valid=player_valid, clip_ppo=config.clip_ppo
            ),
            player_learner_actor_ratio=average(learner_actor_ratio, player_valid),
            player_learner_target_ratio=average(learner_target_ratio, player_valid),
            # Approx KL values
            player_learner_actor_approx_kl=learner_actor_approx_kl,
            player_learner_target_approx_kl=learner_target_approx_kl,
            # Extra stats
            player_value_function_r2=calculate_r2(
                value_prediction=learner_value_head.logits,
                value_target=player_target_scalar_v,
                mask=player_valid,
            ),
        )

    def builder_loss_fn(params: Params):

        pred = promote_map(
            builder_state.apply_fn(
                params,
                builder_actor_input,
                builder_transitions.agent_output.actor_output,
                HeadParams(),
            )
        )

        conditional_entropy_head = pred.conditional_entropy_head
        value_head = pred.value_head
        species_head = pred.species_head
        packed_set_head = pred.packed_set_head

        learner_log_prob = calculate_builder_log_prob(
            species_log_prob=species_head.log_prob,
            packed_set_log_prob=packed_set_head.log_prob,
        )

        learner_actor_log_ratio = learner_log_prob - builder_actor_log_prob
        learner_actor_ratio = jnp.exp(learner_actor_log_ratio)

        learner_target_log_ratio = learner_log_prob - builder_target_log_prob
        learner_target_ratio = jnp.exp(learner_target_log_ratio)

        ratio = builder_actor_target_clipped_ratio * learner_actor_ratio

        normalising_constant = 10.0

        # Calculate the losses.
        loss_pg = policy_gradient_loss(
            policy_ratios=ratio,
            advantages=builder_norm_advantages
            + config.builder_entropy_loss_coef
            * entropy_decay
            * (
                actor_builder_conditional_entropy / normalising_constant
                - builder_actor_conditional_entropy_head.logits
            ),
            valid=builder_valid,
            clip_ppo=config.clip_ppo,
        )

        loss_v = mse_value_loss(
            pred=value_head.logits,
            target=builder_target_scalar_v,
            valid=jnp.ones_like(builder_valid),
        )

        species_entropy = average(species_head.entropy, builder_valid)
        packed_set_entropy = average(packed_set_head.entropy, builder_valid)

        # Estimator for entropy
        loss_entropy = mse_value_loss(
            pred=conditional_entropy_head.logits,
            target=actor_builder_conditional_entropy / normalising_constant,
            valid=builder_valid,
        )

        learner_actor_approx_kl = forward_kl_loss(
            policy_ratio=learner_actor_ratio,
            log_policy_ratio=learner_actor_log_ratio,
            valid=builder_valid,
        )
        learner_target_approx_kl = forward_kl_loss(
            policy_ratio=learner_target_ratio,
            log_policy_ratio=learner_target_log_ratio,
            valid=builder_valid,
        )
        loss_kl_rl = backward_kl_loss(
            policy_ratio=learner_actor_ratio,
            log_policy_ratio=learner_actor_log_ratio,
            valid=builder_valid,
        )
        loss_kl_prior = average(
            optax.kl_divergence(
                species_head.log_policy,
                builder_transitions.env_output.target_species_probs,
            ),
            builder_valid,
        )

        loss = (
            config.builder_policy_loss_coef * loss_pg
            + config.builder_value_loss_coef * loss_v
            + config.builder_kl_loss_coef * loss_kl_rl
            + config.builder_kl_prior_loss_coef * loss_kl_prior
            + loss_entropy
        )

        return loss, dict(
            builder_loss_pg=loss_pg,
            builder_loss_v=loss_v,
            builder_loss_kl_rl=loss_kl_rl,
            builder_loss_kl_prior=loss_kl_prior,
            builder_loss_entropy=loss_entropy,
            # Head entropies
            builder_species_entropy=species_entropy,
            builder_packed_set_entropy=packed_set_entropy,
            # Ratios
            builder_ratio_clip_fraction=clip_fraction(
                policy_ratios=ratio, valid=builder_valid, clip_ppo=config.clip_ppo
            ),
            builder_learner_actor_ratio=average(learner_actor_ratio, builder_valid),
            builder_learner_target_ratio=average(learner_target_ratio, builder_valid),
            # Approx KL values
            builder_learner_actor_approx_kl=learner_actor_approx_kl,
            builder_learner_target_approx_kl=learner_target_approx_kl,
            # Extra stats
            builder_value_function_r2=calculate_r2(
                value_prediction=value_head.logits,
                value_target=builder_target_scalar_v,
                mask=builder_valid,
            ),
        )

    player_grad_fn = jax.value_and_grad(player_loss_fn, has_aux=True)
    (player_loss_val, player_logs), player_grads = player_grad_fn(player_state.params)

    builder_grad_fn = jax.value_and_grad(builder_loss_fn, has_aux=True)
    (builder_loss_val, builder_logs), builder_grads = builder_grad_fn(
        builder_state.params
    )

    training_logs = {}
    training_logs.update(player_logs)
    training_logs.update(
        dict(
            player_loss=player_loss_val,
            player_param_norm=optax.global_norm(player_state.params),
            player_gradient_norm=optax.global_norm(player_grads),
            player_is_ratio=average(player_actor_target_clipped_ratio, player_valid),
            player_norm_adv_mean=average(player_norm_advantages, player_valid),
            player_norm_adv_std=player_norm_advantages.std(where=player_valid),
            player_value_target_mean=average(player_target_scalar_v, player_valid),
            player_value_target_std=player_target_scalar_v.std(where=player_valid),
        )
    )
    training_logs.update(builder_logs)
    training_logs.update(
        dict(
            builder_loss=builder_loss_val,
            builder_param_norm=optax.global_norm(builder_state.params),
            builder_gradient_norm=optax.global_norm(builder_grads),
            builder_is_ratio=average(builder_actor_target_clipped_ratio, builder_valid),
            builder_norm_adv_mean=average(builder_norm_advantages, builder_valid),
            builder_norm_adv_std=builder_norm_advantages.std(where=builder_valid),
            builder_value_target_mean=average(builder_target_scalar_v, builder_valid),
            builder_value_target_std=builder_target_scalar_v.std(where=builder_valid),
        )
    )

    player_state = player_state.apply_gradients(grads=player_grads)
    player_state = player_state.replace(
        # Update target params and adv mean/std.
        target_params=optax.incremental_update(
            player_state.params, player_state.target_params, config.player_ema_decay
        ),
        target_adv_mean=player_state.target_adv_mean * (1 - config.player_ema_decay)
        + adv_mean * config.player_ema_decay,
        target_adv_std=player_state.target_adv_std * (1 - config.player_ema_decay)
        + adv_std * config.player_ema_decay,
        step_count=player_state.step_count + 1,
        frame_count=player_state.frame_count + player_valid.sum(),
    )

    builder_state = builder_state.apply_gradients(grads=builder_grads)
    builder_state = builder_state.replace(
        # Update target params.
        target_params=optax.incremental_update(
            builder_state.params, builder_state.target_params, config.builder_ema_decay
        ),
        step_count=builder_state.step_count + 1,
        frame_count=builder_state.frame_count + builder_valid.sum(),
    )

    training_logs.update(collect_batch_telemetry_data(batch, config))
    training_logs.update(
        dict(
            player_frame_count=player_state.frame_count,
            builder_frame_count=builder_state.frame_count,
            training_step=player_state.step_count,
            entropy_decay_coeff=entropy_decay,
        )
    )

    return player_state, builder_state, training_logs


def stack_batch(
    batch: list[Trajectory],
    player_transition_resolution: int = 50,
    player_history_resolution: int = 128,
):
    stacked_trajectory: Trajectory = jax.tree.map(
        lambda *xs: np.stack(xs, axis=1), *batch
    )

    valid = np.bitwise_not(stacked_trajectory.player_transitions.env_output.done)
    valid_sum = valid.sum(0).max().item()

    num_valid = int(
        np.ceil(valid_sum / player_transition_resolution) * player_transition_resolution
    )

    clipped_trajectory = Trajectory(
        builder_transitions=stacked_trajectory.builder_transitions,
        builder_history=stacked_trajectory.builder_history,
        player_transitions=jax.tree.map(
            lambda x: x[:num_valid], stacked_trajectory.player_transitions
        ),
        player_packed_history=clip_packed_history(
            stacked_trajectory.player_packed_history,
            resolution=player_history_resolution,
        ),
        player_history=clip_history(
            stacked_trajectory.player_history, resolution=player_history_resolution
        ),
    )

    return clipped_trajectory


train_step_jit = jax.jit(train_step, static_argnames=["config"])


class Learner:
    def __init__(
        self,
        player: GenerationalPlayer,
        player_state: Porygon2PlayerTrainState,
        builder_state: Porygon2BuilderTrainState,
        config: Porygon2LearnerConfig,
        wandb_run: wandb.wandb_run.Run | None,
        league: League,
        gpu_lock: LockType | None = None,
        log_prefix: str = "",
    ):
        self.player = player

        self.player_state = player_state
        self.builder_state = builder_state
        self.config = config
        self.wandb_run = wandb_run
        self.gpu_lock = gpu_lock or nullcontext()

        self.done = False
        self.replay = ReplayBuffer(self.config.replay_buffer_capacity)

        self.controller = DirectRatioLimiter(
            target_rr=self.config.target_replay_ratio,
            batch_size=self.config.batch_size,
            warmup_trajectories=self.config.batch_size * 16,
        )
        self.controller.set_replay_buffer_len_fn(lambda: len(self.replay))

        self.device_q: queue.Queue[Trajectory] = queue.Queue(maxsize=1)
        self.league = league

        # progress bars
        self.log_prefix = log_prefix
        self.producer_progress = tqdm(desc=f"{log_prefix} producer", smoothing=0.1)
        self.consumer_progress = tqdm(desc=f"{log_prefix} consumer", smoothing=0.1)
        self.train_progress = tqdm(desc=f"{log_prefix} batches", smoothing=0.1)

        transfer_thread = threading.Thread(target=self.host_to_device_worker)
        transfer_thread.start()

    def serialize(self) -> bytes:
        return cloudpickle.dumps(
            {
                k: v
                for k, v in self.__dict__.items()
                if k in ["player", "player_state", "builder_state", "log_prefix"]
            }
        )

    @classmethod
    def deserialize(
        cls,
        config: Porygon2LearnerConfig,
        wandb_run: wandb.wandb_run.Run | None,
        league: League,
        gpu_lock: LockType | None,
        data: bytes,
    ):
        obj = cloudpickle.loads(data)
        instance = cls(
            player=obj["player"],
            player_state=obj["player_state"],
            builder_state=obj["builder_state"],
            config=config,
            wandb_run=wandb_run,
            league=league,
            gpu_lock=gpu_lock,
            log_prefix=obj["log_prefix"],
        )
        return instance

    def enqueue_traj(self, traj: Trajectory):
        # Block if the ratio is too low (we are too far ahead)
        self.controller.wait_for_produce_permission()

        if self.done:
            return

        self.producer_progress.update(1)
        self.replay.add(traj)

        # Notify the controller that we have produced one trajectory
        self.controller.notify_produced(n_trajectories=1)

    def host_to_device_worker(self):
        max_burst = 8
        batch_size = self.config.batch_size

        while not self.done:
            taken = 0

            while not self.done and taken < max_burst:

                # Wait for permission from the controller
                # This checks both data availability AND the ratio
                self.controller.wait_for_consume_permission()

                if self.done:
                    break

                # Sample & execute one update outside the lock.
                batch = self.replay.sample(self.config.batch_size)
                self.consumer_progress.update(batch_size)

                stacked = stack_batch(batch)
                self.device_q.put(stacked)

                # Notify the controller that we have consumed a batch
                self.controller.notify_consumed(n_trajectories=batch_size)

                taken += 1

        logger.info("host_to_device_worker exiting.")

    def get_usage_counts(self, usage_counts: np.ndarray):
        usage_logs = {}
        names = list(STOI["species"])

        table = wandb.Table(columns=["species", "usage"])
        for name, count in zip(names, usage_counts):
            table.add_data(name, count)

        usage_logs["species_usage"] = table
        return usage_logs

    def get_league_winrates(self):
        league = self.league

        current_player = league.get_player(self.player.player_id)

        historical_players = league._payoff.get_potential_opponents(
            "historical", current_player.player_id
        )
        win_rates = league._payoff[current_player.player_id, historical_players]
        return {
            f"{self.player.player_id}_v_{historical_players[i]}_winrate": win_rate
            for i, win_rate in enumerate(win_rates)
        }

    def update_current_player(self):
        weight_id = self.player.weight_id
        new_weights = ParamsContainer(
            player_type=self.player.__class__.__name__,
            parent=weight_id,
            step_count=int(self.player_state.step_count),
            player_frame_count=int(self.player_state.frame_count),
            builder_frame_count=int(self.builder_state.frame_count),
            player_params=self.player_state.params,
            builder_params=self.builder_state.params,
        )
        self.league.update_weights(weight_id, new_weights)

    def train_step(self):
        batch = self.device_q.get()
        batch: Trajectory = jax.device_put(batch)

        new_player_state: Porygon2PlayerTrainState
        new_builder_state: Porygon2BuilderTrainState
        training_logs: dict

        with self.gpu_lock:
            new_player_state, new_builder_state, training_logs = train_step_jit(
                self.player_state,
                self.builder_state,
                batch,
                self.config,
            )

        training_step = training_logs["training_step"]

        if jnp.isfinite(training_logs["player_loss"]).item():
            self.player_state = new_player_state
        else:
            print("Non-finite loss detected in player @ step", training_step)
            return

        if jnp.isfinite(training_logs["builder_loss"]).item():
            self.builder_state = new_builder_state
        else:
            print("Non-finite loss detected in builder @ step", training_step)
            return

        player_step = int(training_step)

        if (player_step % self.config.save_interval_steps) == 0:
            usage_logs = self.get_usage_counts(self.replay._species_counts)
            training_logs.update(jax.device_get(usage_logs))

        if (player_step % self.config.league_winrate_log_steps) == 0:
            league_winrates = self.get_league_winrates()
            training_logs.update(jax.device_get(league_winrates))

        # Update the tqdm progress bars.
        self.train_progress.update(1)
        rr = self.controller._get_current_rr()
        self.train_progress.set_description(f"{self.log_prefix} batches - rr: {rr:.2f}")

        training_logs["replay_ratio"] = rr

        self.wandb_run.log(training_logs)

        self.update_current_player()

        return training_logs


class LearnerManager:
    def __init__(
        self,
        config: Porygon2LearnerConfig,
        wandb_run: wandb.wandb_run.Run,
        do_initialize: bool = True,
    ):
        self.config = config
        self.wandb_run = wandb_run
        self.gpu_lock = threading.Lock()

        learner_player_conf = get_player_model_config(config.generation, train=True)
        learner_build_conf = get_builder_model_config(config.generation, train=True)

        self.player_net = get_player_model(learner_player_conf)
        self.build_net = get_builder_model(learner_build_conf)

        if do_initialize:
            self.initialize()

    def initialize(self):
        self.league = League()
        self.learners: dict[PlayerID, Learner] = {}

        rng_key_init = 0
        for num_agents, player_fn, add_checkpoint in [
            (self.config.main_agents, MainPlayer, True),
            (self.config.main_exploiters, MainExploiter, False),
            (self.config.league_exploiters, LeagueExploiter, False),
        ]:
            self._init_agents(
                num_agents=num_agents,
                player_fn=player_fn,
                seed_init=rng_key_init,
                add_checkpoint=add_checkpoint,
            )
            rng_key_init += num_agents

        self.current_learner_index = 0
        self.cycle = 0

    @property
    def current_learner(self) -> Learner:
        learner_ids = list(self.learners.keys())
        current_player_id = learner_ids[self.current_learner_index]
        return self.learners[current_player_id]

    def serialize(self):
        data = {
            k: v
            for k, v in self.__dict__.items()
            if k in ["league", "current_learner_index", "cycle"]
        }
        learners = {}
        for player_id, learner in self.learners.items():
            learners[player_id] = learner.serialize()
        data["learners"] = learners
        return cloudpickle.dumps(data)

    @classmethod
    def deserialize(
        cls,
        config: Porygon2LearnerConfig,
        wandb_run: wandb.wandb_run.Run,
        serialized: bytes,
    ):
        state = cloudpickle.loads(serialized)
        obj = cls(config, wandb_run, do_initialize=False)
        obj.league = state["league"]
        obj.current_learner_index = state["current_learner_index"]
        obj.cycle = state["cycle"]
        obj.learners = {}
        for player_id, learner_data in state["learners"].items():
            learner = Learner.deserialize(
                config=config,
                wandb_run=wandb_run,
                league=obj.league,
                gpu_lock=obj.gpu_lock,
                data=learner_data,
            )
            obj.learners[player_id] = learner
        return obj

    def _init_agents(
        self,
        num_agents: int,
        player_fn: Callable[..., Player],
        seed_init: int,
        add_checkpoint: bool = True,
    ):
        seed_start = seed_init
        player_type = player_fn.__name__

        for agent_index in range(num_agents):
            player_state, builder_state = create_train_state(
                self.player_net,
                self.build_net,
                jax.random.key(seed_start + agent_index),
                self.config,
            )
            container = ParamsContainer(
                player_type=player_type,
                parent="initial",
                step_count=0,
                player_frame_count=player_state.frame_count,
                builder_frame_count=builder_state.frame_count,
                player_params=player_state.params,
                builder_params=builder_state.params,
            )
            weight_id = self.league.weight_store.add(container)

            player = player_fn(weight_id, self.league._payoff, self.league.weight_store)

            self.league.add_player(player)
            if add_checkpoint:
                self.league.add_player(player.checkpoint())

            self.learners[player.player_id] = self._init_learner(
                player,
                player_state,
                builder_state,
                log_prefix=f"{player_type}_{player.player_id}",
            )

    def _init_learner(
        self,
        player: Player,
        player_state: Porygon2PlayerTrainState,
        builder_state: Porygon2BuilderTrainState,
        log_prefix: str,
    ) -> Learner:
        logger.info(f"Initializing learner for player {player.player_id}")
        return Learner(
            player,
            player_state=player_state,
            builder_state=builder_state,
            config=self.config,
            wandb_run=self.wandb_run,
            league=self.league,
            gpu_lock=self.gpu_lock,
            log_prefix=log_prefix,
        )

    def train(self):
        while True:
            for learner in self.learners.values():
                self.train_learner(learner)

    def train_learner(self, learner: Learner):
        should_add_player = False
        prev_add_player_check = learner.player.get_weights().step_count

        while not should_add_player:
            try:
                training_logs = learner.train_step()
            except Exception as e:
                logger.error(
                    f"Learner {learner.player.player_id} train step failed: {e}"
                )
                traceback.print_exc()

                raise e

            player_step = training_logs["training_step"]
            if (player_step - prev_add_player_check) >= 10:
                should_add_player = learner.player.ready_to_checkpoint()
                prev_add_player_check = player_step

        self.league.add_player(learner.player.checkpoint())
        logger.info(f"Learner {learner.player.player_id} added checkpoint to league.")

        self.current_learner_index = (self.current_learner_index + 1) % len(
            self.learners
        )
        self.cycle += 1

        save_train_state_locally(self)

    def get_current_player(self):
        return self.current_learner.player


def save_train_state_locally(manager: LearnerManager):
    save_path = os.path.abspath(
        f"ckpts/gen{manager.config.generation}/ckpt_{manager.cycle:03}.pkl"
    )
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_obj = manager.serialize()
    with open(save_path, "wb") as f:
        f.write(save_obj)
    return save_path


if __name__ == "__main__":
    with open("ckpts/gen9/ckpt_00400000", "rb") as f:
        datum = pickle.load(f)

    manager = LearnerManager(config=Porygon2LearnerConfig(), wandb_run=None)

    for _ in range(2):
        for learner in manager.learners.values():
            manager.league.add_player(learner.player.checkpoint())

    for player in manager.league._learning_agents:
        for _ in range(100):
            opponent_id = player.get_match()

    obj = manager.serialize()
    new_manager = LearnerManager.deserialize(
        config=Porygon2LearnerConfig(), wandb_run=None, serialized=obj
    )

    print()
