import functools
import math
import queue
import threading
import traceback
from _thread import LockType
from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import wandb.wandb_run
from loguru import logger
from tqdm import tqdm

import wandb
from rl.environment.data import STOI
from rl.environment.interfaces import (
    BuilderActorInput,
    BuilderHistoryOutput,
    BuilderTransition,
    PlayerActorInput,
    PlayerHiddenInfo,
    PlayerHistoryOutput,
    PlayerTransition,
    Trajectory,
)
from rl.environment.utils import clip_history
from rl.learner.buffer import DirectRatioLimiter, ReplayBuffer
from rl.learner.config import (
    Porygon2BuilderTrainState,
    Porygon2LearnerConfig,
    Porygon2PlayerTrainState,
    save_train_state,
)
from rl.learner.league import MAIN_KEY, League
from rl.learner.utils import calculate_r2, collect_batch_telemetry_data
from rl.model.heads import HeadParams
from rl.model.utils import Params, ParamsContainer, promote_map
from rl.utils import average


def calculate_player_log_prob(
    *,
    action_type_log_prob: jax.Array,
    action_type_index: jax.Array,
    move_log_prob: jax.Array,
    wildcard_log_prob: jax.Array,
    switch_log_prob: jax.Array,
):
    is_move = action_type_index == 0
    is_sw_or_preview = (action_type_index == 1) | (action_type_index == 2)

    return (
        action_type_log_prob
        + jnp.where(is_move, move_log_prob + wildcard_log_prob, 0.0)
        + jnp.where(is_sw_or_preview, switch_log_prob, 0.0)
    )


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


def value_loss(*, pred_v: jax.Array, target_v: jax.Array, valid: jax.Array):
    mse_loss = jnp.square(pred_v - target_v)
    return 0.5 * average(mse_loss, valid)


def approx_backward_kl(*, policy_ratio: jax.Array, log_policy_ratio: jax.Array):
    """
    Calculate the Backward KL approximation.
    """
    return policy_ratio * log_policy_ratio - (policy_ratio - 1)


def approx_forward_kl(*, policy_ratio: jax.Array, log_policy_ratio: jax.Array):
    """
    Calculate the Forward KL approximation.
    """
    return (policy_ratio - 1) - log_policy_ratio


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


def postprocess_log_prob(log_prob: jax.Array, min_p: float = 0.03):
    """Postprocess log probabilities to avoid extremely small values."""
    return jnp.where(
        log_prob < math.log(min_p), jnp.finfo(log_prob.dtype).min, log_prob
    )


def player_train_step(
    player_state: Porygon2PlayerTrainState,
    player_transitions: PlayerTransition,
    player_history: PlayerHistoryOutput,
    config: Porygon2LearnerConfig,
):
    """Train for a single step."""

    player_actor_input = PlayerActorInput(
        env=player_transitions.env_output, history=player_history
    )
    player_target_pred = promote_map(
        player_state.apply_fn(
            player_state.target_params,
            player_actor_input,
            player_transitions.agent_output.actor_output,
            HeadParams(),
        )
    )

    actor_action_type_head = (
        player_transitions.agent_output.actor_output.action_type_head
    )
    actor_move_head = player_transitions.agent_output.actor_output.move_head
    actor_wildcard_head = player_transitions.agent_output.actor_output.wildcard_head
    actor_switch_head = player_transitions.agent_output.actor_output.switch_head

    target_action_type_head = player_target_pred.action_type_head
    target_move_head = player_target_pred.move_head
    target_wildcard_head = player_target_pred.wildcard_head
    target_switch_head = player_target_pred.switch_head

    actor_log_prob = calculate_player_log_prob(
        action_type_log_prob=actor_action_type_head.log_prob,
        action_type_index=actor_action_type_head.action_index,
        move_log_prob=actor_move_head.log_prob,
        wildcard_log_prob=actor_wildcard_head.log_prob,
        switch_log_prob=actor_switch_head.log_prob,
    )
    target_log_prob = calculate_player_log_prob(
        action_type_log_prob=target_action_type_head.log_prob,
        action_type_index=target_action_type_head.action_index,
        move_log_prob=target_move_head.log_prob,
        wildcard_log_prob=target_wildcard_head.log_prob,
        switch_log_prob=target_switch_head.log_prob,
    )

    actor_target_log_ratio = actor_log_prob - target_log_prob
    actor_target_ratio = jnp.exp(actor_target_log_ratio)
    target_actor_ratio = jnp.exp(-actor_target_log_ratio)

    actor_target_clipped_ratio = jnp.clip(actor_target_ratio, min=0.0, max=2.0)

    valid = jnp.bitwise_not(player_transitions.env_output.done)
    move_valid = valid & (actor_action_type_head.action_index == 0)
    switch_valid = valid & (actor_action_type_head.action_index == 1)
    wildcard_valid = move_valid & (
        player_transitions.env_output.wildcard_mask.sum(axis=-1) > 1
    )

    reg_reward = approx_backward_kl(
        policy_ratio=actor_target_ratio, log_policy_ratio=actor_target_log_ratio
    )
    rewards_tm1 = (
        player_transitions.env_output.win_reward - config.player_eta * reg_reward
    )

    v_tm1 = player_target_pred.v
    v_t = jnp.concatenate((v_tm1[1:], v_tm1[-1:]))
    rewards = shift_left_with_zeros(rewards_tm1)
    discounts = shift_left_with_zeros(valid).astype(v_t.dtype) * config.player_gamma

    # Ratio taken from IMPACT paper: https://arxiv.org/pdf/1912.00167.pdf
    vtrace_td_error_and_advantage = jax.vmap(
        functools.partial(
            rlax.vtrace_td_error_and_advantage,
            lambda_=config.player_lambda,
            clip_rho_threshold=config.clip_rho_threshold,
            clip_pg_rho_threshold=config.clip_pg_rho_threshold,
        ),
        in_axes=1,
        out_axes=1,
    )
    vtrace = vtrace_td_error_and_advantage(
        v_tm1, v_t, rewards, discounts, target_actor_ratio
    )
    target_v = vtrace.errors + v_tm1

    player_adv_mean = average(vtrace.pg_advantage, valid)
    player_adv_std = vtrace.pg_advantage.std(where=valid)

    # Normalize by the ema mean and std of the advantages.
    player_norm_advantages = (vtrace.pg_advantage - player_state.target_adv_mean) / (
        player_state.target_adv_std + 1e-8
    )

    def player_loss_fn(params: Params):

        player_pred = promote_map(
            player_state.apply_fn(
                params,
                player_actor_input,
                player_transitions.agent_output.actor_output,
                HeadParams(),
            )
        )

        pred_action_type_head = player_pred.action_type_head
        pred_move_head = player_pred.move_head
        pred_wildcard_head = player_pred.wildcard_head
        pred_switch_head = player_pred.switch_head

        learner_log_prob = calculate_player_log_prob(
            action_type_log_prob=pred_action_type_head.log_prob,
            action_type_index=pred_action_type_head.action_index,
            move_log_prob=pred_move_head.log_prob,
            wildcard_log_prob=pred_wildcard_head.log_prob,
            switch_log_prob=pred_switch_head.log_prob,
        )

        # Calculate the log ratios.
        learner_actor_log_ratio = learner_log_prob - actor_log_prob
        learner_actor_ratio = jnp.exp(learner_actor_log_ratio)

        learner_target_log_ratio = learner_log_prob - target_log_prob
        learner_target_ratio = jnp.exp(learner_target_log_ratio)

        ratio = actor_target_clipped_ratio * learner_actor_ratio

        # Calculate losses.
        loss_pg = policy_gradient_loss(
            policy_ratios=ratio,
            advantages=player_norm_advantages,
            valid=valid,
            clip_ppo=config.clip_ppo,
        )

        loss_v = value_loss(pred_v=player_pred.v, target_v=target_v, valid=valid)

        action_type_head_entropy = average(pred_action_type_head.entropy, valid)
        move_head_entropy = average(pred_move_head.entropy, move_valid)
        switch_head_entropy = average(pred_switch_head.entropy, switch_valid)
        wildcard_head_entropy = average(pred_wildcard_head.entropy, wildcard_valid)

        loss_entropy = (
            action_type_head_entropy
            + move_head_entropy
            + switch_head_entropy
            + wildcard_head_entropy
        )

        learner_actor_approx_kl = forward_kl_loss(
            policy_ratio=learner_actor_ratio,
            log_policy_ratio=learner_actor_log_ratio,
            valid=valid,
        )
        learner_target_approx_kl = forward_kl_loss(
            policy_ratio=learner_target_ratio,
            log_policy_ratio=learner_target_log_ratio,
            valid=valid,
        )
        loss_kl = backward_kl_loss(
            policy_ratio=learner_actor_ratio,
            log_policy_ratio=learner_actor_log_ratio,
            valid=valid,
        )

        loss = (
            config.player_policy_loss_coef * loss_pg
            + config.player_value_loss_coef * loss_v
        )

        return loss, dict(
            # Loss values
            player_loss_pg=loss_pg,
            player_loss_v=loss_v,
            player_loss_entropy=loss_entropy,
            player_loss_kl=loss_kl,
            # Per head entropies
            player_action_type_entropy=action_type_head_entropy,
            player_move_entropy=move_head_entropy,
            player_switch_entropy=switch_head_entropy,
            player_wildcard_entropy=wildcard_head_entropy,
            # Ratios
            player_ratio_clip_fraction=clip_fraction(
                policy_ratios=ratio, valid=valid, clip_ppo=config.clip_ppo
            ),
            player_learner_actor_ratio=average(learner_actor_ratio, valid),
            player_learner_target_ratio=average(learner_target_ratio, valid),
            # Approx KL values
            player_learner_actor_approx_kl=learner_actor_approx_kl,
            player_learner_target_approx_kl=learner_target_approx_kl,
            # Extra stats
            player_value_function_r2=calculate_r2(
                value_prediction=player_pred.v, value_target=target_v, mask=valid
            ),
        )

    player_grad_fn = jax.value_and_grad(player_loss_fn, has_aux=True)
    (player_loss_val, player_logs), player_grads = player_grad_fn(player_state.params)

    player_logs.update(
        dict(
            player_reg_reward_sum=reg_reward.sum(where=valid, axis=0).mean(),
            player_loss=player_loss_val,
            player_param_norm=optax.global_norm(player_state.params),
            player_gradient_norm=optax.global_norm(player_grads),
            player_adv_mean=player_adv_mean,
            player_adv_std=player_adv_std,
            player_is_ratio=average(actor_target_clipped_ratio, valid),
            player_norm_adv_mean=average(player_norm_advantages, valid),
            player_norm_adv_std=player_norm_advantages.std(where=valid),
            player_value_target_mean=average(target_v, valid),
            player_value_target_std=target_v.std(where=valid),
        )
    )

    player_state = player_state.apply_gradients(grads=player_grads)
    player_state = player_state.replace(
        # Update target params and adv mean/std.
        target_params=optax.incremental_update(
            player_state.params, player_state.target_params, config.tau
        ),
        target_adv_mean=player_state.target_adv_mean * (1 - config.tau)
        + player_adv_mean * config.tau,
        target_adv_std=player_state.target_adv_std * (1 - config.tau)
        + player_adv_std * config.tau,
        # Update num steps sampled.
        num_steps=player_state.num_steps + 1,
        # Add 1 for the final step in each trajectory
        actor_steps=player_state.actor_steps + valid.sum(),
    )

    training_logs = dict(
        actor_steps=player_state.actor_steps,
        Step=player_state.num_steps,
    )

    training_logs.update(player_logs)

    return player_state, training_logs


def builder_train_step(
    builder_state: Porygon2BuilderTrainState,
    builder_transitions: BuilderTransition,
    builder_history: BuilderHistoryOutput,
    player_hidden: PlayerHiddenInfo,
    final_reward: jax.Array,
    config: Porygon2LearnerConfig,
):
    """Train for a single step."""

    builder_actor_input = BuilderActorInput(
        env=builder_transitions.env_output,
        hidden=player_hidden,
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

    actor_species_head = builder_transitions.agent_output.actor_output.species_head
    actor_packed_set_head = (
        builder_transitions.agent_output.actor_output.packed_set_head
    )

    target_species_head = builder_target_pred.species_head
    target_packed_set_head = builder_target_pred.packed_set_head

    actor_log_prob = calculate_builder_log_prob(
        species_log_prob=actor_species_head.log_prob,
        packed_set_log_prob=actor_packed_set_head.log_prob,
    )
    target_log_prob = calculate_builder_log_prob(
        species_log_prob=target_species_head.log_prob,
        packed_set_log_prob=target_packed_set_head.log_prob,
    )

    actor_target_log_ratio = actor_log_prob - target_log_prob
    actor_target_ratio = jnp.exp(actor_target_log_ratio)
    target_actor_ratio = jnp.exp(-actor_target_log_ratio)

    actor_target_clipped_ratio = jnp.clip(actor_target_ratio, min=0.0, max=2.0)

    valid = jnp.bitwise_not(builder_transitions.env_output.done)

    reg_reward = approx_backward_kl(
        policy_ratio=actor_target_ratio, log_policy_ratio=actor_target_log_ratio
    )
    rewards_tm1 = (
        jax.nn.one_hot(valid.sum(axis=0), valid.shape[0], axis=0) * final_reward[None]
    ) - config.builder_eta * reg_reward

    v_tm1 = builder_target_pred.v
    v_t = jnp.concatenate((v_tm1[1:], v_tm1[-1:]))
    rewards_t = shift_left_with_zeros(rewards_tm1)
    discounts = shift_left_with_zeros(valid).astype(v_t.dtype) * config.player_gamma

    # Ratio taken from IMPACT paper: https://arxiv.org/pdf/1912.00167.pdf
    vtrace_td_error_and_advantage = jax.vmap(
        functools.partial(
            rlax.vtrace_td_error_and_advantage,
            lambda_=config.builder_lambda,
            clip_rho_threshold=config.clip_rho_threshold,
            clip_pg_rho_threshold=config.clip_pg_rho_threshold,
        ),
        in_axes=1,
        out_axes=1,
    )
    vtrace = vtrace_td_error_and_advantage(
        v_tm1, v_t, rewards_t, discounts, target_actor_ratio
    )
    target_v = vtrace.errors + v_tm1

    builder_adv_mean = average(vtrace.pg_advantage, valid)
    builder_adv_std = vtrace.pg_advantage.std(where=valid)

    # Normalize by the ema mean and std of the advantages.
    builder_norm_advantages = (vtrace.pg_advantage - builder_state.target_adv_mean) / (
        builder_state.target_adv_std + 1e-8
    )

    def builder_loss_fn(params: Params):

        builder_pred = promote_map(
            builder_state.apply_fn(
                params,
                builder_actor_input,
                builder_transitions.agent_output.actor_output,
                HeadParams(),
            )
        )

        learner_species_head = builder_pred.species_head
        learner_packed_set_head = builder_pred.packed_set_head

        learner_log_prob = calculate_builder_log_prob(
            species_log_prob=learner_species_head.log_prob,
            packed_set_log_prob=learner_packed_set_head.log_prob,
        )

        learner_actor_log_ratio = learner_log_prob - actor_log_prob
        learner_actor_ratio = jnp.exp(learner_actor_log_ratio)

        learner_target_log_ratio = learner_log_prob - target_log_prob
        learner_target_ratio = jnp.exp(learner_target_log_ratio)

        ratio = actor_target_clipped_ratio * learner_actor_ratio

        # Calculate the losses.
        loss_pg = policy_gradient_loss(
            policy_ratios=ratio,
            advantages=builder_norm_advantages,
            valid=valid,
            clip_ppo=config.clip_ppo,
        )

        loss_v = value_loss(pred_v=builder_pred.v, target_v=target_v, valid=valid)

        species_entropy = average(learner_species_head.entropy, valid)
        packed_set_entropy = average(learner_packed_set_head.entropy, valid)
        loss_entropy = species_entropy + packed_set_entropy

        learner_actor_approx_kl = forward_kl_loss(
            policy_ratio=learner_actor_ratio,
            log_policy_ratio=learner_actor_log_ratio,
            valid=valid,
        )
        learner_target_approx_kl = forward_kl_loss(
            policy_ratio=learner_target_ratio,
            log_policy_ratio=learner_target_log_ratio,
            valid=valid,
        )
        loss_kl = backward_kl_loss(
            policy_ratio=learner_actor_ratio,
            log_policy_ratio=learner_actor_log_ratio,
            valid=valid,
        )

        loss = (
            config.builder_policy_loss_coef * loss_pg
            + config.builder_value_loss_coef * loss_v
        )

        return loss, dict(
            builder_loss_pg=loss_pg,
            builder_loss_v=loss_v,
            builder_loss_kl=loss_kl,
            builder_loss_entropy=loss_entropy,
            # Head entropies
            builder_species_entropy=species_entropy,
            builder_packed_set_entropy=packed_set_entropy,
            # Ratios
            builder_ratio_clip_fraction=clip_fraction(
                policy_ratios=ratio, valid=valid, clip_ppo=config.clip_ppo
            ),
            builder_learner_actor_ratio=average(learner_actor_ratio, valid),
            builder_learner_target_ratio=average(learner_target_ratio, valid),
            # Approx KL values
            builder_learner_actor_approx_kl=learner_actor_approx_kl,
            builder_learner_target_approx_kl=learner_target_approx_kl,
            # Extra stats
            builder_value_function_r2=calculate_r2(
                value_prediction=builder_pred.v, value_target=target_v, mask=valid
            ),
        )

    builder_grad_fn = jax.value_and_grad(builder_loss_fn, has_aux=True)
    (builder_loss_val, training_logs), builder_grads = builder_grad_fn(
        builder_state.params
    )

    training_logs.update(
        dict(
            builder_reg_reward_sum=reg_reward.sum(where=valid, axis=0).mean(),
            builder_loss=builder_loss_val,
            builder_param_norm=optax.global_norm(builder_state.params),
            builder_gradient_norm=optax.global_norm(builder_grads),
            builder_adv_mean=builder_adv_mean,
            builder_adv_std=builder_adv_std,
            builder_is_ratio=average(actor_target_clipped_ratio, valid),
            builder_norm_adv_mean=average(builder_norm_advantages, valid),
            builder_norm_adv_std=builder_norm_advantages.std(where=valid),
            builder_value_target_mean=average(target_v, valid),
            builder_value_target_std=target_v.std(where=valid),
        )
    )

    builder_state = builder_state.apply_gradients(grads=builder_grads)
    builder_state = builder_state.replace(
        # Update target params.
        target_params=optax.incremental_update(
            builder_state.params, builder_state.target_params, config.tau
        ),
        target_adv_mean=builder_state.target_adv_mean * (1 - config.tau)
        + builder_adv_mean * config.tau,
        target_adv_std=builder_state.target_adv_std * (1 - config.tau)
        + builder_adv_std * config.tau,
        actor_steps=builder_state.actor_steps + valid.sum(),
    )

    return builder_state, training_logs


def calculate_builder_final_reward(batch: Trajectory):
    # my_fainted_count = batch.player_transitions.env_output.info[
    #     ..., InfoFeature.INFO_FEATURE__MY_FAINTED_COUNT
    # ]
    # opp_fainted_count = batch.player_transitions.env_output.info[
    #     ..., InfoFeature.INFO_FEATURE__OPP_FAINTED_COUNT
    # ]
    # phi_t = (opp_fainted_count - my_fainted_count) / MAX_RATIO_TOKEN
    # phi_tp1 = jnp.concatenate((phi_t[1:], jnp.zeros_like(phi_t[:1])), axis=0)
    # shaped_reward = phi_tp1 - phi_t
    # player_valid = jnp.bitwise_not(batch.player_transitions.env_output.done)

    return batch.player_transitions.env_output.win_reward[-1]


class Learner:
    def __init__(
        self,
        player_state: Porygon2PlayerTrainState,
        builder_state: Porygon2BuilderTrainState,
        learner_config: Porygon2LearnerConfig,
        wandb_run: wandb.wandb_run.Run,
        league: League | None = None,
        gpu_lock: LockType | None = None,
    ):
        self.player_state = player_state
        self.builder_state = builder_state
        self.learner_config = learner_config

        self.wandb_run = wandb_run
        self.gpu_lock = gpu_lock or nullcontext()

        self.done = False
        self.replay = ReplayBuffer(self.learner_config.replay_buffer_capacity)

        self.controller = DirectRatioLimiter(
            target_rr=self.learner_config.target_replay_ratio,
            batch_size=self.learner_config.batch_size,
            warmup_trajectories=self.learner_config.batch_size * 16,
        )
        self.controller.set_replay_buffer_len_fn(lambda: len(self.replay))

        self.device_q: queue.Queue[Trajectory] = queue.Queue(maxsize=1)

        self.wandb_run.log_code("rl/")
        self.wandb_run.log_code("inference/")
        self.wandb_run.log_code(
            "service/src/client/", include_fn=lambda x: x.endswith(".ts")
        )

        try:
            init_frame_count = player_state.actor_steps.item()
            init_step_count = player_state.num_steps.item()
        except:
            init_frame_count = int(player_state.actor_steps)
            init_step_count = int(player_state.num_steps)

        if league is not None:
            self.league = league
        else:
            init_params: ParamsContainer = ParamsContainer(
                frame_count=init_frame_count,
                step_count=init_step_count,
                player_params=jax.device_get(self.player_state.params),
                builder_params=jax.device_get(self.builder_state.params),
            )
            self.league = League(
                main_player=init_params,
                players=[init_params],
                league_size=self.learner_config.league_size,
            )

        # progress bars
        self.producer_progress = tqdm(desc="producer", smoothing=0.1)
        self.consumer_progress = tqdm(desc="consumer", smoothing=0.1)
        self.train_progress = tqdm(desc="batches", smoothing=0.1)

    def enqueue_traj(self, traj: Trajectory):
        # Block if the ratio is too low (we are too far ahead)
        self.controller.wait_for_produce_permission()

        if self.done:
            return

        self.producer_progress.update(1)
        self.replay.add(traj)

        # Notify the controller that we have produced one trajectory
        self.controller.notify_produced(n_trajectories=1)

    def stack_batch(
        self,
        batch: list[Trajectory],
        player_transition_resolution: int = 64,
        player_history_resolution: int = 128,
    ):
        stacked_trajectory: Trajectory = jax.tree.map(
            lambda *xs: np.stack(xs, axis=1), *batch
        )

        valid = np.bitwise_not(stacked_trajectory.player_transitions.env_output.done)
        valid_sum = valid.sum(0).max().item()

        num_valid = int(
            np.ceil(valid_sum / player_transition_resolution)
            * player_transition_resolution
        )

        clipped_trajectory = Trajectory(
            builder_transitions=stacked_trajectory.builder_transitions,
            builder_history=stacked_trajectory.builder_history,
            player_transitions=jax.tree.map(
                lambda x: x[:num_valid], stacked_trajectory.player_transitions
            ),
            # builder_history=stacked_trajectory.builder_history,
            player_history=clip_history(
                stacked_trajectory.player_history, resolution=player_history_resolution
            ),
            player_hidden=stacked_trajectory.player_hidden,
        )

        return clipped_trajectory

    def host_to_device_worker(self):
        max_burst = 8
        batch_size = self.learner_config.batch_size

        while not self.done:
            taken = 0

            while not self.done and taken < max_burst:

                # Wait for permission from the controller
                # This checks both data availability AND the ratio
                self.controller.wait_for_consume_permission()

                if self.done:
                    break

                # Sample & execute one update outside the lock.
                batch = self.replay.sample(self.learner_config.batch_size)
                self.consumer_progress.update(batch_size)

                stacked = self.stack_batch(batch)
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
        historical_players = [v for k, v in league.players.items() if k != MAIN_KEY]
        win_rates = league.get_winrate((league.players[MAIN_KEY], historical_players))
        return {
            f"league_main_v_{historical_players[i].step_count}_winrate": win_rate
            for i, win_rate in enumerate(win_rates)
        }

    def ready_to_add_player(self):
        league = self.league
        latest_player = league.get_latest_player()
        steps_passed = self.player_state.actor_steps.item() - latest_player.frame_count
        historical_players = [v for k, v in league.players.items() if k != MAIN_KEY]
        win_rates = league.get_winrate((league.players[MAIN_KEY], historical_players))
        return (win_rates.min() > 0.7) & (
            steps_passed >= self.learner_config.add_player_min_frames
        )

    def train(self):
        transfer_thread = threading.Thread(target=self.host_to_device_worker)
        transfer_thread.start()

        player_train_step_jit = jax.jit(player_train_step, static_argnames=["config"])
        builder_train_step_jit = jax.jit(builder_train_step, static_argnames=["config"])

        for step_idx in range(1, self.learner_config.num_steps + 1):

            try:
                batch = self.device_q.get()
                batch: Trajectory = jax.device_put(batch)
                new_player_state: Porygon2PlayerTrainState
                new_builder_state: Porygon2BuilderTrainState

                num_steps = np.array(self.player_state.num_steps).item()

                with self.gpu_lock:
                    new_player_state, player_logs = player_train_step_jit(
                        self.player_state,
                        batch.player_transitions,
                        batch.player_history,
                        self.learner_config,
                    )

                builder_final_reward = calculate_builder_final_reward(batch)
                with self.gpu_lock:
                    new_builder_state, builder_logs = builder_train_step_jit(
                        self.builder_state,
                        batch.builder_transitions,
                        batch.builder_history,
                        batch.player_hidden,
                        builder_final_reward,
                        self.learner_config,
                    )

                # Safety check: only apply the update if the losses are finite.
                if (
                    jnp.isfinite(player_logs["player_loss"]).item()
                    and jnp.isfinite(builder_logs["builder_loss"]).item()
                ):
                    self.player_state = new_player_state
                    self.builder_state = new_builder_state
                else:
                    print("Non-finite loss detected @ step", step_idx)
                    continue

                training_logs = {"step_idx": step_idx}
                training_logs.update(
                    jax.device_get(
                        collect_batch_telemetry_data(batch, self.learner_config)
                    )
                )
                training_logs.update(jax.device_get(player_logs))
                training_logs.update(jax.device_get(builder_logs))

                if (num_steps % self.learner_config.save_interval_steps) == 0:
                    usage_logs = self.get_usage_counts(self.replay._species_counts)
                    training_logs.update(jax.device_get(usage_logs))

                if (num_steps % 100) == 0:
                    league_winrates = self.get_league_winrates()
                    training_logs.update(jax.device_get(league_winrates))

                # Update the tqdm progress bars.
                self.train_progress.update(1)
                rr = self.controller._get_current_rr()
                self.train_progress.set_description(f"batches - rr: {rr:.2f}")

                training_logs["replay_ratio"] = rr

                self.wandb_run.log(training_logs)

                new_params = ParamsContainer(
                    frame_count=self.player_state.actor_steps.item(),
                    step_count=MAIN_KEY,  # For main agent
                    player_params=self.player_state.params,
                    builder_params=self.builder_state.params,
                )
                self.league.update_main_player(new_params)

                if (num_steps % self.learner_config.save_interval_steps) == 0:
                    save_train_state(
                        self.wandb_run,
                        self.learner_config,
                        self.player_state,
                        self.builder_state,
                        self.league,
                    )

                if self.ready_to_add_player():
                    print("Adding new player to league @ step", num_steps)
                    self.league.add_player(
                        ParamsContainer(
                            frame_count=self.player_state.actor_steps.item(),
                            step_count=num_steps,
                            player_params=self.player_state.params,
                            builder_params=self.builder_state.params,
                        )
                    )

            except Exception as e:
                logger.error(f"Learner train step failed: {e}")
                traceback.print_exc()

                raise e

        self.done = True
        # transfer_thread.join()
        print("Training Finished.")
