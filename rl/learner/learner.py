import math
import queue
import threading

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb.wandb_run
from loguru import logger
from tqdm import tqdm

import wandb
from rl.environment.data import STOI
from rl.environment.interfaces import (
    BuilderTransition,
    PlayerActorInput,
    PlayerHiddenInfo,
    PlayerHistoryOutput,
    PlayerTransition,
    Trajectory,
)
from rl.environment.utils import clip_history
from rl.learner.buffer import ReplayBuffer, ReplayRatioTokenBucket
from rl.learner.config import (
    Porygon2BuilderTrainState,
    Porygon2LearnerConfig,
    Porygon2PlayerTrainState,
    save_train_state,
)
from rl.learner.returns import compute_returns
from rl.learner.utils import calculate_r2, collect_batch_telemetry_data
from rl.model.utils import Params, promote_map
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
    metagame_log_prob: jax.Array,
    continue_log_prob: jax.Array,
    continue_index: jax.Array,
    selection_log_prob: jax.Array,
    species_log_prob: jax.Array,
    packed_set_log_prob: jax.Array,
):
    return continue_log_prob + (continue_index == 0) * (
        selection_log_prob + species_log_prob + packed_set_log_prob + metagame_log_prob
    )


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
    pg_loss = ppo_objective(
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

    player_valid = jnp.bitwise_not(player_transitions.env_output.done)
    move_valid = player_valid & (actor_action_type_head.action_index == 0)
    switch_valid = player_valid & (actor_action_type_head.action_index == 1)
    wildcard_valid = move_valid & (
        player_transitions.env_output.wildcard_mask.sum(axis=-1) > 1
    )

    # my_fainted_count = player_transitions.env_output.info[
    #     ..., InfoFeature.INFO_FEATURE__MY_FAINTED_COUNT
    # ]
    # opp_fainted_count = player_transitions.env_output.info[
    #     ..., InfoFeature.INFO_FEATURE__OPP_FAINTED_COUNT
    # ]
    # phi_t = (opp_fainted_count - my_fainted_count) / MAX_RATIO_TOKEN
    # phi_tp1 = jnp.concatenate((phi_t[1:], jnp.zeros_like(phi_t[:1])), axis=0)
    # shaped_reward = phi_tp1 - phi_t

    # Accounting for revival blessing
    rewards = player_transitions.env_output.win_reward  # + (6 / 7) * shaped_reward

    # Ratio taken from IMPACT paper: https://arxiv.org/pdf/1912.00167.pdf
    player_vtrace = compute_returns(
        player_target_pred.v,
        jnp.concatenate((player_target_pred.v[1:], player_target_pred.v[-1:])),
        jnp.concatenate((rewards[1:], rewards[-1:])),
        jnp.concatenate((player_valid[1:], jnp.zeros_like(player_valid[-1:])))
        * config.player_gamma,
        target_actor_ratio,
        lambda_=config.player_lambda_,
        clip_rho_threshold=config.clip_rho_threshold,
        clip_pg_rho_threshold=config.clip_pg_rho_threshold,
    )
    player_adv_mean = average(player_vtrace.pg_advantage, player_valid)
    player_adv_std = player_vtrace.pg_advantage.std(where=player_valid)

    # Normalize by the ema mean and std of the advantages.
    player_norm_advantages = (
        player_vtrace.pg_advantage - player_state.target_adv_mean
    ) / (player_state.target_adv_std + 1e-8)

    actor_target_clipped_ratio = jnp.clip(actor_target_ratio, min=0.0, max=2.0)

    def player_loss_fn(params: Params):

        player_pred = promote_map(
            player_state.apply_fn(
                params,
                player_actor_input,
                player_transitions.agent_output.actor_output,
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
            valid=player_valid,
            clip_ppo=config.clip_ppo,
        )

        loss_v = value_loss(
            pred_v=player_pred.v, target_v=player_vtrace.returns, valid=player_valid
        )

        action_type_head_entropy = average(pred_action_type_head.entropy, player_valid)
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
            valid=player_valid,
        )
        learner_target_approx_kl = forward_kl_loss(
            policy_ratio=learner_target_ratio,
            log_policy_ratio=learner_target_log_ratio,
            valid=player_valid,
        )
        loss_kl = backward_kl_loss(
            policy_ratio=learner_target_ratio,
            log_policy_ratio=learner_target_log_ratio,
            valid=player_valid,
        )

        loss = (
            config.player_policy_loss_coef * loss_pg
            + config.player_value_loss_coef * loss_v
            + config.player_kl_loss_coef * loss_kl
            - config.player_entropy_loss_coef * loss_entropy
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
                policy_ratios=ratio, valid=player_valid, clip_ppo=config.clip_ppo
            ),
            player_learner_actor_ratio=average(learner_actor_ratio, player_valid),
            player_learner_target_ratio=average(learner_target_ratio, player_valid),
            # Approx KL values
            player_learner_actor_approx_kl=learner_actor_approx_kl,
            player_learner_target_approx_kl=learner_target_approx_kl,
            # Extra stats
            player_value_function_r2=calculate_r2(
                value_prediction=player_pred.v,
                value_target=player_vtrace.returns,
                mask=player_valid,
            ),
        )

    player_grad_fn = jax.value_and_grad(player_loss_fn, has_aux=True)
    (player_loss_val, player_logs), player_grads = player_grad_fn(player_state.params)

    player_logs.update(
        dict(
            player_loss=player_loss_val,
            player_param_norm=optax.global_norm(player_state.params),
            player_gradient_norm=optax.global_norm(player_grads),
            player_adv_mean=player_adv_mean,
            player_adv_std=player_adv_std,
            player_is_ratio=average(actor_target_clipped_ratio, player_valid),
            player_norm_adv_mean=average(player_norm_advantages, player_valid),
            player_norm_adv_std=player_norm_advantages.std(where=player_valid),
            player_value_target_mean=average(player_vtrace.returns, player_valid),
            player_value_target_std=player_vtrace.returns.std(where=player_valid),
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
        actor_steps=player_state.actor_steps + player_valid.sum(),
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
    player_hidden: PlayerHiddenInfo,
    final_reward: jax.Array,
    config: Porygon2LearnerConfig,
):
    """Train for a single step."""

    builder_actor_input = PlayerActorInput(
        env=builder_transitions.env_output, hidden=player_hidden
    )
    builder_target_pred = promote_map(
        builder_state.apply_fn(
            builder_state.target_params,
            builder_actor_input,
            builder_transitions.agent_output.actor_output,
        )
    )

    actor_metagame_head = builder_transitions.agent_output.actor_output.metagame_head
    actor_continue_head = builder_transitions.agent_output.actor_output.continue_head
    actor_selection_head = builder_transitions.agent_output.actor_output.selection_head
    actor_species_head = builder_transitions.agent_output.actor_output.species_head
    actor_packed_set_head = (
        builder_transitions.agent_output.actor_output.packed_set_head
    )

    target_metagame_head = builder_target_pred.metagame_head
    target_continue_head = builder_target_pred.continue_head
    target_selection_head = builder_target_pred.selection_head
    target_species_head = builder_target_pred.species_head
    target_packed_set_head = builder_target_pred.packed_set_head

    actor_builder_log_prob = calculate_builder_log_prob(
        metagame_log_prob=actor_metagame_head.log_prob,
        continue_log_prob=actor_continue_head.log_prob,
        continue_index=actor_continue_head.action_index,
        selection_log_prob=actor_selection_head.log_prob,
        species_log_prob=actor_species_head.log_prob,
        packed_set_log_prob=actor_packed_set_head.log_prob,
    )
    target_builder_log_prob = calculate_builder_log_prob(
        metagame_log_prob=target_metagame_head.log_prob,
        continue_log_prob=target_continue_head.log_prob,
        continue_index=target_continue_head.action_index,
        selection_log_prob=target_selection_head.log_prob,
        species_log_prob=target_species_head.log_prob,
        packed_set_log_prob=target_packed_set_head.log_prob,
    )

    actor_target_builder_log_ratio = actor_builder_log_prob - target_builder_log_prob
    actor_target_builder_ratio = jnp.exp(actor_target_builder_log_ratio)
    target_actor_builder_ratio = jnp.exp(-actor_target_builder_log_ratio)

    actor_target_clipped_ratio = jnp.clip(actor_target_builder_ratio, min=0.0, max=2.0)

    builder_valid = jnp.bitwise_not(builder_transitions.env_output.done)

    builder_rewards = (
        jax.nn.one_hot(builder_valid.sum(axis=0), builder_valid.shape[0], axis=0)
        * final_reward[None]
    )

    builder_vtrace = compute_returns(
        builder_target_pred.v,
        jnp.concatenate((builder_target_pred.v[1:], builder_target_pred.v[-1:])),
        jnp.concatenate((builder_rewards[1:], builder_rewards[-1:])),
        jnp.concatenate((builder_valid[1:], jnp.zeros_like(builder_valid[-1:])))
        * config.builder_gamma,
        target_actor_builder_ratio,
        lambda_=config.builder_lambda_,
        clip_rho_threshold=config.clip_rho_threshold,
        clip_pg_rho_threshold=config.clip_pg_rho_threshold,
    )

    builder_adv_mean = average(builder_vtrace.pg_advantage, builder_valid)
    builder_adv_std = builder_vtrace.pg_advantage.std(where=builder_valid)
    builder_norm_advantages = (
        builder_vtrace.pg_advantage - builder_state.target_adv_mean
    ) / (builder_state.target_adv_std + 1e-8)

    def builder_loss_fn(params: Params):
        """Builder loss function."""
        builder_pred = promote_map(
            builder_state.apply_fn(
                params,
                builder_actor_input,
                builder_transitions.agent_output.actor_output,
            )
        )

        learner_metagame_head = builder_pred.metagame_head
        learner_continue_head = builder_pred.continue_head
        learner_selection_head = builder_pred.selection_head
        learner_species_head = builder_pred.species_head
        learner_packed_set_head = builder_pred.packed_set_head

        learner_builder_log_prob = calculate_builder_log_prob(
            metagame_log_prob=learner_metagame_head.log_prob,
            continue_log_prob=learner_continue_head.log_prob,
            continue_index=learner_continue_head.action_index,
            selection_log_prob=learner_selection_head.log_prob,
            species_log_prob=learner_species_head.log_prob,
            packed_set_log_prob=learner_packed_set_head.log_prob,
        )

        learner_actor_log_ratio = learner_builder_log_prob - actor_builder_log_prob
        learner_actor_ratio = jnp.exp(learner_actor_log_ratio)

        learner_target_log_ratio = learner_builder_log_prob - target_builder_log_prob
        learner_target_ratio = jnp.exp(learner_target_log_ratio)

        ratio = actor_target_clipped_ratio * learner_actor_ratio

        # Calculate the losses.
        loss_pg = policy_gradient_loss(
            policy_ratios=ratio,
            advantages=builder_norm_advantages,
            valid=builder_valid,
            clip_ppo=config.clip_ppo,
        )

        loss_v = value_loss(
            pred_v=builder_pred.v, target_v=builder_vtrace.returns, valid=builder_valid
        )

        continue_entropy = average(learner_continue_head.entropy, builder_valid)
        metagame_entropy = average(learner_metagame_head.entropy, builder_valid)
        selection_entropy = average(learner_selection_head.entropy, builder_valid)
        species_entropy = average(learner_species_head.entropy, builder_valid)
        packed_set_entropy = average(learner_packed_set_head.entropy, builder_valid)
        loss_entropy = (
            continue_entropy
            + metagame_entropy
            + selection_entropy
            + species_entropy
            + packed_set_entropy
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
        loss_kl = backward_kl_loss(
            policy_ratio=learner_target_ratio,
            log_policy_ratio=learner_target_log_ratio,
            valid=builder_valid,
        )

        loss_info_ce = optax.softmax_cross_entropy(
            logits=builder_pred.metagame_pred_logits[1:],
            labels=builder_transitions.env_output.metagame_mask[1:].astype(jnp.float32),
        ).sum(where=builder_valid[1:]) / builder_valid[1:].sum().clip(min=1)

        loss = (
            config.builder_policy_loss_coef * loss_pg
            + config.builder_value_loss_coef * loss_v
            + config.builder_kl_loss_coef * loss_kl
            - config.builder_entropy_loss_coef * loss_entropy
            + loss_info_ce
        )

        return loss, dict(
            builder_loss_pg=loss_pg,
            builder_loss_v=loss_v,
            builder_loss_kl=loss_kl,
            builder_loss_entropy=loss_entropy,
            builder_loss_info_ce=loss_info_ce,
            # Head entropies
            builder_continue_entropy=continue_entropy,
            builder_metagame_entropy=metagame_entropy,
            builder_selection_entropy=selection_entropy,
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
                value_prediction=builder_pred.v,
                value_target=builder_vtrace.returns,
                mask=builder_valid,
            ),
        )

    builder_grad_fn = jax.value_and_grad(builder_loss_fn, has_aux=True)
    (builder_loss_val, training_logs), builder_grads = builder_grad_fn(
        builder_state.params
    )

    training_logs.update(
        dict(
            builder_loss=builder_loss_val,
            builder_param_norm=optax.global_norm(builder_state.params),
            builder_gradient_norm=optax.global_norm(builder_grads),
            builder_adv_mean=builder_adv_mean,
            builder_adv_std=builder_adv_std,
            builder_is_ratio=average(actor_target_clipped_ratio, builder_valid),
            builder_norm_adv_mean=average(builder_norm_advantages, builder_valid),
            builder_norm_adv_std=builder_norm_advantages.std(where=builder_valid),
            builder_value_target_mean=average(builder_vtrace.returns, builder_valid),
            builder_value_target_std=builder_vtrace.returns.std(where=builder_valid),
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
        actor_steps=builder_state.actor_steps + builder_valid.sum(),
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
    ):
        self.player_state = player_state
        self.builder_state = builder_state
        self.learner_config = learner_config

        self.wandb_run = wandb_run

        # Replay ratio bookkeeping.
        self.target_replay_ratio = self.learner_config.target_replay_ratio
        self.min_replay_size = 8 * self.learner_config.batch_size
        self.cv = threading.Condition()

        self.done = False
        self.replay = ReplayBuffer(self.learner_config.replay_buffer_capacity)
        self.controller = ReplayRatioTokenBucket(
            target_rr=self.learner_config.target_replay_ratio,
            capacity=int(
                self.target_replay_ratio * self.learner_config.batch_size * 64
            ),
            headroom=0.05,
        )
        self.device_q: queue.Queue[Trajectory] = queue.Queue(maxsize=1)

        self.wandb_run.log_code("rl/")
        self.wandb_run.log_code("inference/")
        self.wandb_run.log_code(
            "service/src/client/", include_fn=lambda x: x.endswith(".ts")
        )

        try:
            init_steps = player_state.num_steps.item()
        except:
            init_steps = int(player_state.num_steps)

        self.params_for_actor: tuple[int, Params, Params] = (
            init_steps,
            jax.device_get(self.player_state.params),
            jax.device_get(self.builder_state.params),
        )

        # progress bars
        self.producer_progress = tqdm(desc="producer", smoothing=0.1)
        self.consumer_progress = tqdm(desc="consumer", smoothing=0.1)
        self.train_progress = tqdm(desc="batches", smoothing=0.1)

    def enqueue_traj(self, traj):
        # Block if minting this trajectory would overflow capacity.
        with self.cv:
            self.cv.wait_for(
                lambda: self.done
                or (self.controller.tokens + self.target_replay_ratio)
                <= self.controller.capacity
            )
            if self.done:
                return
            # Mint tokens and ingest ONE trajectory (trajectory-based RR)
            self.controller.on_trajectories(1)
            self.producer_progress.update(1)
            self.replay.add(traj)
            self.cv.notify_all()  # wake learner

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
            # --- Warmup: need min replay size AND at least one env frame counted.
            with self.cv:
                self.cv.wait_for(lambda: self.done or (len(self.replay) >= batch_size))
                if self.done:
                    break

            taken = 0

            while not self.done and taken < max_burst:
                # Pre-check under the CV to sleep efficiently if we lack tokens/data.
                with self.cv:
                    ready = len(
                        self.replay
                    ) >= batch_size and self.controller.can_consume(batch_size)
                    if not ready and not self.done:
                        self.cv.wait()  # wait for more frames/tokens (enqueue_traj notifies)
                        continue
                    if self.done:
                        break

                # Sample & execute one update outside the lock.
                batch = self.replay.sample(self.learner_config.batch_size)
                stacked = self.stack_batch(batch)
                self.device_q.put(stacked)

                self.controller.consume(batch_size)
                taken += 1

                with self.cv:
                    self.cv.notify_all()

        logger.info("host_to_device_worker exiting.")

    def get_usage_counts(self, usage_counts: np.ndarray):
        usage_logs = {}

        if self.player_state.num_steps % 200 == 0:
            names = list(STOI["species"])
            counts = usage_counts / usage_counts.sum()

            table = wandb.Table(columns=["species", "usage"])
            for name, count in zip(names, counts):
                table.add_data(name, count)

            usage_logs["species_usage"] = table

        return usage_logs

    def train(self):

        batch_size = self.learner_config.batch_size

        transfer_thread = threading.Thread(target=self.host_to_device_worker)
        transfer_thread.start()

        player_train_step_jit = jax.jit(player_train_step, static_argnames=["config"])
        builder_train_step_jit = jax.jit(builder_train_step, static_argnames=["config"])

        for step_idx in range(1, self.learner_config.num_steps + 1):
            batch = self.device_q.get()
            batch: Trajectory = jax.device_put(batch)

            self.player_state, player_logs = player_train_step_jit(
                self.player_state,
                batch.player_transitions,
                batch.player_history,
                self.learner_config,
            )

            builder_final_reward = calculate_builder_final_reward(batch)
            self.builder_state, builder_logs = builder_train_step_jit(
                self.builder_state,
                batch.builder_transitions,
                batch.player_hidden,
                builder_final_reward,
                self.learner_config,
            )

            # if (
            #     player_logs["player_loss_kl"].item() <= self.learner_config.target_kl
            # ) and (
            #     builder_logs["builder_loss_kl"].item() <= self.learner_config.target_kl
            # ):
            #     self.player_state = new_player_state
            #     self.builder_state = new_builder_state
            # else:
            #     logger.info(
            #         f"Skipping params update due to high KL: player {player_logs['player_loss_kl'].item():.3f}, builder {builder_logs['builder_loss_kl'].item():.3f}"
            #     )
            #     continue

            training_logs = {"step_idx": step_idx}
            training_logs.update(jax.device_get(collect_batch_telemetry_data(batch)))
            training_logs.update(jax.device_get(player_logs))
            training_logs.update(jax.device_get(builder_logs))

            usage_logs = self.get_usage_counts(self.replay._species_counts)
            training_logs.update(jax.device_get(usage_logs))

            self.params_for_actor = (
                self.player_state.num_steps.item(),
                jax.device_get(self.player_state.params),
                jax.device_get(self.builder_state.params),
            )

            # Update the tqdm progress bars.
            self.consumer_progress.update(batch_size)
            self.train_progress.update(1)
            rr = self.consumer_progress.n / max(self.producer_progress.n, 1)
            self.train_progress.set_description(f"batches - rr: {rr:.2f}")

            training_logs["replay_ratio"] = rr

            self.wandb_run.log(training_logs)

            if (self.player_state.num_steps % 5000) == 0:
                save_train_state(
                    self.wandb_run,
                    self.learner_config,
                    self.player_state,
                    self.builder_state,
                )

        self.done = True
        # transfer_thread.join()
        print("Training Finished.")
