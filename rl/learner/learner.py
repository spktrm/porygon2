import functools
import math
import queue
import threading
import traceback
from _thread import LockType

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
    BuilderPerformance,
    BuilderTrajectory,
    PlayerActorInput,
    PlayerTrajectory,
)
from rl.environment.utils import clip_history
from rl.learner.buffer import (
    BuilderMetadata,
    BuilderReplayBuffer,
    DirectRatioLimiter,
    PlayerReplayBuffer,
)
from rl.learner.config import (
    BuilderLearnerConfig,
    BuilderTrainState,
    PlayerLearnerConfig,
    PlayerTrainState,
)
from rl.learner.utils import (
    calculate_r2,
    collect_builder_batch_telemetry_data,
    collect_player_batch_telemetry_data,
)
from rl.model.heads import HeadParams
from rl.model.utils import Params, promote_map
from rl.utils import average


def calculate_player_log_prob(
    *,
    action_log_prob: jax.Array,
    action_index: jax.Array,
    wildcard_log_prob: jax.Array,
):
    is_move = action_index < 4

    return action_log_prob + jnp.where(is_move, wildcard_log_prob, 0.0)


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


def mse_value_loss(*, pred_v: jax.Array, target_v: jax.Array, valid: jax.Array):
    mse_loss = jnp.square(pred_v - target_v)
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


def scalar_to_gaussian_dist(scalar_target: jax.Array, sigma: float = 0.5) -> jax.Array:
    """
    Projects scalar [-1, 1] into a Gaussian distribution over bins {-1, 0, 1}.
    """
    # 1. Define Bin Centers
    # Shape: [1, 3] broadcastable
    bins = jnp.array([-1.0, 0.0, 1.0])

    # 2. Compute Squared Distance from Scalar to each Bin
    # scalar: [Batch, 1] - bins: [1, 3] -> [Batch, 3]
    dist_sq = jnp.square(scalar_target[..., None] - bins)

    # 3. Softmax with Temperature (Sigma)
    # exp(-dist^2 / 2sigma^2)
    logits = -dist_sq / (2 * sigma**2)
    return jax.nn.softmax(logits, axis=-1)


def player_train_step(
    state: PlayerTrainState, batch: PlayerTrajectory, config: PlayerLearnerConfig
):
    """Train for a single step."""

    transitions = batch.transitions
    history = batch.history

    actor_input = PlayerActorInput(env=transitions.env_output, history=history)
    target_pred = promote_map(
        state.apply_fn(
            state.target_params,
            actor_input,
            transitions.agent_output.actor_output,
            HeadParams(),
        )
    )

    actor_action_head = transitions.agent_output.actor_output.action_head
    actor_wildcard_head = transitions.agent_output.actor_output.wildcard_head

    target_value_head = target_pred.value_head
    target_action_head = target_pred.action_head
    target_wildcard_head = target_pred.wildcard_head

    actor_log_prob = calculate_player_log_prob(
        action_log_prob=actor_action_head.log_prob,
        action_index=actor_action_head.action_index,
        wildcard_log_prob=actor_wildcard_head.log_prob,
    )
    target_log_prob = calculate_player_log_prob(
        action_log_prob=target_action_head.log_prob,
        action_index=target_action_head.action_index,
        wildcard_log_prob=target_wildcard_head.log_prob,
    )

    actor_target_log_ratio = actor_log_prob - target_log_prob
    actor_target_ratio = jnp.exp(actor_target_log_ratio)
    target_actor_ratio = jnp.exp(-actor_target_log_ratio)

    actor_target_clipped_ratio = jnp.clip(actor_target_ratio, min=0.0, max=2.0)

    valid = jnp.bitwise_not(transitions.env_output.done)
    rewards_tm1 = transitions.env_output.win_reward

    v_tm1 = target_value_head.expectation
    v_t = jnp.concatenate((v_tm1[1:], v_tm1[-1:]))
    rewards = shift_left_with_zeros(rewards_tm1)
    discounts = shift_left_with_zeros(valid).astype(v_t.dtype) * config.gamma

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
    target_dist = scalar_to_two_hot(target_scalar_v)

    player_adv_mean = average(vtrace.pg_advantage, valid)
    player_adv_std = vtrace.pg_advantage.std(where=valid)

    # Normalize by the ema mean and std of the advantages.
    player_norm_advantages = (vtrace.pg_advantage - state.target_adv_mean) / (
        state.target_adv_std + 1e-8
    )

    wildcard_valid = valid & (transitions.env_output.wildcard_mask.sum(axis=-1) > 1)

    def loss_fn(params: Params):

        player_pred = promote_map(
            state.apply_fn(
                params,
                actor_input,
                transitions.agent_output.actor_output,
                HeadParams(),
            )
        )

        learner_value_head = player_pred.value_head
        learner_action_head = player_pred.action_head
        learner_wildcard_head = player_pred.wildcard_head

        learner_log_prob = calculate_player_log_prob(
            action_log_prob=learner_action_head.log_prob,
            action_index=learner_action_head.action_index,
            wildcard_log_prob=learner_wildcard_head.log_prob,
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

        # Softmax cross-entropy loss for value head
        loss_v = average(
            -jnp.sum(learner_value_head.log_probs * target_dist, axis=-1), valid
        )

        action_head_entropy = average(learner_action_head.entropy, valid)
        wildcard_head_entropy = average(learner_wildcard_head.entropy, wildcard_valid)
        loss_entropy = action_head_entropy + wildcard_head_entropy

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
            config.policy_loss_coef * loss_pg
            + config.value_loss_coef * loss_v
            + config.kl_loss_coef * loss_kl
            - config.entropy_loss_coef * loss_entropy
        )

        return loss, dict(
            # Loss values
            player_loss_pg=loss_pg,
            player_loss_v=loss_v,
            player_loss_entropy=loss_entropy,
            player_loss_kl=loss_kl,
            # Per head entropies
            player_action_entropy=action_head_entropy,
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
                value_prediction=learner_value_head.expectation,
                value_target=target_scalar_v,
                mask=valid,
            ),
        )

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, logs), grads = grad_fn(state.params)

    logs.update(
        dict(
            player_loss=loss_val,
            player_param_norm=optax.global_norm(state.params),
            player_gradient_norm=optax.global_norm(grads),
            player_adv_mean=player_adv_mean,
            player_adv_std=player_adv_std,
            player_is_ratio=average(actor_target_clipped_ratio, valid),
            player_norm_adv_mean=average(player_norm_advantages, valid),
            player_norm_adv_std=player_norm_advantages.std(where=valid),
            player_value_target_mean=average(target_scalar_v, valid),
            player_value_target_std=target_scalar_v.std(where=valid),
        )
    )

    state = state.apply_gradients(grads=grads)
    state = state.replace(
        # Update target params and adv mean/std.
        target_params=optax.incremental_update(
            state.params, state.target_params, config.ema_decay
        ),
        target_adv_mean=state.target_adv_mean * (1 - config.ema_decay)
        + player_adv_mean * config.ema_decay,
        target_adv_std=state.target_adv_std * (1 - config.ema_decay)
        + player_adv_std * config.ema_decay,
        step_count=state.step_count + 1,
        frame_count=state.frame_count + valid.sum(),
    )

    training_logs = dict(
        player_frame_count=state.frame_count,
        player_step_count=state.step_count,
    )

    training_logs.update(logs)

    return state, training_logs


def builder_train_step(
    state: BuilderTrainState, batch: BuilderTrajectory, config: BuilderLearnerConfig
):
    """Train for a single step."""

    transitions = batch.transitions
    history = batch.history

    actor_input = BuilderActorInput(env=transitions.env_output, history=history)
    builder_target_pred = promote_map(
        state.apply_fn(
            state.target_params,
            actor_input,
            transitions.agent_output.actor_output,
            HeadParams(),
        )
    )

    actor_species_head = transitions.agent_output.actor_output.species_head
    actor_packed_set_head = transitions.agent_output.actor_output.packed_set_head

    target_value_head = builder_target_pred.value_head
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

    valid = jnp.bitwise_not(transitions.env_output.done)

    winrates = batch.performance.winrate.reshape(-1)
    winrate_mean = winrates.mean()
    winrate_std = winrates.std()
    norm_winrates = (winrates - state.winrate_mean) / (state.winrate_std + 1e-8)

    rewards_tm1 = (
        jax.nn.one_hot(valid.sum(axis=0), valid.shape[0], axis=0) * norm_winrates[None]
    )

    v_tm1 = target_value_head.logits
    v_t = jnp.concatenate((v_tm1[1:], v_tm1[-1:]))
    rewards_t = shift_left_with_zeros(rewards_tm1)
    discounts = shift_left_with_zeros(valid).astype(v_t.dtype) * config.gamma

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
        v_tm1, v_t, rewards_t, discounts, target_actor_ratio
    )

    target_scalar_v = vtrace.errors + v_tm1

    adv_mean = average(vtrace.pg_advantage, valid)
    adv_std = vtrace.pg_advantage.std(where=valid)

    # Normalize by the ema mean and std of the advantages.
    norm_advantages = (vtrace.pg_advantage - state.target_adv_mean) / (
        state.target_adv_std + 1e-8
    )

    def loss_fn(params: Params):

        builder_pred = promote_map(
            state.apply_fn(
                params,
                actor_input,
                transitions.agent_output.actor_output,
                HeadParams(),
            )
        )

        learner_value_head = builder_pred.value_head
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
            advantages=norm_advantages,
            valid=valid,
            clip_ppo=config.clip_ppo,
        )

        # Softmax cross-entropy loss for value head
        loss_v = mse_value_loss(
            pred_v=learner_value_head.logits,
            target_v=target_scalar_v,
            valid=valid,
        )

        species_entropy = average(learner_species_head.entropy, valid)
        packed_set_entropy = average(learner_packed_set_head.entropy, valid)
        loss_entropy = -average(learner_log_prob, valid)  # Estimator for entropy

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
            config.policy_loss_coef * loss_pg
            + config.value_loss_coef * loss_v
            + config.kl_loss_loss_coef * loss_kl
            - config.entropy_loss_coef * loss_entropy
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
                value_prediction=learner_value_head.logits,
                value_target=target_scalar_v,
                mask=valid,
            ),
        )

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, training_logs), grads = grad_fn(state.params)

    training_logs.update(
        dict(
            builder_loss=loss_val,
            builder_param_norm=optax.global_norm(state.params),
            builder_gradient_norm=optax.global_norm(grads),
            builder_adv_mean=adv_mean,
            builder_adv_std=adv_std,
            builder_is_ratio=average(actor_target_clipped_ratio, valid),
            builder_norm_adv_mean=state.target_adv_mean,
            builder_norm_adv_std=state.target_adv_std,
            builder_value_target_mean=average(target_scalar_v, valid),
            builder_value_target_std=target_scalar_v.std(where=valid),
            builder_winrate_mean=winrate_mean,
            builder_winrate_std=winrate_std,
            builder_norm_winrate_mean=state.winrate_mean,
            builder_norm_winrate_std=state.winrate_std,
        )
    )

    state = state.apply_gradients(grads=grads)
    state = state.replace(
        # Update target params.
        target_params=optax.incremental_update(
            state.params, state.target_params, config.ema_decay
        ),
        target_adv_mean=state.target_adv_mean * (1 - config.ema_decay)
        + adv_mean * config.ema_decay,
        target_adv_std=state.target_adv_std * (1 - config.ema_decay)
        + adv_std * config.ema_decay,
        winrate_mean=state.winrate_mean * (1 - config.ema_decay)
        + winrate_mean * config.ema_decay,
        winrate_std=state.winrate_std * (1 - config.ema_decay)
        + winrate_std * config.ema_decay,
        step_count=state.step_count + 1,
        frame_count=state.frame_count + valid.sum(),
    )

    training_logs.update(
        dict(
            builder_frame_count=state.frame_count,
            builder_step_count=state.step_count,
        )
    )

    return state, training_logs


class BuilderLearner:
    def __init__(
        self,
        state: BuilderTrainState,
        config: BuilderLearnerConfig,
        wandb_run: wandb.wandb_run.Run,
        gpu_lock: LockType,
    ):
        self._state = state
        self._config = config
        self._wandb_run = wandb_run
        self._gpu_lock = gpu_lock

        self._done = False
        self._replay_buffer = BuilderReplayBuffer(self._config.replay_buffer_capacity)

        self._ratio_controller = DirectRatioLimiter(
            target_rr=self._config.target_replay_ratio,
            batch_size=self._config.batch_size,
            warmup_trajectories=self._config.batch_size * 16,
        )
        self._ratio_controller.set_replay_buffer_len_fn(
            lambda: len(self._replay_buffer)
        )

        self._device_q: queue.Queue[BuilderTrajectory] = queue.Queue(maxsize=1)

        # progress bars
        self._producer_progress = tqdm(desc="builder producer", smoothing=0.1)
        self._consumer_progress = tqdm(desc="builder consumer", smoothing=0.1)
        self._train_progress = tqdm(desc="builder batches", smoothing=0.1)

    def enqueue_traj(self, traj: BuilderTrajectory):
        # Block if the ratio is too low (we are too far ahead)
        self._ratio_controller.wait_for_produce_permission()

        if self._done:
            return

        self._producer_progress.update(1)
        self._replay_buffer.add(traj)

        # Notify the controller that we have produced one trajectory
        self._ratio_controller.notify_produced(n_trajectories=1)

    def stack_batch(
        self, batch: list[BuilderTrajectory], metadatas: list[BuilderMetadata]
    ) -> BuilderTrajectory:

        stacked_trajectory: BuilderTrajectory = jax.tree.map(
            lambda *xs: np.stack(xs, axis=1), *batch
        )
        stacked_metadata: BuilderMetadata = jax.tree.map(
            lambda *xs: np.stack(xs, axis=1), *metadatas
        )

        return BuilderTrajectory(
            transitions=stacked_trajectory.transitions,
            history=stacked_trajectory.history,
            performance=BuilderPerformance(
                n_sampled=stacked_metadata.n_sampled,
                winrate=stacked_metadata.avg_reward,
            ),
        )

    def host_to_device_worker(self):
        max_burst = 8
        batch_size = self._config.batch_size

        while not self._done:
            taken = 0

            while not self._done and taken < max_burst:

                # Wait for permission from the controller
                # This checks both data availability AND the ratio
                self._ratio_controller.wait_for_consume_permission()

                if self._done:
                    break

                # Sample & execute one update outside the lock.
                batch, metadatas = self._replay_buffer.sample_for_learner(
                    self._config.batch_size
                )
                self._consumer_progress.update(batch_size)

                stacked = self.stack_batch(batch, metadatas)
                self._device_q.put(stacked)

                # Notify the controller that we have consumed a batch
                self._ratio_controller.notify_consumed(n_trajectories=batch_size)

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

    def train(self):
        transfer_thread = threading.Thread(target=self.host_to_device_worker)
        transfer_thread.start()

        train_step_jit = jax.jit(builder_train_step, static_argnames=["config"])

        for _ in range(self._config.num_steps):

            try:
                batch = self._device_q.get()
                batch: BuilderTrajectory = jax.device_put(batch)

                new_state: BuilderTrainState

                step = np.array(self._state.step_count).item()
                training_logs = {"builder_steps": step}

                with self._gpu_lock:
                    new_state, logs = train_step_jit(self._state, batch, self._config)

                if jnp.isfinite(logs["builder_loss"]).item():
                    self._state = new_state
                else:
                    print(
                        "Non-finite loss detected in builder @ step",
                        training_logs["builder_steps"],
                    )
                    continue

                training_logs.update(
                    jax.device_get(
                        collect_builder_batch_telemetry_data(batch, self._config)
                    )
                )
                training_logs.update(jax.device_get(logs))

                if (step % self._config.usage_count_interval) == 0:
                    usage_logs = self.get_usage_counts(
                        self._replay_buffer._species_counts
                    )
                    training_logs.update(jax.device_get(usage_logs))

                # Update the tqdm progress bars.
                self._train_progress.update(1)
                rr = self._ratio_controller._get_current_rr()
                self._train_progress.set_description(f"batches - rr: {rr:.2f}")

                training_logs["builder_replay_ratio"] = rr

                self._wandb_run.log(training_logs)

            except Exception as e:
                logger.error(f"Learner train step failed: {e}")
                traceback.print_exc()

                raise e

        self.done = True
        # transfer_thread.join()
        print("Training Finished.")


class PlayerLearner:
    def __init__(
        self,
        state: PlayerTrainState,
        config: PlayerLearnerConfig,
        wandb_run: wandb.wandb_run.Run,
        gpu_lock: LockType,
    ):
        self._state = state
        self._config = config
        self._wandb_run = wandb_run
        self._gpu_lock = gpu_lock

        self._done = False
        self._replay_buffer = PlayerReplayBuffer(self._config.replay_buffer_capacity)

        self._ratio_controller = DirectRatioLimiter(
            target_rr=self._config.target_replay_ratio,
            batch_size=self._config.batch_size,
            warmup_trajectories=self._config.batch_size * 16,
        )
        self._ratio_controller.set_replay_buffer_len_fn(
            lambda: len(self._replay_buffer)
        )

        self._device_q: queue.Queue[BuilderTrajectory] = queue.Queue(maxsize=1)

        # progress bars
        self._producer_progress = tqdm(desc="player producer", smoothing=0.1)
        self._consumer_progress = tqdm(desc="player consumer", smoothing=0.1)
        self._train_progress = tqdm(desc="player batches", smoothing=0.1)

    def enqueue_traj(self, traj: BuilderTrajectory):
        # Block if the ratio is too low (we are too far ahead)
        self._ratio_controller.wait_for_produce_permission()

        if self._done:
            return

        self._producer_progress.update(1)
        self._replay_buffer.add(traj)

        # Notify the controller that we have produced one trajectory
        self._ratio_controller.notify_produced(n_trajectories=1)

    def stack_batch(
        self,
        batch: list[BuilderTrajectory],
        transition_resolution: int = 64,
        history_resolution: int = 128,
    ):
        stacked_trajectory: BuilderTrajectory = jax.tree.map(
            lambda *xs: np.stack(xs, axis=1), *batch
        )

        valid = np.bitwise_not(stacked_trajectory.transitions.env_output.done)
        valid_sum = valid.sum(0).max().item()

        num_valid = int(
            np.ceil(valid_sum / transition_resolution) * transition_resolution
        )

        clipped_trajectory = BuilderTrajectory(
            transitions=jax.tree.map(
                lambda x: x[:num_valid], stacked_trajectory.transitions
            ),
            # builder_history=stacked_trajectory.builder_history,
            history=clip_history(
                stacked_trajectory.history, resolution=history_resolution
            ),
        )

        return clipped_trajectory

    def host_to_device_worker(self):
        max_burst = 8
        batch_size = self._config.batch_size

        while not self._done:
            taken = 0

            while not self._done and taken < max_burst:

                # Wait for permission from the controller
                # This checks both data availability AND the ratio
                self._ratio_controller.wait_for_consume_permission()

                if self._done:
                    break

                # Sample & execute one update outside the lock.
                batch = self._replay_buffer.sample(self._config.batch_size)
                self._consumer_progress.update(batch_size)

                stacked = self.stack_batch(batch)
                self._device_q.put(stacked)

                # Notify the controller that we have consumed a batch
                self._ratio_controller.notify_consumed(n_trajectories=batch_size)

                taken += 1

        logger.info("host_to_device_worker exiting.")

    def train(self):
        transfer_thread = threading.Thread(target=self.host_to_device_worker)
        transfer_thread.start()

        train_step_jit = jax.jit(player_train_step, static_argnames=["config"])

        for _ in range(self._config.num_steps):

            try:
                batch = self._device_q.get()
                batch: PlayerTrajectory = jax.device_put(batch)

                new_player_state: PlayerTrainState

                player_step = np.array(self._state.step_count).item()
                training_logs = {"player_steps": player_step}

                with self._gpu_lock:
                    new_player_state, player_logs = train_step_jit(
                        self._state, batch, self._config
                    )
                if jnp.isfinite(player_logs["player_loss"]).item():
                    self._state = new_player_state
                else:
                    print(
                        "Non-finite loss detected in player @ step",
                        training_logs["player_steps"],
                    )
                    continue

                training_logs.update(
                    jax.device_get(
                        collect_player_batch_telemetry_data(batch, self._config)
                    )
                )
                training_logs.update(jax.device_get(player_logs))

                # Update the tqdm progress bars.
                self._train_progress.update(1)
                rr = self._ratio_controller._get_current_rr()
                self._train_progress.set_description(f"batches - rr: {rr:.2f}")

                training_logs["player_replay_ratio"] = rr

                self._wandb_run.log(training_logs)

            except Exception as e:
                logger.error(f"Learner train step failed: {e}")
                traceback.print_exc()

                raise e

        self._done = True
        # transfer_thread.join()
        print("Training Finished.")
