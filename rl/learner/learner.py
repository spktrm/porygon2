import functools
import threading
import time
import traceback

import jax
import jax.numpy as jnp
import optax
import wandb.wandb_run
from tqdm import tqdm

import wandb
from rl.environment.interfaces import Transition
from rl.learner.buffer import ReplayBuffer, ReplayRatioController
from rl.learner.config import (
    Porygon2BuilderTrainState,
    Porygon2LearnerConfig,
    Porygon2PlayerTrainState,
    save_train_state,
)
from rl.learner.returns import compute_returns
from rl.learner.utils import calculate_r2, collect_batch_telemetry_data
from rl.model.utils import Params


def get_action_value(arr: jax.Array, action: jax.Array):
    """Get the value of an action from an array."""
    return jnp.take_along_axis(arr, action[..., None], axis=-1).squeeze(-1)


@functools.partial(jax.jit, static_argnames=["config"])
def train_step(
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
    batch: Transition,
    config: Porygon2LearnerConfig,
):
    """Train for a single step."""

    target_pred = player_state.apply_fn(player_state.target_params, batch.timestep)
    target_log_pi = get_action_value(target_pred.log_pi, batch.actor_step.action)
    actor_log_pi = get_action_value(
        batch.actor_step.model_output.log_pi, batch.actor_step.action
    )

    actor_target_log_ratio = actor_log_pi - target_log_pi
    actor_target_ratio = jnp.exp(actor_target_log_ratio)

    valid = jnp.bitwise_not(batch.timestep.env.done)
    rewards = batch.timestep.env.win_reward

    # Ratio taken from IMPACT paper: https://arxiv.org/pdf/1912.00167.pdf
    vtrace = compute_returns(
        target_pred.v,
        jnp.concatenate((target_pred.v[1:], target_pred.v[-1:])),
        jnp.concatenate((rewards[1:], rewards[-1:])),
        jnp.concatenate((valid[1:], jnp.zeros_like(valid[-1:]))) * config.gamma,
        actor_target_ratio,
        lambda_=config.lambda_,
        clip_rho_threshold=config.clip_rho_threshold,
        clip_pg_rho_threshold=config.clip_pg_rho_threshold,
    )
    player_adv_mean = vtrace.pg_advantage.mean(where=valid)
    player_adv_std = vtrace.pg_advantage.std(where=valid)

    # Normalize by the ema mean and std of the advantages.
    player_advantages = (vtrace.pg_advantage - player_state.target_adv_mean) / (
        player_state.target_adv_std + 1e-8
    )

    player_is_ratio = jnp.clip(actor_target_ratio, min=0.0, max=2.0)

    # Update entropy schedule coefficient.
    ent_kl_coef_mult = jnp.sqrt(config.num_steps / (player_state.actor_steps + 1000))

    def player_loss_fn(params: Params):

        pred = player_state.apply_fn(params, batch.timestep)

        learner_log_pi = get_action_value(pred.log_pi, batch.actor_step.action)

        # Calculate the log ratios.
        learner_actor_log_ratio = learner_log_pi - actor_log_pi
        learner_actor_ratio = jnp.exp(learner_actor_log_ratio)

        learner_target_log_ratio = learner_log_pi - target_log_pi
        learner_target_ratio = jnp.exp(learner_target_log_ratio)

        # Calculate the policy gradient loss.
        # Objective taken from IMPACT paper: https://arxiv.org/pdf/1912.00167.pdf
        learner_actor_ratio_is = player_is_ratio * learner_actor_ratio

        pg_loss1 = player_advantages * learner_actor_ratio_is
        pg_loss2 = player_advantages * jnp.clip(
            learner_actor_ratio_is, min=1 - config.clip_ppo, max=1 + config.clip_ppo
        )
        pg_loss = jnp.minimum(pg_loss1, pg_loss2)
        loss_pg = -pg_loss.mean(where=valid)

        # Calculate the value loss.
        loss_v = 0.5 * jnp.square(pred.v - vtrace.returns).mean(where=valid)

        # Calculate the entropy loss.
        loss_entropy = -(pred.pi * pred.log_pi).sum(axis=-1).mean(where=valid)

        # Calculate the Backward KL loss.
        # Taken from the MMD paper: https://arxiv.org/pdf/2206.05825
        # as well as: https://arxiv.org/pdf/2502.08938
        backward_kl_approx = learner_target_ratio * learner_target_log_ratio - (
            learner_target_ratio - 1
        )
        loss_kl = backward_kl_approx.mean(where=valid)

        loss = (
            loss_pg
            + config.value_loss_coef * loss_v
            + ent_kl_coef_mult
            * (config.kl_loss_coef * loss_kl - config.entropy_loss_coef * loss_entropy)
        )
        learner_actor_approx_kl = (-learner_actor_log_ratio).mean(where=valid)
        learner_target_approx_kl = (-learner_target_log_ratio).mean(where=valid)

        return loss, dict(
            # Loss values
            loss_pg=loss_pg,
            loss_v=loss_v,
            loss_entropy=loss_entropy,
            loss_kl=loss_kl,
            # Ratios
            learner_actor_ratio=learner_actor_ratio.mean(where=valid),
            learner_target_ratio=learner_target_ratio.mean(where=valid),
            # Approx KL values
            learner_actor_approx_kl=learner_actor_approx_kl,
            learner_target_approx_kl=learner_target_approx_kl,
            # Extra stats
            ent_kl_coef_mult=ent_kl_coef_mult,
            value_function_r2=calculate_r2(
                value_prediction=pred.v, value_target=vtrace.returns, mask=valid
            ),
        )

    actor_builder_keys = batch.actor_reset.key
    actor_builder_log_pi = batch.actor_reset.log_pi
    actor_builder_tokens = batch.actor_reset.tokens
    builder_apply_fn = jax.vmap(
        builder_state.apply_fn, in_axes=(None, 1, 1), out_axes=1
    )

    target_builder_output = builder_apply_fn(
        builder_state.target_params, actor_builder_keys, actor_builder_tokens
    )

    actor_target_builder_log_ratio = actor_builder_log_pi - target_builder_output.log_pi
    actor_target_builder_ratio = jnp.exp(actor_target_builder_log_ratio)

    builder_is_ratio = jnp.clip(actor_target_builder_ratio, min=0.0, max=2.0)

    final_reward = batch.timestep.env.win_reward[-1]
    builder_rewards = jnp.concatenate(
        (jnp.zeros_like(builder_is_ratio[:-1]), final_reward[None])
    )
    builder_vtrace = compute_returns(
        target_builder_output.v,
        jnp.concatenate((target_builder_output.v[1:], target_builder_output.v[-1:])),
        builder_rewards,
        jnp.concatenate(
            (
                jnp.ones_like(builder_is_ratio[:-1], dtype=jnp.bool),
                jnp.zeros_like(final_reward[None], dtype=jnp.bool),
            )
        )
        * config.gamma,
        actor_target_builder_ratio,
        lambda_=config.lambda_,
        clip_rho_threshold=config.clip_rho_threshold,
        clip_pg_rho_threshold=config.clip_pg_rho_threshold,
    )
    builder_valids = jnp.ones_like(builder_rewards)

    builder_adv_mean = builder_vtrace.pg_advantage.mean(where=builder_valids)
    builder_adv_std = builder_vtrace.pg_advantage.std(where=builder_valids)
    builder_norm_advantages = (
        builder_vtrace.pg_advantage - builder_state.target_adv_mean
    ) / (builder_state.target_adv_std + 1e-8)

    def builder_loss_fn(params: Params):
        """Builder loss function."""
        learner_builder_output = builder_apply_fn(
            params, actor_builder_keys, actor_builder_tokens
        )

        learner_actor_log_ratio = learner_builder_output.log_pi - actor_builder_log_pi
        learner_actor_ratio = jnp.exp(learner_actor_log_ratio)

        learner_target_log_ratio = (
            learner_builder_output.log_pi - target_builder_output.log_pi
        )
        learner_target_ratio = jnp.exp(learner_target_log_ratio)

        # Calculate the policy gradient loss.
        # Objective taken from IMPACT paper: https://arxiv.org/pdf/1912.00167.pdf
        learner_actor_ratio_is = builder_is_ratio * learner_actor_ratio

        pg_loss1 = builder_norm_advantages * learner_actor_ratio_is
        pg_loss2 = builder_norm_advantages * jnp.clip(
            learner_actor_ratio_is, min=1 - config.clip_ppo, max=1 + config.clip_ppo
        )
        pg_loss = jnp.minimum(pg_loss1, pg_loss2)
        loss_pg = -pg_loss.mean(where=builder_valids)

        pred_v = learner_builder_output.v
        loss_v = 0.5 * jnp.square(pred_v - builder_vtrace.returns).mean(
            where=builder_valids
        )

        loss_entropy = learner_builder_output.entropy.mean(where=builder_valids)

        backward_kl_approx = learner_target_ratio * learner_target_log_ratio - (
            learner_target_ratio - 1
        )
        loss_kl = backward_kl_approx.mean(where=builder_valids)

        builder_loss = (
            loss_pg
            + config.value_loss_coef * loss_v
            + ent_kl_coef_mult
            * (config.kl_loss_coef * loss_kl - config.entropy_loss_coef * loss_entropy)
        )

        return builder_loss, dict(
            builder_loss_pg=loss_pg,
            builder_loss_v=loss_v,
            builder_loss_kl=loss_kl,
            builder_loss_entropy=loss_entropy,
            builder_value_function_r2=calculate_r2(
                value_prediction=pred_v,
                value_target=builder_vtrace.returns,
                mask=builder_valids,
            ),
        )

    player_grad_fn = jax.value_and_grad(player_loss_fn, has_aux=True)
    (player_loss_val, player_logs), player_grads = player_grad_fn(player_state.params)

    player_logs.update(
        dict(
            loss=player_loss_val,
            param_norm=optax.global_norm(player_state.params),
            gradient_norm=optax.global_norm(player_grads),
            adv_mean=player_adv_mean,
            adv_std=player_adv_std,
            is_ratio=player_is_ratio.mean(where=valid),
            norm_adv_mean=player_advantages.mean(where=valid),
            norm_adv_std=player_advantages.std(where=valid),
            value_target_mean=vtrace.returns.mean(where=valid),
            value_target_std=vtrace.returns.std(where=valid),
            Step=player_state.num_steps,
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

    player_logs.update(dict(actor_steps=player_state.actor_steps))
    player_logs.update(collect_batch_telemetry_data(batch))

    builder_grad_fn = jax.value_and_grad(builder_loss_fn, has_aux=True)
    (builder_loss_val, builder_logs), builder_grads = builder_grad_fn(
        builder_state.params
    )

    player_logs.update(builder_logs)
    player_logs.update(
        dict(
            builder_loss_val=builder_loss_val,
            builder_param_norm=optax.global_norm(builder_state.params),
            builder_gradient_norm=optax.global_norm(builder_grads),
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
    )

    return player_state, builder_state, player_logs


class Learner:
    def __init__(
        self,
        player_state: Porygon2PlayerTrainState,
        builder_state: Porygon2BuilderTrainState,
        learner_config: Porygon2LearnerConfig,
        replay_buffer: ReplayBuffer,
        controller: ReplayRatioController,
        wandb_run: wandb.wandb_run.Run,
        gpu_lock: threading.Lock,
        num_samples: list[int],
    ):
        self.player_state = player_state
        self.builder_state = builder_state
        self.learner_config = learner_config
        self.replay_buffer = replay_buffer
        self.controller = controller
        self.wandb_run = wandb_run
        self.gpu_lock = gpu_lock
        self.num_samples = num_samples

        self.update_params_for_actor()

    def update_params_for_actor(self):
        """Updates the parameters for the actor."""
        self.params_for_actor = (
            int(self.player_state.num_steps),
            jax.device_get(self.player_state.params),
            jax.device_get(self.builder_state.params),
        )

    def train(self):
        consumer_progress = tqdm(desc="consumer", smoothing=0.1)
        train_progress = tqdm(desc="batches", smoothing=0.1)
        batch_size = self.learner_config.batch_size
        last_oom = time.time()

        for _ in range(self.learner_config.num_steps):
            try:
                self.controller.learner_wait()

                batch = self.replay_buffer.sample(batch_size)
                with self.gpu_lock:
                    self.player_state, self.builder_state, logs = train_step(
                        self.player_state,
                        self.builder_state,
                        batch,
                        self.learner_config,
                    )

                self.update_params_for_actor()
                self.wandb_run.log(logs)

                # Update the tqdm progress bars.
                consumer_progress.update(batch_size)
                train_progress.update(1)
                self.num_samples[0] += batch_size

                self.controller.signal_actors()

                if self.player_state.num_steps % 5000 == 0:
                    save_train_state(self.player_state, self.builder_state)

            except Exception as e:
                traceback.print_exc()
                if "RESOURCE_EXHAUSTED" in str(e):
                    batch_size = max(2, batch_size // 2)
                    print(
                        f"Resource exhausted, reducing batch size to {batch_size} and retrying."
                    )
                    last_oom = time.time()
                else:
                    raise e
            else:
                # If no OOM for 60 minutes, double the batch size
                if time.time() - last_oom > 60 * 60:
                    new_batch_size = min(self.learner_config.batch_size, 2 * batch_size)
                    if new_batch_size != batch_size:
                        batch_size = new_batch_size
                        print(
                            f"No OOM for 60 minutes, doubling batch size to {batch_size}."
                        )
                    last_oom = time.time()
