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
from rl.environment.interfaces import PlayerActorInput, Trajectory
from rl.learner.buffer import ReplayBuffer, ReplayRatioController
from rl.learner.config import (
    Porygon2BuilderTrainState,
    Porygon2LearnerConfig,
    Porygon2PlayerTrainState,
    save_train_state,
)
from rl.learner.returns import compute_returns
from rl.learner.utils import calculate_r2, collect_batch_telemetry_data
from rl.model.utils import Params, legal_log_policy, legal_policy
from rl.utils import average


def get_action_value(arr: jax.Array, action: jax.Array):
    """Get the value of an action from an array."""
    return jnp.take_along_axis(arr, action[..., None], axis=-1).squeeze(-1)


def calculate_player_log_prob(
    action_type_log_pi: jax.Array,
    action_type: jax.Array,
    move_log_pi: jax.Array,
    move: jax.Array,
    wildcard_log_pi: jax.Array,
    wildcard: jax.Array,
    switch_log_pi: jax.Array,
    switch: jax.Array,
):
    action_type_log_prob = get_action_value(action_type_log_pi, action_type)

    move_log_prob = get_action_value(move_log_pi, move)
    switch_log_prob = get_action_value(switch_log_pi, switch)
    wild_card_log_prob = get_action_value(wildcard_log_pi, wildcard)

    # safe selectors (avoid 0 * -inf -> NaN)
    is_move = action_type == 0
    is_sw_or_preview = (action_type == 1) | (action_type == 2)

    return (
        action_type_log_prob
        + jnp.where(is_move, move_log_prob + wild_card_log_prob, 0.0)
        + jnp.where(is_sw_or_preview, switch_log_prob, 0.0)
    )


def calculate_builder_log_prob(
    species_log_pi: jax.Array,
    species: jax.Array,
    packed_set_log_pi: jax.Array,
    packed_set: jax.Array,
    pos: jax.Array,
):
    species_log_prob = get_action_value(species_log_pi, species)
    packed_set_log_prob = get_action_value(packed_set_log_pi, packed_set)
    return jnp.where(pos < 6, species_log_prob, packed_set_log_prob)


def policy_gradient_loss(
    policy_ratios: jax.Array, advantages: jax.Array, valid: jax.Array, clip_ppo: float
):
    """Objective taken from SPO paper: https://arxiv.org/pdf/2401.16025"""
    pg_loss = policy_ratios * advantages - (
        jnp.abs(advantages) * (1 - policy_ratios) ** 2
    ) / (2 * clip_ppo)
    return -average(pg_loss, valid)


def value_loss(pred_v: jax.Array, target_v: jax.Array, valid: jax.Array):
    mse_loss = jnp.square(pred_v - target_v)
    return 0.5 * average(mse_loss, valid)


def entropy_loss(pi: jax.Array, log_pi: jax.Array) -> jax.Array:
    return -jnp.sum(pi * log_pi, axis=-1)


def player_entropy_loss(
    action_type_log_pi: jax.Array,  # T, B, 2
    action_type_pi: jax.Array,  # T, B, 2
    move_log_pi: jax.Array,  # T, B, 4
    move_pi: jax.Array,  # T, B, 4
    wildcard_log_pis: jax.Array,  # T, B, 4, 5
    wildcard_pis: jax.Array,  # T, B, 4, 5
    switch_log_pi: jax.Array,  # T, B, 6
    switch_pi: jax.Array,  # T, B, 6
    valid: jax.Array,  # T, B
):
    action_type_entropy = entropy_loss(action_type_pi, action_type_log_pi)

    wildcard_entropy = entropy_loss(wildcard_pis, wildcard_log_pis)
    move_entropy = entropy_loss(move_pi, move_log_pi) + (
        move_pi * wildcard_entropy
    ).sum(axis=-1)
    switch_entropy = entropy_loss(switch_pi, switch_log_pi)
    sub_entropy = jnp.stack((move_entropy, switch_entropy, switch_entropy), axis=-1)

    total_entropy = action_type_entropy + (action_type_pi * sub_entropy).sum(axis=-1)
    return average(total_entropy, valid)


def builder_entropy_loss(
    species_log_pi: jax.Array,
    species_pi: jax.Array,
    packed_set_log_pi: jax.Array,
    packed_set_pi: jax.Array,
    pos: jax.Array,
    valid: jax.Array,
):
    species_entropy = entropy_loss(species_pi, species_log_pi)
    packed_set_entropy = entropy_loss(packed_set_pi, packed_set_log_pi)
    loss = jnp.where(pos < 6, species_entropy, packed_set_entropy)
    return average(loss, valid)


def backward_kl_loss(
    policy_ratio: jax.Array, log_policy_ratio: jax.Array, valid: jax.Array
):
    """
    Calculate the Backward KL loss.
    Taken from the MMD paper: https://arxiv.org/pdf/2206.05825
    as well as: https://arxiv.org/pdf/2502.08938
    """
    loss = policy_ratio * log_policy_ratio - (policy_ratio - 1)
    return average(loss, valid)


@functools.partial(jax.jit, static_argnames=["config"])
def train_step(
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
    batch: Trajectory,
    config: Porygon2LearnerConfig,
):
    """Train for a single step."""
    player_actor_input = PlayerActorInput(
        env=batch.player_transitions.env_output, history=batch.player_history
    )

    target_pred = player_state.apply_fn(player_state.target_params, player_actor_input)
    actor_log_prob = calculate_player_log_prob(
        action_type_log_pi=legal_log_policy(
            batch.player_transitions.agent_output.actor_output.action_type_logits,
            batch.player_transitions.env_output.action_type_mask,
        ),
        action_type=batch.player_transitions.agent_output.action_type,
        move_log_pi=legal_log_policy(
            batch.player_transitions.agent_output.actor_output.move_logits,
            batch.player_transitions.env_output.move_mask,
        ),
        move=batch.player_transitions.agent_output.move_slot,
        wildcard_log_pi=legal_log_policy(
            jnp.take_along_axis(
                batch.player_transitions.agent_output.actor_output.wildcard_logits,
                batch.player_transitions.agent_output.move_slot[..., None, None],
                axis=-2,
            ).squeeze(-2),
            batch.player_transitions.env_output.wildcard_mask,
        ),
        wildcard=batch.player_transitions.agent_output.wildcard_slot,
        switch_log_pi=legal_log_policy(
            batch.player_transitions.agent_output.actor_output.switch_logits,
            batch.player_transitions.env_output.switch_mask,
        ),
        switch=batch.player_transitions.agent_output.switch_slot,
    )
    target_log_prob = calculate_player_log_prob(
        action_type_log_pi=legal_log_policy(
            target_pred.action_type_logits,
            batch.player_transitions.env_output.action_type_mask,
        ),
        action_type=batch.player_transitions.agent_output.action_type,
        move_log_pi=legal_log_policy(
            target_pred.move_logits,
            batch.player_transitions.env_output.move_mask,
        ),
        move=batch.player_transitions.agent_output.move_slot,
        wildcard_log_pi=legal_log_policy(
            jnp.take_along_axis(
                target_pred.wildcard_logits,
                batch.player_transitions.agent_output.move_slot[..., None, None],
                axis=-2,
            ).squeeze(-2),
            batch.player_transitions.env_output.wildcard_mask,
        ),
        wildcard=batch.player_transitions.agent_output.wildcard_slot,
        switch_log_pi=legal_log_policy(
            target_pred.switch_logits,
            batch.player_transitions.env_output.switch_mask,
        ),
        switch=batch.player_transitions.agent_output.switch_slot,
    )

    actor_target_log_ratio = actor_log_prob - target_log_prob
    actor_target_ratio = jnp.exp(actor_target_log_ratio)

    player_valid = jnp.bitwise_not(batch.player_transitions.env_output.done)
    rewards = batch.player_transitions.env_output.win_reward

    # Ratio taken from IMPACT paper: https://arxiv.org/pdf/1912.00167.pdf
    vtrace = compute_returns(
        target_pred.v,
        jnp.concatenate((target_pred.v[1:], target_pred.v[-1:])),
        jnp.concatenate((rewards[1:], rewards[-1:])),
        jnp.concatenate((player_valid[1:], jnp.zeros_like(player_valid[-1:])))
        * config.player_gamma,
        actor_target_ratio,
        lambda_=config.player_lambda_,
        clip_rho_threshold=config.clip_rho_threshold,
        clip_pg_rho_threshold=config.clip_pg_rho_threshold,
    )
    player_adv_mean = average(vtrace.pg_advantage, player_valid)
    player_adv_std = vtrace.pg_advantage.std(where=player_valid)

    # Normalize by the ema mean and std of the advantages.
    player_norm_advantages = (vtrace.pg_advantage - player_state.target_adv_mean) / (
        player_state.target_adv_std + 1e-8
    )

    player_is_ratio = jnp.clip(actor_target_ratio, min=0.0, max=2.0)

    # Update entropy schedule coefficient.
    ent_kl_coef_mult = jnp.sqrt(config.num_steps / (player_state.actor_steps + 1000))

    def player_loss_fn(params: Params):

        pred = player_state.apply_fn(params, player_actor_input)
        pred_action_type_log_pi = legal_log_policy(
            pred.action_type_logits,
            batch.player_transitions.env_output.action_type_mask,
        )
        pred_move_log_pi = legal_log_policy(
            pred.move_logits,
            batch.player_transitions.env_output.move_mask,
        )
        pred_wildcard_log_pi = legal_log_policy(
            pred.wildcard_logits,
            batch.player_transitions.env_output.wildcard_mask[..., None, :],
        )
        pred_switch_log_pi = legal_log_policy(
            pred.switch_logits,
            batch.player_transitions.env_output.switch_mask,
        )
        pred_action_type_pi = legal_policy(
            pred.action_type_logits,
            batch.player_transitions.env_output.action_type_mask,
        )
        pred_move_pi = legal_policy(
            pred.move_logits,
            batch.player_transitions.env_output.move_mask,
        )
        pred_wildcard_pi = legal_policy(
            pred.wildcard_logits,
            batch.player_transitions.env_output.wildcard_mask[..., None, :],
        )
        pred_switch_pi = legal_policy(
            pred.switch_logits,
            batch.player_transitions.env_output.switch_mask,
        )

        learner_log_prob = calculate_player_log_prob(
            action_type_log_pi=pred_action_type_log_pi,
            action_type=batch.player_transitions.agent_output.action_type,
            move_log_pi=pred_move_log_pi,
            move=batch.player_transitions.agent_output.move_slot,
            wildcard_log_pi=jnp.take_along_axis(
                pred_wildcard_log_pi,
                batch.player_transitions.agent_output.move_slot[..., None, None],
                axis=-2,
            ).squeeze(-2),
            wildcard=batch.player_transitions.agent_output.wildcard_slot,
            switch_log_pi=pred_switch_log_pi,
            switch=batch.player_transitions.agent_output.switch_slot,
        )

        # Calculate the log ratios.
        learner_actor_log_ratio = learner_log_prob - actor_log_prob
        learner_actor_ratio = jnp.exp(learner_actor_log_ratio)

        learner_target_log_ratio = learner_log_prob - target_log_prob
        learner_target_ratio = jnp.exp(learner_target_log_ratio)

        ratio = player_is_ratio * learner_actor_ratio

        # Calculate losses.
        loss_pg = policy_gradient_loss(
            ratio, player_norm_advantages, player_valid, config.clip_ppo
        )

        loss_v = value_loss(pred.v, vtrace.returns, player_valid)

        loss_entropy = player_entropy_loss(
            pred_action_type_log_pi,
            pred_action_type_pi,
            pred_move_log_pi,
            pred_move_pi,
            pred_wildcard_log_pi,
            pred_wildcard_pi,
            pred_switch_log_pi,
            pred_switch_pi,
            player_valid,
        )

        loss_kl = backward_kl_loss(
            learner_target_ratio, learner_target_log_ratio, player_valid
        )

        loss = (
            config.player_policy_loss_coef * loss_pg
            + config.player_value_loss_coef * loss_v
            + ent_kl_coef_mult
            * (
                config.player_kl_loss_coef * loss_kl
                - config.player_entropy_loss_coef * loss_entropy
            )
        )
        learner_actor_approx_kl = average(-learner_actor_log_ratio, player_valid)
        learner_target_approx_kl = average(-learner_target_log_ratio, player_valid)

        return loss, dict(
            # Loss values
            loss_pg=loss_pg,
            loss_v=loss_v,
            loss_entropy=loss_entropy,
            loss_kl=loss_kl,
            # Ratios
            learner_actor_ratio=average(learner_actor_ratio, player_valid),
            learner_target_ratio=average(learner_target_ratio, player_valid),
            # Approx KL values
            learner_actor_approx_kl=learner_actor_approx_kl,
            learner_target_approx_kl=learner_target_approx_kl,
            # Extra stats
            ent_kl_coef_mult=ent_kl_coef_mult,
            value_function_r2=calculate_r2(
                value_prediction=pred.v, value_target=vtrace.returns, mask=player_valid
            ),
        )

    target_builder_output = builder_state.apply_fn(
        builder_state.target_params, batch.builder_transitions.env_output
    )

    target_builder_log_prob = calculate_builder_log_prob(
        legal_log_policy(
            target_builder_output.species_logits,
            batch.builder_transitions.env_output.species_mask,
        ),
        batch.builder_transitions.agent_output.species,
        legal_log_policy(
            target_builder_output.packed_set_logits,
            batch.builder_transitions.env_output.packed_set_mask,
        ),
        batch.builder_transitions.agent_output.packed_set,
        batch.builder_transitions.env_output.pos,
    )
    actor_builder_log_prob = calculate_builder_log_prob(
        legal_log_policy(
            batch.builder_transitions.agent_output.actor_output.species_logits,
            batch.builder_transitions.env_output.species_mask,
        ),
        batch.builder_transitions.agent_output.species,
        legal_log_policy(
            batch.builder_transitions.agent_output.actor_output.packed_set_logits,
            batch.builder_transitions.env_output.packed_set_mask,
        ),
        batch.builder_transitions.agent_output.packed_set,
        batch.builder_transitions.env_output.pos,
    )

    actor_target_builder_log_ratio = actor_builder_log_prob - target_builder_log_prob
    actor_target_builder_ratio = jnp.exp(actor_target_builder_log_ratio)

    builder_is_ratio = jnp.clip(actor_target_builder_ratio, min=0.0, max=2.0)

    final_reward = rewards[-1]
    builder_rewards = jnp.concatenate(
        (jnp.zeros_like(builder_is_ratio[:-1]), final_reward[None])
    )
    builder_valid = jnp.bitwise_not(batch.builder_transitions.env_output.done)
    builder_vtrace = compute_returns(
        target_builder_output.v,
        jnp.concatenate((target_builder_output.v[1:], target_builder_output.v[-1:])),
        jnp.concatenate((builder_rewards[1:], builder_rewards[-1:])),
        jnp.concatenate((builder_valid[1:], jnp.zeros_like(builder_valid[-1:])))
        * config.builder_gamma,
        actor_target_builder_ratio,
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
        learner_builder_output = builder_state.apply_fn(
            params, batch.builder_transitions.env_output
        )
        leaner_builder_species_log_pi = legal_log_policy(
            batch.builder_transitions.agent_output.actor_output.species_logits,
            batch.builder_transitions.env_output.species_mask,
        )
        leaner_builder_packed_set_log_pi = legal_log_policy(
            batch.builder_transitions.agent_output.actor_output.packed_set_logits,
            batch.builder_transitions.env_output.packed_set_mask,
        )
        leaner_builder_species_pi = legal_policy(
            batch.builder_transitions.agent_output.actor_output.species_logits,
            batch.builder_transitions.env_output.species_mask,
        )
        leaner_builder_packed_set_pi = legal_policy(
            batch.builder_transitions.agent_output.actor_output.packed_set_logits,
            batch.builder_transitions.env_output.packed_set_mask,
        )

        learner_builder_log_prob = calculate_builder_log_prob(
            leaner_builder_species_log_pi,
            batch.builder_transitions.agent_output.species,
            leaner_builder_packed_set_log_pi,
            batch.builder_transitions.agent_output.packed_set,
            batch.builder_transitions.env_output.pos,
        )

        learner_actor_log_ratio = learner_builder_log_prob - actor_builder_log_prob
        learner_actor_ratio = jnp.exp(learner_actor_log_ratio)

        learner_target_log_ratio = learner_builder_log_prob - target_builder_log_prob
        learner_target_ratio = jnp.exp(learner_target_log_ratio)

        ratio = builder_is_ratio * learner_actor_ratio

        # Calculate the losses.
        loss_pg = policy_gradient_loss(
            ratio, builder_norm_advantages, builder_valid, config.clip_ppo
        )

        loss_v = value_loss(
            learner_builder_output.v,
            builder_vtrace.returns,
            # Do this so we learn the value of final teams
            jnp.ones_like(builder_valid),
        )

        loss_entropy = builder_entropy_loss(
            leaner_builder_species_log_pi,
            leaner_builder_species_pi,
            leaner_builder_packed_set_log_pi,
            leaner_builder_packed_set_pi,
            batch.builder_transitions.env_output.pos,
            builder_valid,
        )

        loss_kl = backward_kl_loss(
            learner_target_ratio, learner_target_log_ratio, builder_valid
        )

        builder_loss = (
            config.builder_policy_loss_coef * loss_pg
            + config.builder_value_loss_coef * loss_v
            + ent_kl_coef_mult
            * (
                config.builder_kl_loss_coef * loss_kl
                - config.builder_entropy_loss_coef * loss_entropy
            )
        )

        return builder_loss, dict(
            builder_loss_pg=loss_pg,
            builder_loss_v=loss_v,
            builder_loss_kl=loss_kl,
            builder_loss_entropy=loss_entropy,
            builder_value_function_r2=calculate_r2(
                value_prediction=learner_builder_output.v,
                value_target=builder_vtrace.returns,
                mask=builder_valid,
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
            is_ratio=average(player_is_ratio, player_valid),
            norm_adv_mean=average(player_norm_advantages, player_valid),
            norm_adv_std=player_norm_advantages.std(where=player_valid),
            value_target_mean=average(vtrace.returns, player_valid),
            value_target_std=vtrace.returns.std(where=player_valid),
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
        actor_steps=player_state.actor_steps + player_valid.sum(),
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

        self.wandb_run.log_code("inference/")
        self.wandb_run.log_code(
            "service/src/client/", include_fn=lambda x: x.endswith(".ts")
        )

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
                    save_train_state(
                        self.wandb_run,
                        self.learner_config,
                        self.player_state,
                        self.builder_state,
                    )

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
