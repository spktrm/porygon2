import logging
import queue
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
from rl.environment.data import CAT_VF_SUPPORT, STOI, PackedSetFeature
from rl.environment.interfaces import BuilderActorInput, PlayerActorInput, Trajectory
from rl.environment.utils import clip_history, clip_packed_history
from rl.learner.buffer import BuilderTrajectoryStore, PlayerTrajectoryStore
from rl.learner.config import (
    Porygon2BuilderTrainState,
    Porygon2LearnerConfig,
    Porygon2PlayerTrainState,
    save_train_state,
)
from rl.learner.league import MAIN_KEY, League
from rl.learner.loss import (
    backward_kl_loss,
    clip_fraction,
    forward_kl_loss,
    mse_value_loss,
    policy_gradient_loss,
)
from rl.learner.targets import compute_builder_targets, compute_player_targets
from rl.learner.utils import calculate_r2, collect_batch_telemetry_data, promote_map
from rl.model.heads import HeadParams
from rl.model.utils import Params, ParamsContainer
from rl.utils import average

logger = logging.getLogger(__name__)


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

    player_target_pred = player_state.apply_fn(
        player_state.target_params,
        player_actor_input,
        player_transitions.agent_output.actor_output,
        HeadParams(),
    )

    player_actor_action_head = player_transitions.agent_output.actor_output.action_head

    # Calculate importance sampling ratios for off-policy correction.
    player_actor_log_prob = player_actor_action_head.log_prob
    player_target_log_prob = player_target_pred.action_head.log_prob
    player_actor_target_log_ratio = player_actor_log_prob - player_target_log_prob
    player_actor_target_ratio = jnp.exp(player_actor_target_log_ratio)
    player_target_actor_ratio = jnp.exp(-player_actor_target_log_ratio)
    player_actor_target_clipped_ratio = jnp.clip(
        player_actor_target_ratio, min=0.0, max=2.0
    )

    float_dtype = player_actor_log_prob.dtype

    cat_vf_support = jnp.asarray(CAT_VF_SUPPORT, dtype=float_dtype)

    # --- Player ---
    # Compute targets inside train_step (JAX/JIT compatible).
    player_targets = compute_player_targets(
        batch,
        player_target_pred,
        importance_sampling_ratios=player_target_actor_ratio,
        config=config,
    )
    player_returns = promote_map(player_targets, float_dtype)

    player_valid = jnp.bitwise_not(player_transitions.env_output.done)

    player_advantages = (
        player_targets.win_advantages
        + config.player_entropy_advantage_scale * player_targets.ent_advantages
        + config.player_potential_advantage_scale * player_targets.potential_advantages
    )

    action_mask = player_transitions.env_output.action_mask
    action_mask_flat = jax.lax.collapse(action_mask, -2)
    selected_action = (
        batch.player_transitions.agent_output.actor_output.action_head.action_index
    )
    q_values = player_target_pred.value_head.expectation[..., None] + (
        jax.nn.one_hot(selected_action, action_mask_flat.shape[-1], dtype=float_dtype)
        * player_advantages[..., None]
    )

    win_return_correction = player_targets.win_returns.sum(axis=-1, keepdims=True)
    player_win_returns = player_targets.win_returns / win_return_correction

    num_valid_actions = player_transitions.env_output.action_mask.sum((-2, -1))
    safe_valid_action_sum = jnp.maximum(num_valid_actions, 2)
    dynamic_threshold = 0.5 * jnp.log(
        (safe_valid_action_sum - 1)
        * (1 - config.exploration_fraction)
        / config.exploration_fraction
    )

    training_logs = {}

    def player_loss_fn(params: Params):

        learner_player_pred = player_state.apply_fn(
            params,
            player_actor_input,
            player_transitions.agent_output.actor_output,
            HeadParams(),
        )

        learner_value_head = learner_player_pred.value_head
        learner_action_head = learner_player_pred.action_head
        learner_conditional_entropy_head = learner_player_pred.conditional_entropy_head
        learner_potential_value_head = learner_player_pred.potential_value_head
        learner_log_prob = learner_action_head.log_prob

        learner_actor_log_ratio = learner_log_prob - player_actor_log_prob
        learner_actor_ratio = jnp.exp(learner_actor_log_ratio)

        learner_target_log_ratio = learner_log_prob - player_target_log_prob
        learner_target_ratio = jnp.exp(learner_target_log_ratio)

        player_policy_ratio = player_actor_target_clipped_ratio * learner_actor_ratio

        # Calculate losses.
        loss_pg = policy_gradient_loss(
            logits=learner_action_head.logits,
            policy=learner_action_head.policy,
            mask=action_mask_flat,
            policy_ratios=player_policy_ratio,
            q_values=q_values,
            valid=player_valid,
            clip_ppo=config.clip_ppo,
            threshold=dynamic_threshold,
        )

        # Softmax cross-entropy loss for value head
        loss_v = average(
            optax.softmax_cross_entropy(
                logits=learner_value_head.logits, labels=player_win_returns
            ),
            player_valid,
        )

        action_head_entropy = average(learner_action_head.entropy, player_valid)

        loss_forward_kl = forward_kl_loss(
            policy_ratio=learner_actor_ratio,
            log_policy_ratio=learner_actor_log_ratio,
            valid=player_valid,
        )
        loss_backward_kl = backward_kl_loss(
            policy_ratio=learner_actor_ratio,
            log_policy_ratio=learner_actor_log_ratio,
            valid=player_valid,
        )

        loss_conditional_entropy = mse_value_loss(
            pred=learner_conditional_entropy_head.logits,
            target=player_targets.ent_returns,
            valid=player_valid,
        )

        loss_potential_value = mse_value_loss(
            pred=learner_potential_value_head.logits,
            target=player_targets.potential_returns,
            valid=player_valid,
        )

        loss_magnet_kl = average(learner_action_head.kl_prior, valid=player_valid)

        loss = (
            config.player_policy_loss_coef * loss_pg
            + config.player_value_loss_coef * loss_v
            + config.player_kl_loss_coef * loss_backward_kl
            + config.player_entropy_loss_coef * loss_magnet_kl
            + config.player_conditional_entropy_loss_coef * loss_conditional_entropy
            + config.player_state_potential_loss_coef * loss_potential_value
        )

        return loss, dict(
            # Loss values
            player_loss_pg=loss_pg,
            player_loss_v=loss_v,
            player_loss_kl=loss_backward_kl,
            player_loss_conditional_entropy=loss_conditional_entropy,
            player_loss_potential_value=loss_potential_value,
            player_loss_magnet_kl=loss_magnet_kl,
            # Per head entropies
            player_action_entropy=action_head_entropy,
            # Ratios
            player_ratio_clip_fraction=clip_fraction(
                policy_ratios=player_policy_ratio,
                valid=player_valid,
                clip_ppo=config.clip_ppo,
            ),
            player_learner_actor_ratio=average(learner_actor_ratio, player_valid),
            player_learner_target_ratio=average(learner_target_ratio, player_valid),
            # Approx KL values
            player_learner_actor_approx_kl=loss_forward_kl,
            # Extra stats
            player_value_function_r2=calculate_r2(
                value_prediction=learner_value_head.expectation,
                value_target=player_returns.win_returns @ cat_vf_support,
                mask=player_valid,
            ),
            player_conditional_entropy_head_mean=average(
                learner_conditional_entropy_head.logits, player_valid
            ),
            player_conditional_entropy_head_std=jnp.std(
                learner_conditional_entropy_head.logits, where=player_valid
            ),
        )

    mean_abs_win = average(jnp.abs(player_targets.win_advantages), player_valid)
    mean_abs_ent = average(
        jnp.abs(config.player_entropy_advantage_scale * player_targets.ent_advantages),
        player_valid,
    )
    mean_abs_pot = average(
        jnp.abs(
            config.player_potential_advantage_scale
            * player_targets.potential_advantages
        ),
        player_valid,
    )
    total_signal_denom = mean_abs_win + mean_abs_ent + mean_abs_pot + 1e-8

    player_ent_win_adv_ratio = mean_abs_ent / total_signal_denom
    player_pot_win_adv_ratio = mean_abs_pot / total_signal_denom

    player_grad_fn = jax.value_and_grad(player_loss_fn, has_aux=True)
    (player_loss_val, player_logs), player_grads = player_grad_fn(player_state.params)
    training_logs.update(player_logs)
    training_logs.update(
        dict(
            player_loss=player_loss_val,
            player_nll_sum=(
                batch.player_transitions.agent_output.actor_output.action_head.log_prob
                * player_valid
            )
            .sum(axis=0)
            .mean(),
            player_win_return_correction=average(
                win_return_correction.reshape(player_valid.shape), player_valid
            ),
            player_param_norm=optax.global_norm(player_state.params),
            player_gradient_norm=optax.global_norm(player_grads),
            player_action_head_gradient_norm=optax.global_norm(
                player_grads["params"]["action_head"]
            ),
            player_winloss_value_head_gradient_norm=optax.global_norm(
                player_grads["params"]["winloss_head"]
            ),
            player_conditional_entropy_head_gradient_norm=optax.global_norm(
                player_grads["params"]["conditional_entropy_head"]
            ),
            player_potential_value_head_gradient_norm=optax.global_norm(
                player_grads["params"]["potential_value_head"]
            ),
            player_local_timestep_decoder_gradient_norm=optax.global_norm(
                player_grads["params"]["encoder"]["local_timestep_decoder"]
            ),
            player_history_decoder_gradient_norm=optax.global_norm(
                player_grads["params"]["encoder"]["history_decoder"]
            ),
            player_state_transformer_gradient_norm=optax.global_norm(
                player_grads["params"]["encoder"]["state_transformer"]
            ),
            player_norm_adv_mean=average(player_advantages, player_valid),
            player_norm_adv_std=player_advantages.std(where=player_valid),
            player_ent_win_adv_ratio=player_ent_win_adv_ratio,
            player_pot_win_adv_ratio=player_pot_win_adv_ratio,
        )
    )

    player_state = player_state.apply_gradients(grads=player_grads)
    player_state = player_state.replace(
        step_count=player_state.step_count + 1,
        frame_count=player_state.frame_count + player_valid.sum(),
        target_params=optax.incremental_update(
            player_state.params,
            player_state.target_params,
            config.player_ema_update_rate,
        ),
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
                clip_ppo=config.clip_ppo,
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
                learner_action_head.kl_prior, valid=builder_valid & human_valid_mask
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
                builder_ratio_clip_fraction=clip_fraction(
                    policy_ratios=learner_actor_ratio,
                    valid=builder_valid,
                    clip_ppo=config.clip_ppo,
                ),
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

        builder_mean_abs_ent = average(
            jnp.abs(builder_targets.ent_advantages), builder_valid
        )
        builder_mean_abs_win = average(
            jnp.abs(builder_targets.win_advantages), builder_valid
        )
        builder_total_signal_denom = builder_mean_abs_win + builder_mean_abs_ent + 1e-8

        builder_ent_win_adv_ratio = builder_mean_abs_ent / builder_total_signal_denom

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
                builder_ent_win_adv_ratio=builder_ent_win_adv_ratio,
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

    training_logs.update(collect_batch_telemetry_data(batch, config, player_targets))
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
    player_transition_resolution: int = 50,
    player_history_resolution: int = 128,
) -> Trajectory:
    """Stacks a list of trajectories and pads them to a fixed resolution."""
    stacked_trajectory: Trajectory = jax.tree.map(
        lambda *xs: np.stack(xs, axis=1), *batch
    )

    valid = np.bitwise_not(stacked_trajectory.player_transitions.env_output.done)
    valid_sum = valid.sum(0).max().item()

    num_valid = int(
        np.ceil(valid_sum / player_transition_resolution) * player_transition_resolution
    )

    return Trajectory(
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


class Learner:
    def __init__(
        self,
        player_state: Porygon2PlayerTrainState,
        builder_state: Porygon2BuilderTrainState,
        config: Porygon2LearnerConfig,
        wandb_run: wandb.wandb_run.Run,
        league: League,
        gpu_lock: LockType | None = None,
        debug: bool = False,
    ):
        self.player_state = player_state
        self.builder_state = builder_state
        self.config = config
        self.wandb_run = wandb_run
        self.league = league
        self.gpu_lock = gpu_lock or nullcontext()

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

        # Threading
        self.device_q: queue.Queue[Trajectory] = queue.Queue(maxsize=1)

        # Progress Bars
        self.consumer_progress = tqdm(desc="consumer", smoothing=0.1)
        self.train_progress = tqdm(desc="batches", smoothing=0.1)

        # JIT Compile
        if debug:
            self._train_step_jit = train_step
        else:
            self._train_step_jit = jax.jit(train_step, static_argnames=["config"])

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
                stacked = _stack_and_pad_batch(batch)
                self.device_q.put(stacked)

        logger.info("host_to_device_worker exiting.")

    def train(self):
        """
        High-level training loop.
        Delegates computation to _execute_model_update and I/O to _handle_periodic_tasks.
        """
        transfer_thread = threading.Thread(
            target=self.host_to_device_worker, daemon=True
        )
        transfer_thread.start()

        try:
            prev_league_check_step = 0

            for _ in range(self.config.num_steps):

                # 1. Fetch Data (Blocking)
                batch = self.device_q.get()
                batch = jax.device_put(batch)

                # 2. Update Model
                logs = self._train_step(batch)

                if logs is None:
                    continue  # Skip this step if update failed

                # 3. Logging & Checkpointing
                step = jax.device_get(logs["training_step"]).item()
                self._handle_periodic_tasks(step, logs)

                # 4. League Logic (Periodic)
                if (step - prev_league_check_step) >= 10:
                    self._manage_league(step)
                    prev_league_check_step = step

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Saving checkpoint...")
            save_train_state(
                self.wandb_run,
                self.config,
                self.player_state,
                self.builder_state,
                self.league,
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
            print("Training Finished.")

    def _train_step(self, batch: Trajectory) -> dict | None:
        """
        Runs the JAX update, verifies gradients/loss, and updates internal state.
        Returns training logs on success, or None if the step was invalid (e.g. NaN loss).
        """
        # 1. Run JAX Step (thread-safe)
        with self.gpu_lock:
            new_player_state, new_builder_state, logs = self._train_step_jit(
                self.player_state,
                self.builder_state,
                batch,
                self.config,
            )

        # 2. Validate Numerical Stability
        # Convert JAX arrays to python scalars for cheap comparison
        if self.config.check_finite_loss:
            p_loss_valid = jnp.isfinite(logs["player_loss"]).item()
            b_loss_valid = True
            if self.config.smogon_format != "randombattle":
                b_loss_valid = jnp.isfinite(logs["builder_loss"]).item()

            if not p_loss_valid or not b_loss_valid:
                step = logs["training_step"]
                logger.warning(
                    f"Skipping update: Non-finite loss detected @ step {step}"
                )
                return None

        # 3. Apply State Update
        self.player_state = new_player_state
        self.builder_state = new_builder_state

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

        self.wandb_run.log(logs)

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
            )

    def _manage_league(self, step: int):
        """Checks if a new player should be added to the league."""
        if self._should_add_new_player():
            print(f"Adding new player to league @ {step}")
            self.league.add_player(self._create_params_container(step))
            self.player_replay.reset_usage_counts()

    def _should_add_new_player(self) -> bool:
        latest = self.league.get_latest_player()
        current = self.league.get_main_player()

        # Calculate frames passed
        latest_frames = 0 if latest == current else latest.player_frame_count
        frames_passed = int(current.player_frame_count - latest_frames)

        # Basic gate: minimum frames
        if frames_passed < self.config.add_player_min_frames:
            return False

        historical_players = [
            v for k, v in self.league.players.items() if k != MAIN_KEY
        ]

        # Initial population check
        if not historical_players:
            return (
                int(self.player_state.step_count)
                > self.config.minimum_historical_player_steps
            )

        # Winrate check
        win_rates = self.league.get_winrate((current, historical_players))

        is_dominant = win_rates.min() > 0.7
        is_overdue = frames_passed >= self.config.add_player_max_frames

        return is_dominant or is_overdue

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
