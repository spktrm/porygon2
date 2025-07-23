import functools
import os
import pickle
import threading
import time
import traceback
from copy import deepcopy
from pprint import pprint
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import wandb.wandb_run
from chex import PRNGKey
from flax import core, struct
from flax.training import train_state
from tqdm import tqdm

import wandb
from rl.environment.interfaces import ModelOutput, Transition
from rl.environment.utils import get_ex_step
from rl.learner.buffer import ReplayBuffer, ReplayRatioController
from rl.learner.config import MMDConfig
from rl.learner.returns import Targets, compute_returns
from rl.learner.utils import calculate_r2, collect_batch_telemetry_data
from rl.model.utils import Params


class TrainState(train_state.TrainState):
    target_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    num_steps: int = 0
    num_samples: int = 0
    actor_steps: int = 0

    target_adv_mean: float = 0
    target_adv_std: float = 1


def create_train_state(module: nn.Module, rng: PRNGKey, config: MMDConfig):
    """Creates an initial `TrainState`."""
    ts = jax.tree.map(lambda x: x[:, 0], get_ex_step())

    params = module.init(rng, ts)
    target_params = deepcopy(params)

    tx = optax.chain(
        optax.clip_by_global_norm(config.clip_gradient),
        optax.scale_by_adam(
            b1=config.adam.b1,
            b2=config.adam.b2,
            eps=config.adam.eps,
        ),
        optax.scale_by_schedule(
            optax.linear_schedule(
                init_value=config.learning_rate,
                end_value=0.0,
                transition_steps=config.num_steps,
            )
        ),
        optax.scale(-1.0),
    )

    return TrainState.create(
        apply_fn=jax.vmap(module.apply, in_axes=(None, 1), out_axes=1),
        params=params,
        target_params=target_params,
        tx=tx,
    )


def save_train_state(state: TrainState):
    with open(os.path.abspath(f"ckpts/mmd_ckpt_{state.num_steps:08}"), "wb") as f:
        pickle.dump(
            dict(
                params=state.params,
                target_params=state.target_params,
                opt_state=state.opt_state,
                num_steps=state.num_steps,
                num_samples=state.num_samples,
                actor_steps=state.actor_steps,
                target_adv_mean=state.target_adv_mean,
                target_adv_std=state.target_adv_std,
            ),
            f,
        )


def load_train_state(state: TrainState, path: str):
    print(f"loading checkpoint from {path}")
    with open(path, "rb") as f:
        ckpt_data = pickle.load(f)

    print("Checkpoint data:")
    pprint(
        {
            k: v
            for k, v in ckpt_data.items()
            if k not in ["opt_state", "params", "target_params"]
        }
    )

    state = state.replace(
        params=ckpt_data["params"],
        target_params=ckpt_data["target_params"],
        opt_state=ckpt_data["opt_state"],
        num_steps=ckpt_data["num_steps"],
        num_samples=ckpt_data["num_samples"],
        actor_steps=ckpt_data["actor_steps"],
        target_adv_mean=ckpt_data.get("target_adv_mean", 0.0),
        target_adv_std=ckpt_data.get("target_adv_std", 1.0),
    )

    return state


@functools.partial(jax.jit, static_argnums=(3,))
def train_step(
    state: TrainState, batch: Transition, targets: Targets, config: MMDConfig
):
    """Train for a single step."""

    def loss_fn(params: Params):

        pred: ModelOutput = state.apply_fn(params, batch.timestep)

        action = jax.lax.stop_gradient(batch.actorstep.action[..., None])
        learner_log_pi = jnp.take_along_axis(pred.log_pi, action, axis=-1).squeeze(-1)
        actor_log_pi = jnp.take_along_axis(
            batch.actorstep.model_output.log_pi, action, axis=-1
        ).squeeze(-1)

        # Calculate the log ratios.
        learner_actor_log_ratio = learner_log_pi - actor_log_pi
        learner_actor_ratio = jnp.exp(learner_actor_log_ratio)

        learner_target_log_ratio = learner_log_pi - targets.target_log_pi
        learner_target_ratio = jnp.exp(learner_target_log_ratio)

        actor_target_log_ratio = actor_log_pi - targets.target_log_pi
        actor_target_ratio = jnp.exp(actor_target_log_ratio)

        valid = jnp.bitwise_not(batch.timestep.env.done)

        advantages = jax.lax.stop_gradient(targets.vtrace.pg_advantage)
        adv_mean = advantages.mean(where=valid)
        adv_std = advantages.std(where=valid)

        # Normalize by the ema mean and std of the advantages.
        advantages = (advantages - state.target_adv_mean) / (
            state.target_adv_std + 1e-8
        )

        # Calculate the policy gradient loss.
        # Objective taken from IMPACT paper: https://arxiv.org/pdf/1912.00167.pdf
        is_ratio = jnp.clip(actor_target_ratio, min=0.0, max=2.0)
        learner_actor_ratio_is = is_ratio * learner_actor_ratio

        pg_loss1 = -advantages * learner_actor_ratio_is
        pg_loss2 = -advantages * jnp.clip(
            learner_actor_ratio_is, 1 - config.clip_ppo, 1 + config.clip_ppo
        )
        pg_loss = jnp.maximum(pg_loss1, pg_loss2)
        loss_pg = pg_loss.mean(where=valid)

        # Calculate the value loss.
        pred_v = pred.v.reshape(*valid.shape)
        target_v = targets.vtrace.returns
        loss_v = 0.5 * jnp.square(pred_v - target_v).mean(where=valid)

        # Calculate the entropy loss.
        loss_entropy = -(pred.pi * pred.log_pi).sum(axis=-1).mean(where=valid)

        # Calculate the Backward KL loss.
        # Taken from the MMD paper: https://arxiv.org/pdf/2206.05825
        # as well as: https://arxiv.org/pdf/2502.08938
        backward_kl_approx = learner_target_ratio * learner_target_log_ratio - (
            learner_target_ratio - 1
        )
        loss_kl = backward_kl_approx.mean(where=valid)

        # Update entropy schedule coefficient.
        ent_kl_coef_mult = jnp.sqrt(config.num_steps / (state.actor_steps + 1000))

        loss = (
            loss_pg
            + config.value_loss_coef * loss_v
            - config.entropy_loss_coef * ent_kl_coef_mult * loss_entropy
            + config.kl_loss_coef * ent_kl_coef_mult * loss_kl
        )
        learner_actor_approx_kl = (-learner_actor_log_ratio).mean(where=valid)
        learner_target_approx_kl = (-learner_target_log_ratio).mean(where=valid)

        logs = dict(
            # Loss values
            loss_pg=loss_pg,
            loss_v=loss_v,
            loss_entropy=loss_entropy,
            loss_kl=loss_kl,
            # Ratios
            learner_actor_ratio=learner_actor_ratio.mean(where=valid),
            learner_target_ratio=learner_target_ratio.mean(where=valid),
            is_ratio=is_ratio.mean(where=valid),
            # Approx KL values
            learner_actor_approx_kl=learner_actor_approx_kl,
            learner_target_approx_kl=learner_target_approx_kl,
            # Extra stats
            ent_kl_coef_mult=ent_kl_coef_mult,
            adv_mean=adv_mean,
            adv_std=adv_std,
            norm_adv_mean=advantages.mean(where=valid),
            norm_adv_std=advantages.std(where=valid),
            value_function_r2=calculate_r2(
                value_prediction=pred_v, value_target=targets.vtrace.returns, mask=valid
            ),
        )

        return loss, logs

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, logs), grads = grad_fn(state.params)

    valid = jnp.bitwise_not(batch.timestep.env.done)

    logs.update(
        dict(
            loss=loss_val,
            param_norm=optax.global_norm(state.params),
            gradient_norm=optax.global_norm(grads),
            value_target_mean=targets.vtrace.returns.mean(where=valid),
            value_target_std=targets.vtrace.returns.std(where=valid),
            Step=state.num_steps,
        )
    )

    state = state.apply_gradients(grads=grads)
    state = state.replace(
        # Update target params and adv mean/std.
        target_params=optax.incremental_update(
            state.params, state.target_params, config.tau
        ),
        target_adv_mean=state.target_adv_mean * (1 - config.tau)
        + logs["adv_mean"] * config.tau,
        target_adv_std=state.target_adv_std * (1 - config.tau)
        + logs["adv_std"] * config.tau,
        # Update num steps sampled.
        num_steps=state.num_steps + 1,
        # Add 1 for the final step in each trajectory
        actor_steps=state.actor_steps + (valid.sum(0) + 1).sum(),
    )

    logs.update(dict(actor_steps=state.actor_steps))
    logs.update(collect_batch_telemetry_data(batch))

    return state, logs


class Learner:
    def __init__(
        self,
        state: TrainState,
        learner_config: MMDConfig,
        replay_buffer: ReplayBuffer,
        controller: ReplayRatioController,
        wandb_run: wandb.wandb_run.Run,
        gpu_lock: threading.Lock,
        num_samples: list[int],
    ):
        self.state = state
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
            int(self.state.num_steps),
            jax.device_get(self.state.params),
        )

    def train(self):
        consumer_progress = tqdm(desc="consumer", smoothing=0)
        train_progress = tqdm(desc="batches", smoothing=0)
        batch_size = self.learner_config.batch_size
        last_oom = time.time()

        for _ in range(self.learner_config.num_steps):
            try:
                self.controller.learner_wait()

                batch = self.replay_buffer.sample(batch_size)
                with self.gpu_lock:
                    targets = compute_returns(self.state, batch, self.learner_config)
                with self.gpu_lock:
                    self.state, logs = train_step(
                        self.state, batch, targets, self.learner_config
                    )

                self.update_params_for_actor()
                self.wandb_run.log(logs)

                # Update the tqdm progress bars.
                consumer_progress.update(batch_size)
                train_progress.update(1)
                self.num_samples[0] += batch_size

                self.controller.signal_actors()

                if self.state.num_steps % 5000 == 0:
                    save_train_state(self.state)

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
