import collections
import functools
import json
import os
import pickle
import queue
import random
import threading
import time
import traceback
from copy import deepcopy
from pprint import pprint
from typing import Any, Callable, NamedTuple

import chex
import flax.linen as nn
import jax
import jax.experimental
import jax.experimental.compilation_cache
import jax.experimental.compilation_cache.compilation_cache
import jax.numpy as jnp
import numpy as np
import optax
from chex import PRNGKey
from flax import core, struct
from flax.training import train_state
from tqdm import tqdm

import wandb
from ml.arch.config import get_model_cfg
from ml.arch.model import get_model, get_num_params
from ml.learners.func import calculate_r2, collect_batch_telemetry_data
from ml.utils import Params, get_most_recent_file
from rlenv.env import clip_history, get_ex_step
from rlenv.interfaces import ModelOutput, TimeStep, Transition
from rlenv.main import Actor, Agent, SinglePlayerSyncEnvironment
from rlenv.utils import FairLock


@chex.dataclass(frozen=True)
class AdamConfig:
    """Adam optimizer related params."""

    b1: float
    b2: float
    eps: float


@chex.dataclass(frozen=True)
class MMDConfig:
    num_steps = 10_000_000
    num_actors: int = 32
    unroll_length: int = 108
    replay_buffer_capacity: int = 16

    # Batch iteration params
    batch_size: int = 4
    target_replay_ratio: int = 2

    # Learning params
    adam: AdamConfig = AdamConfig(b1=0.9, b2=0.999, eps=1e-5)
    learning_rate: float = 3e-5
    clip_gradient: float = 1
    tau: float = 1e-2

    # Vtrace params
    lambda_: float = 0.95
    gamma: float = 1.0
    clip_rho_threshold: float = 1.0
    clip_pg_rho_threshold: float = 1.0
    clip_ppo: float = 0.2

    # Loss coefficients
    value_loss_coef: float = 0.5
    policy_loss_coef: float = 1.0
    entropy_loss_coef: float = 0.05
    kl_loss_coef: float = 0.05


def get_config():
    return MMDConfig()


class TrainState(train_state.TrainState):
    target_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    num_samples: int = 0
    actor_steps: int = 0

    target_adv_mean: float = 0
    target_adv_std: float = 1


def create_train_state(module: nn.Module, rng: PRNGKey, config: MMDConfig):
    """Creates an initial `TrainState`."""
    ts = get_ex_step()
    ts = jax.tree.map(lambda x: x[:, 0], get_ex_step())

    params = module.init(rng, ts)
    target_params = deepcopy(params)

    tx = optax.chain(
        optax.clip_by_global_norm(config.clip_gradient),
        optax.adam(
            learning_rate=config.learning_rate,
            eps_root=0.0,
            b1=config.adam.b1,
            b2=config.adam.b2,
            eps=config.adam.eps,
        ),
    )

    return TrainState.create(
        apply_fn=jax.vmap(module.apply, in_axes=(None, 1), out_axes=1),
        params=params,
        target_params=target_params,
        tx=tx,
    )


def save(state: TrainState, replay_buffer: "ReplayBuffer"):
    with open(os.path.abspath(f"ckpts/mmd_ckpt_{state.step:08}"), "wb") as f:
        pickle.dump(
            dict(
                params=state.params,
                target_params=state.target_params,
                opt_state=state.opt_state,
                step=state.step,
                num_samples=state.num_samples,
                actor_steps=state.actor_steps,
                target_adv_mean=state.target_adv_mean,
                target_adv_std=state.target_adv_std,
                total_added=replay_buffer.total_added,
            ),
            f,
        )


def load(state: TrainState, replay_buffer: "ReplayBuffer", path: str):
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
        step=ckpt_data["step"],
        num_samples=ckpt_data["num_samples"],
        actor_steps=ckpt_data["actor_steps"],
        target_adv_mean=ckpt_data.get("target_adv_mean", 0.0),
        target_adv_std=ckpt_data.get("target_adv_std", 1.0),
    )

    replay_buffer._total_added = ckpt_data.get["total_added"]

    return state, replay_buffer


class VTraceOutput(NamedTuple):
    returns: jax.Array
    pg_advantage: jax.Array
    q_estimate: jax.Array


class Targets(NamedTuple):
    vtrace: VTraceOutput
    target_log_pi: jax.Array


def vtrace(
    v_tm1: jax.Array,
    v_t: jax.Array,
    r_t: jax.Array,
    discount_t: jax.Array,
    rho_tm1: jax.Array,
    lambda_: float = 1.0,
    clip_rho_threshold: float = 1.0,
    stop_target_gradients: bool = True,
) -> jax.Array:
    """Calculates V-Trace errors from importance weights.

    V-trace computes TD-errors from multistep trajectories by applying
    off-policy corrections based on clipped importance sampling ratios.

    See "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor
    Learner Architectures" by Espeholt et al. (https://arxiv.org/abs/1802.01561).

    Args:
      v_tm1: values at time t-1.
      v_t: values at time t.
      r_t: reward at time t.
      discount_t: discount at time t.
      rho_tm1: importance sampling ratios at time t-1.
      lambda_: mixing parameter; a scalar or a vector for timesteps t.
      clip_rho_threshold: clip threshold for importance weights.
      stop_target_gradients: whether or not to apply stop gradient to targets.

    Returns:
      V-Trace error.
    """
    chex.assert_rank(
        [v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_], [1, 1, 1, 1, 1, {0, 1}]
    )
    chex.assert_type(
        [v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_],
        [float, float, float, float, float, float],
    )
    chex.assert_equal_shape([v_tm1, v_t, r_t, discount_t, rho_tm1])

    # Clip importance sampling ratios.
    c_tm1 = jnp.minimum(1.0, rho_tm1) * lambda_
    clipped_rhos_tm1 = jnp.minimum(clip_rho_threshold, rho_tm1)

    # Compute the temporal difference errors.
    td_errors = clipped_rhos_tm1 * (r_t + discount_t * v_t - v_tm1)

    # Work backwards computing the td-errors.
    def _body(acc, xs):
        td_error, discount, c = xs
        acc = td_error + discount * c * acc
        return acc, acc

    _, errors = jax.lax.scan(_body, 0.0, (td_errors, discount_t, c_tm1), reverse=True)

    # Return errors, maybe disabling gradient flow through bootstrap targets.
    return jax.lax.select(
        stop_target_gradients, jax.lax.stop_gradient(errors + v_tm1) - v_tm1, errors
    )


def vtrace_td_error_and_advantage(
    v_tm1: jax.Array,
    v_t: jax.Array,
    r_t: jax.Array,
    discount_t: jax.Array,
    rho_tm1: jax.Array,
    lambda_: float = 1.0,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
    stop_target_gradients: bool = True,
) -> VTraceOutput:
    """Calculates V-Trace errors and PG advantage from importance weights.

    This functions computes the TD-errors and policy gradient Advantage terms
    as used by the IMPALA distributed actor-critic agent.

    See "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor
    Learner Architectures" by Espeholt et al. (https://arxiv.org/abs/1802.01561)

    Args:
      v_tm1: values at time t-1.
      v_t: values at time t.
      r_t: reward at time t.
      discount_t: discount at time t.
      rho_tm1: importance weights at time t-1.
      lambda_: mixing parameter; a scalar or a vector for timesteps t.
      clip_rho_threshold: clip threshold for importance ratios.
      clip_pg_rho_threshold: clip threshold for policy gradient importance ratios.
      stop_target_gradients: whether or not to apply stop gradient to targets.

    Returns:
      a tuple of V-Trace error, policy gradient advantage, and estimated Q-values.
    """
    chex.assert_rank(
        [v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_], [1, 1, 1, 1, 1, {0, 1}]
    )
    chex.assert_type(
        [v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_],
        [float, float, float, float, float, float],
    )
    chex.assert_equal_shape([v_tm1, v_t, r_t, discount_t, rho_tm1])

    # If scalar make into vector.
    lambda_ = jnp.ones_like(discount_t) * lambda_

    errors = vtrace(
        v_tm1,
        v_t,
        r_t,
        discount_t,
        rho_tm1,
        lambda_,
        clip_rho_threshold,
        stop_target_gradients,
    )
    targets_tm1 = errors + v_tm1
    q_bootstrap = jnp.concatenate(
        [
            lambda_[:-1] * targets_tm1[1:] + (1 - lambda_[:-1]) * v_tm1[1:],
            v_t[-1:],
        ],
        axis=0,
    )
    q_estimate = r_t + discount_t * q_bootstrap
    clipped_pg_rho_tm1 = jnp.minimum(clip_pg_rho_threshold, rho_tm1)
    pg_advantages = clipped_pg_rho_tm1 * (q_estimate - v_tm1)
    return VTraceOutput(
        returns=targets_tm1, pg_advantage=pg_advantages, q_estimate=q_estimate
    )


def _compute_returns(
    v_tm1: chex.Array, rho_tm1: chex.Array, batch: Transition, config: MMDConfig
):
    """Train for a single step."""

    valid = jnp.bitwise_not(batch.timestep.env.done)
    rewards = batch.timestep.env.win_reward

    rewards = jnp.concatenate((rewards[1:], rewards[-1:]))
    v_t = jnp.concatenate((v_tm1[1:], v_tm1[-1:]))
    valids = jnp.concatenate((valid[1:], jnp.zeros_like(valid[-1:])))

    discount_t = valids * config.gamma

    with jax.default_device(jax.devices("cpu")[0]):
        return jax.vmap(
            functools.partial(
                vtrace_td_error_and_advantage,
                lambda_=config.lambda_,
                clip_rho_threshold=config.clip_rho_threshold,
                clip_pg_rho_threshold=config.clip_pg_rho_threshold,
            ),
            in_axes=1,
            out_axes=1,
        )(v_tm1, v_t, rewards, discount_t, rho_tm1)


@functools.partial(jax.jit, static_argnums=(2,))
def compute_returns(state: TrainState, batch: Transition, config: MMDConfig):

    target_pred: ModelOutput = state.apply_fn(state.target_params, batch.timestep)

    valid = jnp.bitwise_not(batch.timestep.env.done)

    action = jax.lax.stop_gradient(batch.actorstep.action[..., None])
    target_log_pi = jnp.take_along_axis(target_pred.log_pi, action, axis=-1).squeeze(-1)
    log_mu = jnp.take_along_axis(
        batch.actorstep.model_output.log_pi, action, axis=-1
    ).squeeze(-1)

    # Objective taken from IMPACT paper: https://arxiv.org/pdf/1912.00167.pdf
    log_ratio = log_mu - target_log_pi
    ratio = jnp.exp(log_ratio)

    return Targets(
        vtrace=_compute_returns(
            target_pred.v.reshape(*valid.shape), ratio, batch, config
        ),
        target_log_pi=jax.lax.stop_gradient(target_log_pi),
    )


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
            Step=state.step,
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
        # Update num trajectories sampled.
        num_samples=state.num_samples + valid.shape[1],
        # Add 1 for the final step in each trajectory
        actor_steps=state.actor_steps + (valid.sum(0) + 1).sum(),
    )

    logs.update(dict(actor_steps=state.actor_steps))
    logs.update(collect_batch_telemetry_data(batch))

    return state, logs


def run_training_actor(
    actor: Actor, stop_signal: list[bool], controller: "ReplayRatioController"
):
    """Runs an actor to produce trajectories, checking the ratio each time."""

    while not stop_signal[0]:
        try:
            controller.actor_wait()
            step_count, params = actor.pull_params()
            actor.unroll_and_push(step_count, params)
        except Exception as e:
            traceback.print_exc()
            raise e


def run_eval_actor(actor: Actor, stop_signal: list[bool]):
    """Runs an actor to produce num_trajectories trajectories."""

    old_step_count, _ = actor.pull_params()
    session_id = actor._env.username
    win_reward_sum = {old_step_count: (0, 0)}

    while not stop_signal[0]:
        try:
            step_count, params = actor.pull_params()
            assert step_count >= old_step_count, (
                f"Actor {session_id} tried to pull params with frame count "
                f"{step_count} but expected at least {old_step_count}."
            )
            if step_count not in win_reward_sum:
                win_reward_sum[step_count] = (0, 0)

            if step_count > old_step_count:
                reward_count, reward_sum = win_reward_sum.pop(old_step_count)
                wandb.log(
                    {
                        "Step": old_step_count,
                        f"wr-{session_id}": reward_sum / max(1, reward_count),
                    }
                )
                old_step_count = step_count

            params = jax.device_put(params)
            subkey = actor.split_rng()
            eval_trajectory = actor.unroll(subkey, step_count, params)

            win_rewards = np.sign(eval_trajectory.timestep.env.win_reward[-1])
            # Update the win reward sum for this step count.
            reward_count, reward_sum = win_reward_sum[step_count]
            win_reward_sum[step_count] = (reward_count + 1, reward_sum + win_rewards)

            # Log the win reward mean for this step count if we have enough data.
        except Exception:
            traceback.print_exc()
            continue


class ReplayBuffer:
    """A simple, thread-safe FIFO experience replay buffer."""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._buffer = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._pbar = tqdm(desc="producer", smoothing=0)
        self._total_added = 0

    @property
    def total_added(self):
        """Total number of transitions added to the buffer."""
        return self._total_added

    def is_ready(self, min_size: int):
        return len(self) >= min_size

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, transition: Transition):
        with self._lock:
            self._buffer.append(transition)
            self._pbar.update(1)
            self._total_added += 1

    def sample(self, batch_size: int) -> Transition:
        with self._lock:
            if len(self._buffer) < batch_size:
                raise ValueError(
                    f"Not enough transitions in buffer to sample batch of size {batch_size}."
                    f" Buffer size: {len(self._buffer)}"
                )
            batch = random.sample(self._buffer, batch_size)

        stacked_batch: Transition = jax.tree.map(
            lambda *xs: np.stack(xs, axis=1), *batch
        )

        # resolution = 64
        # valid = jnp.bitwise_not(stacked_batch.timestep.env.done)
        # num_valid = valid.sum(0).max().item() + 1
        # num_valid = int(np.ceil(num_valid / resolution) * resolution)

        stacked_batch = Transition(
            timestep=TimeStep(
                env=stacked_batch.timestep.env,
                # env=jax.tree.map(lambda x: x[:num_valid], stacked_batch.timestep.env),
                history=clip_history(stacked_batch.timestep.history, resolution=128),
            ),
            actorstep=stacked_batch.actorstep,
            # actorstep=jax.tree.map(lambda x: x[:num_valid], stacked_batch.actorstep),
        )

        return jax.device_put(stacked_batch)


class ReplayRatioController:
    """Manages a target replay ratio by controlling both learners and actors."""

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        get_num_samples: Callable[[], int],
        learner_config: MMDConfig,
    ):
        self.replay_buffer = replay_buffer
        self.get_num_samples = get_num_samples
        self.target_replay_ratio = learner_config.target_replay_ratio
        self.batch_size = learner_config.batch_size

        # A condition to make the LEARNER wait
        self._learner_can_proceed = threading.Condition()
        # A separate condition to make the ACTORS wait
        self._actor_can_proceed = threading.Condition()

    def _get_current_ratio(self) -> float:
        """Calculates the current consumer/producer ratio."""
        num_samples = self.get_num_samples()
        producer_steps = max(1, self.replay_buffer.total_added)
        return num_samples / producer_steps

    def _is_safe_to_train(self) -> bool:
        """Checks if the learner is allowed to proceed"""
        not_enough_learning = self._get_current_ratio() <= self.target_replay_ratio
        buffer_ready = self.replay_buffer.is_ready(self.batch_size)
        return not_enough_learning and buffer_ready

    def _is_safe_to_produce(self) -> bool:
        """Checks if actors are allowed to produce"""
        too_much_learning = self._get_current_ratio() > self.target_replay_ratio
        no_samples = self.get_num_samples() == 0
        return too_much_learning or no_samples

    def learner_wait(self):
        """Called by the learner; blocks until it's safe to train."""
        with self._learner_can_proceed:
            while not self._is_safe_to_train():
                self._learner_can_proceed.wait()

    def actor_wait(self):
        """Called by an actor; blocks until it's safe to produce data."""
        with self._actor_can_proceed:
            while not self._is_safe_to_produce():
                self._actor_can_proceed.wait()

    def signal_learner(self):
        """Called by a producer after adding data."""
        with self._learner_can_proceed:
            self._learner_can_proceed.notify_all()

    def signal_actors(self):
        """Called by the learner after consuming data."""
        with self._actor_can_proceed:
            self._actor_can_proceed.notify_all()


def host_to_device_worker(
    trajectory_queue: queue.Queue[Transition],
    stop_signal: list[bool],
    replay_buffer: ReplayBuffer,
    controller: ReplayRatioController,
):
    """Elementary data pipeline."""
    while not stop_signal[0]:
        try:
            trajectory = trajectory_queue.get(timeout=10)
        except queue.Empty:
            continue

        replay_buffer.add(trajectory)

        controller.signal_learner()


JAX_JIT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "../../.jax_jit_cache")


def init_jax_jit_cache(jax_jit_cache_path: str = JAX_JIT_CACHE_PATH):
    if not os.path.exists(jax_jit_cache_path):
        os.mkdir(jax_jit_cache_path)
    jax.experimental.compilation_cache.compilation_cache.set_cache_dir(
        jax_jit_cache_path
    )


def set_jax_env_vars():
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_TRACEBACK_FILTERING"] = "off"
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_triton_gemm_any=True "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
    )


def main():
    """Main function to run the MMD learner."""
    set_jax_env_vars()
    init_jax_jit_cache()

    learner_config = get_config()
    model_config = get_model_cfg()
    pprint(learner_config)

    network = get_model(model_config)
    # network = get_dummy_model()

    actor_threads: list[threading.Thread] = []
    stop_signal = [False]

    num_eval_actors = 4
    trajectory_queue: queue.Queue[Transition] = queue.Queue(
        maxsize=2 * learner_config.num_actors
    )

    state = create_train_state(network, jax.random.PRNGKey(42), learner_config)

    gpu_lock = FairLock()  # threading.Lock()
    agent = Agent(state.apply_fn, gpu_lock)

    replay_buffer = ReplayBuffer(
        capacity=max(
            learner_config.replay_buffer_capacity, learner_config.batch_size * 2
        )
    )

    latest_ckpt = get_most_recent_file("./ckpts", "mmd")
    if latest_ckpt:
        state, replay_buffer = load(state, replay_buffer, latest_ckpt)

    controller = ReplayRatioController(
        replay_buffer, lambda: int(state.num_samples), learner_config
    )

    def params_for_actor():
        return int(state.step), jax.device_get(state.params)

    for game_id in range(learner_config.num_actors // 2):
        for player_id in range(2):
            actor = Actor(
                agent=agent,
                env=SinglePlayerSyncEnvironment(f"train-{game_id:02d}{player_id:02d}"),
                unroll_length=learner_config.unroll_length,
                queue=trajectory_queue,
                params_for_actor=params_for_actor,
                rng_seed=len(actor_threads),
            )
            args = (actor, stop_signal, controller)
            actor_threads.append(
                threading.Thread(
                    target=run_training_actor,
                    args=args,
                    name=f"Actor-{game_id}-{player_id}",
                )
            )

    for eval_id in range(num_eval_actors):
        actor = Actor(
            agent=agent,
            env=SinglePlayerSyncEnvironment(f"eval-{eval_id:04d}"),
            unroll_length=learner_config.unroll_length,
            params_for_actor=params_for_actor,
            rng_seed=len(actor_threads),
        )
        args = (actor, stop_signal)
        actor_threads.append(
            threading.Thread(
                target=run_eval_actor, args=args, name=f"EvalActor-{eval_id}"
            )
        )

    wandb.init(
        project="pokemon-rl",
        config={
            "num_params": get_num_params(state.params),
            "learner_config": learner_config,
            "model_config": json.loads(model_config.to_json_best_effort()),
        },
    )

    # Start the actors and learner.
    for t in actor_threads:
        t.start()

    transfer_thread = threading.Thread(
        target=host_to_device_worker,
        args=(trajectory_queue, stop_signal, replay_buffer, controller),
    )
    transfer_thread.start()

    consumer_progress = tqdm(desc="consumer", smoothing=0)
    train_progress = tqdm(desc="batches", smoothing=0)
    batch_size = learner_config.batch_size
    last_oom = time.time()

    for _ in range(learner_config.num_steps):
        try:
            controller.learner_wait()

            batch = replay_buffer.sample(batch_size)
            with gpu_lock:
                targets = compute_returns(state, batch, learner_config)
            with gpu_lock:
                state, logs = train_step(state, batch, targets, learner_config)

            wandb.log(logs)

            # Update the tqdm progress bars.
            consumer_progress.update(batch_size)
            train_progress.update(1)

            controller.signal_actors()

            if state.step % 5000 == 0:
                save(state, replay_buffer)

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
                batch_size *= 2
                print(f"No OOM for 60 minutes, doubling batch size to {batch_size}.")
                last_oom = time.time()

    stop_signal[0] = True
    for t in actor_threads:
        t.join()

    print("done")


if __name__ == "__main__":
    main()
