import collections
import functools
import json
import logging
import os
import pickle
import queue
import random
import sys
import tempfile
import threading
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from pprint import pprint

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
from flax.training import train_state
from tqdm import tqdm

import wandb
from ml.arch.config import get_model_cfg
from ml.arch.model import get_model, get_num_params
from ml.config import AdamConfig
from ml.learners.func import (
    collect_batch_telemetry_data,
    collect_parameter_and_gradient_telemetry_data,
    collect_policy_stats_telemetry_data,
    collect_state_telemetry_data,
    collect_value_stats_telemetry_data,
)
from ml.utils import Params, get_most_recent_file
from rlenv.env import clip_history, get_ex_step
from rlenv.interfaces import ModelOutput, TimeStep, Transition
from rlenv.main import Actor, Agent, SinglePlayerSyncEnvironment
from rlenv.utils import NoOpLock


@chex.dataclass(frozen=True)
class MMDConfig:
    num_steps = 10_000_000
    num_actors: int = 32
    do_eval: bool = True
    num_eval_games: int = 200
    unroll_length: int = 108
    replay_buffer_capacity: int = 512

    # Batch iteration params
    batch_size: int = 2

    # Learning params
    adam: AdamConfig = AdamConfig(b1=0, b2=0.999, eps=1e-8, weight_decay=0)
    learning_rate: float = 3e-5
    clip_gradient: float = 1

    # Vtrace params
    lambda_: float = 0.95
    gamma: float = 1.0

    # Loss coefficients
    value_loss_coef: float = 0.25
    policy_loss_coef: float = 1.0
    entropy_loss_coef: float = 0.05
    kl_loss_coef: float = 0.05

    # Stopping param
    kl_target: float = 0.15


def get_config():
    return MMDConfig()


class TrainState(train_state.TrainState):
    actor_steps: int = 0

    target_adv_mean: float = 0
    target_adv_std: float = 1


def create_train_state(module: nn.Module, rng: PRNGKey, config: MMDConfig):
    """Creates an initial `TrainState`."""
    ts = get_ex_step()
    ts = jax.tree.map(lambda x: x[:, 0], get_ex_step())

    params = module.init(rng, ts)

    tx = optax.chain(
        optax.clip_by_global_norm(config.clip_gradient),
        optax.adamw(
            learning_rate=config.learning_rate,
            eps_root=0.0,
            b1=config.adam.b1,
            b2=config.adam.b2,
            eps=config.adam.eps,
            weight_decay=config.adam.weight_decay,
        ),
    )

    return TrainState.create(
        apply_fn=jax.vmap(module.apply, in_axes=(None, 1), out_axes=1),
        params=params,
        tx=tx,
    )


def save(state: TrainState, replay_buffer: "ReplayBuffer"):
    with replay_buffer._lock:
        with open(os.path.abspath(f"ckpts/mmd_ckpt_{state.step:08}"), "wb") as f:
            pickle.dump(
                dict(
                    params=state.params,
                    opt_state=state.opt_state,
                    step=state.step,
                    actor_steps=state.actor_steps,
                    target_adv_mean=state.target_adv_mean,
                    target_adv_std=state.target_adv_std,
                    replay_buffer_deque=deepcopy(replay_buffer._buffer),
                ),
                f,
            )


def load(state: TrainState, replay_buffer: "ReplayBuffer", path: str):
    print(f"loading checkpoint from {path}")
    with open(path, "rb") as f:
        ckpt_data = pickle.load(f)

    step_no = ckpt_data.get("step", 0)
    print(f"Learner steps: {step_no:08}")

    actor_steps = ckpt_data.get("actor_steps", 0)
    print("Actor steps: ", actor_steps)

    params = ckpt_data["params"]
    state = state.replace(
        step=step_no,
        params=params,
        actor_steps=actor_steps,
        opt_state=ckpt_data["opt_state"],
        target_adv_mean=ckpt_data.get("target_adv_mean", 0),
        target_adv_std=ckpt_data.get("target_adv_std", 1),
    )

    if "replay_buffer_deque" in ckpt_data:
        replay_buffer = ReplayBuffer.from_deque(ckpt_data["replay_buffer_deque"])
        print(f"Loaded replay buffer with {len(replay_buffer)} transitions.")

    return state, replay_buffer


VTraceOutput = collections.namedtuple(
    "vtrace_output", ["returns", "pg_advantage", "q_estimate"]
)


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
    lambda_ = valids * config.lambda_

    return jax.vmap(vtrace_td_error_and_advantage, in_axes=1, out_axes=1)(
        v_tm1, v_t, rewards, discount_t, rho_tm1, lambda_
    )


@functools.partial(jax.jit, static_argnums=(2,))
def compute_returns(state: TrainState, batch: Transition, config: MMDConfig):

    pred: ModelOutput = state.apply_fn(state.params, batch.timestep)

    valid = jnp.bitwise_not(batch.timestep.env.done)

    action = jax.lax.stop_gradient(batch.actorstep.action[..., None])
    log_pi = jnp.take_along_axis(pred.log_pi, action, axis=-1).squeeze()
    log_mu = jnp.take_along_axis(
        batch.actorstep.model_output.log_pi, action, axis=-1
    ).squeeze()
    log_ratio = log_pi - log_mu
    ratio = jnp.exp(log_ratio)

    return _compute_returns(pred.v.reshape(*valid.shape), ratio, batch, config)


@functools.partial(jax.jit, static_argnums=(3,))
def train_step(
    state: TrainState, batch: Transition, targets: VTraceOutput, config: MMDConfig
):
    """Train for a single step."""

    def loss_fn(params: Params):

        pred: ModelOutput = state.apply_fn(params, batch.timestep)

        action = jax.lax.stop_gradient(batch.actorstep.action[..., None])
        log_pi = jnp.take_along_axis(pred.log_pi, action, axis=-1).squeeze()
        log_mu = jnp.take_along_axis(
            batch.actorstep.model_output.log_pi, action, axis=-1
        ).squeeze()
        log_ratio = log_pi - log_mu
        ratio = jnp.exp(log_ratio)

        approx_kl = (ratio - 1) - log_ratio

        valid = jnp.bitwise_not(batch.timestep.env.done) & (
            approx_kl <= config.kl_target
        )

        advantages: jax.Array = jax.lax.stop_gradient(targets.pg_advantage)
        adv_mean = advantages.mean(where=valid)
        adv_std = advantages.std(where=valid)
        advantages = (advantages - state.target_adv_mean) / (
            state.target_adv_std + 1e-8
        )

        pg_loss = -advantages * log_pi
        loss_pg = pg_loss.mean(where=valid)
        loss_v = jnp.square(pred.v.reshape(*valid.shape) - targets.returns).mean(
            where=valid
        )
        loss_entropy = -(pred.pi * pred.log_pi).sum(axis=-1).mean(where=valid)

        backward_kl_approx = ratio * log_ratio - (ratio - 1)
        loss_kl = backward_kl_approx.mean(where=valid)

        ent_kl_coef_mult = jnp.sqrt(10_000_000 / (state.actor_steps + 1000))

        loss = (
            loss_pg
            + config.value_loss_coef * loss_v
            - config.entropy_loss_coef * ent_kl_coef_mult * loss_entropy
            + config.kl_loss_coef * ent_kl_coef_mult * loss_kl
        )

        old_approx_kl = (-log_ratio).mean(where=valid)

        logs = dict(
            old_approx_kl=old_approx_kl,
            approx_kl=approx_kl.mean(where=valid),
            ent_kl_coef_mult=ent_kl_coef_mult,
            loss_pg=loss_pg,
            loss_v=loss_v,
            loss_entropy=loss_entropy,
            loss_kl=loss_kl,
        )
        logs.update(
            collect_policy_stats_telemetry_data(
                pred.logit,
                pred.pi,
                pred.log_pi,
                batch.timestep.env.legal,
                valid,
                batch.actorstep.model_output.pi,
                ratio,
                advantages,
            )
        )
        logs.update(collect_value_stats_telemetry_data(pred.v, targets.returns, valid))
        logs.update(dict(adv_mean=adv_mean, adv_std=adv_std))

        return loss, logs

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, logs), grads = grad_fn(state.params)

    logs.update(dict(loss=loss_val))
    logs.update(collect_parameter_and_gradient_telemetry_data(state.params, grads))

    state = state.apply_gradients(grads=grads)

    adv_tau = 1e-1
    # Add 1 for the final step in each trajectory
    new_steps = (jnp.bitwise_not(batch.timestep.env.done).sum(0) + 1).sum()
    state = state.replace(
        actor_steps=state.actor_steps + new_steps,
        target_adv_mean=state.target_adv_mean * (1 - adv_tau)
        + logs["adv_mean"] * adv_tau,
        target_adv_std=state.target_adv_std * (1 - adv_tau) + logs["adv_std"] * adv_tau,
    )

    return state, logs


DEADLOCK_TIMEOUT_S = 60 * 5  # 5 minutes


def start_deadlock_watchdog(timeout_s: int = DEADLOCK_TIMEOUT_S) -> callable:
    """
    Starts a watchdog thread that monitors for deadlock-like pauses.
    If no thread makes progress for `timeout_s`, it writes every live
    thread's stack trace to a file inside a tmp directory and tells you
    where to find them.

    Returns
    -------
    touch : callable
        Call this whenever meaningful work is completed to reset the timer.
    """
    last_progress = {"t": time.time()}

    def touch() -> None:
        """Reset the watchdog timer — call after each unit of work."""
        last_progress["t"] = time.time()

    def watchdog() -> None:
        # One temp dir per watchdog lifetime, so all dumps live together
        dump_root = Path(
            tempfile.mkdtemp(prefix="deadlock_watchdog_", dir=tempfile.gettempdir())
        )

        logging.info("🛡️  Deadlock watchdog armed; dumps will go to %s", dump_root)

        while True:
            time.sleep(timeout_s)
            if time.time() - last_progress["t"] > timeout_s:
                logging.error("⚠️  Possible deadlock – dumping stacks to %s", dump_root)

                timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                dump_path = dump_root / f"{timestamp}_thread.txt"
                with dump_path.open("w", encoding="utf-8") as fh:
                    for tid, frame in sys._current_frames().items():
                        fh.write(f"Thread {tid} at {timestamp} UTC\n")
                        fh.write("".join(traceback.format_stack(frame)))

                # Point the user at the folder once per dump cycle
                logging.error(
                    "📄  Stack traces written. Inspect files in %s", dump_root
                )

    threading.Thread(target=watchdog, daemon=True, name="DeadlockWatchdog").start()
    return touch


def run_training_actor(actor: Actor, stop_signal: list[bool]):
    """Runs an actor to produce num_trajectories trajectories."""
    touch = start_deadlock_watchdog()

    old_step_count = 0
    while not stop_signal[0]:
        try:
            session_id = actor._env.username
            step_count, params = actor.pull_params()
            assert step_count >= old_step_count, (
                f"Actor {session_id} tried to pull params with frame count "
                f"{step_count} but expected at least {old_step_count}."
            )
            actor.unroll_and_push(step_count, params)
            old_step_count = step_count
        except Exception as e:
            traceback.print_exc()
            raise e

        touch()


def run_eval_actor(actor: Actor, stop_signal: list[bool]):
    """Runs an actor to produce num_trajectories trajectories."""
    touch = start_deadlock_watchdog()

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

            win_rewards = np.sign(eval_trajectory.timestep.env.win_reward.sum(0))
            # Update the win reward sum for this step count.
            reward_count, reward_sum = win_reward_sum[step_count]
            win_reward_sum[step_count] = (reward_count + 1, reward_sum + win_rewards)

            # Log the win reward mean for this step count if we have enough data.
        except Exception:
            traceback.print_exc()
            continue

        touch()


class ReplayBuffer:
    """A simple, thread-safe FIFO experience replay buffer."""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._buffer = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._pbar = tqdm(desc="producer", smoothing=0.1)

    @classmethod
    def from_deque(cls, deque_buffer: collections.deque):
        instance = cls(deque_buffer.maxlen)
        instance._buffer = deque_buffer
        instance._pbar.update(len(instance._buffer))
        return instance

    def is_ready(self):
        return len(self) > (self._capacity // 2)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, transition: Transition):
        with self._lock:
            self._buffer.append(transition)
            self._pbar.update(1)

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

        resolution = 64
        valid = valid = jnp.bitwise_not(stacked_batch.timestep.env.done)
        num_valid = valid.sum(0).max().item() + 1
        num_valid = int(np.ceil(num_valid / resolution) * resolution)

        stacked_batch = Transition(
            timestep=TimeStep(
                env=jax.tree.map(lambda x: x[:num_valid], stacked_batch.timestep.env),
                history=clip_history(
                    stacked_batch.timestep.history, resolution=resolution
                ),
            ),
            actorstep=jax.tree.map(lambda x: x[:num_valid], stacked_batch.actorstep),
        )

        return jax.device_put(stacked_batch)


def host_to_device_worker(
    trajectory_queue: queue.Queue[Transition],
    stop_signal: list[bool],
    replay_buffer: ReplayBuffer,
    start_condition: threading.Condition,
):
    """Elementary data pipeline."""

    notified = False
    while not stop_signal[0]:
        try:
            trajectory = trajectory_queue.get(timeout=10)
        except queue.Empty:
            continue

        replay_buffer.add(trajectory)

        if not notified and replay_buffer.is_ready():
            with start_condition:
                start_condition.notify_all()
            notified = True


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

    gpu_lock = threading.Lock()
    agent = Agent(state.apply_fn, gpu_lock)
    replay_buffer = ReplayBuffer(
        capacity=max(
            learner_config.replay_buffer_capacity, learner_config.batch_size * 2
        )
    )

    latest_ckpt = get_most_recent_file("./ckpts", "mmd")
    if latest_ckpt:
        state, replay_buffer = load(state, replay_buffer, latest_ckpt)

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
            args = (actor, stop_signal)
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

    start_condition = threading.Condition()

    transfer_thread = threading.Thread(
        target=host_to_device_worker,
        args=(trajectory_queue, stop_signal, replay_buffer, start_condition),
    )
    transfer_thread.start()

    with start_condition:
        start_condition.wait()

    consumer_progress = tqdm(desc="consumer", smoothing=0.1)
    train_progress = tqdm(desc="batches", smoothing=0.1)

    for _ in range(learner_config.num_steps):
        try:
            batch = replay_buffer.sample(learner_config.batch_size)

            with gpu_lock:
                targets = compute_returns(state, batch, learner_config)

            with gpu_lock:
                state, logs = train_step(state, batch, targets, learner_config)

            logs.update(collect_state_telemetry_data(state))
            logs.update(collect_batch_telemetry_data(batch))
            # logs.update(collect_action_prob_telemetry_data(minibatch))

            logs["Step"] = state.step
            wandb.log(logs)

            # Update the tqdm progress bars.
            consumer_progress.update(learner_config.batch_size)
            train_progress.update(1)

            if state.step % 5000 == 0:
                save(state, replay_buffer)

        except Exception:
            traceback.print_exc()

    stop_signal[0] = True
    for t in actor_threads:
        t.join()

    print("done")


if __name__ == "__main__":
    main()
