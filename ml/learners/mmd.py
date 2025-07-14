import collections
import functools
import json
import math
import os
import pickle
from pprint import pprint
from typing import Iterator

import chex
import flax.linen as nn
import jax
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
    collect_nn_telemetry_data,
    collect_parameter_and_gradient_telemetry_data,
    collect_policy_stats_telemetry_data,
)
from ml.utils import Params, get_most_recent_file
from rlenv.env import get_ex_step
from rlenv.interfaces import ModelOutput, TimeStep
from rlenv.main import DoubleTrajectoryTrainingBatchCollector, EvalBatchCollector


@chex.dataclass(frozen=True)
class MMDConfig:
    num_steps = 10_000_000
    num_actors: int = 64
    do_eval: bool = True
    num_eval_games: int = 200

    # Batch iteration params
    num_epochs: int = 40
    minibatch_size: int = 4

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
    ex, hx = get_ex_step()

    params = module.init(rng, ex, hx)

    tx = optax.chain(
        optax.adamw(
            learning_rate=config.learning_rate,
            eps_root=0.0,
            b1=config.adam.b1,
            b2=config.adam.b2,
            eps=config.adam.eps,
            weight_decay=config.adam.weight_decay,
        ),
        optax.clip_by_global_norm(config.clip_gradient),
    )

    return TrainState.create(apply_fn=module.apply, params=params, tx=tx)


def save(state: TrainState):
    with open(os.path.abspath(f"ckpts/mmd_ckpt_{state.step:08}"), "wb") as f:
        pickle.dump(
            dict(
                params=state.params,
                opt_state=state.opt_state,
                step=state.step,
                actor_steps=state.actor_steps,
                target_adv_mean=state.target_adv_mean,
                target_adv_std=state.target_adv_std,
            ),
            f,
        )


def load(state: TrainState, path: str):
    print(f"loading checkpoint from {path}")
    with open(path, "rb") as f:
        step: TrainState = pickle.load(f)

    step_no = step.get("step", 0)
    print(f"Learner steps: {step_no:08}")

    actor_steps = step.get("actor_steps", 0)
    print("Actor steps: ", actor_steps)

    params = step["params"]
    state = state.replace(
        step=step_no,
        params=params,
        actor_steps=actor_steps,
        opt_state=step["opt_state"],
        target_adv_mean=step.get("target_adv_mean", 0),
        target_adv_std=step.get("target_adv_std", 1),
    )

    return state


VTraceOutput = collections.namedtuple(
    "vtrace_output", ["errors", "pg_advantage", "q_estimate"]
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
        errors=errors, pg_advantage=pg_advantages, q_estimate=q_estimate
    )


def compute_returns(
    v_tm1: chex.Array, rho_tm1: chex.Array, batch: TimeStep, config: MMDConfig
):
    """Train for a single step."""

    valid = batch.env.valid

    rewards = jnp.take_along_axis(
        batch.env.rewards.win_rewards, batch.env.player_id[..., None], axis=-1
    ).squeeze()

    rewards = jnp.concatenate((rewards[1:], rewards[-1:]))
    v_t = jnp.concatenate((v_tm1[1:], v_tm1[-1:]))
    valids = jnp.concatenate((valid[1:], valid[-1:]))

    discount_t = valids * config.gamma
    lambda_ = valids * config.lambda_

    return jax.vmap(vtrace_td_error_and_advantage, in_axes=1, out_axes=1)(
        v_tm1, v_t, rewards, discount_t, rho_tm1, lambda_
    )


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(state: TrainState, batch: TimeStep, config: MMDConfig):
    """Train for a single step."""

    def loss_fn(params: Params):

        pred: ModelOutput = jax.vmap(state.apply_fn, (None, 1, 1), 1)(
            params, batch.env, batch.history
        )

        valid = batch.env.valid

        action = jax.lax.stop_gradient(batch.actor.action[..., None])
        log_pi = jnp.take_along_axis(pred.log_pi, action, axis=-1).squeeze()
        log_mu = jnp.take_along_axis(batch.actor.log_policy, action, axis=-1).squeeze()
        log_ratio = log_pi - log_mu
        ratio = jnp.exp(log_ratio)

        targets = compute_returns(pred.v.reshape(*valid.shape), ratio, batch, config)

        advantages: jax.Array = jax.lax.stop_gradient(targets.pg_advantage)
        adv_mean = advantages.mean(where=valid)
        adv_std = advantages.std(where=valid)
        advantages = (advantages - state.target_adv_mean) / (
            state.target_adv_std + 1e-8
        )

        pg_loss = -advantages * log_pi
        loss_pg = pg_loss.mean(where=valid)
        loss_v = jnp.square(targets.errors).mean(where=valid)
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
        approx_kl = ((ratio - 1) - log_ratio).mean(where=valid)

        logs = dict(
            old_approx_kl=old_approx_kl,
            approx_kl=approx_kl,
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
                batch.env.legal,
                valid,
                batch.actor.policy,
                ratio,
                advantages,
            )
        )
        logs.update(dict(adv_mean=adv_mean, adv_std=adv_std))

        return loss, logs

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, logs), grads = grad_fn(state.params)

    logs.update(dict(loss=loss_val))
    logs.update(collect_parameter_and_gradient_telemetry_data(state.params, grads))

    state = state.apply_gradients(grads=grads)

    adv_tau = 1e-1
    state = state.replace(
        actor_steps=state.actor_steps + batch.env.valid.sum(),
        target_adv_mean=state.target_adv_mean * (1 - adv_tau)
        + logs["adv_mean"] * adv_tau,
        target_adv_std=state.target_adv_std * (1 - adv_tau) + logs["adv_std"] * adv_tau,
    )

    return state, logs


def iterate(batch: TimeStep, minibatch_size: int = 4) -> Iterator[TimeStep]:
    _, batch_size, *__ = batch.env.valid.shape
    indices = np.arange(batch_size)
    np.random.shuffle(indices)
    for batch_index in range(math.ceil(batch_size / minibatch_size)):
        minibatch_indices = indices[
            batch_index * minibatch_size : (batch_index + 1) * minibatch_size
        ]
        yield jax.tree.map(lambda x: x[:, minibatch_indices], batch)


def evaluate(evaluation_collector: EvalBatchCollector, state: train_state.TrainState):
    eval_batch = evaluation_collector.collect_batch_trajectory(state.params)

    win_rewards = np.sign(
        (
            eval_batch.actor.rewards.win_rewards[..., 0].squeeze()
            * eval_batch.env.valid.squeeze()
        ).sum(0)
    )

    fainted_rewards = (
        eval_batch.actor.rewards.fainted_rewards[..., 0].squeeze()
        * eval_batch.env.valid.squeeze()
    ).sum(0)

    winrates = {f"wr{i}": wr for i, wr in enumerate(win_rewards)}
    winrates.update({f"hp{i}": f for i, f in enumerate(fainted_rewards)})

    return winrates


def main():
    learner_config = get_config()
    model_config = get_model_cfg()
    pprint(learner_config)

    training_network = inference_network = get_model(model_config)
    # training_network = inference_network = get_dummy_model()

    training_collector = DoubleTrajectoryTrainingBatchCollector(
        inference_network, learner_config.num_actors
    )
    evaluation_collector = EvalBatchCollector(inference_network, 4)

    state = create_train_state(training_network, jax.random.PRNGKey(42), learner_config)

    latest_ckpt = get_most_recent_file("./ckpts", "mmd")
    if latest_ckpt:
        state = load(state, latest_ckpt)

    wandb.init(
        project="pokemon-rl",
        config={
            "num_params": get_num_params(state.params),
            "learner_config": learner_config,
            "model_config": json.loads(model_config.to_json_best_effort()),
        },
    )

    eval_freq = 5000
    save_freq = 1000

    train_progress = tqdm(desc="training", smoothing=0)

    for _ in range(learner_config.num_steps):
        logs: dict

        batch = training_collector.collect_batch_trajectory(state.params)

        for _ in range(learner_config.num_epochs):
            should_early_stop = False

            # Do the learning updates
            for minibatch in iterate(batch, learner_config.minibatch_size):
                winrates = {}

                time_to_eval = (
                    state.step % (eval_freq // learner_config.num_eval_games) == 0
                )
                if time_to_eval and learner_config.do_eval:
                    winrates = evaluate(evaluation_collector, state)

                new_state, logs = train_step(state, minibatch, learner_config)

                should_early_stop = (
                    logs["approx_kl"] > learner_config.kl_target * 1.5
                ).item()
                if should_early_stop:
                    break

                state = new_state

                logs.update(collect_nn_telemetry_data(state))
                logs.update(collect_batch_telemetry_data(minibatch))
                # logs.update(collect_action_prob_telemetry_data(minibatch))

                logs["Step"] = state.step
                wandb.log({**logs, **winrates})
                train_progress.update(1)

                if state.step % save_freq == 0 and state.step > 0:
                    save(state)

            if should_early_stop:
                print("Early stopping")
                break

    print("done")


if __name__ == "__main__":
    main()
