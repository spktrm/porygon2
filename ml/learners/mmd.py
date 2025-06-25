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
    collect_value_stats_telemetry_data,
)
from ml.utils import Params, get_most_recent_file
from rlenv.env import clip_history, get_ex_step
from rlenv.interfaces import ModelOutput, Targets, TimeStep
from rlenv.main import DoubleTrajectoryTrainingBatchCollector, EvalBatchCollector


@chex.dataclass(frozen=True)
class MMDConfig:
    num_steps = 1_000_000
    num_actors: int = 32
    do_eval: bool = True
    num_eval_games: int = 200

    # Batch iteration params
    num_epochs: int = 40
    minibatch_size: int = 4

    # Learning params
    adam: AdamConfig = AdamConfig(b1=0, b2=0.999, eps=1e-8, weight_decay=0)
    learning_rate: float = 3e-5
    clip_gradient: float = 1

    # PPO params
    clip_coef: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 0.99

    # Loss coefficients
    value_loss_coef: float = 0.25
    policy_loss_coef: float = 1.0
    entropy_loss_coef: float = 0.05
    kl_loss_coef: float = 0.05

    # Stopping param
    kl_target: float = 0.025


def get_config():
    return MMDConfig()


class TrainState(train_state.TrainState):

    actor_steps: int = 0


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
    )

    return state


def compute_gae(
    r_t: chex.Array,
    discount_t: chex.Array,
    lambda_: float,
    values: chex.Array,
    stop_target_gradients: bool = False,
) -> chex.Array:
    """Computes truncated generalized advantage estimates for a sequence length k.

    The advantages are computed in a backwards fashion according to the equation:
    Âₜ = δₜ + (γλ) * δₜ₊₁ + ... + ... + (γλ)ᵏ⁻ᵗ⁺¹ * δₖ₋₁
    where δₜ = rₜ₊₁ + γₜ₊₁ * v(sₜ₊₁) - v(sₜ).

    See Proximal Policy Optimization Algorithms, Schulman et al.:
    https://arxiv.org/abs/1707.06347

    Note: This paper uses a different notation than the RLax standard
    convention that follows Sutton & Barto. We use rₜ₊₁ to denote the reward
    received after acting in state sₜ, while the PPO paper uses rₜ.

    Args:
      r_t: Sequence of rewards at times [1, k]
      discount_t: Sequence of discounts at times [1, k]
      lambda_: Mixing parameter; a scalar or sequence of lambda_t at times [1, k]
      values: Sequence of values under π at times [0, k]
      stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.

    Returns:
      Multistep truncated generalized advantage estimation at times [0, k-1].
    """
    chex.assert_rank([r_t, values, discount_t], 1)
    chex.assert_type([r_t, values, discount_t], float)
    lambda_ = jnp.ones_like(discount_t) * lambda_  # If scalar, make into vector.

    delta_t = r_t + discount_t * values[1:] - values[:-1]

    # Iterate backwards to calculate advantages.
    def _body(acc, xs):
        deltas, discounts, lambda_ = xs
        acc = deltas + discounts * lambda_ * acc
        return acc, acc

    _, advantage_t = jax.lax.scan(
        _body, 0.0, (delta_t, discount_t, lambda_), reverse=True
    )

    return jax.lax.select(
        stop_target_gradients, jax.lax.stop_gradient(advantage_t), advantage_t
    )


@functools.partial(jax.jit, static_argnums=(2,))
def compute_returns(state: TrainState, batch: TimeStep, config: MMDConfig):
    """Train for a single step."""

    pred: ModelOutput = jax.vmap(state.apply_fn, (None, 1, 1), 1)(
        state.params, batch.env, batch.history
    )

    valid = batch.env.valid

    rewards = jnp.take_along_axis(
        batch.actor.rewards.win_rewards, batch.env.player_id[..., None], axis=-1
    ).squeeze()

    value_pred = jnp.squeeze(pred.v, axis=-1)
    baselines = jnp.concatenate([value_pred, jnp.zeros_like(value_pred[-1:])])
    discounts = valid * config.gamma
    lambda_ = jnp.ones_like(valid) * config.gae_lambda

    advantages = jax.vmap(
        functools.partial(compute_gae, stop_target_gradients=True),
        in_axes=(1, 1, 1, 1),
    )(rewards, discounts, lambda_, baselines).T
    returns = jax.lax.stop_gradient(advantages + baselines[:-1])

    return Targets(advantages=advantages, returns=returns)


def compute_target_statistics(
    batch: TimeStep, iterations: list[tuple[TimeStep, Targets]]
):
    valid_sum = batch.env.valid.sum()
    target_adv_mean = (
        sum([jnp.sum(t.advantages, where=m.env.valid) for m, t in iterations])
        / valid_sum
    )
    target_adv_std = jnp.sqrt(
        sum(
            [
                jnp.square(t.advantages - target_adv_mean).sum(where=m.env.valid)
                for m, t in iterations
            ]
        )
        / valid_sum
    )
    return target_adv_mean, target_adv_std


@functools.partial(jax.jit, static_argnums=(3,))
def train_step(state: TrainState, batch: TimeStep, targets: Targets, config: MMDConfig):
    """Train for a single step."""

    def loss_fn(params: Params):

        pred: ModelOutput = jax.vmap(state.apply_fn, (None, 1, 1), 1)(
            params, batch.env, batch.history
        )

        valid = batch.env.valid
        value_pred = jnp.squeeze(pred.v, axis=-1)

        action = batch.actor.action[..., None]
        log_pi = jnp.take_along_axis(pred.log_pi, action, axis=-1).squeeze()
        log_mu = jnp.take_along_axis(batch.actor.log_policy, action, axis=-1).squeeze()
        log_ratio = log_pi - log_mu
        ratio = jnp.exp(log_ratio)

        advantages = (targets.advantages - targets.advantage_mean) / (
            targets.advantage_std + 1e-8
        )

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * ratio.clip(1 - config.clip_coef, 1 + config.clip_coef)

        loss_pg = jnp.maximum(pg_loss1, pg_loss2).mean(where=valid)
        loss_v = jnp.square(targets.returns - value_pred).mean(where=valid)
        loss_entropy = -(pred.pi * pred.log_pi).sum(axis=-1).mean(where=valid)

        backward_kl_approx = ratio * log_ratio - (ratio - 1)
        loss_kl = backward_kl_approx.mean(where=valid)

        ent_kl_coef_mult = 1  # jnp.sqrt(10_000_000 / (state.actor_steps + 1000))

        loss = (
            loss_pg
            + config.value_loss_coef * loss_v
            - config.entropy_loss_coef * ent_kl_coef_mult * loss_entropy
            + config.kl_loss_coef * ent_kl_coef_mult * loss_kl
        )

        old_approx_kl = (-log_ratio).mean(where=valid)
        approx_kl = ((ratio - 1) - log_ratio).mean(where=valid)
        clipfracs = (
            ~jnp.isclose(ratio.clip(1 - config.clip_coef, 1 + config.clip_coef), ratio)
        ).mean(where=valid)

        logs = dict(
            old_approx_kl=old_approx_kl,
            approx_kl=approx_kl,
            clipfracs=clipfracs,
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
        logs.update(
            collect_value_stats_telemetry_data(value_pred, targets.returns, valid)
        )

        return loss, logs

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, logs), grads = grad_fn(state.params)

    logs.update(dict(loss=loss_val))
    logs.update(collect_parameter_and_gradient_telemetry_data(state.params, grads))

    state = state.apply_gradients(grads=grads)
    state = state.replace(
        actor_steps=state.actor_steps + batch.env.valid.sum(),
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

    training_network = get_model(model_config)
    inference_network = get_model(model_config)

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

            # Compute targets for the batch
            # This is done outside the minibatch loop to avoid recomputing the
            # model output for each minibatch.
            iterations: list[tuple[TimeStep, Targets]] = []
            for minibatch in iterate(batch, learner_config.minibatch_size):
                minibatch = TimeStep(
                    env=minibatch.env,
                    history=clip_history(minibatch.history, resolution=64),
                    actor=minibatch.actor,
                )
                targets = compute_returns(state, minibatch, learner_config)
                iterations.append((minibatch, targets))

            target_adv_mean, target_adv_std = compute_target_statistics(
                batch, iterations
            )

            # Do the learning updates
            for minibatch, target in iterations:
                winrates = {}

                time_to_eval = (
                    state.step % (eval_freq // learner_config.num_eval_games) == 0
                )
                if time_to_eval and learner_config.do_eval:
                    winrates = evaluate(evaluation_collector, state)

                new_state, logs = train_step(
                    state,
                    minibatch,
                    Targets(
                        advantages=target.advantages,
                        returns=target.returns,
                        advantage_mean=target_adv_mean,
                        advantage_std=target_adv_std,
                    ),
                    learner_config,
                )

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
