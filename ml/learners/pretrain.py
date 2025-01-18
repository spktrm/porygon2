import functools
import os
import pickle
from typing import Any

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import PRNGKey
from flax import core, struct
from flax.training import train_state

from ml.config import ActorCriticConfig, AdamConfig
from ml.func import (
    get_loss_entropy,
    get_loss_nerd,
    get_loss_v_mse,
    reg_v_trace,
    renormalize,
)
from ml.learners.rnad import NerdConfig
from ml.utils import Params
from rlenv.env import get_ex_step
from rlenv.interfaces import TimeStep


@chex.dataclass(frozen=True)
class PretrainConfig(ActorCriticConfig):
    adam: AdamConfig = AdamConfig(b1=0.99, b2=0.999, eps=1e-8, weight_decay=1e-5)

    learning_rate: float = 1e-3

    entropy_loss_coef: float = 1e-3
    target_network_avg: float = 1e-2

    nerd: NerdConfig = NerdConfig()


def get_config():
    return PretrainConfig()


class TrainState(train_state.TrainState):

    params_target: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    learner_steps: int = 0
    actor_steps: int = 0


def create_train_state(module: nn.Module, rng: PRNGKey, config: ActorCriticConfig):
    """Creates an initial `TrainState`."""
    ex = get_ex_step()

    params = module.init(rng, ex)
    params_target = module.init(rng, ex)

    tx = optax.chain(
        optax.adamw(
            learning_rate=config.learning_rate,
            eps_root=0.0,
            b1=config.adam.b1,
            b2=config.adam.b2,
            eps=config.adam.eps,
            weight_decay=config.adam.weight_decay,
        ),
        optax.clip(config.clip_gradient),
    )

    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        params_target=params_target,
        tx=tx,
    )


def save(state: TrainState):
    with open(os.path.abspath(f"ckpts/ckpt_{state.learner_steps:08}"), "wb") as f:
        pickle.dump(
            dict(
                params=state.params,
                params_target=state.params_target,
                opt_state=state.opt_state,
                step=state.step,
            ),
            f,
        )


def load(state: TrainState, path: str):
    print(f"loading checkpoint from {path}")
    with open(path, "rb") as f:
        step = pickle.load(f)

    state = state.replace(
        params=step["params"],
        params_target=step["params_target"],
        opt_state=step["opt_state"],
        step=step["step"],
    )

    return state


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(state: TrainState, batch: TimeStep, config: PretrainConfig):
    """Train for a single step."""

    rollout = jax.vmap(jax.vmap(state.apply_fn, (None, 0)), (None, 0))
    pred_targ = rollout(state.params_target, batch.env)
    vtrace_v = pred_targ.v

    def loss_fn(params: Params):
        rollout_w_grad = jax.vmap(jax.vmap(state.apply_fn, (None, 0)), (None, 0))

        pred = rollout_w_grad(params, batch.env)

        logs = {}

        policy_pprocessed = config.finetune(
            pred.pi, batch.env.legal, state.learner_steps
        )

        valid = batch.env.valid * (batch.env.legal.sum(axis=-1) > 1)

        v_target_list, has_played_list, v_trace_policy_target_list = [], [], []
        action_oh = jax.nn.one_hot(batch.actor.action, batch.actor.policy.shape[-1])

        rewards = (
            batch.actor.win_rewards
            # + 1e-1 * batch.actor.fainted_rewards / 6
            # + 1e-2 * batch.actor.hp_rewards / 6
        )

        for player in range(config.num_players):
            reward = rewards[:, :, player]  # [T, B, Player]
            v_target_, has_played, policy_target_ = reg_v_trace(
                vtrace_v,
                valid,
                batch.env.player_id,
                batch.actor.policy,
                policy_pprocessed,
                action_oh,
                reward,
                player,
                lambda_=1.0,
                c=config.c_vtrace,
                rho=jnp.inf,
                gamma=config.gamma,
            )
            v_target_list.append(jax.lax.stop_gradient(v_target_))
            has_played_list.append(has_played)
            v_trace_policy_target_list.append(jax.lax.stop_gradient(policy_target_))

        loss_v = get_loss_v_mse(
            [pred.v] * config.num_players,
            v_target_list,
            has_played_list,
        )

        is_vector = jnp.expand_dims(jnp.ones_like(valid), axis=-1)
        importance_sampling_correction = [is_vector] * config.num_players

        loss_nerd = get_loss_nerd(
            [pred.logit] * config.num_players,
            [pred.pi] * config.num_players,
            v_trace_policy_target_list,
            valid,
            batch.env.player_id,
            batch.env.legal,
            importance_sampling_correction,
            clip=config.nerd.clip,
            threshold=config.nerd.beta,
        )

        loss_norm = renormalize(
            jnp.square(pred.logit).mean(axis=-1, where=batch.env.legal), valid
        )

        loss_entropy = get_loss_entropy(pred.pi, valid)

        heuristic_target = jax.nn.one_hot(
            batch.env.heuristic_action.clip(min=0), pred.logit.shape[-1]
        )
        loss_heuristic = -renormalize(
            (pred.log_pi * heuristic_target).sum(axis=-1),
            valid & (batch.env.heuristic_action >= 0),
        )

        loss = (
            config.value_loss_coef * loss_v
            # + config.policy_loss_coef * loss_nerd
            # + config.entropy_loss_coef * loss_entropy
            + loss_heuristic
            # + 1e-5 * loss_norm
        )

        move_entropy = get_loss_entropy(
            pred.pi[..., :4],
            valid & batch.env.legal[..., :4].any(axis=-1),
        )
        switch_entropy = get_loss_entropy(
            pred.pi[..., 4:],
            valid & batch.env.legal[..., 4:].any(axis=-1),
        )

        logs["loss_v"] = loss_v
        logs["loss_nerd"] = loss_nerd
        logs["loss_entropy"] = loss_entropy
        logs["move_entropy"] = move_entropy
        logs["switch_entropy"] = switch_entropy
        logs["loss_norm"] = loss_norm

        return loss, logs

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, logs), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    new_params_target = optax.incremental_update(
        new_tensors=state.params,
        old_tensors=state.params_target,
        step_size=config.target_network_avg,
    )
    state = state.replace(
        params_target=new_params_target,
        actor_steps=state.actor_steps + batch.env.valid.sum(),
        learner_steps=state.learner_steps + 1,
    )

    valid = batch.env.valid
    lengths = valid.sum(0)

    can_move = batch.env.legal[..., :4].any(axis=-1)
    can_switch = batch.env.legal[..., 4:].any(axis=-1)

    move_ratio = renormalize(batch.actor.action < 4, can_switch & valid)
    switch_ratio = renormalize(batch.actor.action >= 4, can_move & valid)

    extra_logs = dict(
        actor_steps=valid.sum(),
        loss=loss_val,
        trajectory_length_mean=lengths.mean(),
        trajectory_length_min=lengths.min(),
        trajectory_length_max=lengths.max(),
        early_finish_ratio=(
            jnp.abs(batch.actor.win_rewards[..., 1] * batch.env.valid).sum(0) < 1
        ).mean(),
        reward_sum=jnp.abs(batch.actor.win_rewards[..., 1] * batch.env.valid)
        .sum(0)
        .mean(),
        gradient_norm=optax.global_norm(grads),
        param_norm=optax.global_norm(state.params),
        move_ratio=move_ratio,
        switch_ratio=switch_ratio,
    )
    logs.update(extra_logs)

    return state, logs
