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

from ml.config import ActorCriticConfig
from ml.func import (
    _player_others,
    get_loss_entropy,
    get_loss_nerd,
    get_loss_v_huber,
    rnad_v_trace,
)
from ml.learners.func import (
    collect_loss_value_telemetry_data,
    collect_parameter_and_gradient_telemetry_data,
    collect_policy_stats_telemetry_data,
    collect_regularisation_telemetry_data,
    collect_value_stats_telemetry_data,
)
from ml.utils import Params
from rlenv.env import get_ex_step
from rlenv.interfaces import ModelOutput, TimeStep


@chex.dataclass(frozen=True)
class NerdConfig:
    """Nerd related params."""

    beta: float = 2
    clip: float = 10


@chex.dataclass(frozen=True)
class VtraceConfig(ActorCriticConfig):
    entropy_loss_coef: float = 1e-2
    target_network_avg: float = 1e-3

    nerd: NerdConfig = NerdConfig()
    clip_gradient: float = 200


def get_config():
    return VtraceConfig()


class TrainState(train_state.TrainState):

    params_target: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    params_reg: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    actor_steps: int = 0


def create_train_state(module: nn.Module, rng: PRNGKey, config: ActorCriticConfig):
    """Creates an initial `TrainState`."""
    ex, hx = get_ex_step()

    params = module.init(rng, ex, hx)
    params_target = module.init(rng, ex, hx)
    params_reg = module.init(rng, ex, hx)

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

    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        params_target=params_target,
        params_reg=params_reg,
        tx=tx,
    )


def save(state: TrainState):
    with open(os.path.abspath(f"ckpts/ckpt_{state.step:08}"), "wb") as f:
        pickle.dump(
            dict(
                params=state.params,
                params_target=state.params_target,
                params_reg=state.params_reg,
                opt_state=state.opt_state,
                step=state.step,
            ),
            f,
        )


def load(state: TrainState, path: str):
    print(f"loading checkpoint from {path}")
    with open(path, "rb") as f:
        step = pickle.load(f)

    step_no = step.get("step", 0)

    params = step["params"]
    state = state.replace(step=step["step"], params=params)

    if step_no > 0:
        print(f"Learner steps: {step_no:08}")
        print(f"Loading target and regularisation nets")
        print(f"Loading optimizer state")

        params_target = step.get("params_target", params)
        params_reg = step.get("params_reg", params_target)

        state = state.replace(
            params_target=params_target,
            params_reg=params_reg,
            opt_state=step["opt_state"],
        )

    return state


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(state: TrainState, batch: TimeStep, config: VtraceConfig):
    """Train for a single step."""

    def loss_fn(params: Params):
        # Define a checkpointed function
        def rollout_fn(model_params):
            return jax.vmap(jax.vmap(state.apply_fn, (None, 0, 0)), (None, 0, 0))(
                model_params, batch.env, batch.history
            )

        pred: ModelOutput = rollout_fn(params)
        pred_targ: ModelOutput = rollout_fn(state.params_target)
        pred_reg: ModelOutput = rollout_fn(state.params_reg)

        logs = {}

        policy_pprocessed = config.finetune(pred.pi, batch.env.legal, state.step)

        log_policy_reg = pred.log_pi - pred_reg.log_pi
        logs.update(
            collect_regularisation_telemetry_data(
                pred.pi, log_policy_reg, batch.env.legal, batch.env.valid
            )
        )

        valid = batch.env.valid

        v_target_list, has_played_list, v_trace_policy_target_list = [], [], []
        action_oh = jax.nn.one_hot(batch.actor.action, batch.actor.policy.shape[-1])

        rewards = batch.actor.rewards.win_rewards

        for player in range(config.num_players):
            reward = rewards[:, :, player]  # [T, B, Player]
            v_target_, has_played, policy_target_ = rnad_v_trace(
                pred_targ.v,
                valid,
                batch.env.player_id,
                batch.actor.policy,
                policy_pprocessed,
                log_policy_reg,
                _player_others(batch.env.player_id, valid, player),
                action_oh,
                reward,
                player,
                lambda_=1.0,
                c=config.c_vtrace,
                rho=jnp.inf,
                eta=0.1,
                gamma=config.gamma,
            )
            v_target_list.append(jax.lax.stop_gradient(v_target_))
            has_played_list.append(jax.lax.stop_gradient(has_played))
            v_trace_policy_target_list.append(jax.lax.stop_gradient(policy_target_))

        loss_v = get_loss_v_huber(
            [pred.v] * config.num_players,
            v_target_list,
            has_played_list,
        )

        logs.update(
            collect_value_stats_telemetry_data(
                pred.v.squeeze(),
                v_target_list[0].squeeze(),
                has_played_list[0].squeeze(),
            )
        )

        policy_ratio = (action_oh * jnp.exp(pred.log_pi - batch.actor.log_policy)).sum(
            axis=-1
        )
        is_vector = jnp.expand_dims(jnp.ones_like(valid), axis=-1)
        importance_sampling_correction = [is_vector] * config.num_players

        loss_nerd, adv_pi = get_loss_nerd(
            [pred.logit] * config.num_players,
            [pred.pi] * config.num_players,
            v_trace_policy_target_list,
            valid * (batch.env.legal.sum(axis=-1) > 1),
            batch.env.player_id,
            batch.env.legal,
            importance_sampling_correction,
            clip=config.nerd.clip,
            beta=config.nerd.beta,
        )

        logs.update(
            collect_policy_stats_telemetry_data(
                pred.logit,
                pred.pi,
                pred.log_pi,
                batch.env.legal,
                valid,
                pred_targ.pi,
                policy_ratio,
                sum(v_trace_policy_target_list),
                adv_pi,
            )
        )

        loss_entropy = get_loss_entropy(
            pred.pi,
            valid * (batch.env.legal.sum(axis=-1) > 1),
        )
        loss = config.value_loss_coef * loss_v + config.policy_loss_coef * loss_nerd
        logs.update(collect_loss_value_telemetry_data(loss_v, loss_nerd, loss_entropy))

        return loss, logs

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, logs), grads = grad_fn(state.params)

    logs.update(dict(loss=loss_val))
    logs.update(collect_parameter_and_gradient_telemetry_data(state.params, grads))
    # logs.update(efficient_per_module_gradient_stats(grads))

    state = state.apply_gradients(grads=grads)

    ema_val = jnp.maximum(1 / (state.step + 1), config.target_network_avg)
    params_target = optax.incremental_update(
        new_tensors=state.params,
        old_tensors=state.params_target,
        step_size=ema_val,
    )
    params_reg = optax.incremental_update(
        new_tensors=state.params_target,
        old_tensors=state.params_reg,
        step_size=ema_val,
    )
    state = state.replace(
        params_target=params_target,
        params_reg=params_reg,
        actor_steps=state.actor_steps + batch.env.valid.sum(),
    )

    logs["ema_val"] = ema_val

    return state, logs
