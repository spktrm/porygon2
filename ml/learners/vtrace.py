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
from ml.func import get_loss_entropy, get_loss_nerd, get_loss_v, reg_v_trace
from ml.learners.func import (
    collect_gradient_telemetry_data,
    collect_loss_value_telemetry_data,
    collect_policy_stats_telemetry_data,
    collect_value_stats_telemetry_data,
)
from ml.learners.rnad import NerdConfig
from ml.utils import Params
from rlenv.env import get_ex_step
from rlenv.interfaces import ModelOutput, TimeStep


@chex.dataclass(frozen=True)
class VtraceConfig(ActorCriticConfig):
    entropy_loss_coef: float = 1e-3
    target_network_avg: float = 1e-2

    nerd: NerdConfig = NerdConfig()
    clip_gradient: float = 1.0


def get_config():
    return VtraceConfig()


class TrainState(train_state.TrainState):

    params_prev: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    params_target: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    learner_steps: int = 0
    actor_steps: int = 0


def create_train_state(module: nn.Module, rng: PRNGKey, config: ActorCriticConfig):
    """Creates an initial `TrainState`."""
    ex = get_ex_step()

    params = module.init(rng, ex)
    params_target = module.init(rng, ex)
    params_prev = module.init(rng, ex)

    tx = optax.chain(
        optax.adam(
            learning_rate=config.learning_rate,
            eps_root=0.0,
            b1=config.adam.b1,
            b2=config.adam.b2,
            eps=config.adam.eps,
        ),
        optax.clip_by_global_norm(config.clip_gradient),
        # optax.clip(config.clip_gradient),
    )

    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        params_target=params_target,
        params_prev=params_prev,
        tx=optax.MultiSteps(
            tx, every_k_schedule=config.batch_size // config.minibatch_size
        ),
    )


def save(state: TrainState):
    with open(os.path.abspath(f"ckpts/ckpt_{state.learner_steps:08}"), "wb") as f:
        pickle.dump(
            dict(
                params=state.params,
                params_target=state.params_target,
                params_prev=state.params_prev,
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
        params_prev=step["params_prev"],
        params_target=step["params_target"],
        opt_state=step["opt_state"],
        step=step["step"],
    )

    return state


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(state: TrainState, batch: TimeStep, config: VtraceConfig):
    """Train for a single step."""

    def loss_fn(params: Params):
        # Define a checkpointed function
        def rollout_fn(params, env):
            return jax.vmap(jax.vmap(state.apply_fn, (None, 0)), (None, 0))(params, env)

        pred: ModelOutput = rollout_fn(params, batch.env)
        pred_targ: ModelOutput = rollout_fn(state.params_target, batch.env)
        pred_prev: ModelOutput = rollout_fn(state.params_prev, batch.env)

        logs = {}

        policy_pprocessed = config.finetune(
            pred.pi, batch.env.legal, state.learner_steps
        )

        valid = batch.env.valid

        v_target_list, has_played_list, v_trace_policy_target_list = [], [], []
        action_oh = jax.nn.one_hot(batch.actor.action, batch.actor.policy.shape[-1])

        rewards = batch.actor.win_rewards

        for player in range(config.num_players):
            reward = rewards[:, :, player]  # [T, B, Player]
            v_target_, has_played, policy_target_ = reg_v_trace(
                pred_targ.v,
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

        loss_v = get_loss_v(
            [pred.v] * config.num_players,
            v_target_list,
            has_played_list,
        )
        logs.update(
            collect_value_stats_telemetry_data(
                pred.v, v_target_list[0], has_played_list[0]
            )
        )

        ratio = (action_oh * jnp.exp(pred.log_pi - pred_prev.log_pi)).sum(axis=-1)
        # ratio = jnp.ones_like(batch.env.valid)
        is_vector = jnp.expand_dims(ratio, axis=-1)
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
        logs.update(
            collect_policy_stats_telemetry_data(
                pred.logit,
                pred.pi,
                pred.log_pi,
                batch.env.legal,
                batch.env.valid,
                pred_prev.pi,
                ratio,
            )
        )

        loss_entropy = get_loss_entropy(pred.pi, pred.log_pi, batch.env.legal, valid)
        loss = (
            config.value_loss_coef * loss_v
            + config.policy_loss_coef * loss_nerd
            + config.entropy_loss_coef * loss_entropy
        )
        logs.update(collect_loss_value_telemetry_data(loss_v, loss_nerd, loss_entropy))

        return loss, logs

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, logs), grads = grad_fn(state.params)

    state = state.replace(params_prev=state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(
        params_target=optax.incremental_update(
            new_tensors=state.params,
            old_tensors=state.params_target,
            step_size=config.target_network_avg,
        ),
        actor_steps=state.actor_steps + batch.env.valid.sum(),
        learner_steps=state.learner_steps + 1,
    )

    logs.update(dict(loss=loss_val))
    logs.update(collect_gradient_telemetry_data(grads))

    return state, logs
