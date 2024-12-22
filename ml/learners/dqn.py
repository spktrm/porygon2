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
from ml.func import get_loss_entropy, renormalize
from ml.utils import Params
from rlenv.env import get_ex_step
from rlenv.interfaces import ModelOutput, TimeStep


@chex.dataclass(frozen=True)
class PPOConfig(ActorCriticConfig):
    entropy_loss_coef: float = 1e-3
    target_network_avg: float = 1e-2
    value_loss_coef: float = 0.5
    clip_eps: float = 0.2


def get_config():
    return PPOConfig()


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
        optax.adam(
            learning_rate=config.learning_rate,
            eps_root=0.0,
            b1=config.adam.b1,
            b2=config.adam.b2,
            eps=config.adam.eps,
            # weight_decay=config.adam.weight_decay,
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


def calculate_gae(
    r_t: chex.Array,
    discount_t: chex.Array,
    v_t: chex.Array,
    lambda_: float = 1.0,
    stop_target_gradients: bool = False,
) -> chex.Array:
    # If scalar make into vector.
    lambda_ = jnp.ones_like(discount_t) * lambda_

    # Work backwards to compute `G_{T-1}`, ..., `G_0`.
    def _body(acc, xs):
        returns, discounts, values, lambda_ = xs
        acc = returns + discounts * ((1 - lambda_) * values + lambda_ * acc)
        return acc, acc

    _, returns = jax.lax.scan(
        _body, v_t[-1], (r_t, discount_t, v_t, lambda_), reverse=True
    )

    return jax.lax.select(
        stop_target_gradients, jax.lax.stop_gradient(returns), returns
    )


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(state: TrainState, batch: TimeStep, config: PPOConfig):
    """Train for a single step."""

    def loss_fn(params: Params):
        rollout = jax.vmap(jax.vmap(state.apply_fn, (None, 0)), (None, 0))

        pred: ModelOutput = rollout(params, batch.env)
        pred_target: ModelOutput = rollout(state.params_target, batch.env)

        q_values = pred.logit
        target_q_values = pred_target.logit[1:]
        target_q_values = jnp.concatenate(
            (target_q_values, jnp.zeros_like(target_q_values[0])[None])
        )

        # Sum a large negative constant to illegal action logits before taking the
        # max. This prevents illegal action values from being considered as target.
        max_next_q = jnp.max(
            target_q_values + (1 - batch.env.legal) * 1e-9,
            axis=-1,
        )
        max_next_q = jnp.where(
            1 - batch.env.valid, max_next_q, jnp.zeros_like(max_next_q)
        )
        target = batch.env.win_rewards + (1 - batch.env.valid) * 1.0 * max_next_q
        target = jax.lax.stop_gradient(target)
        predictions = jnp.sum(
            q_values * jax.nn.one_hot(batch.actor.action, pred.pi.shape[-1]), axis=-1
        )
        loss = renormalize((predictions - target) ** 2, batch.env.valid)

        logs = {}

        move_entropy = get_loss_entropy(
            pred.pi[..., :4],
            pred.log_pi[..., :4],
            batch.env.legal[..., :4],
            batch.env.valid & batch.env.legal[..., :4].any(axis=-1),
        )
        switch_entropy = get_loss_entropy(
            pred.pi[..., 4:],
            pred.log_pi[..., 4:],
            batch.env.legal[..., 4:],
            batch.env.valid & batch.env.legal[..., 4:].any(axis=-1),
        )

        logs["loss"] = loss
        logs["move_entropy"] = move_entropy
        logs["switch_entropy"] = switch_entropy

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
            jnp.abs(batch.actor.win_rewards * batch.env.valid).sum(0) != 1
        ).mean(),
        reward_sum=jnp.abs(batch.actor.win_rewards * batch.env.valid).sum(0).mean(),
        gradient_norm=optax.global_norm(grads),
        param_norm=optax.global_norm(state.params),
        move_ratio=move_ratio,
        switch_ratio=switch_ratio,
    )
    logs.update(extra_logs)

    return state, logs
