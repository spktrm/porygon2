import json
import os
import pickle
from typing import Any, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import PRNGKey
from flax import core, struct
from flax.training import train_state
from tqdm import tqdm, trange

import wandb
from ml.arch.config import get_model_cfg
from ml.arch.model import get_model, get_num_params
from ml.config import ActorCriticConfig
from ml.learners.buffer import OfflineReplayBuffer
from ml.learners.func import collect_parameter_and_gradient_telemetry_data
from ml.utils import Params
from rlenv.data import NUM_ACTIONS
from rlenv.env import get_ex_step
from rlenv.interfaces import EnvStep, ModelOutput


class TrainState(train_state.TrainState):

    params_target: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    actor_steps: int = 0


def create_train_state(module: nn.Module, rng: PRNGKey):
    """Creates an initial `TrainState`."""
    ex = get_ex_step()

    ActorCriticConfig()

    params = module.init(rng, ex)
    params_target = module.init(rng, ex)
    tx = optax.chain(
        optax.adamw(learning_rate=3e-4, b1=0.9, b2=0.999, weight_decay=1e-5),
    )

    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        params_target=params_target,
        tx=tx,
    )


@jax.jit
def train_step(state: TrainState, batch: Tuple[EnvStep, chex.Array, chex.Array]):
    """Train for a single step."""

    samples, targets, labels, values = batch

    def loss_fn(params: Params):
        logs = {}

        rollout = jax.vmap(state.apply_fn, (None, 0))
        pred: ModelOutput = rollout(params, samples)
        head_loss = -(pred.log_pi * targets).sum(axis=-1).mean()

        onehot_labels = jax.nn.one_hot(labels, NUM_ACTIONS)
        offline_head_loss = -(pred.offline_log_pi * onehot_labels).sum(axis=-1).mean()

        value_loss = jnp.square(pred.v.squeeze() - values).mean()

        loss = offline_head_loss  # + head_loss + value_loss

        logs["head_loss"] = head_loss
        logs["offline_head_loss"] = offline_head_loss
        logs["value_loss"] = value_loss

        return loss, logs

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, logs), grads = grad_fn(state.params)
    logs.update(dict(loss=loss_val))

    state = state.apply_gradients(grads=grads)
    logs.update(collect_parameter_and_gradient_telemetry_data(grads))

    logs = {f"train_{k}": v for k, v in logs.items()}

    return state, logs


@jax.jit
def val_step(state: TrainState, batch: Tuple[EnvStep, chex.Array, chex.Array]):
    """Train for a single step."""

    samples, targets, labels, values = batch

    def loss_fn(params: Params):
        logs = {}

        rollout = jax.vmap(state.apply_fn, (None, 0))
        pred: ModelOutput = rollout(params, samples)
        head_loss = -(pred.log_pi * targets).sum(axis=-1).mean()

        onehot_labels = jax.nn.one_hot(labels, NUM_ACTIONS)
        offline_head_loss = -(pred.offline_log_pi * onehot_labels).sum(axis=-1).mean()

        value_loss = jnp.square(pred.v.squeeze() - values).mean()

        loss = offline_head_loss  # + head_loss + value_loss

        logs["head_loss"] = head_loss
        logs["offline_head_loss"] = offline_head_loss
        logs["value_loss"] = value_loss

        return loss, logs

    (loss_val, logs) = loss_fn(state.params)
    logs.update(dict(loss=loss_val))
    logs = {f"val_{k}": v for k, v in logs.items()}

    return state, logs


def save(state: TrainState):
    with open(os.path.abspath(f"ckpts/ckpt_{state.step:08}"), "wb") as f:
        pickle.dump(
            dict(
                params=state.params,
                opt_state=state.opt_state,
                step=state.step,
            ),
            f,
        )


def main():
    training_buffer = OfflineReplayBuffer.from_replay_dir(
        "replays/data/gen3randombattle/"
    )
    training_buffer, validation_buffer = OfflineReplayBuffer.split_train_test(
        training_buffer
    )

    model_config = get_model_cfg()

    network = get_model(model_config)

    state = create_train_state(network, jax.random.PRNGKey(42))

    wandb.init(
        project="pokemon-rl",
        config={
            "num_params": get_num_params(state.params),
            "model_config": json.loads(model_config.to_json_best_effort()),
        },
    )

    ti = 0
    vi = 0

    for epoch in trange(0, 64, desc="epochs: "):

        for batch in tqdm(
            iter(training_buffer),
            total=len(training_buffer) // training_buffer.batch_size,
        ):

            state, logs = train_step(state, batch)

            logs["train_idx"] = ti
            logs["epoch"] = epoch
            wandb.log(logs)
            ti += 1

        save(state)

        for batch in tqdm(
            iter(validation_buffer),
            total=len(validation_buffer) // validation_buffer.batch_size,
        ):

            _, logs = val_step(state, batch)

            logs["val_idx"] = vi
            logs["epoch"] = epoch
            wandb.log(logs)
            vi += 1

    print("done")


if __name__ == "__main__":
    main()
