import os
import pickle
from copy import deepcopy
from pprint import pprint
from typing import Any, Callable

import chex
import flax.linen as nn
import jax
import optax
from chex import PRNGKey
from flax import core, struct
from flax.training import train_state

from rl.environment.interfaces import ModelOutput, TimeStep
from rl.environment.utils import get_ex_step
from rl.model.utils import Params


@chex.dataclass(frozen=True)
class AdamConfig:
    """Adam optimizer related params."""

    b1: float
    b2: float
    eps: float


@chex.dataclass(frozen=True)
class Porygon2LearnerConfig:
    num_steps = 10_000_000
    num_actors: int = 32
    unroll_length: int = 108
    replay_buffer_capacity: int = 16

    # Batch iteration params
    batch_size: int = 4
    target_replay_ratio: int = 2

    # Learning params
    adam: AdamConfig = AdamConfig(b1=0.9, b2=0.999, eps=1e-5)
    learning_rate: float = 5e-5
    clip_gradient: float = 2.0
    tau: float = 1e-3

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


def get_learner_config():
    return Porygon2LearnerConfig()


class Porygon2TrainState(train_state.TrainState):
    apply_fn: Callable[[Params, TimeStep], ModelOutput] = struct.field(
        pytree_node=False
    )

    target_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    num_steps: int = 0
    num_samples: int = 0
    actor_steps: int = 0

    target_adv_mean: float = 0
    target_adv_std: float = 1


def create_train_state(module: nn.Module, rng: PRNGKey, config: Porygon2LearnerConfig):
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

    return Porygon2TrainState.create(
        apply_fn=jax.vmap(module.apply, in_axes=(None, 1), out_axes=1),
        params=params,
        target_params=target_params,
        tx=tx,
    )


def save_train_state(state: Porygon2TrainState):
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


def load_train_state(state: Porygon2TrainState, path: str):
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
