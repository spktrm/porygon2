import os
import pickle
from pprint import pprint
from typing import Any, Callable, Literal

import chex
import flax.linen as nn
import jax
import optax
import wandb.wandb_run
from flax import core, struct
from flax.training import train_state

from rl.environment.interfaces import (
    BuilderActorInput,
    BuilderActorOutput,
    PlayerActorInput,
    PlayerActorOutput,
)
from rl.environment.utils import get_ex_builder_step, get_ex_player_step
from rl.model.utils import Params, get_most_recent_file


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
    num_eval_actors: int = 5
    unroll_length: int = 128 * 2
    replay_buffer_capacity: int = 512

    # Batch iteration params
    batch_size: int = 4
    target_replay_ratio: float = 2.5

    # Learning params
    adam: AdamConfig = AdamConfig(b1=0.9, b2=0.999, eps=1e-5)
    learning_rate: float = 5e-5
    clip_gradient: float = 2.0
    tau: float = 1e-3

    # Discount params
    player_lambda_: float = 0.95
    player_gamma: float = 1.0

    builder_lambda_: float = 0.90
    builder_gamma: float = 1.0

    # Vtrace params
    clip_rho_threshold: float = 1.0
    clip_pg_rho_threshold: float = 1.0
    clip_ppo: float = 0.3

    # Loss coefficients
    player_value_loss_coef: float = 0.5
    player_policy_loss_coef: float = 1.0
    player_entropy_loss_coef: float = 0.05
    player_kl_loss_coef: float = 0.05

    builder_value_loss_coef: float = 0.5
    builder_policy_loss_coef: float = 1.0
    builder_entropy_loss_coef: float = 0.05
    builder_kl_loss_coef: float = 0.05

    # Smogon Generation
    generation: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9] = 9


def get_learner_config():
    return Porygon2LearnerConfig()


class Porygon2PlayerTrainState(train_state.TrainState):
    apply_fn: Callable[
        [Params, PlayerActorInput, PlayerActorOutput], PlayerActorOutput
    ] = struct.field(pytree_node=False)

    target_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    num_steps: int = 0
    num_samples: int = 0
    actor_steps: int = 0

    target_adv_mean: float = 0
    target_adv_std: float = 1


class Porygon2BuilderTrainState(train_state.TrainState):
    apply_fn: Callable[
        [Params, BuilderActorInput, BuilderActorOutput], BuilderActorOutput
    ] = struct.field(pytree_node=False)
    target_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    target_adv_mean: float = 0
    target_adv_std: float = 1


def create_train_state(
    player_network: nn.Module,
    builder_network: nn.Module,
    rng: jax.Array,
    config: Porygon2LearnerConfig,
):
    """Creates an initial `TrainState`."""
    ex_player_actor_inp, ex_player_actor_out = jax.tree.map(
        lambda x: x[:, 0], get_ex_player_step()
    )
    ex_builder_actor_inp, ex_builder_actor_out = jax.tree.map(
        lambda x: x[:, 0], get_ex_builder_step()
    )

    player_params = player_network.init(rng, ex_player_actor_inp, ex_player_actor_out)
    builder_params = builder_network.init(
        rng, ex_builder_actor_inp, ex_builder_actor_out
    )

    player_train_state = Porygon2PlayerTrainState.create(
        apply_fn=jax.vmap(player_network.apply, in_axes=(None, 1, 1), out_axes=1),
        params=player_params,
        target_params=player_params,
        tx=optax.chain(
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
        ),
    )

    builder_train_state = Porygon2BuilderTrainState.create(
        apply_fn=jax.vmap(builder_network.apply, in_axes=(None, 1, 1), out_axes=1),
        params=builder_params,
        target_params=builder_params,
        tx=optax.chain(
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
        ),
    )

    return player_train_state, builder_train_state


def save_train_state(
    wandb_run: wandb.wandb_run.Run,
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
):
    save_path = save_train_state_locally(learner_config, player_state, builder_state)
    wandb_run.log_artifact(
        artifact_or_path=save_path,
        name=f"latest-gen{learner_config.generation}",
        type="model",
    )


def save_train_state_locally(
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
):
    save_path = os.path.abspath(
        f"ckpts/gen{learner_config.generation}/ckpt_{player_state.num_steps:08}"
    )
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(
            dict(
                player_state=dict(
                    params=player_state.params,
                    target_params=player_state.target_params,
                    opt_state=player_state.opt_state,
                    num_steps=player_state.num_steps,
                    num_samples=player_state.num_samples,
                    actor_steps=player_state.actor_steps,
                    target_adv_mean=player_state.target_adv_mean,
                    target_adv_std=player_state.target_adv_std,
                ),
                builder_state=dict(
                    params=builder_state.params,
                    target_params=builder_state.target_params,
                    opt_state=builder_state.opt_state,
                    target_adv_mean=builder_state.target_adv_mean,
                    target_adv_std=builder_state.target_adv_std,
                ),
            ),
            f,
        )
    return save_path


def load_train_state(
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
):
    save_path = f"./ckpts/gen{learner_config.generation}/"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    latest_ckpt = get_most_recent_file(save_path)
    if not latest_ckpt:
        return player_state, builder_state

    print(f"loading checkpoint from {latest_ckpt}")
    with open(latest_ckpt, "rb") as f:
        ckpt_data = pickle.load(f)

    print("Checkpoint data:")
    ckpt_player_state = ckpt_data.get("player_state", {})
    ckpt_builder_state = ckpt_data.get("builder_state", {})

    pprint(
        {
            k: v
            for k, v in ckpt_player_state.items()
            if k not in ["opt_state", "params", "target_params"]
        }
    )
    pprint(
        {
            k: v
            for k, v in ckpt_builder_state.items()
            if k not in ["opt_state", "params", "target_params"]
        }
    )

    player_state = player_state.replace(
        params=ckpt_player_state["params"],
        target_params=ckpt_player_state["target_params"],
        opt_state=ckpt_player_state["opt_state"],
        num_steps=ckpt_player_state["num_steps"],
        num_samples=ckpt_player_state["num_samples"],
        actor_steps=ckpt_player_state["actor_steps"],
        target_adv_mean=ckpt_player_state.get("target_adv_mean", 0.0),
        target_adv_std=ckpt_player_state.get("target_adv_std", 1.0),
    )

    builder_state = builder_state.replace(
        params=ckpt_builder_state["params"],
        target_params=ckpt_builder_state["target_params"],
        opt_state=ckpt_builder_state["opt_state"],
        target_adv_mean=ckpt_builder_state.get("target_adv_mean", 0.0),
        target_adv_std=ckpt_builder_state.get("target_adv_std", 1.0),
    )

    return player_state, builder_state
