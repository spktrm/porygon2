import functools
import os
from pprint import pprint
from typing import Any, Callable, Literal

import chex
import cloudpickle as pickle
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb.wandb_run
from flax import core, struct
from flax.training import train_state

from rl.environment.interfaces import (
    BuilderActorInput,
    BuilderActorOutput,
    PlayerActorInput,
    PlayerActorOutput,
    Trajectory,
)
from rl.environment.utils import get_ex_builder_step, get_ex_player_step
from rl.learner.league import League
from rl.model.heads import HeadParams
from rl.model.utils import Params, ParamsContainer, get_most_recent_file


@chex.dataclass(frozen=True)
class AdamConfig:
    """Adam optimizer related params."""

    b1: float
    b2: float
    eps: float


GenT = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9]


@chex.dataclass(frozen=True)
class Porygon2LearnerConfig:
    num_steps = 5_000_000
    num_actors: int = 16
    num_eval_actors: int = 5
    unroll_length: int = 128
    replay_buffer_capacity: int = 512

    # Num metagame tokens
    metagame_vocab_size: int = 32

    # False for the beginning
    builder_start_step: int = 100_000

    # Self-play evaluation params
    save_interval_steps: int = 20_000
    new_player_interval: int = 10_000
    league_size: int = 16

    # Batch iteration params
    batch_size: int = 4
    target_replay_ratio: float = 2.5

    # Learning params
    adam: AdamConfig = AdamConfig(b1=0.9, b2=0.999, eps=1e-6)
    learning_rate: float = 3e-5
    player_clip_gradient: float = 1.0
    builder_clip_gradient: float = 1.0
    tau: float = 1e-3

    # Discount params
    player_gamma: float = 1.0
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
    generation: GenT = 9


def get_learner_config():
    return Porygon2LearnerConfig()


class Porygon2PlayerTrainState(train_state.TrainState):
    apply_fn: Callable[
        [Params, PlayerActorInput, PlayerActorOutput, HeadParams], PlayerActorOutput
    ] = struct.field(pytree_node=False)

    target_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    num_steps: int = 0
    num_samples: int = 0
    actor_steps: int = 0

    target_adv_mean: float = 0
    target_adv_std: float = 1


class Porygon2BuilderTrainState(train_state.TrainState):
    apply_fn: Callable[
        [Params, BuilderActorInput, BuilderActorOutput, HeadParams], BuilderActorOutput
    ] = struct.field(pytree_node=False)

    target_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    actor_steps: int = 0

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
        lambda x: jnp.asarray(x[:, 0]), get_ex_player_step()
    )
    ex_builder_actor_inp, ex_builder_actor_out = jax.tree.map(
        lambda x: jnp.asarray(x[:, 0]), get_ex_builder_step()
    )

    player_params_init_fn = lambda: functools.partial(
        player_network.init,
        head_params=HeadParams(),
    )(rng, ex_player_actor_inp, ex_player_actor_out)
    builder_params_init_fn = lambda: functools.partial(
        builder_network.init,
        head_params=HeadParams(),
    )(rng, ex_builder_actor_inp, ex_builder_actor_out)

    player_train_state = Porygon2PlayerTrainState.create(
        apply_fn=jax.vmap(player_network.apply, in_axes=(None, 1, 1, None), out_axes=1),
        params=player_params_init_fn(),
        target_params=player_params_init_fn(),
        tx=optax.chain(
            optax.clip_by_global_norm(config.player_clip_gradient),
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
        apply_fn=jax.vmap(
            builder_network.apply, in_axes=(None, 1, 1, None), out_axes=1
        ),
        params=builder_params_init_fn(),
        target_params=builder_params_init_fn(),
        tx=optax.chain(
            optax.clip_by_global_norm(config.builder_clip_gradient),
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
    league: League,
):
    save_path = save_train_state_locally(
        learner_config, player_state, builder_state, league
    )
    wandb_run.log_artifact(
        artifact_or_path=save_path,
        name=f"latest-gen{learner_config.generation}",
        type="model",
    )


def save_train_state_locally(
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState = None,
    builder_state: Porygon2BuilderTrainState = None,
    league: League = None,
    batch: Trajectory = None,
):
    save_path = os.path.abspath(
        f"ckpts/gen{learner_config.generation}/ckpt_{player_state.num_steps:08}"
    )
    return save_state(
        save_path, learner_config, player_state, builder_state, league, batch
    )


def save_state(
    save_path: str,
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState = None,
    builder_state: Porygon2BuilderTrainState = None,
    league: League = None,
    batch: Trajectory = None,
):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = dict(learner_config=learner_config)
    if player_state is not None:
        data["player_state"] = dict(
            params=player_state.params,
            target_params=player_state.target_params,
            opt_state=player_state.opt_state,
            num_steps=player_state.num_steps,
            num_samples=player_state.num_samples,
            actor_steps=player_state.actor_steps,
            target_adv_mean=player_state.target_adv_mean,
            target_adv_std=player_state.target_adv_std,
        )
    if builder_state is not None:
        data["builder_state"] = dict(
            params=builder_state.params,
            target_params=builder_state.target_params,
            opt_state=builder_state.opt_state,
            actor_steps=builder_state.actor_steps,
            target_adv_mean=builder_state.target_adv_mean,
            target_adv_std=builder_state.target_adv_std,
        )
    if league is not None:
        data["league"] = league.serialize()
    if batch is not None:
        data["batch"] = batch
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    return save_path


def load_train_state(
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
    from_scratch: bool = False,
):
    save_path = f"./ckpts/gen{learner_config.generation}/"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    latest_ckpt = get_most_recent_file(save_path)
    if not latest_ckpt or from_scratch:
        if from_scratch:
            print("Starting training from scratch.")
        initial_params = ParamsContainer(
            frame_count=np.array(player_state.actor_steps).item(),
            step_count=np.array(player_state.num_steps).item(),
            player_params=player_state.params,
            builder_params=builder_state.params,
        )
        league = League(
            main_player=initial_params,
            players=[initial_params],
            league_size=learner_config.league_size,
        )
        return player_state, builder_state, league

    print(f"loading checkpoint from {latest_ckpt}")
    with open(latest_ckpt, "rb") as f:
        ckpt_data = pickle.load(f)

    print("Checkpoint data:")
    ckpt_player_state = ckpt_data.get("player_state", {})
    ckpt_builder_state = ckpt_data.get("builder_state", {})
    ckpt_league_bytes = ckpt_data.get("league", None)

    if ckpt_league_bytes is not None:
        league = League.deserialize(ckpt_league_bytes)
    else:
        initial_params = ParamsContainer(
            frame_count=np.array(player_state.actor_steps).item(),
            step_count=np.array(player_state.num_steps).item(),
            player_params=player_state.params,
            builder_params=builder_state.params,
        )
        league = League(
            main_player=initial_params,
            players=[initial_params],
            league_size=learner_config.league_size,
        )

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
        actor_steps=ckpt_builder_state["actor_steps"],
        target_adv_mean=ckpt_builder_state.get("target_adv_mean", 0.0),
        target_adv_std=ckpt_builder_state.get("target_adv_std", 1.0),
    )

    return player_state, builder_state, league
