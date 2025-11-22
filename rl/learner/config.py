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
    num_eval_actors: int = 0
    num_controlled_eval_actors: int = 0
    unroll_length: int = 128
    replay_buffer_capacity: int = 512

    # Self-play evaluation params
    save_interval_steps: int = 20_000
    league_winrate_log_steps: int = 1_000
    add_player_min_frames: int = int(2e6)
    add_player_max_frames: int = int(3e7)
    league_size: int = 16

    # Controlled evaluation params
    controlled_eval_fixed_species: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    controlled_eval_fixed_sets: tuple[int, ...] = (0, 0, 0, 0, 0, 0)

    # Batch iteration params
    batch_size: int = 4
    target_replay_ratio: float = 4

    # Learning params
    adam: AdamConfig = AdamConfig(b1=0, b2=0.99, eps=1e-6)
    learning_rate: float = 5e-5
    player_clip_gradient: float = 1.0
    builder_clip_gradient: float = 1.0
    mix_noise_ratio: float = 0.5

    # EMA params
    player_ema_decay: float = 1e-3
    builder_ema_decay: float = 1e-4

    # Discount params
    player_gamma: float = 1.0
    builder_gamma: float = 1.0

    # Vtrace params
    player_lambda: float = 0.95
    builder_lambda: float = 0.5
    clip_rho_threshold: float = 1.0
    clip_pg_rho_threshold: float = 1.0
    clip_ppo: float = 0.2
    consistency_coef: float = 0.01

    # Loss coefficients
    player_value_loss_coef: float = 1.0
    player_policy_loss_coef: float = 1.0
    player_kl_loss_coef: float = 0.1
    player_entropy_loss_coef: float = 0.01

    builder_value_loss_coef: float = 0.5
    builder_policy_loss_coef: float = 1.0
    builder_kl_loss_loss_coef: float = 0.1
    builder_entropy_loss_coef: float = 0.1

    # Smogon Generation
    generation: GenT = 9


def get_learner_config():
    return Porygon2LearnerConfig()


class Porygon2PlayerTrainState(train_state.TrainState):
    apply_fn: Callable[
        [Params, PlayerActorInput, PlayerActorOutput, HeadParams], PlayerActorOutput
    ] = struct.field(pytree_node=False)
    init_fn: Callable[[jax.Array], Params] = struct.field(pytree_node=False)

    target_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    num_steps: int = 0
    actor_steps: int = 0

    target_adv_mean: float = 0
    target_adv_std: float = 1


class Porygon2BuilderTrainState(train_state.TrainState):
    apply_fn: Callable[
        [Params, BuilderActorInput, BuilderActorOutput, HeadParams], BuilderActorOutput
    ] = struct.field(pytree_node=False)
    init_fn: Callable[[jax.Array], Params] = struct.field(pytree_node=False)

    target_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    num_steps: int = 0
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

    player_params_init_fn = functools.partial(
        player_network.init,
        head_params=HeadParams(),
        actor_input=ex_player_actor_inp,
        actor_output=ex_player_actor_out,
    )
    player_train_state = Porygon2PlayerTrainState.create(
        apply_fn=jax.vmap(
            player_network.apply,
            in_axes=(None, 1, 1, None),
            out_axes=1,
        ),
        init_fn=player_params_init_fn,
        params=player_params_init_fn(rng),
        target_params=player_params_init_fn(rng),
        tx=optax.chain(
            optax.clip_by_global_norm(config.player_clip_gradient),
            optax.adam(
                learning_rate=config.learning_rate,
                b1=config.adam.b1,
                b2=config.adam.b2,
                eps=config.adam.eps,
            ),
        ),
    )

    builder_params_init_fn = functools.partial(
        builder_network.init,
        actor_input=ex_builder_actor_inp,
        actor_output=ex_builder_actor_out,
        head_params=HeadParams(),
    )
    builder_train_state = Porygon2BuilderTrainState.create(
        apply_fn=jax.vmap(
            builder_network.apply,
            in_axes=(None, 1, 1, None),
            out_axes=1,
        ),
        init_fn=builder_params_init_fn,
        params=builder_params_init_fn(rng),
        target_params=builder_params_init_fn(rng),
        tx=optax.chain(
            optax.clip_by_global_norm(config.builder_clip_gradient),
            optax.adam(
                learning_rate=config.learning_rate,
                b1=config.adam.b1,
                b2=config.adam.b2,
                eps=config.adam.eps,
            ),
        ),
    )

    return player_train_state, builder_train_state


def save_train_state(
    wandb_run: wandb.wandb_run.Run,
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
    league: League,
    metadata: dict | None = None,
):
    save_path = save_train_state_locally(
        learner_config, player_state, builder_state, league, metadata
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
    metadata: dict | None = None,
):
    save_path = os.path.abspath(
        f"ckpts/gen{learner_config.generation}/ckpt_{player_state.num_steps:08}"
    )
    return save_state(
        save_path, learner_config, player_state, builder_state, league, metadata
    )


def save_state(
    save_path: str,
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState = None,
    builder_state: Porygon2BuilderTrainState = None,
    league: League = None,
    metadata: dict | None = None,
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
    if metadata is not None:
        data["metadata"] = metadata
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    return save_path


def load_train_state(
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
    from_scratch: bool = False,
    metadata: dict | None = None,
) -> tuple[Porygon2PlayerTrainState, Porygon2BuilderTrainState, League, dict | None]:
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
        return player_state, builder_state, league, None

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

    metadata = ckpt_data.get("metadata", None)

    return player_state, builder_state, league, metadata
