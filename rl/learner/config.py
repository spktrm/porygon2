import functools
import os
from pprint import pprint
from typing import Any, Callable, Literal

import chex
import cloudpickle as pickle
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
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
from rl.model.utils import Params, get_most_recent_file


@chex.dataclass(frozen=True)
class AdamConfig:
    """Adam optimizer related params."""

    b1: float
    b2: float
    eps: float


GenT = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9]


@chex.dataclass(frozen=True)
class PlayerLearnerConfig:
    num_steps = 5_000_000
    num_actors: int = 16
    num_eval_actors: int = 0
    unroll_length: int = 128
    replay_buffer_capacity: int = 512

    # Batch iteration params
    batch_size: int = 4
    target_replay_ratio: float = 4

    # Learning params
    adam: AdamConfig = AdamConfig(b1=0, b2=0.99, eps=1e-6)
    learning_rate: float = 2e-5
    clip_gradient: float = 1.0

    # EMA params
    ema_decay: float = 1e-3

    # Discount params
    gamma: float = 1.0

    # Vtrace params
    lambda_: float = 0.95
    clip_rho_threshold: float = 1.0
    clip_pg_rho_threshold: float = 1.0
    clip_ppo: float = 0.2

    # Loss coefficients
    value_loss_coef: float = 1.0
    policy_loss_coef: float = 1.0
    kl_loss_coef: float = 0.1
    entropy_loss_coef: float = 0.01


@chex.dataclass(frozen=True)
class BuilderLearnerConfig:
    num_steps = 5_000_000
    num_actors: int = 4
    unroll_length: int = 7
    replay_buffer_capacity: int = 512

    # Batch iteration params
    batch_size: int = 4
    target_replay_ratio: float = 4

    # Learning params
    adam: AdamConfig = AdamConfig(b1=0, b2=0.99, eps=1e-6)
    learning_rate: float = 2e-5
    clip_gradient: float = 1.0

    # EMA params
    ema_decay: float = 1e-3

    # Discount params
    gamma: float = 1.0

    # Vtrace params
    lambda_: float = 1.0
    clip_rho_threshold: float = 1.0
    clip_pg_rho_threshold: float = 1.0
    clip_ppo: float = 0.2

    # Loss coefficients
    value_loss_coef: float = 0.5
    policy_loss_coef: float = 1.0
    kl_loss_loss_coef: float = 0.1
    entropy_loss_coef: float = 0.01


@chex.dataclass(frozen=True)
class LearnerConfig:
    player_config: PlayerLearnerConfig = PlayerLearnerConfig()
    builder_config: BuilderLearnerConfig = BuilderLearnerConfig()

    # Smogon Generation
    generation: GenT = 9

    # Self-play evaluation params
    save_interval_steps: int = 20_000
    league_winrate_log_steps: int = 1_000
    add_player_min_frames: int = int(2e6)
    add_player_max_frames: int = int(3e8)
    league_size: int = 16


def get_learner_config():
    return LearnerConfig()


class PlayerTrainState(train_state.TrainState):
    apply_fn: Callable[
        [Params, PlayerActorInput, PlayerActorOutput, HeadParams], PlayerActorOutput
    ] = struct.field(pytree_node=False)
    init_fn: Callable[[jax.Array], Params] = struct.field(pytree_node=False)

    target_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    step_count: int = 0
    frame_count: int = 0

    target_adv_mean: float = 0
    target_adv_std: float = 1


class BuilderTrainState(train_state.TrainState):
    apply_fn: Callable[
        [Params, BuilderActorInput, BuilderActorOutput, HeadParams], BuilderActorOutput
    ] = struct.field(pytree_node=False)
    init_fn: Callable[[jax.Array], Params] = struct.field(pytree_node=False)

    target_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    step_count: int = 0
    frame_count: int = 0

    target_adv_mean: float = 0
    target_adv_std: float = 1


def create_train_state(
    player_network: nn.Module,
    builder_network: nn.Module,
    rng: jax.Array,
    config: PlayerLearnerConfig,
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
    player_train_state = PlayerTrainState.create(
        apply_fn=jax.vmap(
            player_network.apply,
            in_axes=(None, 1, 1, None),
            out_axes=1,
        ),
        init_fn=player_params_init_fn,
        params=player_params_init_fn(rng),
        target_params=player_params_init_fn(rng),
        tx=optax.chain(
            optax.clip_by_global_norm(config.clip_gradient),
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
    builder_train_state = BuilderTrainState.create(
        apply_fn=jax.vmap(
            builder_network.apply,
            in_axes=(None, 1, 1, None),
            out_axes=1,
        ),
        init_fn=builder_params_init_fn,
        params=builder_params_init_fn(rng),
        target_params=builder_params_init_fn(rng),
        tx=optax.chain(
            optax.clip_by_global_norm(config.clip_gradient),
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
    learner_config: PlayerLearnerConfig,
    player_state: PlayerTrainState,
    builder_state: BuilderTrainState,
    league: League,
):
    save_train_state_locally(learner_config, player_state, builder_state, league)


def save_train_state_locally(
    learner_config: LearnerConfig,
    player_state: PlayerTrainState,
    builder_state: BuilderTrainState,
    league: League,
):
    save_path = os.path.abspath(
        f"ckpts/gen{learner_config.generation}/ckpt_{player_state.step_count:08}"
    )
    return save_state(save_path, learner_config, player_state, builder_state, league)


def save_state(
    save_path: str,
    learner_config: PlayerLearnerConfig,
    player_state: PlayerTrainState,
    builder_state: BuilderTrainState,
    league: League,
):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = dict(learner_config=learner_config)
    data["player_state"] = dict(
        params=player_state.params,
        target_params=player_state.target_params,
        opt_state=player_state.opt_state,
        step_count=player_state.step_count,
        frame_count=player_state.frame_count,
        target_adv_mean=player_state.target_adv_mean,
        target_adv_std=player_state.target_adv_std,
    )

    data["builder_state"] = dict(
        params=builder_state.params,
        target_params=builder_state.target_params,
        opt_state=builder_state.opt_state,
        step_count=builder_state.step_count,
        frame_count=builder_state.frame_count,
        target_adv_mean=builder_state.target_adv_mean,
        target_adv_std=builder_state.target_adv_std,
    )
    data["league"] = league.serialize()

    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    return save_path


def load_train_state(
    learner_config: LearnerConfig,
    player_state: PlayerTrainState,
    builder_state: BuilderTrainState,
) -> tuple[PlayerTrainState, BuilderTrainState, League]:
    save_path = f"./ckpts/gen{learner_config.generation}/"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    latest_ckpt = get_most_recent_file(save_path)

    print(f"loading checkpoint from {latest_ckpt}")
    with open(latest_ckpt, "rb") as f:
        ckpt_data = pickle.load(f)

    ckpt_player_state = ckpt_data.get("player_state")
    ckpt_builder_state = ckpt_data.get("builder_state")
    ckpt_league_bytes = ckpt_data.get("league")

    league = League.deserialize(ckpt_league_bytes)

    print("Checkpoint data:")
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
        step_count=ckpt_player_state["step_count"],
        frame_count=ckpt_player_state["frame_count"],
        target_adv_mean=ckpt_player_state.get("target_adv_mean", 0.0),
        target_adv_std=ckpt_player_state.get("target_adv_std", 1.0),
    )

    builder_state = builder_state.replace(
        params=ckpt_builder_state["params"],
        target_params=ckpt_builder_state["target_params"],
        opt_state=ckpt_builder_state["opt_state"],
        step_count=ckpt_builder_state["step_count"],
        frame_count=ckpt_builder_state["frame_count"],
        target_adv_mean=ckpt_builder_state.get("target_adv_mean", 0.0),
        target_adv_std=ckpt_builder_state.get("target_adv_std", 1.0),
    )

    return player_state, builder_state, league
