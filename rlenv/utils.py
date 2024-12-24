import functools
import math
from typing import Sequence, TypeVar

import jax
import numpy as np
import jax.numpy as jnp

from rlenv.data import NUM_HISTORY
from rlenv.interfaces import ActorStep, EnvStep, HistoryContainer, HistoryStep, TimeStep
from rlenv.protos.features_pb2 import FeatureEdge

T = TypeVar("T")


def add_batch(step: T, axis: int = 0) -> T:
    return jax.tree.map(lambda xs: np.expand_dims(xs, axis), step)


# @jax.jit
def stack_steps(steps: Sequence[T], axis: int = 0) -> T:
    return jax.tree.map(lambda *xs: np.stack(xs, axis=axis), *steps)


def trim_container(
    container: HistoryContainer, resolution: int = 32
) -> HistoryContainer:
    traj_length = jnp.max(container.edges[..., FeatureEdge.EDGE_VALID].sum(axis=0))
    traj_length = resolution * jnp.ceil(traj_length / resolution).astype(int)

    def dynamic_slice(x, length):
        slice_length = jnp.minimum(x.shape[0], length)  # Ensure bounds are safe
        return jax.lax.dynamic_slice(x, (0,) * x.ndim, (slice_length,) + x.shape[1:])

    return jax.tree_map(lambda x: dynamic_slice(x, traj_length), container)


def trim_history(step: HistoryStep) -> HistoryStep:
    return HistoryStep(
        major_history=trim_container(step.major_history),
        minor_history=trim_container(step.minor_history),
    )


def concatenate_steps(steps: Sequence[T], axis: int = 0) -> T:
    return jax.tree.map(lambda *xs: np.concatenate(xs, axis=axis), *steps)


def stack_trajectories(trajectories: Sequence[TimeStep], resolution: int = 32) -> T:
    valid_lengths = np.array([tx.env.valid.sum(0) for tx in trajectories]).reshape(-1)
    traj_length = np.max(valid_lengths)
    traj_length = resolution * math.ceil(traj_length / resolution)
    trajectories = jax.tree.map(
        lambda t: np.resize(t, (traj_length, *t.shape[1:])), trajectories
    )
    batch = concatenate_steps(trajectories, axis=1)
    return TimeStep(
        env=EnvStep(
            ts=batch.env.ts.squeeze(),
            draw_ratio=batch.env.draw_ratio.squeeze(),
            valid=(np.arange(traj_length)[:, None] < valid_lengths),
            turn=batch.env.turn.squeeze(),
            game_id=batch.env.game_id.squeeze(),
            player_id=batch.env.player_id.squeeze(),
            moveset=batch.env.moveset.squeeze(),
            legal=batch.env.legal.squeeze(),
            team=batch.env.team.squeeze(),
            win_rewards=np.take_along_axis(
                batch.env.win_rewards, batch.env.player_id[..., None], axis=-1
            ).squeeze(),
            fainted_rewards=np.take_along_axis(
                batch.env.fainted_rewards, batch.env.player_id[..., None], axis=-1
            ).squeeze(),
            switch_rewards=np.take_along_axis(
                batch.env.switch_rewards, batch.env.player_id[..., None], axis=-1
            ).squeeze(),
            longevity_rewards=np.take_along_axis(
                batch.env.longevity_rewards, batch.env.player_id[..., None], axis=-1
            ).squeeze(),
            hp_rewards=np.take_along_axis(
                batch.env.hp_rewards, batch.env.player_id[..., None], axis=-1
            ).squeeze(),
            history_edges=batch.env.history_edges.squeeze(),
            history_entities=batch.env.history_entities.squeeze(),
            history_side_conditions=batch.env.history_side_conditions.squeeze(),
            history_field=batch.env.history_field.squeeze(),
        ),
        actor=ActorStep(
            action=batch.actor.action.squeeze(),
            policy=batch.actor.policy.squeeze(),
            win_rewards=np.take_along_axis(
                batch.actor.win_rewards, batch.env.player_id[..., None], axis=-1
            ).squeeze(),
            fainted_rewards=np.take_along_axis(
                batch.actor.fainted_rewards, batch.env.player_id[..., None], axis=-1
            ).squeeze(),
            switch_rewards=np.take_along_axis(
                batch.actor.switch_rewards, batch.env.player_id[..., None], axis=-1
            ).squeeze(),
            longevity_rewards=np.take_along_axis(
                batch.actor.longevity_rewards, batch.env.player_id[..., None], axis=-1
            ).squeeze(),
            hp_rewards=np.take_along_axis(
                batch.actor.hp_rewards, batch.env.player_id[..., None], axis=-1
            ).squeeze(),
        ),
    )


# @jax.jit
def padnstack(arr: np.ndarray, padding: int = NUM_HISTORY) -> np.ndarray:
    stacked = np.resize(arr, (padding, *arr.shape[1:]))
    mask = np.arange(padding) < arr.shape[0]
    return np.where(mask[..., *((None,) * (len(arr.shape) - 1))], stacked, 0)
