import jax
import jax.numpy as jnp
import numpy as np

from rlenv.data import (
    EX_STATE,
    NUM_ABSOLUTE_EDGE_FIELDS,
    NUM_ENTITY_FIELDS,
    NUM_HISTORY,
    NUM_MOVE_FIELDS,
    NUM_RELATIVE_EDGE_FIELDS,
)
from rlenv.interfaces import EnvStep, HistoryContainer, HistoryStep, RewardStep
from rlenv.protos.features_pb2 import AbsoluteEdgeFeature
from rlenv.protos.history_pb2 import History
from rlenv.protos.state_pb2 import State
from rlenv.utils import padnstack


def get_history(history: History, padding_length: int = NUM_HISTORY):
    history_length = history.length

    entities = np.frombuffer(history.entities, dtype=np.int16).reshape(
        (history_length, 2, NUM_ENTITY_FIELDS)
    )
    relative_edges = np.frombuffer(history.relative_edges, dtype=np.int16).reshape(
        (history_length, 2, NUM_RELATIVE_EDGE_FIELDS)
    )
    absolute_edges = np.frombuffer(history.absolute_edge, dtype=np.int16).reshape(
        (history_length, NUM_ABSOLUTE_EDGE_FIELDS)
    )

    return HistoryContainer(
        entities=padnstack(entities, padding_length).astype(int),
        relative_edges=padnstack(relative_edges, padding_length).astype(int),
        absolute_edges=padnstack(absolute_edges, padding_length).astype(int),
    )


def expand_dims(x, axis: int):
    return jax.tree_map(lambda i: np.expand_dims(i, axis=axis), x)


def clip_history(history: HistoryStep, resolution: int = 64) -> HistoryStep:
    history_length = np.max(
        history.major_history.absolute_edges[
            ..., AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__VALID
        ].sum(0),
        axis=0,
    ).item()

    # Round history length up to the nearest multiple of resolution
    rounded_length = int(np.ceil(history_length / resolution) * resolution) + 1

    return jax.tree_map(lambda x: x[:rounded_length], history)


def get_legal_mask(state: State):
    buffer = np.frombuffer(state.legal_actions, dtype=np.uint8)
    mask = np.unpackbits(buffer, axis=-1)
    return mask[:10].astype(bool)


def process_state(state: State):
    player_index = int(state.info.player_index)

    history_step = get_history(state.history, NUM_HISTORY)

    moveset = (
        np.frombuffer(state.moveset, dtype=np.int16)
        .reshape(10, NUM_MOVE_FIELDS)
        .astype(int)
    )
    private_team = (
        np.frombuffer(state.private_team, dtype=np.int16)
        .reshape(6, NUM_ENTITY_FIELDS)
        .astype(int)
    )
    public_team = (
        np.frombuffer(state.public_team, dtype=np.int16)
        .reshape(12, NUM_ENTITY_FIELDS)
        .astype(int)
    )

    rewards = state.info.rewards
    heuristics = state.info.heuristics

    reward_step = RewardStep(
        win_rewards=np.array([rewards.win_reward, -rewards.win_reward], dtype=float),
        hp_rewards=np.array([rewards.hp_reward, -rewards.hp_reward], dtype=float),
        fainted_rewards=np.array(
            [rewards.fainted_reward, -rewards.fainted_reward], dtype=float
        ),
        scaled_hp_rewards=np.array(
            [rewards.scaled_hp_reward, -rewards.scaled_hp_reward], dtype=float
        ),
        scaled_fainted_rewards=np.array(
            [rewards.scaled_fainted_reward, -rewards.scaled_fainted_reward], dtype=float
        ),
    )

    env_step = EnvStep(
        ts=np.array(state.info.ts),
        draw_ratio=np.array(state.info.draw_ratio, dtype=float),
        valid=~np.array(state.info.done, dtype=bool),
        draw=np.array(state.info.draw, dtype=bool),
        player_id=np.array(player_index, dtype=int),
        game_id=np.array(state.info.game_id, dtype=int),
        turn=np.array(state.info.turn, dtype=int),
        timestamp=np.array(state.info.timestamp, dtype=int),
        legal=get_legal_mask(state),
        rewards=reward_step,
        private_team=private_team.astype(int),
        public_team=public_team.astype(int),
        moveset=moveset.astype(int),
        seed_hash=np.array(state.info.seed).astype(int),
        request_count=np.array(state.info.request_count).astype(int),
        heuristic_action=np.array(heuristics.heuristic_action).astype(int),
    )
    history_step = HistoryStep(
        major_history=history_step,
    )

    return expand_dims(env_step, axis=0), history_step


def as_jax_arr(x):
    return jax.tree.map(lambda i: jnp.asarray(i), x)


def get_ex_step():
    ex, hx = process_state(EX_STATE)
    ex = as_jax_arr(ex)
    hx = as_jax_arr(hx)
    return ex, hx
