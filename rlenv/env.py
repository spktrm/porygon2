import jax
import jax.numpy as jnp
import numpy as np

from rlenv.data import (
    EX_STATE,
    NUM_EDGE_FIELDS,
    NUM_ENTITY_FIELDS,
    NUM_HISTORY,
    NUM_MOVE_FIELDS,
)
from rlenv.interfaces import EnvStep, HistoryContainer, HistoryStep, RewardStep
from rlenv.protos.history_pb2 import History
from rlenv.protos.state_pb2 import State
from rlenv.utils import padnstack


def get_history(history: History, padding_length: int = NUM_HISTORY):
    history_length = history.length

    edges = np.frombuffer(history.edges, dtype=np.int16).reshape(
        (history_length, NUM_EDGE_FIELDS)
    )
    entities = np.frombuffer(history.entities, dtype=np.int16).reshape(
        (history_length, 2, NUM_ENTITY_FIELDS)
    )
    side_conditions = np.frombuffer(history.sideConditions, dtype=np.uint8).reshape(
        (history_length, 2, -1)
    )
    field = np.frombuffer(history.field, dtype=np.uint8).reshape((history_length, -1))
    return HistoryContainer(
        edges=padnstack(edges, padding_length).astype(int),
        entities=padnstack(entities, padding_length).astype(int),
        side_conditions=padnstack(side_conditions, padding_length).astype(int),
        field=padnstack(field, padding_length).astype(int),
    )


def get_legal_mask(state: State):
    buffer = np.frombuffer(state.legalActions, dtype=np.uint8)
    mask = np.unpackbits(buffer, axis=-1)
    return mask[:10].astype(bool)


def process_state(state: State):
    player_index = int(state.info.playerIndex)

    major_history_step = get_history(state.majorHistory, NUM_HISTORY)
    minor_history_step = get_history(state.minorHistory, NUM_HISTORY)

    moveset = (
        np.frombuffer(state.moveset, dtype=np.int16)
        .reshape(2, -1, NUM_MOVE_FIELDS)
        .astype(int)
    )
    team = (
        np.frombuffer(state.team, dtype=np.int16)
        .reshape(2, 6, NUM_ENTITY_FIELDS)
        .astype(int)
    )

    rewards = state.info.rewards
    heuristics = state.info.heuristics

    reward_step = RewardStep(
        win_rewards=np.array([rewards.winReward, -rewards.winReward], dtype=float),
        hp_rewards=np.array([rewards.hpReward, -rewards.hpReward], dtype=float),
        fainted_rewards=np.array(
            [rewards.faintedReward, -rewards.faintedReward], dtype=float
        ),
        switch_rewards=np.array(
            [rewards.switchReward, -rewards.switchReward], dtype=float
        ),
        longevity_rewards=np.array(
            [rewards.longevityReward, -rewards.longevityReward], dtype=float
        ),
    )

    env_step = EnvStep(
        ts=np.array(state.info.ts),
        draw_ratio=np.array(state.info.drawRatio, dtype=float),
        valid=~np.array(state.info.done, dtype=bool),
        draw=np.array(state.info.draw, dtype=bool),
        player_id=np.array(player_index, dtype=int),
        game_id=np.array(state.info.gameId, dtype=int),
        turn=np.array(state.info.turn, dtype=int),
        legal=get_legal_mask(state),
        rewards=reward_step,
        team=team.astype(int),
        moveset=moveset.astype(int),
        seed_hash=np.array(state.info.seed).astype(int),
        request_count=np.array(state.info.requestCount).astype(int),
        heuristic_action=np.array(heuristics.heuristicAction).astype(int),
    )
    history_step = HistoryStep(
        major_history=major_history_step,
        minor_history=minor_history_step,
    )

    return env_step, history_step


def as_jax_arr(x):
    return jax.tree.map(lambda i: jnp.asarray(i), x)


def get_ex_step():
    ex, hx = process_state(EX_STATE)
    ex = as_jax_arr(ex)
    hx = as_jax_arr(hx)
    return ex, hx
