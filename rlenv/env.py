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

    edges = np.frombuffer(bytearray(history.edges), dtype=np.int16).reshape(
        (history_length, NUM_EDGE_FIELDS)
    )
    entities = np.frombuffer(bytearray(history.entities), dtype=np.int16).reshape(
        (history_length, 2, NUM_ENTITY_FIELDS)
    )
    side_conditions = np.frombuffer(
        bytearray(history.sideConditions), dtype=np.uint8
    ).reshape((history_length, 2, -1))
    field = np.frombuffer(bytearray(history.field), dtype=np.uint8).reshape(
        (history_length, -1)
    )
    return HistoryContainer(
        edges=padnstack(edges, padding_length).astype(np.int32),
        entities=padnstack(entities, padding_length).astype(np.int32),
        side_conditions=padnstack(side_conditions, padding_length).astype(np.int32),
        field=padnstack(field, padding_length).astype(np.int32),
    )


def get_legal_mask(state: State):
    buffer = np.frombuffer(state.legalActions, dtype=np.uint8)
    mask = np.unpackbits(buffer, axis=-1)
    return mask[:10].astype(bool)


def process_state(state: State):
    player_index = int(state.info.playerIndex)

    major_history_step = get_history(state.majorHistory, 64)
    minor_history_step = get_history(state.minorHistory, 64)

    moveset = np.frombuffer(bytearray(state.moveset), dtype=np.int16).reshape(
        2, -1, NUM_MOVE_FIELDS
    )
    team = np.frombuffer(bytearray(state.team), dtype=np.int16).reshape(
        2, 6, NUM_ENTITY_FIELDS
    )

    rewards = state.info.rewards
    heuristics = state.info.heuristics

    reward_step = RewardStep(
        win_rewards=np.array([rewards.winReward, -rewards.winReward], dtype=np.float32),
        hp_rewards=np.array([rewards.hpReward, -rewards.hpReward], dtype=np.float32),
        fainted_rewards=np.array(
            [rewards.faintedReward, -rewards.faintedReward], dtype=np.float32
        ),
        switch_rewards=np.array(
            [rewards.switchReward, -rewards.switchReward], dtype=np.float32
        ),
        longevity_rewards=np.array(
            [rewards.longevityReward, -rewards.longevityReward], dtype=np.float32
        ),
    )

    env_step = EnvStep(
        ts=np.array(state.info.ts),
        draw_ratio=np.array(state.info.drawRatio),
        valid=~np.array(state.info.done, dtype=bool),
        draw=np.array(state.info.draw, dtype=bool),
        player_id=np.array(player_index, dtype=np.int32),
        game_id=np.array(state.info.gameId, dtype=np.int32),
        turn=np.array(state.info.turn, dtype=np.int32),
        legal=get_legal_mask(state),
        rewards=reward_step,
        team=team.astype(np.int32),
        moveset=moveset.astype(np.int32),
        seed_hash=np.array(state.info.seed).astype(np.int32),
        request_count=np.array(state.info.requestCount).astype(np.int32),
        heuristic_action=np.array(heuristics.heuristicAction).astype(np.int32),
    )
    history_step = HistoryStep(
        major_history=major_history_step,
        minor_history=minor_history_step,
    )

    return env_step, history_step


def get_ex_step():
    return process_state(EX_STATE)
