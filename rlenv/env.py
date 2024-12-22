import numpy as np

from rlenv.data import EX_STATE, NUM_EDGE_FIELDS, NUM_ENTITY_FIELDS, NUM_MOVE_FIELDS
from rlenv.interfaces import EnvStep, HistoryStep, RewardStep
from rlenv.protos.state_pb2 import State
from rlenv.utils import padnstack


def get_history(state: State):
    history = state.history
    history_length = history.length

    history_edges = np.frombuffer(bytearray(history.edges), dtype=np.int16).reshape(
        (history_length, NUM_EDGE_FIELDS)
    )
    history_entities = np.frombuffer(
        bytearray(history.entities), dtype=np.int16
    ).reshape((history_length, 2, NUM_ENTITY_FIELDS))
    history_side_conditions = np.frombuffer(
        bytearray(history.sideConditions), dtype=np.uint8
    ).reshape((history_length, 2, -1))
    history_field = np.frombuffer(bytearray(history.field), dtype=np.uint8).reshape(
        (history_length, -1)
    )
    return HistoryStep(
        history_edges=padnstack(history_edges).astype(np.int32),
        history_entities=padnstack(history_entities).astype(np.int32),
        history_side_conditions=padnstack(history_side_conditions).astype(np.int32),
        history_field=padnstack(history_field).astype(np.int32),
    )


def get_legal_mask(state: State):
    buffer = np.frombuffer(state.legalActions, dtype=np.uint8)
    mask = np.unpackbits(buffer, axis=-1)
    return mask[:10].astype(bool)


def process_state(state: State):
    player_index = int(state.info.playerIndex)

    history_step = get_history(state)

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
        heuristic_action=np.array(heuristics.heuristicAction).astype(np.int32),
    )

    return env_step, history_step


def get_ex_step() -> EnvStep:
    return process_state(EX_STATE)
