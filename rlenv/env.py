import numpy as np

from rlenv.data import EX_STATE, NUM_EDGE_FIELDS, NUM_ENTITY_FIELDS, NUM_MOVE_FIELDS
from rlenv.interfaces import EnvStep
from rlenv.protos.features_pb2 import FeatureEdge, FeatureEntity, FeatureMoveset
from rlenv.protos.state_pb2 import State
from rlenv.utils import padnstack


def get_history(state: State, player_index: int):
    history = state.history
    history_length = history.length
    moveset = np.frombuffer(bytearray(state.moveset), dtype=np.int16).reshape(
        (2, -1, NUM_MOVE_FIELDS)
    )
    moveset[..., FeatureMoveset.MOVESET_SIDE] ^= player_index
    moveset.flags.writeable = False
    team = np.frombuffer(bytearray(state.team), dtype=np.int16).reshape(
        (2, 6, NUM_ENTITY_FIELDS)
    )
    team[..., FeatureEntity.ENTITY_SIDE] ^= player_index
    team.flags.writeable = False
    history_edges = np.frombuffer(bytearray(history.edges), dtype=np.int16).reshape(
        (history_length, NUM_EDGE_FIELDS)
    )
    edge_affecting_side = history_edges[..., FeatureEdge.EDGE_AFFECTING_SIDE]
    history_edges[..., FeatureEdge.EDGE_AFFECTING_SIDE] = np.where(
        edge_affecting_side < 2,
        edge_affecting_side ^ player_index,
        edge_affecting_side,
    )
    history_edges.flags.writeable = False
    history_entities = np.frombuffer(
        bytearray(history.entities), dtype=np.int16
    ).reshape((history_length, 2, NUM_ENTITY_FIELDS))
    history_entities[..., FeatureEntity.ENTITY_SIDE] ^= player_index
    history_entities.flags.writeable = False
    history_side_conditions = np.frombuffer(
        bytearray(history.sideConditions), dtype=np.uint8
    ).reshape((history_length, 2, -1))
    history_field = np.frombuffer(bytearray(history.field), dtype=np.uint8).reshape(
        (history_length, -1)
    )
    return (
        moveset,
        team,
        padnstack(history_edges),
        padnstack(history_entities),
        padnstack(history_side_conditions),
        padnstack(history_field),
    )


def get_legal_mask(state: State):
    buffer = np.frombuffer(state.legalActions, dtype=np.uint8)
    mask = np.unpackbits(buffer, axis=-1)
    return mask[:10].astype(bool)


def process_state(state: State) -> EnvStep:
    player_index = state.info.playerIndex
    (
        moveset,
        team,
        history_edges,
        history_entities,
        history_side_conditions,
        history_field,
    ) = get_history(state, int(player_index))
    rewards = state.info.rewards
    return EnvStep(
        ts=np.array(state.info.ts),
        draw_ratio=np.array(state.info.drawRatio),
        valid=~np.array(state.info.done, dtype=bool),
        player_id=np.array(player_index, dtype=np.int32),
        game_id=np.array(state.info.gameId, dtype=np.int32),
        turn=np.array(state.info.turn, dtype=np.int32),
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
        legal=get_legal_mask(state),
        team=team,
        moveset=moveset,
        history_edges=history_edges,
        history_entities=history_entities,
        history_side_conditions=history_side_conditions,
        history_field=history_field,
        seed_hash=state.info.seed,
    )


def get_ex_step() -> EnvStep:
    return process_state(EX_STATE)
