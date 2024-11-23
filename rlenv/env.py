import numpy as np

from rlenv.data import EX_STATE, NUM_EDGE_FIELDS, NUM_MOVE_FIELDS
from rlenv.interfaces import EnvStep
from rlenv.protos.features_pb2 import FeatureEdge, FeatureEntity
from rlenv.protos.state_pb2 import State
from rlenv.utils import padnstack


def get_history(state: State, player_index: int):
    history = state.history
    history_length = history.length
    moveset = np.frombuffer(state.moveset, dtype=np.int16).reshape(
        (2, -1, NUM_MOVE_FIELDS)
    )
    team = np.frombuffer(bytearray(state.team), dtype=np.int16).reshape((2, 6, -1))
    team[..., FeatureEntity.ENTITY_SIDE] ^= player_index
    team.flags.writeable = False
    history_edges = np.frombuffer(bytearray(history.edges), dtype=np.int16).reshape(
        (history_length, -1, NUM_EDGE_FIELDS)
    )
    edge_affecting_side = history_edges[..., FeatureEdge.EDGE_AFFECTING_SIDE]
    history_edges[..., FeatureEdge.EDGE_AFFECTING_SIDE] = np.where(
        edge_affecting_side < 2,
        edge_affecting_side ^ player_index,
        edge_affecting_side,
    )
    history_edges.flags.writeable = False
    history_nodes = np.frombuffer(bytearray(history.nodes), dtype=np.int16).reshape(
        (history_length, 12, -1)
    )
    history_nodes[..., FeatureEntity.ENTITY_SIDE] ^= player_index
    history_nodes.flags.writeable = False
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
        padnstack(history_nodes),
        padnstack(history_side_conditions),
        padnstack(history_field),
    )


def get_legal_mask(state: State):
    buffer = np.frombuffer(state.legalActions, dtype=np.uint8)
    mask = np.unpackbits(buffer, axis=-1)
    return mask[:10].astype(bool)


def process_state(state: State, is_eval: bool = False, done: bool = False) -> EnvStep:
    player_index = state.info.playerIndex
    (
        moveset,
        team,
        history_edges,
        history_nodes,
        history_side_conditions,
        history_field,
    ) = get_history(state, player_index)
    return EnvStep(
        ts=state.info.ts,
        draw_ratio=(
            1 - (1 - state.info.turn / 100) ** 2 if is_eval else state.info.drawRatio
        ),
        valid=~np.array(done, dtype=bool),
        player_id=np.array(player_index, dtype=np.int32),
        game_id=np.array(state.info.gameId, dtype=np.int32),
        turn=np.array(state.info.turn, dtype=np.int32),
        heuristic_action=np.array(state.info.heuristicAction, dtype=np.int32),
        heuristic_dist=np.frombuffer(state.info.heuristicDist, dtype=np.float32),
        prev_action=np.array(state.info.lastAction, dtype=np.int32),
        prev_move=np.array(state.info.lastMove, dtype=np.int32),
        win_rewards=(
            np.array([state.info.winReward, -state.info.winReward], dtype=np.float32)
            if done
            else np.zeros(2)
        ),
        hp_rewards=np.array(
            [state.info.hpReward, -state.info.hpReward], dtype=np.float32
        ),
        fainted_rewards=np.array(
            [state.info.faintedReward, -state.info.faintedReward], dtype=np.float32
        ),
        switch_rewards=np.array(
            [state.info.switchReward, -state.info.switchReward], dtype=np.float32
        ),
        longevity_rewards=np.array(
            [state.info.longevityReward, -state.info.longevityReward], dtype=np.float32
        ),
        legal=get_legal_mask(state),
        team=team,
        moveset=moveset,
        history_edges=history_edges,
        history_nodes=history_nodes,
        history_side_conditions=history_side_conditions,
        history_field=history_field,
    )


def get_ex_step() -> EnvStep:
    return process_state(EX_STATE)
