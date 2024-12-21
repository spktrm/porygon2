import numpy as np
import jax.numpy as jnp

from rlenv.data import EX_STATE, NUM_EDGE_FIELDS, NUM_ENTITY_FIELDS
from rlenv.interfaces import EnvStep
from rlenv.protos.state_pb2 import State
from rlenv.utils import padnstack


def get_history(state: State):
    history = state.history
    history_length = history.length
    moveset = np.frombuffer(bytearray(state.moveset), dtype=np.int16)
    team = np.frombuffer(bytearray(state.team), dtype=np.int16)
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
    return (
        jnp.array(moveset),
        jnp.array(team),
        jnp.array(padnstack(history_edges)),
        jnp.array(padnstack(history_entities)),
        jnp.array(padnstack(history_side_conditions)),
        jnp.array(padnstack(history_field)),
    )


def get_legal_mask(state: State):
    buffer = jnp.frombuffer(state.legalActions, dtype=jnp.uint8)
    mask = jnp.unpackbits(buffer, axis=-1)
    return mask[:10].astype(bool)


def process_state(state: State) -> EnvStep:
    player_index = int(state.info.playerIndex)
    (
        moveset,
        team,
        history_edges,
        history_entities,
        history_side_conditions,
        history_field,
    ) = get_history(state)
    rewards = state.info.rewards
    heuristics = state.info.heuristics
    return EnvStep(
        ts=jnp.array(state.info.ts),
        draw_ratio=jnp.array(state.info.drawRatio),
        valid=~jnp.array(state.info.done, dtype=bool),
        draw=jnp.array(state.info.draw, dtype=bool),
        player_id=jnp.array(player_index, dtype=jnp.int32),
        game_id=jnp.array(state.info.gameId, dtype=jnp.int32),
        turn=jnp.array(state.info.turn, dtype=jnp.int32),
        win_rewards=jnp.array(
            [rewards.winReward, -rewards.winReward], dtype=jnp.float32
        ),
        hp_rewards=jnp.array([rewards.hpReward, -rewards.hpReward], dtype=jnp.float32),
        fainted_rewards=jnp.array(
            [rewards.faintedReward, -rewards.faintedReward], dtype=jnp.float32
        ),
        switch_rewards=jnp.array(
            [rewards.switchReward, -rewards.switchReward], dtype=jnp.float32
        ),
        longevity_rewards=jnp.array(
            [rewards.longevityReward, -rewards.longevityReward], dtype=jnp.float32
        ),
        legal=get_legal_mask(state),
        team=team.astype(jnp.int32),
        moveset=moveset.astype(jnp.int32),
        history_edges=history_edges.astype(jnp.int32),
        history_entities=history_entities.astype(jnp.int32),
        history_side_conditions=history_side_conditions.astype(jnp.int32),
        history_field=history_field.astype(jnp.int32),
        seed_hash=jnp.array(state.info.seed).astype(jnp.int32),
        heuristic_action=jnp.array(heuristics.heuristicAction).astype(jnp.int32),
    )


def get_ex_step() -> EnvStep:
    return process_state(EX_STATE)
