from typing import Sequence, TypeVar

import jax
import numpy as np
from rl.environment.data import (
    EX_STATE,
    MAX_RATIO_TOKEN,
    NUM_ABSOLUTE_EDGE_FIELDS,
    NUM_CONTEXT_FIELDS,
    NUM_ENTITY_FIELDS,
    NUM_HISTORY,
    NUM_MOVE_FIELDS,
    NUM_RELATIVE_EDGE_FIELDS,
)

from rl.environment.interfaces import EnvStep, HistoryStep, TimeStep
from rl.environment.protos.features_pb2 import AbsoluteEdgeFeature, InfoFeature
from rl.environment.protos.service_pb2 import EnvironmentState

T = TypeVar("T")


def stack_steps(steps: Sequence[T], axis: int = 0) -> T:
    return jax.tree.map(lambda *xs: np.stack(xs, axis=axis), *steps)


def concatenate_steps(steps: Sequence[T], axis: int = 0) -> T:
    return jax.tree.map(lambda *xs: np.concatenate(xs, axis=axis), *steps)


def padnstack(arr: np.ndarray, padding: int = NUM_HISTORY) -> np.ndarray:
    output_shape = (padding, *arr.shape[1:])
    result = np.zeros(output_shape, dtype=arr.dtype)
    length_to_copy = min(padding, arr.shape[0])
    result[:length_to_copy] = arr[:length_to_copy]
    return result


def expand_dims(x, axis: int):
    return jax.tree.map(lambda i: np.expand_dims(i, axis=axis), x)


def clip_history(history: HistoryStep, resolution: int = 64) -> HistoryStep:
    history_length = np.max(
        history.absolute_edges[
            ..., AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__VALID
        ].sum(0),
        axis=0,
    ).item()

    # Round history length up to the nearest multiple of resolution
    rounded_length = int(np.ceil(history_length / resolution) * resolution)

    return jax.tree.map(lambda x: x[:rounded_length], history)


def get_legal_mask(state: EnvironmentState):
    buffer = np.frombuffer(state.legal_actions, dtype=np.uint8)
    mask = np.unpackbits(buffer, axis=-1)
    return mask[:10].astype(bool)


def process_state(state: EnvironmentState) -> tuple[EnvStep, HistoryStep]:
    history_length = state.history_length

    info = np.frombuffer(state.info, dtype=np.int16)

    history_entities = padnstack(
        np.frombuffer(state.history_entities, dtype=np.int16).reshape(
            (history_length, 2, NUM_ENTITY_FIELDS)
        ),
        NUM_HISTORY,
    ).astype(np.int32)

    history_relative_edges = padnstack(
        np.frombuffer(state.history_relative_edges, dtype=np.int16).reshape(
            (history_length, 2, NUM_RELATIVE_EDGE_FIELDS)
        ),
        NUM_HISTORY,
    ).astype(np.int32)

    history_absolute_edge = padnstack(
        np.frombuffer(state.history_absolute_edge, dtype=np.int16).reshape(
            (history_length, NUM_ABSOLUTE_EDGE_FIELDS)
        ),
        NUM_HISTORY,
    ).astype(np.int32)

    moveset = (
        np.frombuffer(state.moveset, dtype=np.int16)
        .reshape(10, NUM_MOVE_FIELDS)
        .astype(np.int32)
    )
    private_team = (
        np.frombuffer(state.private_team, dtype=np.int16)
        .reshape(6, NUM_ENTITY_FIELDS)
        .astype(np.int32)
    )
    public_team = (
        np.frombuffer(state.public_team, dtype=np.int16)
        .reshape(12, NUM_ENTITY_FIELDS)
        .astype(np.int32)
    )

    current_context = (
        np.frombuffer(state.current_context, dtype=np.int16)
        .reshape(NUM_CONTEXT_FIELDS)
        .astype(np.int32)
    )

    win_reward_token = info[InfoFeature.INFO_FEATURE__WIN_REWARD]
    # Divide by MAX_RATIO_TOKEN to normalize the win reward to [-1, 1] since we store as int16
    win_reward = win_reward_token / MAX_RATIO_TOKEN

    env_step = EnvStep(
        info=info,
        done=info[InfoFeature.INFO_FEATURE__DONE].astype(np.bool_),
        win_reward=win_reward.astype(np.float32),
        private_team=private_team,
        public_team=public_team,
        current_context=current_context,
        moveset=moveset.astype(np.int32),
        legal=get_legal_mask(state),
    )
    history_step = HistoryStep(
        entities=history_entities,
        relative_edges=history_relative_edges,
        absolute_edges=history_absolute_edge,
    )

    return env_step, history_step


def get_ex_step():
    ex, hx = process_state(EX_STATE)
    ex = jax.tree.map(lambda x: x[None, None, ...], ex)
    hx = jax.tree.map(lambda x: x[:, None, ...], hx)
    return TimeStep(env=ex, history=hx)
