from typing import Sequence, TypeVar

import jax
import numpy as np

from rl.environment.data import (
    EX_STATE,
    MAX_RATIO_TOKEN,
    NUM_ENTITY_EDGE_FEATURES,
    NUM_ENTITY_NODE_FEATURES,
    NUM_FIELD_FEATURES,
    NUM_HISTORY,
    NUM_MOVE_FEATURES,
)
from rl.environment.interfaces import EnvStep, HistoryStep, TimeStep
from rl.environment.protos.features_pb2 import FieldFeature, InfoFeature
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
        history.field[..., FieldFeature.FIELD_FEATURE__VALID].sum(0),
        axis=0,
    ).item()

    # Round history length up to the nearest multiple of resolution
    rounded_length = int(np.ceil(history_length / resolution) * resolution)

    return jax.tree.map(lambda x: x[:rounded_length], history)


def get_legal_mask(state: EnvironmentState):
    buffer = np.frombuffer(state.legal_actions, dtype=np.uint8)
    mask = np.unpackbits(buffer, axis=-1)
    return mask[:10].astype(bool)


def process_state(state: EnvironmentState) -> TimeStep:
    history_length = state.history_length

    info = np.frombuffer(state.info, dtype=np.int16).astype(np.int32)

    history_entity_nodes = padnstack(
        np.frombuffer(state.history_entity_nodes, dtype=np.int16).reshape(
            (history_length, 12, NUM_ENTITY_NODE_FEATURES)
        ),
        NUM_HISTORY,
    ).astype(np.int32)

    history_entity_edges = padnstack(
        np.frombuffer(state.history_entity_edges, dtype=np.int16).reshape(
            (history_length, 12, NUM_ENTITY_EDGE_FEATURES)
        ),
        NUM_HISTORY,
    ).astype(np.int32)

    history_field = padnstack(
        np.frombuffer(state.history_field, dtype=np.int16).reshape(
            (history_length, NUM_FIELD_FEATURES)
        ),
        NUM_HISTORY,
    ).astype(np.int32)

    my_actions = (
        np.frombuffer(state.my_actions, dtype=np.int16)
        .reshape(10, NUM_MOVE_FEATURES)
        .astype(np.int32)
    )
    private_team = (
        np.frombuffer(state.private_team, dtype=np.int16)
        .reshape(6, NUM_ENTITY_NODE_FEATURES)
        .astype(np.int32)
    )
    public_team = (
        np.frombuffer(state.public_team, dtype=np.int16)
        .reshape(12, NUM_ENTITY_NODE_FEATURES)
        .astype(np.int32)
    )

    field = (
        np.frombuffer(state.field, dtype=np.int16)
        .reshape(NUM_FIELD_FEATURES)
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
        field=field,
        moveset=my_actions,
        action_mask=get_legal_mask(state),
    )
    history_step = HistoryStep(
        nodes=history_entity_nodes,
        edges=history_entity_edges,
        field=history_field,
    )

    return TimeStep(env=env_step, history=history_step)


def get_ex_step(expand: bool = True) -> TimeStep:
    ts = process_state(EX_STATE)
    if expand:
        ex = jax.tree.map(lambda x: x[None, None, ...], ts.env)
        hx = jax.tree.map(lambda x: x[:, None, ...], ts.history)
        return TimeStep(env=ex, history=hx)
    else:
        return ts
