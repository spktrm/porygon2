import jax
import jax.numpy as jnp
import numpy as np

from rlenv.data import (
    EX_STATE,
    NUM_ABSOLUTE_EDGE_FIELDS,
    NUM_CONTEXT_FIELDS,
    NUM_ENTITY_FIELDS,
    NUM_HISTORY,
    NUM_MOVE_FIELDS,
    NUM_RELATIVE_EDGE_FIELDS,
)
from rlenv.interfaces import EnvStep, HistoryStep, TimeStep
from rlenv.protos.features_pb2 import AbsoluteEdgeFeature, InfoFeature
from rlenv.protos.service_pb2 import EnvironmentState
from rlenv.utils import padnstack


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
    ).astype(int)

    history_relative_edges = padnstack(
        np.frombuffer(state.history_relative_edges, dtype=np.int16).reshape(
            (history_length, 2, NUM_RELATIVE_EDGE_FIELDS)
        ),
        NUM_HISTORY,
    ).astype(int)

    history_absolute_edge = padnstack(
        np.frombuffer(state.history_absolute_edge, dtype=np.int16).reshape(
            (history_length, NUM_ABSOLUTE_EDGE_FIELDS)
        ),
        NUM_HISTORY,
    ).astype(int)

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

    current_context = (
        np.frombuffer(state.current_context, dtype=np.int16)
        .reshape(NUM_CONTEXT_FIELDS)
        .astype(int)
    )

    env_step = EnvStep(
        info=info,
        done=info[InfoFeature.INFO_FEATURE__DONE].astype(np.bool_),
        win_reward=info[InfoFeature.INFO_FEATURE__WIN_REWARD].astype(np.float32),
        private_team=private_team,
        public_team=public_team,
        current_context=current_context,
        moveset=moveset.astype(int),
        legal=get_legal_mask(state),
    )
    history_step = HistoryStep(
        entities=history_entities,
        relative_edges=history_relative_edges,
        absolute_edges=history_absolute_edge,
    )

    return env_step, history_step


def as_jax_arr(x):
    return jax.tree.map(lambda i: jnp.asarray(i), x)


def get_ex_step():
    ex, hx = process_state(EX_STATE)
    ex = jax.tree.map(lambda x: x[None, None, ...], ex)
    hx = jax.tree.map(lambda x: x[:, None, ...], hx)
    return TimeStep(env=ex, history=hx)
