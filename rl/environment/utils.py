from typing import Sequence, TypeVar

import jax
import numpy as np

from rl.environment.data import (
    EX_TRAJECTORY,
    MAX_RATIO_TOKEN,
    NUM_ACTION_MASK_FEATURES,
    NUM_ENTITY_EDGE_FEATURES,
    NUM_ENTITY_NODE_FEATURES,
    NUM_FIELD_FEATURES,
    NUM_HISTORY,
    NUM_MOVE_FEATURES,
    PACKED_SETS,
)
from rl.environment.interfaces import (
    BuilderEnvOutput,
    PlayerActorInput,
    PlayerEnvOutput,
    PlayerHistoryOutput,
)
from rl.environment.protos.features_pb2 import (
    ActionMaskFeature,
    FieldFeature,
    InfoFeature,
)
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


def clip_history(
    history: PlayerHistoryOutput, resolution: int = 64
) -> PlayerHistoryOutput:
    history_length = np.max(
        history.field[..., FieldFeature.FIELD_FEATURE__VALID].sum(0),
        axis=0,
    ).item()

    # Round history length up to the nearest multiple of resolution
    rounded_length = int(np.ceil(history_length / resolution) * resolution)

    return jax.tree.map(lambda x: x[:rounded_length], history)


def get_action_mask(state: EnvironmentState):
    buffer = np.frombuffer(state.action_mask, dtype=np.uint8)
    mask = np.unpackbits(buffer, axis=-1)
    return mask[:NUM_ACTION_MASK_FEATURES].astype(bool)


def get_action_type_mask(mask: jax.Array):
    mask = mask[
        ...,
        ActionMaskFeature.ACTION_MASK_FEATURE__CAN_MOVE : ActionMaskFeature.ACTION_MASK_FEATURE__CAN_TEAMPREVIEW
        + 1,
    ]
    return mask | (~mask).all(axis=-1, keepdims=True)


def get_move_mask(mask: jax.Array):
    mask = mask[
        ...,
        ActionMaskFeature.ACTION_MASK_FEATURE__MOVE_SLOT_1 : ActionMaskFeature.ACTION_MASK_FEATURE__MOVE_SLOT_4
        + 1,
    ]
    return mask | (~mask).all(axis=-1, keepdims=True)


def get_switch_mask(mask: jax.Array):
    mask = mask[
        ...,
        ActionMaskFeature.ACTION_MASK_FEATURE__SWITCH_SLOT_1 : ActionMaskFeature.ACTION_MASK_FEATURE__SWITCH_SLOT_6
        + 1,
    ]
    return mask | (~mask).all(axis=-1, keepdims=True)


def get_tera_mask(mask: jax.Array):
    mask = mask[
        ...,
        ActionMaskFeature.ACTION_MASK_FEATURE__CAN_NORMAL : ActionMaskFeature.ACTION_MASK_FEATURE__CAN_TERA
        + 1,
    ]
    return mask | (~mask).all(axis=-1, keepdims=True)


def process_state(state: EnvironmentState) -> PlayerActorInput:
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

    moveset = (
        np.frombuffer(state.moveset, dtype=np.int16)
        .reshape(4, NUM_MOVE_FEATURES)
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

    action_mask = get_action_mask(state)

    env_step = PlayerEnvOutput(
        info=info,
        done=info[InfoFeature.INFO_FEATURE__DONE].astype(np.bool_),
        win_reward=win_reward.astype(np.float32),
        private_team=private_team,
        public_team=public_team,
        field=field,
        moveset=moveset,
        action_type_mask=get_action_type_mask(action_mask),
        move_mask=get_move_mask(action_mask),
        switch_mask=get_switch_mask(action_mask),
        wildcard_mask=get_tera_mask(action_mask),
    )
    history_step = PlayerHistoryOutput(
        nodes=history_entity_nodes,
        edges=history_entity_edges,
        field=history_field,
    )

    return PlayerActorInput(env=env_step, history=history_step)


def get_ex_trajectory() -> PlayerActorInput:
    states = []
    for state in EX_TRAJECTORY.states:
        processed_state = process_state(state)
        states.append(processed_state.env)
    return PlayerActorInput(
        env=jax.tree.map(lambda *xs: np.stack(xs), *states),
        history=processed_state.history,
    )


def get_ex_player_step() -> PlayerActorInput:
    ts = get_ex_trajectory()
    ex = jax.tree.map(lambda x: x[:, None, ...], ts.env)
    hx = jax.tree.map(lambda x: x[:, None, ...], ts.history)
    return PlayerActorInput(env=ex, history=hx)


def get_ex_builder_step(generation: int = 3) -> BuilderEnvOutput:
    data = PACKED_SETS[f"gen{generation}ou"]
    num_sets = len(data["sets"])
    return BuilderEnvOutput(
        tokens=np.ones((1, 1, 6), dtype=np.int32) * -1,
        mask=np.ones((1, 1, num_sets), dtype=bool),
    )
