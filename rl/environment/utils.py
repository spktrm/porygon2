from typing import Sequence, TypeVar

import jax
import numpy as np
import rlax

from rl.environment.data import (
    EX_TRAJECTORY,
    MAX_RATIO_TOKEN,
    NUM_ACTION_FEATURES,
    NUM_ENTITY_EDGE_FEATURES,
    NUM_ENTITY_PRIVATE_FEATURES,
    NUM_ENTITY_PUBLIC_FEATURES,
    NUM_ENTITY_REVEALED_FEATURES,
    NUM_FIELD_FEATURES,
    NUM_HISTORY,
    NUM_MOVE_FEATURES,
    NUM_SPECIES,
)
from rl.environment.interfaces import (
    BuilderActorInput,
    BuilderActorOutput,
    BuilderEnvOutput,
    BuilderHistoryOutput,
    PlayerActorInput,
    PlayerActorOutput,
    PlayerEnvOutput,
    PlayerHistoryOutput,
    PlayerPackedHistoryOutput,
    PolicyHeadOutput,
)
from rl.environment.protos.enums_pb2 import SpeciesEnum
from rl.environment.protos.features_pb2 import (
    EntityRevealedNodeFeature,
    FieldFeature,
    InfoFeature,
)
from rl.environment.protos.service_pb2 import EnvironmentState

T = TypeVar("T")


def split_rng(key: jax.Array, num_splits: int = 2) -> tuple[jax.Array, jax.Array]:
    return jax.random.split(key, num_splits)


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
    rounded_length = max(
        resolution, int(np.ceil(history_length / resolution) * resolution)
    )

    return jax.tree.map(lambda x: x[:rounded_length], history)


def clip_packed_history(
    packed_history: PlayerPackedHistoryOutput, resolution: int = 64
) -> PlayerPackedHistoryOutput:
    history_length = np.max(
        (
            packed_history.revealed[
                ..., EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
            ]
            != SpeciesEnum.SPECIES_ENUM___UNSPECIFIED
        ).sum(0),
        axis=0,
    ).item()

    # Round history length up to the nearest multiple of resolution
    rounded_length = max(
        resolution, int(np.ceil(history_length / resolution) * resolution)
    )

    return jax.tree.map(lambda x: x[:rounded_length], packed_history)


def get_action_mask(state: EnvironmentState):
    buffer = np.frombuffer(state.action_mask, dtype=np.uint8)
    mask = np.unpackbits(buffer, axis=-1)[: NUM_ACTION_FEATURES**2]
    return mask.astype(bool).reshape(NUM_ACTION_FEATURES, NUM_ACTION_FEATURES)


def process_state(
    state: EnvironmentState, max_history: int = NUM_HISTORY
) -> PlayerActorInput:
    history_length = state.history_length
    history_packed_length = max(1, state.history_packed_length)

    info = np.frombuffer(state.info, dtype=np.int16).astype(np.int32)
    max_packed_history = 2 * max_history

    history_entity_public = padnstack(
        np.frombuffer(state.history_entity_public, dtype=np.int16).reshape(
            (history_packed_length, NUM_ENTITY_PUBLIC_FEATURES)
        ),
        max_packed_history,
    ).astype(np.int32)
    history_entity_revealed = padnstack(
        np.frombuffer(state.history_entity_revealed, dtype=np.int16).reshape(
            (history_packed_length, NUM_ENTITY_REVEALED_FEATURES)
        ),
        max_packed_history,
    ).astype(np.int32)
    history_entity_edges = padnstack(
        np.frombuffer(state.history_entity_edges, dtype=np.int16).reshape(
            (history_packed_length, NUM_ENTITY_EDGE_FEATURES)
        ),
        max_packed_history,
    ).astype(np.int32)
    history_field = padnstack(
        np.frombuffer(state.history_field, dtype=np.int16).reshape(
            (history_length, NUM_FIELD_FEATURES)
        ),
        max_history,
    ).astype(np.int32)

    moveset = (
        np.frombuffer(state.moveset, dtype=np.int16)
        .reshape(16, NUM_MOVE_FEATURES)
        .astype(np.int32)
    )
    private_team = (
        np.frombuffer(state.private_team, dtype=np.int16)
        .reshape(6, NUM_ENTITY_PRIVATE_FEATURES)
        .astype(np.int32)
    )
    revealed_team = (
        np.frombuffer(state.revealed_team, dtype=np.int16)
        .reshape(11 * 2, NUM_ENTITY_REVEALED_FEATURES)
        .astype(np.int32)
    )
    public_team = (
        np.frombuffer(state.public_team, dtype=np.int16)
        .reshape(11 * 2, NUM_ENTITY_PUBLIC_FEATURES)
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

    fib_reward_token = info[InfoFeature.INFO_FEATURE__FIB_REWARD]
    # Divide by MAX_RATIO_TOKEN to normalize the fib reward to [-1, 1] since we store as int16
    fib_reward = fib_reward_token / MAX_RATIO_TOKEN

    env_step = PlayerEnvOutput(
        info=info,
        done=info[InfoFeature.INFO_FEATURE__DONE].astype(np.bool_),
        win_reward=win_reward.astype(np.float32),
        fib_reward=fib_reward.astype(np.float32),
        private_team=private_team,
        public_team=public_team,
        revealed_team=revealed_team,
        field=field,
        moveset=moveset,
        action_mask=get_action_mask(state),
    )
    packed_history_step = PlayerPackedHistoryOutput(
        public=history_entity_public,
        revealed=history_entity_revealed,
        edges=history_entity_edges,
    )
    history_step = PlayerHistoryOutput(field=history_field)

    return PlayerActorInput(
        env=env_step, packed_history=packed_history_step, history=history_step
    )


def get_ex_trajectory() -> PlayerActorInput:
    states = []
    for state in EX_TRAJECTORY.states:
        processed_state = process_state(state)
        states.append(processed_state.env)
    return PlayerActorInput(
        env=jax.tree.map(lambda *xs: np.stack(xs), *states),
        packed_history=processed_state.packed_history,
        history=processed_state.history,
    )


def get_ex_player_step() -> tuple[PlayerActorInput, PlayerActorOutput]:
    ts = get_ex_trajectory()
    env: PlayerEnvOutput = jax.tree.map(lambda x: x[:, None, ...], ts.env)
    packed_history: PlayerPackedHistoryOutput = jax.tree.map(
        lambda x: x[:, None, ...], ts.packed_history
    )
    history: PlayerHistoryOutput = jax.tree.map(lambda x: x[:, None, ...], ts.history)
    return (
        PlayerActorInput(env=env, packed_history=packed_history, history=history),
        PlayerActorOutput(
            value_head=np.zeros_like(env.info[..., 0], dtype=np.float32),
            action_head=PolicyHeadOutput(
                action_index=env.action_mask.reshape(
                    env.action_mask.shape[:-2] + (-1,)
                ).argmax(-1)
            ),
        ),
    )


def get_ex_builder_step() -> tuple[BuilderActorInput, BuilderActorOutput]:
    trajectory_length = 7
    history_length = 6
    done = np.zeros((trajectory_length, 1), dtype=np.bool_)
    done[-1] = True
    ts = np.arange(trajectory_length, dtype=np.int32)[:, None]
    return (
        BuilderActorInput(
            env=BuilderEnvOutput(
                species_mask=np.ones(
                    (trajectory_length, 1, NUM_SPECIES), dtype=np.bool
                ),
                ts=ts,
                done=done,
            ),
            history=BuilderHistoryOutput(
                species_tokens=np.zeros((history_length, 1), dtype=np.int32),
                packed_set_tokens=np.zeros((history_length, 1), dtype=np.int32),
            ),
        ),
        BuilderActorOutput(
            value_head=np.zeros_like(done, dtype=np.float32),
            species_head=PolicyHeadOutput(
                action_index=np.zeros_like(done, dtype=np.int32)
            ),
            packed_set_head=PolicyHeadOutput(
                action_index=np.zeros_like(done, dtype=np.int32)
            ),
        ),
    )


def main():
    """Main function for testing the utilities."""
    ex_player_input, ex_player_output = get_ex_player_step()
    ex_builder_input, ex_builder_output = get_ex_builder_step()

    rewards_tm1 = ex_player_input.env.win_reward.squeeze(-1)
    valid = np.bitwise_not(ex_player_input.env.done).squeeze(-1).astype(np.float32)

    rewards = np.concatenate((rewards_tm1[1:], np.zeros_like(rewards_tm1[-1:])))

    oracle_value = rewards[::-1].cumsum()[::-1]
    perturbed_oracle_value = oracle_value + np.random.normal(
        scale=0.5, size=oracle_value.shape
    )

    v_tm1 = valid * perturbed_oracle_value
    v_t = np.concatenate((v_tm1[1:], v_tm1[-1:]))
    discounts = (np.concatenate((valid[1:], np.zeros_like(valid[-1:])))).astype(
        v_t.dtype
    )

    vtrace = rlax.vtrace_td_error_and_advantage(
        v_tm1, v_t, rewards, discounts, valid, lambda_=0.95
    )
    v_tm1 + vtrace.errors
    print()


if __name__ == "__main__":
    main()
