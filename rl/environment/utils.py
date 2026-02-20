from typing import Sequence, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from rl.environment.data import (
    EX_TRAJECTORY,
    MAX_RATIO_TOKEN,
    NUM_ABILITIES,
    NUM_ACTION_FEATURES,
    NUM_ENTITY_EDGE_FEATURES,
    NUM_ENTITY_PRIVATE_FEATURES,
    NUM_ENTITY_PUBLIC_FEATURES,
    NUM_ENTITY_REVEALED_FEATURES,
    NUM_FIELD_FEATURES,
    NUM_GENDERS,
    NUM_HISTORY,
    NUM_ITEMS,
    NUM_MOVE_FEATURES,
    NUM_MOVES,
    NUM_NATURES,
    NUM_PACKED_SET_FEATURES,
    NUM_SPECIES,
    NUM_TYPECHART,
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
    RegressionValueHeadOutput,
)
from rl.environment.protos.enums_pb2 import SpeciesEnum
from rl.environment.protos.features_pb2 import (
    EntityRevealedNodeFeature,
    FieldFeature,
    InfoFeature,
)
from rl.environment.protos.service_pb2 import EnvironmentState
from rl.model.heads import CategoricalValueHeadOutput

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


def jax_segmented_cumsum(x: jnp.ndarray, discount: jnp.ndarray) -> jnp.ndarray:
    """
    Parallel implementation of your @torch.jit.script loop.
    Replaces O(T) sequential loop with O(log T) associative scan.
    """

    def binary_op(right, left):
        # The scan operator for: y[t] = x[t] + discount[t] * y[t+1]
        val_l, disc_l = left
        val_r, disc_r = right
        return val_l + disc_l * val_r, disc_l * disc_r

    # Flip to treat the 'future' as the prefix for the scan
    vals, _ = jax.lax.associative_scan(binary_op, (x[::-1], discount[::-1]))
    return vals[::-1]


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
        .reshape(6 * 2, NUM_ENTITY_REVEALED_FEATURES)
        .astype(np.int32)
    )
    public_team = (
        np.frombuffer(state.public_team, dtype=np.int16)
        .reshape(6 * 2, NUM_ENTITY_PUBLIC_FEATURES)
        .astype(np.int32)
    )

    field = (
        np.frombuffer(state.field, dtype=np.int16)
        .reshape(NUM_FIELD_FEATURES)
        .astype(np.int32)
    )

    is_done = info[InfoFeature.INFO_FEATURE__DONE].astype(np.bool_)

    # Rewards are stored as int16 in the info array, so we need to convert them back to float32
    win_reward = np.array(
        [
            info[InfoFeature.INFO_FEATURE__LOSS_REWARD],
            info[InfoFeature.INFO_FEATURE__TIE_REWARD],
            info[InfoFeature.INFO_FEATURE__WIN_REWARD],
        ]
    )

    fib_reward_token = info[InfoFeature.INFO_FEATURE__FIB_REWARD]
    # Divide by MAX_RATIO_TOKEN to normalize the fib reward to [-1, 1] since we store as int16
    fib_reward = fib_reward_token / MAX_RATIO_TOKEN

    env_step = PlayerEnvOutput(
        info=info,
        done=is_done,
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
            value_head=CategoricalValueHeadOutput(
                logits=np.zeros((env.done.shape[0], 1, 3), dtype=np.float32),
                log_probs=np.zeros((env.done.shape[0], 1, 3), dtype=np.float32),
                expectation=np.zeros((env.done.shape[0], 1), dtype=np.float32),
            ),
            action_head=PolicyHeadOutput(
                action_index=env.action_mask.reshape(
                    env.action_mask.shape[:-2] + (-1,)
                ).argmax(-1)
            ),
        ),
    )


@jax.jit(static_argnames=["r", "N"])
def generate_order(key, r, N):
    """
    Generates a build order where Species (Index 1) is always selected first
    for each team member, followed by a random permutation of the remaining attributes.

    Args:
        key: JAX random key
        r: Number of team members (rows)
        N: Number of features per member (NUM_PACKED_SET_FEATURES)
    """
    # 1. Define the specific index for Species (Enum Value 1)
    SPECIES_IDX = 1

    # 2. Identify "Other" attributes to shuffle (Indices 2 to N-1)
    # We skip 0 (UNSPECIFIED) and 1 (SPECIES)
    other_indices = jnp.arange(SPECIES_IDX + 1, N)

    # 3. Define a function to generate the order for a single team member
    def get_member_order(k):
        # Shuffle only the non-species attributes
        shuffled_others = jax.random.permutation(k, other_indices)
        # Prepend Species index to the shuffled others
        return jnp.concatenate([jnp.array([SPECIES_IDX]), shuffled_others])

    # 4. Generate keys for each team member
    keys = jax.random.split(key, r)

    # 5. Apply logic to all members (r, N-1)
    local_orders = jax.vmap(get_member_order)(keys)

    # 6. Add offsets to convert local indices (0..N-1) to global flattened indices
    # Shape (r, 1) broadcasted to (r, N-1)
    offsets = (jnp.arange(r) * N)[:, None]
    global_orders = local_orders + offsets

    # 7. Flatten to match the expected 1D output shape
    return global_orders.reshape(-1)


def get_ex_builder_step() -> tuple[BuilderActorInput, BuilderActorOutput]:
    trajectory_length = 6 * (NUM_PACKED_SET_FEATURES - 1)
    6 * NUM_PACKED_SET_FEATURES
    done = np.zeros((trajectory_length, 1), dtype=np.bool_)
    done[-1] = True
    ts = np.arange(trajectory_length, dtype=np.int32)[:, None]

    packed_team_member_tokens = np.zeros(
        (6 * NUM_PACKED_SET_FEATURES, 1), dtype=np.int32
    )

    order = generate_order(jax.random.key(42), 6, NUM_PACKED_SET_FEATURES)[:, None]
    member_position = order // NUM_PACKED_SET_FEATURES
    member_attribute = order % NUM_PACKED_SET_FEATURES

    return (
        BuilderActorInput(
            env=BuilderEnvOutput(
                species_mask=np.ones(
                    (trajectory_length, 1, NUM_SPECIES), dtype=np.bool
                ),
                item_mask=np.ones((trajectory_length, 1, NUM_ITEMS), dtype=np.bool),
                ability_mask=np.ones(
                    (trajectory_length, 1, NUM_ABILITIES), dtype=np.bool
                ),
                move_mask=np.ones((trajectory_length, 1, NUM_MOVES), dtype=np.bool),
                ev_mask=np.ones((trajectory_length, 1, 64), dtype=np.bool),
                nature_mask=np.ones((trajectory_length, 1, NUM_NATURES), dtype=np.bool),
                gender_mask=np.ones((trajectory_length, 1, NUM_GENDERS), dtype=np.bool),
                teratype_mask=np.ones(
                    (trajectory_length, 1, NUM_TYPECHART), dtype=np.bool
                ),
                ts=ts,
                curr_order=order,
                curr_attribute=member_attribute,
                curr_position=member_position,
                done=done,
            ),
            history=BuilderHistoryOutput(
                packed_team_member_tokens=packed_team_member_tokens,
                order=order,
                member_attribute=member_attribute,
                member_position=member_position,
            ),
        ),
        BuilderActorOutput(
            conditional_entropy_head=RegressionValueHeadOutput(
                logits=np.zeros_like(done, dtype=np.float32)
            ),
            value_head=CategoricalValueHeadOutput(
                logits=np.zeros((done.shape[0], 1, 3), dtype=np.float32),
                log_probs=np.zeros((done.shape[0], 1, 3), dtype=np.float32),
                expectation=np.zeros((done.shape[0], 1), dtype=np.float32),
            ),
            action_head=PolicyHeadOutput(
                action_index=np.zeros_like(done, dtype=np.int32)
            ),
        ),
    )


def main():
    """Main function for testing the utilities."""
    ex_player_input, ex_player_output = get_ex_player_step()
    ex_builder_input, ex_builder_output = get_ex_builder_step()

    valid = jnp.concatenate(
        (
            np.ones_like(ex_builder_input.env.done),
            np.bitwise_not(ex_player_input.env.done),
        )
    )

    value_probs = jnp.exp(
        jnp.concatenate(
            (
                ex_builder_output.value_head.log_probs,
                ex_player_output.value_head.log_probs,
            ),
            axis=0,
        )
    )
    value_probs = value_probs / value_probs.sum(axis=-1, keepdims=True)
    value_expectation = value_probs @ (
        jnp.arange(value_probs.shape[-1], dtype=jnp.float32) - 1
    )

    cat_reward = jnp.concatenate(
        (
            jnp.zeros_like(ex_builder_output.value_head.log_probs),
            ex_player_input.env.win_reward,
        )
    )
    value_target = (
        cat_reward
        + jnp.concatenate((value_probs[1:], jnp.zeros_like(value_probs[0:1])))
        * valid[..., None]
    )
    scalar_value_target = cat_reward @ (
        jnp.arange(value_probs.shape[-1], dtype=jnp.float32) - 1
    )

    cat_value_delta = value_target - value_probs
    scalar_value_delta = value_expectation - scalar_value_target

    td_lambda = 0.8
    gae_lambda = 0.5

    td_lambdas = td_lambda * valid.astype(cat_value_delta.dtype)[..., None]
    gae_lambdas = gae_lambda * valid.astype(cat_value_delta.dtype)

    segmented_cumsum = jax.vmap(jax_segmented_cumsum, in_axes=(1, 1), out_axes=1)
    returns = segmented_cumsum(cat_value_delta, td_lambdas) + value_probs
    advantages = segmented_cumsum(scalar_value_delta, gae_lambdas)
    print(returns, advantages)


if __name__ == "__main__":
    main()
