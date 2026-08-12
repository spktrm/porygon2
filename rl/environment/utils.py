import heapq
import math
import threading
from collections.abc import Sequence
from typing import TypeVar

_tqdm_position_lock = threading.Lock()
_tqdm_free_positions: list[int] = []
_tqdm_positions_issued = 0


def next_tqdm_position() -> int:
    """Assigns each tqdm progress bar a unique, stable terminal row (via
    tqdm's position= kwarg / ANSI cursor movement) instead of letting
    concurrent bars fight over "the current line" with bare \\r redraws —
    with up to 4 bars per population (player_producer/builder_producer/
    consumer/batches) across 3 concurrently-training populations, that
    fight is what corrupted terminal output into garbled interleaved
    text. Call once per tqdm() construction, at bar-creation time, and
    pair with close_tqdm_bar() at teardown: exploiter populations are
    reset repeatedly, and without recycling rows each reset would place
    its 4 new bars one screen-row lower, leaving the dead rows above
    permanently occupied for the life of the process."""
    global _tqdm_positions_issued
    with _tqdm_position_lock:
        if _tqdm_free_positions:
            return heapq.heappop(_tqdm_free_positions)
        position = _tqdm_positions_issued
        _tqdm_positions_issued += 1
        return position


def close_tqdm_bar(bar) -> None:
    """Closes a bar created with position=next_tqdm_position() and returns
    its terminal row to the pool for the next bar. tqdm stores an
    explicitly-passed position negated in .pos (its marker for "fixed
    position"), so abs() recovers what next_tqdm_position() issued.
    Safe to call more than once: close() sets .disable, which gates the
    release here so the same row can't be pushed to the free pool twice
    (a double release would hand one terminal row to two live bars)."""
    if bar.disable:
        return
    position = abs(bar.pos)
    bar.close()
    with _tqdm_position_lock:
        heapq.heappush(_tqdm_free_positions, position)

import jax
import jax.numpy as jnp
import numpy as np

from constants import NUM_HISTORY
from rl.environment.data import (
    EX_BATCH,
    NUM_ABILITIES,
    NUM_ACTION_FEATURES,
    NUM_ENTITY_EDGE_FEATURES,
    NUM_ENTITY_PRIVATE_FEATURES,
    NUM_ENTITY_PUBLIC_FEATURES,
    NUM_ENTITY_REVEALED_FEATURES,
    NUM_FIELD_FEATURES,
    NUM_GENDERS,
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
    result[:length_to_copy] = arr[-length_to_copy:]
    return result


def expand_dims(x, axis: int):
    return jax.tree.map(lambda i: np.expand_dims(i, axis=axis), x)


def _bucket_level(length: int, lo: int) -> int:
    """Number of lo-doublings needed to reach at least `length` (>= 0),
    uncapped by any hi — see geometric_bucket. Exposed separately so
    callers with multiple correlated length signals (e.g. _stack_and_pad_
    batch's player_transitions/history/packed_history, which all describe
    the same underlying game length) can take one shared max level instead
    of each independently picking its own — see geometric_bucket's
    docstring for why that matters."""
    if length <= lo:
        return 0
    return math.ceil(math.log2(length / lo))


def _bucket_value(level: int, lo: int, hi: int) -> int:
    return min(hi, lo * 2**level)


def geometric_bucket(length: int, lo: int, hi: int) -> int:
    """Rounds length up to the next lo * 2^k, capped at hi.

    Geometric buckets bound the number of distinct clipped shapes (and thus
    JIT recompilations) to log2(hi / lo) + 1 for a SINGLE length signal —
    if a jitted function's batch depends on multiple independently-bucketed
    fields, the actual number of distinct shape combinations XLA sees is
    the PRODUCT across fields, not the sum. When those fields are
    correlated (e.g. all describing the same trajectory's game length),
    prefer computing one shared level via _bucket_level per field, taking
    their max, and applying it uniformly via _bucket_value — see
    rl/online/learner.py's _stack_and_pad_batch.
    """
    return _bucket_value(_bucket_level(length, lo), lo, hi)


def _history_level(history: PlayerHistoryOutput, min_length: int) -> int:
    history_length = np.max(
        history.field[..., FieldFeature.FIELD_FEATURE__VALID].sum(0),
        axis=0,
    ).item()
    return _bucket_level(history_length, min_length)


def clip_history(
    history: PlayerHistoryOutput, min_length: int = 64, level: int | None = None
) -> PlayerHistoryOutput:
    if level is None:
        level = _history_level(history, min_length)
    rounded_length = _bucket_value(level, min_length, history.field.shape[0])
    return jax.tree.map(lambda x: x[:rounded_length], history)


def _packed_history_level(
    packed_history: PlayerPackedHistoryOutput, min_length: int
) -> int:
    history_length = np.max(
        (
            packed_history.revealed_cache[
                ..., EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
            ]
            != SpeciesEnum.SPECIES_ENUM___UNSPECIFIED
        ).sum(0),
        axis=0,
    ).item()
    return _bucket_level(history_length, min_length)


def clip_packed_history(
    packed_history: PlayerPackedHistoryOutput,
    min_length: int = 64,
    level: int | None = None,
) -> PlayerPackedHistoryOutput:
    if level is None:
        level = _packed_history_level(packed_history, min_length)
    rounded_length = _bucket_value(
        level, min_length, packed_history.revealed_cache.shape[0]
    )
    return jax.tree.map(lambda x: x[:rounded_length], packed_history)


def get_action_mask(state: EnvironmentState):
    buffer = np.frombuffer(state.action_mask, dtype=np.uint8)
    mask = np.unpackbits(buffer, axis=-1)[: NUM_ACTION_FEATURES**2]
    return mask.astype(bool).reshape(NUM_ACTION_FEATURES, NUM_ACTION_FEATURES)


def process_state(
    state: EnvironmentState,
    max_history: int = NUM_HISTORY,
    with_history: bool = True,
) -> PlayerActorInput:
    """Decodes an EnvironmentState proto into a PlayerActorInput.

    with_history=False skips materialising the (large, padded) history
    caches and returns empty history pytrees — use it when only the env
    fields are needed (e.g. the offline dataset keeps just the final
    state's histories per trajectory).
    """
    info = np.frombuffer(state.info, dtype=np.int16).astype(np.int32)
    # Older exports predate trailing InfoFeature additions (e.g. the rating
    # features); zero-pad so static feature indexing stays in bounds — zero
    # is every such feature's "unknown" value.
    num_info_features = len(InfoFeature.keys())
    if info.shape[0] < num_info_features:
        info = np.pad(info, (0, num_info_features - info.shape[0]))

    if with_history:
        history_length = state.history_length
        history_packed_length = max(1, state.history_packed_length)
        max_packed_history = 2 * max_history

        history_entity_public_cache = padnstack(
            np.frombuffer(state.history_entity_public_cache, dtype=np.int16).reshape(
                (history_packed_length, NUM_ENTITY_PUBLIC_FEATURES)
            ),
            max_packed_history,
        ).astype(np.int32)
        history_entity_revealed_cache = padnstack(
            np.frombuffer(state.history_entity_revealed_cache, dtype=np.int16).reshape(
                (history_packed_length, NUM_ENTITY_REVEALED_FEATURES)
            ),
            max_packed_history,
        ).astype(np.int32)
        history_entity_edge_cache = padnstack(
            np.frombuffer(state.history_entity_edge_cache, dtype=np.int16).reshape(
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

    my_moveset = (
        np.frombuffer(state.my_moveset, dtype=np.int16)
        .reshape(16, NUM_MOVE_FEATURES)
        .astype(np.int32)
    )
    opp_moveset = (
        np.frombuffer(state.opp_moveset, dtype=np.int16)
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
        ],
        dtype=np.float32,
    )

    env_step = PlayerEnvOutput(
        info=info,
        done=is_done,
        win_reward=win_reward.astype(np.float32),
        private_team=private_team,
        public_team=public_team,
        revealed_team=revealed_team,
        field=field,
        my_moveset=my_moveset,
        opp_moveset=opp_moveset,
        action_mask=get_action_mask(state),
    )
    if with_history:
        packed_history_step = PlayerPackedHistoryOutput(
            public_cache=history_entity_public_cache,
            revealed_cache=history_entity_revealed_cache,
            edge_cache=history_entity_edge_cache,
        )
        history_step = PlayerHistoryOutput(field=history_field)
    else:
        packed_history_step = PlayerPackedHistoryOutput()
        history_step = PlayerHistoryOutput()

    return PlayerActorInput(
        env=env_step, packed_history=packed_history_step, history=history_step
    )


def get_ex_batch(min_length: int = 64) -> PlayerActorInput:
    processed_states = []
    for i, unprocessed_states in enumerate(EX_BATCH.trajectories):
        states = []
        for state in unprocessed_states.states:
            processed_state = process_state(state)
            states.append(processed_state.env)

        done_state = processed_state.env.replace(
            done=np.ones_like(processed_state.env.done)
        )
        states += [done_state] * (EX_BATCH.max_trajectory_length - len(states))

        processed_states.append(
            PlayerActorInput(
                env=jax.tree.map(lambda *xs: np.stack(xs), *states),
                packed_history=processed_state.packed_history,
                history=processed_state.history,
            )
        )

    ex_batch: PlayerActorInput = jax.tree.map(
        lambda *xs: np.stack(xs, axis=1), *processed_states
    )
    ex_batch = ex_batch.replace(
        packed_history=clip_packed_history(
            ex_batch.packed_history, min_length=min_length
        ),
        history=clip_history(ex_batch.history, min_length=min_length),
    )

    return ex_batch


def get_ex_trajectory() -> PlayerActorInput:
    states = []
    for state in EX_BATCH.trajectories[0].states:
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


def generate_order(key: jax.Array, r: int, N: int):
    total_size = r * N
    selection_order = jax.random.permutation(key, jnp.arange(total_size))

    # 1. Entry point is now i % N == 1
    entry_priorities = selection_order.reshape(r, N)[:, 1]
    block_gate_priority = jnp.repeat(entry_priorities, N)

    # 2. Effective priority
    effective_priority = jnp.maximum(selection_order, block_gate_priority)

    # 3. Get the full sorted order
    sorted_indices = jnp.argsort(effective_priority)

    # 4. FIXED: Instead of boolean masking, we use a static filter
    # We find which positions in the 'sorted_indices' do NOT contain an i % N == 0
    # But wait—it's easier to just calculate the valid indices first!

    # Alternative JIT-safe approach:
    # Use jnp.where with a fixed-size size argument or simple slicing if possible.
    # Since we must return r * (N-1), we use jnp.take with static indices.

    is_valid = (sorted_indices % N) != 0
    # Sort the boolean mask to push all 'True' values to the front
    # and then slice the first r*(N-1) elements.
    valid_positions = jnp.argsort(~is_valid)

    return sorted_indices[valid_positions[: r * (N - 1)]]


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
                hp_ev_mask=np.ones((trajectory_length, 1, 64), dtype=np.bool),
                atk_ev_mask=np.ones((trajectory_length, 1, 64), dtype=np.bool),
                def_ev_mask=np.ones((trajectory_length, 1, 64), dtype=np.bool),
                spa_ev_mask=np.ones((trajectory_length, 1, 64), dtype=np.bool),
                spd_ev_mask=np.ones((trajectory_length, 1, 64), dtype=np.bool),
                spe_ev_mask=np.ones((trajectory_length, 1, 64), dtype=np.bool),
                nature_mask=np.ones((trajectory_length, 1, NUM_NATURES), dtype=np.bool),
                gender_mask=np.ones((trajectory_length, 1, NUM_GENDERS), dtype=np.bool),
                teratype_mask=np.ones(
                    (trajectory_length, 1, NUM_TYPECHART), dtype=np.bool
                ),
                species_usage=np.ones(
                    (trajectory_length, 1, NUM_SPECIES), dtype=np.bool
                ),
                item_usage=np.ones((trajectory_length, 1, NUM_ITEMS), dtype=np.bool),
                ability_usage=np.ones(
                    (trajectory_length, 1, NUM_ABILITIES), dtype=np.bool
                ),
                move_usage=np.ones((trajectory_length, 1, NUM_MOVES), dtype=np.bool),
                hp_ev_usage=np.ones((trajectory_length, 1, 64), dtype=np.bool),
                atk_ev_usage=np.ones((trajectory_length, 1, 64), dtype=np.bool),
                def_ev_usage=np.ones((trajectory_length, 1, 64), dtype=np.bool),
                spa_ev_usage=np.ones((trajectory_length, 1, 64), dtype=np.bool),
                spd_ev_usage=np.ones((trajectory_length, 1, 64), dtype=np.bool),
                spe_ev_usage=np.ones((trajectory_length, 1, 64), dtype=np.bool),
                nature_usage=np.ones(
                    (trajectory_length, 1, NUM_NATURES), dtype=np.bool
                ),
                gender_usage=np.ones(
                    (trajectory_length, 1, NUM_GENDERS), dtype=np.bool
                ),
                teratype_usage=np.ones(
                    (trajectory_length, 1, NUM_TYPECHART), dtype=np.bool
                ),
                ts=ts,
                curr_order=order,
                curr_attribute=member_attribute,
                curr_position=member_position,
                done=done,
                ev_reward=np.zeros_like(done, dtype=np.float32),
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
