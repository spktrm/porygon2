import heapq
import math
import threading
from typing import TypeVar

_tqdm_position_lock = threading.Lock()
_tqdm_free_positions: list[int] = []
_tqdm_positions_issued = 0


def next_tqdm_position() -> int:
    """Assigns each tqdm progress bar a unique, stable terminal row (via
    tqdm's position= kwarg / ANSI cursor movement) instead of letting
    concurrent bars fight over "the current line" with bare \\r redraws —
    with 4 concurrent bars (player_producer/builder_producer/consumer/
    batches), that fight is what corrupted terminal output into garbled
    interleaved text. Call once per tqdm() construction, at bar-creation
    time, and pair with close_tqdm_bar() at teardown: bars are rebuilt on
    restart, and without recycling rows each rebuild would place
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
    ALLY_SWITCH_INDICES,
    EX_BATCH,
    MOVE_CELL_OFFSET,
    MOVE_INDICES,
    NUM_ABILITIES,
    NUM_ACTION_CELLS,
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
    NUM_SWITCH_CELLS,
    NUM_TARGET_SLOTS,
    NUM_TYPECHART,
    OTHER_CELL_OFFSET,
    RESERVE_ENTITY_INDICES,
    TARGET_SLOT_INDICES,
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
from rl.environment.protos.service_pb2 import (
    ActionMask,
    ActionRequestKind,
    EnvironmentState,
)
from rl.model.heads import CategoricalValueHeadOutput

T = TypeVar("T")


def split_rng(key: jax.Array, num_splits: int = 2) -> tuple[jax.Array, jax.Array]:
    return jax.random.split(key, num_splits)


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
    rl/online/training/batching.py (now fixed-shape stack_batch).
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


# All eight RELEVANT_ENTITY_IDX columns (the model's _RELEVANT_ENTITY_
# FEATURES reads the first four; the service writes up to eight).
_ALL_RELEVANT_IDX_COLUMNS = np.array(
    [FieldFeature.Value(f"FIELD_FEATURE__RELEVANT_ENTITY_IDX{k}") for k in range(8)]
)


def _packed_valid_rows(packed_history: PlayerPackedHistoryOutput) -> int:
    """Occupied packed-cache rows, inferred from the species sentinel (the
    known-open derivation -- CLAUDE.md; the suffix cut and the tail cut
    share it so both flip together)."""
    return int(
        np.asarray(
            packed_history.revealed_cache[
                ..., EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
            ]
            != SpeciesEnum.SPECIES_ENUM___UNSPECIFIED
        ).sum()
    )


def _cut_history_windows(
    history: PlayerHistoryOutput,
    packed_history: PlayerPackedHistoryOutput,
    start_step: int,
    keep_steps: int,
    start_row: int,
    packed_end: int,
    history_rows: int,
    packed_rows: int,
) -> tuple[PlayerHistoryOutput, PlayerPackedHistoryOutput]:
    """Cut field steps [start_step, start_step + keep_steps) and packed rows
    [start_row, packed_end) into zero-padded windows of exactly
    (history_rows, packed_rows), rebasing every RELEVANT_ENTITY_IDX column
    of the kept steps by -start_row -- the ONE place the two axes are cut
    together (state.ts getHistory's counterpart), shared by the trailing
    window and the actor's suffix."""
    field = np.asarray(history.field)
    new_field = np.zeros((history_rows, field.shape[1]), dtype=field.dtype)
    new_field[:keep_steps] = field[start_step : start_step + keep_steps]
    if start_row > 0 and keep_steps > 0:
        # Rebase every index column of the kept rows; entries past a row's
        # NUM_RELEVANT are padding the consumers mask out — clip keeps them
        # in gather range regardless.
        rebase_at = np.ix_(np.arange(keep_steps), _ALL_RELEVANT_IDX_COLUMNS)
        new_field[rebase_at] = np.clip(
            new_field[rebase_at] - start_row, 0, packed_rows - 1
        )

    keep_rows = max(0, packed_end - start_row)

    def cut_packed(x) -> np.ndarray:
        x = np.asarray(x)
        out = np.zeros((packed_rows, *x.shape[1:]), dtype=x.dtype)
        out[:keep_rows] = x[start_row:packed_end]
        return out

    return (
        history.replace(field=new_field),
        jax.tree.map(cut_packed, packed_history),
    )


def clip_history_windows_tail(
    history: PlayerHistoryOutput,
    packed_history: PlayerPackedHistoryOutput,
    history_length: int,
) -> tuple[PlayerHistoryOutput, PlayerPackedHistoryOutput]:
    """Host-side fixed-length trailing windows: the last ``history_length``
    field steps and the ``2 * history_length`` packed-cache rows they
    reference, zero-padded to exactly those shapes. One fixed shape means
    the learner never recompiles (chunked unrolls, 2026-08-16), unlike the
    geometric clip_* functions above which round variable lengths UP.

    The two axes CANNOT be cut independently: each field step names its
    packed rows by absolute row index (RELEVANT_ENTITY_IDX*), so this
    mirrors the service's own windowing (state.ts getHistory) exactly —
    shrink the field window until its oldest step's first packed row
    (IDX0; rows are appended per step, so per-step row blocks are
    contiguous and ascending) fits the packed budget, slice the caches
    from that row, and rebase the index columns to the new start."""
    field = np.asarray(history.field)
    valid_steps = int(field[:, FieldFeature.FIELD_FEATURE__VALID].sum())
    packed_valid = _packed_valid_rows(packed_history)
    max_packed_rows = 2 * history_length

    keep_steps = min(valid_steps, history_length)
    start_row = 0
    while keep_steps > 0:
        oldest_step = valid_steps - keep_steps
        oldest_row = int(
            field[oldest_step, FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX0]
        )
        if packed_valid - oldest_row <= max_packed_rows:
            start_row = oldest_row
            break
        keep_steps -= 1

    if keep_steps == 0 and valid_steps > 0:
        # Degenerate guard, mirroring the service's max(1, ...): a single
        # step referencing more than the whole packed budget cannot occur
        # for real games, but never ship an empty window for a non-empty
        # history.
        keep_steps = 1
        start_row = int(
            field[valid_steps - 1, FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX0]
        )

    packed_end = min(packed_valid, start_row + max_packed_rows)
    return _cut_history_windows(
        history,
        packed_history,
        start_step=valid_steps - keep_steps,
        keep_steps=keep_steps,
        start_row=start_row,
        packed_end=packed_end,
        history_rows=history_length,
        packed_rows=max_packed_rows,
    )


# The actor path's geometric-bucket base for BOTH history axes: the actor
# clips to these bucket values and the inference server re-buckets the
# batch against the same base (a shared level per group), so the two must
# agree -- written once, read by both.
ACTOR_HISTORY_MIN_LENGTH = 64


def clip_history_suffix(
    actor_input: PlayerActorInput,
    last_step_index: int,
    min_length: int = ACTOR_HISTORY_MIN_LENGTH,
) -> tuple[PlayerActorInput | None, int]:
    """The actor's incremental window: the field steps AFTER absolute step
    ``last_step_index`` (FIELD_FEATURE__INDEX, the carry's last consumed
    step; -1 = nothing consumed) and exactly the packed rows they reference,
    each axis rounded up to its own geometric bucket -> (the clipped input,
    the number of new steps). Returns (None, 0) when the window cannot be
    resumed from that point -- the carried step is neither in the window
    nor immediately before it -- and the caller recomputes from scratch.
    Zero new steps is a valid suffix: an all-zero window, from which the
    encoder returns the carry itself."""
    field = np.asarray(actor_input.history.field)
    valid_steps = int(field[:, FieldFeature.FIELD_FEATURE__VALID].sum())
    index = field[:valid_steps, FieldFeature.FIELD_FEATURE__INDEX]
    first = int(np.searchsorted(index, last_step_index, side="right"))
    if first == 0:
        if valid_steps == 0:
            contiguous = last_step_index == -1
        else:
            contiguous = int(index[0]) == last_step_index + 1
    else:
        contiguous = int(index[first - 1]) == last_step_index
    if not contiguous:
        return None, 0

    keep_steps = valid_steps - first
    packed_end = _packed_valid_rows(actor_input.packed_history)
    if keep_steps > 0:
        start_row = int(field[first, FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX0])
    else:
        start_row = packed_end
    history_rows = _bucket_value(
        _bucket_level(keep_steps, min_length), min_length, field.shape[0]
    )
    packed_rows = _bucket_value(
        _bucket_level(packed_end - start_row, min_length),
        min_length,
        actor_input.packed_history.revealed_cache.shape[0],
    )
    history, packed_history = _cut_history_windows(
        actor_input.history,
        actor_input.packed_history,
        start_step=first,
        keep_steps=keep_steps,
        start_row=start_row,
        packed_end=packed_end,
        history_rows=history_rows,
        packed_rows=packed_rows,
    )
    return (
        actor_input.replace(history=history, packed_history=packed_history),
        keep_steps,
    )


def _cells_from_structured_mask(mask: ActionMask) -> np.ndarray:
    """The block-space legal mask (NUM_ACTION_CELLS,) from the wire's mask.

    The block layout is the flattening of ActionMask's own fields in field
    order (proto/service.proto `Action`): the 6 switch bits, then the 16x17
    move_targets rows, then the standalone bits. Bit positions index the
    ActionEnum slot lists both sides build, never raw enum values, and `kind`
    only matters to the DECODER (lead vs switch choice string) -- the mask
    cells are the same either way.
    """
    cells = np.zeros(NUM_ACTION_CELLS, dtype=bool)

    if mask.kind in (
        ActionRequestKind.ACTION_REQUEST_KIND___UNSPECIFIED,
        ActionRequestKind.ACTION_REQUEST_KIND__WAIT,
    ):
        # Nothing is being asked. The service lights every cell so masked
        # averages downstream never meet an empty row, and the decoder answers
        # "default" whichever cell comes back.
        cells[:] = True
        return cells

    for switch_bit in range(NUM_SWITCH_CELLS):
        cells[switch_bit] = (mask.switch_slots >> switch_bit) & 1

    for move_slot, targets in enumerate(mask.move_targets):
        if targets == 0:
            continue
        base = MOVE_CELL_OFFSET + move_slot * NUM_TARGET_SLOTS
        for target_bit in range(NUM_TARGET_SLOTS):
            cells[base + target_bit] = (targets >> target_bit) & 1

    for other_bit in range(NUM_TARGET_SLOTS):
        cells[OTHER_CELL_OFFSET + other_bit] = (mask.other_srcs >> other_bit) & 1
    return cells


def _cells_from_packed_grid(grid: np.ndarray) -> np.ndarray:
    """Block mask from a legacy 41x41 grid (replay shards only).

    The inverse of the retired scatter over the ~18% reachable cells: battle
    switches lived at (ALLY_i_SWITCH, RESERVE_j) and team-preview leads at
    (RESERVE_j, tgt), both folding onto switch cell j; moves at
    (MOVE_INDICES[m], TARGET_SLOT_INDICES[t]); standalone on the target-slot
    diagonal. An all-lit grid (the WAIT sentinel) folds onto all-lit cells.
    """
    cells = np.zeros(NUM_ACTION_CELLS, dtype=bool)
    switch_via_tgt = grid[ALLY_SWITCH_INDICES][:, RESERVE_ENTITY_INDICES].any(axis=0)
    switch_via_src = grid[RESERVE_ENTITY_INDICES].any(axis=-1)
    cells[:NUM_SWITCH_CELLS] = switch_via_tgt | switch_via_src
    move_block = grid[MOVE_INDICES][:, TARGET_SLOT_INDICES]
    cells[MOVE_CELL_OFFSET:OTHER_CELL_OFFSET] = move_block.reshape(-1)
    cells[OTHER_CELL_OFFSET:] = grid[TARGET_SLOT_INDICES, TARGET_SLOT_INDICES]
    return cells


def _decode_private_rows(raw: bytes) -> np.ndarray:
    """One private sheet -> (6, NUM_ENTITY_PRIVATE_FEATURES) int32.

    Right-pads a short buffer with zero COLUMNS to the current width: the
    replay shards (spectator logs, no |request|) store private blocks frozen
    at an older feature count, and 0 is UNSPECIFIED for every appended field
    -- padding is semantically exact where a reshape would raise. Appending
    private features is therefore safe; renumbering never is. An EMPTY buffer
    (opp_private_team on old shards and at deploy) decodes as all zeros --
    the documented "does not exist" encoding.
    """
    flat = np.frombuffer(raw, dtype=np.int16)
    if flat.shape[0] == 0:
        return np.zeros((6, NUM_ENTITY_PRIVATE_FEATURES), dtype=np.int32)
    rows = flat.reshape(6, flat.shape[0] // 6).astype(np.int32)
    missing_columns = NUM_ENTITY_PRIVATE_FEATURES - rows.shape[1]
    if missing_columns > 0:
        rows = np.pad(rows, ((0, 0), (0, missing_columns)))
    return rows


def get_action_mask(state: EnvironmentState):
    if state.HasField("structured_action_mask"):
        return _cells_from_structured_mask(state.structured_action_mask)
    # Replay shards predate the structured mask (2026-08-29) and carry the
    # 1681-bit packed grid instead. Delete this branch, and the proto field it
    # reads, when replays/shards is next rebuilt.
    buffer = np.frombuffer(state.packed_action_mask, dtype=np.uint8)
    mask = np.unpackbits(buffer, axis=-1)[: NUM_ACTION_FEATURES**2]
    grid = mask.astype(bool).reshape(NUM_ACTION_FEATURES, NUM_ACTION_FEATURES)
    return _cells_from_packed_grid(grid)


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
    # Right-pad a short private buffer with zero COLUMNS to the current
    # width: the replay shards (spectator logs, no |request|) store private
    # blocks frozen at an older feature count, and 0 is UNSPECIFIED for
    # every appended field -- padding is semantically exact where a reshape
    # would raise. Appending private features is therefore safe; renumbering
    # never is.
    private_team = _decode_private_rows(state.private_team)
    # The opponent truth channel (2026-09-01): absent entirely on old shards
    # and at deploy -- all-zero rows are the documented "does not exist"
    # encoding, matching the service's own zero-buffer branches.
    opp_private_team = _decode_private_rows(state.opp_private_team)
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
        opp_private_team=opp_private_team,
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
            action_head=PolicyHeadOutput(action_index=env.action_mask.argmax(-1)),
        ),
    )


def generate_order(key: jax.Array, r: int, N: int):
    total_size = r * N
    selection_order = jax.random.permutation(key, jnp.arange(total_size))

    # A block's entry slot is i % N == 1, and every member inherits that
    # slot's priority, so blocks stay contiguous through the sort.
    entry_priorities = selection_order.reshape(r, N)[:, 1]
    block_gate_priority = jnp.repeat(entry_priorities, N)
    effective_priority = jnp.maximum(selection_order, block_gate_priority)
    sorted_indices = jnp.argsort(effective_priority)

    # Drop the i % N == 0 entries by SORTING the mask, not filtering on it: a
    # boolean filter's output shape is data-dependent and jit cannot trace it,
    # whereas argsort(~is_valid) pushes the keepers to the front and leaves a
    # static r * (N - 1) slice.
    is_valid = (sorted_indices % N) != 0
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
