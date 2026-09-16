"""Per-step event labels over a trajectory's public history.

One history step is one service edge (a |move|, |switch|/|drag|/|replace|,
|cant| or |faint| line with its minor lines folded in; |turn| and |done|
only commit the pending edge). The world model's decoder predicts, from
the state after step k, the event that step k+1 records, so every label
here is indexed by k and describes event k+1. Numpy only: this runs in the
dataset workers.
"""

from enum import IntEnum

import chex
import numpy as np
from jaxtyping import ArrayLike

from rl.environment.protos.enums_pb2 import BattlemajorargsEnum, MovesEnum
from rl.environment.protos.features_pb2 import (
    EntityEdgeFeature,
    EntityPublicNodeFeature,
    EntityRevealedNodeFeature,
    FieldFeature,
)
from rl.model.constants import NUM_PUBLIC_SLOTS, RELEVANT_ENTITY_FEATURES

SIDE_MINE = 1
NO_SLOT = NUM_PUBLIC_SLOTS


class EventKind(IntEnum):
    MOVE = 0
    SWITCH = 1
    DRAG = 2
    CANT = 3
    FAINT = 4
    RESIDUAL = 5
    END = 6


NUM_EVENT_KINDS = len(EventKind)


class DeclaredKind(IntEnum):
    UNKNOWN = 0
    MOVE = 1
    SWITCH = 2


NUM_DECLARED_KINDS = len(DeclaredKind)

_MAJOR_TO_KIND = {
    BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__MOVE: EventKind.MOVE,
    BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__SWITCH: EventKind.SWITCH,
    BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__REPLACE: EventKind.SWITCH,
    BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__DRAG: EventKind.DRAG,
    BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__CANT: EventKind.CANT,
    BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__FAINT: EventKind.FAINT,
}
_KIND_TABLE = np.full(
    max(int(value) for value in BattlemajorargsEnum.values()) + 1,
    EventKind.RESIDUAL,
    dtype=np.int32,
)
for _major, _kind in _MAJOR_TO_KIND.items():
    _KIND_TABLE[_major] = _kind

_MOVE_ID_FEATURES = np.array(
    [
        EntityRevealedNodeFeature.Value(f"ENTITY_REVEALED_NODE_FEATURE__MOVEID{i}")
        for i in range(4)
    ]
)
_FIELD_CONDITION_FEATURES = np.arange(
    FieldFeature.FIELD_FEATURE__WEATHER_ID, FieldFeature.FIELD_FEATURE__OPP_SPIKES + 1
)
_REAL_MOVE_FLOOR = MovesEnum.MOVES_ENUM___SWITCH_IN


@chex.dataclass
class EventLabels:
    """(H,) unless noted; entry k describes event k+1 from the state after
    step k. `valid` marks pairs whose target exists; at the last real step
    the target is END with every other field at its sentinel."""

    kind: ArrayLike = ()
    actor: ArrayLike = ()
    actor_side: ArrayLike = ()
    move: ArrayLike = ()
    move_valid: ArrayLike = ()
    target: ArrayLike = ()
    touched: ArrayLike = ()  # (H, 12)
    field_touched: ArrayLike = ()
    new_turn: ArrayLike = ()
    boundary: ArrayLike = ()
    declared_kind: ArrayLike = ()
    declared_arg: ArrayLike = ()
    num_revealed: ArrayLike = ()
    move_previously_revealed: ArrayLike = ()
    valid: ArrayLike = ()
    terminal: ArrayLike = ()


@chex.dataclass
class StepEvents:
    """Per-step description of the event step k itself records."""

    valid: ArrayLike = ()
    kind: ArrayLike = ()
    actor: ArrayLike = ()
    actor_side: ArrayLike = ()
    move: ArrayLike = ()
    move_valid: ArrayLike = ()
    target: ArrayLike = ()
    touched: ArrayLike = ()
    field_touched: ArrayLike = ()
    new_turn: ArrayLike = ()
    turn: ArrayLike = ()
    request_count: ArrayLike = ()
    num_revealed: ArrayLike = ()
    move_previously_revealed: ArrayLike = ()
    last_row: ArrayLike = ()  # (H, 12) each slot's last packed row as of k, -1 never


def relevant_edges(history_field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Numpy twin of rl.model.history_encoder.relevant_edges."""
    relevant = history_field[:, RELEVANT_ENTITY_FEATURES]
    num_relevant = history_field[:, FieldFeature.FIELD_FEATURE__NUM_RELEVANT]
    edge_mask = np.arange(relevant.shape[1])[None] < num_relevant[:, None]
    return relevant, edge_mask


def step_events(
    history_field: np.ndarray,
    edge_cache: np.ndarray,
    public_cache: np.ndarray,
    revealed_cache: np.ndarray,
) -> StepEvents:
    num_steps = history_field.shape[0]
    num_rows = edge_cache.shape[0]
    valid = history_field[:, FieldFeature.FIELD_FEATURE__VALID] > 0
    relevant, edge_mask = relevant_edges(history_field)
    edge_mask = edge_mask & valid[:, None]
    rows = relevant.clip(0, num_rows - 1)
    major = edge_cache[rows, EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG]
    slots = edge_cache[rows, EntityEdgeFeature.ENTITY_EDGE_FEATURE__ENTITY_IDX].clip(
        0, NUM_PUBLIC_SLOTS - 1
    )
    source = (major > BattlemajorargsEnum.BATTLEMAJORARGS_ENUM___PAD) & edge_mask
    has_source = source.any(axis=1)
    first_source = source.argmax(axis=1)
    steps = np.arange(num_steps)
    source_row = rows[steps, first_source]
    kind = np.where(
        has_source, _KIND_TABLE[major[steps, first_source].clip(0)], EventKind.RESIDUAL
    )
    kind = np.where(valid, kind, EventKind.RESIDUAL).astype(np.int32)
    actor = np.where(has_source, slots[steps, first_source], NO_SLOT).astype(np.int32)
    actor_side = np.where(
        has_source,
        public_cache[
            source_row, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE
        ],
        -1,
    ).astype(np.int32)
    move = edge_cache[source_row, EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN]
    move_kind = (kind == EventKind.MOVE) | (kind == EventKind.CANT)
    move_valid = has_source & move_kind & (move >= _REAL_MOVE_FLOOR)
    move = np.where(move_valid, move, 0).astype(np.int32)

    affected = edge_mask & ~source
    has_target = affected.any(axis=1)
    target = np.where(
        has_target, slots[steps, affected.argmax(axis=1)], NO_SLOT
    ).astype(np.int32)

    touched = np.zeros((num_steps, NUM_PUBLIC_SLOTS), dtype=bool)
    step_index, row_index = np.nonzero(edge_mask)
    touched[step_index, slots[step_index, row_index]] = True

    conditions = history_field[:, _FIELD_CONDITION_FEATURES]
    previous = np.zeros_like(conditions)
    previous[1:] = conditions[:-1]
    field_touched = (conditions != previous).any(axis=1) & valid

    new_turn = (
        history_field[:, FieldFeature.FIELD_FEATURE__TURN_ORDER_VALUE] == 0
    ) & valid
    turn = history_field[:, FieldFeature.FIELD_FEATURE__TURN_VALUE]
    request_count = history_field[:, FieldFeature.FIELD_FEATURE__REQUEST_COUNT]

    highest = np.where(
        touched.any(axis=1), touched.shape[1] - 1 - touched[:, ::-1].argmax(axis=1), -1
    )
    num_revealed = (np.maximum.accumulate(highest) + 1).astype(np.int32)

    # Packed rows are appended in step order, so a running maximum is each
    # slot's latest row as of every step.
    last_row = np.full((num_steps, NUM_PUBLIC_SLOTS), -1, dtype=np.int32)
    np.maximum.at(
        last_row,
        (step_index, slots[step_index, row_index]),
        rows[step_index, row_index],
    )
    last_row = np.maximum.accumulate(last_row, axis=0)
    previous_row = np.full_like(last_row, -1)
    previous_row[1:] = last_row[:-1]
    actor_previous_row = previous_row[steps, actor.clip(0, NUM_PUBLIC_SLOTS - 1)]
    known_moves = revealed_cache[actor_previous_row.clip(0)][:, _MOVE_ID_FEATURES]
    move_previously_revealed = (
        move_valid
        & (actor_previous_row >= 0)
        & (known_moves == move[:, None]).any(axis=1)
    )

    return StepEvents(
        valid=valid,
        kind=kind,
        actor=actor,
        actor_side=actor_side,
        move=move,
        move_valid=move_valid,
        target=target,
        touched=touched & valid[:, None],
        field_touched=field_touched,
        new_turn=new_turn,
        turn=turn.astype(np.int32),
        request_count=request_count.astype(np.int32),
        num_revealed=num_revealed,
        move_previously_revealed=move_previously_revealed,
        last_row=last_row,
    )


def _own_declaration(events: StepEvents, step: int) -> tuple[int, int]:
    kind = events.kind[step]
    if kind == EventKind.SWITCH:
        return DeclaredKind.SWITCH, int(events.actor[step])
    if kind in (EventKind.MOVE, EventKind.CANT) and events.move_valid[step]:
        return DeclaredKind.MOVE, int(events.move[step])
    return DeclaredKind.UNKNOWN, 0


def declarations(events: StepEvents) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The own-side declaration in force at each step, and whether an
    own-side faint at the step is answered by an own switch before the
    next turn (the forced-switch decision). A turn's declaration is the
    own side's first move, switch or cant-with-a-move; a drag is never a
    declaration; nothing observable is UNKNOWN -- execution is not
    submission."""
    num_steps = events.valid.shape[0]
    declared_kind = np.zeros(num_steps, dtype=np.int32)
    declared_arg = np.zeros(num_steps, dtype=np.int32)
    forced_switch = np.zeros(num_steps, dtype=bool)
    valid_steps = np.flatnonzero(events.valid)
    if valid_steps.size == 0:
        return declared_kind, declared_arg, forced_switch
    starts = [int(step) for step in valid_steps if events.new_turn[step]]
    if not starts or starts[0] != valid_steps[0]:
        starts = [int(valid_steps[0])] + starts
    bounds = starts + [int(valid_steps[-1]) + 1]
    own = events.actor_side == SIDE_MINE
    for start, end in zip(bounds[:-1], bounds[1:]):
        kind, arg = DeclaredKind.UNKNOWN, 0
        for step in range(start, end):
            if own[step] and events.kind[step] in (
                EventKind.MOVE,
                EventKind.SWITCH,
                EventKind.CANT,
            ):
                kind, arg = _own_declaration(events, step)
                break
        declared_kind[start:end] = kind
        declared_arg[start:end] = arg
        for step in range(start, end):
            if own[step] and events.kind[step] == EventKind.FAINT:
                answer = next(
                    (
                        later
                        for later in range(step + 1, end)
                        if own[later] and events.kind[later] == EventKind.SWITCH
                    ),
                    None,
                )
                if answer is None:
                    declared_kind[step + 1 : end] = DeclaredKind.UNKNOWN
                    declared_arg[step + 1 : end] = 0
                else:
                    forced_switch[step] = True
                    declared_kind[step + 1 : end] = DeclaredKind.SWITCH
                    declared_arg[step + 1 : end] = events.actor[answer]
    return declared_kind, declared_arg, forced_switch


def event_labels(
    history_field: np.ndarray,
    edge_cache: np.ndarray,
    public_cache: np.ndarray,
    revealed_cache: np.ndarray,
) -> EventLabels:
    events = step_events(history_field, edge_cache, public_cache, revealed_cache)
    declared_kind, declared_arg, forced_switch = declarations(events)
    num_steps = events.valid.shape[0]
    valid_steps = np.flatnonzero(events.valid)
    last = -1
    if valid_steps.size > 0:
        last = int(valid_steps[-1])
    terminal = np.arange(num_steps) == last

    def shifted(values, sentinel):
        out = np.full_like(values, sentinel)
        out[:-1] = values[1:]
        return out

    has_next = shifted(events.valid, False)
    valid = events.valid & (has_next | terminal)
    kind = np.where(terminal, EventKind.END, shifted(events.kind, EventKind.RESIDUAL))
    return EventLabels(
        kind=np.where(valid, kind, EventKind.RESIDUAL).astype(np.int32),
        actor=shifted(events.actor, NO_SLOT),
        actor_side=shifted(events.actor_side, -1),
        move=shifted(events.move, 0),
        move_valid=shifted(events.move_valid, False) & valid,
        target=shifted(events.target, NO_SLOT),
        touched=shifted(events.touched, False) & valid[:, None],
        field_touched=shifted(events.field_touched, False) & valid,
        new_turn=shifted(events.new_turn, False) & valid,
        boundary=(shifted(events.new_turn, False) | forced_switch) & valid & ~terminal,
        declared_kind=shifted(declared_kind, DeclaredKind.UNKNOWN),
        declared_arg=shifted(declared_arg, 0),
        num_revealed=events.num_revealed,
        move_previously_revealed=shifted(events.move_previously_revealed, False)
        & valid,
        valid=valid,
        terminal=terminal & events.valid,
    )
