"""Model-side layout constants: the token vocabulary and the one sequence.

WHY NOT IN proto/enums.proto. That file is this project's source of
constants truth BETWEEN `service/` and `rl/`, and the discriminating test is
whether both sides read it. Token types fail that test: the service has
never emitted one, and nothing here crosses the wire. Routing them through
the proto was tried on 2026-08-25 and reverted, because it bought nothing
an IntEnum does not (the count is derived either way) while costing a
generated-but-unused TypeScript enum and, worse, an extra table row --
protolint mandates a `___UNSPECIFIED` zero value, which would have taken
the token-type table from 12 rows to 13 with row 0 never indexed.

The derived count is the point. Every `NUM_*` below is a `len()`, not a
literal: on 2026-08-25 a token type was deleted and the literal `13` had to
be hand-edited to `12` -- exactly the edit that silently leaves a dead
embedding row, or an out-of-range gather, when someone forgets.

Environment-side layout -- feature counts, action-slot partitions, modality
masks -- lives in `rl/environment/data.py` and is imported here; this module
adds only what the model layers on top.
"""

from enum import IntEnum

import numpy as np

from rl.environment.data import (
    ALLY_TARGET_INDICES,
    ENEMY_TARGET_INDICES,
    MOVE_INDICES,
    NUM_ACTION_CELLS,
    NUM_ACTION_FEATURES,
    RESERVE_ENTITY_INDICES,
    TARGET_SLOT_INDICES,
    WILDCARD_MOVE_INDICES,
)

# Public slots the history encoder tracks: 6 per side, both sides.
NUM_PUBLIC_SLOTS = 12
# Rows of my private sheet.
NUM_PRIVATE_SLOTS = 6
# The field triple, mirrored by the history field triple: global, mine, theirs.
NUM_FIELD_ROWS = 3


class TokenType(IntEnum):
    """Rows of the intra-entity token-type bias table.

    These are the ATTRIBUTES of one entity, consumed inside
    `EntityAttentionPool`, which gives the otherwise permutation-invariant
    intra-entity attention a field identity per token. They are not rows of
    the trunk's sequence -- that is `SequenceGroup`.

    The four moves share one type: movesets are unordered. Public entities
    carry TWO state tokens: a persistent one (hp, status, level, ...; it
    survives switching) and an active-only one (volatiles, boosts,
    typechange, trapped, ...) masked out for benched entities, so "not
    applicable" is an ABSENT token rather than a default-valued vector.

    No UNSPECIFIED/PAD/UNK sentinels: these are table rows, not a vocabulary
    with an "unknown" case, so a reserved id is a never-indexed embedding
    row. Ids are load-bearing only as rows, so they may be renumbered -- but
    a checkpoint's table is indexed by them, so renumbering invalidates one.
    """

    SPECIES = 0
    ABILITY = 1
    ITEM = 2
    MOVE = 3
    LEARNSET = 4
    PUBLIC_STATE = 5
    ACTIVE_STATE = 6
    PRIVATE_STATE = 7


NUM_TOKEN_TYPES = len(TokenType)
assert max(TokenType) == NUM_TOKEN_TYPES - 1, "TokenType ids must be contiguous from 0"

# The attribute vector each entity pool is handed, in the order its tokens
# appear.
PUBLIC_TOKEN_TYPES = np.array(
    [TokenType.SPECIES, TokenType.ABILITY, TokenType.ITEM]
    + 4 * [TokenType.MOVE]
    + [TokenType.LEARNSET, TokenType.PUBLIC_STATE, TokenType.ACTIVE_STATE],
    dtype=np.int32,
)
PRIVATE_TOKEN_TYPES = np.array(
    [TokenType.SPECIES, TokenType.ABILITY, TokenType.ITEM]
    + 4 * [TokenType.MOVE]
    + [TokenType.PRIVATE_STATE],
    dtype=np.int32,
)


class SequenceGroup(IntEnum):
    """Rows of the trunk's one sequence, one group per kind of thing.

    ONE TOKEN PER THING (2026-08-29). Before this the board was unpacked into
    189 attribute tokens -- 10 or 11 per entity -- and a Perceiver read
    compressed them to 48 latents for a trunk that could not afford the rows.
    With entities pooled to a vector each the whole board is 61 rows, the
    trunk carries them directly, and the read, the latents and the separate
    action stream all go.

    What that trades away is on the record: with one token per mon, THEIR
    individual revealed moves no longer coexist with anything as separate
    rows, so a move-token x species-token comparison across two mons can only
    happen inside a pooled vector (CLAUDE.md 13). MY sixteen candidate moves
    stay their own rows and the four entity-derived target rows are built
    from the opposing actives, so the matchup direction a decision actually
    turns on keeps both operands. If matchup reasoning proves to be the
    deficit the fix is explicit matchup rows, not re-unpacking attributes.
    """

    CLS = 0
    PUBLIC_ENTITY = 1
    PRIVATE_ENTITY = 2
    MOVE_SLOT = 3
    TARGET_SLOT = 4
    FIELD = 5
    HISTORY_FIELD = 6
    PREV_ACTION = 7
    INFO = 8


NUM_SEQUENCE_GROUPS = len(SequenceGroup)
assert max(SequenceGroup) == NUM_SEQUENCE_GROUPS - 1, "SequenceGroup must be contiguous"

# (group, row count), in sequence order. The single source of the layout: the
# offsets, the slices and the per-row group vector are all derived from it, so
# the arithmetic exists once rather than as a comment beside three literals.
SEQUENCE_LAYOUT = (
    (SequenceGroup.CLS, 1),
    (SequenceGroup.PUBLIC_ENTITY, NUM_PUBLIC_SLOTS),
    (SequenceGroup.PRIVATE_ENTITY, NUM_PRIVATE_SLOTS),
    (SequenceGroup.MOVE_SLOT, len(MOVE_INDICES)),
    (SequenceGroup.TARGET_SLOT, len(TARGET_SLOT_INDICES)),
    (SequenceGroup.FIELD, NUM_FIELD_ROWS),
    (SequenceGroup.HISTORY_FIELD, NUM_FIELD_ROWS),
    (SequenceGroup.PREV_ACTION, 2),
    (SequenceGroup.INFO, 1),
)

_offsets = np.cumsum([0] + [rows for _, rows in SEQUENCE_LAYOUT])
NUM_SEQUENCE_ROWS = int(_offsets[-1])

SEQUENCE_SLICES = {
    group: slice(int(_offsets[index]), int(_offsets[index + 1]))
    for index, (group, _) in enumerate(SEQUENCE_LAYOUT)
}
# Per-row group id, for the additive group bias.
SEQUENCE_GROUP_IDS = np.concatenate(
    [np.full(rows, int(group), dtype=np.int32) for group, rows in SEQUENCE_LAYOUT]
)

# The rows each head reads. Named so a head never carries an offset literal.
CLS_ROW = int(_offsets[0])
PUBLIC_ROWS = SEQUENCE_SLICES[SequenceGroup.PUBLIC_ENTITY]
PRIVATE_ROWS = SEQUENCE_SLICES[SequenceGroup.PRIVATE_ENTITY]
MOVE_ROWS = SEQUENCE_SLICES[SequenceGroup.MOVE_SLOT]
TARGET_ROWS = SEQUENCE_SLICES[SequenceGroup.TARGET_SLOT]

assert NUM_SEQUENCE_ROWS == 61, NUM_SEQUENCE_ROWS
assert len(SEQUENCE_GROUP_IDS) == NUM_SEQUENCE_ROWS
assert MOVE_ROWS.stop - MOVE_ROWS.start == len(MOVE_INDICES)
assert TARGET_ROWS.stop - TARGET_ROWS.start == len(TARGET_SLOT_INDICES)
assert PRIVATE_ROWS.stop - PRIVATE_ROWS.start == len(RESERVE_ENTITY_INDICES)

# Public rows 0-5 are mine and 6-11 theirs, actives first, so my active i is
# public row i and theirs is row NUM_PUBLIC_SLOTS // 2 + i. The four
# entity-derived target slots (ALLY_i_TARGET, ENEMY_i_TARGET) read those rows,
# which is what lets a move score against the actual mon it would hit.
NUM_ACTIVES_PER_SIDE = 2
MY_ACTIVE_PUBLIC_ROWS = np.arange(NUM_ACTIVES_PER_SIDE)
OPP_ACTIVE_PUBLIC_ROWS = NUM_PUBLIC_SLOTS // 2 + np.arange(NUM_ACTIVES_PER_SIDE)

assert (
    NUM_ACTION_FEATURES
    == len(MOVE_INDICES) + len(TARGET_SLOT_INDICES) + len(RESERVE_ENTITY_INDICES) + 2
), "the action slots must partition into move / target / reserve / ally-switch"

# Which of the 16 move rows are the wildcard (tera / mega / Z-move) shadow of
# a regular slot. `my_moveset` row k IS action slot MOVE_INDICES[k], so this
# indexes the move rows directly.
IS_WILDCARD_MOVE_SLOT = np.isin(MOVE_INDICES, WILDCARD_MOVE_INDICES)
assert IS_WILDCARD_MOVE_SLOT.sum() == len(WILDCARD_MOVE_INDICES)

# Where inside the 17 target rows the four entity-derived targets sit. These
# rows add the entity they name, which is what lets a move score against the
# actual pokemon it would hit rather than a bare positional slot.
_target_row_of = {int(slot): row for row, slot in enumerate(TARGET_SLOT_INDICES)}
ALLY_TARGET_ROWS = np.array(
    [_target_row_of[int(slot)] for slot in ALLY_TARGET_INDICES], dtype=np.int32
)
ENEMY_TARGET_ROWS = np.array(
    [_target_row_of[int(slot)] for slot in ENEMY_TARGET_INDICES], dtype=np.int32
)
assert len(ALLY_TARGET_ROWS) == len(ENEMY_TARGET_ROWS) == NUM_ACTIVES_PER_SIDE

# Block cell -> row-bank index, for the doubles SlotConditioning gather. The
# bank is the readout's own input rows stacked in order --
# private(6) | move(16) | target(17) -- and each cell names the row(s) that
# produced its logit: a switch cell its private row (both halves -- the
# ALLY_i_SWITCH pseudo-slots of the grid era had no row and gathered zeros),
# a move cell its move row and its target row, a standalone cell its target
# row twice.
_BANK_MOVE_OFFSET = NUM_PRIVATE_SLOTS
_BANK_TARGET_OFFSET = NUM_PRIVATE_SLOTS + len(MOVE_INDICES)
CELL_BANK_SRC = np.concatenate(
    [
        np.arange(NUM_PRIVATE_SLOTS),
        np.repeat(
            _BANK_MOVE_OFFSET + np.arange(len(MOVE_INDICES)),
            len(TARGET_SLOT_INDICES),
        ),
        _BANK_TARGET_OFFSET + np.arange(len(TARGET_SLOT_INDICES)),
    ]
).astype(np.int32)
CELL_BANK_TGT = np.concatenate(
    [
        np.arange(NUM_PRIVATE_SLOTS),
        np.tile(
            _BANK_TARGET_OFFSET + np.arange(len(TARGET_SLOT_INDICES)),
            len(MOVE_INDICES),
        ),
        _BANK_TARGET_OFFSET + np.arange(len(TARGET_SLOT_INDICES)),
    ]
).astype(np.int32)
assert len(CELL_BANK_SRC) == len(CELL_BANK_TGT) == NUM_ACTION_CELLS
