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
    """Rows of the entity token-type bias table.

    These are the ATTRIBUTES of one entity, consumed inside `EntitySumPool`,
    which adds each token its field identity before the masked sum -- without
    it the sum could not tell an item's embedding from an ability's. They are
    not rows of the trunk's sequence -- that is `SequenceGroup`.

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
    # The learner-only partition (2026-09-01). OPP_PRIVATE_ENTITY rows carry
    # the opponent's request truth (their discrete code embedding); VALUE_CLS
    # is the one row the privileged value head reads. Both sit at the END of
    # the layout so every pre-existing offset -- and the adjacency the flat
    # readout pins -- survives unchanged.
    OPP_PRIVATE_ENTITY = 9
    VALUE_CLS = 10
    # History as its own rows (2026-09-01): entity i's GRU diary + latest
    # raw snapshot, freed from being an additive attribute on the public
    # row so attention routes board-now vs memory instead of one vector
    # carrying their sum. Policy-readable.
    HISTORY_ENTITY = 11


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
    (SequenceGroup.OPP_PRIVATE_ENTITY, NUM_PRIVATE_SLOTS),
    (SequenceGroup.VALUE_CLS, 1),
    (SequenceGroup.HISTORY_ENTITY, NUM_PUBLIC_SLOTS),
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
OPP_PRIVATE_ROWS = SEQUENCE_SLICES[SequenceGroup.OPP_PRIVATE_ENTITY]
HISTORY_ENTITY_ROWS = SEQUENCE_SLICES[SequenceGroup.HISTORY_ENTITY]
VALUE_CLS_ROW = SEQUENCE_SLICES[SequenceGroup.VALUE_CLS].start
FIELD_ROWS = SEQUENCE_SLICES[SequenceGroup.FIELD]

# The dynamics head's rows (2026-09-03): the entity rows whose NEXT-step
# pre-trunk content is a prediction target -- public 12, my private 6, the
# field triple, in that order. One index array so the encoder's target slice
# and the head's input slice cannot disagree; the per-group panels split it
# by these offsets.
DYNAMICS_TARGET_ROWS = np.concatenate(
    [
        np.arange(PUBLIC_ROWS.start, PUBLIC_ROWS.stop),
        np.arange(PRIVATE_ROWS.start, PRIVATE_ROWS.stop),
        np.arange(FIELD_ROWS.start, FIELD_ROWS.stop),
    ]
).astype(np.int32)
DYNAMICS_GROUP_SLICES = {
    "public": slice(0, NUM_PUBLIC_SLOTS),
    "private": slice(NUM_PUBLIC_SLOTS, NUM_PUBLIC_SLOTS + NUM_PRIVATE_SLOTS),
    "field": slice(NUM_PUBLIC_SLOTS + NUM_PRIVATE_SLOTS, len(DYNAMICS_TARGET_ROWS)),
}
NUM_DYNAMICS_ROWS = len(DYNAMICS_TARGET_ROWS)

assert NUM_SEQUENCE_ROWS == 80, NUM_SEQUENCE_ROWS
assert len(SEQUENCE_GROUP_IDS) == NUM_SEQUENCE_ROWS
assert MOVE_ROWS.stop - MOVE_ROWS.start == len(MOVE_INDICES)
assert TARGET_ROWS.stop - TARGET_ROWS.start == len(TARGET_SLOT_INDICES)
assert PRIVATE_ROWS.stop - PRIVATE_ROWS.start == len(RESERVE_ENTITY_INDICES)

# ---- the leak partition (2026-09-01) ---------------------------------------
# R[q, k]: query row q may attend to key row k. Three sets:
#   POLICY_READABLE -- every pre-existing row (0..60): reads only itself.
#   SECRET (OPP_PRIVATE_ROWS) -- the opponent's request truth: readable ONLY
#     by VALUE_CLS; may itself read the policy-readable rows and its
#     siblings, because a row's READS leak nothing.
#   VALUE_CLS -- reads everything, read by NOTHING (out-degree 0). Reading
#     the policy-readable rows (history included) AS WELL AS the secret
#     partition is what makes the privileged V the (history, state)-
#     conditioned asymmetric critic -- unbiased for the policy's returns
#     (Baisero & Amato 2022); a state-only critic is the biased form.
# Leak-freedom is transitive by induction over blocks: a row's content after
# block b is a function of its in-edges' contents at block b-1 (plus its own
# residual), and a policy-readable row's in-edges are policy-readable at
# every block, so no secret content can enter the set at any depth; and
# VALUE_CLS, with no out-edge, aggregates without re-broadcasting. The trunk
# ANDs this matrix into its validity mask every block.
_is_secret = np.zeros(NUM_SEQUENCE_ROWS, dtype=bool)
_is_secret[OPP_PRIVATE_ROWS] = True
_is_value_cls = np.zeros(NUM_SEQUENCE_ROWS, dtype=bool)
_is_value_cls[VALUE_CLS_ROW] = True
_policy_readable = ~(_is_secret | _is_value_cls)
SEQUENCE_READ_MASK = np.zeros((NUM_SEQUENCE_ROWS, NUM_SEQUENCE_ROWS), dtype=bool)
SEQUENCE_READ_MASK[np.ix_(_policy_readable, _policy_readable)] = True
SEQUENCE_READ_MASK[np.ix_(_is_secret, _policy_readable | _is_secret)] = True
SEQUENCE_READ_MASK[_is_value_cls, :] = True
assert not SEQUENCE_READ_MASK[
    np.ix_(_policy_readable, ~_policy_readable)
].any(), "leak: a policy-readable row may attend to the learner-only partition"
assert not SEQUENCE_READ_MASK[:, _is_value_cls][
    ~_is_value_cls
].any(), "leak: VALUE_CLS must have out-degree 0"

# The ACTOR's sequence (2026-09-04): the policy-readable rows alone. At act
# time the learner-only partition is all-zero input that no policy output
# reads -- the read mask gives the policy-readable rows no in-edge from it
# at any block -- yet its rows still cost the private embedder, the code
# softmax and seven rows of every trunk block. Under cfg.train=False the
# encoder assembles only these rows and the trunk runs on them with the
# partition's sub-mask (all True by construction), which computes the SAME
# policy-readable rows the learner computes, up to GEMM shape numerics.
# Every head reads rows BELOW the first dropped one, so a head's absolute
# index means the same row in either sequence -- asserted, not assumed.
LEARNER_ONLY_GROUPS = frozenset(
    {SequenceGroup.OPP_PRIVATE_ENTITY, SequenceGroup.VALUE_CLS}
)
POLICY_READABLE_ROWS = np.flatnonzero(_policy_readable)
NUM_POLICY_READABLE_ROWS = len(POLICY_READABLE_ROWS)
assert (
    POLICY_READABLE_ROWS
    == np.flatnonzero(
        ~np.isin(SEQUENCE_GROUP_IDS, [int(group) for group in LEARNER_ONLY_GROUPS])
    )
).all(), "the leak partition and the actor's dropped groups disagree"
assert SEQUENCE_READ_MASK[np.ix_(POLICY_READABLE_ROWS, POLICY_READABLE_ROWS)].all()
_first_dropped_row = min(OPP_PRIVATE_ROWS.start, VALUE_CLS_ROW)
assert (
    POLICY_READABLE_ROWS[:_first_dropped_row] == np.arange(_first_dropped_row)
).all()
for _head_row in (
    CLS_ROW,
    PUBLIC_ROWS.stop - 1,
    PRIVATE_ROWS.stop - 1,
    MOVE_ROWS.stop - 1,
    TARGET_ROWS.stop - 1,
    int(DYNAMICS_TARGET_ROWS.max()),
):
    assert _head_row < _first_dropped_row, "a head reads past the actor's prefix"

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

# Which block cells are a wildcard (tera / mega / Z) move: the move-block
# cells whose source slot is a wildcard shadow. What an offline intervention
# masks to hold tera back.
_cell_move_slot = CELL_BANK_SRC - _BANK_MOVE_OFFSET
_cell_is_move = (_cell_move_slot >= 0) & (_cell_move_slot < len(MOVE_INDICES))
IS_WILDCARD_CELL = np.zeros(NUM_ACTION_CELLS, dtype=bool)
IS_WILDCARD_CELL[_cell_is_move] = IS_WILDCARD_MOVE_SLOT[_cell_move_slot[_cell_is_move]]
assert IS_WILDCARD_CELL.sum() == len(WILDCARD_MOVE_INDICES) * len(TARGET_SLOT_INDICES)
