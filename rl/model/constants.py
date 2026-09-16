"""Model-side layout constants: the token vocabulary and the one sequence.

WHY NOT IN proto/enums.proto. That file is this project's source of
constants truth BETWEEN `service/` and `rl/`, and the discriminating test is
whether both sides read it. Token types fail that test: the service has
never emitted one, and nothing here crosses the wire. An IntEnum gives the
same derived count without a generated-but-unused TypeScript enum and
without the extra table row protolint's mandatory `___UNSPECIFIED` zero
value would add, never indexed.

The derived count is the point. Every `NUM_*` below is a `len()`, not a
literal -- a hand-edited literal is exactly the edit that silently leaves a
dead embedding row, or an out-of-range gather, when someone forgets.

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
    RESERVE_ENTITY_INDICES,
    TARGET_SLOT_INDICES,
    WILDCARD_MOVE_INDICES,
)
from rl.environment.protos.features_pb2 import FieldFeature

# Public slots the history encoder tracks: 6 per side, both sides.
NUM_PUBLIC_SLOTS = 12
# All EIGHT columns the service writes (state.ts maxRelevant = 8). Listing
# fewer silently drops the rows of any step touching more entities than are
# listed -- spread moves, hazard cascades -- before the scatter sees them.
RELEVANT_ENTITY_FEATURES = np.array(
    [
        FieldFeature.Value(f"FIELD_FEATURE__RELEVANT_ENTITY_IDX{index}")
        for index in range(8)
    ]
)
# Rows of my private sheet.
NUM_PRIVATE_SLOTS = 6
# The field triple, mirrored by the history field triple: global, mine, theirs.
NUM_FIELD_ROWS = 3
NUM_HISTORY_REGISTERS = 4
HISTORY_SLOT_STATE_ROWS = slice(0, NUM_PUBLIC_SLOTS)
HISTORY_FIELD_STATE_ROWS = slice(NUM_PUBLIC_SLOTS, NUM_PUBLIC_SLOTS + NUM_FIELD_ROWS)
HISTORY_REGISTER_STATE_ROWS = slice(
    HISTORY_FIELD_STATE_ROWS.stop, HISTORY_FIELD_STATE_ROWS.stop + NUM_HISTORY_REGISTERS
)
NUM_HISTORY_STATE_ROWS = HISTORY_REGISTER_STATE_ROWS.stop
HISTORY_STATE_GROUP_IDS = np.repeat(
    np.arange(3, dtype=np.int32),
    [NUM_PUBLIC_SLOTS, NUM_FIELD_ROWS, NUM_HISTORY_REGISTERS],
)


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

    Entity content is pooled to one vector per Pokémon; recurrent history
    registers provide additional global memory rows.

    What that trades away is on the record: with one token per mon, THEIR
    individual revealed moves no longer coexist with anything as separate
    rows, so a move-token x species-token comparison across two mons can only
    happen inside a pooled vector (LESSONS.md 13). MY sixteen candidate moves
    stay their own rows and the four entity-derived target rows are built
    from the opposing actives, so the matchup direction a decision actually
    turns on keeps both operands. If matchup reasoning proves to be the
    deficit the fix is explicit matchup rows, not re-unpacking attributes.
    """

    # Ordered by tier, which is also the layout order: the public tier (what
    # both players can see), then the private tier (my request truth), then
    # the learner-only partition -- so the actor's sequence is the layout's
    # policy-readable PREFIX and every head index means the same row on
    # either path.
    PUBLIC_ENTITY = 0
    TARGET_SLOT = 1
    FIELD = 2
    HISTORY_FIELD = 3
    INFO = 4
    HISTORY_ENTITY = 5
    HISTORY_REGISTER = 6
    PUBLIC_REGISTER = 7
    CLS = 8
    PRIVATE_ENTITY = 9
    MOVE_SLOT = 10
    PREV_ACTION = 11
    PRIVATE_REGISTER = 12
    # The learner-only partition (2026-09-01). OPP_PRIVATE_ENTITY rows carry
    # the opponent's request truth (their sheet latent), PRIVILEGED_REGISTER
    # the privileged critic's workspace, PUBLIC_CLS and VALUE_CLS the rows
    # the public and privileged value heads read. The three register groups
    # (2026-09-15) are learned workspace rows declared here so the read mask
    # governs them like every other row rather than the trunk deriving a
    # read set for rows it appends itself.
    OPP_PRIVATE_ENTITY = 13
    PRIVILEGED_REGISTER = 14
    PUBLIC_CLS = 15
    VALUE_CLS = 16


NUM_SEQUENCE_GROUPS = len(SequenceGroup)
assert max(SequenceGroup) == NUM_SEQUENCE_GROUPS - 1, "SequenceGroup must be contiguous"

# (group, row count), in sequence order. The single source of the layout: the
# offsets, the slices and the per-row group vector are all derived from it, so
# the arithmetic exists once rather than as a comment beside three literals.
NUM_TRUNK_REGISTERS_PER_TIER = 2

SEQUENCE_LAYOUT = (
    (SequenceGroup.PUBLIC_ENTITY, NUM_PUBLIC_SLOTS),
    (SequenceGroup.TARGET_SLOT, len(TARGET_SLOT_INDICES)),
    (SequenceGroup.FIELD, NUM_FIELD_ROWS),
    (SequenceGroup.HISTORY_FIELD, NUM_FIELD_ROWS),
    (SequenceGroup.INFO, 1),
    (SequenceGroup.HISTORY_ENTITY, NUM_PUBLIC_SLOTS),
    (SequenceGroup.HISTORY_REGISTER, NUM_HISTORY_REGISTERS),
    (SequenceGroup.PUBLIC_REGISTER, NUM_TRUNK_REGISTERS_PER_TIER),
    (SequenceGroup.CLS, 1),
    (SequenceGroup.PRIVATE_ENTITY, NUM_PRIVATE_SLOTS),
    (SequenceGroup.MOVE_SLOT, len(MOVE_INDICES)),
    (SequenceGroup.PREV_ACTION, 2),
    (SequenceGroup.PRIVATE_REGISTER, NUM_TRUNK_REGISTERS_PER_TIER),
    (SequenceGroup.OPP_PRIVATE_ENTITY, NUM_PRIVATE_SLOTS),
    (SequenceGroup.PRIVILEGED_REGISTER, NUM_TRUNK_REGISTERS_PER_TIER),
    (SequenceGroup.PUBLIC_CLS, 1),
    (SequenceGroup.VALUE_CLS, 1),
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
CLS_ROW = SEQUENCE_SLICES[SequenceGroup.CLS].start
PUBLIC_ROWS = SEQUENCE_SLICES[SequenceGroup.PUBLIC_ENTITY]
OPP_PUBLIC_ROWS = slice(PUBLIC_ROWS.start + NUM_PUBLIC_SLOTS // 2, PUBLIC_ROWS.stop)
PRIVATE_ROWS = SEQUENCE_SLICES[SequenceGroup.PRIVATE_ENTITY]
MOVE_ROWS = SEQUENCE_SLICES[SequenceGroup.MOVE_SLOT]
TARGET_ROWS = SEQUENCE_SLICES[SequenceGroup.TARGET_SLOT]
OPP_PRIVATE_ROWS = SEQUENCE_SLICES[SequenceGroup.OPP_PRIVATE_ENTITY]
HISTORY_ENTITY_ROWS = SEQUENCE_SLICES[SequenceGroup.HISTORY_ENTITY]
HISTORY_REGISTER_ROWS = SEQUENCE_SLICES[SequenceGroup.HISTORY_REGISTER]
VALUE_CLS_ROW = SEQUENCE_SLICES[SequenceGroup.VALUE_CLS].start
PUBLIC_CLS_ROW = SEQUENCE_SLICES[SequenceGroup.PUBLIC_CLS].start
PUBLIC_REGISTER_ROWS = SEQUENCE_SLICES[SequenceGroup.PUBLIC_REGISTER]
PRIVATE_REGISTER_ROWS = SEQUENCE_SLICES[SequenceGroup.PRIVATE_REGISTER]
PRIVILEGED_REGISTER_ROWS = SEQUENCE_SLICES[SequenceGroup.PRIVILEGED_REGISTER]
FIELD_ROWS = SEQUENCE_SLICES[SequenceGroup.FIELD]

assert (
    NUM_SEQUENCE_ROWS == 81 + NUM_HISTORY_REGISTERS + 3 * NUM_TRUNK_REGISTERS_PER_TIER
), NUM_SEQUENCE_ROWS
assert len(SEQUENCE_GROUP_IDS) == NUM_SEQUENCE_ROWS
assert MOVE_ROWS.stop - MOVE_ROWS.start == len(MOVE_INDICES)
assert TARGET_ROWS.stop - TARGET_ROWS.start == len(TARGET_SLOT_INDICES)
assert PRIVATE_ROWS.stop - PRIVATE_ROWS.start == len(RESERVE_ENTITY_INDICES)

# ---- the leak partition (2026-09-01; public tier 2026-09-15) --------------
# R[q, k]: query row q may attend to key row k. Four nested sets, each
# reading itself and everything below it:
#   PUBLIC -- the rows both players can see (the public entity views, the
#     targets built from them, the field, the request info, the history):
#     reads only itself, so its content is a function of public state alone
#     -- the common-knowledge representation a human replay also contains.
#   PRIVATE -- my request truth (CLS, my sheet, my move slots, and the
#     PREV_ACTION rows the doubles actor's second slot reads for the first
#     slot's choice this turn): reads PUBLIC and itself. PUBLIC | PRIVATE is the policy's
#     information set, the POLICY_READABLE partition.
#   SECRET (OPP_PRIVATE_ROWS, PRIVILEGED_REGISTER_ROWS) -- the opponent's
#     request truth and the privileged workspace: readable ONLY by
#     VALUE_CLS; may itself read the policy-readable rows and its siblings,
#     because a row's READS leak nothing.
#   Each tier carries two learned register rows (PUBLIC_REGISTER,
#     PRIVATE_REGISTER, PRIVILEGED_REGISTER): workspace with no input of
#     its own, governed by its tier's rule and nothing else.
#   VALUE_CLS -- reads everything but PUBLIC_CLS, read by NOTHING
#     (out-degree 0). Reading the policy-readable rows (history included)
#     AS WELL AS the secret partition is what makes the privileged V the
#     (history, state)-conditioned asymmetric critic -- unbiased for the
#     policy's returns (Baisero & Amato 2022); a state-only critic is the
#     biased form.
#   PUBLIC_CLS -- reads the PUBLIC tier and itself, read by NOTHING: the
#     public critic's row, a value of the common-knowledge state that a
#     human replay could also label.
# Leak-freedom is transitive by induction over blocks: a row's content after
# block b is a function of its in-edges' contents at block b-1 (plus its own
# residual), and a row's in-edges never rise above its own tier at any
# block, so no higher-tier content can enter a tier at any depth; and
# VALUE_CLS, with no out-edge, aggregates without re-broadcasting. The trunk
# ANDs this matrix into its validity mask every block and adds no rows of
# its own.
PUBLIC_TIER_GROUPS = frozenset(
    {
        SequenceGroup.PUBLIC_ENTITY,
        SequenceGroup.TARGET_SLOT,
        SequenceGroup.FIELD,
        SequenceGroup.HISTORY_FIELD,
        SequenceGroup.INFO,
        SequenceGroup.HISTORY_ENTITY,
        SequenceGroup.HISTORY_REGISTER,
        SequenceGroup.PUBLIC_REGISTER,
    }
)
_is_public = np.isin(SEQUENCE_GROUP_IDS, [int(group) for group in PUBLIC_TIER_GROUPS])
_is_secret = np.zeros(NUM_SEQUENCE_ROWS, dtype=bool)
_is_secret[OPP_PRIVATE_ROWS] = True
_is_secret[PRIVILEGED_REGISTER_ROWS] = True
_is_value_cls = np.zeros(NUM_SEQUENCE_ROWS, dtype=bool)
_is_value_cls[VALUE_CLS_ROW] = True
_is_public_cls = np.zeros(NUM_SEQUENCE_ROWS, dtype=bool)
_is_public_cls[PUBLIC_CLS_ROW] = True
_policy_readable = ~(_is_secret | _is_value_cls | _is_public_cls)
_is_private = _policy_readable & ~_is_public
SEQUENCE_READ_MASK = np.zeros((NUM_SEQUENCE_ROWS, NUM_SEQUENCE_ROWS), dtype=bool)
SEQUENCE_READ_MASK[np.ix_(_is_public, _is_public)] = True
SEQUENCE_READ_MASK[np.ix_(_is_private, _policy_readable)] = True
SEQUENCE_READ_MASK[np.ix_(_is_secret, _policy_readable | _is_secret)] = True
SEQUENCE_READ_MASK[np.ix_(_is_public_cls, _is_public | _is_public_cls)] = True
SEQUENCE_READ_MASK[np.ix_(_is_value_cls, ~_is_public_cls)] = True
assert not SEQUENCE_READ_MASK[
    np.ix_(_is_public, ~_is_public)
].any(), "leak: a public row may attend outside the public tier"
assert not SEQUENCE_READ_MASK[
    np.ix_(_policy_readable, ~_policy_readable)
].any(), "leak: a policy-readable row may attend to the learner-only partition"
assert not SEQUENCE_READ_MASK[:, _is_value_cls][
    ~_is_value_cls
].any(), "leak: VALUE_CLS must have out-degree 0"
assert not SEQUENCE_READ_MASK[:, _is_public_cls][
    ~_is_public_cls
].any(), "leak: PUBLIC_CLS must have out-degree 0"
PUBLIC_TIER_ROWS = np.flatnonzero(_is_public)
PRIVATE_TIER_ROWS = np.flatnonzero(_is_private)

# The ACTOR's sequence (2026-09-04; tier-ordered layout 2026-09-15): the
# policy-readable rows alone. At act time the learner-only partition is
# all-zero input that no policy output reads -- the read mask gives the
# policy-readable rows no in-edge from it at any block -- yet its rows
# would still cost the private embedder and ten rows of every trunk block.
# Under cfg.train=False the encoder assembles only the policy-readable
# rows and the trunk runs on them with the mask's leading sub-block, which
# computes the SAME rows the learner computes, up to GEMM shape numerics.
# Because the layout lists the learner-only partition LAST, those rows are
# the prefix arange(NUM_POLICY_READABLE_ROWS): every absolute row index a
# head carries names the same row in either sequence -- asserted, not
# assumed.
LEARNER_ONLY_GROUPS = frozenset(
    {
        SequenceGroup.OPP_PRIVATE_ENTITY,
        SequenceGroup.PRIVILEGED_REGISTER,
        SequenceGroup.PUBLIC_CLS,
        SequenceGroup.VALUE_CLS,
    }
)
POLICY_READABLE_ROWS = np.flatnonzero(_policy_readable)
# The public-only sequence: the public tier (a layout prefix, asserted) plus
# PUBLIC_CLS -- the rows the event world model and the public critic read.
# Closed under the read mask, so the trunk on these rows alone reproduces
# the learner's public rows.
PUBLIC_SEQUENCE_ROWS = np.concatenate([PUBLIC_TIER_ROWS, [PUBLIC_CLS_ROW]])
NUM_PUBLIC_SEQUENCE_ROWS = len(PUBLIC_SEQUENCE_ROWS)
PUBLIC_CLS_LOCAL_ROW = len(PUBLIC_TIER_ROWS)
assert (PUBLIC_TIER_ROWS == np.arange(len(PUBLIC_TIER_ROWS))).all()
assert not SEQUENCE_READ_MASK[
    np.ix_(
        PUBLIC_SEQUENCE_ROWS,
        np.setdiff1d(np.arange(NUM_SEQUENCE_ROWS), PUBLIC_SEQUENCE_ROWS),
    )
].any()
NUM_POLICY_READABLE_ROWS = len(POLICY_READABLE_ROWS)
assert (
    POLICY_READABLE_ROWS
    == np.flatnonzero(
        ~np.isin(SEQUENCE_GROUP_IDS, [int(group) for group in LEARNER_ONLY_GROUPS])
    )
).all(), "the leak partition and the actor's dropped groups disagree"
assert (
    POLICY_READABLE_ROWS == np.arange(NUM_POLICY_READABLE_ROWS)
).all(), "the learner-only partition must be the layout's suffix"
assert SEQUENCE_READ_MASK[np.ix_(PRIVATE_TIER_ROWS, POLICY_READABLE_ROWS)].all()
assert SEQUENCE_READ_MASK[np.ix_(POLICY_READABLE_ROWS, PUBLIC_TIER_ROWS)].all()

# Public rows 0-5 are mine and 6-11 theirs, actives first, so my active i is
# public row i and theirs is row NUM_PUBLIC_SLOTS // 2 + i. The four
# entity-derived target slots (ALLY_i_TARGET, ENEMY_i_TARGET) read those rows,
# which is what lets a move score against the actual mon it would hit.
NUM_ACTIVES_PER_SIDE = 2
MY_ACTIVE_PUBLIC_ROWS = np.arange(NUM_ACTIVES_PER_SIDE)
OPP_ACTIVE_PUBLIC_ROWS = NUM_PUBLIC_SLOTS // 2 + np.arange(NUM_ACTIVES_PER_SIDE)

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
# produced its logit: a switch cell its private row and the ALLY_1_TARGET
# row of the active it replaces (the switch pair, 2026-09-11; a doubles
# stage-2 switch reads ALLY_2 -- the known-open doubles workstream),
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
        np.full(NUM_PRIVATE_SLOTS, _BANK_TARGET_OFFSET + ALLY_TARGET_ROWS[0]),
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
