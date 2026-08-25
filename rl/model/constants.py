"""Model-side layout constants: token types and the action-slot grouping.

WHY NOT IN proto/enums.proto. That file is this project's source of
constants truth BETWEEN `service/` and `rl/` — the discriminating test is
whether both sides read it. Token types fail that test: the service has
never emitted one, and nothing here crosses the wire. Routing them through
the proto was tried on 2026-08-25 and reverted, because it bought nothing
an IntEnum does not (the count is derived either way) while costing a
generated-but-unused TypeScript enum and, worse, an extra table row —
protolint mandates a `___UNSPECIFIED` zero value, which would have taken
the token-type table from 12 rows to 13 with row 0 never indexed. Anything
BOTH languages need still belongs in the proto.

The derived count is the point. `NUM_TOKEN_TYPES` is `len(TokenType)`, not
a literal: on 2026-08-25 a token type was deleted and the literal `13` had
to be hand-edited to `12` — exactly the edit that silently leaves a dead
embedding row, or an out-of-range gather, when someone forgets. An IntEnum
cannot drift from its own length.

Environment-side layout — feature counts, action-slot partitions, modality
masks — lives in `rl/environment/data.py` and is imported here; this module
adds only what the model layers on top.
"""

from enum import IntEnum

import numpy as np

from rl.environment.data import (
    MOVE_SLOT_INDICES,
    SWITCH_SLOT_INDICES,
    TARGET_SLOT_INDICES,
)

# Public slots the history encoder tracks: 6 per side, both sides.
NUM_PUBLIC_SLOTS = 12


class TokenType(IntEnum):
    """Rows of the shared token-type bias table.

    The table gives the (otherwise permutation-invariant) intra-entity
    attention a field identity per token, and types the flat input set the
    latent read consumes.

    ENTITY ATTRIBUTE TOKENS. The four moves share one type — movesets are
    unordered. Public entities carry TWO state tokens: a persistent one
    (hp, status, level, ...; survives switching) and an active-only one
    (volatiles, boosts, typechange, trapped, ...) masked out for benched
    entities, so "not applicable" is an ABSENT token rather than a
    default-valued vector.

    NON-ENTITY TOKENS (2026-08-21): field / side conditions, the two
    prev-action slots, the per-slot recurrent history states and the field
    history state.

    No UNSPECIFIED/PAD/UNK sentinels: these are table rows, not a
    vocabulary with an "unknown" case, so a reserved id is a never-indexed
    embedding row. Ids are load-bearing only as rows, so they may be
    renumbered — but a checkpoint's table is indexed by them, so
    renumbering invalidates one. (A LATENT type existed until 2026-08-25;
    it typed the public latents when the deleted privileged read re-read
    them.)
    """

    SPECIES = 0
    ABILITY = 1
    ITEM = 2
    MOVE = 3
    LEARNSET = 4
    PUBLIC_STATE = 5
    ACTIVE_STATE = 6
    PRIVATE_STATE = 7
    FIELD = 8
    PREV_ACTION = 9
    HISTORY_SLOT = 10
    HISTORY_FIELD = 11


NUM_TOKEN_TYPES = len(TokenType)
assert max(TokenType) == NUM_TOKEN_TYPES - 1, "TokenType ids must be contiguous from 0"

_SPECIES = TokenType.SPECIES
_ABILITY = TokenType.ABILITY
_ITEM = TokenType.ITEM
_MOVE = TokenType.MOVE
_LEARNSET = TokenType.LEARNSET
_PUBLIC_STATE = TokenType.PUBLIC_STATE
_ACTIVE_STATE = TokenType.ACTIVE_STATE
_PRIVATE_STATE = TokenType.PRIVATE_STATE
_FIELD = TokenType.FIELD
_PREV_ACTION = TokenType.PREV_ACTION
_HISTORY_SLOT = TokenType.HISTORY_SLOT
_HISTORY_FIELD = TokenType.HISTORY_FIELD

# The type vector for each token group the latent read is handed, in the
# order that group's tokens appear.
PUBLIC_TOKEN_TYPES = np.array(
    [_SPECIES, _ABILITY, _ITEM]
    + 4 * [_MOVE]
    + [_LEARNSET, _PUBLIC_STATE, _ACTIVE_STATE],
    dtype=np.int32,
)
PRIVATE_TOKEN_TYPES = np.array(
    [_SPECIES, _ABILITY, _ITEM] + 4 * [_MOVE] + [_PRIVATE_STATE], dtype=np.int32
)
FIELD_TOKEN_TYPES = np.array(3 * [_FIELD], dtype=np.int32)
PREV_ACTION_TOKEN_TYPES = np.array(2 * [_PREV_ACTION], dtype=np.int32)
HISTORY_TOKEN_TYPES = np.array(
    NUM_PUBLIC_SLOTS * [_HISTORY_SLOT] + [_HISTORY_FIELD], dtype=np.int32
)

# The flat input token count the latent read cross-attends, ASSERTED rather
# than commented: 12 public x 10 + 6 private x 8 + field 3 + prev-action 2
# + history 13 = 186. A group changing width silently changes the read's
# key count, and the arithmetic was previously only a comment in config.py.
NUM_INPUT_TOKENS = (
    NUM_PUBLIC_SLOTS * len(PUBLIC_TOKEN_TYPES)
    + 6 * len(PRIVATE_TOKEN_TYPES)
    + len(FIELD_TOKEN_TYPES)
    + len(PREV_ACTION_TOKEN_TYPES)
    + len(HISTORY_TOKEN_TYPES)
)
assert NUM_INPUT_TOKENS == 186, NUM_INPUT_TOKENS

_MOVE_SLOTS = np.asarray(MOVE_SLOT_INDICES)
_SWITCH_SLOTS = np.asarray(SWITCH_SLOT_INDICES)
_TARGET_STATIC_SLOTS = np.asarray(TARGET_SLOT_INDICES)

# (name, static slot indices) per decoder, used to gather/scatter action
# embeddings.
ACTION_DECODER_SLOT_GROUPS = (
    ("move", _MOVE_SLOTS),
    ("switch", _SWITCH_SLOTS),
    ("target", _TARGET_STATIC_SLOTS),
)
# Slot-aligned indices of the concatenated action stream's rows
# ([move | switch | target] order) and the split points between groups — the
# gather on the way into the trunk and the scatter back out are the same
# permutation.
ACTION_GROUP_SLOTS = np.concatenate(
    [slot_indices for _, slot_indices in ACTION_DECODER_SLOT_GROUPS]
)
ACTION_GROUP_SPLITS = np.cumsum(
    [len(slot_indices) for _, slot_indices in ACTION_DECODER_SLOT_GROUPS]
)[:-1]
