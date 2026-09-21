"""Shared semantic identities added to normalised trunk content."""

import jax
import jax.numpy as jnp
import numpy as np

from rl.environment.data import TARGET_SLOT_INDICES
from rl.environment.interfaces import PlayerEnvOutput
from rl.environment.protos.features_pb2 import (
    EntityPrivateNodeFeature,
    EntityPublicNodeFeature,
    InfoFeature,
)
from rl.environment.protos.service_pb2 import TargetSlot
from rl.model.constants import (
    ACTIVE_STATE_ROWS,
    HISTORY_ACTIVE_ROWS,
    HISTORY_ENTITY_ROWS,
    MOVE_ROWS,
    NUM_ACTIVE_SLOTS,
    NUM_ACTIVES_PER_SIDE,
    NUM_SEQUENCE_ROWS,
    OPP_PRIVATE_ROWS,
    PRIVATE_ROWS,
    PUBLIC_ROWS,
    SEQUENCE_SLICES,
    TARGET_ROWS,
    SequenceGroup,
)
from rl.model.heads import chosen_bank_rows

SIDE_OPPONENT = 0
SIDE_MINE = 1
BENCH_POSITION = 0
FIRST_ACTIVE_POSITION = 2
SECOND_ACTIVE_POSITION = 1

_TARGET_SIDES = {
    TargetSlot.TARGET_SLOT___UNSPECIFIED: (),
    TargetSlot.TARGET_SLOT__DEFAULT: (),
    TargetSlot.TARGET_SLOT__ALLY_1: (SIDE_MINE,),
    TargetSlot.TARGET_SLOT__ALLY_1_PASS: (SIDE_MINE,),
    TargetSlot.TARGET_SLOT__ALLY_2: (SIDE_MINE,),
    TargetSlot.TARGET_SLOT__ALLY_2_PASS: (SIDE_MINE,),
    TargetSlot.TARGET_SLOT__ENEMY_1: (SIDE_OPPONENT,),
    TargetSlot.TARGET_SLOT__ENEMY_2: (SIDE_OPPONENT,),
    TargetSlot.TARGET_SLOT__AUTO: (),
    TargetSlot.TARGET_SLOT__ALL: (SIDE_MINE, SIDE_OPPONENT),
    TargetSlot.TARGET_SLOT__ALLY_SIDE: (SIDE_MINE,),
    TargetSlot.TARGET_SLOT__FOE_SIDE: (SIDE_OPPONENT,),
    TargetSlot.TARGET_SLOT__ALLY_TEAM: (SIDE_MINE,),
    TargetSlot.TARGET_SLOT__RANDOM_NORMAL: (SIDE_OPPONENT,),
    TargetSlot.TARGET_SLOT__ALL_ADJACENT: (SIDE_MINE, SIDE_OPPONENT),
    TargetSlot.TARGET_SLOT__ALL_ADJACENT_FOES: (SIDE_OPPONENT,),
    TargetSlot.TARGET_SLOT__ALLIES: (SIDE_MINE,),
}
_TARGET_POSITIONS = {
    TargetSlot.TARGET_SLOT__ALLY_1: FIRST_ACTIVE_POSITION,
    TargetSlot.TARGET_SLOT__ALLY_1_PASS: FIRST_ACTIVE_POSITION,
    TargetSlot.TARGET_SLOT__ALLY_2: SECOND_ACTIVE_POSITION,
    TargetSlot.TARGET_SLOT__ALLY_2_PASS: SECOND_ACTIVE_POSITION,
    TargetSlot.TARGET_SLOT__ENEMY_1: FIRST_ACTIVE_POSITION,
    TargetSlot.TARGET_SLOT__ENEMY_2: SECOND_ACTIVE_POSITION,
}
TARGET_SIDE_WEIGHTS = np.asarray(
    [
        [side in _TARGET_SIDES[slot] for side in range(2)]
        for slot in TARGET_SLOT_INDICES
    ],
    dtype=np.float32,
)
TARGET_POSITION_WEIGHTS = np.asarray(
    [
        [_TARGET_POSITIONS.get(slot, -1) == position for position in range(3)]
        for slot in TARGET_SLOT_INDICES
    ],
    dtype=np.float32,
)


# The active-slot rows, mine then theirs, first position then second -- the
# order `ACTIVE_STATE` and the history's active-slot rows share.
ACTIVE_SLOT_SIDES = np.repeat([SIDE_MINE, SIDE_OPPONENT], NUM_ACTIVES_PER_SIDE)
ACTIVE_SLOT_POSITIONS = np.tile([FIRST_ACTIVE_POSITION, SECOND_ACTIVE_POSITION], 2)


def active_slot_identities(
    side_embeddings: jax.Array, position_embeddings: jax.Array
) -> jax.Array:
    return (
        side_embeddings[ACTIVE_SLOT_SIDES] + position_embeddings[ACTIVE_SLOT_POSITIONS]
    )


def active_slot_index(sides: jax.Array, positions: jax.Array) -> jax.Array:
    """The active-slot row an entity with this side and position occupies;
    NUM_ACTIVE_SLOTS for a benched one."""
    matches = (sides[..., None] == ACTIVE_SLOT_SIDES) & (
        positions[..., None] == ACTIVE_SLOT_POSITIONS
    )
    return jnp.where(matches.any(axis=-1), matches.argmax(axis=-1), NUM_ACTIVE_SLOTS)


def field_identities(side_embeddings: jax.Array) -> jax.Array:
    return jnp.stack(
        (
            jnp.zeros_like(side_embeddings[0]),
            side_embeddings[SIDE_MINE],
            side_embeddings[SIDE_OPPONENT],
        )
    )


def private_positions(
    env_step: PlayerEnvOutput, private_team: jax.Array, side: int
) -> jax.Array:
    private_keys = private_team[
        :, EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ENTITY_IDX
    ]
    public_order = env_step.info[
        InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0 : InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11
        + 1
    ]
    public_sides = env_step.public_team[
        :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE
    ]
    public_positions = env_step.public_team[
        :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE
    ]
    matches = (
        (private_keys[:, None] > 0)
        & (public_order[None, :] >= 0)
        & (private_keys[:, None] == public_order[None, :] + 1)
        & (public_sides[None, :] == side)
    )
    # Unrevealed sheet entries have no public key and are on the bench.
    return jnp.max(
        jnp.where(matches, public_positions[None, :], BENCH_POSITION), axis=-1
    )


def public_identities(
    public_sides: jax.Array,
    public_positions: jax.Array,
    side_embeddings: jax.Array,
    position_embeddings: jax.Array,
    target_embeddings: jax.Array,
) -> jax.Array:
    """The full-layout identity table with only the public tier's rows
    set: entity and history-entity rows carry their own side and
    position, the target rows their fixed side/position mixes, the
    active-slot rows their slot's side and position, the field triples the
    side pair. Written once for the request path and the
    per-event path."""
    identities = jnp.zeros(
        (NUM_SEQUENCE_ROWS, side_embeddings.shape[-1]), side_embeddings.dtype
    )
    entity_identities = (
        side_embeddings[public_sides] + position_embeddings[public_positions]
    )
    identities = identities.at[PUBLIC_ROWS].set(entity_identities)
    identities = identities.at[HISTORY_ENTITY_ROWS].set(entity_identities)
    targets = (
        target_embeddings
        + jnp.asarray(TARGET_SIDE_WEIGHTS, side_embeddings.dtype) @ side_embeddings
        + jnp.asarray(TARGET_POSITION_WEIGHTS, position_embeddings.dtype)
        @ position_embeddings
    )
    identities = identities.at[TARGET_ROWS].set(targets)
    for rows in (ACTIVE_STATE_ROWS, HISTORY_ACTIVE_ROWS):
        identities = identities.at[rows].set(
            active_slot_identities(side_embeddings, position_embeddings)
        )
    for group in (SequenceGroup.FIELD, SequenceGroup.HISTORY_FIELD):
        identities = identities.at[SEQUENCE_SLICES[group]].set(
            field_identities(side_embeddings)
        )
    return identities


def sequence_identities(
    env_step: PlayerEnvOutput,
    side_embeddings: jax.Array,
    position_embeddings: jax.Array,
    target_embeddings: jax.Array,
    *,
    include_opponent: bool,
) -> jax.Array:
    identities = public_identities(
        env_step.public_team[
            :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE
        ],
        env_step.public_team[
            :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE
        ],
        side_embeddings,
        position_embeddings,
        target_embeddings,
    )
    targets = identities[TARGET_ROWS]
    identities = identities.at[PRIVATE_ROWS].set(
        side_embeddings[SIDE_MINE]
        + position_embeddings[
            private_positions(env_step, env_step.private_team, SIDE_MINE)
        ]
    )
    if include_opponent:
        identities = identities.at[OPP_PRIVATE_ROWS].set(
            side_embeddings[SIDE_OPPONENT]
            + position_embeddings[
                private_positions(env_step, env_step.opp_private_team, SIDE_OPPONENT)
            ]
        )
    previous_source, previous_target = chosen_bank_rows(
        identities[PRIVATE_ROWS],
        identities[MOVE_ROWS],
        targets,
        env_step.info[InfoFeature.INFO_FEATURE__PREV_ACTION_CELL],
    )
    return identities.at[SEQUENCE_SLICES[SequenceGroup.PREV_ACTION]].set(
        jnp.stack((previous_source, previous_target))
    )
