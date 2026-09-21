"""Observed self-play outcomes; targets have no dependency on model parameters."""

import chex
import jax
import jax.numpy as jnp

from constants import MAX_RATIO_TOKEN
from rl.environment.data import MOVE_CELL_OFFSET, OTHER_CELL_OFFSET
from rl.environment.protos.enums_pb2 import BattlemajorargsEnum
from rl.environment.protos.features_pb2 import (
    EntityEdgeFeature,
    EntityPublicNodeFeature,
    FieldFeature,
    InfoFeature,
)
from rl.model.constants import NUM_PUBLIC_SLOTS, RELEVANT_ENTITY_FEATURES

HP_CHANGE_SUPPORT = tuple(index / 10 for index in range(-10, 11))
NUM_HP_CHANGE_BINS = len(HP_CHANGE_SUPPORT)
NUM_OBSERVABLE_LOGITS = 1 + NUM_PUBLIC_SLOTS * (NUM_HP_CHANGE_BINS + 1)
_PUBLIC_ORDER = slice(
    InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0,
    InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0 + NUM_PUBLIC_SLOTS,
)


@chex.dataclass
class ObservedConsequences:
    executed: jax.Array
    execution_valid: jax.Array
    hp_change: jax.Array
    hp_valid: jax.Array
    fainted: jax.Array
    faint_valid: jax.Array


def following(values):
    return jnp.concatenate((values[1:], values[-1:]), axis=0)


def _execution_events(field, edges, public):
    relevant = field[:, RELEVANT_ENTITY_FEATURES]
    in_bounds = (relevant >= 0) & (relevant < edges.shape[0])
    edge_valid = (
        jnp.arange(relevant.shape[-1])[None]
        < field[:, FieldFeature.FIELD_FEATURE__NUM_RELEVANT, None]
    )
    indices = jnp.clip(relevant, 0, edges.shape[0] - 1)
    major = edges[indices, EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG]
    source = (
        in_bounds
        & edge_valid
        & (major > BattlemajorargsEnum.BATTLEMAJORARGS_ENUM___PAD)
    )
    first = jnp.argmax(source, axis=-1)
    source_index = jnp.take_along_axis(indices, first[:, None], axis=-1)[:, 0]
    source_major = jnp.take_along_axis(major, first[:, None], axis=-1)[:, 0]
    mine = (
        public[source_index, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE]
        == 1
    )
    valid = source.any(-1) & mine & (field[:, FieldFeature.FIELD_FEATURE__VALID] > 0)
    return (
        valid & (source_major == BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__MOVE),
        valid & (source_major == BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__CANT),
    )


def observed_consequences(env, history, packed, action_index, acted_mask):
    order = env.info[..., _PUBLIC_ORDER]
    next_order = following(order)
    matches = (order[..., :, None] == next_order[..., None, :]) & (
        order[..., :, None] >= 0
    )
    unique_now = (order[..., :, None] == order[..., None, :]).sum(-1) == 1
    identity_valid = (matches.sum(-1) == 1) & unique_now
    next_index = jnp.argmax(matches, axis=-1)
    next_public = jnp.take_along_axis(
        following(env.public_team), next_index[..., None], axis=2
    )
    hp_feature = EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO
    faint_feature = EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED
    hp_change = (
        next_public[..., hp_feature].astype(jnp.float32)
        - env.public_team[..., hp_feature]
    ) / MAX_RATIO_TOKEN
    was_fainted = env.public_team[..., faint_feature] > 0
    fainted = (next_public[..., faint_feature] > 0) & ~was_fainted
    paired_action = acted_mask & (action_index < OTHER_CELL_OFFSET)
    entity_valid = paired_action[..., None] & identity_valid
    executed, unable = jax.vmap(_execution_events, in_axes=(1, 1, 1), out_axes=1)(
        history.field, packed.edge_cache, packed.public_cache
    )
    next_count = following(env.info[..., InfoFeature.INFO_FEATURE__REQUEST_COUNT])
    interval = (
        next_count[..., None]
        == history.field[..., FieldFeature.FIELD_FEATURE__REQUEST_COUNT].T[None]
    )
    own_move = (interval & executed.T[None]).any(-1)
    own_cant = (interval & unable.T[None]).any(-1)
    is_move = (action_index >= MOVE_CELL_OFFSET) & (action_index < OTHER_CELL_OFFSET)
    return ObservedConsequences(
        executed=own_move,
        execution_valid=acted_mask & is_move & (own_move | own_cant),
        hp_change=jnp.clip(hp_change, -1, 1),
        hp_valid=entity_valid,
        fainted=fainted,
        faint_valid=entity_valid & ~was_fainted,
    )
