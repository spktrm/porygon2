"""Pure feature-encoding helpers shared by the encoder.

Everything here is a parameter-free function of raw observation arrays:
one-hot codecs (plus their per-feature-table partials), species-token
validity masks, boost-stage codecs and masked pooling. Modules with
parameters live in modules.py; the Encoder itself composes these in
encoder.py.
"""

import math
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np

from rl.environment.data import (
    ACTION_MAX_VALUES,
    ENTITY_EDGE_MAX_VALUES,
    ENTITY_PRIVATE_MAX_VALUES,
    ENTITY_PUBLIC_MAX_VALUES,
    FIELD_MAX_VALUES,
)
from rl.environment.protos.enums_pb2 import SpeciesEnum
from rl.environment.protos.features_pb2 import (
    EntityPrivateNodeFeature,
    EntityRevealedNodeFeature,
)


def binary_scale_encoding(
    to_encode: jax.Array, world_dim: int, dtype: jnp.dtype = jnp.float32
) -> jax.Array:
    """Encode the feature using its binary representation."""
    chex.assert_rank(to_encode, 0)
    chex.assert_type(to_encode, jnp.int32)
    num_bits = (world_dim - 1).bit_length()
    bit_mask = 1 << np.arange(num_bits)
    pos = jnp.broadcast_to(to_encode[jnp.newaxis], num_bits)
    result = jnp.not_equal(jnp.bitwise_and(pos, bit_mask), 0)
    return result.astype(dtype)


def encode_one_hot(
    entity: jax.Array,
    feature_idx: int,
    max_values: dict[int, int],
    value_offset: int = 0,
) -> tuple[int, int]:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    width = max_values[feature_idx] + 1
    # Clip like the sqrt/divided variants: one_hot_concat_jax lays the
    # blocks end to end, so an out-of-range value would silently light a
    # bit in the NEXT feature's block.
    return jnp.clip(entity[feature_idx] + value_offset, 0, width - 1), width


def encode_sqrt_one_hot(
    entity: jax.Array,
    feature_idx: int,
    max_values: dict[int, int],
    dtype: jnp.dtype = jnp.int32,
) -> tuple[int, int]:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    max_value = max_values[feature_idx]
    max_sqrt_value = int(math.floor(math.sqrt(max_value)))
    x = jnp.floor(jnp.sqrt(entity[feature_idx].astype(dtype)))
    x = jnp.minimum(x.astype(jnp.int32), max_sqrt_value)
    return x, max_sqrt_value + 1


def encode_divided_one_hot(
    entity: jax.Array, feature_idx: int, divisor: int, max_values: dict[int, int]
) -> tuple[int, int]:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    max_value = max_values[feature_idx]
    max_divided_value = max_value // divisor
    x = jnp.floor_divide(entity[feature_idx], divisor)
    x = jnp.minimum(x, max_divided_value)
    return x, max_divided_value + 1


encode_one_hot_public_entity = partial(
    encode_one_hot, max_values=ENTITY_PUBLIC_MAX_VALUES
)
encode_one_hot_private_entity = partial(
    encode_one_hot, max_values=ENTITY_PRIVATE_MAX_VALUES
)
encode_one_hot_action = partial(encode_one_hot, max_values=ACTION_MAX_VALUES)
encode_one_hot_edge = partial(encode_one_hot, max_values=ENTITY_EDGE_MAX_VALUES)
encode_one_hot_field = partial(encode_one_hot, max_values=FIELD_MAX_VALUES)
encode_sqrt_one_hot_public_entity = partial(
    encode_sqrt_one_hot, max_values=ENTITY_PUBLIC_MAX_VALUES
)
encode_sqrt_one_hot_action = partial(encode_sqrt_one_hot, max_values=ACTION_MAX_VALUES)
encode_divided_one_hot_public_entity = partial(
    encode_divided_one_hot, max_values=ENTITY_PUBLIC_MAX_VALUES
)
encode_divided_one_hot_edge = partial(
    encode_divided_one_hot, max_values=ENTITY_EDGE_MAX_VALUES
)


def get_public_entity_mask(revealed: jax.Array) -> jax.Array:
    """
    Generate a mask to identify valid entities based on species tokens.
    """
    species_token = revealed[
        EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
    ]
    return ~(
        (species_token == SpeciesEnum.SPECIES_ENUM___NULL)
        | (species_token == SpeciesEnum.SPECIES_ENUM___PAD)
        | (species_token == SpeciesEnum.SPECIES_ENUM___UNSPECIFIED)
    )


def get_private_entity_mask(private: jax.Array) -> jax.Array:
    """
    Generate a mask to identify valid entities based on species tokens.
    """
    species_token = private[
        EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SPECIES
    ]
    return ~(
        (species_token == SpeciesEnum.SPECIES_ENUM___NULL)
        | (species_token == SpeciesEnum.SPECIES_ENUM___PAD)
        | (species_token == SpeciesEnum.SPECIES_ENUM___UNSPECIFIED)
    )


def encode_boosts(boosts: jax.Array, offset: int):
    return jnp.where(
        boosts > 0,
        (offset + boosts) / offset,
        offset / (offset - boosts),
    )


def encode_reg_boosts(boosts: jax.Array):
    """Encodes according to https://bulbapedia.bulbagarden.net/wiki/Stat_modifier#Stage_multipliers"""
    return (1 / math.log(2)) * jnp.log(encode_boosts(boosts, 2))


def encode_spe_boosts(boosts: jax.Array):
    """Encodes according to https://bulbapedia.bulbagarden.net/wiki/Stat_modifier#Stage_multipliers"""
    return (2 / math.log(3)) * jnp.log(encode_boosts(boosts, 3))
