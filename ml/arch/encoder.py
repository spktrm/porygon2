import functools
import math
from typing import Mapping, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from ml.arch.modules import (
    MLP,
    CompressSequence,
    PretrainedEmbedding,
    TransformerDecoder,
    TransformerEncoder,
    create_attention_mask,
)
from rlenv.data import (
    NUM_ABILITIES,
    NUM_ACTIONS,
    NUM_EDGE_FROM_TYPES,
    NUM_EDGE_TYPES,
    NUM_EFFECTS,
    NUM_GENDERS,
    NUM_ITEM_EFFECTS,
    NUM_ITEMS,
    NUM_MAJOR_ARGS,
    NUM_MINOR_ARGS,
    NUM_MOVES,
    NUM_SPECIES,
    NUM_STATUS,
    NUM_VOLATILE_STATUS,
    NUM_WEATHER,
)
from rlenv.interfaces import EnvStep, HistoryStep
from rlenv.protos.enums_pb2 import ActionsEnum, SideconditionEnum, SpeciesEnum
from rlenv.protos.features_pb2 import (
    EdgeTypes,
    FeatureEdge,
    FeatureEntity,
    FeatureMoveset,
    FeatureWeather,
)

SPECIES_ONEHOT = PretrainedEmbedding("data/data/gen3/species.npy")
ABILITY_ONEHOT = PretrainedEmbedding("data/data/gen3/abilities.npy")
ITEM_ONEHOT = PretrainedEmbedding("data/data/gen3/items.npy")
MOVE_ONEHOT = PretrainedEmbedding("data/data/gen3/moves.npy")


def get_move_mask(move: chex.Array) -> chex.Array:
    move_id_token = astype(move[FeatureMoveset.MOVESET_ACTION_ID], jnp.int32)
    return (
        jnp.not_equal(move_id_token, ActionsEnum.ACTIONS_MOVE__NULL)
        & jnp.not_equal(move_id_token, ActionsEnum.ACTIONS_SWITCH__NULL)
        & jnp.not_equal(move_id_token, ActionsEnum.ACTIONS_MOVE__PAD)
        & jnp.not_equal(move_id_token, ActionsEnum.ACTIONS_SWITCH__PAD)
    )


def astype(x: chex.Array, dtype: jnp.dtype) -> chex.Array:
    """Cast x if necessary."""
    if x.dtype != dtype:
        return x.astype(dtype)
    else:
        return x


def _encode_one_hot(entity: chex.Array, feature_idx: int, num_classes: int):
    chex.assert_rank(entity, 1)
    return jax.nn.one_hot(entity[feature_idx], num_classes)


def _encode_boost_one_hot(entity: chex.Array, feature_idx: int):
    chex.assert_rank(entity, 1)
    return jax.nn.one_hot(entity[feature_idx] + 6, 13)


def _encode_volatiles_onehot(entity: chex.Array) -> chex.Array:
    chex.assert_rank(entity, 1)
    volatiles = jax.lax.slice(
        entity,
        (FeatureEntity.ENTITY_VOLATILES0,),
        (FeatureEntity.ENTITY_VOLATILES8 + 1,),
    )
    onehot = jax.vmap(functools.partial(_binary_scale_embedding, world_dim=16))(
        volatiles
    )
    return astype(onehot.reshape(-1)[:NUM_VOLATILE_STATUS], jnp.float32)


def _binary_scale_embedding(to_encode: chex.Array, world_dim: int) -> chex.Array:
    """Encode the feature using its binary representation."""
    chex.assert_rank(to_encode, 0)
    num_bits = (world_dim - 1).bit_length()
    bit_mask = 1 << np.arange(num_bits)
    pos = jnp.broadcast_to(to_encode[jnp.newaxis], num_bits)
    result = jnp.not_equal(jnp.bitwise_and(pos, bit_mask), 0)
    return astype(result, jnp.float32)


def _encode_sqrt_one_hot(to_encode: chex.Array, max_value: int) -> chex.Array:
    max_sqrt_value = int(math.floor(math.sqrt(max_value)))
    x = jnp.floor(jnp.sqrt(to_encode.astype(jnp.float32)))
    x = jnp.minimum(x.astype(jnp.int32), max_sqrt_value)
    return jax.nn.one_hot(x, max_sqrt_value + 1)


def _features_embedding(
    entity: chex.Array, rescales: Mapping[int, float]
) -> chex.Array:
    """Select features in `rescales`, rescale and concatenate them."""
    chex.assert_rank(entity, 1)

    assert rescales
    selected_features = []
    feature_indices = sorted(rescales.keys())
    i_min = 0
    while i_min < len(feature_indices):
        i_max = i_min
        while (i_max < len(feature_indices) - 1) and (
            feature_indices[i_max + 1] == feature_indices[i_max] + 1
        ):
            i_max += 1
        consecutive_features = entity[
            feature_indices[i_min] : feature_indices[i_max] + 1
        ]
        consecutive_rescales = jnp.asarray(
            [rescales[feature_indices[i]] for i in range(i_min, i_max + 1)], jnp.float32
        )
        i_min = i_max + 1
        rescaled_features = jnp.multiply(consecutive_features, consecutive_rescales)
        selected_features.append(rescaled_features)
    return astype(jnp.concatenate(selected_features, axis=0), jnp.float32)


def get_entity_mask(entity: chex.Array) -> chex.Array:
    species_token = astype(entity[FeatureEntity.ENTITY_SPECIES], jnp.int32)
    return ~jnp.logical_or(
        jnp.equal(species_token, SpeciesEnum.SPECIES__NULL),
        jnp.equal(species_token, SpeciesEnum.SPECIES__PAD),
    )


def get_edge_mask(edge: chex.Array) -> chex.Array:
    edge_type_token = astype(edge[FeatureEdge.EDGE_TYPE_TOKEN], jnp.int32)
    edge_valid = astype(edge[FeatureEdge.EDGE_VALID], jnp.int32)
    return jnp.not_equal(edge_type_token, EdgeTypes.EDGE_TYPE_NONE) & edge_valid


class SideEncoder(nn.Module):
    entity_size: int

    @nn.compact
    def __call__(self, side: chex.Array) -> chex.Array:

        # Embeddings (to feed to nn.Dense modules):
        one_hot_encoded = [
            astype(side > 0, jnp.float32),
            _encode_one_hot(side, SideconditionEnum.SIDECONDITION_SPIKES, 4),
            _encode_one_hot(side, SideconditionEnum.SIDECONDITION_TOXICSPIKES, 3),
        ]
        boolean_code = jnp.concatenate(one_hot_encoded, axis=-1)

        embedding = nn.Dense(self.entity_size)(boolean_code)
        return embedding


class FieldEncoder(nn.Module):
    entity_size: int

    @nn.compact
    def __call__(self, field: chex.Array) -> chex.Array:

        # Embeddings (to feed to nn.Dense modules):
        one_hot_encoded = [
            _encode_one_hot(field, FeatureWeather.WEATHER_ID, NUM_WEATHER),
            _encode_one_hot(field, FeatureWeather.MAX_DURATION, 9),
            _encode_one_hot(field, FeatureWeather.MIN_DURATION, 9),
        ]
        boolean_code = jnp.concatenate(one_hot_encoded, axis=-1)

        embedding = nn.Dense(self.entity_size)(boolean_code)
        return embedding


class Encoder(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(
        self, env_step: EnvStep, history_step: HistoryStep
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Forward pass of the Encoder model. Processes an environment step and outputs
        the current state and action embeddings.
        """

        # Initialize various encoders and transformers with configurations.
        side_condition_encoder = SideEncoder(
            **self.cfg.side_condition_encoder.to_dict()
        )
        field_encoder = FieldEncoder(**self.cfg.field_encoder.to_dict())

        timestep_linear = lambda x: MLP((self.cfg.entity_size,))(
            nn.Dense(self.cfg.entity_size)(x)
        )
        entity_linear = lambda x: MLP((self.cfg.entity_size,))(
            nn.Dense(self.cfg.entity_size)(x)
        )
        edge_linear = lambda x: MLP((self.cfg.entity_size,))(
            nn.Dense(self.cfg.entity_size)(x)
        )
        action_linear = lambda x: MLP((self.cfg.entity_size,))(
            nn.Dense(self.cfg.entity_size)(x)
        )

        def _encode_entity(entity: chex.Array) -> chex.Array:

            onehot_move = lambda x: jax.nn.one_hot(x, NUM_MOVES)
            moveset_onehot = (
                onehot_move(entity[FeatureEntity.ENTITY_MOVEID0])
                + onehot_move(entity[FeatureEntity.ENTITY_MOVEID1])
                + onehot_move(entity[FeatureEntity.ENTITY_MOVEID2])
                + onehot_move(entity[FeatureEntity.ENTITY_MOVEID3])
            )

            # Encoded one-hots (to pass to jax.nn.one_hot then nn.Dense):
            one_hot_encoded = [
                SPECIES_ONEHOT(entity[FeatureEntity.ENTITY_SPECIES]),
                ABILITY_ONEHOT(entity[FeatureEntity.ENTITY_ABILITY]),
                ITEM_ONEHOT(entity[FeatureEntity.ENTITY_ITEM]),
                _encode_one_hot(entity, FeatureEntity.ENTITY_SPECIES, NUM_SPECIES),
                _encode_one_hot(entity, FeatureEntity.ENTITY_ABILITY, NUM_ABILITIES),
                _encode_one_hot(entity, FeatureEntity.ENTITY_ITEM, NUM_ITEMS),
                moveset_onehot,
                _encode_volatiles_onehot(entity),
                _encode_sqrt_one_hot(entity[FeatureEntity.ENTITY_LEVEL], 100),
                _encode_sqrt_one_hot(entity[FeatureEntity.ENTITY_HP_TOKEN], 1023),
                _encode_one_hot(entity, FeatureEntity.ENTITY_GENDER, NUM_GENDERS),
                _encode_one_hot(entity, FeatureEntity.ENTITY_STATUS, NUM_STATUS),
                _encode_one_hot(
                    entity, FeatureEntity.ENTITY_ITEM_EFFECT, NUM_ITEM_EFFECTS
                ),
                # _encode_one_hot(entity, FeatureEntity.ENTITY_BEING_CALLED_BACK, 2),
                _encode_one_hot(entity, FeatureEntity.ENTITY_TRAPPED, 2),
                # _encode_one_hot(entity, FeatureEntity.ENTITY_NEWLY_SWITCHED, 2),
                _encode_one_hot(entity, FeatureEntity.ENTITY_TOXIC_TURNS, 8),
                _encode_one_hot(entity, FeatureEntity.ENTITY_SLEEP_TURNS, 4),
                _encode_one_hot(entity, FeatureEntity.ENTITY_FAINTED, 2),
                _encode_one_hot(entity, FeatureEntity.ENTITY_ACTIVE, 2),
                _encode_boost_one_hot(entity, FeatureEntity.ENTITY_BOOST_ATK_VALUE),
                _encode_boost_one_hot(entity, FeatureEntity.ENTITY_BOOST_DEF_VALUE),
                _encode_boost_one_hot(entity, FeatureEntity.ENTITY_BOOST_SPA_VALUE),
                _encode_boost_one_hot(entity, FeatureEntity.ENTITY_BOOST_SPD_VALUE),
                _encode_boost_one_hot(entity, FeatureEntity.ENTITY_BOOST_SPE_VALUE),
                _encode_boost_one_hot(entity, FeatureEntity.ENTITY_BOOST_EVASION_VALUE),
                _encode_boost_one_hot(
                    entity, FeatureEntity.ENTITY_BOOST_ACCURACY_VALUE
                ),
            ]

            boolean_code = jnp.concatenate(one_hot_encoded, axis=-1)
            embedding = entity_linear(boolean_code)

            mask = get_entity_mask(entity)
            embedding = jnp.where(mask, embedding, 0)
            return embedding, mask

        def _encode_edge(edge: chex.Array) -> chex.Array:

            # Embeddings (to feed to nn.Dense modules):
            one_hot_encoded = [
                ABILITY_ONEHOT(edge[FeatureEdge.ABILITY_TOKEN]),
                ITEM_ONEHOT(edge[FeatureEdge.ITEM_TOKEN]),
                _encode_one_hot(edge, FeatureEdge.ACTION_TOKEN, NUM_ACTIONS),
                _encode_one_hot(edge, FeatureEdge.ITEM_TOKEN, NUM_ITEMS),
                _encode_one_hot(edge, FeatureEdge.ABILITY_TOKEN, NUM_ABILITIES),
                _encode_one_hot(edge, FeatureEdge.MAJOR_ARG, NUM_MAJOR_ARGS),
                _encode_one_hot(edge, FeatureEdge.MINOR_ARG, NUM_MINOR_ARGS),
                _encode_one_hot(edge, FeatureEdge.FROM_SOURCE_TOKEN, NUM_EFFECTS),
                _encode_one_hot(edge, FeatureEdge.FROM_TYPE_TOKEN, NUM_EDGE_FROM_TYPES),
                _encode_one_hot(edge, FeatureEdge.EDGE_TYPE_TOKEN, NUM_EDGE_TYPES),
                # Healing
                _encode_sqrt_one_hot(
                    (edge[FeatureEdge.DAMAGE_TOKEN]).clip(min=0), 1023
                ),
                # Damage
                _encode_sqrt_one_hot(
                    jnp.abs(edge[FeatureEdge.DAMAGE_TOKEN].clip(max=0)), 1023
                ),
                _binary_scale_embedding(
                    edge[FeatureEdge.EDGE_AFFECTING_SIDE].astype(jnp.int32), 3
                ),
                _binary_scale_embedding(
                    edge[FeatureEdge.TURN_ORDER_VALUE].astype(jnp.int32), 32
                ),
                _binary_scale_embedding(
                    edge[FeatureEdge.TURN_VALUE].astype(jnp.int32), 512
                ),
                _encode_one_hot(edge, FeatureEdge.STATUS_TOKEN, NUM_STATUS),
                _encode_boost_one_hot(edge, FeatureEdge.BOOST_ATK_VALUE),
                _encode_boost_one_hot(edge, FeatureEdge.BOOST_DEF_VALUE),
                _encode_boost_one_hot(edge, FeatureEdge.BOOST_SPA_VALUE),
                _encode_boost_one_hot(edge, FeatureEdge.BOOST_SPD_VALUE),
                _encode_boost_one_hot(edge, FeatureEdge.BOOST_SPE_VALUE),
                _encode_boost_one_hot(edge, FeatureEdge.BOOST_EVASION_VALUE),
                _encode_boost_one_hot(edge, FeatureEdge.BOOST_ACCURACY_VALUE),
            ]

            boolean_code = jnp.concatenate(one_hot_encoded, axis=-1)
            embedding = edge_linear(boolean_code)

            mask = get_edge_mask(edge)

            embedding = jnp.where(mask, embedding, 0)

            return embedding, mask

        # Encode each timestep's nodes, edges, side conditions, and field data
        def _encode_timestep(
            entities_per_edge: chex.Array,
            edge_features: chex.Array,
            side_conditions_per_edge: chex.Array,
            field_per_edge: chex.Array,
        ):
            # Encode nodes (entities) and generate masks
            entity_embeddings, _ = jax.vmap(_encode_entity)(entities_per_edge)

            # Encode edges, incorporating entity embeddings
            edge_embeddings, edge_mask = _encode_edge(edge_features)

            # Encode side conditions and field data
            side_condition_embeddings = jax.vmap(side_condition_encoder)(
                side_conditions_per_edge
            )

            field_embedding = field_encoder(field_per_edge)

            # Merge aggregated embeddings with timestep context

            timestep_embedding = jnp.concatenate(
                (
                    edge_embeddings,
                    entity_embeddings[0],
                    entity_embeddings[1],
                    side_condition_embeddings[0],
                    side_condition_embeddings[1],
                    field_embedding,
                ),
                axis=-1,
            )

            timestep_embedding = timestep_linear(timestep_embedding)
            timestep_embedding = jnp.where(edge_mask, timestep_embedding, 0)

            # Return combined timestep embedding and mask
            return timestep_embedding, edge_mask

        # Process history across timesteps
        timestep_embeddings, valid_timestep_mask = jax.vmap(_encode_timestep)(
            history_step.history_entities,
            history_step.history_edges,
            history_step.history_side_conditions,
            history_step.history_field,
        )

        (
            compressed_embeddings0,
            compressed_embeddings1,
            timestep_self_attn_mask0,
            timestep_self_attn_mask1,
            timestep_causal_attn_mask0,
            timestep_causal_attn_mask1,
        ) = CompressSequence(self.cfg.entity_size)(
            timestep_embeddings, history_step, env_step
        )

        timestep_transformer = TransformerEncoder(
            **self.cfg.timestep_transformer.to_dict()
        )

        causal_mask = jnp.expand_dims(jnp.eye(128), axis=0)

        contextual_timestep_embeddings0 = timestep_transformer(
            compressed_embeddings0,
            mask=timestep_self_attn_mask0,
            causal_mask=causal_mask,
        )
        contextual_timestep_embeddings1 = timestep_transformer(
            compressed_embeddings1,
            mask=timestep_self_attn_mask1,
            causal_mask=causal_mask,
        )

        # Process private entities and generate masks
        entity_embeddings, valid_entity_mask = jax.vmap(jax.vmap(_encode_entity))(
            env_step.team.reshape(env_step.team.shape[0], -1, env_step.team.shape[-1])
        )

        entity_timestep_transformer = TransformerDecoder(
            **self.cfg.entity_timestep_transformer.to_dict()
        )
        cast_timestep_to_entity = jax.vmap(
            entity_timestep_transformer, in_axes=(0, None, 0, 0)
        )

        contextual_entity_embeddings0 = cast_timestep_to_entity(
            entity_embeddings,
            contextual_timestep_embeddings0,
            valid_entity_mask & jnp.expand_dims(env_step.player_id == 0, axis=-1),
            timestep_causal_attn_mask0,
        )
        contextual_entity_embeddings1 = cast_timestep_to_entity(
            entity_embeddings,
            contextual_timestep_embeddings1,
            valid_entity_mask & jnp.expand_dims(env_step.player_id == 1, axis=-1),
            timestep_causal_attn_mask1,
        )
        contextual_entity_embeddings = (
            contextual_entity_embeddings0 + contextual_entity_embeddings1
        )

        # Compute action embeddings
        def _encode_move(move: chex.Array) -> chex.Array:
            # Encoded one-hots (to pass to jax.nn.one_hot then nn.Dense):
            one_hot_encoded = [
                SPECIES_ONEHOT(move[FeatureMoveset.MOVESET_SPECIES_ID]),
                MOVE_ONEHOT(move[FeatureMoveset.MOVESET_MOVE_ID]),
                _encode_one_hot(move, FeatureMoveset.MOVESET_ACTION_TYPE, 2),
                _encode_one_hot(move, FeatureMoveset.MOVESET_ACTION_ID, NUM_ACTIONS),
                _encode_sqrt_one_hot(
                    move[FeatureMoveset.MOVESET_PPUSED].astype(jnp.int32), 65
                ),
            ]

            mask = get_move_mask(move)

            boolean_code = jnp.concatenate(one_hot_encoded, axis=-1)
            embedding = action_linear(boolean_code)

            return embedding, mask

        action_embeddings, _ = jax.vmap(jax.vmap(_encode_move))(env_step.moveset[:, 0])

        action_entity_transformer = TransformerDecoder(
            **self.cfg.action_entity_transformer.to_dict()
        )
        contextual_action_embeddings = jax.vmap(action_entity_transformer)(
            action_embeddings,
            contextual_entity_embeddings,
            env_step.legal,
            valid_entity_mask,
        )

        return (
            contextual_entity_embeddings,
            valid_entity_mask,
            contextual_action_embeddings,
        )
