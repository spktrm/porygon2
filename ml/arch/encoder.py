from functools import partial
import math
from typing import Dict, Mapping, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from ml.arch.modules import (
    MLP,
    BinaryEncoder,
    PretrainedEmbedding,
    SumEmbeddings,
    TransformerDecoder,
    TransformerEncoder,
    one_hot_concat_jax,
)
from rlenv.data import (
    ABSOLUTE_EDGE_MAX_VALUES,
    ACTION_MAX_VALUES,
    ENTITY_MAX_VALUES,
    MAX_RATIO_TOKEN,
    NUM_ABILITIES,
    NUM_ACTION_TYPES,
    NUM_ACTIONS,
    NUM_EFFECTS,
    NUM_ITEMS,
    NUM_MOVES,
    NUM_SPECIES,
    RELATIVE_EDGE_MAX_VALUES,
)
from rlenv.interfaces import EnvStep, HistoryContainer, HistoryStep
from rlenv.protos.enums_pb2 import ActionsEnum, EffectEnum, SpeciesEnum
from rlenv.protos.features_pb2 import (
    AbsoluteEdgeFeature,
    EntityFeature,
    MovesetFeature,
    RelativeEdgeFeature,
)

# Load pretrained embeddings for various features.
SPECIES_ONEHOT = PretrainedEmbedding(fpath="data/data/gen3/species.npy")
ABILITY_ONEHOT = PretrainedEmbedding(fpath="data/data/gen3/abilities.npy")
ITEM_ONEHOT = PretrainedEmbedding(fpath="data/data/gen3/items.npy")
MOVE_ONEHOT = PretrainedEmbedding(fpath="data/data/gen3/moves.npy")

# # Initialize a binary encoder for specific features.
OCT_ENCODER = BinaryEncoder(num_bits=8)
HEX_ENCODER = BinaryEncoder(num_bits=16)


def get_move_mask(move: chex.Array) -> chex.Array:
    """
    Generate a mask to filter valid moves based on move identifiers.
    """
    action_id_token = move[MovesetFeature.MOVESET_FEATURE__ACTION_ID].astype(jnp.int32)
    return ~(
        (action_id_token == ActionsEnum.ACTIONS_ENUM__MOVE__NULL)
        | (action_id_token == ActionsEnum.ACTIONS_ENUM__SWITCH__NULL)
        | (action_id_token == ActionsEnum.ACTIONS_ENUM__MOVE__PAD)
        | (action_id_token == ActionsEnum.ACTIONS_ENUM__SWITCH__PAD)
    )


def _binary_scale_embedding(to_encode: chex.Array, world_dim: int) -> chex.Array:
    """Encode the feature using its binary representation."""
    chex.assert_rank(to_encode, 0)
    chex.assert_type(to_encode, jnp.int32)
    num_bits = (world_dim - 1).bit_length()
    bit_mask = 1 << np.arange(num_bits)
    pos = jnp.broadcast_to(to_encode[jnp.newaxis], num_bits)
    result = jnp.not_equal(jnp.bitwise_and(pos, bit_mask), 0)
    return result.astype(jnp.float32)


def _encode_one_hot(
    entity: chex.Array,
    feature_idx: int,
    max_values: Dict[int, int],
    value_offset: int = 0,
) -> chex.Array:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    return entity[feature_idx] + value_offset, max_values[feature_idx] + 1


_encode_one_hot_entity = partial(_encode_one_hot, max_values=ENTITY_MAX_VALUES)
_encode_one_hot_action = partial(_encode_one_hot, max_values=ACTION_MAX_VALUES)
_encode_one_hot_relative_edge = partial(
    _encode_one_hot, max_values=RELATIVE_EDGE_MAX_VALUES
)
_encode_one_hot_absolute_edge = partial(
    _encode_one_hot, max_values=ABSOLUTE_EDGE_MAX_VALUES
)
_encode_one_hot_entity_boost = partial(_encode_one_hot_entity, value_offset=6)
_encode_one_hot_relative_edge_boost = partial(
    _encode_one_hot_relative_edge, value_offset=6
)


def _encode_capped_one_hot(
    entity: chex.Array, feature_idx: int, max_values: Dict[int, int]
) -> chex.Array:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    max_value = max_values[feature_idx]
    return jnp.minimum(entity[feature_idx], max_value), max_value + 1


_encode_capped_one_hot_entity = partial(
    _encode_capped_one_hot, max_values=ENTITY_MAX_VALUES
)
_encode_capped_one_hot_relative_edge = partial(
    _encode_capped_one_hot, max_values=RELATIVE_EDGE_MAX_VALUES
)
_encode_capped_one_hot_absolute_edge = partial(
    _encode_capped_one_hot, max_values=ABSOLUTE_EDGE_MAX_VALUES
)


def _encode_sqrt_one_hot(
    entity: chex.Array, feature_idx: int, max_values: Dict[int, int]
) -> chex.Array:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    max_value = max_values[feature_idx]
    max_sqrt_value = int(math.floor(math.sqrt(max_value)))
    x = jnp.floor(jnp.sqrt(entity[feature_idx].astype(jnp.float32)))
    x = jnp.minimum(x.astype(jnp.int32), max_sqrt_value)
    return x, max_sqrt_value + 1


_encode_sqrt_one_hot_entity = partial(
    _encode_sqrt_one_hot, max_values=ENTITY_MAX_VALUES
)
_encode_sqrt_one_hot_action = partial(
    _encode_sqrt_one_hot, max_values=ACTION_MAX_VALUES
)
_encode_sqrt_one_hot_relative_edge = partial(
    _encode_sqrt_one_hot, max_values=RELATIVE_EDGE_MAX_VALUES
)
_encode_sqrt_one_hot_absolute_edge = partial(
    _encode_sqrt_one_hot, max_values=ABSOLUTE_EDGE_MAX_VALUES
)


def _encode_divided_one_hot(
    entity: chex.Array, feature_idx: int, divisor: int, max_values: Dict[int, int]
) -> chex.Array:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    max_value = max_values[feature_idx]
    max_divided_value = max_value // divisor
    x = jnp.floor_divide(entity[feature_idx], divisor)
    x = jnp.minimum(x, max_divided_value)
    return x, max_divided_value + 1


_encode_divided_one_hot_entity = partial(
    _encode_divided_one_hot, max_values=ENTITY_MAX_VALUES
)
_encode_divided_one_hot_relative_edge = partial(
    _encode_divided_one_hot, max_values=RELATIVE_EDGE_MAX_VALUES
)
_encode_divided_one_hot_absolute_edge = partial(
    _encode_divided_one_hot, max_values=ABSOLUTE_EDGE_MAX_VALUES
)


def _features_embedding(
    raw_unit: chex.Array, rescales: Mapping[int, float]
) -> chex.Array:
    """Select features in `rescales`, rescale and concatenate them."""
    chex.assert_rank(raw_unit, 1)
    chex.assert_type(raw_unit, jnp.int32)
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
        consecutive_features = raw_unit[
            feature_indices[i_min] : feature_indices[i_max] + 1
        ]
        consecutive_rescales = jnp.asarray(
            [rescales[feature_indices[i]] for i in range(i_min, i_max + 1)], jnp.float32
        )
        i_min = i_max + 1
        rescaled_features = jnp.multiply(consecutive_features, consecutive_rescales)
        selected_features.append(rescaled_features)
    return jnp.concatenate(selected_features, axis=0).astype(jnp.float32)


def get_entity_mask(entity: chex.Array) -> chex.Array:
    """
    Generate a mask to identify valid entities based on species tokens.
    """
    species_token = entity[EntityFeature.ENTITY_FEATURE__SPECIES].astype(jnp.int32)
    return ~(
        (species_token == SpeciesEnum.SPECIES_ENUM___NULL)
        | (species_token == SpeciesEnum.SPECIES_ENUM___PAD)
    )


def get_edge_mask(edge: chex.Array) -> chex.Array:
    """
    Generate a mask for edges based on their validity tokens.
    """
    return edge[AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__VALID].astype(jnp.int32)


class Encoder(nn.Module):
    """
    Encoder model for processing environment steps and history to generate embeddings.
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(
        self, env_step: EnvStep, history_step: HistoryStep
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Forward pass of the Encoder model. Processes an environment step and outputs
        contextual embeddings for actions.
        """

        # Extract configuration parameters for embedding sizes.
        entity_size = self.cfg.entity_size

        embed_kwargs = dict(
            features=entity_size, embedding_init=nn.initializers.lecun_normal()
        )

        # Initialize embeddings for various features.
        species_embedding = nn.Embed(NUM_SPECIES, **embed_kwargs)
        abilities_embedding = nn.Embed(NUM_ABILITIES, **embed_kwargs)
        items_embedding = nn.Embed(NUM_ITEMS, **embed_kwargs)
        actions_embedding = nn.Embed(NUM_ACTIONS, **embed_kwargs)
        effects_embedding = nn.Embed(NUM_EFFECTS, **embed_kwargs)

        # Initialize aggregation modules for combining feature embeddings.
        entity_aggregate = SumEmbeddings(entity_size)
        entity_mlp = MLP((entity_size,))

        relative_edge_aggregate = SumEmbeddings(entity_size)
        relative_edge_mlp = MLP((entity_size,))

        absolute_edge_aggregate = SumEmbeddings(entity_size)
        absolute_edge_mlp = MLP((entity_size,))

        timestep_aggregate = MLP((entity_size,))

        action_aggregate = SumEmbeddings(entity_size)
        action_mlp = MLP((entity_size,))

        def _encode_entity(entity: chex.Array) -> chex.Array:
            # Encode volatile and type-change indices using the binary encoder.
            volatiles_indices = entity[
                EntityFeature.ENTITY_FEATURE__VOLATILES0 : EntityFeature.ENTITY_FEATURE__VOLATILES8
                + 1
            ]
            volatiles_encoding = HEX_ENCODER(
                volatiles_indices.astype(jnp.uint16)
            ).reshape(-1)

            typechange_indices = entity[
                EntityFeature.ENTITY_FEATURE__TYPECHANGE0 : EntityFeature.ENTITY_FEATURE__TYPECHANGE1
                + 1
            ]
            typechange_encoding = HEX_ENCODER(
                typechange_indices.astype(jnp.uint16)
            ).reshape(-1)

            boolean_code = one_hot_concat_jax(
                [
                    _encode_divided_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__LEVEL, 1
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__ACTIVE
                    ),
                    _encode_one_hot_entity(entity, EntityFeature.ENTITY_FEATURE__SIDE),
                    _encode_divided_one_hot_entity(
                        entity,
                        EntityFeature.ENTITY_FEATURE__HP_RATIO,
                        MAX_RATIO_TOKEN / 32,
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__GENDER
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__STATUS
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__ITEM_EFFECT
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__BEING_CALLED_BACK
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__TRAPPED
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__NEWLY_SWITCHED
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__TOXIC_TURNS
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__SLEEP_TURNS
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__FAINTED
                    ),
                    _encode_one_hot_entity_boost(
                        entity, EntityFeature.ENTITY_FEATURE__BOOST_ATK_VALUE
                    ),
                    _encode_one_hot_entity_boost(
                        entity, EntityFeature.ENTITY_FEATURE__BOOST_DEF_VALUE
                    ),
                    _encode_one_hot_entity_boost(
                        entity, EntityFeature.ENTITY_FEATURE__BOOST_SPA_VALUE
                    ),
                    _encode_one_hot_entity_boost(
                        entity, EntityFeature.ENTITY_FEATURE__BOOST_SPD_VALUE
                    ),
                    _encode_one_hot_entity_boost(
                        entity, EntityFeature.ENTITY_FEATURE__BOOST_SPE_VALUE
                    ),
                    _encode_one_hot_entity_boost(
                        entity, EntityFeature.ENTITY_FEATURE__BOOST_EVASION_VALUE
                    ),
                    _encode_one_hot_entity_boost(
                        entity, EntityFeature.ENTITY_FEATURE__BOOST_ACCURACY_VALUE
                    ),
                    # (entity[EntityFeature.ENTITY_FEATURE__MOVEID0], 0),
                    # (entity[EntityFeature.ENTITY_FEATURE__MOVEID0], 0),
                    # (entity[EntityFeature.ENTITY_FEATURE__MOVEID0], 0),
                    # (entity[EntityFeature.ENTITY_FEATURE__MOVEID0], NUM_MOVES),
                ]
            )

            def _get_action_embedding(feature_index: int):
                mask = (entity[feature_index] == ActionsEnum.ACTIONS_ENUM___NULL) | (
                    entity[feature_index] == ActionsEnum.ACTIONS_ENUM___UNSPECIFIED
                )
                return jnp.where(
                    mask[None],
                    0,
                    actions_embedding(entity[feature_index]),
                )

            embeddings = [
                species_embedding(entity[EntityFeature.ENTITY_FEATURE__SPECIES]),
                abilities_embedding(entity[EntityFeature.ENTITY_FEATURE__ABILITY]),
                items_embedding(entity[EntityFeature.ENTITY_FEATURE__ITEM]),
                boolean_code,
                volatiles_encoding,
                typechange_encoding,
                SPECIES_ONEHOT(entity[EntityFeature.ENTITY_FEATURE__SPECIES]),
                ABILITY_ONEHOT(entity[EntityFeature.ENTITY_FEATURE__ABILITY]),
                ITEM_ONEHOT(entity[EntityFeature.ENTITY_FEATURE__ITEM]),
                (
                    _get_action_embedding(EntityFeature.ENTITY_FEATURE__ACTION_ID0)
                    + _get_action_embedding(EntityFeature.ENTITY_FEATURE__ACTION_ID1)
                    + _get_action_embedding(EntityFeature.ENTITY_FEATURE__ACTION_ID2)
                    + _get_action_embedding(EntityFeature.ENTITY_FEATURE__ACTION_ID3)
                )
                / entity[EntityFeature.ENTITY_FEATURE__NUM_MOVES].clip(min=1),
            ]

            embedding = entity_aggregate(embeddings)
            embedding = entity_mlp(embedding)

            # Apply mask to filter out invalid entities.
            mask = get_entity_mask(entity)
            embedding = jnp.where(mask, embedding, 0)

            return embedding, mask

        def _encode_relative_edge(relative_edge: chex.Array) -> chex.Array:
            # Encode minor arguments and side conditions using the binary encoder.
            minor_args_indices = relative_edge[
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MINOR_ARG0 : RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MINOR_ARG3
                + 1
            ]
            minor_args_encoding = HEX_ENCODER(
                minor_args_indices.astype(jnp.uint16)
            ).reshape(-1)

            side_condition_indices = relative_edge[
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SIDECONDITIONS0 : RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SIDECONDITIONS1
                + 1
            ]
            side_condition_encoding = HEX_ENCODER(
                side_condition_indices.astype(jnp.uint16)
            ).reshape(-1)

            # Aggregate embeddings for the relative edge.
            boolean_code = one_hot_concat_jax(
                [
                    _encode_one_hot_relative_edge(
                        relative_edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MAJOR_ARG,
                    ),
                    _encode_divided_one_hot_relative_edge(
                        relative_edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__DAMAGE_RATIO,
                        MAX_RATIO_TOKEN / 32,
                    ),
                    _encode_divided_one_hot_relative_edge(
                        relative_edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__HEAL_RATIO,
                        MAX_RATIO_TOKEN / 32,
                    ),
                    _encode_one_hot_relative_edge(
                        relative_edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__STATUS_TOKEN,
                    ),
                    _encode_one_hot_relative_edge_boost(
                        relative_edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_ATK_VALUE,
                    ),
                    _encode_one_hot_relative_edge_boost(
                        relative_edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_DEF_VALUE,
                    ),
                    _encode_one_hot_relative_edge_boost(
                        relative_edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_SPA_VALUE,
                    ),
                    _encode_one_hot_relative_edge_boost(
                        relative_edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_SPD_VALUE,
                    ),
                    _encode_one_hot_relative_edge_boost(
                        relative_edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_SPE_VALUE,
                    ),
                    _encode_one_hot_relative_edge_boost(
                        relative_edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_EVASION_VALUE,
                    ),
                    _encode_one_hot_relative_edge_boost(
                        relative_edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_ACCURACY_VALUE,
                    ),
                    _encode_one_hot_relative_edge(
                        relative_edge, RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SPIKES
                    ),
                    _encode_one_hot_relative_edge(
                        relative_edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__TOXIC_SPIKES,
                    ),
                ]
            )

            def _get_effect_embedding(feature_index: int):
                mask = (
                    relative_edge[feature_index] == EffectEnum.EFFECT_ENUM___NULL
                ) | (
                    relative_edge[feature_index] == EffectEnum.EFFECT_ENUM___UNSPECIFIED
                )
                return jnp.where(
                    mask[None],
                    0,
                    effects_embedding(relative_edge[feature_index]),
                )

            effect_embedding = (
                _get_effect_embedding(
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN0
                )
                + _get_effect_embedding(
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN1
                )
                + _get_effect_embedding(
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN2
                )
                + _get_effect_embedding(
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN3
                )
                + _get_effect_embedding(
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN4
                )
            ) / relative_edge[
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_SOURCES
            ].clip(
                min=1
            )

            ability_token = relative_edge[
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ABILITY_TOKEN
            ]
            item_token = relative_edge[
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ITEM_TOKEN
            ]
            move_token = relative_edge[
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MOVE_TOKEN
            ]
            action_token = relative_edge[
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ACTION_TOKEN
            ]
            embeddings = [
                ABILITY_ONEHOT(ability_token),
                ITEM_ONEHOT(item_token),
                MOVE_ONEHOT(move_token),
                items_embedding(item_token),
                abilities_embedding(ability_token),
                actions_embedding(action_token),
                minor_args_encoding,
                effect_embedding,
                side_condition_encoding,
                boolean_code,
                _features_embedding(
                    relative_edge,
                    {
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MOVE_TOKEN: 1
                        / MAX_RATIO_TOKEN,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__HEAL_RATIO: 1
                        / MAX_RATIO_TOKEN,
                    },
                ),
            ]

            embedding = relative_edge_aggregate(embeddings)
            embedding = relative_edge_mlp(embedding)

            # Apply mask to filter out invalid edges.
            mask = get_edge_mask(relative_edge)
            embedding = jnp.where(mask, embedding, 0)

            return embedding, mask

        def _encode_absolute_edge(
            absolute_edge: chex.Array,
            turn_offset: chex.Array,
            request_count_offset: chex.Array,
        ) -> chex.Array:
            """
            Encode features of an absolute edge, including turn and request offsets.
            """
            # Compute turn and request count differences for encoding.
            turn = jnp.abs(
                absolute_edge[AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TURN_VALUE]
                - turn_offset
            )
            request_count = jnp.abs(
                absolute_edge[AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__REQUEST_COUNT]
                - request_count_offset
            )

            # Aggregate embeddings for the absolute edge.
            boolean_code = one_hot_concat_jax(
                [
                    _encode_one_hot_absolute_edge(
                        absolute_edge,
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_ID,
                    ),
                    _encode_one_hot_absolute_edge(
                        absolute_edge,
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_MAX_DURATION,
                    ),
                    _encode_one_hot_absolute_edge(
                        absolute_edge,
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_MIN_DURATION,
                    ),
                    _encode_one_hot_absolute_edge(
                        absolute_edge,
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TERRAIN_ID,
                    ),
                    _encode_one_hot_absolute_edge(
                        absolute_edge,
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TERRAIN_MAX_DURATION,
                    ),
                    _encode_one_hot_absolute_edge(
                        absolute_edge,
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TERRAIN_MIN_DURATION,
                    ),
                    _encode_one_hot_absolute_edge(
                        absolute_edge,
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__PSEUDOWEATHER_ID,
                    ),
                    _encode_one_hot_absolute_edge(
                        absolute_edge,
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__PSEUDOWEATHER_MAX_DURATION,
                    ),
                    _encode_one_hot_absolute_edge(
                        absolute_edge,
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__PSEUDOWEATHER_MIN_DURATION,
                    ),
                ]
            )

            embeddings = [
                boolean_code,
                _binary_scale_embedding(turn, 32),
                _binary_scale_embedding(request_count, 32),
                _binary_scale_embedding(
                    absolute_edge[
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TURN_ORDER_VALUE
                    ],
                    16,
                ),
            ]

            embedding = absolute_edge_aggregate(embeddings)
            embedding = absolute_edge_mlp(embedding)

            # Apply mask to filter out invalid edges.
            mask = get_edge_mask(absolute_edge)
            embedding = jnp.where(mask, embedding, 0)

            return embedding, mask

        # Encode each timestep's features, including nodes and edges.
        def _encode_timestep(
            history_container: HistoryContainer,
            turn_offset: chex.Array,
            request_count_offset: chex.Array,
        ):
            """
            Encode features of a single timestep, including entities and edges.
            """
            # Encode entities and their masks.
            entity_embeddings, _ = jax.vmap(_encode_entity)(history_container.entities)

            # Encode relative edges.
            relative_edge_embeddings, _ = jax.vmap(_encode_relative_edge)(
                history_container.relative_edges
            )

            # Encode absolute edges.
            absolute_edge_embedding, edge_mask = _encode_absolute_edge(
                history_container.absolute_edges, turn_offset, request_count_offset
            )

            # Combine all embeddings for the timestep.
            timestep_embeddings = jnp.concatenate(
                [
                    entity_embeddings[0],
                    entity_embeddings[1],
                    relative_edge_embeddings[0],
                    relative_edge_embeddings[1],
                    absolute_edge_embedding,
                ],
                axis=-1,
            )
            timestep_embedding = timestep_aggregate(timestep_embeddings)

            # Apply mask to the timestep embeddings.
            timestep_embedding = jnp.where(edge_mask, timestep_embedding, 0)

            return timestep_embedding, edge_mask

        # Encode history data across multiple timesteps.
        def _encode_timesteps(history_container: HistoryContainer):
            """
            Encode all timesteps in the history container.
            """
            turn_offset = history_container.absolute_edges[
                ..., AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TURN_VALUE
            ].max(0)
            request_count_offset = history_container.absolute_edges[
                ..., AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__REQUEST_COUNT
            ].max(0)
            return jax.vmap(_encode_timestep, in_axes=(0, None, None))(
                history_container, turn_offset, request_count_offset
            )

        # Encode actions for the current environment step.
        def _encode_action(action: chex.Array, legal: chex.Array) -> chex.Array:
            """
            Encode features of a move, including its type, species, and action ID.
            """
            boolean_code = one_hot_concat_jax(
                [
                    _encode_one_hot_action(
                        action, MovesetFeature.MOVESET_FEATURE__ACTION_TYPE
                    ),
                    _encode_sqrt_one_hot_action(
                        action, MovesetFeature.MOVESET_FEATURE__PP
                    ),
                    _encode_sqrt_one_hot_action(
                        action, MovesetFeature.MOVESET_FEATURE__MAXPP
                    ),
                    _encode_one_hot_action(
                        action, MovesetFeature.MOVESET_FEATURE__HAS_PP
                    ),
                ]
            )

            embeddings = [
                MOVE_ONEHOT(action[MovesetFeature.MOVESET_FEATURE__MOVE_ID]),
                SPECIES_ONEHOT(action[MovesetFeature.MOVESET_FEATURE__SPECIES_ID]),
                actions_embedding(action[MovesetFeature.MOVESET_FEATURE__ACTION_ID]),
                boolean_code,
                _features_embedding(
                    action,
                    {MovesetFeature.MOVESET_FEATURE__PP_RATIO: 1 / MAX_RATIO_TOKEN},
                ),
            ]

            embedding = action_aggregate(embeddings)
            embedding = action_mlp(embedding)

            # Apply mask to the move embeddings.
            mask = get_move_mask(action)
            embedding = jnp.where(mask, embedding, 0)

            return embedding, mask

        _encode_entities = jax.vmap(_encode_entity)
        private_entity_embeddings, private_entity_mask = _encode_entities(
            env_step.private_team
        )
        public_entity_embeddings, public_entity_mask = _encode_entities(
            env_step.public_team
        )

        timestep_embeddings, valid_timestep_mask = _encode_timesteps(
            history_step.major_history
        )

        action_embeddings, _ = jax.vmap(_encode_action)(
            env_step.moveset, env_step.legal.astype(int)
        )

        contextual_private_entity_embeddings = TransformerEncoder(
            **self.cfg.private_entity_encoder.to_dict()
        )(private_entity_embeddings, private_entity_mask)

        contextual_public_entity_embeddings = TransformerEncoder(
            **self.cfg.public_entity_encoder.to_dict()
        )(public_entity_embeddings, public_entity_mask)

        contextual_entity_embeddings = TransformerDecoder(
            **self.cfg.entity_decoder.to_dict()
        )(
            contextual_private_entity_embeddings,
            contextual_public_entity_embeddings,
            private_entity_mask,
            public_entity_mask,
        )

        contextual_timestep_embeddings = TransformerEncoder(
            **self.cfg.timestep_encoder.to_dict()
        )(timestep_embeddings, valid_timestep_mask)

        contextual_entity_embeddings = TransformerDecoder(
            **self.cfg.entity_timestep_decoder.to_dict()
        )(
            contextual_entity_embeddings,
            contextual_timestep_embeddings,
            private_entity_mask,
            valid_timestep_mask,
        )

        contextual_action_embeddings = TransformerDecoder(
            **self.cfg.action_entity_decoder.to_dict()
        )(
            action_embeddings,
            contextual_entity_embeddings,
            env_step.legal,
            private_entity_mask,
        )

        return (
            contextual_entity_embeddings,
            private_entity_mask,
            contextual_action_embeddings,
        )
