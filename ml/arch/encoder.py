import math
from functools import partial
from typing import Dict, Mapping, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from ml.arch.modules import (
    MLP,
    BinaryEncoder,
    DenseMultiHeadProjection,
    GatedResidualLayer,
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
    NUM_EFFECTS,
    NUM_ITEMS,
    NUM_MOVES,
    NUM_SPECIES,
    RELATIVE_EDGE_MAX_VALUES,
)
from rlenv.interfaces import EnvStep, HistoryContainer, HistoryStep
from rlenv.protos.enums_pb2 import (
    AbilitiesEnum,
    ActionsEnum,
    EffectEnum,
    ItemsEnum,
    MovesEnum,
    SpeciesEnum,
)
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


def _binary_scale_encoding(to_encode: chex.Array, world_dim: int) -> chex.Array:
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
) -> Tuple[int, int]:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    return entity[feature_idx] + value_offset, max_values[feature_idx] + 1


def _encode_capped_one_hot(
    entity: chex.Array, feature_idx: int, max_values: Dict[int, int]
) -> Tuple[int, int]:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    max_value = max_values[feature_idx]
    return jnp.minimum(entity[feature_idx], max_value), max_value + 1


def _encode_sqrt_one_hot(
    entity: chex.Array, feature_idx: int, max_values: Dict[int, int]
) -> Tuple[int, int]:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    max_value = max_values[feature_idx]
    max_sqrt_value = int(math.floor(math.sqrt(max_value)))
    x = jnp.floor(jnp.sqrt(entity[feature_idx].astype(jnp.float32)))
    x = jnp.minimum(x.astype(jnp.int32), max_sqrt_value)
    return x, max_sqrt_value + 1


def _encode_divided_one_hot(
    entity: chex.Array, feature_idx: int, divisor: int, max_values: Dict[int, int]
) -> Tuple[int, int]:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    max_value = max_values[feature_idx]
    max_divided_value = max_value // divisor
    x = jnp.floor_divide(entity[feature_idx], divisor)
    x = jnp.minimum(x, max_divided_value)
    return x, max_divided_value + 1


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
_encode_sqrt_one_hot_entity = partial(
    _encode_sqrt_one_hot, max_values=ENTITY_MAX_VALUES
)
_encode_sqrt_one_hot_action = partial(
    _encode_sqrt_one_hot, max_values=ACTION_MAX_VALUES
)
_encode_divided_one_hot_entity = partial(
    _encode_divided_one_hot, max_values=ENTITY_MAX_VALUES
)
_encode_divided_one_hot_relative_edge = partial(
    _encode_divided_one_hot, max_values=RELATIVE_EDGE_MAX_VALUES
)


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

        species_embedding = nn.Embed(
            NUM_SPECIES, name="species_embedding", **embed_kwargs
        )
        items_embedding = nn.Embed(NUM_ITEMS, name="items_embedding", **embed_kwargs)
        abilities_embedding = nn.Embed(
            NUM_ABILITIES, name="abilities_embedding", **embed_kwargs
        )
        moves_embedding = nn.Embed(NUM_MOVES, name="moves_embedding", **embed_kwargs)

        species_linear = nn.Dense(entity_size, use_bias=False, name="species_linear")
        items_linear = nn.Dense(entity_size, use_bias=False, name="items_linear")
        abilities_linear = nn.Dense(
            entity_size, use_bias=False, name="abilities_linear"
        )
        moves_linear = nn.Dense(entity_size, use_bias=False, name="moves_linear")

        def _encode_species(token: chex.Array):
            return jnp.where(
                token == SpeciesEnum.SPECIES_ENUM___UNSPECIFIED,
                0,
                species_linear(SPECIES_ONEHOT(token)),
            )

        def _encode_item(token: chex.Array):
            return jnp.where(
                token == ItemsEnum.ITEMS_ENUM___UNSPECIFIED,
                0,
                items_linear(ITEM_ONEHOT(token)),
            )

        def _encode_ability(token: chex.Array):
            return jnp.where(
                token == AbilitiesEnum.ABILITIES_ENUM___UNSPECIFIED,
                0,
                abilities_linear(ABILITY_ONEHOT(token)),
            )

        def _encode_move(token: chex.Array):
            return jnp.where(
                token == MovesEnum.MOVES_ENUM___UNSPECIFIED,
                0,
                moves_linear(MOVE_ONEHOT(token)),
            )

        # Initialize aggregation modules for combining feature embeddings.
        entity_combine = SumEmbeddings(entity_size, name="entity_combine")
        relative_edge_combine = SumEmbeddings(entity_size, name="relative_edge_combine")
        absolute_edge_combine = SumEmbeddings(entity_size, name="absolute_edge_combine")
        timestep_mlp = MLP((entity_size,), name="timestep_mlp")
        action_combine = SumEmbeddings(entity_size, name="action_combine")

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
                    _encode_sqrt_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__LEVEL
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__ACTIVE
                    ),
                    _encode_one_hot_entity(entity, EntityFeature.ENTITY_FEATURE__SIDE),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__IS_PUBLIC
                    ),
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
                ]
            )

            move_indices = np.array(
                [
                    EntityFeature.ENTITY_FEATURE__MOVEID0,
                    EntityFeature.ENTITY_FEATURE__MOVEID1,
                    EntityFeature.ENTITY_FEATURE__MOVEID2,
                    EntityFeature.ENTITY_FEATURE__MOVEID3,
                ]
            )
            move_pp_indices = np.array(
                [
                    EntityFeature.ENTITY_FEATURE__MOVEPP0,
                    EntityFeature.ENTITY_FEATURE__MOVEPP1,
                    EntityFeature.ENTITY_FEATURE__MOVEPP2,
                    EntityFeature.ENTITY_FEATURE__MOVEPP3,
                ]
            )
            move_tokens = entity[move_indices]

            move_pp_onehot = (
                jnp.where(
                    (
                        (move_tokens != MovesEnum.MOVES_ENUM___NULL)
                        | (move_tokens != MovesEnum.MOVES_ENUM___UNSPECIFIED)
                    )[..., None],
                    jax.nn.one_hot(move_tokens, NUM_MOVES)
                    * move_pp_indices[..., None]
                    / 31,
                    0,
                )
                .sum(0)
                .clip(min=0, max=1)
            )

            moveset_embedding_avg = moves_embedding(move_tokens).sum(0) / entity[
                EntityFeature.ENTITY_FEATURE__NUM_MOVES
            ].clip(min=1)

            move_embeddings = jax.vmap(_encode_move)(move_tokens)
            moveset_linear_embedding_avg = move_embeddings.sum(0) / entity[
                EntityFeature.ENTITY_FEATURE__NUM_MOVES
            ].clip(min=1)

            species_token = entity[EntityFeature.ENTITY_FEATURE__SPECIES]
            ability_token = entity[EntityFeature.ENTITY_FEATURE__ABILITY]
            item_token = entity[EntityFeature.ENTITY_FEATURE__ITEM]
            # last_move_token = entity[EntityFeature.ENTITY_FEATURE__LAST_MOVE]

            embedding = entity_combine(
                encodings=[
                    boolean_code,
                    volatiles_encoding,
                    typechange_encoding,
                    move_pp_onehot,
                ],
                embeddings=[
                    _encode_species(species_token),
                    _encode_ability(ability_token),
                    _encode_item(item_token),
                    species_embedding(species_token),
                    abilities_embedding(ability_token),
                    items_embedding(item_token),
                    moveset_embedding_avg,
                    moveset_linear_embedding_avg,
                ],
            )

            # Apply mask to filter out invalid entities.
            mask = get_entity_mask(entity)
            embedding = jnp.where(mask, embedding, 0)

            return embedding, mask

        def _encode_relative_edge(edge: chex.Array) -> chex.Array:
            # Encode minor arguments and side conditions using the binary encoder.
            minor_args_indices = edge[
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MINOR_ARG0 : RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MINOR_ARG3
                + 1
            ]
            minor_args_encoding = HEX_ENCODER(
                minor_args_indices.astype(jnp.uint16)
            ).reshape(-1)

            side_condition_indices = edge[
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
                        edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MAJOR_ARG,
                    ),
                    _encode_divided_one_hot_relative_edge(
                        edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__DAMAGE_RATIO,
                        MAX_RATIO_TOKEN / 32,
                    ),
                    _encode_divided_one_hot_relative_edge(
                        edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__HEAL_RATIO,
                        MAX_RATIO_TOKEN / 32,
                    ),
                    _encode_one_hot_relative_edge(
                        edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__STATUS_TOKEN,
                    ),
                    _encode_one_hot_relative_edge_boost(
                        edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_ATK_VALUE,
                    ),
                    _encode_one_hot_relative_edge_boost(
                        edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_DEF_VALUE,
                    ),
                    _encode_one_hot_relative_edge_boost(
                        edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_SPA_VALUE,
                    ),
                    _encode_one_hot_relative_edge_boost(
                        edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_SPD_VALUE,
                    ),
                    _encode_one_hot_relative_edge_boost(
                        edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_SPE_VALUE,
                    ),
                    _encode_one_hot_relative_edge_boost(
                        edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_EVASION_VALUE,
                    ),
                    _encode_one_hot_relative_edge_boost(
                        edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_ACCURACY_VALUE,
                    ),
                    _encode_one_hot_relative_edge(
                        edge, RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SPIKES
                    ),
                    _encode_one_hot_relative_edge(
                        edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__TOXIC_SPIKES,
                    ),
                ]
            )

            effct_from_source_onehot = one_hot_concat_jax(
                [
                    (
                        edge[
                            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN0
                        ],
                        0,
                    ),
                    (
                        edge[
                            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN1
                        ],
                        0,
                    ),
                    (
                        edge[
                            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN2
                        ],
                        0,
                    ),
                    (
                        edge[
                            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN3
                        ],
                        0,
                    ),
                    (
                        edge[
                            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN4
                        ],
                        NUM_EFFECTS,
                    ),
                ]
            ).clip(max=1)
            effct_from_source_onehot = effct_from_source_onehot.at[
                ..., EffectEnum.EFFECT_ENUM___UNSPECIFIED
            ].set(0)
            effct_from_source_onehot = effct_from_source_onehot.at[
                ..., EffectEnum.EFFECT_ENUM___NULL
            ].set(0)
            effct_from_source_onehot = effct_from_source_onehot / edge[
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_SOURCES
            ].clip(min=1)

            ability_token = edge[
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ABILITY_TOKEN
            ]
            item_token = edge[RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ITEM_TOKEN]
            move_token = edge[RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MOVE_TOKEN]
            action_token = edge[RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ACTION_TOKEN]

            embedding = relative_edge_combine(
                encodings=[
                    minor_args_encoding,
                    side_condition_encoding,
                    boolean_code,
                    effct_from_source_onehot,
                ],
                embeddings=[
                    _encode_ability(ability_token),
                    _encode_item(item_token),
                    _encode_move(move_token),
                    items_embedding(item_token),
                    abilities_embedding(ability_token),
                    moves_embedding(action_token),
                ],
            )

            return embedding

        def _encode_absolute_edge(edge: chex.Array) -> chex.Array:
            """
            Encode features of an absolute edge, including turn and request offsets.
            """
            # Compute turn and request count differences for encoding.
            edge[AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TURN_VALUE]
            request_count = edge[
                AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__REQUEST_COUNT
            ]

            # Aggregate embeddings for the absolute edge.
            boolean_code = one_hot_concat_jax(
                [
                    _encode_one_hot_absolute_edge(
                        edge,
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_ID,
                    ),
                    _encode_one_hot_absolute_edge(
                        edge,
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_MAX_DURATION,
                    ),
                    _encode_one_hot_absolute_edge(
                        edge,
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_MIN_DURATION,
                    ),
                    _encode_one_hot_absolute_edge(
                        edge,
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TERRAIN_ID,
                    ),
                    _encode_one_hot_absolute_edge(
                        edge,
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TERRAIN_MAX_DURATION,
                    ),
                    _encode_one_hot_absolute_edge(
                        edge,
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TERRAIN_MIN_DURATION,
                    ),
                    _encode_one_hot_absolute_edge(
                        edge,
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__PSEUDOWEATHER_ID,
                    ),
                    _encode_one_hot_absolute_edge(
                        edge,
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__PSEUDOWEATHER_MAX_DURATION,
                    ),
                    _encode_one_hot_absolute_edge(
                        edge,
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__PSEUDOWEATHER_MIN_DURATION,
                    ),
                ]
            )

            embedding = absolute_edge_combine(
                encodings=[
                    boolean_code,
                    # _binary_scale_embedding(turn, 32),
                    _binary_scale_encoding(request_count, 256),
                    _binary_scale_encoding(
                        edge[
                            AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TURN_ORDER_VALUE
                        ],
                        32,
                    ),
                ]
            )

            # Apply mask to filter out invalid edges.
            mask = get_edge_mask(edge)

            return embedding, mask, request_count

        # Encode each timestep's features, including nodes and edges.
        def _encode_timestep(history_container: HistoryContainer):
            """
            Encode features of a single timestep, including entities and edges.
            """
            # Encode entities and their masks.
            entity_embeddings, _ = jax.vmap(_encode_entity)(history_container.entities)

            # Encode relative edges.
            relative_edge_embeddings = jax.vmap(_encode_relative_edge)(
                history_container.relative_edges
            )

            # Encode absolute edges.
            absolute_edge_embedding, edge_mask, timestep_position = (
                _encode_absolute_edge(history_container.absolute_edges)
            )

            # Combine all embeddings for the timestep.
            timestep_embedding = jnp.concatenate(
                [
                    entity_embeddings[0],
                    entity_embeddings[1],
                    relative_edge_embeddings[0],
                    relative_edge_embeddings[1],
                    absolute_edge_embedding,
                ],
                axis=-1,
            )

            timestep_embedding = timestep_mlp(timestep_embedding)

            # Apply mask to the timestep embeddings.
            timestep_embedding = jnp.where(edge_mask, timestep_embedding, 0)

            return timestep_embedding, edge_mask, timestep_position

        # Encode actions for the current environment step.
        def _encode_action(
            action: chex.Array, legal: chex.Array, entity_embedding: chex.Array
        ) -> chex.Array:
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

            embedding = action_combine(
                encodings=[boolean_code],
                embeddings=[
                    _encode_move(action[MovesetFeature.MOVESET_FEATURE__MOVE_ID]),
                    moves_embedding(action[MovesetFeature.MOVESET_FEATURE__MOVE_ID]),
                    entity_embedding[
                        action[MovesetFeature.MOVESET_FEATURE__ENTITY_IDX]
                    ],
                ],
            )

            return embedding

        _encode_entities = jax.vmap(jax.vmap(_encode_entity))
        private_entity_embeddings, private_entity_mask = _encode_entities(
            env_step.private_team
        )
        public_entity_embeddings, public_entity_mask = _encode_entities(
            env_step.public_team
        )

        timestep_embeddings, valid_timestep_mask, _ = jax.vmap(_encode_timestep)(
            history_step.major_history
        )

        entity_mask = jnp.concatenate(
            (private_entity_mask, public_entity_mask), axis=-1
        )
        entity_embeddings = jnp.concatenate(
            (private_entity_embeddings, public_entity_embeddings), axis=-2
        )
        entity_context = jax.vmap(
            TransformerEncoder(**self.cfg.entity_encoder.to_dict())
        )(entity_embeddings, entity_mask)

        history_seq_length = valid_timestep_mask.shape[-1]
        casual_attn_mask = jnp.tril(
            jnp.ones((1, history_seq_length, history_seq_length))
        )
        timestep_embeddings = TransformerEncoder(**self.cfg.timestep_encoder.to_dict())(
            timestep_embeddings, valid_timestep_mask, casual_attn_mask
        )

        history_request_count = jnp.where(
            valid_timestep_mask,
            history_step.major_history.absolute_edges[
                ..., AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__REQUEST_COUNT
            ],
            1e9,
        )
        per_example_timestep_mask = (
            env_step.request_count[..., None] > history_request_count
        )

        fused_temporal = jax.vmap(
            TransformerDecoder(**self.cfg.entity_timestep_decoder.to_dict()),
            in_axes=(0, None, 0, 0),
        )(entity_context, timestep_embeddings, entity_mask, per_example_timestep_mask)

        action_embeddings = jax.vmap(jax.vmap(_encode_action, in_axes=(0, 0, None)))(
            env_step.moveset, env_step.legal.astype(int), fused_temporal[:, :6]
        )

        cross_modal = jax.vmap(
            TransformerDecoder(**self.cfg.action_entity_decoder.to_dict())
        )(
            fused_temporal,
            action_embeddings,
            entity_mask,
            env_step.legal,
        )

        final_embeddings = jax.vmap(
            GatedResidualLayer(**self.cfg.entity_gating.to_dict())
        )(entity_context, cross_modal)
        entity_output = DenseMultiHeadProjection(
            **self.cfg.entity_projection.to_dict()
        )(final_embeddings)

        contextual_action_embeddings = jax.vmap(
            TransformerDecoder(**self.cfg.action_entity_decoder.to_dict())
        )(
            action_embeddings,
            final_embeddings,
            env_step.legal,
            entity_mask,
        )

        return entity_output, entity_mask, contextual_action_embeddings
