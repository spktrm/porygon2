from typing import Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from ml.arch.modules import (
    MLP,
    BinaryEncoder,
    PretrainedEmbedding,
    SumEmbeddings,
    TransformerDecoder,
    TransformerEncoder,
    feature_encode_entity,
)
from rlenv.data import (
    ABSOLUTE_EDGE_MAX_VALUES,
    ENTITY_MAX_VALUES,
    NUM_ABILITIES,
    NUM_ACTIONS,
    NUM_EFFECTS,
    NUM_ITEMS,
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
    action_id_token = astype(move[MovesetFeature.MOVESET_FEATURE__ACTION_ID], jnp.int32)
    return ~(
        (action_id_token == ActionsEnum.ACTIONS_ENUM__MOVE__NULL)
        | (action_id_token == ActionsEnum.ACTIONS_ENUM__SWITCH__NULL)
        | (action_id_token == ActionsEnum.ACTIONS_ENUM__MOVE__PAD)
        | (action_id_token == ActionsEnum.ACTIONS_ENUM__SWITCH__PAD)
    )


def one_hot_encode_entity_feature(entity: chex.Array, feature_index: int):
    max_value = ENTITY_MAX_VALUES[feature_index]
    return jax.nn.one_hot(entity[feature_index].clip(max=max_value), max_value)


def one_hot_encode_relative_edge_feature(edge: chex.Array, feature_index: int):
    max_value = RELATIVE_EDGE_MAX_VALUES[feature_index]
    return jax.nn.one_hot(edge[feature_index].clip(max=max_value), max_value)


def one_hot_encode_relative_edge_boost_feature(edge: chex.Array, feature_index: int):
    max_value = RELATIVE_EDGE_MAX_VALUES[feature_index] + 6
    return jax.nn.one_hot(edge[feature_index].clip(max=max_value), max_value)


def one_hot_encode_absolute_edge_feature(edge: chex.Array, feature_index: int):
    max_value = ABSOLUTE_EDGE_MAX_VALUES[feature_index]
    return jax.nn.one_hot(edge[feature_index].clip(max=max_value), max_value)


def astype(x: chex.Array, dtype: jnp.dtype) -> chex.Array:
    """
    Cast array `x` to a specified `dtype` if not already matching.
    """
    if x.dtype != dtype:
        return x.astype(dtype)
    else:
        return x


def get_entity_mask(entity: chex.Array) -> chex.Array:
    """
    Generate a mask to identify valid entities based on species tokens.
    """
    species_token = astype(entity[EntityFeature.ENTITY_FEATURE__SPECIES], jnp.int32)
    return ~(
        (species_token == SpeciesEnum.SPECIES_ENUM___NULL)
        | (species_token == SpeciesEnum.SPECIES_ENUM___PAD)
    )


def get_edge_mask(edge: chex.Array) -> chex.Array:
    """
    Generate a mask for edges based on their validity tokens.
    """
    return astype(edge[AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__VALID], jnp.int32)


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

            encodings = [
                SPECIES_ONEHOT(entity[EntityFeature.ENTITY_FEATURE__SPECIES]),
                ABILITY_ONEHOT(entity[EntityFeature.ENTITY_FEATURE__ABILITY]),
                ITEM_ONEHOT(entity[EntityFeature.ENTITY_FEATURE__ITEM]),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__LEVEL
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__ACTIVE
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__SIDE
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__HP_RATIO
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__GENDER
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__STATUS
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__ITEM_EFFECT
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__BEING_CALLED_BACK
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__TRAPPED
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__NEWLY_SWITCHED
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__TOXIC_TURNS
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__SLEEP_TURNS
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__FAINTED
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__BOOST_ATK_VALUE
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__BOOST_DEF_VALUE
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__BOOST_SPA_VALUE
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__BOOST_SPD_VALUE
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__BOOST_SPE_VALUE
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__BOOST_EVASION_VALUE
                ),
                one_hot_encode_entity_feature(
                    entity, EntityFeature.ENTITY_FEATURE__BOOST_ACCURACY_VALUE
                ),
                volatiles_encoding,
                typechange_encoding,
                feature_encode_entity(entity),
            ]

            embeddings = [
                species_embedding(entity[EntityFeature.ENTITY_FEATURE__SPECIES]),
                abilities_embedding(entity[EntityFeature.ENTITY_FEATURE__ABILITY]),
                items_embedding(entity[EntityFeature.ENTITY_FEATURE__ITEM]),
                actions_embedding(entity[EntityFeature.ENTITY_FEATURE__MOVEID0]),
                actions_embedding(entity[EntityFeature.ENTITY_FEATURE__MOVEID1]),
                actions_embedding(entity[EntityFeature.ENTITY_FEATURE__MOVEID2]),
                actions_embedding(entity[EntityFeature.ENTITY_FEATURE__MOVEID3]),
            ]

            embedding = entity_aggregate(
                encodings=[jnp.concatenate(encodings, axis=-1)], embeddings=embeddings
            )
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

            from_type_onehot = (
                one_hot_encode_relative_edge_feature(
                    relative_edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_TYPE_TOKEN0,
                )
                + one_hot_encode_relative_edge_feature(
                    relative_edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_TYPE_TOKEN1,
                )
                + one_hot_encode_relative_edge_feature(
                    relative_edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_TYPE_TOKEN2,
                )
                + one_hot_encode_relative_edge_feature(
                    relative_edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_TYPE_TOKEN3,
                )
                + one_hot_encode_relative_edge_feature(
                    relative_edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_TYPE_TOKEN4,
                )
            ) / relative_edge[
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_TYPES
            ].clip(
                min=1
            )

            # Aggregate embeddings for the relative edge.
            encodings = [
                ABILITY_ONEHOT(
                    relative_edge[
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ABILITY_TOKEN
                    ]
                ),
                ITEM_ONEHOT(
                    relative_edge[RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ITEM_TOKEN]
                ),
                minor_args_encoding,
                side_condition_encoding,
                one_hot_encode_relative_edge_feature(
                    relative_edge, RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MAJOR_ARG
                ),
                from_type_onehot,
                HEX_ENCODER(
                    relative_edge[
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__DAMAGE_RATIO
                    ]
                ),
                HEX_ENCODER(
                    relative_edge[RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__HEAL_RATIO]
                ),
                one_hot_encode_relative_edge_feature(
                    relative_edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__STATUS_TOKEN,
                ),
                one_hot_encode_relative_edge_boost_feature(
                    relative_edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_ATK_VALUE,
                ),
                one_hot_encode_relative_edge_boost_feature(
                    relative_edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_DEF_VALUE,
                ),
                one_hot_encode_relative_edge_boost_feature(
                    relative_edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_SPA_VALUE,
                ),
                one_hot_encode_relative_edge_boost_feature(
                    relative_edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_SPD_VALUE,
                ),
                one_hot_encode_relative_edge_boost_feature(
                    relative_edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_SPE_VALUE,
                ),
                one_hot_encode_relative_edge_boost_feature(
                    relative_edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_EVASION_VALUE,
                ),
                one_hot_encode_relative_edge_boost_feature(
                    relative_edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_ACCURACY_VALUE,
                ),
                one_hot_encode_relative_edge_feature(
                    relative_edge, RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SPIKES
                ),
                one_hot_encode_relative_edge_feature(
                    relative_edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__TOXIC_SPIKES,
                ),
            ]

            def _get_effect_embedding(feature_index: int):
                return jnp.where(
                    (relative_edge[feature_index] != EffectEnum.EFFECT_ENUM___NULL)[
                        None
                    ],
                    effects_embedding(relative_edge[feature_index]),
                    0,
                )

            edge_effect_embedding = (
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

            embeddings = [
                items_embedding(
                    relative_edge[RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ITEM_TOKEN]
                ),
                abilities_embedding(
                    relative_edge[
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ABILITY_TOKEN
                    ]
                ),
                actions_embedding(
                    relative_edge[
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ACTION_TOKEN
                    ]
                ),
                edge_effect_embedding,
            ]

            embedding = relative_edge_aggregate(
                encodings=[jnp.concatenate(encodings, axis=-1)], embeddings=embeddings
            )
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
            encodings = [
                one_hot_encode_absolute_edge_feature(
                    absolute_edge, AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_ID
                ),
                one_hot_encode_absolute_edge_feature(
                    absolute_edge,
                    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_MAX_DURATION,
                ),
                one_hot_encode_absolute_edge_feature(
                    absolute_edge,
                    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_MIN_DURATION,
                ),
                one_hot_encode_absolute_edge_feature(
                    absolute_edge, AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TERRAIN_ID
                ),
                one_hot_encode_absolute_edge_feature(
                    absolute_edge,
                    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TERRAIN_MAX_DURATION,
                ),
                one_hot_encode_absolute_edge_feature(
                    absolute_edge,
                    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TERRAIN_MIN_DURATION,
                ),
                one_hot_encode_absolute_edge_feature(
                    absolute_edge,
                    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__PSEUDOWEATHER_ID,
                ),
                one_hot_encode_absolute_edge_feature(
                    absolute_edge,
                    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__PSEUDOWEATHER_MAX_DURATION,
                ),
                one_hot_encode_absolute_edge_feature(
                    absolute_edge,
                    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__PSEUDOWEATHER_MIN_DURATION,
                ),
                OCT_ENCODER(turn),
                OCT_ENCODER(request_count),
                OCT_ENCODER(
                    absolute_edge[
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TURN_ORDER_VALUE
                    ]
                ),
            ]

            embedding = absolute_edge_aggregate(
                encodings=[jnp.concatenate(encodings, axis=-1)], embeddings=None
            )
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
        def _encode_action(move: chex.Array, legal: chex.Array) -> chex.Array:
            """
            Encode features of a move, including its type, species, and action ID.
            """
            encodings = [
                MOVE_ONEHOT(move[MovesetFeature.MOVESET_FEATURE__MOVE_ID]),
                SPECIES_ONEHOT(move[MovesetFeature.MOVESET_FEATURE__SPECIES_ID]),
                jnp.concatenate(
                    (
                        # jax.nn.one_hot(legal, 2),
                        jax.nn.one_hot(
                            move[MovesetFeature.MOVESET_FEATURE__ACTION_TYPE], 2
                        ),
                        jax.nn.one_hot(
                            move[MovesetFeature.MOVESET_FEATURE__PPUSED].clip(
                                min=0, max=31
                            ),
                            32,
                        ),
                        move[MovesetFeature.MOVESET_FEATURE__PPUSED][None],
                    ),
                    axis=-1,
                ),
            ]

            embeddings = [
                actions_embedding(move[MovesetFeature.MOVESET_FEATURE__ACTION_ID])
            ]

            embedding = action_aggregate(encodings=encodings, embeddings=embeddings)
            embedding = action_mlp(embedding)

            # Apply mask to the move embeddings.
            mask = get_move_mask(move)
            embedding = jnp.where(mask, embedding, 0)

            return embedding, mask

        entity_embeddings, valid_entity_mask = jax.vmap(_encode_entity)(
            env_step.team.reshape(-1, env_step.team.shape[-1])
        )

        timestep_embeddings, valid_timestep_mask = _encode_timesteps(
            history_step.major_history
        )

        action_embeddings, _ = jax.vmap(_encode_action)(
            env_step.moveset, env_step.legal.astype(int)
        )

        contextual_entity_embeddings = TransformerEncoder(
            **self.cfg.entity_encoder.to_dict()
        )(entity_embeddings, valid_entity_mask)

        contextual_timestep_embeddings = TransformerEncoder(
            **self.cfg.timestep_encoder.to_dict()
        )(timestep_embeddings, valid_timestep_mask)

        contextual_entity_embeddings = TransformerDecoder(
            **self.cfg.entity_timestep_decoder.to_dict()
        )(
            contextual_entity_embeddings,
            contextual_timestep_embeddings,
            valid_entity_mask,
            valid_timestep_mask,
        )

        contextual_action_embeddings = TransformerDecoder(
            **self.cfg.action_entity_decoder.to_dict()
        )(
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
