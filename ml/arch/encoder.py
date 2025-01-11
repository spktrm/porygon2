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
    one_hot_encode_absolute_edge,
    one_hot_encode_entity,
    one_hot_encode_relative_edge,
)
from rlenv.data import NUM_ABILITIES, NUM_ACTIONS, NUM_EFFECTS, NUM_ITEMS, NUM_SPECIES
from rlenv.interfaces import EnvStep, HistoryContainer, HistoryStep
from rlenv.protos.enums_pb2 import ActionsEnum, SpeciesEnum
from rlenv.protos.features_pb2 import (
    FeatureAbsoluteEdge,
    FeatureEntity,
    FeatureMoveset,
    FeatureRelativeEdge,
)

# Load pretrained embeddings for various features.
SPECIES_ONEHOT = PretrainedEmbedding(fpath="data/data/gen3/species.npy")
ABILITY_ONEHOT = PretrainedEmbedding(fpath="data/data/gen3/abilities.npy")
ITEM_ONEHOT = PretrainedEmbedding(fpath="data/data/gen3/items.npy")
MOVE_ONEHOT = PretrainedEmbedding(fpath="data/data/gen3/moves.npy")

# Initialize a binary encoder for specific features.
HEX_ENCODER = BinaryEncoder(num_bits=16)


def get_move_mask(move: chex.Array) -> chex.Array:
    """
    Generate a mask to filter valid moves based on move identifiers.
    """
    move_id_token = astype(move[FeatureMoveset.MOVESET_ACTION_ID], jnp.int32)
    return (
        jnp.not_equal(move_id_token, ActionsEnum.ACTIONS_MOVE__NULL)
        & jnp.not_equal(move_id_token, ActionsEnum.ACTIONS_SWITCH__NULL)
        & jnp.not_equal(move_id_token, ActionsEnum.ACTIONS_MOVE__PAD)
        & jnp.not_equal(move_id_token, ActionsEnum.ACTIONS_SWITCH__PAD)
    )


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
    species_token = astype(entity[FeatureEntity.ENTITY_SPECIES], jnp.int32)
    return ~jnp.logical_or(
        jnp.equal(species_token, SpeciesEnum.SPECIES__NULL),
        jnp.equal(species_token, SpeciesEnum.SPECIES__PAD),
    )


def get_edge_mask(edge: chex.Array) -> chex.Array:
    """
    Generate a mask for edges based on their validity tokens.
    """
    return astype(edge[FeatureAbsoluteEdge.EDGE_VALID], jnp.int32)


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
            features=entity_size, embedding_init=jax.nn.initializers.lecun_normal()
        )

        # Initialize embeddings for various features.
        species_embedding = nn.Embed(NUM_SPECIES, **embed_kwargs)
        abilities_embedding = nn.Embed(NUM_ABILITIES, **embed_kwargs)
        items_embedding = nn.Embed(NUM_ITEMS, **embed_kwargs)
        actions_embedding = nn.Embed(NUM_ACTIONS, **embed_kwargs)
        effects_embedding = nn.Embed(NUM_EFFECTS, **embed_kwargs)
        side_embedding = nn.Embed(2, **embed_kwargs)

        # Initialize aggregation modules for combining feature embeddings.
        entity_aggregate = SumEmbeddings(entity_size)
        relative_edge_aggregate = SumEmbeddings(entity_size)
        absolute_edge_aggregate = SumEmbeddings(entity_size)
        timestep_aggregate = MLP((entity_size,))
        action_aggregate = SumEmbeddings(entity_size)

        def _encode_entity(entity: chex.Array) -> chex.Array:
            """
            Encode features of an entity, including species, ability, item, and moves.
            """
            # Process move-related features.
            moveset_embedding = actions_embedding(
                entity[FeatureEntity.ENTITY_MOVEID0 : FeatureEntity.ENTITY_MOVEID3 + 1]
            ).sum(0)

            # Encode volatile and type-change indices using the binary encoder.
            volatiles_indices = entity[
                FeatureEntity.ENTITY_VOLATILES0 : FeatureEntity.ENTITY_VOLATILES8 + 1
            ]
            volatiles_encoding = HEX_ENCODER(
                volatiles_indices.astype(jnp.uint16)
            ).reshape(-1)

            typechange_indices = entity[
                FeatureEntity.ENTITY_TYPECHANGE0 : FeatureEntity.ENTITY_TYPECHANGE1 + 1
            ]
            typechange_encoding = HEX_ENCODER(
                typechange_indices.astype(jnp.uint16)
            ).reshape(-1)

            # Perform one-hot encoding for the entity and aggregate embeddings.
            one_hot_encoded = one_hot_encode_entity(entity)
            embeddings = [
                SPECIES_ONEHOT(entity[FeatureEntity.ENTITY_SPECIES]),
                ABILITY_ONEHOT(entity[FeatureEntity.ENTITY_ABILITY]),
                ITEM_ONEHOT(entity[FeatureEntity.ENTITY_ITEM]),
                one_hot_encoded,
                feature_encode_entity(entity),
                volatiles_encoding,
                typechange_encoding,
            ]

            embedding = (
                entity_aggregate(embeddings)
                + species_embedding(entity[FeatureEntity.ENTITY_SPECIES])
                + abilities_embedding(entity[FeatureEntity.ENTITY_ABILITY])
                + items_embedding(entity[FeatureEntity.ENTITY_ITEM])
                + side_embedding(entity[FeatureEntity.ENTITY_SIDE])
                + moveset_embedding
            )

            # Apply mask to filter out invalid entities.
            mask = get_entity_mask(entity)
            embedding = jnp.where(mask, embedding, 0)

            return embedding, mask

        def _encode_relative_edge(relative_edge: chex.Array) -> chex.Array:
            """
            Encode features of a relative edge using various embeddings and encodings.
            """
            # Encode minor arguments and side conditions using the binary encoder.
            minor_args_indices = relative_edge[
                FeatureRelativeEdge.EDGE_MINOR_ARG0 : FeatureRelativeEdge.EDGE_MINOR_ARG3
                + 1
            ]
            minor_args_encoding = HEX_ENCODER(
                minor_args_indices.astype(jnp.uint16)
            ).reshape(-1)

            side_condition_indices = relative_edge[
                FeatureRelativeEdge.EDGE_SIDECONDITIONS0 : FeatureRelativeEdge.EDGE_SIDECONDITIONS1
                + 1
            ]
            side_condition_encoding = HEX_ENCODER(
                side_condition_indices.astype(jnp.uint16)
            ).reshape(-1)

            # Perform one-hot encoding for the relative edge.
            one_hot_encoded = one_hot_encode_relative_edge(relative_edge)

            # Aggregate embeddings for the relative edge.
            embeddings = [
                ABILITY_ONEHOT(relative_edge[FeatureRelativeEdge.EDGE_ABILITY_TOKEN]),
                ITEM_ONEHOT(relative_edge[FeatureRelativeEdge.EDGE_ITEM_TOKEN]),
                minor_args_encoding,
                side_condition_encoding,
                one_hot_encoded,
            ]

            embedding = (
                relative_edge_aggregate(embeddings)
                + items_embedding(relative_edge[FeatureRelativeEdge.EDGE_ITEM_TOKEN])
                + abilities_embedding(
                    relative_edge[FeatureRelativeEdge.EDGE_ABILITY_TOKEN]
                )
                + actions_embedding(
                    relative_edge[FeatureRelativeEdge.EDGE_ACTION_TOKEN]
                )
                + effects_embedding(
                    relative_edge[FeatureRelativeEdge.EDGE_FROM_SOURCE_TOKEN]
                )
            )

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
                absolute_edge[FeatureAbsoluteEdge.EDGE_TURN_VALUE] - turn_offset
            )
            request_count = jnp.abs(
                absolute_edge[FeatureAbsoluteEdge.EDGE_REQUEST_COUNT]
                - request_count_offset
            )

            # Aggregate embeddings for the absolute edge.
            embeddings = [
                one_hot_encode_absolute_edge(absolute_edge),
                HEX_ENCODER(turn),
                HEX_ENCODER(request_count),
                HEX_ENCODER(absolute_edge[FeatureAbsoluteEdge.EDGE_TURN_ORDER_VALUE]),
            ]

            embedding = absolute_edge_aggregate(embeddings)

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
                ]
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
                ..., FeatureAbsoluteEdge.EDGE_TURN_VALUE
            ].max(0)
            request_count_offset = history_container.absolute_edges[
                ..., FeatureAbsoluteEdge.EDGE_REQUEST_COUNT
            ].max(0)
            return jax.vmap(_encode_timestep, in_axes=(0, None, None))(
                history_container, turn_offset, request_count_offset
            )

        # Encode actions for the current environment step.
        def _encode_action(move: chex.Array, legal: chex.Array) -> chex.Array:
            """
            Encode features of a move, including its type, species, and action ID.
            """
            one_hot_encoded = [
                MOVE_ONEHOT(move[FeatureMoveset.MOVESET_MOVE_ID]),
                SPECIES_ONEHOT(move[FeatureMoveset.MOVESET_SPECIES_ID]),
                jnp.concatenate(
                    (
                        # jax.nn.one_hot(legal, 2),
                        jax.nn.one_hot(move[FeatureMoveset.MOVESET_ACTION_TYPE], 2),
                        jax.nn.one_hot(
                            move[FeatureMoveset.MOVESET_PPUSED].clip(min=0, max=31), 32
                        ),
                        move[FeatureMoveset.MOVESET_PPUSED][None],
                    ),
                    axis=-1,
                ),
            ]

            embedding = action_aggregate(one_hot_encoded) + actions_embedding(
                move[FeatureMoveset.MOVESET_ACTION_ID]
            )

            # Apply mask to the move embeddings.
            mask = get_move_mask(move)
            embedding = jnp.where(mask, embedding, 0)

            return embedding, mask

        # Process entities in the environment step.
        entity_embeddings, valid_entity_mask = jax.vmap(_encode_entity)(
            env_step.team.reshape(-1, env_step.team.shape[-1])
        )

        # Process timestep embeddings from history.
        timestep_embeddings, valid_timestep_mask = _encode_timesteps(
            history_step.major_history
        )

        # Process action embeddings from the environment step.
        action_embeddings, _ = jax.vmap(_encode_action)(
            env_step.moveset[0], env_step.legal.astype(int)
        )

        # Compute contextual embeddings for timesteps using a transformer.
        contextual_timestep_embeddings = TransformerEncoder(
            **self.cfg.timestep_transformer_encoder.to_dict()
        )(timestep_embeddings, valid_timestep_mask)

        # Compute contextual embeddings for entities using a transformer.
        contextual_entity_embeddings = TransformerEncoder(
            **self.cfg.entity_transformer_encoder.to_dict()
        )(entity_embeddings, valid_entity_mask)

        # Decode entity embeddings using timestep embeddings for additional context.
        contextual_entity_embeddings = TransformerDecoder(
            **self.cfg.entity_timestep_transformer_decoder.to_dict()
        )(
            contextual_entity_embeddings,
            contextual_timestep_embeddings,
            valid_entity_mask,
            valid_timestep_mask,
        )

        # Decode action embeddings using entity embeddings for context.
        contextual_action_embeddings = TransformerDecoder(
            **self.cfg.action_entity_transformer_decoder.to_dict()
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
