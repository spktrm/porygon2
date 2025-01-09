from typing import Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from ml.arch.modules import (
    BinaryEncoder,
    MergeEmbeddings,
    PretrainedEmbedding,
    SumEmbeddings,
    TransformerDecoder,
    TransformerEncoder,
    activation_fn,
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

SPECIES_ONEHOT = PretrainedEmbedding(fpath="data/data/gen3/species.npy")
ABILITY_ONEHOT = PretrainedEmbedding(fpath="data/data/gen3/abilities.npy")
ITEM_ONEHOT = PretrainedEmbedding(fpath="data/data/gen3/items.npy")
MOVE_ONEHOT = PretrainedEmbedding(fpath="data/data/gen3/moves.npy")

HEX_ENCODER = BinaryEncoder(num_bits=16)


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


def get_entity_mask(entity: chex.Array) -> chex.Array:
    species_token = astype(entity[FeatureEntity.ENTITY_SPECIES], jnp.int32)
    return ~jnp.logical_or(
        jnp.equal(species_token, SpeciesEnum.SPECIES__NULL),
        jnp.equal(species_token, SpeciesEnum.SPECIES__PAD),
    )


def get_edge_mask(edge: chex.Array) -> chex.Array:
    return astype(edge[FeatureAbsoluteEdge.EDGE_VALID], jnp.int32)


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
        entity_size = self.cfg.entity_size

        species_embedding = nn.Embed(NUM_SPECIES, entity_size)
        abilities_embedding = nn.Embed(NUM_ABILITIES, entity_size)
        items_embedding = nn.Embed(NUM_ITEMS, entity_size)
        actions_embedding = nn.Embed(NUM_ACTIONS, entity_size)
        effects_embedding = nn.Embed(NUM_EFFECTS, entity_size)
        side_embedding = nn.Embed(2, entity_size)

        entity_aggregate = SumEmbeddings(entity_size)
        relative_edge_aggregate = SumEmbeddings(entity_size)
        absolute_edge_aggregate = SumEmbeddings(entity_size)
        timestep_aggregate = MergeEmbeddings(entity_size)
        action_aggregate = SumEmbeddings(entity_size)

        def _encode_entity(entity: chex.Array) -> chex.Array:
            # Encoded one-hots (to pass to jax.nn.one_hot then nn.Dense):

            moveset_embedding = actions_embedding(
                entity[FeatureEntity.ENTITY_MOVEID0 : FeatureEntity.ENTITY_MOVEID3 + 1]
            ).sum(0)

            volatiles_indices = entity[
                FeatureEntity.ENTITY_VOLATILES0 : FeatureEntity.ENTITY_VOLATILES8 + 1
            ]
            volatiles_embedding = HEX_ENCODER(
                volatiles_indices.astype(jnp.uint16)
            ).reshape(-1)

            typechange_indices = entity[
                FeatureEntity.ENTITY_TYPECHANGE0 : FeatureEntity.ENTITY_TYPECHANGE1 + 1
            ]
            typechange_embedding = HEX_ENCODER(
                typechange_indices.astype(jnp.uint16)
            ).reshape(-1)

            one_hot_encoded = one_hot_encode_entity(entity)

            embeddings = [
                SPECIES_ONEHOT(entity[FeatureEntity.ENTITY_SPECIES]),
                ABILITY_ONEHOT(entity[FeatureEntity.ENTITY_ABILITY]),
                ITEM_ONEHOT(entity[FeatureEntity.ENTITY_ITEM]),
                activation_fn(species_embedding(entity[FeatureEntity.ENTITY_SPECIES])),
                activation_fn(
                    abilities_embedding(entity[FeatureEntity.ENTITY_ABILITY])
                ),
                activation_fn(items_embedding(entity[FeatureEntity.ENTITY_ITEM])),
                activation_fn(side_embedding(entity[FeatureEntity.ENTITY_SIDE])),
                activation_fn(moveset_embedding / 4),
                one_hot_encoded,
                feature_encode_entity(entity),
                volatiles_embedding,
                typechange_embedding,
            ]

            embedding = entity_aggregate(embeddings)
            mask = get_entity_mask(entity)
            embedding = jnp.where(mask, embedding, 0)
            return embedding, mask

        def _encode_relative_edge(relative_edge: chex.Array) -> chex.Array:
            # Embeddings (to feed to nn.Dense modules):

            minor_args_indices = relative_edge[
                FeatureRelativeEdge.EDGE_MINOR_ARG0 : FeatureRelativeEdge.EDGE_MINOR_ARG3
                + 1
            ]
            minor_args_embedding = HEX_ENCODER(
                minor_args_indices.astype(jnp.uint16)
            ).reshape(-1)

            side_condition_indices = relative_edge[
                FeatureRelativeEdge.EDGE_SIDECONDITIONS0 : FeatureRelativeEdge.EDGE_SIDECONDITIONS1
                + 1
            ]
            side_condition_embedding = HEX_ENCODER(
                side_condition_indices.astype(jnp.uint16)
            ).reshape(-1)

            one_hot_encoded = one_hot_encode_relative_edge(relative_edge)

            embeddings = [
                ABILITY_ONEHOT(relative_edge[FeatureRelativeEdge.EDGE_ABILITY_TOKEN]),
                ITEM_ONEHOT(relative_edge[FeatureRelativeEdge.EDGE_ITEM_TOKEN]),
                activation_fn(
                    items_embedding(relative_edge[FeatureRelativeEdge.EDGE_ITEM_TOKEN])
                ),
                activation_fn(
                    abilities_embedding(
                        relative_edge[FeatureRelativeEdge.EDGE_ABILITY_TOKEN]
                    )
                ),
                activation_fn(
                    actions_embedding(
                        relative_edge[FeatureRelativeEdge.EDGE_ACTION_TOKEN]
                    )
                ),
                activation_fn(
                    effects_embedding(
                        relative_edge[FeatureRelativeEdge.EDGE_FROM_SOURCE_TOKEN]
                    )
                ),
                minor_args_embedding,
                side_condition_embedding,
                one_hot_encoded,
            ]

            embedding = relative_edge_aggregate(embeddings)

            mask = get_edge_mask(relative_edge)
            embedding = jnp.where(mask, embedding, 0)
            return embedding, mask

        def _encode_absolute_edge(
            absolute_edge: chex.Array,
            turn_offset: chex.Array,
            request_count_offset: chex.Array,
        ) -> chex.Array:

            turn = jnp.abs(
                absolute_edge[FeatureAbsoluteEdge.EDGE_TURN_VALUE] - turn_offset
            )
            request_count = jnp.abs(
                absolute_edge[FeatureAbsoluteEdge.EDGE_REQUEST_COUNT]
                - request_count_offset
            )

            embeddings = [
                one_hot_encode_absolute_edge(absolute_edge),
                HEX_ENCODER(turn),
                HEX_ENCODER(request_count),
                HEX_ENCODER(absolute_edge[FeatureAbsoluteEdge.EDGE_TURN_ORDER_VALUE]),
            ]

            embedding = absolute_edge_aggregate(embeddings)

            mask = get_edge_mask(absolute_edge)
            embedding = jnp.where(mask, embedding, 0)
            return embedding, mask

        # Encode each timestep's nodes, edges, side conditions, and field data
        def _encode_timestep(
            history_container: HistoryContainer,
            turn_offset: chex.Array,
            request_count_offset: chex.Array,
        ):
            # Encode nodes (entities) and generate masks
            entity_embeddings, _ = jax.vmap(_encode_entity)(history_container.entities)

            # Encode edges, incorporating entity embeddings
            relative_edge_embeddings, _ = jax.vmap(_encode_relative_edge)(
                history_container.relative_edges
            )

            # Encode side conditions and field data
            absolute_edge_embedding, edge_mask = _encode_absolute_edge(
                history_container.absolute_edges, turn_offset, request_count_offset
            )

            # Merge aggregated embeddings with timestep context

            timestep_embeddings = [
                entity_embeddings[0],
                entity_embeddings[1],
                relative_edge_embeddings[0],
                relative_edge_embeddings[1],
                absolute_edge_embedding,
            ]
            timestep_embedding = timestep_aggregate(timestep_embeddings)
            timestep_embedding = jnp.where(edge_mask, timestep_embedding, 0)

            # Return combined timestep embedding and mask
            return timestep_embedding, edge_mask

        # Process history across timesteps
        def _encode_timesteps(history_container: HistoryContainer):
            turn_offset = history_container.absolute_edges[
                ..., FeatureAbsoluteEdge.EDGE_TURN_VALUE
            ].max(0)
            request_count_offset = history_container.absolute_edges[
                ..., FeatureAbsoluteEdge.EDGE_REQUEST_COUNT
            ].max(0)
            return jax.vmap(_encode_timestep, in_axes=(0, None, None))(
                history_container, turn_offset, request_count_offset
            )

        timestep_embeddings, valid_timestep_mask = _encode_timesteps(
            history_step.major_history
        )

        timestep_encoder = TransformerEncoder(
            **self.cfg.timestep_transformer_encoder.to_dict()
        )

        contextual_timestep_embeddings = timestep_encoder(
            timestep_embeddings, valid_timestep_mask
        )

        # Process private entities and generate masks
        entity_embeddings, valid_entity_mask = jax.vmap(_encode_entity)(
            env_step.team.reshape(-1, env_step.team.shape[-1])
        )

        entity_transformer_encoder = TransformerEncoder(
            **self.cfg.entity_transformer_encoder.to_dict()
        )

        contextual_entity_embeddings = entity_transformer_encoder(
            entity_embeddings, valid_entity_mask
        )

        entity_timestep_decoder = TransformerDecoder(
            **self.cfg.entity_timestep_transformer_decoder.to_dict()
        )

        contextual_entity_embeddings = entity_timestep_decoder(
            contextual_entity_embeddings,
            contextual_timestep_embeddings,
            valid_entity_mask,
            valid_timestep_mask,
        )

        # Compute action embeddings
        def _encode_move(move: chex.Array) -> chex.Array:
            # Encoded one-hots (to pass to jax.nn.one_hot then nn.Dense):
            one_hot_encoded = [
                SPECIES_ONEHOT(move[FeatureMoveset.MOVESET_SPECIES_ID]),
                MOVE_ONEHOT(move[FeatureMoveset.MOVESET_MOVE_ID]),
                jax.nn.one_hot(move[FeatureMoveset.MOVESET_ACTION_TYPE], 2),
                activation_fn(
                    actions_embedding(move[FeatureMoveset.MOVESET_ACTION_ID])
                ),
                jax.nn.one_hot(move[FeatureMoveset.MOVESET_PPUSED], 65),
            ]

            embedding = action_aggregate(one_hot_encoded)

            mask = get_move_mask(move)
            embedding = jnp.where(mask, embedding, 0)

            return embedding, mask

        action_embeddings, _ = jax.vmap(_encode_move)(env_step.moveset[0])

        contextual_action_embeddings = TransformerDecoder(
            **self.cfg.action_entity_transformer_decoder.to_dict()
        )(
            action_embeddings,
            contextual_entity_embeddings,
            env_step.legal,
            valid_entity_mask,
        )

        return contextual_action_embeddings
