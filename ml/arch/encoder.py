import functools
from typing import Mapping, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from ml.arch.modules import (
    CrossTransformer,
    PretrainedEmbedding,
    Resnet,
    ToAvgVector,
    Transformer,
    VectorMerge,
)
from rlenv.data import (
    NUM_ABILITIES,
    NUM_EDGE_FROM_TYPES,
    NUM_EDGE_TYPES,
    NUM_EFFECTS,
    NUM_GENDERS,
    NUM_HISTORY,
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
from rlenv.interfaces import EnvStep
from rlenv.protos.enums_pb2 import MovesEnum, SideconditionEnum, SpeciesEnum
from rlenv.protos.features_pb2 import (
    EdgeTypes,
    FeatureEdge,
    FeatureEntity,
    FeatureMoveset,
    FeatureWeather,
)

SPECIES_ONEHOT = PretrainedEmbedding("data/data/gen3/randombattle/species.npy")
ABILITY_ONEHOT = PretrainedEmbedding("data/data/gen3/randombattle/abilities.npy")
ITEM_ONEHOT = PretrainedEmbedding("data/data/gen3/randombattle/items.npy")
MOVE_ONEHOT = PretrainedEmbedding("data/data/gen3/randombattle/moves.npy")


def get_move_mask(move: chex.Array) -> chex.Array:
    move_id_token = astype(move[FeatureMoveset.MOVEID], jnp.int32)
    return jnp.logical_or(
        jnp.not_equal(move_id_token, MovesEnum.MOVES__NULL),
        jnp.not_equal(move_id_token, MovesEnum.MOVES__PAD),
    )


class MoveEncoder(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, move: chex.Array, is_legal: bool):

        embeddings = [
            _encode_one_hot(move, FeatureMoveset.MOVEID, NUM_MOVES),
            MOVE_ONEHOT(move[FeatureMoveset.MOVEID]),
            _binary_scale_embedding(move[FeatureMoveset.PPUSED].astype(jnp.int32), 65),
        ]

        # Encoded one-hots (to pass to jax.nn.one_hot then nn.Dense):
        is_switch = move[FeatureMoveset.MOVEID] == MovesEnum.MOVES__SWITCH
        one_hot_encoded = [
            jax.nn.one_hot(is_switch.astype(jnp.int32), 2),
            jax.nn.one_hot(is_legal.astype(jnp.int32), 2),
        ]

        boolean_code = jnp.concatenate(one_hot_encoded, axis=-1)
        embeddings.append(astype(boolean_code, jnp.float32))
        mask = get_move_mask(move)
        embedding = sum([nn.Dense(self.cfg.entity_size)(x) for x in embeddings])
        return embedding, mask


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


def _encode_moveset_onehot(entity: chex.Array) -> chex.Array:
    """Select features in `rescales`, rescale and concatenate them."""
    chex.assert_rank(entity, 1)

    # Moves and PP encoding using JAX indexing
    move_indices = jax.lax.slice(
        entity, (FeatureEntity.ENTITY_MOVEID0,), (FeatureEntity.ENTITY_MOVEPP0,)
    )
    move_indices_onehot = jax.nn.one_hot(move_indices, NUM_MOVES)
    pp_indices = jax.lax.slice(
        entity,
        (FeatureEntity.ENTITY_MOVEPP0,),
        (FeatureEntity.ENTITY_HAS_STATUS,),
    )
    pp_indices_broadcasted = (
        jnp.expand_dims(pp_indices / 1023, axis=-1) * move_indices_onehot
    )
    return jnp.concatenate((move_indices_onehot, pp_indices_broadcasted), axis=-1).sum(
        0
    )


def get_entity_mask(entity: chex.Array) -> chex.Array:
    species_token = astype(entity[FeatureEntity.ENTITY_SPECIES], jnp.int32)
    return ~jnp.logical_or(
        jnp.equal(species_token, SpeciesEnum.SPECIES__NULL),
        jnp.equal(species_token, SpeciesEnum.SPECIES__PAD),
    )


def get_edge_mask(edge: chex.Array) -> chex.Array:
    edge_type_token = astype(edge[FeatureEdge.EDGE_TYPE_TOKEN], jnp.int32)
    return jnp.not_equal(edge_type_token, EdgeTypes.EDGE_TYPE_NONE)


class EntityEncoder(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, entity: chex.Array):

        # Embeddings (to feed to nn.Dense modules):
        embeddings = [
            _encode_one_hot(entity, FeatureEntity.ENTITY_SPECIES, NUM_SPECIES),
            _encode_one_hot(entity, FeatureEntity.ENTITY_ABILITY, NUM_ABILITIES),
            _encode_one_hot(entity, FeatureEntity.ENTITY_ITEM, NUM_ITEMS),
            SPECIES_ONEHOT(entity[FeatureEntity.ENTITY_SPECIES]),
            ABILITY_ONEHOT(entity[FeatureEntity.ENTITY_ABILITY]),
            ITEM_ONEHOT(entity[FeatureEntity.ENTITY_ITEM]),
            _encode_moveset_onehot(entity),
            _binary_scale_embedding(entity[FeatureEntity.ENTITY_LEVEL], 100),
            _binary_scale_embedding(entity[FeatureEntity.ENTITY_HP_TOKEN], 1023),
            _encode_volatiles_onehot(entity),
            _features_embedding(
                entity,
                {
                    FeatureEntity.ENTITY_BOOST_ATK_VALUE: 1 / 2,
                    FeatureEntity.ENTITY_BOOST_DEF_VALUE: 1 / 2,
                    FeatureEntity.ENTITY_BOOST_SPA_VALUE: 1 / 2,
                    FeatureEntity.ENTITY_BOOST_SPD_VALUE: 1 / 2,
                    FeatureEntity.ENTITY_BOOST_SPE_VALUE: 1 / 2,
                    FeatureEntity.ENTITY_BOOST_EVASION_VALUE: 1 / 2,
                    FeatureEntity.ENTITY_BOOST_ACCURACY_VALUE: 1 / 2,
                    FeatureEntity.ENTITY_LEVEL: 1 / 100,
                    FeatureEntity.ENTITY_HP_TOKEN: 1 / 1023,
                },
            ),
        ]

        # Encoded one-hots (to pass to jax.nn.one_hot then nn.Dense):
        one_hot_encoded = [
            _encode_one_hot(entity, FeatureEntity.ENTITY_GENDER, NUM_GENDERS),
            _encode_one_hot(entity, FeatureEntity.ENTITY_STATUS, NUM_STATUS),
            _encode_one_hot(entity, FeatureEntity.ENTITY_ITEM_EFFECT, NUM_ITEM_EFFECTS),
            _encode_one_hot(entity, FeatureEntity.ENTITY_BEING_CALLED_BACK, 2),
            _encode_one_hot(entity, FeatureEntity.ENTITY_TRAPPED, 2),
            _encode_one_hot(entity, FeatureEntity.ENTITY_NEWLY_SWITCHED, 2),
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
            _encode_boost_one_hot(entity, FeatureEntity.ENTITY_BOOST_ACCURACY_VALUE),
        ]

        boolean_code = jnp.concatenate(one_hot_encoded, axis=-1)
        embeddings.append(astype(boolean_code, jnp.float32))

        embedding = sum([nn.Dense(self.cfg.entity_size)(x) for x in embeddings])
        mask = get_entity_mask(entity)
        embedding = jnp.where(mask, embedding, 0)
        return embedding, mask


def _select_entity_embedding(
    entity_embeddings: chex.Array, entity_index: chex.Array
) -> chex.Array:
    selected_embedding = jnp.take(entity_embeddings, entity_index, axis=0)
    return jnp.where(entity_index >= 0, selected_embedding, 0)


class EdgeEncoder(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, edge: chex.Array, poke_embeddings: chex.Array) -> chex.Array:

        # Embeddings (to feed to nn.Dense modules):
        embeddings = [
            _select_entity_embedding(poke_embeddings, edge[FeatureEdge.POKE1_INDEX]),
            _select_entity_embedding(poke_embeddings, edge[FeatureEdge.POKE2_INDEX]),
            _encode_one_hot(edge, FeatureEdge.MOVE_TOKEN, NUM_MOVES),
            _encode_one_hot(edge, FeatureEdge.ITEM_TOKEN, NUM_ITEMS),
            _encode_one_hot(edge, FeatureEdge.ABILITY_TOKEN, NUM_ABILITIES),
            MOVE_ONEHOT(edge[FeatureEdge.MOVE_TOKEN]),
            ITEM_ONEHOT(edge[FeatureEdge.ITEM_TOKEN]),
            ABILITY_ONEHOT(edge[FeatureEdge.ABILITY_TOKEN]),
            _encode_one_hot(edge, FeatureEdge.MAJOR_ARG, NUM_MAJOR_ARGS),
            _encode_one_hot(edge, FeatureEdge.MINOR_ARG, NUM_MINOR_ARGS),
            _encode_one_hot(edge, FeatureEdge.FROM_SOURCE_TOKEN, NUM_EFFECTS),
            _encode_one_hot(edge, FeatureEdge.FROM_TYPE_TOKEN, NUM_EDGE_FROM_TYPES),
            _encode_one_hot(edge, FeatureEdge.EDGE_TYPE_TOKEN, NUM_EDGE_TYPES),
            # Healing
            _binary_scale_embedding((edge[FeatureEdge.DAMAGE_TOKEN]).clip(min=0), 1023),
            # Damage
            _binary_scale_embedding(
                jnp.abs(edge[FeatureEdge.DAMAGE_TOKEN]).clip(min=0), 1023
            ),
            _binary_scale_embedding(edge[FeatureEdge.TURN_ORDER_VALUE], 20),
            _binary_scale_embedding(
                edge[FeatureEdge.EDGE_AFFECTING_SIDE].astype(jnp.int32), 3
            ),
            _features_embedding(
                edge,
                {
                    FeatureEdge.DAMAGE_TOKEN: 1 / 1023,
                    FeatureEdge.BOOST_ATK_VALUE: 1 / 2,
                    FeatureEdge.BOOST_DEF_VALUE: 1 / 2,
                    FeatureEdge.BOOST_SPA_VALUE: 1 / 2,
                    FeatureEdge.BOOST_SPD_VALUE: 1 / 2,
                    FeatureEdge.BOOST_SPE_VALUE: 1 / 2,
                    FeatureEdge.BOOST_EVASION_VALUE: 1 / 2,
                    FeatureEdge.BOOST_ACCURACY_VALUE: 1 / 2,
                },
            ),
        ]

        # Encoded one-hots (to pass to jax.nn.one_hot then nn.Dense):
        one_hot_encoded = [
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
        embeddings.append(astype(boolean_code, jnp.float32))

        embedding = sum([nn.Dense(self.cfg.entity_size)(x) for x in embeddings])
        mask = get_edge_mask(edge)
        embedding = jnp.where(mask, embedding, 0)
        return embedding, mask


class SideEncoder(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, side: chex.Array) -> chex.Array:

        # Embeddings (to feed to nn.Dense modules):
        one_hot_encoded = [
            astype(side > 0, jnp.float32),
            _encode_one_hot(side, SideconditionEnum.SIDECONDITION_SPIKES, 4),
            _encode_one_hot(side, SideconditionEnum.SIDECONDITION_TOXICSPIKES, 3),
        ]
        boolean_code = jnp.concatenate(one_hot_encoded, axis=-1)

        return nn.Dense(self.cfg.entity_size)(boolean_code)


class FieldEncoder(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, field: chex.Array) -> chex.Array:

        # Embeddings (to feed to nn.Dense modules):
        one_hot_encoded = [
            _encode_one_hot(field, FeatureWeather.WEATHER_ID, NUM_WEATHER),
            _encode_one_hot(field, FeatureWeather.MAX_DURATION, 9),
            _encode_one_hot(field, FeatureWeather.MIN_DURATION, 9),
        ]
        boolean_code = jnp.concatenate(one_hot_encoded, axis=-1)

        return nn.Dense(self.cfg.entity_size)(boolean_code)


class Encoder(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, env_step: EnvStep) -> Tuple[chex.Array, chex.Array]:
        """
        Forward pass of the Encoder model. Processes an environment step and outputs
        the current state and action embeddings.
        """

        # Initialize the entity and edge encoders using configurations.
        entity_encoder = EntityEncoder(self.cfg.entity_encoder)
        edge_encoder = EdgeEncoder(self.cfg.edge_encoder)
        side_condition_encoder = SideEncoder(self.cfg.edge_encoder)
        field_encoder = FieldEncoder(self.cfg.edge_encoder)
        Transformer(**self.cfg.context_transformer.to_dict())
        Transformer(**self.cfg.context_transformer.to_dict())
        timestep_transformer = Transformer(**self.cfg.context_transformer.to_dict())
        node_entity_cross_transformer = CrossTransformer(
            **self.cfg.context_transformer.to_dict()
        )

        # Encode each timestep's nodes, edges, side conditions, and field data
        def _encode_timestep(
            nodes_per_turn: chex.Array,
            edges_per_turn: chex.Array,
            side_conditions_per_turn: chex.Array,
            field_per_turn: chex.Array,
        ):
            # Encode entities (nodes) for the current timestep, along with their masks
            entity_embeddings, entity_mask = jax.vmap(entity_encoder)(nodes_per_turn)
            # entity_embeddings = entity_transformer(entity_embeddings, entity_mask)

            # Encode edges for the current timestep, using entity embeddings as input
            edge_embeddings, edge_mask = jax.vmap(edge_encoder, in_axes=(0, None))(
                edges_per_turn, entity_embeddings
            )
            # edge_embeddings = edge_transformer(edge_embeddings, edge_mask)

            # Use a cross-transformer to contextualize the node and edge embeddings
            contextualized_nodes, contextualized_edges = node_entity_cross_transformer(
                entity_embeddings,
                edge_embeddings,
                entity_mask,
                edge_mask,
            )

            # Aggregate contextualized node and edge embeddings based on their respective masks
            aggregate = lambda embeddings, mask: jnp.einsum("ij,i->j", embeddings, mask)
            aggregate_nodes = aggregate(contextualized_nodes, entity_mask)
            aggregate_edges = aggregate(contextualized_edges, edge_mask)

            # Concatenate the aggregated node and edge embeddings into a single timestep vector
            timestep_embedding = jnp.concatenate(
                (aggregate_nodes, aggregate_edges), axis=-1
            )

            # Encode side conditions for the current timestep
            side_condition_embeddings = jax.vmap(side_condition_encoder)(
                side_conditions_per_turn
            )

            # Encode field data for the current timestep
            field_embedding = field_encoder(field_per_turn)

            # Merge timestep embeddings with side condition and field embeddings
            timestep_embedding = VectorMerge(**self.cfg.context_merge.to_dict())(
                timestep_embedding,
                side_condition_embeddings.reshape(
                    -1
                ),  # Flatten side condition embeddings
                field_embedding,  # Field embeddings
            )

            # Return the combined timestep embedding and a boolean indicating if the timestep has valid edges
            return timestep_embedding, entity_mask.sum() > 0

        # Encode the history of nodes and edges across multiple timesteps
        timestep_embeddings, valid_timestep_mask = jax.vmap(_encode_timestep)(
            env_step.history_nodes,
            env_step.history_edges,
            env_step.history_side_conditions,
            env_step.history_field,
        )
        positional_embeddings = nn.Embed(NUM_HISTORY, self.cfg.entity_size)(
            jnp.arange(NUM_HISTORY)
        )
        timestep_embeddings = timestep_transformer(
            timestep_embeddings
            + positional_embeddings,  # Add positional embeddings to timestep embeddings
            valid_timestep_mask,
        )

        # Encode private entities (team) and obtain valid masks for each entity
        private_entity_embeddings, valid_private_mask = jax.vmap(entity_encoder)(
            env_step.team.reshape(-1, env_step.team.shape[-1])
        )
        # private_entity_embeddings = entity_transformer(
        #     private_entity_embeddings, valid_private_mask
        # )

        # Apply the history transformer to combine public timestep embeddings with private entity embeddings
        history_transformer = CrossTransformer(**self.cfg.history_transformer.to_dict())
        private_entity_embeddings, _ = history_transformer(
            private_entity_embeddings,
            timestep_embeddings,
            valid_private_mask,
            valid_timestep_mask,
        )

        # Compute action embeddings by merging private entity embeddings with move embeddings
        move_encoder = MoveEncoder(self.cfg.move_encoder)
        action_merge = VectorMerge(**self.cfg.action_merge)

        # Helper function to compute action embeddings for each set of private entity embeddings and moveset
        def _compute_action_embeddings(
            current_entity_embeddings: chex.Array,
            moveset: chex.Array,
            legal_mask: chex.Array,
        ) -> chex.Array:
            """
            Computes action embeddings by merging private entity embeddings with move embeddings.
            """

            scaling_factor = self.param(
                "scaling_factor", lambda rng, shape: jnp.ones(shape) / 4, (1,)
            )

            # Repeat the first entity embedding and concatenate with the remaining entity embeddings
            active_embedding = current_entity_embeddings[:1]
            scaled_active = active_embedding * scaling_factor

            active_embedding = scaled_active + jax.lax.stop_gradient(
                active_embedding - scaled_active
            )

            repeated_embedding = jnp.tile(active_embedding, (4, 1))

            entity_embeddings = jnp.concatenate(
                (repeated_embedding, current_entity_embeddings),
                axis=0,
            )

            # Encode moves and merge them with the entity embeddings
            action_embeddings, action_mask = jax.vmap(move_encoder)(moveset, legal_mask)
            action_embeddings = jax.vmap(action_merge)(
                entity_embeddings, action_embeddings
            )
            return action_embeddings, action_mask

        # Compute action embeddings for each moveset
        action_embeddings, action_mask = _compute_action_embeddings(
            private_entity_embeddings[:6], env_step.moveset[0], env_step.legal
        )

        action_transformer = Transformer(**self.cfg.action_transformer.to_dict())
        action_embeddings = action_transformer(action_embeddings, action_mask)

        # Compute the current state by averaging private embeddings, then passing through a ResNet
        average_embedding = ToAvgVector(**self.cfg.to_vector.to_dict())(
            private_entity_embeddings,
            valid_private_mask,
        )

        def _compute_state_embedding(
            average_entity_embedding: chex.Array, env_step: EnvStep
        ) -> chex.Array:
            turn_value = jnp.clip(env_step.turn, max=49)
            can_move_onehot = jax.nn.one_hot(
                env_step.legal[:4].any().astype(jnp.int32), 2
            )
            can_switch_onehot = jax.nn.one_hot(
                env_step.legal[4:].any().astype(jnp.int32), 2
            )
            boolean_code = jnp.concatenate(
                (can_move_onehot, can_switch_onehot), axis=-1
            )

            turn_embedding = nn.Embed(50, self.cfg.vector_size)
            onehot_linear = nn.Dense(features=self.cfg.vector_size)
            state_resnet = Resnet(**self.cfg.state_resnet)

            current_state = (
                average_entity_embedding
                + turn_embedding(turn_value)
                + onehot_linear(boolean_code)
            )

            return state_resnet(current_state)

        current_state = _compute_state_embedding(average_embedding, env_step)

        return current_state, action_embeddings
