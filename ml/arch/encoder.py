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
    PretrainedEmbedding,
    Resnet,
    ToAvgVector,
    Transformer,
    VectorMerge,
)
from ml.func import cosine_similarity
from rlenv.data import (
    NUM_ABILITIES,
    NUM_EDGE_TYPES,
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
)
from rlenv.interfaces import EnvStep
from rlenv.protos.enums_pb2 import SpeciesEnum
from rlenv.protos.features_pb2 import (
    EdgeTypes,
    FeatureEdge,
    FeatureEntity,
    FeatureMoveset,
)


SPECIES_ONEHOT = PretrainedEmbedding("data/data/gen3/randombattle/species.npy")
ABILITY_ONEHOT = PretrainedEmbedding("data/data/gen3/randombattle/abilities.npy")
ITEM_ONEHOT = PretrainedEmbedding("data/data/gen3/randombattle/items.npy")
MOVE_ONEHOT = PretrainedEmbedding("data/data/gen3/randombattle/moves.npy")


class MoveEncoder(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, move: chex.Array):
        pp_left = move[FeatureMoveset.PPUSED]
        move_id = move[FeatureMoveset.MOVEID]
        pp_onehot = _binary_scale_embedding(pp_left.astype(jnp.int32), 65)
        move_onehot = jax.nn.one_hot(move_id, NUM_MOVES)
        move_embedding = nn.Dense(features=self.cfg.entity_size)(move_onehot)
        pp_embedding = nn.Dense(features=self.cfg.entity_size)(pp_onehot)
        return move_embedding + pp_embedding


def astype(x: chex.Array, dtype: jnp.dtype) -> chex.Array:
    """Cast x if necessary."""
    if x.dtype != dtype:
        return x.astype(dtype)
    else:
        return x


ENTITY_MAX_VALUES = {
    FeatureEntity.ENTITY_SPECIES: NUM_SPECIES,
    FeatureEntity.ENTITY_ABILITY: NUM_ABILITIES,
    FeatureEntity.ENTITY_ITEM: NUM_ITEMS,
    FeatureEntity.ENTITY_ACTIVE: 1,
    FeatureEntity.ENTITY_GENDER: NUM_GENDERS,
    FeatureEntity.ENTITY_STATUS: NUM_STATUS,
    FeatureEntity.ENTITY_BEING_CALLED_BACK: 2,
    FeatureEntity.ENTITY_TRAPPED: 2,
    FeatureEntity.ENTITY_NEWLY_SWITCHED: 2,
    FeatureEntity.ENTITY_TOXIC_TURNS: 8,
    FeatureEntity.ENTITY_SLEEP_TURNS: 4,
    FeatureEntity.ENTITY_FAINTED: 2,
    FeatureEntity.ENTITY_ACTIVE: 2,
    FeatureEntity.ENTITY_ITEM_EFFECT: NUM_ITEM_EFFECTS,
}

EDGE_MAX_VALUES = {
    FeatureEdge.MOVE_TOKEN: NUM_MOVES,
    FeatureEdge.ITEM_TOKEN: NUM_ITEMS,
    FeatureEdge.ABILITY_TOKEN: NUM_ABILITIES,
    FeatureEdge.STATUS_TOKEN: NUM_STATUS,
}


def _encode_one_hot(entity: chex.Array, feature_idx: int):
    chex.assert_rank(entity, 1)
    return entity[feature_idx], ENTITY_MAX_VALUES[feature_idx] + 1


def _encode_boost_one_hot(entity: chex.Array, feature_idx: int):
    chex.assert_rank(entity, 1)
    return entity[feature_idx] + 6, 13


def _encode_sqrt_one_hot(entity: chex.Array, feature_idx: int):
    chex.assert_rank(entity, 1)

    max_value = ENTITY_MAX_VALUES[feature_idx]
    max_sqrt_value = int(math.floor(math.sqrt(max_value)))
    x = jnp.floor(jnp.sqrt(astype(entity[feature_idx], jnp.float32)))
    x = jnp.minimum(astype(x, jnp.int32), max_sqrt_value)
    return x, max_sqrt_value + 1


def _encode_capped_one_hot(entity: chex.Array, feature_idx: int):
    chex.assert_rank(entity, 1)

    max_value = ENTITY_MAX_VALUES[feature_idx]
    return jnp.minimum(entity[feature_idx], max_value), max_value + 1


def _encode_divided_one_hot(entity: chex.Array, feature_idx: int, divisor: int):
    chex.assert_rank(entity, 1)

    max_value = ENTITY_MAX_VALUES[feature_idx]
    max_divided_value = max_value // divisor
    x = jnp.floor_divide(entity[feature_idx], divisor)
    x = jnp.minimum(x, max_divided_value)
    return x, max_divided_value + 1


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
    return astype(onehot.reshape(-1), jnp.float32)


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

        # Embeddings (to feed to hk.Linear modules):
        embeddings = [
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_SPECIES], NUM_SPECIES),
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_ABILITY], NUM_ABILITIES),
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_ITEM], NUM_ITEMS),
            _encode_moveset_onehot(entity),
            _binary_scale_embedding(entity[FeatureEntity.ENTITY_LEVEL], 100),
            _binary_scale_embedding(entity[FeatureEntity.ENTITY_HP_TOKEN], 1023),
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

        # Encoded one-hots (to pass to jax.nn.one_hot then hk.Linear):
        one_hot_encoded = [
            _encode_one_hot(entity, FeatureEntity.ENTITY_GENDER),
            _encode_one_hot(entity, FeatureEntity.ENTITY_STATUS),
            _encode_one_hot(entity, FeatureEntity.ENTITY_BEING_CALLED_BACK),
            _encode_one_hot(entity, FeatureEntity.ENTITY_TRAPPED),
            _encode_one_hot(entity, FeatureEntity.ENTITY_NEWLY_SWITCHED),
            _encode_one_hot(entity, FeatureEntity.ENTITY_TOXIC_TURNS),
            _encode_one_hot(entity, FeatureEntity.ENTITY_SLEEP_TURNS),
            _encode_one_hot(entity, FeatureEntity.ENTITY_FAINTED),
            _encode_one_hot(entity, FeatureEntity.ENTITY_ACTIVE),
            _encode_boost_one_hot(entity, FeatureEntity.ENTITY_BOOST_ATK_VALUE),
            _encode_boost_one_hot(entity, FeatureEntity.ENTITY_BOOST_DEF_VALUE),
            _encode_boost_one_hot(entity, FeatureEntity.ENTITY_BOOST_SPA_VALUE),
            _encode_boost_one_hot(entity, FeatureEntity.ENTITY_BOOST_SPD_VALUE),
            _encode_boost_one_hot(entity, FeatureEntity.ENTITY_BOOST_SPE_VALUE),
            _encode_boost_one_hot(entity, FeatureEntity.ENTITY_BOOST_EVASION_VALUE),
            _encode_boost_one_hot(entity, FeatureEntity.ENTITY_BOOST_ACCURACY_VALUE),
        ]

        sum_offsets = np.cumsum([0] + [offset for _, offset in one_hot_encoded])
        indices = jnp.stack(
            [
                idx + offset
                for (idx, _), offset in zip(one_hot_encoded, sum_offsets[:-1])
            ]
        )
        boolean_code = jnp.matmul(
            jnp.ones((len(indices),), jnp.float32),
            indices[:, jnp.newaxis] == jnp.arange(sum_offsets[-1]),
        )
        embeddings.append(astype(boolean_code, jnp.float32))

        embedding = sum([nn.Dense(self.cfg.entity_size)(x) for x in embeddings])
        mask = get_entity_mask(entity)
        embedding = jnp.where(mask, embedding, 0)
        return embedding, mask


def _select_entity_embedding(
    entity_embeddings: chex.Array, entity_index: chex.Array
) -> chex.Array:
    return jnp.where(
        entity_index >= 0,
        jnp.take(entity_embeddings, entity_index, axis=0),
        0,
    )


class EdgeEncoder(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, edge: chex.Array, poke_embeddings: chex.Array) -> chex.Array:

        # Embeddings (to feed to hk.Linear modules):
        embeddings = [
            _select_entity_embedding(poke_embeddings, edge[FeatureEdge.POKE1_INDEX]),
            _select_entity_embedding(poke_embeddings, edge[FeatureEdge.POKE2_INDEX]),
            jax.nn.one_hot(edge[FeatureEdge.MOVE_TOKEN], NUM_MOVES),
            jax.nn.one_hot(edge[FeatureEdge.ITEM_TOKEN], NUM_ITEMS),
            jax.nn.one_hot(edge[FeatureEdge.ABILITY_TOKEN], NUM_ABILITIES),
            jax.nn.one_hot(edge[FeatureEdge.MAJOR_ARG], NUM_MAJOR_ARGS),
            jax.nn.one_hot(edge[FeatureEdge.MINOR_ARG], NUM_MINOR_ARGS),
            jax.nn.one_hot(edge[FeatureEdge.EDGE_TYPE_TOKEN], NUM_EDGE_TYPES),
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

        # Encoded one-hots (to pass to jax.nn.one_hot then hk.Linear):
        one_hot_encoded = [
            (edge[FeatureEdge.STATUS_TOKEN], NUM_STATUS + 1),
            _encode_boost_one_hot(edge, FeatureEdge.BOOST_ATK_VALUE),
            _encode_boost_one_hot(edge, FeatureEdge.BOOST_DEF_VALUE),
            _encode_boost_one_hot(edge, FeatureEdge.BOOST_SPA_VALUE),
            _encode_boost_one_hot(edge, FeatureEdge.BOOST_SPD_VALUE),
            _encode_boost_one_hot(edge, FeatureEdge.BOOST_SPE_VALUE),
            _encode_boost_one_hot(edge, FeatureEdge.BOOST_EVASION_VALUE),
            _encode_boost_one_hot(edge, FeatureEdge.BOOST_ACCURACY_VALUE),
        ]

        sum_offsets = np.cumsum([0] + [offset for _, offset in one_hot_encoded])
        indices = jnp.stack(
            [
                idx + offset
                for (idx, _), offset in zip(one_hot_encoded, sum_offsets[:-1])
            ]
        )
        boolean_code = jnp.matmul(
            jnp.ones((len(indices),), jnp.float32),
            indices[:, jnp.newaxis] == jnp.arange(sum_offsets[-1]),
        )
        embeddings.append(astype(boolean_code, jnp.float32))

        embedding = sum([nn.Dense(self.cfg.entity_size)(x) for x in embeddings])
        mask = get_edge_mask(edge)
        embedding = jnp.where(mask, embedding, 0)

        return embedding, mask


class Encoder(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, env_step: EnvStep) -> Tuple[chex.Array, chex.Array]:
        """
        Forward pass of the Encoder model, processing an environment step to produce
        the current state and action embeddings.
        """

        entity_encoder = EntityEncoder(self.cfg.entity_encoder)
        edge_encoder = EdgeEncoder(self.cfg.edge_encoder)

        nodes = env_step.history_nodes
        edges = env_step.history_edges

        def _encode_timestep(nodes_per_turn: chex.Array, edges_per_turn: chex.Array):
            entity_embeddings, entity_mask = jax.vmap(entity_encoder)(nodes_per_turn)
            edge_embeddings, edge_mask = jax.vmap(edge_encoder, in_axes=(0, None))(
                edges_per_turn, entity_embeddings
            )
            node_context_transformer = Transformer(
                **self.cfg.context_transformer.to_dict()
            )
            edge_context_transformer = Transformer(
                **self.cfg.context_transformer.to_dict()
            )
            contextualized_nodes = node_context_transformer(
                entity_embeddings, edge_embeddings, entity_mask, edge_mask
            )
            contextualized_edges = edge_context_transformer(
                edge_embeddings, entity_embeddings, edge_mask, entity_mask
            )
            aggregate = lambda embeddings, mask: jnp.einsum("ij,i->j", embeddings, mask)
            return (
                aggregate(contextualized_nodes, entity_mask),
                aggregate(contextualized_edges, edge_mask),
                edge_mask.sum() > 0,
            )

        node_embedding, edge_embedding, valid_public_mask = jax.vmap(_encode_timestep)(
            nodes, edges
        )

        # Private entity embeddings and valid mask
        private_entity_embeddings, valid_private_mask = jax.vmap(
            jax.vmap(entity_encoder)
        )(env_step.team)

        public_embeddings = jnp.concatenate((node_embedding, edge_embedding), axis=-1)

        # Apply context transformer to merge public and private entity embeddings
        history_transformer = Transformer(**self.cfg.history_transformer.to_dict())
        private_entity_embeddings = history_transformer(
            private_entity_embeddings,
            public_embeddings,
            valid_private_mask,
            valid_public_mask,
        )

        # Compute the current state using the updated private embeddings and valid mask
        average_embedding = ToAvgVector(**self.cfg.to_vector.to_dict())(
            private_entity_embeddings.reshape(-1, private_entity_embeddings.shape[-1]),
            valid_private_mask.reshape(-1),
        )
        current_state = Resnet(**self.cfg.state_resnet)(average_embedding) + nn.Embed(
            50, self.cfg.vector_size
        )(jnp.clip(env_step.turn, max=49))

        # Compute action embeddings by merging private entity embeddings with move embeddings
        move_encoder = MoveEncoder(self.cfg.move_encoder)
        action_merge = VectorMerge(**self.cfg.action_merge)

        def _compute_action_embeddings(
            current_entity_embeddings: chex.Array, moveset: chex.Array
        ) -> chex.Array:
            """
            Computes action embeddings by merging private entity embeddings and move embeddings.
            """
            # Concatenate private entity embeddings
            entity_embeddings = jnp.concatenate(
                (
                    jnp.tile(current_entity_embeddings[:1], (4, 1)),
                    current_entity_embeddings,
                ),
                axis=0,
            )

            # Encode moves and merge them with entity embeddings
            action_embeddings = jax.vmap(move_encoder)(moveset)
            action_embeddings = jax.vmap(action_merge)(
                entity_embeddings, action_embeddings
            )
            return action_embeddings

        action_embeddings = jax.vmap(_compute_action_embeddings)(
            private_entity_embeddings, env_step.moveset
        )
        action_transformer = Transformer(**self.cfg.action_transformer.to_dict())
        action_embeddings = action_transformer(
            action_embeddings[0],
            action_embeddings[1],
            env_step.legal,
            jnp.ones_like(env_step.legal),
        )

        return current_state, action_embeddings
