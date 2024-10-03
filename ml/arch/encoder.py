from re import L
import jax
import chex
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from ml_collections import ConfigDict
from functools import partial

from ml.arch.modules import (
    MLP,
    PretrainedEmbedding,
    Resnet,
    ToAvgVector,
    Transformer,
    VectorMerge,
)

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
    FeatureAdditionalInformation,
    FeatureEdge,
    FeatureEntity,
    FeatureMoveset,
)


def _binary_scale_embedding(to_encode: chex.Array, world_dim: int) -> chex.Array:
    """Encode the feature using its binary representation."""
    chex.assert_type(to_encode, jnp.int32)
    num_bits = (world_dim - 1).bit_length()
    bit_mask = 1 << np.arange(num_bits)
    pos = jnp.broadcast_to(to_encode[jnp.newaxis], num_bits)
    result = jnp.not_equal(jnp.bitwise_and(pos, bit_mask), 0)
    return result.astype(jnp.float32)


SPECIES_ONEHOT = PretrainedEmbedding("data/data/gen3/randombattle/species.npy")
ABILITY_ONEHOT = PretrainedEmbedding("data/data/gen3/randombattle/abilities.npy")
ITEM_ONEHOT = PretrainedEmbedding("data/data/gen3/randombattle/items.npy")
MOVE_ONEHOT = PretrainedEmbedding("data/data/gen3/randombattle/moves.npy")


class MoveEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.embed_moves = nn.Embed(NUM_MOVES, features=self.cfg.entity_size)
        self.move_linear = nn.Dense(features=self.cfg.entity_size)
        self.pp_linear = nn.Dense(features=self.cfg.entity_size)

    def encode_move(self, move: chex.Array):
        pp_left = move[FeatureMoveset.PPUSED]
        move_id = move[FeatureMoveset.MOVEID]
        pp_onehot = _binary_scale_embedding(pp_left.astype(jnp.int32), 65)
        move_onehot = self.embed_moves(move_id)
        embedding = self.move_linear(move_onehot)
        return embedding + self.pp_linear(pp_onehot)

    def __call__(self, movesets: chex.Array):
        _encode = jax.vmap(self.encode_move)
        return _encode(movesets)


class EntityEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        entity_size = self.cfg.entity_size

        self.embed_species = nn.Embed(NUM_SPECIES, features=entity_size)
        self.embed_ability = nn.Embed(NUM_ABILITIES, features=entity_size)
        self.embed_item = nn.Embed(NUM_ITEMS, features=entity_size)
        self.embed_moves = nn.Embed(NUM_MOVES, features=entity_size)

        self.item_linear = nn.Dense(features=entity_size)
        self.moves_linear = nn.Dense(features=entity_size)
        self.onehot_linear = nn.Dense(features=entity_size)
        self.side_linear = nn.Dense(features=entity_size)

        self.level_linear = nn.Dense(features=entity_size)
        self.hp_linear = nn.Dense(features=entity_size)
        self.active_linear = nn.Dense(features=entity_size)
        self.volatiles_linear = nn.Dense(features=entity_size)
        self.boosts_linear = nn.Dense(features=entity_size)

        self.output = nn.Dense(features=entity_size)

    def encode_hp(self, entity: chex.Array):
        hp = entity[FeatureEntity.ENTITY_HP]
        maxhp = jnp.clip(entity[FeatureEntity.ENTITY_MAXHP], a_min=1)
        return jnp.clip(hp / maxhp, a_min=0, a_max=1)

    def encode_moveset(self, entity: chex.Array):
        # Moves and PP encoding using JAX indexing
        move_indices = jax.lax.slice(
            entity, (FeatureEntity.ENTITY_MOVEID0,), (FeatureEntity.ENTITY_MOVEPP0,)
        )
        pp_indices = jax.lax.slice(
            entity, (FeatureEntity.ENTITY_MOVEPP0,), (FeatureEntity.ENTITY_HAS_STATUS,)
        )
        # Use jax.vmap to handle moves and PP encoding for batch efficiency
        return jnp.concatenate(
            (
                jax.vmap(self.embed_moves)(move_indices),
                jax.vmap(partial(_binary_scale_embedding, world_dim=64))(
                    pp_indices.astype(jnp.int32)
                ),
            ),
            axis=-1,
        )

    def encode_boosts(self, entity: chex.Array):
        boosts = jax.lax.slice(
            entity,
            (FeatureEntity.ENTITY_BOOST_ATK_VALUE,),
            (FeatureEntity.ENTITY_VOLATILES0,),
        )
        return boosts / 2.0

    def encode_item(self, entity: chex.Array):
        return jnp.concatenate(
            (
                self.embed_item(entity[FeatureEntity.ENTITY_ITEM]),
                jax.nn.one_hot(
                    entity[FeatureEntity.ENTITY_ITEM_EFFECT], NUM_ITEM_EFFECTS
                ),
            )
        )

    def encode_volatiles(self, entity: chex.Array):
        volatiles = jax.lax.slice(
            entity,
            (FeatureEntity.ENTITY_VOLATILES0,),
            (FeatureEntity.ENTITY_VOLATILES8 + 1,),
        )
        bitmask = jnp.arange(16)
        onehot = (volatiles[..., None] & bitmask) > 0
        return jax.lax.slice(
            onehot.flatten().astype(jnp.float32), (0,), (NUM_VOLATILE_STATUS,)
        )

    def encode_entity(self, entity: chex.Array):
        hp_ratio = self.encode_hp(entity)
        hp_token = jnp.floor(1023 * hp_ratio).astype(jnp.int32)

        # Embeddings for basic features (using one-hot encodings)
        basic_embeddings = [
            hp_ratio[jnp.newaxis],
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_GENDER], NUM_GENDERS),
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_STATUS], NUM_STATUS),
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_BEING_CALLED_BACK], 2),
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_TRAPPED], 2),
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_NEWLY_SWITCHED], 2),
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_TOXIC_TURNS], 8),
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_SLEEP_TURNS], 4),
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_FAINTED], 2),
        ]

        # Concatenate boolean features into a single vector
        boolean_code = jnp.concatenate(basic_embeddings, axis=0)

        item_onehot = self.encode_item(entity)
        boosts = self.encode_boosts(entity)
        moveset_onehot = self.encode_moveset(entity)
        volatiles_onehot = self.encode_volatiles(entity)

        # Final embedding combination (linear layers applied to encoded features)
        embedding = (
            self.hp_linear(_binary_scale_embedding(hp_token, 1024))
            + self.level_linear(
                _binary_scale_embedding(
                    entity[FeatureEntity.ENTITY_LEVEL].astype(jnp.int32), 101
                )
            )
            + self.active_linear(jax.nn.one_hot(entity[FeatureEntity.ENTITY_ACTIVE], 2))
            + self.onehot_linear(boolean_code.astype(jnp.float32))
            + self.boosts_linear(boosts)
            + self.volatiles_linear(volatiles_onehot)
            + self.embed_species(entity[FeatureEntity.ENTITY_SPECIES])
            + self.embed_ability(entity[FeatureEntity.ENTITY_ABILITY])
            + self.item_linear(item_onehot)
            + self.side_linear(jax.nn.one_hot(entity[FeatureEntity.ENTITY_SIDE], 2))
            + jax.vmap(self.moves_linear)(moveset_onehot).sum(axis=0)  # Sum over moves
        )

        return embedding

    def batch_encode(self, entities: chex.Array) -> chex.Array:
        func = jax.vmap(jax.vmap(self.encode_entity))
        return func(entities)

    def __call__(self, entities: chex.Array):
        return self.batch_encode(entities)


class EdgeEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        entity_size = self.cfg.entity_size

        self.linear_pokemon1 = nn.Dense(entity_size)
        self.linear_pokemon2 = nn.Dense(entity_size)

        # Embedding layers for categorical tokens
        self.embedding_move = nn.Embed(num_embeddings=NUM_MOVES, features=entity_size)
        self.embedding_item = nn.Embed(num_embeddings=NUM_ITEMS, features=entity_size)
        self.embedding_ability = nn.Embed(
            num_embeddings=NUM_ABILITIES, features=entity_size
        )
        self.embedding_status = nn.Embed(
            num_embeddings=NUM_STATUS, features=entity_size
        )
        self.embedding_edge_type = nn.Embed(
            num_embeddings=NUM_EDGE_TYPES, features=entity_size
        )
        self.embedding_major_arg = nn.Embed(
            num_embeddings=NUM_MAJOR_ARGS, features=entity_size
        )
        self.embedding_minor_arg = nn.Embed(
            num_embeddings=NUM_MINOR_ARGS, features=entity_size
        )
        self.embedding_turn_order = nn.Embed(20, entity_size)

        # Linear layers for numeric features
        self.linear_boosts = nn.Dense(entity_size)

        self.linear_side_affecting = nn.Dense(entity_size)

        # Dense layers for damage scalar and one-hot encoding
        self.linear_damage_scalar = nn.Dense(entity_size)
        self.linear_damage_onehot = nn.Dense(entity_size)

        self.output = nn.Dense(entity_size)

    def encode_damage_category(self, damage: chex.Array, num_bins: int = 16):
        divisor = 2048 / num_bins
        token = jnp.floor((damage + 1023) / divisor)
        token = jnp.where(damage == 0, num_bins + 1, token)
        return jax.nn.one_hot(token, num_bins + 1)

    def encode_edge(
        self,
        edge: chex.Array,
        poke1_embedding: chex.Array,
        poke2_embedding: chex.Array,
    ) -> chex.Array:
        # Process categorical features with embeddings

        has_poke1 = edge[FeatureEdge.POKE1_INDEX] >= 0
        poke1_embed = self.linear_pokemon1(jnp.where(has_poke1, poke1_embedding, 0))
        poke2_embed = self.linear_pokemon2(
            jnp.where(edge[FeatureEdge.POKE2_INDEX] >= 0, poke2_embedding, 0)
        )

        move_embed = self.embedding_move(edge[FeatureEdge.MOVE_TOKEN])
        item_embed = self.embedding_item(edge[FeatureEdge.ITEM_TOKEN])
        ability_embed = self.embedding_ability(edge[FeatureEdge.ABILITY_TOKEN])
        status_embed = self.embedding_status(edge[FeatureEdge.STATUS_TOKEN])
        major_args_embed = self.embedding_major_arg(edge[FeatureEdge.MAJOR_ARG])
        minor_args_embed = self.embedding_minor_arg(edge[FeatureEdge.MINOR_ARG])
        edge_type_embed = self.embedding_edge_type(edge[FeatureEdge.EDGE_TYPE_TOKEN])

        # # Process numeric and binary features with linear layers
        boosts = jax.lax.slice(
            edge, (FeatureEdge.BOOST_ATK_VALUE,), (FeatureEdge.BOOST_EVASION_VALUE + 1,)
        )
        boosts_embed = self.linear_boosts(boosts)

        # Scaling damage value

        damage_token = edge[FeatureEdge.DAMAGE_TOKEN]
        damage_raw = damage_token / 1023
        damage_features = jnp.concat(
            (
                damage_raw[..., None],  # raw
                jnp.abs(damage_raw)[..., None],  # magnitude
                jnp.sign(damage_token)[..., None],  # direction
                self.encode_damage_category(damage_token),  # category
            ),
            axis=-1,
        )
        damage_embed = self.linear_damage_scalar(damage_features)

        # # Normalize turn order value
        turn_order_embed = self.embedding_turn_order(edge[FeatureEdge.TURN_ORDER_VALUE])

        side_affecting_embed = self.linear_side_affecting(
            _binary_scale_embedding(
                edge[FeatureEdge.EDGE_AFFECTING_SIDE].astype(jnp.int32), 3
            )
        )

        # Concatenate all features into a single embedding
        combined = (
            poke1_embed
            + poke2_embed
            + move_embed
            + item_embed
            + ability_embed
            + status_embed
            + boosts_embed
            + damage_embed
            + turn_order_embed
            + edge_type_embed
            + major_args_embed
            + minor_args_embed
            + jnp.where(has_poke1, side_affecting_embed, 0)
        )

        return combined

    def batch_encode(
        self,
        edges: chex.Array,
        poke1_embeddings: chex.Array,
        poke2_embeddings: chex.Array,
    ) -> chex.Array:
        func = jax.vmap(jax.vmap(self.encode_edge))
        return func(edges, poke1_embeddings, poke2_embeddings)

    def __call__(
        self,
        edges: chex.Array,
        poke1_embeddings: chex.Array,
        poke2_embeddings: chex.Array,
    ) -> chex.Array:
        return self.batch_encode(edges, poke1_embeddings, poke2_embeddings)


def cosine_similarity(arr1: jnp.ndarray, arr2: jnp.ndarray) -> jnp.ndarray:

    # Compute dot products along the last dimension (D)
    dot_product = np.sum(arr1 * arr2, axis=-1)

    # Compute the L2 norms along the last dimension (D)
    norm_arr1 = jnp.linalg.norm(arr1, axis=-1)
    norm_arr2 = jnp.linalg.norm(arr2, axis=-1)

    # Compute cosine similarity, handle division by zero if needed
    cosine_sim = dot_product / (norm_arr1 * norm_arr2 + 1e-8)

    return cosine_sim


class PublicEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        entity_size = self.cfg.entity_size

        self.positional_embedding = nn.Embed(NUM_HISTORY, entity_size)

        self.entity_encoder = EntityEncoder(self.cfg.entity_encoder)
        self.edge_encoder = EdgeEncoder(self.cfg.edge_encoder)

        self.node_transformer = Transformer(**self.cfg.context_transformer.to_dict())
        self.edge_transformer = Transformer(**self.cfg.context_transformer.to_dict())

        self.context_transformer = Transformer(**self.cfg.context_transformer.to_dict())
        self.history_transformer = Transformer(**self.cfg.history_transformer.to_dict())

        self.linear_latent_action = MLP((entity_size,))
        self.linear_dynamics = MLP((entity_size,))

    def repr_learn(self, st: chex.Array, stp1: chex.Array):
        # Concatenate st and stp1 to get latent action
        latent_action = self.linear_latent_action(jnp.concatenate((st, stp1), axis=-1))
        latent_action_probs = jax.nn.softmax(latent_action.reshape(-1, 32), axis=-1)
        latent_action_ohe = jax.nn.one_hot(
            jnp.argmax(latent_action_probs, axis=-1), num_classes=32
        ).reshape(latent_action.shape)
        latent_action_probs = latent_action_probs.reshape(latent_action.shape)
        latent_action = latent_action_probs + jax.lax.stop_gradient(
            latent_action_ohe - latent_action_probs
        )

        # Predict the next state (stp1) using dynamics and latent action
        pred_stp1 = self.linear_dynamics(jnp.concatenate((st, latent_action), axis=-1))

        def normalize(arr: chex.Array) -> chex.Array:
            return arr / jnp.linalg.norm(arr, axis=-1)

        stp1 = normalize(stp1)
        pred_stp1 = normalize(pred_stp1)

        # Compute cosine similarity (negative for loss minimization)
        return -(pred_stp1 * stp1).sum(axis=-1)

    def __call__(self, env_step: EnvStep):
        node_embeddings = self.entity_encoder.batch_encode(env_step.history_nodes)

        poke1_index_data = jax.lax.dynamic_index_in_dim(
            env_step.history_edges,
            FeatureEdge.POKE1_INDEX,
            axis=-1,
            keepdims=False,
        )
        poke2_index_data = jax.lax.dynamic_index_in_dim(
            env_step.history_edges,
            FeatureEdge.POKE1_INDEX,
            axis=-1,
            keepdims=False,
        )
        poke1_embeddings = jnp.einsum(
            "ijk,ilj->ilk",
            node_embeddings,
            jnp.where(
                jnp.expand_dims(poke1_index_data, axis=-1) >= 0,
                jax.nn.one_hot(poke1_index_data, 12),
                0,
            ),
        )
        poke2_embeddings = jnp.einsum(
            "ijk,ilj->ilk",
            node_embeddings,
            jnp.where(
                jnp.expand_dims(poke2_index_data, axis=-1) >= 0,
                jax.nn.one_hot(poke2_index_data, 12),
                0,
            ),
        )
        edge_embeddings = self.edge_encoder.batch_encode(
            env_step.history_edges, poke1_embeddings, poke2_embeddings
        )

        species_token = env_step.history_nodes[..., FeatureEntity.ENTITY_SPECIES]
        invalid_node_mask = (species_token == SpeciesEnum.SPECIES__NULL) | (
            species_token == SpeciesEnum.SPECIES__PAD
        )
        valid_node_mask = ~invalid_node_mask

        edge_type_token = env_step.history_edges[..., FeatureEdge.EDGE_TYPE_TOKEN]
        invalid_edge_mask = edge_type_token == EdgeTypes.EDGE_TYPE_NONE

        node_embeddings = jax.vmap(
            self.node_transformer, in_axes=(0, None, 0, None), out_axes=0
        )(node_embeddings, None, valid_node_mask, None)

        edge_embeddings = jax.vmap(
            self.edge_transformer, in_axes=(0, None, 0, None), out_axes=0
        )(edge_embeddings, None, ~invalid_edge_mask, None)

        cross_node_embeddings = jax.vmap(self.context_transformer)(
            node_embeddings, edge_embeddings, valid_node_mask, ~invalid_edge_mask
        )

        repr_loss = jax.vmap(self.repr_learn)(
            cross_node_embeddings[1], cross_node_embeddings[0]
        )

        positional_embeddings = self.positional_embedding(jnp.arange(NUM_HISTORY))
        contextual_node_embeddings_w_pos = cross_node_embeddings + jnp.expand_dims(
            positional_embeddings, axis=1
        )

        contextual_node_embeddings_w_pos = jax.vmap(
            self.history_transformer, in_axes=(1, None, 1, None), out_axes=1
        )(contextual_node_embeddings_w_pos, None, valid_node_mask, None)

        denominator = valid_node_mask.sum(axis=0)
        contextual_node_embeddings_w_pos = jnp.einsum(
            "ij,ijk->jk", valid_node_mask, contextual_node_embeddings_w_pos
        ) / denominator[..., None].clip(min=1)

        return contextual_node_embeddings_w_pos, denominator > 0, repr_loss.mean()


class Encoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.turn_encoder = nn.Embed(50, self.cfg.vector_size)
        self.public_encoder = PublicEncoder(self.cfg.public)
        self.context_transformer = Transformer(**self.cfg.context_transformer.to_dict())
        self.to_vector = ToAvgVector(**self.cfg.to_vector.to_dict())
        self.move_encoder = MoveEncoder(self.cfg.move_encoder)
        self.state_resnet = Resnet(**self.cfg.state_resnet)

    def __call__(self, env_step: EnvStep):
        species_token = env_step.team[..., FeatureEntity.ENTITY_SPECIES]
        invalid_private_mask = (species_token == SpeciesEnum.SPECIES__NULL) | (
            species_token == SpeciesEnum.SPECIES__PAD
        )
        valid_private_mask = ~invalid_private_mask

        _encode_entity = jax.vmap(self.public_encoder.entity_encoder.encode_entity)

        private_entity_embeddings = _encode_entity(env_step.team)
        public_entity_embeddings, valid_public_mask, repr_loss = self.public_encoder(
            env_step
        )

        private_entity_embeddings = self.context_transformer(
            private_entity_embeddings,
            public_entity_embeddings,
            valid_private_mask,
            valid_public_mask,
        )

        current_state = self.to_vector(private_entity_embeddings, valid_private_mask)
        current_state = self.state_resnet(current_state) + self.turn_encoder(
            jnp.clip(env_step.turn, max=49)
        )

        move_embeddings = self.move_encoder(env_step.moveset[0, :4])

        return current_state, move_embeddings, private_entity_embeddings, repr_loss
