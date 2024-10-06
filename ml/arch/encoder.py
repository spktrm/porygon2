from typing import Tuple

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

        self.moveset_linear = nn.Dense(features=entity_size)
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
        move_indices_onehot = jax.nn.one_hot(move_indices, NUM_MOVES)
        pp_indices = jax.lax.slice(
            entity,
            (FeatureEntity.ENTITY_MOVEPP0,),
            (FeatureEntity.ENTITY_HAS_STATUS,),
        )
        pp_indices_broadcasted = (
            jnp.expand_dims(pp_indices / 1024, axis=-1) * move_indices_onehot
        )
        return jnp.concatenate(
            (move_indices_onehot, pp_indices_broadcasted), axis=-1
        ).sum(0)

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
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_GENDER], NUM_GENDERS),
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_STATUS], NUM_STATUS),
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_BEING_CALLED_BACK], 2),
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_TRAPPED], 2),
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_NEWLY_SWITCHED], 2),
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_TOXIC_TURNS], 8),
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_SLEEP_TURNS], 4),
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_FAINTED], 2),
            jax.nn.one_hot(entity[FeatureEntity.ENTITY_ACTIVE], 2),
            _binary_scale_embedding(
                entity[FeatureEntity.ENTITY_LEVEL].astype(jnp.int32), 101
            ),
            hp_ratio[jnp.newaxis],
            _binary_scale_embedding(hp_token, 1024),
            self.encode_item(entity),
            self.encode_boosts(entity),
            self.encode_volatiles(entity),
        ]

        # Concatenate boolean features into a single vector
        boolean_code = jnp.concatenate(basic_embeddings, axis=0)
        moveset_onehot = self.encode_moveset(entity)

        # Final embedding combination (linear layers applied to encoded features)
        embedding = (
            self.onehot_linear(boolean_code.astype(jnp.float32))
            + self.embed_species(entity[FeatureEntity.ENTITY_SPECIES])
            + self.embed_ability(entity[FeatureEntity.ENTITY_ABILITY])
            + self.side_linear(jax.nn.one_hot(entity[FeatureEntity.ENTITY_SIDE], 2))
            + self.moveset_linear(moveset_onehot)
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


class PublicEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """
        Initializes the necessary encoders and transformers for the model.
        """
        entity_size = self.cfg.entity_size

        self.positional_embedding = nn.Embed(NUM_HISTORY, entity_size)
        self.entity_encoder = EntityEncoder(self.cfg.entity_encoder)
        self.edge_encoder = EdgeEncoder(self.cfg.edge_encoder)

        self.context_transformer = Transformer(**self.cfg.context_transformer.to_dict())
        self.history_transformer = Transformer(**self.cfg.history_transformer.to_dict())

        self.projection = MLP((entity_size, entity_size, entity_size))
        self.projection_head = MLP((entity_size, entity_size))

    def _compute_embeddings(
        self, env_step: EnvStep
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Computes embeddings for nodes, edges, and positional embeddings.
        """
        node_embeddings = self.entity_encoder.batch_encode(env_step.history_nodes)
        poke1_embeddings, poke2_embeddings = self._get_poke_embeddings(
            node_embeddings, env_step
        )
        edge_embeddings = self.edge_encoder.batch_encode(
            env_step.history_edges, poke1_embeddings, poke2_embeddings
        )
        positional_embeddings = self.positional_embedding(jnp.arange(NUM_HISTORY))

        return node_embeddings, edge_embeddings, positional_embeddings

    def _get_poke_embeddings(
        self, node_embeddings: chex.Array, env_step: EnvStep
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Extracts poke embeddings for both ends of the edges (poke1 and poke2).
        """
        poke1_index_data = self._get_index_data(
            env_step.history_edges, FeatureEdge.POKE1_INDEX
        )
        poke2_index_data = self._get_index_data(
            env_step.history_edges, FeatureEdge.POKE2_INDEX
        )

        poke1_embeddings = self._apply_one_hot_embeddings(
            node_embeddings, poke1_index_data
        )
        poke2_embeddings = self._apply_one_hot_embeddings(
            node_embeddings, poke2_index_data
        )

        return poke1_embeddings, poke2_embeddings

    def _get_index_data(self, edges: chex.Array, index: int) -> chex.Array:
        """
        Retrieves index data for a given feature from edges.
        """
        return jax.lax.dynamic_index_in_dim(edges, index, axis=-1, keepdims=False)

    def _apply_one_hot_embeddings(
        self, node_embeddings: chex.Array, index_data: chex.Array
    ) -> chex.Array:
        """
        Applies one-hot encoding to index data and performs matrix multiplication with node embeddings.
        """
        return jnp.einsum(
            "ijk,ilj->ilk",
            node_embeddings,
            jnp.where(
                jnp.expand_dims(index_data, axis=-1) >= 0,
                jax.nn.one_hot(index_data, 12),
                0,
            ),
        )

    def _create_masks(self, env_step: EnvStep) -> Tuple[chex.Array, chex.Array]:
        """
        Creates masks to indicate valid nodes and edges.
        """
        species_token = env_step.history_nodes[..., FeatureEntity.ENTITY_SPECIES]
        invalid_node_mask = (species_token == SpeciesEnum.SPECIES__NULL) | (
            species_token == SpeciesEnum.SPECIES__PAD
        )
        valid_node_mask = ~invalid_node_mask

        edge_type_token = env_step.history_edges[..., FeatureEdge.EDGE_TYPE_TOKEN]
        invalid_edge_mask = edge_type_token == EdgeTypes.EDGE_TYPE_NONE

        return valid_node_mask, invalid_edge_mask

    def _add_positional_embeddings(
        self, embeddings: chex.Array, positional_embeddings: chex.Array
    ) -> chex.Array:
        """
        Adds positional embeddings to the given embeddings.
        """
        return embeddings + jnp.expand_dims(positional_embeddings, axis=1)

    def _repr_loss(
        self,
        predicted_node_embeddings: chex.Array,
        actual_node_embeddings: chex.Array,
        loss_mask: chex.Array,
    ) -> chex.Array:
        dynamic_project = self.projection_head(
            self.projection(predicted_node_embeddings)
        )
        obs_project = self.projection(actual_node_embeddings)
        loss_value = -cosine_similarity(  # We want the negative since we want to maximise similarity
            dynamic_project, jax.lax.stop_gradient(obs_project)
        )
        num_diffs = loss_mask.sum().clip(min=1)
        return jnp.where(loss_mask, loss_value, 0).sum() / num_diffs

    def _apply_transformers(
        self,
        node_embeddings: chex.Array,
        edge_embeddings: chex.Array,
        positional_embeddings: chex.Array,
        valid_node_mask: chex.Array,
        invalid_edge_mask: chex.Array,
    ) -> chex.Array:
        """
        Applies the context and history transformers to the node embeddings,
        edge embeddings, and positional information.
        """
        # Apply context transformer across nodes and edges
        cross_node_embeddings = jax.vmap(self.context_transformer)(
            node_embeddings, edge_embeddings, valid_node_mask, ~invalid_edge_mask
        )

        node_diff = cosine_similarity(node_embeddings[0], node_embeddings[1])
        repr_loss = self._repr_loss(
            cross_node_embeddings[1], node_embeddings[0], node_diff < 1
        )

        # Add positional information and apply the history transformer
        contextual_node_embeddings_w_pos = self._add_positional_embeddings(
            cross_node_embeddings, positional_embeddings
        )
        contextual_node_embeddings_w_pos = jax.vmap(
            self.history_transformer, in_axes=(1, None, 1, None), out_axes=1
        )(contextual_node_embeddings_w_pos, None, valid_node_mask, None)

        return contextual_node_embeddings_w_pos, repr_loss

    def _aggregate_embeddings(
        self, embeddings: chex.Array, valid_mask: chex.Array
    ) -> chex.Array:
        """
        Averages the valid embeddings across nodes.
        """
        denominator = valid_mask.sum(axis=0)
        return (
            jnp.einsum("ij,ijk->jk", valid_mask, embeddings)
            / denominator[..., None].clip(min=1),
            denominator > 0,
        )

    def __call__(self, env_step: EnvStep) -> Tuple[chex.Array, chex.Array]:
        """
        Forward pass for the PublicEncoder model, processing an environment step to produce embeddings.
        """
        # Compute node, edge, and positional embeddings
        node_embeddings, edge_embeddings, positional_embeddings = (
            self._compute_embeddings(env_step)
        )

        # Create masks for valid nodes and edges
        valid_node_mask, invalid_edge_mask = self._create_masks(env_step)

        # Apply transformers to contextualize embeddings
        contextual_node_embeddings_w_pos, repr_loss = self._apply_transformers(
            node_embeddings,
            edge_embeddings,
            positional_embeddings,
            valid_node_mask,
            invalid_edge_mask,
        )

        # Aggregate embeddings and return final result
        public_embeddings, valid_mask = self._aggregate_embeddings(
            contextual_node_embeddings_w_pos, valid_node_mask
        )
        return public_embeddings, valid_mask, repr_loss


class Encoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """
        Initializes necessary encoders and transformers for the encoder model.
        """
        self.turn_encoder = nn.Embed(50, self.cfg.vector_size)
        self.public_encoder = PublicEncoder(self.cfg.public)
        self.context_transformer = Transformer(**self.cfg.context_transformer.to_dict())
        self.to_vector = ToAvgVector(**self.cfg.to_vector.to_dict())
        self.move_encoder = MoveEncoder(self.cfg.move_encoder)
        self.state_resnet = Resnet(**self.cfg.state_resnet)
        self.action_merge = VectorMerge(**self.cfg.action_merge)
        self.action_transformer = Transformer(**self.cfg.context_transformer.to_dict())

    def _create_private_entity_embeddings(
        self, env_step: EnvStep
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Creates private entity embeddings and generates the mask for valid private entities.
        """
        species_token = env_step.team[..., FeatureEntity.ENTITY_SPECIES]
        invalid_private_mask = (species_token == SpeciesEnum.SPECIES__NULL) | (
            species_token == SpeciesEnum.SPECIES__PAD
        )
        valid_private_mask = ~invalid_private_mask

        # Encode private entities using the public encoder's entity encoder
        _encode_entity = jax.vmap(
            jax.vmap(self.public_encoder.entity_encoder.encode_entity)
        )
        private_entity_embeddings = _encode_entity(env_step.team)

        private_entity_embeddings = private_entity_embeddings.reshape(
            -1, private_entity_embeddings.shape[-1]
        )
        return private_entity_embeddings, valid_private_mask.reshape(-1)

    def _get_public_entity_embeddings(
        self, env_step: EnvStep
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Retrieves the public entity embeddings and the valid public mask from the public encoder.
        """
        return self.public_encoder(env_step)

    def _apply_context_transformer(
        self,
        private_entity_embeddings: chex.Array,
        public_entity_embeddings: chex.Array,
        valid_private_mask: chex.Array,
        valid_public_mask: chex.Array,
    ) -> chex.Array:
        """
        Applies the context transformer to merge private and public entity embeddings.
        """
        return self.context_transformer(
            private_entity_embeddings,
            public_entity_embeddings,
            valid_private_mask,
            valid_public_mask,
        )

    def _compute_current_state(
        self,
        private_entity_embeddings: chex.Array,
        valid_private_mask: chex.Array,
        turn: chex.Array,
    ) -> chex.Array:
        """
        Computes the current state representation using the entity embeddings,
        ResNet, and turn embeddings.
        """
        current_state = self.to_vector(private_entity_embeddings, valid_private_mask)
        current_state = self.state_resnet(current_state) + self.turn_encoder(
            jnp.clip(turn, max=49)
        )
        return current_state

    def _compute_action_embeddings(
        self, private_entity_embeddings: chex.Array, moveset: chex.Array
    ) -> chex.Array:
        """
        Computes action embeddings by merging private entity embeddings and move embeddings.
        """
        # Concatenate private entity embeddings
        entity_embeddings = jnp.concatenate(
            (
                jnp.tile(private_entity_embeddings[:1], (4, 1)),
                private_entity_embeddings,
            ),
            axis=0,
        )

        # Encode moves and merge them with entity embeddings
        action_embeddings = self.move_encoder(moveset)
        action_embeddings = jax.vmap(self.action_merge)(
            entity_embeddings, action_embeddings
        )

        return action_embeddings

    def __call__(self, env_step: EnvStep) -> Tuple[chex.Array, chex.Array]:
        """
        Forward pass of the Encoder model, processing an environment step to produce
        the current state and action embeddings.
        """
        # Private entity embeddings and valid mask
        private_entity_embeddings, valid_private_mask = (
            self._create_private_entity_embeddings(env_step)
        )

        # Public entity embeddings and valid mask
        public_entity_embeddings, valid_public_mask, repr_loss = (
            self._get_public_entity_embeddings(env_step)
        )

        # Apply context transformer to merge public and private entity embeddings
        private_entity_embeddings = self._apply_context_transformer(
            private_entity_embeddings,
            public_entity_embeddings,
            valid_private_mask,
            valid_public_mask,
        )

        # Compute the current state using the updated private embeddings and valid mask
        current_state = self._compute_current_state(
            private_entity_embeddings, valid_private_mask, env_step.turn
        )

        # Compute action embeddings by merging private entity embeddings with move embeddings
        action_embeddings = jax.vmap(self._compute_action_embeddings)(
            private_entity_embeddings.reshape(2, 6, -1), env_step.moveset
        )
        action_embeddings = self.action_transformer(
            action_embeddings[0],
            action_embeddings[1],
            env_step.legal,
            jnp.ones_like(env_step.legal),
        )

        return current_state, action_embeddings, repr_loss
