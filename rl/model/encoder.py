import functools
import math
from functools import partial
from typing import Dict, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from rl.environment.data import (
    ACTION_MAX_VALUES,
    ENTITY_EDGE_MAX_VALUES,
    ENTITY_NODE_MAX_VALUES,
    FIELD_MAX_VALUES,
    MAX_RATIO_TOKEN,
    NUM_FROM_SOURCE_EFFECTS,
    NUM_MOVES,
)
from rl.environment.interfaces import EnvStep, HistoryStep
from rl.environment.protos.enums_pb2 import (
    AbilitiesEnum,
    BattlemajorargsEnum,
    EffectEnum,
    ItemsEnum,
    MovesEnum,
    SpeciesEnum,
)
from rl.environment.protos.features_pb2 import (
    EntityEdgeFeature,
    EntityNodeFeature,
    FieldFeature,
    InfoFeature,
    MovesetFeature,
)
from rl.model.modules import (
    FeedForwardResidual,
    MergeEmbeddings,
    PretrainedEmbedding,
    SumEmbeddings,
    TransformerDecoder,
    TransformerEncoder,
    one_hot_concat_jax,
)

# Load pretrained embeddings for various features.
SPECIES_ONEHOT = PretrainedEmbedding(
    fpath="data/data/gen3/species.npy", dtype=jnp.bfloat16
)
ABILITY_ONEHOT = PretrainedEmbedding(
    fpath="data/data/gen3/abilities.npy", dtype=jnp.bfloat16
)
ITEM_ONEHOT = PretrainedEmbedding(fpath="data/data/gen3/items.npy", dtype=jnp.bfloat16)
MOVE_ONEHOT = PretrainedEmbedding(fpath="data/data/gen3/moves.npy", dtype=jnp.bfloat16)


def _binary_scale_encoding(
    to_encode: chex.Array, world_dim: int, dtype: jnp.dtype = jnp.float32
) -> chex.Array:
    """Encode the feature using its binary representation."""
    chex.assert_rank(to_encode, 0)
    chex.assert_type(to_encode, jnp.int32)
    num_bits = (world_dim - 1).bit_length()
    bit_mask = 1 << np.arange(num_bits)
    pos = jnp.broadcast_to(to_encode[jnp.newaxis], num_bits)
    result = jnp.not_equal(jnp.bitwise_and(pos, bit_mask), 0)
    return result.astype(dtype)


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
    entity: chex.Array,
    feature_idx: int,
    max_values: Dict[int, int],
    dtype: jnp.dtype = jnp.int32,
) -> Tuple[int, int]:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    max_value = max_values[feature_idx]
    max_sqrt_value = int(math.floor(math.sqrt(max_value)))
    x = jnp.floor(jnp.sqrt(entity[feature_idx].astype(dtype)))
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


_encode_one_hot_entity = partial(_encode_one_hot, max_values=ENTITY_NODE_MAX_VALUES)
_encode_one_hot_action = partial(_encode_one_hot, max_values=ACTION_MAX_VALUES)
_encode_one_hot_edge = partial(_encode_one_hot, max_values=ENTITY_EDGE_MAX_VALUES)
_encode_one_hot_field = partial(_encode_one_hot, max_values=FIELD_MAX_VALUES)
_encode_one_hot_entity_boost = partial(_encode_one_hot_entity, value_offset=6)
_encode_one_hot_edge_boost = partial(_encode_one_hot_edge, value_offset=6)
_encode_sqrt_one_hot_entity = partial(
    _encode_sqrt_one_hot, max_values=ENTITY_NODE_MAX_VALUES
)
_encode_sqrt_one_hot_action = partial(
    _encode_sqrt_one_hot, max_values=ACTION_MAX_VALUES
)
_encode_divided_one_hot_entity = partial(
    _encode_divided_one_hot, max_values=ENTITY_NODE_MAX_VALUES
)
_encode_divided_one_hot_edge = partial(
    _encode_divided_one_hot, max_values=ENTITY_EDGE_MAX_VALUES
)


def get_entity_mask(entity: chex.Array) -> chex.Array:
    """
    Generate a mask to identify valid entities based on species tokens.
    """
    species_token = entity[EntityNodeFeature.ENTITY_NODE_FEATURE__SPECIES]
    return ~(
        (species_token == SpeciesEnum.SPECIES_ENUM___NULL)
        | (species_token == SpeciesEnum.SPECIES_ENUM___PAD)
        | (species_token == SpeciesEnum.SPECIES_ENUM___UNSPECIFIED)
    )


class Encoder(nn.Module):
    """
    Encoder model for processing environment steps and history to generate embeddings.
    """

    cfg: ConfigDict

    def setup(self):

        # Extract configuration parameters for embedding sizes.
        entity_size = self.cfg.entity_size

        embed_kwargs = dense_kwargs = dict(features=entity_size, dtype=self.cfg.dtype)

        # Initialize embeddings for various entities and features.

        self.effect_from_source_embedding = nn.Embed(
            num_embeddings=NUM_FROM_SOURCE_EFFECTS,
            name="effect_from_source_embedding",
            **embed_kwargs,
        )

        # Initialize linear layers for encoding various entity features.
        self.species_linear = nn.Dense(name="species_linear", **dense_kwargs)
        self.items_linear = nn.Dense(name="items_linear", **dense_kwargs)
        self.abilities_linear = nn.Dense(name="abilities_linear", **dense_kwargs)
        self.moves_linear = nn.Dense(name="moves_linear", **dense_kwargs)

        # Initialize aggregation modules for combining feature embeddings.
        self.entity_sum = SumEmbeddings(
            output_size=entity_size, dtype=self.cfg.dtype, name="entity_node_sum"
        )
        self.entity_edge_sum = SumEmbeddings(
            output_size=entity_size, dtype=self.cfg.dtype, name="entity_edge_sum"
        )
        self.field_sum = SumEmbeddings(
            output_size=entity_size, dtype=self.cfg.dtype, name="field_sum"
        )
        self.action_sum = SumEmbeddings(
            output_size=entity_size, dtype=self.cfg.dtype, name="action_sum"
        )
        self.latent_merge = MergeEmbeddings(
            output_size=entity_size, dtype=self.cfg.dtype, name="latent_merge"
        )

        # Feed-forward layers for processing entity and timestep features.
        self.entity_ff = FeedForwardResidual(
            hidden_dim=entity_size, dtype=self.cfg.dtype
        )
        self.timestep_ff = FeedForwardResidual(
            hidden_dim=entity_size, dtype=self.cfg.dtype
        )
        self.action_ff = FeedForwardResidual(
            hidden_dim=entity_size, dtype=self.cfg.dtype
        )

        # Transformer encoders for processing sequences of entities and edges.
        self.entity_encoder = TransformerEncoder(**self.cfg.entity_encoder.to_dict())
        self.per_timestep_node_encoder = TransformerEncoder(
            **self.cfg.per_timestep_node_encoder.to_dict()
        )
        self.per_timestep_node_edge_encoder = TransformerEncoder(
            **self.cfg.per_timestep_node_edge_encoder.to_dict()
        )
        self.timestep_encoder = TransformerEncoder(
            **self.cfg.timestep_encoder.to_dict()
        )
        self.action_encoder = TransformerEncoder(**self.cfg.action_encoder.to_dict())
        self.latent_encoder = TransformerEncoder(**self.cfg.latent_encoder.to_dict())

        # Transformer Decoders
        self.latent_timestep_decoder = TransformerDecoder(
            **self.cfg.latent_timestep_decoder.to_dict()
        )
        self.latent_entity_decoder = TransformerDecoder(
            **self.cfg.latent_entity_decoder.to_dict()
        )
        self.latent_action_decoder = TransformerDecoder(
            **self.cfg.latent_action_decoder.to_dict()
        )

        # Latents
        self.latent_embeddings = self.param(
            "latent_embeddings",
            nn.initializers.normal(dtype=self.cfg.dtype),
            (self.cfg.num_latents, entity_size),
        )

    def _encode_species(self, token: chex.Array):
        mask = ~(
            (token == SpeciesEnum.SPECIES_ENUM___UNSPECIFIED)
            | (token == SpeciesEnum.SPECIES_ENUM___PAD)
            | (token == SpeciesEnum.SPECIES_ENUM___NULL)
        )
        return mask * self.species_linear(SPECIES_ONEHOT(token))

    def _encode_item(self, token: chex.Array):
        mask = ~(
            (token == ItemsEnum.ITEMS_ENUM___UNSPECIFIED)
            | (token == ItemsEnum.ITEMS_ENUM___PAD)
            | (token == ItemsEnum.ITEMS_ENUM___NULL)
        )
        return mask * self.items_linear(ITEM_ONEHOT(token))

    def _encode_ability(self, token: chex.Array):
        mask = ~(
            (token == AbilitiesEnum.ABILITIES_ENUM___UNSPECIFIED)
            | (token == AbilitiesEnum.ABILITIES_ENUM___PAD)
            | (token == AbilitiesEnum.ABILITIES_ENUM___NULL)
        )
        return mask * self.abilities_linear(ABILITY_ONEHOT(token))

    def _encode_move(self, token: chex.Array):
        mask = ~(
            (token == MovesEnum.MOVES_ENUM___UNSPECIFIED)
            | (token == MovesEnum.MOVES_ENUM___PAD)
            | (token == MovesEnum.MOVES_ENUM___NULL)
        )
        return mask * self.moves_linear(MOVE_ONEHOT(token))

    def _encode_entity(self, entity: chex.Array):
        # Encode volatile and type-change indices using the binary encoder.
        _encode_hex = jax.vmap(
            functools.partial(_binary_scale_encoding, world_dim=65535)
        )
        volatiles_indices = entity[
            EntityNodeFeature.ENTITY_NODE_FEATURE__VOLATILES0 : EntityNodeFeature.ENTITY_NODE_FEATURE__VOLATILES8
            + 1
        ]
        volatiles_encoding = _encode_hex(volatiles_indices).reshape(-1)

        typechange_indices = entity[
            EntityNodeFeature.ENTITY_NODE_FEATURE__TYPECHANGE0 : EntityNodeFeature.ENTITY_NODE_FEATURE__TYPECHANGE1
            + 1
        ]
        typechange_encoding = _encode_hex(typechange_indices).reshape(-1)

        hp_ratio_token = entity[EntityNodeFeature.ENTITY_NODE_FEATURE__HP_RATIO]
        hp_ratio = (
            entity[EntityNodeFeature.ENTITY_NODE_FEATURE__HP_RATIO] / MAX_RATIO_TOKEN
        )
        hp_features = jnp.stack(
            [
                hp_ratio,
                hp_ratio == 0,
                (0 < hp_ratio) & (hp_ratio < 0.25),
                (0.25 <= hp_ratio) & (hp_ratio < 0.5),
                (0.5 <= hp_ratio) & (hp_ratio < 0.75),
                (0.75 <= hp_ratio) & (hp_ratio < 1),
                hp_ratio_token == MAX_RATIO_TOKEN,
            ],
            axis=-1,
        ).reshape(-1)

        boolean_code = one_hot_concat_jax(
            [
                _encode_sqrt_one_hot_entity(
                    entity,
                    EntityNodeFeature.ENTITY_NODE_FEATURE__LEVEL,
                    dtype=self.cfg.dtype,
                ),
                _encode_one_hot_entity(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__ACTIVE
                ),
                _encode_one_hot_entity(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__SIDE
                ),
                _encode_one_hot_entity(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__IS_PUBLIC
                ),
                _encode_divided_one_hot_entity(
                    entity,
                    EntityNodeFeature.ENTITY_NODE_FEATURE__HP_RATIO,
                    MAX_RATIO_TOKEN / 32,
                ),
                _encode_one_hot_entity(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__GENDER
                ),
                _encode_one_hot_entity(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__STATUS
                ),
                _encode_one_hot_entity(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__ITEM_EFFECT
                ),
                _encode_one_hot_entity(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__BEING_CALLED_BACK
                ),
                _encode_one_hot_entity(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__TRAPPED
                ),
                _encode_one_hot_entity(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__NEWLY_SWITCHED
                ),
                _encode_one_hot_entity(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__TOXIC_TURNS
                ),
                _encode_one_hot_entity(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__SLEEP_TURNS
                ),
                _encode_one_hot_entity(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__FAINTED
                ),
                _encode_one_hot_entity_boost(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_ATK_VALUE
                ),
                _encode_one_hot_entity_boost(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_DEF_VALUE
                ),
                _encode_one_hot_entity_boost(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_SPA_VALUE
                ),
                _encode_one_hot_entity_boost(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_SPD_VALUE
                ),
                _encode_one_hot_entity_boost(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_SPE_VALUE
                ),
                _encode_one_hot_entity_boost(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_EVASION_VALUE
                ),
                _encode_one_hot_entity_boost(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_ACCURACY_VALUE
                ),
            ]
        )

        move_indices = np.array(
            [
                EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEID0,
                EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEID1,
                EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEID2,
                EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEID3,
            ]
        )
        move_pp_indices = np.array(
            [
                EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEPP0,
                EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEPP1,
                EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEPP2,
                EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEPP3,
            ]
        )
        move_tokens = entity[move_indices]

        moves_onehot = jax.nn.one_hot(move_tokens, NUM_MOVES)
        is_valid_move = (move_tokens != MovesEnum.MOVES_ENUM___NULL) | (
            move_tokens != MovesEnum.MOVES_ENUM___UNSPECIFIED
        )
        move_pp_ratios = move_pp_indices[..., None] / 31
        move_pp_onehot = (
            jnp.where(is_valid_move[..., None], moves_onehot * move_pp_ratios, 0)
            .sum(0)
            .clip(min=0, max=1)
        )

        move_encodings = jax.vmap(self._encode_move)(move_tokens)

        species_token = entity[EntityNodeFeature.ENTITY_NODE_FEATURE__SPECIES]
        ability_token = entity[EntityNodeFeature.ENTITY_NODE_FEATURE__ABILITY]
        item_token = entity[EntityNodeFeature.ENTITY_NODE_FEATURE__ITEM]
        # last_move_token = entity[EntityNodeFeature.ENTITY_NODE_FEATURE__LAST_MOVE]

        embedding = self.entity_sum(
            boolean_code,
            volatiles_encoding,
            typechange_encoding,
            move_pp_onehot,
            hp_features,
            self._encode_species(species_token),
            self._encode_ability(ability_token),
            self._encode_item(item_token),
            move_encodings.sum(axis=0),
        )

        # Apply mask to filter out invalid entities.
        mask = get_entity_mask(entity)
        embedding = mask * embedding

        return embedding, mask

    def _encode_edge(self, edge: chex.Array):
        _encode_hex = jax.vmap(
            functools.partial(
                _binary_scale_encoding, world_dim=65535, dtype=self.cfg.dtype
            )
        )

        minor_args_indices = edge[
            EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG0 : EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG3
            + 1
        ]
        minor_args_encoding = _encode_hex(minor_args_indices).reshape(-1)

        # Aggregate embeddings for the relative edge.
        boolean_code = one_hot_concat_jax(
            [
                _encode_one_hot_edge(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG,
                ),
                _encode_divided_one_hot_edge(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__DAMAGE_RATIO,
                    MAX_RATIO_TOKEN / 32,
                ),
                _encode_divided_one_hot_edge(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__HEAL_RATIO,
                    MAX_RATIO_TOKEN / 32,
                ),
                _encode_one_hot_edge(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__STATUS_TOKEN,
                ),
                _encode_one_hot_edge_boost(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_ATK_VALUE,
                ),
                _encode_one_hot_edge_boost(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_DEF_VALUE,
                ),
                _encode_one_hot_edge_boost(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_SPA_VALUE,
                ),
                _encode_one_hot_edge_boost(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_SPD_VALUE,
                ),
                _encode_one_hot_edge_boost(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_SPE_VALUE,
                ),
                _encode_one_hot_edge_boost(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_EVASION_VALUE,
                ),
                _encode_one_hot_edge_boost(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_ACCURACY_VALUE,
                ),
            ]
        )

        effect_from_source_indices = np.array(
            [
                EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN0,
                EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN1,
                EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN2,
                EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN3,
                EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN4,
            ]
        )
        effect_from_source_tokens = edge[effect_from_source_indices]
        effect_from_source_mask = ~(
            (effect_from_source_tokens == EffectEnum.EFFECT_ENUM___UNSPECIFIED)
            | (effect_from_source_tokens == EffectEnum.EFFECT_ENUM___PAD)
            | (effect_from_source_tokens == EffectEnum.EFFECT_ENUM___NULL)
        )
        effect_from_source_embeddings = self.effect_from_source_embedding(
            effect_from_source_tokens
        )
        effect_from_source_embedding = effect_from_source_embeddings.sum(
            axis=0, where=effect_from_source_mask[..., None]
        )

        ability_token = edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__ABILITY_TOKEN]
        item_token = edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__ITEM_TOKEN]
        move_token = edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN]

        embedding = self.entity_edge_sum(
            minor_args_encoding,
            boolean_code,
            self._encode_ability(ability_token),
            self._encode_item(item_token),
            self._encode_move(move_token),
            effect_from_source_embedding,
        )

        mask = (
            edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG]
            != BattlemajorargsEnum.BATTLEMAJORARGS_ENUM___UNSPECIFIED
        ) | (minor_args_indices.sum(axis=-1) > 0)

        return embedding, mask

    def _encode_field(self, edge: chex.Array):
        """
        Encode features of an absolute edge, including turn and request offsets.
        """
        # Compute turn and request count differences for encoding.

        request_count = edge[FieldFeature.FIELD_FEATURE__REQUEST_COUNT]

        _encode_hex = jax.vmap(
            functools.partial(
                _binary_scale_encoding, world_dim=65535, dtype=self.cfg.dtype
            )
        )

        my_side_condition_indices = edge[
            FieldFeature.FIELD_FEATURE__MY_SIDECONDITIONS0 : FieldFeature.FIELD_FEATURE__MY_SIDECONDITIONS1
            + 1
        ]
        opp_side_condition_indices = edge[
            FieldFeature.FIELD_FEATURE__OPP_SIDECONDITIONS0 : FieldFeature.FIELD_FEATURE__OPP_SIDECONDITIONS1
            + 1
        ]
        my_side_condition_encoding = _encode_hex(my_side_condition_indices).reshape(-1)
        opp_side_condition_encoding = _encode_hex(opp_side_condition_indices).reshape(
            -1
        )

        # Aggregate embeddings for the absolute edge.
        boolean_code = one_hot_concat_jax(
            [
                _encode_one_hot_field(
                    edge,
                    FieldFeature.FIELD_FEATURE__WEATHER_ID,
                ),
                _encode_one_hot_field(
                    edge,
                    FieldFeature.FIELD_FEATURE__WEATHER_MAX_DURATION,
                ),
                _encode_one_hot_field(
                    edge,
                    FieldFeature.FIELD_FEATURE__WEATHER_MIN_DURATION,
                ),
                _encode_one_hot_field(
                    edge,
                    FieldFeature.FIELD_FEATURE__TERRAIN_ID,
                ),
                _encode_one_hot_field(
                    edge,
                    FieldFeature.FIELD_FEATURE__TERRAIN_MAX_DURATION,
                ),
                _encode_one_hot_field(
                    edge,
                    FieldFeature.FIELD_FEATURE__TERRAIN_MIN_DURATION,
                ),
                _encode_one_hot_field(
                    edge,
                    FieldFeature.FIELD_FEATURE__PSEUDOWEATHER_ID,
                ),
                _encode_one_hot_field(
                    edge,
                    FieldFeature.FIELD_FEATURE__PSEUDOWEATHER_MAX_DURATION,
                ),
                _encode_one_hot_field(
                    edge,
                    FieldFeature.FIELD_FEATURE__PSEUDOWEATHER_MIN_DURATION,
                ),
                _encode_one_hot_field(
                    edge,
                    FieldFeature.FIELD_FEATURE__MY_SPIKES,
                ),
                _encode_one_hot_field(
                    edge,
                    FieldFeature.FIELD_FEATURE__OPP_SPIKES,
                ),
                _encode_one_hot_field(
                    edge,
                    FieldFeature.FIELD_FEATURE__MY_TOXIC_SPIKES,
                ),
                _encode_one_hot_field(
                    edge,
                    FieldFeature.FIELD_FEATURE__OPP_TOXIC_SPIKES,
                ),
            ]
        )

        embedding = self.field_sum(
            boolean_code,
            my_side_condition_encoding,
            opp_side_condition_encoding,
            _binary_scale_encoding(
                to_encode=edge[FieldFeature.FIELD_FEATURE__TURN_ORDER_VALUE],
                world_dim=32,
            ),
        )

        # Apply mask to filter out invalid edges.
        mask = edge[FieldFeature.FIELD_FEATURE__VALID].astype(jnp.bool)
        request_count = jnp.where(mask, request_count, 1e9)

        return embedding, mask, request_count

    def _encode_entities(self, env_step: EnvStep):
        entity_encodings = entity_embeddings = jnp.concatenate(
            (env_step.private_team, env_step.public_team), axis=-2
        )
        encode_entities = jax.vmap(self._encode_entity)

        def _batched_fn(encodings: jax.Array):
            embeddings, mask = encode_entities(encodings)
            return (
                embeddings,
                self.entity_encoder(self.entity_ff(embeddings), mask),
                mask,
            )

        entity_embeddings, contextual_entity_embeddings, entity_mask = jax.vmap(
            _batched_fn
        )(entity_encodings)
        private_entity_embeddings = entity_embeddings[:, :6]

        return private_entity_embeddings, contextual_entity_embeddings, entity_mask

    # Encode each timestep's features, including nodes and edges.
    def _encode_timestep(self, history: HistoryStep):
        """
        Encode features of a single timestep, including entities and edges.
        """

        # Encode nodes.
        history_node_embedding, node_mask = jax.vmap(self._encode_entity)(history.nodes)

        # Encode edges.
        history_edge_embedding, edge_mask = jax.vmap(self._encode_edge)(history.edges)

        # Encode field
        field_embedding, valid_timestep_mask, history_request_count = (
            self._encode_field(history.field)
        )

        contextual_history_nodes = self.per_timestep_node_encoder(
            history_node_embedding, node_mask
        )

        node_edge_mask = node_mask & edge_mask
        contextual_history_nodes = self.per_timestep_node_edge_encoder(
            contextual_history_nodes + history_edge_embedding, node_edge_mask
        )

        timestep_embedding = (
            node_edge_mask.astype(self.cfg.dtype) @ contextual_history_nodes
            + field_embedding
        )

        return (
            timestep_embedding,
            valid_timestep_mask & edge_mask.any(),
            history_request_count,
        )

    def _encode_timesteps(self, history: HistoryStep):
        timestep_embedding, valid_timestep_mask, history_request_count = jax.vmap(
            self._encode_timestep
        )(history)

        timestep_embedding = self.timestep_ff(timestep_embedding)

        # Apply mask to the timestep embeddings.
        timestep_embedding = jnp.where(
            valid_timestep_mask[..., None], timestep_embedding, 0
        )

        seq_len = timestep_embedding.shape[0]

        contextual_timestep_embedding = self.timestep_encoder(
            timestep_embedding,
            valid_timestep_mask,
            jnp.tril(jnp.ones((seq_len, seq_len))),
        )

        return contextual_timestep_embedding, history_request_count

    # Encode actions for the current environment step.
    def _encode_action(
        self,
        action: chex.Array,
        legal: chex.Array,
        entity_embedding: chex.Array,
        context_encoding: chex.Array,
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
                    action, MovesetFeature.MOVESET_FEATURE__PP, dtype=self.cfg.dtype
                ),
                _encode_sqrt_one_hot_action(
                    action, MovesetFeature.MOVESET_FEATURE__MAXPP, dtype=self.cfg.dtype
                ),
                _encode_one_hot_action(action, MovesetFeature.MOVESET_FEATURE__HAS_PP),
            ]
        )

        embedding = self.action_sum(
            boolean_code,
            context_encoding,
            entity_embedding,
            self._encode_move(action[MovesetFeature.MOVESET_FEATURE__MOVE_ID]),
        )

        return embedding

    def _encode_actions(
        self, env_step: EnvStep, entity_embeddings: chex.Array
    ) -> chex.Array:

        def _batched_forward(env_step: EnvStep, entity_embeddings: chex.Array):
            context_encoding, _, _ = self._encode_field(env_step.current_context)
            action_entites = jnp.take(
                entity_embeddings,
                env_step.moveset[..., MovesetFeature.MOVESET_FEATURE__ENTITY_IDX],
                axis=0,
            )
            action_embeddings = jax.vmap(self._encode_action, in_axes=(0, 0, 0, None))(
                env_step.moveset,
                env_step.legal.astype(int),
                action_entites,
                context_encoding,
            )
            action_embeddings = self.action_ff(action_embeddings)
            return self.action_encoder(action_embeddings, env_step.legal)

        return jax.vmap(_batched_forward)(env_step, entity_embeddings)

    def __call__(
        self, env_step: EnvStep, history_step: HistoryStep
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Forward pass of the Encoder model. Processes an environment step and outputs
        contextual embeddings for actions.
        """
        private_entity_embeddings, contextual_entity_embeddings, entity_mask = (
            self._encode_entities(env_step)
        )

        timestep_embeddings, history_request_count = self._encode_timesteps(
            history_step
        )
        request_count = env_step.info[..., InfoFeature.INFO_FEATURE__REQUEST_COUNT]
        timestep_mask = request_count[..., None] >= history_request_count

        contextual_action_embeddings = self._encode_actions(
            env_step, private_entity_embeddings
        )
        action_mask = env_step.legal

        def _batched_forward(
            timestep_mask: jax.Array,
            entity_embeddings: jax.Array,
            entity_mask: jax.Array,
            action_embeddings: jax.Array,
            action_mask: jax.Array,
        ):
            latent_timesteps = self.latent_timestep_decoder(
                self.latent_embeddings, timestep_embeddings, None, timestep_mask
            )
            latent_entities = self.latent_entity_decoder(
                self.latent_embeddings, entity_embeddings, None, entity_mask
            )
            latent_actions = self.latent_action_decoder(
                self.latent_embeddings, action_embeddings, None, action_mask
            )

            latent_embeddings = jax.vmap(self.latent_merge)(
                latent_timesteps, latent_entities, latent_actions
            )
            contextual_latent_embeddings = self.latent_encoder(latent_embeddings)

            return contextual_latent_embeddings

        contextual_latent_embeddings = jax.vmap(_batched_forward)(
            timestep_mask,
            contextual_entity_embeddings,
            entity_mask,
            contextual_action_embeddings,
            action_mask,
        )

        return contextual_latent_embeddings, contextual_action_embeddings
