import functools
import math
from functools import partial

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
    ONEHOT_ENCODERS,
)
from rl.environment.interfaces import PlayerEnvOutput, PlayerHistoryOutput
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
    RMSNorm,
    SumEmbeddings,
    TransformerDecoder,
    TransformerEncoder,
    create_attention_mask,
    one_hot_concat_jax,
)
from rl.model.utils import LARGE_NEGATIVE_BIAS


def _binary_scale_encoding(
    to_encode: jax.Array, world_dim: int, dtype: jnp.dtype = jnp.float32
) -> jax.Array:
    """Encode the feature using its binary representation."""
    chex.assert_rank(to_encode, 0)
    chex.assert_type(to_encode, jnp.int32)
    num_bits = (world_dim - 1).bit_length()
    bit_mask = 1 << np.arange(num_bits)
    pos = jnp.broadcast_to(to_encode[jnp.newaxis], num_bits)
    result = jnp.not_equal(jnp.bitwise_and(pos, bit_mask), 0)
    return result.astype(dtype)


def _encode_one_hot(
    entity: jax.Array,
    feature_idx: int,
    max_values: dict[int, int],
    value_offset: int = 0,
) -> tuple[int, int]:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    return entity[feature_idx] + value_offset, max_values[feature_idx] + 1


def _encode_capped_one_hot(
    entity: jax.Array, feature_idx: int, max_values: dict[int, int]
) -> tuple[int, int]:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    max_value = max_values[feature_idx]
    return jnp.minimum(entity[feature_idx], max_value), max_value + 1


def _encode_sqrt_one_hot(
    entity: jax.Array,
    feature_idx: int,
    max_values: dict[int, int],
    dtype: jnp.dtype = jnp.int32,
) -> tuple[int, int]:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    max_value = max_values[feature_idx]
    max_sqrt_value = int(math.floor(math.sqrt(max_value)))
    x = jnp.floor(jnp.sqrt(entity[feature_idx].astype(dtype)))
    x = jnp.minimum(x.astype(jnp.int32), max_sqrt_value)
    return x, max_sqrt_value + 1


def _encode_divided_one_hot(
    entity: jax.Array, feature_idx: int, divisor: int, max_values: dict[int, int]
) -> tuple[int, int]:
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


def get_entity_mask(entity: jax.Array) -> jax.Array:
    """
    Generate a mask to identify valid entities based on species tokens.
    """
    species_token = entity[EntityNodeFeature.ENTITY_NODE_FEATURE__SPECIES]
    return ~(
        (species_token == SpeciesEnum.SPECIES_ENUM___NULL)
        | (species_token == SpeciesEnum.SPECIES_ENUM___PAD)
        | (species_token == SpeciesEnum.SPECIES_ENUM___UNSPECIFIED)
    )


def encode_boosts(boosts: jax.Array, offset: int):
    return jnp.where(
        boosts > 0,
        (offset + boosts) / offset,
        offset / (offset - boosts),
    )


def encode_reg_boosts(boosts: jax.Array):
    """Encodes according to https://bulbapedia.bulbagarden.net/wiki/Stat_modifier#Stage_multipliers"""
    return (1 / math.log(2)) * jnp.log(encode_boosts(boosts, 2))


def encode_spe_boosts(boosts: jax.Array):
    """Encodes according to https://bulbapedia.bulbagarden.net/wiki/Stat_modifier#Stage_multipliers"""
    return (2 / math.log(3)) * jnp.log(encode_boosts(boosts, 3))


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

        # Cls Embeddings
        self.move_cls_embedding = self.param(
            "move_cls_embedding",
            nn.initializers.truncated_normal(),
            (1, entity_size),
            dtype=self.cfg.dtype,
        )
        self.switch_cls_embedding = self.param(
            "switch_cls_embedding",
            nn.initializers.truncated_normal(),
            (1, entity_size),
            dtype=self.cfg.dtype,
        )

        # Initialize linear layers for encoding various entity features.
        self.species_linear = nn.Dense(
            name="species_linear", use_bias=False, **dense_kwargs
        )
        self.items_linear = nn.Dense(
            name="items_linear", use_bias=False, **dense_kwargs
        )
        self.abilities_linear = nn.Dense(
            name="abilities_linear", use_bias=False, **dense_kwargs
        )
        self.moves_linear = nn.Dense(
            name="moves_linear", use_bias=False, **dense_kwargs
        )
        self.learnset_linear = nn.Dense(
            name="learnset_linear", use_bias=False, **dense_kwargs
        )

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

        # Layer normalization for action embeddings.
        self.timestep_ln = RMSNorm()
        self.entities_ln = RMSNorm()
        self.moves_ln = RMSNorm()
        self.switch_ln = RMSNorm()

        # Feed-forward layers for processing entity and timestep features.
        self.entity_ff = FeedForwardResidual(
            hidden_dim=entity_size, dtype=self.cfg.dtype
        )
        self.timestep_ff = FeedForwardResidual(
            hidden_dim=entity_size, dtype=self.cfg.dtype
        )
        self.move_ff = FeedForwardResidual(hidden_dim=entity_size, dtype=self.cfg.dtype)
        self.switch_ff = FeedForwardResidual(
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
        self.move_encoder = TransformerEncoder(**self.cfg.move_encoder.to_dict())
        self.switch_encoder = TransformerEncoder(**self.cfg.switch_encoder.to_dict())

        # Transformer Decoders
        self.entity_timestep_decoder = TransformerDecoder(
            **self.cfg.entity_timestep_decoder.to_dict()
        )
        self.entity_action_decoder = TransformerDecoder(
            **self.cfg.entity_action_decoder.to_dict()
        )
        self.action_entity_decoder = TransformerDecoder(
            **self.cfg.action_entity_decoder.to_dict()
        )

    def _embed_species(self, token: jax.Array):
        mask = ~(
            (token == SpeciesEnum.SPECIES_ENUM___UNSPECIFIED)
            | (token == SpeciesEnum.SPECIES_ENUM___PAD)
            | (token == SpeciesEnum.SPECIES_ENUM___NULL)
        )
        _ohe_encoder = ONEHOT_ENCODERS[self.cfg.generation]["species"]
        return mask * self.species_linear(_ohe_encoder(token))

    def _embed_learnset(self, token: jax.Array):
        mask = ~(
            (token == SpeciesEnum.SPECIES_ENUM___UNSPECIFIED)
            | (token == SpeciesEnum.SPECIES_ENUM___PAD)
            | (token == SpeciesEnum.SPECIES_ENUM___NULL)
        )
        _ohe_encoder = ONEHOT_ENCODERS[self.cfg.generation]["learnset"]
        return mask * self.learnset_linear(_ohe_encoder(token))

    def _embed_item(self, token: jax.Array):
        mask = ~(
            (token == ItemsEnum.ITEMS_ENUM___UNSPECIFIED)
            | (token == ItemsEnum.ITEMS_ENUM___PAD)
            | (token == ItemsEnum.ITEMS_ENUM___NULL)
        )
        _ohe_encoder = ONEHOT_ENCODERS[self.cfg.generation]["items"]
        return mask * self.items_linear(_ohe_encoder(token))

    def _embed_ability(self, token: jax.Array):
        mask = ~(
            (token == AbilitiesEnum.ABILITIES_ENUM___UNSPECIFIED)
            | (token == AbilitiesEnum.ABILITIES_ENUM___PAD)
            | (token == AbilitiesEnum.ABILITIES_ENUM___NULL)
        )
        _ohe_encoder = ONEHOT_ENCODERS[self.cfg.generation]["abilities"]
        return mask * self.abilities_linear(_ohe_encoder(token))

    def _embed_move(self, token: jax.Array):
        mask = ~(
            (token == MovesEnum.MOVES_ENUM___UNSPECIFIED)
            | (token == MovesEnum.MOVES_ENUM___PAD)
            | (token == MovesEnum.MOVES_ENUM___NULL)
        )
        _ohe_encoder = ONEHOT_ENCODERS[self.cfg.generation]["moves"]
        return mask * self.moves_linear(_ohe_encoder(token))

    def _embed_entity(self, entity: jax.Array):
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

        ev_features = entity[
            np.array(
                [
                    EntityNodeFeature.ENTITY_NODE_FEATURE__EV_HP,
                    EntityNodeFeature.ENTITY_NODE_FEATURE__EV_ATK,
                    EntityNodeFeature.ENTITY_NODE_FEATURE__EV_DEF,
                    EntityNodeFeature.ENTITY_NODE_FEATURE__EV_SPA,
                    EntityNodeFeature.ENTITY_NODE_FEATURE__EV_SPD,
                    EntityNodeFeature.ENTITY_NODE_FEATURE__EV_SPE,
                ]
            )
        ]
        iv_features = entity[
            np.array(
                [
                    EntityNodeFeature.ENTITY_NODE_FEATURE__IV_HP,
                    EntityNodeFeature.ENTITY_NODE_FEATURE__IV_ATK,
                    EntityNodeFeature.ENTITY_NODE_FEATURE__IV_DEF,
                    EntityNodeFeature.ENTITY_NODE_FEATURE__IV_SPA,
                    EntityNodeFeature.ENTITY_NODE_FEATURE__IV_SPD,
                    EntityNodeFeature.ENTITY_NODE_FEATURE__IV_SPE,
                ]
            )
        ]
        reg_boost_features = entity[
            np.array(
                [
                    EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_ATK_VALUE,
                    EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_DEF_VALUE,
                    EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_SPA_VALUE,
                    EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_SPD_VALUE,
                    EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_SPE_VALUE,
                ]
            )
        ]
        spe_boost_features = entity[
            np.array(
                [
                    EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_ACCURACY_VALUE,
                    EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_EVASION_VALUE,
                ]
            )
        ]
        stat_features = jnp.concatenate(
            (
                hp_features,
                ev_features / 255,
                iv_features / 31,
                encode_reg_boosts(reg_boost_features),
                encode_spe_boosts(spe_boost_features),
            ),
            axis=-1,
        )

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
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__NATURE
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
                _encode_one_hot_entity(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__TERA_TYPE
                ),
                _encode_one_hot_entity(
                    entity, EntityNodeFeature.ENTITY_NODE_FEATURE__TERASTALLIZED
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

        move_encodings = jax.vmap(self._embed_move)(move_tokens)

        species_token = entity[EntityNodeFeature.ENTITY_NODE_FEATURE__SPECIES]
        ability_token = entity[EntityNodeFeature.ENTITY_NODE_FEATURE__ABILITY]
        item_token = entity[EntityNodeFeature.ENTITY_NODE_FEATURE__ITEM]
        # last_move_token = entity[EntityNodeFeature.ENTITY_NODE_FEATURE__LAST_MOVE]

        embedding = self.entity_sum(
            boolean_code,
            volatiles_encoding,
            typechange_encoding,
            move_pp_onehot,
            stat_features,
            self._embed_species(species_token),
            self._embed_learnset(species_token),
            self._embed_ability(ability_token),
            self._embed_item(item_token),
            move_encodings.sum(axis=0),
        )

        embedding = embedding / (
            jnp.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-6
        )

        # Apply mask to filter out invalid entities.
        mask = get_entity_mask(entity)
        embedding = mask * embedding

        return embedding, mask

    def _embed_edge(self, edge: jax.Array):
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

        reg_boost_features = edge[
            np.array(
                [
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_ATK_VALUE,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_DEF_VALUE,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_SPA_VALUE,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_SPD_VALUE,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_SPE_VALUE,
                ]
            )
        ]
        spe_boost_features = edge[
            np.array(
                [
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_ACCURACY_VALUE,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_EVASION_VALUE,
                ]
            )
        ]
        stat_features = jnp.concatenate(
            (
                edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__DAMAGE_RATIO, None]
                / MAX_RATIO_TOKEN,
                edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__HEAL_RATIO, None]
                / MAX_RATIO_TOKEN,
                encode_reg_boosts(reg_boost_features),
                encode_spe_boosts(spe_boost_features),
            ),
            axis=-1,
        )

        embedding = self.entity_edge_sum(
            minor_args_encoding,
            boolean_code,
            stat_features,
            self._embed_ability(ability_token),
            self._embed_item(item_token),
            self._embed_move(move_token),
            effect_from_source_embedding,
        )

        embedding = embedding / (
            jnp.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-6
        )

        mask = (
            edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG]
            != BattlemajorargsEnum.BATTLEMAJORARGS_ENUM___UNSPECIFIED
        ) | (minor_args_indices.sum(axis=-1) > 0)

        return embedding, mask

    def _embed_field(self, edge: jax.Array):
        """
        Embed features of the field
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

        embedding = embedding / (
            jnp.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-6
        )

        # Apply mask to filter out invalid edges.
        mask = edge[FieldFeature.FIELD_FEATURE__VALID].astype(jnp.bool)

        return embedding, mask, request_count

    def _embed_entities(self, env_step: PlayerEnvOutput):
        entity_encodings = entity_embeddings = jnp.concatenate(
            (env_step.private_team, env_step.public_team)
        )

        entity_embeddings, entity_mask = jax.vmap(self._embed_entity)(entity_encodings)
        entity_embeddings = self.entity_ff(entity_embeddings)
        contextual_entities = self.entity_encoder(
            entity_embeddings, create_attention_mask(entity_mask)
        )

        return contextual_entities, entity_mask

    # Encode each timestep's features, including nodes and edges.
    def _embed_timestep(self, history: PlayerHistoryOutput):
        """
        Encode features of a single timestep, including entities and edges.
        """

        # Encode nodes.
        history_node_embedding, node_mask = jax.vmap(self._embed_entity)(history.nodes)

        # Encode edges.
        history_edge_embedding, edge_mask = jax.vmap(self._embed_edge)(history.edges)

        # Encode field
        field_embedding, valid_timestep_mask, history_request_count = self._embed_field(
            history.field
        )

        contextual_history_nodes = self.per_timestep_node_encoder(
            history_node_embedding, create_attention_mask(node_mask)
        )

        node_edge_mask = node_mask & edge_mask
        contextual_history_nodes = self.per_timestep_node_edge_encoder(
            contextual_history_nodes + history_edge_embedding,
            create_attention_mask(node_edge_mask),
        )

        timestep_embedding = (
            # More stable for summing over affected historical nodes
            self.timestep_ln(
                node_edge_mask.astype(self.cfg.dtype) @ contextual_history_nodes
            )
            + field_embedding
        )

        return (
            timestep_embedding,
            valid_timestep_mask & edge_mask.any(),
            history_request_count,
        )

    def _embed_timesteps(self, history: PlayerHistoryOutput):
        timestep_embedding, valid_timestep_mask, history_request_count = jax.vmap(
            self._embed_timestep
        )(history)

        timestep_embedding = self.timestep_ff(timestep_embedding)

        # Apply mask to the timestep embeddings.
        timestep_embedding = jnp.where(
            valid_timestep_mask[..., None], timestep_embedding, 0
        )

        seq_len = timestep_embedding.shape[0]

        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))
        contextual_timestep_embedding = self.timestep_encoder(
            timestep_embedding,
            create_attention_mask(valid_timestep_mask)
            & jnp.expand_dims(causal_mask, axis=0),
        )

        return contextual_timestep_embedding, valid_timestep_mask, history_request_count

    # Encode actions for the current environment step.
    def _embed_action(
        self, action: jax.Array, entity_embedding: jax.Array
    ) -> jax.Array:
        """
        Encode features of a move, including its type, species, and action ID.
        """
        boolean_code = one_hot_concat_jax(
            [
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
            entity_embedding,
            self._embed_move(action[MovesetFeature.MOVESET_FEATURE__MOVE_ID]),
        )

        embedding = embedding / (
            jnp.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-6
        )

        return embedding

    def _embed_moves(
        self, moveset: jax.Array, active_embedding: jax.Array, move_mask: jax.Array
    ) -> jax.Array:
        move_embeddings = jax.vmap(self._embed_action)(moveset, active_embedding)
        move_embeddings = self.move_ff(move_embeddings)
        return self.moves_ln(
            self.move_encoder(move_embeddings, create_attention_mask(move_mask))
        )

    def _embed_switches(
        self, switch_embeddings: jax.Array, switch_mask: jax.Array
    ) -> jax.Array:
        switch_embeddings = self.switch_ff(switch_embeddings)
        return self.switch_ln(
            self.switch_encoder(switch_embeddings, create_attention_mask(switch_mask))
        )

    def __call__(self, env_step: PlayerEnvOutput, history_step: PlayerHistoryOutput):
        """
        Forward pass of the Encoder model. Processes an environment step and outputs
        contextual embeddings for actions.
        """

        timestep_embeddings, history_valid_mask, history_request_count = (
            self._embed_timesteps(history_step)
        )
        request_count = env_step.info[..., InfoFeature.INFO_FEATURE__REQUEST_COUNT]
        # For padded timesteps, request count is 0, so we use a large bias value.
        timestep_mask = request_count[..., None] >= jnp.where(
            history_valid_mask, history_request_count, -LARGE_NEGATIVE_BIAS
        )

        def _batched_forward(
            env_step: PlayerEnvOutput,
            timestep_mask: jax.Array,
            current_position: jax.Array,
        ):
            entity_embeddings, entity_mask = self._embed_entities(env_step)
            field_embedding, *_ = self._embed_field(env_step.field)

            entity_embeddings = self.entity_timestep_decoder(
                entity_embeddings + field_embedding,
                timestep_embeddings,
                create_attention_mask(entity_mask, timestep_mask),
                q_positions=current_position,
                kv_positions=history_request_count,
            )
            entity_embeddings = self.entities_ln(entity_embeddings)

            entity_idx = env_step.moveset[
                ..., MovesetFeature.MOVESET_FEATURE__ENTITY_IDX
            ]
            move_embeddings = self._embed_moves(
                env_step.moveset,
                jnp.take(entity_embeddings[:6], entity_idx, axis=0),
                env_step.move_mask,
            )
            switch_embeddings = self._embed_switches(
                entity_embeddings[:6], env_step.switch_mask
            )
            action_embeddings = jnp.concatenate(
                (
                    move_embeddings + self.move_cls_embedding,
                    switch_embeddings + self.switch_cls_embedding,
                ),
                axis=0,
            )
            move_switch_mask = jnp.concatenate(
                (
                    env_step.action_type_mask[0] * env_step.move_mask,
                    env_step.action_type_mask[1:].any(axis=-1) * env_step.switch_mask,
                ),
                axis=-1,
            )
            action_embeddings = self.action_encoder(
                action_embeddings, create_attention_mask(move_switch_mask)
            )

            contextual_entities = self.entity_action_decoder(
                entity_embeddings,
                action_embeddings,
                create_attention_mask(entity_mask, move_switch_mask),
            )
            contextual_actions = self.action_entity_decoder(
                action_embeddings,
                entity_embeddings,
                create_attention_mask(move_switch_mask, entity_mask),
            )

            return contextual_entities, contextual_actions, entity_mask

        contextual_entities, contextual_actions, entity_mask = jax.vmap(
            _batched_forward
        )(env_step, timestep_mask, jnp.expand_dims(request_count, -1))

        return contextual_entities, contextual_actions, entity_mask
