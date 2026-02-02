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
    ENTITY_PRIVATE_MAX_VALUES,
    ENTITY_PUBLIC_MAX_VALUES,
    FIELD_MAX_VALUES,
    MAX_RATIO_TOKEN,
    NUM_FROM_SOURCE_EFFECTS,
    NUM_MOVES,
    ONEHOT_ENCODERS,
)
from rl.environment.interfaces import (
    PlayerEnvOutput,
    PlayerHistoryOutput,
    PlayerPackedHistoryOutput,
)
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
    EntityPrivateNodeFeature,
    EntityPublicNodeFeature,
    EntityRevealedNodeFeature,
    FieldFeature,
    InfoFeature,
    MovesetFeature,
)
from rl.environment.protos.service_pb2 import ActionEnum
from rl.model.modules import (
    SumEmbeddings,
    TransformerDecoder,
    TransformerEncoder,
    create_attention_mask,
    dense_layer,
    one_hot_concat_jax,
)

MOVE_INDICES = np.array(
    [
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_1,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_2,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_3,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_4,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_1_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_2_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_3_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_4_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_1,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_2,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_3,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_4,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_1_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_2_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_3_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_4_WILDCARD,
    ]
)


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


_encode_one_hot_public_entity = partial(
    _encode_one_hot, max_values=ENTITY_PUBLIC_MAX_VALUES
)
_encode_one_hot_private_entity = partial(
    _encode_one_hot, max_values=ENTITY_PRIVATE_MAX_VALUES
)
_encode_one_hot_action = partial(_encode_one_hot, max_values=ACTION_MAX_VALUES)
_encode_one_hot_edge = partial(_encode_one_hot, max_values=ENTITY_EDGE_MAX_VALUES)
_encode_one_hot_field = partial(_encode_one_hot, max_values=FIELD_MAX_VALUES)
_encode_sqrt_one_hot_public_entity = partial(
    _encode_sqrt_one_hot, max_values=ENTITY_PUBLIC_MAX_VALUES
)
_encode_sqrt_one_hot_action = partial(
    _encode_sqrt_one_hot, max_values=ACTION_MAX_VALUES
)
_encode_divided_one_hot_public_entity = partial(
    _encode_divided_one_hot, max_values=ENTITY_PUBLIC_MAX_VALUES
)
_encode_divided_one_hot_edge = partial(
    _encode_divided_one_hot, max_values=ENTITY_EDGE_MAX_VALUES
)


def get_public_entity_mask(revealed: jax.Array) -> jax.Array:
    """
    Generate a mask to identify valid entities based on species tokens.
    """
    species_token = revealed[
        EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
    ]
    return ~(
        (species_token == SpeciesEnum.SPECIES_ENUM___NULL)
        | (species_token == SpeciesEnum.SPECIES_ENUM___PAD)
        | (species_token == SpeciesEnum.SPECIES_ENUM___UNSPECIFIED)
    )


def get_private_entity_mask(private: jax.Array) -> jax.Array:
    """
    Generate a mask to identify valid entities based on species tokens.
    """
    species_token = private[
        EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SPECIES
    ]
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


def _mean_pool(x: jax.Array, m: jax.Array) -> jax.Array:
    """Mean pool over axis=0 with mask m (same shape as x)."""
    denom = m.sum(axis=0, keepdims=True).clip(min=1)
    return jnp.where(m, x, 0).sum(axis=0, keepdims=True) / denom


def _max_pool(x: jax.Array, m: jax.Array) -> jax.Array:
    """Masked max pool over axis=0; returns minimum dtype value where mask is False."""
    dtype_finfo = jnp.finfo(x.dtype)
    return m.any(axis=0, keepdims=True) * jnp.where(m, x, dtype_finfo.min).max(
        axis=0, keepdims=True
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
        self.wildcard_embedding = nn.Embed(
            num_embeddings=2, name="wildcard_embedding", **embed_kwargs
        )

        # Positional / Modality Embeddings
        self.move_embedding = self.param(
            "move_embedding",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, entity_size),
        )
        self.private_embedding = self.param(
            "private_embedding",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, entity_size),
        )
        self.public_embedding = self.param(
            "public_embedding",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, entity_size),
        )
        self.prev_action_src_embedding = self.param(
            "prev_action_src_embedding",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, entity_size),
        )
        self.prev_action_tgt_embedding = self.param(
            "prev_action_tgt_embedding",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, entity_size),
        )
        self.timestep_embedding = self.param(
            "timestep_embedding",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, entity_size),
        )
        self.switch_positional_embeddings = self.param(
            "switch_positional_embeddings",
            nn.initializers.truncated_normal(stddev=0.02),
            (6, entity_size),
        )
        self.timestep_positional_embeddings = self.param(
            "timestep_positional_embeddings",
            nn.initializers.truncated_normal(stddev=0.02),
            (self.cfg.num_history_timesteps, entity_size),
        )

        # Initialize linear layers for encoding various entity features.
        self.species_linear = dense_layer(
            name="species_linear", use_bias=False, **dense_kwargs
        )
        self.items_linear = dense_layer(
            name="items_linear", use_bias=False, **dense_kwargs
        )
        self.abilities_linear = dense_layer(
            name="abilities_linear", use_bias=False, **dense_kwargs
        )
        self.moves_linear = dense_layer(
            name="moves_linear", use_bias=False, **dense_kwargs
        )
        self.learnset_linear = dense_layer(
            name="learnset_linear", use_bias=False, **dense_kwargs
        )

        # Initialize aggregation modules for combining feature embeddings.
        self.private_entity_sum = SumEmbeddings(
            output_size=entity_size, dtype=self.cfg.dtype, name="private_entity_sum"
        )
        self.public_entity_sum = SumEmbeddings(
            output_size=entity_size, dtype=self.cfg.dtype, name="public_entity_sum"
        )
        self.action_linear = dense_layer(
            features=entity_size, dtype=self.cfg.dtype, name="action_linear"
        )
        self.entity_edge_sum = SumEmbeddings(
            output_size=entity_size, dtype=self.cfg.dtype, name="entity_edge_sum"
        )
        self.field_linear = dense_layer(
            features=entity_size, dtype=self.cfg.dtype, name="field_linear"
        )
        self.side_condition_linear = dense_layer(
            features=entity_size, dtype=self.cfg.dtype, name="side_condition_linear"
        )

        # Timestep wise graph attention layers
        self.local_timestep_cls_embedding = self.param(
            "local_timestep_cls_embedding",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, entity_size),
        )
        self.local_timestep_encoder = TransformerEncoder(
            **self.cfg.timestep_encoder.to_dict()
        )

        # Transformer Decoders
        self.state_embeddings = self.param(
            "state_embeddings",
            nn.initializers.truncated_normal(stddev=0.02),
            (2, entity_size),
        )
        self.extra_embeddings = self.param(
            "extra_embeddings",
            nn.initializers.truncated_normal(stddev=0.02),
            (3, entity_size),
        )
        self.latent_embeddings = self.param(
            "latent_embeddings",
            nn.initializers.truncated_normal(stddev=0.02),
            (self.cfg.num_latent_embeddings, entity_size),
        )
        self.input_decoder = TransformerDecoder(
            **self.cfg.input_decoder.to_dict(),
        )
        self.latent_state_encoder = TransformerEncoder(
            **self.cfg.state_encoder.to_dict()
        )
        self.output_decoder = TransformerDecoder(
            **self.cfg.output_decoder.to_dict(),
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

    def _embed_public_entity(self, public: jax.Array, revealed: jax.Array):
        # Encode volatile and type-change indices using the binary encoder.
        _encode_hex = jax.vmap(
            functools.partial(
                _binary_scale_encoding, dtype=self.cfg.dtype, world_dim=65535
            )
        )
        volatiles_indices = public[
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES0 : EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES8
            + 1
        ]
        volatiles_encoding = _encode_hex(volatiles_indices).reshape(-1)

        typechange_indices = public[
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE0 : EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE1
            + 1
        ]
        typechange_encoding = _encode_hex(typechange_indices).reshape(-1)

        hp_ratio = (
            public[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO]
            / MAX_RATIO_TOKEN
        ).astype(self.cfg.dtype)
        hp_features = jnp.concatenate(
            [
                hp_ratio[..., None],
                jax.nn.one_hot(jnp.floor(32 * hp_ratio), 32, dtype=self.cfg.dtype),
            ],
            axis=-1,
        ).reshape(-1)

        reg_boost_features = public[
            np.array(
                [
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_ATK_VALUE,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_DEF_VALUE,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_SPA_VALUE,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_SPD_VALUE,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_SPE_VALUE,
                ]
            )
        ]
        spe_boost_features = public[
            np.array(
                [
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_ACCURACY_VALUE,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_EVASION_VALUE,
                ]
            )
        ]

        boolean_code = one_hot_concat_jax(
            [
                _encode_sqrt_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__LEVEL,
                    dtype=self.cfg.dtype,
                ),
                _encode_one_hot_public_entity(
                    public, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE
                ),
                _encode_divided_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO,
                    MAX_RATIO_TOKEN / 32,
                ),
                _encode_one_hot_public_entity(
                    public, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__GENDER
                ),
                _encode_one_hot_public_entity(
                    public, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__STATUS
                ),
                _encode_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ITEM_EFFECT,
                ),
                _encode_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BEING_CALLED_BACK,
                ),
                _encode_one_hot_public_entity(
                    public, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TRAPPED
                ),
                _encode_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__NEWLY_SWITCHED,
                ),
                _encode_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TOXIC_TURNS,
                ),
                _encode_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SLEEP_TURNS,
                ),
                _encode_one_hot_public_entity(
                    public, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED
                ),
                _encode_one_hot_public_entity(
                    public, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE
                ),
            ],
            dtype=self.cfg.dtype,
        )

        move_indices = np.array(
            [
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID0,
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID1,
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID2,
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID3,
            ]
        )
        move_pp_indices = np.array(
            [
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP0,
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP1,
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP2,
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP3,
            ]
        )
        move_tokens = revealed[move_indices]
        move_pp_tokens = public[move_pp_indices]

        is_valid_move = (move_tokens != MovesEnum.MOVES_ENUM___NULL) & (
            move_tokens != MovesEnum.MOVES_ENUM___UNSPECIFIED
        )
        move_pp_ratios = is_valid_move * (move_pp_tokens / 31).astype(self.cfg.dtype)
        move_pp_onehot = (
            jnp.zeros(NUM_MOVES, dtype=move_pp_ratios.dtype)
            .at[move_tokens]
            .set(move_pp_ratios)
            .clip(min=0, max=1)
        )

        move_embeddings = jax.vmap(self._embed_move)(move_tokens)

        species_token = revealed[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
        ]
        ability_token = revealed[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ABILITY
        ]
        item_token = revealed[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ITEM
        ]

        public_encoding = jnp.concatenate(
            [
                jnp.concatenate(
                    [
                        boolean_code,
                        volatiles_encoding,
                        typechange_encoding,
                        hp_features,
                    ]
                ),
                move_pp_onehot,
                self._embed_learnset(species_token),
                encode_reg_boosts(reg_boost_features),
                encode_spe_boosts(spe_boost_features),
            ],
            axis=-1,
        )

        revealed_embedding = self.public_entity_sum(
            self._embed_species(species_token),
            self._embed_ability(ability_token),
            self._embed_item(item_token),
            move_embeddings.mean(0),
            public_encoding,
        )

        # Apply mask to filter out invalid entities.
        mask = get_public_entity_mask(revealed)

        return revealed_embedding, mask

    def _embed_private_entity(self, private: jax.Array):
        move_indices = np.array(
            [
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID0,
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID1,
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID2,
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID3,
            ]
        )
        move_tokens = private[move_indices]

        move_embeddings = jax.vmap(self._embed_move)(move_tokens)

        species_token = private[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
        ]
        ability_token = private[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ABILITY
        ]
        item_token = private[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ITEM
        ]

        boolean_code = one_hot_concat_jax(
            [
                _encode_one_hot_private_entity(
                    private,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__NATURE,
                ),
                _encode_one_hot_private_entity(
                    private,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__TERA_TYPE,
                ),
            ],
            dtype=self.cfg.dtype,
        )

        ev_features = private[
            np.array(
                [
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__EV_HP,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__EV_ATK,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__EV_DEF,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__EV_SPA,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__EV_SPD,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__EV_SPE,
                ]
            )
        ]
        iv_features = private[
            np.array(
                [
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__IV_HP,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__IV_ATK,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__IV_DEF,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__IV_SPA,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__IV_SPD,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__IV_SPE,
                ]
            )
        ]

        private_encoding = jnp.concatenate(
            [
                boolean_code,
                (ev_features / 255).astype(self.cfg.dtype),
                (iv_features / 31).astype(self.cfg.dtype),
            ],
            axis=-1,
        )
        private_embedding = self.private_entity_sum(
            self._embed_species(species_token),
            self._embed_ability(ability_token),
            self._embed_item(item_token),
            move_embeddings.mean(0),
            private_encoding,
        )

        # Apply mask to filter out invalid entities.
        mask = get_private_entity_mask(private)

        return private_embedding, mask

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
            ],
            dtype=self.cfg.dtype,
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
            stat_features.astype(self.cfg.dtype),
            self._embed_ability(ability_token),
            self._embed_item(item_token),
            self._embed_move(move_token),
            effect_from_source_embedding,
        )

        mask = (
            edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG]
            != BattlemajorargsEnum.BATTLEMAJORARGS_ENUM___UNSPECIFIED
        ) | (minor_args_indices.sum(axis=-1) > 0)

        embedding = mask * embedding

        return embedding, mask

    def _embed_field(self, field: jax.Array, side_token: jax.Array):
        """
        Embed features of the field
        """
        # Compute turn and request count differences for encoding.

        request_count = field[FieldFeature.FIELD_FEATURE__REQUEST_COUNT]

        _encode_hex = jax.vmap(
            functools.partial(
                _binary_scale_encoding, world_dim=65535, dtype=self.cfg.dtype
            )
        )

        my_side_condition_indices = field[
            FieldFeature.FIELD_FEATURE__MY_SIDECONDITIONS0 : FieldFeature.FIELD_FEATURE__MY_SIDECONDITIONS1
            + 1
        ]
        opp_side_condition_indices = field[
            FieldFeature.FIELD_FEATURE__OPP_SIDECONDITIONS0 : FieldFeature.FIELD_FEATURE__OPP_SIDECONDITIONS1
            + 1
        ]
        my_side_condition_encoding = _encode_hex(my_side_condition_indices).reshape(-1)
        opp_side_condition_encoding = _encode_hex(opp_side_condition_indices).reshape(
            -1
        )

        # Aggregate embeddings for the absolute edge.
        field_encoding = one_hot_concat_jax(
            [
                _encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__WEATHER_ID,
                ),
                _encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__WEATHER_MAX_DURATION,
                ),
                _encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__WEATHER_MIN_DURATION,
                ),
                _encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__TERRAIN_ID,
                ),
                _encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__TERRAIN_MAX_DURATION,
                ),
                _encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__TERRAIN_MIN_DURATION,
                ),
                _encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__PSEUDOWEATHER_ID,
                ),
                _encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__PSEUDOWEATHER_MAX_DURATION,
                ),
                _encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__PSEUDOWEATHER_MIN_DURATION,
                ),
            ],
            dtype=self.cfg.dtype,
        )

        my_side_condition_encoding = jnp.concatenate(
            (
                my_side_condition_encoding,
                one_hot_concat_jax(
                    [
                        _encode_one_hot_field(
                            field,
                            FieldFeature.FIELD_FEATURE__MY_SPIKES,
                        ),
                        _encode_one_hot_field(
                            field,
                            FieldFeature.FIELD_FEATURE__MY_TOXIC_SPIKES,
                        ),
                    ],
                    dtype=self.cfg.dtype,
                ),
            )
        )

        opp_side_condition_encoding = jnp.concatenate(
            (
                opp_side_condition_encoding,
                one_hot_concat_jax(
                    [
                        _encode_one_hot_field(
                            field,
                            FieldFeature.FIELD_FEATURE__OPP_SPIKES,
                        ),
                        _encode_one_hot_field(
                            field,
                            FieldFeature.FIELD_FEATURE__OPP_TOXIC_SPIKES,
                        ),
                    ],
                    dtype=self.cfg.dtype,
                ),
            )
        )

        mask = field[FieldFeature.FIELD_FEATURE__VALID].astype(jnp.bool)[..., None]
        field_embedding = self.field_linear(field_encoding)

        embed_side_con = lambda enc: jnp.where(
            mask, self.side_condition_linear(enc) + field_embedding, 0
        )

        my_field_embedding = embed_side_con(my_side_condition_encoding)
        opp_field_embedding = embed_side_con(opp_side_condition_encoding)

        field_embedding = jnp.where(
            side_token[..., None], my_field_embedding, opp_field_embedding
        )

        return field_embedding, mask, request_count

    def _embed_public_entities(
        self, env_step: PlayerEnvOutput
    ) -> tuple[jax.Array, jax.Array]:
        revealed_embedding, mask = jax.vmap(self._embed_public_entity)(
            env_step.public_team, env_step.revealed_team
        )
        public_team_side_token = env_step.public_team[
            ..., EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE
        ]
        field_embedding, *_ = self._embed_field(env_step.field, public_team_side_token)

        return revealed_embedding + field_embedding, mask

    def _embed_private_entities(self, private_team: jax.Array):
        private_embeddings, mask = jax.vmap(self._embed_private_entity)(private_team)
        return private_embeddings

    # Encode each timestep's features, including nodes and edges.
    def _embed_local_timestep(
        self,
        history: PlayerHistoryOutput,
        entity_embedding_cache: jax.Array,
        edge_embedding_cache: jax.Array,
        side_token_cache: jax.Array,
    ):
        """
        Encode features of a single timestep, including entities and edges.
        """
        relevant_indices = history.field[
            np.array(
                [
                    FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX0,
                    FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX1,
                    FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX2,
                    FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX3,
                    # FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX4,
                    # FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX5,
                    # FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX6,
                    # FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX7,
                ]
            )
        ]

        # Encode field
        side_token = jnp.take(side_token_cache, relevant_indices, axis=0)
        field_embeddings, valid_timestep_mask, history_request_count = (
            self._embed_field(history.field, side_token)
        )

        num_relevant = history.field[FieldFeature.FIELD_FEATURE__NUM_RELEVANT]
        node_edge_mask = jnp.arange(relevant_indices.shape[0]) < num_relevant
        valid_timestep_mask = (valid_timestep_mask & node_edge_mask.any()).squeeze()

        node_embeddings = jnp.take(entity_embedding_cache, relevant_indices, axis=0)
        edge_embeddings = jnp.take(edge_embedding_cache, relevant_indices, axis=0)

        local_sequence = jnp.concatenate(
            (
                self.local_timestep_cls_embedding.astype(self.cfg.dtype),
                node_embeddings + edge_embeddings + field_embeddings,
            ),
            axis=0,
        )
        sequence_mask = jnp.insert(node_edge_mask, 0, True)

        local_sequence = self.local_timestep_encoder(
            local_sequence,
            attn_mask=create_attention_mask(sequence_mask),
            qkv_positions=jnp.arange(local_sequence.shape[0]),
        )

        # Extract the timestep embedding corresponding to the CLS token.
        local_timestep_embedding = local_sequence[0]

        return local_timestep_embedding, valid_timestep_mask, history_request_count

    def _embed_global_timestep(
        self, history: PlayerHistoryOutput, packed_history: PlayerPackedHistoryOutput
    ):

        entity_embedding_cache, _ = jax.vmap(self._embed_public_entity)(
            packed_history.public, packed_history.revealed
        )

        edge_embedding_cache, _ = jax.vmap(self._embed_edge)(packed_history.edges)

        (
            local_timestep_embedding,
            valid_timestep_mask,
            history_request_count,
        ) = jax.vmap(self._embed_local_timestep, in_axes=(0, None, None, None))(
            history,
            entity_embedding_cache,
            edge_embedding_cache,
            packed_history.public[
                ..., EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE
            ],
        )

        # Apply mask to the timestep embeddings.
        local_timestep_embedding = jnp.where(
            valid_timestep_mask[..., None], local_timestep_embedding, 0
        )

        seq_len = local_timestep_embedding.shape[0]
        timestep_positions = jnp.arange(seq_len)

        # causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))
        # attn_mask = create_attention_mask(valid_timestep_mask)
        # attn_mask = attn_mask & jnp.expand_dims(causal_mask, axis=0)

        # global_timestep_embedding = self.global_timestep_encoder(
        #     local_timestep_embedding,
        #     attn_mask=attn_mask,
        #     qkv_positions=timestep_positions,
        # )

        return (
            local_timestep_embedding,
            valid_timestep_mask,
            history_request_count,
            timestep_positions,
        )

    # Encode actions for the current environment step.
    def _embed_action(self, action: jax.Array) -> jax.Array:
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
                _encode_one_hot_action(
                    action, MovesetFeature.MOVESET_FEATURE__DISABLED
                ),
            ],
            dtype=self.cfg.dtype,
        )
        embedding = (
            self._embed_move(action[MovesetFeature.MOVESET_FEATURE__MOVE_ID])
            + self.action_linear(boolean_code)
            + self.wildcard_embedding(
                action[MovesetFeature.MOVESET_FEATURE__IS_WILDCARD]
            )
        )

        mask = ~(
            (
                action[MovesetFeature.MOVESET_FEATURE__MOVE_ID]
                == MovesEnum.MOVES_ENUM___NULL
            )
            | (
                action[MovesetFeature.MOVESET_FEATURE__MOVE_ID]
                == MovesEnum.MOVES_ENUM___PAD
            )
        )

        return embedding, mask

    def _embed_moves(self, moveset: jax.Array) -> jax.Array:
        return jax.vmap(self._embed_action)(moveset)

    def _get_latest_timestep_embeddings(
        self,
        timestep_embeddings: jax.Array,
        timestep_mask: jax.Array,
        timestep_positions: jax.Array,
        timestep_arange: jax.Array,
        num_timesteps: int = 24,
    ):
        upper_index = jnp.where(timestep_mask, timestep_arange, -1).max(axis=0)
        latest_timestep_indices = upper_index - jnp.arange(num_timesteps)
        latest_timestep_indices_clipped = latest_timestep_indices.clip(min=0)
        latest_timestep_embeddings = jnp.take(
            timestep_embeddings, latest_timestep_indices_clipped, axis=0
        )
        latest_timestep_mask = jnp.take(
            timestep_mask, latest_timestep_indices_clipped, axis=0
        ) & (latest_timestep_indices >= 0)
        latest_timestep_positions = jnp.take(
            timestep_positions, latest_timestep_indices_clipped, axis=0
        )
        return (
            latest_timestep_embeddings,
            latest_timestep_mask,
            latest_timestep_positions,
        )

    def _batched_forward(
        self,
        env_step: PlayerEnvOutput,
        timestep_mask: jax.Array,
        current_position: jax.Array,
        timestep_embeddings: jax.Array,
        timestep_positions: jax.Array,
        timestep_arange: jax.Array,
        private_embeddings: jax.Array,
    ):
        entity_embeddings, entity_mask = self._embed_public_entities(env_step)

        move_embeddings, move_mask = self._embed_moves(env_step.moveset)
        move_mask = move_mask & jnp.take(
            env_step.action_mask, MOVE_INDICES, axis=0
        ).any(axis=-1)

        private_mask = jnp.ones_like(private_embeddings[..., 0], dtype=jnp.bool)

        latest_timestep_embeddings, latest_timestep_mask, latest_timestep_positions = (
            self._get_latest_timestep_embeddings(
                timestep_embeddings,
                timestep_mask,
                timestep_positions,
                timestep_arange,
                self.cfg.num_history_timesteps,
            )
        )
        switch_order_indices = np.array(
            [
                InfoFeature.INFO_FEATURE__SWITCH_ORDER_VALUE0,
                InfoFeature.INFO_FEATURE__SWITCH_ORDER_VALUE1,
                InfoFeature.INFO_FEATURE__SWITCH_ORDER_VALUE2,
                InfoFeature.INFO_FEATURE__SWITCH_ORDER_VALUE3,
                InfoFeature.INFO_FEATURE__SWITCH_ORDER_VALUE4,
                InfoFeature.INFO_FEATURE__SWITCH_ORDER_VALUE5,
            ]
        )
        switch_order_values = env_step.info[switch_order_indices]
        private_embeddings = jnp.take(private_embeddings, switch_order_values, axis=0)

        output_state_sequence = jnp.concatenate(
            [
                self.state_embeddings.astype(self.cfg.dtype),
                move_embeddings,
                private_embeddings,
                entity_embeddings[:2],
                entity_embeddings[6:8],
                self.extra_embeddings.astype(self.cfg.dtype),
            ],
            axis=0,
        )

        has_prev_action = env_step.info[
            InfoFeature.INFO_FEATURE__HAS_PREV_ACTION
        ].reshape(1, 1)
        prev_action_src = has_prev_action * jnp.take(
            output_state_sequence,
            env_step.info[InfoFeature.INFO_FEATURE__PREV_ACTION_SRC],
            axis=0,
        )
        prev_action_tgt = has_prev_action * jnp.take(
            output_state_sequence,
            env_step.info[InfoFeature.INFO_FEATURE__PREV_ACTION_TGT],
            axis=0,
        )

        move_embeddings = move_embeddings + self.move_embedding.astype(self.cfg.dtype)
        private_embeddings = (
            private_embeddings
            + self.switch_positional_embeddings.astype(self.cfg.dtype)
            + self.private_embedding.astype(self.cfg.dtype)
        )
        public_embeddings = entity_embeddings + self.public_embedding.astype(
            self.cfg.dtype
        )
        prev_action_src_embedding = (
            prev_action_src + self.prev_action_src_embedding.astype(self.cfg.dtype)
        )
        prev_action_tgt_embedding = (
            prev_action_tgt + self.prev_action_tgt_embedding.astype(self.cfg.dtype)
        )
        history_embeddings = (
            latest_timestep_embeddings
            + self.timestep_embedding.astype(self.cfg.dtype)
            + self.timestep_positional_embeddings.astype(self.cfg.dtype)
        )

        input_state_sequence = jnp.concatenate(
            (
                move_embeddings,
                private_embeddings,
                public_embeddings,
                prev_action_src_embedding,
                prev_action_tgt_embedding,
                history_embeddings,
            ),
            axis=0,
        )

        state_mask = jnp.concatenate(
            (
                move_mask,
                private_mask,
                entity_mask,
                jnp.ones_like(move_mask[:2], dtype=jnp.bool),
                latest_timestep_mask,
            )
        )
        latent_mask = jnp.ones_like(self.latent_embeddings[..., 0], dtype=jnp.bool)

        latent_state_embeddings = self.latent_embeddings.astype(self.cfg.dtype)
        for _ in range(self.cfg.num_perceiver_steps):
            latent_state_embeddings = self.input_decoder(
                q=latent_state_embeddings,
                kv=input_state_sequence,
                attn_mask=create_attention_mask(latent_mask, state_mask),
            )

            for _ in range(self.cfg.num_thinking_steps):
                latent_state_embeddings = self.latent_state_encoder(
                    qkv=latent_state_embeddings,
                    attn_mask=create_attention_mask(latent_mask),
                )

        output_state_embeddings = self.output_decoder(
            output_state_sequence, latent_state_embeddings
        )

        state_embeddings = output_state_embeddings[:2]
        action_embeddings = output_state_embeddings

        state_embedding = state_embeddings.reshape(-1)

        return state_embedding, action_embeddings

    def __call__(
        self,
        env_step: PlayerEnvOutput,
        packed_history_step: PlayerHistoryOutput,
        history_step: PlayerHistoryOutput,
    ):
        (
            timestep_embeddings,
            history_valid_mask,
            history_request_count,
            timestep_positions,
        ) = self._embed_global_timestep(history_step, packed_history_step)

        request_count = env_step.info[..., InfoFeature.INFO_FEATURE__REQUEST_COUNT]
        # For padded timesteps, request count is 0, so we use a large bias value.
        timestep_mask = request_count[..., None] >= jnp.where(
            history_valid_mask,
            history_request_count,
            jnp.iinfo(request_count.dtype).max,
        )
        timestep_arange = jnp.arange(timestep_mask.shape[-1])

        switch_embeddings = self._embed_private_entities(env_step.private_team[0])

        state_embedding, action_embeddings = jax.vmap(
            self._batched_forward, in_axes=(0, 0, 0, None, None, None, None)
        )(
            env_step,
            timestep_mask,
            jnp.expand_dims(request_count, -1),
            timestep_embeddings,
            timestep_positions,
            timestep_arange,
            switch_embeddings,
        )

        return state_embedding, action_embeddings
