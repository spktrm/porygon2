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
    EntityPrivateNodeFeature,
    EntityPublicNodeFeature,
    EntityRevealedNodeFeature,
    FieldFeature,
    InfoFeature,
    MovesetFeature,
)
from rl.model.modules import (
    GLU,
    GatNet,
    SumEmbeddings,
    Transformer,
    TransformerDecoder,
    TransformerEncoder,
    create_attention_mask,
    one_hot_concat_jax,
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
        self.private_entity_sum = SumEmbeddings(
            output_size=entity_size, dtype=self.cfg.dtype, name="private_entity_sum"
        )
        self.public_entity_sum = SumEmbeddings(
            output_size=entity_size, dtype=self.cfg.dtype, name="public_entity_sum"
        )
        self.public_entity_glu = GLU()
        self.side_embedding = nn.Embed(
            num_embeddings=2, name="side_embedding", **embed_kwargs
        )
        self.edge_sum = SumEmbeddings(
            output_size=entity_size, dtype=self.cfg.dtype, name="entity_edge_sum"
        )
        self.field_sum = SumEmbeddings(
            output_size=entity_size, dtype=self.cfg.dtype, name="field_sum"
        )
        self.move_sum = SumEmbeddings(
            output_size=entity_size, dtype=self.cfg.dtype, name="move_sum"
        )

        self.expanded_entity_merge = SumEmbeddings(
            output_size=entity_size, dtype=self.cfg.dtype, name="expanded_entity_merge"
        )

        # Latent embeddings for global state representation.
        self.latent_state_embeddings = self.param(
            "latent_state_embeddings",
            nn.initializers.truncated_normal(stddev=0.02),
            (self.cfg.num_state_latents, entity_size),
        )

        # Timestep wise graph attention layers
        self.timestep_gat = GatNet(**self.cfg.timestep_gat.to_dict())

        # Transformer encoders for processing sequences of entities and edges.
        self.timestep_encoder = TransformerEncoder(
            **self.cfg.timestep_encoder.to_dict()
        )

        # Transformer Decoders
        self.entity_everything_transformer = Transformer(
            **self.cfg.entity_timestep_transformer.to_dict()
        )
        self.state_pool = TransformerDecoder(**self.cfg.query_pool.to_dict())

        self.move_entity_decoder = TransformerDecoder(
            **self.cfg.action_entity_decoder.to_dict()
        )
        self.switch_entity_decoder = TransformerDecoder(
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

    def _embed_public_entity(self, public: jax.Array, revealed: jax.Array):
        # Encode volatile and type-change indices using the binary encoder.
        _encode_hex = jax.vmap(
            functools.partial(_binary_scale_encoding, world_dim=65535)
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
        )
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

        moves_onehot = jax.nn.one_hot(move_tokens, NUM_MOVES)
        is_valid_move = (move_tokens != MovesEnum.MOVES_ENUM___NULL) & (
            move_tokens != MovesEnum.MOVES_ENUM___UNSPECIFIED
        )
        move_pp_ratios = move_pp_indices[..., None] / 31
        move_pp_onehot = (
            jnp.where(is_valid_move[..., None], moves_onehot * move_pp_ratios, 0)
            .sum(0)
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
                boolean_code,
                volatiles_encoding,
                typechange_encoding,
                move_pp_onehot,
                hp_features,
                encode_reg_boosts(reg_boost_features),
                encode_spe_boosts(spe_boost_features),
            ],
            axis=-1,
        )

        revealed_embedding = self.public_entity_sum(
            self._embed_species(species_token),
            self._embed_learnset(species_token),
            self._embed_ability(ability_token),
            self._embed_item(item_token),
            move_embeddings.sum(axis=0),
        )

        embedding = self.public_entity_glu(public_encoding, revealed_embedding)

        embedding = embedding + self.side_embedding(
            public[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE]
        )

        # Apply mask to filter out invalid entities.
        mask = get_public_entity_mask(revealed)
        embedding = mask * embedding

        return embedding, mask

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

        embedding = self.private_entity_sum(
            boolean_code,
            ev_features / 255,
            iv_features / 31,
            self._embed_species(species_token),
            self._embed_ability(ability_token),
            self._embed_item(item_token),
            move_embeddings.sum(axis=0),
        )

        # Apply mask to filter out invalid entities.
        mask = get_private_entity_mask(private)
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

        embedding = self.edge_sum(
            minor_args_encoding,
            boolean_code,
            stat_features,
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
            ],
            dtype=self.cfg.dtype,
        )

        embedding = self.field_sum(
            boolean_code,
            my_side_condition_encoding,
            opp_side_condition_encoding,
            _binary_scale_encoding(
                to_encode=edge[FieldFeature.FIELD_FEATURE__TURN_ORDER_VALUE],
                world_dim=32,
                dtype=self.cfg.dtype,
            ),
        )

        # Apply mask to filter out invalid edges.
        mask = edge[FieldFeature.FIELD_FEATURE__VALID].astype(jnp.bool)
        embedding = mask * embedding

        return embedding, mask, request_count

    def _embed_public_entities(self, env_step: PlayerEnvOutput):
        return jax.vmap(self._embed_public_entity)(
            env_step.public_team, env_step.revealed_team
        )

    def _embed_private_entities(self, env_step: PlayerEnvOutput):
        return jax.vmap(self._embed_private_entity)(env_step.private_team)

    # Encode each timestep's features, including nodes and edges.
    def _embed_timestep(self, history: PlayerHistoryOutput):
        """
        Encode features of a single timestep, including entities and edges.
        """

        # Encode nodes.
        history_node_embedding, node_mask = jax.vmap(self._embed_public_entity)(
            history.public, history.revealed
        )

        # Encode edges.
        history_edge_embedding, edge_mask = jax.vmap(self._embed_edge)(history.edges)

        # Encode field
        field_embedding, valid_timestep_mask, history_request_count = self._embed_field(
            history.field
        )

        node_edge_mask = node_mask & edge_mask
        side_token = history.public[
            ..., EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE
        ]
        side0_mask = side_token == 0
        side1_mask = side_token == 1

        def agg_nodes(x: jax.Array, m: jax.Array):
            m = m.astype(self.cfg.dtype)
            return m @ x / jnp.maximum(m.sum(axis=0, keepdims=True), 1)

        mean_pool_nodes_per_side = jnp.where(
            side0_mask[..., None],
            agg_nodes(history_node_embedding, side0_mask),
            agg_nodes(history_node_embedding, side1_mask),
        )

        history_node_embedding = self.timestep_gat(
            jnp.concatenate(
                (history_node_embedding, mean_pool_nodes_per_side), axis=-1
            ),
            history_edge_embedding,
            field_embedding,
            node_edge_mask,
        )

        timestep_embedding = agg_nodes(history_node_embedding, node_edge_mask)

        return (
            timestep_embedding,
            valid_timestep_mask & edge_mask.any(),
            history_request_count,
        )

    def _embed_timesteps(self, history: PlayerHistoryOutput):
        timestep_embedding, valid_timestep_mask, history_request_count = jax.vmap(
            self._embed_timestep
        )(history)

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
            ],
            dtype=self.cfg.dtype,
        )
        embedding = self.move_sum(
            boolean_code,
            self._embed_move(action[MovesetFeature.MOVESET_FEATURE__MOVE_ID]),
        )

        return embedding

    def _embed_moves(self, moveset: jax.Array) -> jax.Array:
        return jax.vmap(self._embed_action)(moveset)

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
            history_valid_mask,
            history_request_count,
            jnp.iinfo(request_count.dtype).max,
        )
        timestep_arange = jnp.arange(timestep_mask.shape[-1])

        def _batched_forward(
            env_step: PlayerEnvOutput,
            timestep_mask: jax.Array,
            current_position: jax.Array,
        ):
            entity_embeddings, entity_mask = self._embed_public_entities(env_step)
            field_embedding, *_ = self._embed_field(env_step.field)

            upper_index = jnp.where(timestep_mask, timestep_arange, -1).max(axis=0)

            latest_timestep_indices = upper_index - jnp.arange(32)

            kv_positions = jnp.take(
                history_request_count, latest_timestep_indices, axis=0
            )

            move_mask = env_step.action_type_mask[0] * env_step.move_mask
            switch_mask = env_step.action_type_mask[1] * env_step.switch_mask

            move_embeddings = self._embed_moves(env_step.moveset)
            switch_embeddings, valid_switch_mask = self._embed_private_entities(
                env_step.private_team
            )

            switch_mask = switch_mask * valid_switch_mask

            contextual_entity_embeddings = self.entity_everything_transformer(
                entity_embeddings + field_embedding[None],
                contexts=[
                    (
                        timestep_embeddings,
                        create_attention_mask(entity_mask, timestep_mask),
                        kv_positions,
                    ),
                    (
                        move_embeddings,
                        create_attention_mask(entity_mask, move_mask),
                        jnp.broadcast_to(current_position, move_embeddings.shape[:-1]),
                    ),
                    (
                        switch_embeddings,
                        create_attention_mask(entity_mask, switch_mask),
                        jnp.broadcast_to(
                            current_position, switch_embeddings.shape[:-1]
                        ),
                    ),
                ],
                encoder_attn_mask=create_attention_mask(entity_mask, entity_mask),
                q_positions=jnp.broadcast_to(
                    current_position, entity_embeddings.shape[:-1]
                ),
            )

            latent_state_mask = jnp.ones(
                (self.cfg.num_state_latents,), dtype=entity_mask.dtype
            )
            state_embeddings = self.state_pool(
                self.latent_state_embeddings.astype(self.cfg.dtype),
                contextual_entity_embeddings,
                create_attention_mask(latent_state_mask, entity_mask),
            )

            contextual_moves = self.move_entity_decoder(
                move_embeddings,
                contextual_entity_embeddings,
                create_attention_mask(move_mask, entity_mask),
            )

            contextual_switches = self.switch_entity_decoder(
                switch_embeddings,
                contextual_entity_embeddings,
                create_attention_mask(switch_mask, entity_mask),
            )

            return state_embeddings.reshape(-1), contextual_moves, contextual_switches

        state_query, contextual_moves, contextual_switches = jax.vmap(_batched_forward)(
            env_step, timestep_mask, jnp.expand_dims(request_count, -1)
        )

        return state_query, contextual_moves, contextual_switches
