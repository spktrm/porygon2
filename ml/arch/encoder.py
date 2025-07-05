import math
from functools import partial
from typing import Dict, Mapping, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from ml.arch.modules import (
    MLP,
    BinaryEncoder,
    FeedForwardResidual,
    MergeEmbeddings,
    PretrainedEmbedding,
    SumEmbeddings,
    TransformerDecoder,
    TransformerEncoder,
    one_hot_concat_jax,
)
from rlenv.data import (
    ABSOLUTE_EDGE_MAX_VALUES,
    ACTION_MAX_VALUES,
    ENTITY_MAX_VALUES,
    MAX_RATIO_TOKEN,
    NUM_FROM_SOURCE_EFFECTS,
    NUM_MOVES,
    RELATIVE_EDGE_MAX_VALUES,
)
from rlenv.interfaces import EnvStep, HistoryContainer, HistoryStep
from rlenv.protos.enums_pb2 import (
    AbilitiesEnum,
    ActionsEnum,
    EffectEnum,
    ItemsEnum,
    MovesEnum,
    SpeciesEnum,
)
from rlenv.protos.features_pb2 import (
    AbsoluteEdgeFeature,
    EntityFeature,
    MovesetFeature,
    RelativeEdgeFeature,
)

# Load pretrained embeddings for various features.
SPECIES_ONEHOT = PretrainedEmbedding(fpath="data/data/gen3/species.npy")
ABILITY_ONEHOT = PretrainedEmbedding(fpath="data/data/gen3/abilities.npy")
ITEM_ONEHOT = PretrainedEmbedding(fpath="data/data/gen3/items.npy")
MOVE_ONEHOT = PretrainedEmbedding(fpath="data/data/gen3/moves.npy")

# # Initialize a binary encoder for specific features.
OCT_ENCODER = BinaryEncoder(num_bits=8)
HEX_ENCODER = BinaryEncoder(num_bits=16)


def get_move_mask(move: chex.Array) -> chex.Array:
    """
    Generate a mask to filter valid moves based on move identifiers.
    """
    action_id_token = move[MovesetFeature.MOVESET_FEATURE__ACTION_ID].astype(jnp.int32)
    return ~(
        (action_id_token == ActionsEnum.ACTIONS_ENUM__MOVE__NULL)
        | (action_id_token == ActionsEnum.ACTIONS_ENUM__SWITCH__NULL)
        | (action_id_token == ActionsEnum.ACTIONS_ENUM__MOVE__PAD)
        | (action_id_token == ActionsEnum.ACTIONS_ENUM__SWITCH__PAD)
    )


def _binary_scale_encoding(to_encode: chex.Array, world_dim: int) -> chex.Array:
    """Encode the feature using its binary representation."""
    chex.assert_rank(to_encode, 0)
    chex.assert_type(to_encode, jnp.int32)
    num_bits = (world_dim - 1).bit_length()
    bit_mask = 1 << np.arange(num_bits)
    pos = jnp.broadcast_to(to_encode[jnp.newaxis], num_bits)
    result = jnp.not_equal(jnp.bitwise_and(pos, bit_mask), 0)
    return result.astype(jnp.float32)


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
    entity: chex.Array, feature_idx: int, max_values: Dict[int, int]
) -> Tuple[int, int]:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    max_value = max_values[feature_idx]
    max_sqrt_value = int(math.floor(math.sqrt(max_value)))
    x = jnp.floor(jnp.sqrt(entity[feature_idx].astype(jnp.float32)))
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


def _features_embedding(
    raw_unit: chex.Array, rescales: Mapping[int, float]
) -> chex.Array:
    """Select features in `rescales`, rescale and concatenate them."""
    chex.assert_rank(raw_unit, 1)
    chex.assert_type(raw_unit, jnp.int32)
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
        consecutive_features = raw_unit[
            feature_indices[i_min] : feature_indices[i_max] + 1
        ]
        consecutive_rescales = jnp.asarray(
            [rescales[feature_indices[i]] for i in range(i_min, i_max + 1)], jnp.float32
        )
        i_min = i_max + 1
        rescaled_features = jnp.multiply(consecutive_features, consecutive_rescales)
        selected_features.append(rescaled_features)
    return jnp.concatenate(selected_features, axis=0).astype(jnp.float32)


_encode_one_hot_entity = partial(_encode_one_hot, max_values=ENTITY_MAX_VALUES)
_encode_one_hot_action = partial(_encode_one_hot, max_values=ACTION_MAX_VALUES)
_encode_one_hot_relative_edge = partial(
    _encode_one_hot, max_values=RELATIVE_EDGE_MAX_VALUES
)
_encode_one_hot_absolute_edge = partial(
    _encode_one_hot, max_values=ABSOLUTE_EDGE_MAX_VALUES
)
_encode_one_hot_entity_boost = partial(_encode_one_hot_entity, value_offset=6)
_encode_one_hot_relative_edge_boost = partial(
    _encode_one_hot_relative_edge, value_offset=6
)
_encode_sqrt_one_hot_entity = partial(
    _encode_sqrt_one_hot, max_values=ENTITY_MAX_VALUES
)
_encode_sqrt_one_hot_action = partial(
    _encode_sqrt_one_hot, max_values=ACTION_MAX_VALUES
)
_encode_divided_one_hot_entity = partial(
    _encode_divided_one_hot, max_values=ENTITY_MAX_VALUES
)
_encode_divided_one_hot_relative_edge = partial(
    _encode_divided_one_hot, max_values=RELATIVE_EDGE_MAX_VALUES
)


def get_entity_mask(entity: chex.Array) -> chex.Array:
    """
    Generate a mask to identify valid entities based on species tokens.
    """
    species_token = entity[EntityFeature.ENTITY_FEATURE__SPECIES].astype(jnp.int32)
    return ~(
        (species_token == SpeciesEnum.SPECIES_ENUM___NULL)
        | (species_token == SpeciesEnum.SPECIES_ENUM___PAD)
        | (species_token == SpeciesEnum.SPECIES_ENUM___UNSPECIFIED)
    )


def get_edge_mask(edge: chex.Array) -> chex.Array:
    """
    Generate a mask for edges based on their validity tokens.
    """
    return edge[AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__VALID].astype(jnp.int32)


class Encoder(nn.Module):
    """
    Encoder model for processing environment steps and history to generate embeddings.
    """

    cfg: ConfigDict

    def setup(self):

        # Extract configuration parameters for embedding sizes.
        entity_size = self.cfg.entity_size

        embed_kwargs = dense_kwargs = dict(features=entity_size)

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
        self.entity_sum = SumEmbeddings(output_size=entity_size, name="entity_sum")
        self.relative_edge_sum = SumEmbeddings(
            output_size=entity_size, name="relative_edge_sum"
        )
        self.absolute_edge_sum = SumEmbeddings(
            output_size=entity_size, name="absolute_edge_sum"
        )
        self.timestep_linear = MLP(layer_sizes=(entity_size,), use_layer_norm=True)
        self.action_sum = SumEmbeddings(output_size=entity_size, name="action_sum")
        self.latent_merge = MergeEmbeddings(
            output_size=entity_size, name="latent_merge"
        )

        # Feed-forward layers for processing entity and timestep features.
        self.entity_ff = FeedForwardResidual(hidden_dim=entity_size)
        self.timestep_ff = FeedForwardResidual(hidden_dim=entity_size)
        self.action_ff = FeedForwardResidual(hidden_dim=entity_size)

        # Transformer encoders for processing sequences of entities and edges.
        self.entity_encoder = TransformerEncoder(**self.cfg.entity_encoder.to_dict())
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
            nn.initializers.normal(),
            (self.cfg.num_latents, entity_size),
        )

    def _encode_species(self, token: chex.Array):
        return jnp.where(
            token == SpeciesEnum.SPECIES_ENUM___UNSPECIFIED,
            0,
            self.species_linear(SPECIES_ONEHOT(token)),
        )

    def _encode_item(self, token: chex.Array):
        return jnp.where(
            token == ItemsEnum.ITEMS_ENUM___UNSPECIFIED,
            0,
            self.items_linear(ITEM_ONEHOT(token)),
        )

    def _encode_ability(self, token: chex.Array):
        return jnp.where(
            token == AbilitiesEnum.ABILITIES_ENUM___UNSPECIFIED,
            0,
            self.abilities_linear(ABILITY_ONEHOT(token)),
        )

    def _encode_move(self, token: chex.Array):
        return jnp.where(
            token == MovesEnum.MOVES_ENUM___UNSPECIFIED,
            0,
            self.moves_linear(MOVE_ONEHOT(token)),
        )

    def _encode_effect_from_source(self, token: chex.Array):
        return self.effect_from_source_embedding(token)

    def _encode_entity(self, entity: chex.Array):
        # Encode volatile and type-change indices using the binary encoder.
        volatiles_indices = entity[
            EntityFeature.ENTITY_FEATURE__VOLATILES0 : EntityFeature.ENTITY_FEATURE__VOLATILES8
            + 1
        ]
        volatiles_encoding = HEX_ENCODER(volatiles_indices.astype(jnp.uint16)).reshape(
            -1
        )

        typechange_indices = entity[
            EntityFeature.ENTITY_FEATURE__TYPECHANGE0 : EntityFeature.ENTITY_FEATURE__TYPECHANGE1
            + 1
        ]
        typechange_encoding = HEX_ENCODER(
            typechange_indices.astype(jnp.uint16)
        ).reshape(-1)

        hp_ratio_token = entity[EntityFeature.ENTITY_FEATURE__HP_RATIO]
        hp_ratio = entity[EntityFeature.ENTITY_FEATURE__HP_RATIO] / MAX_RATIO_TOKEN
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
                    entity, EntityFeature.ENTITY_FEATURE__LEVEL
                ),
                _encode_one_hot_entity(entity, EntityFeature.ENTITY_FEATURE__ACTIVE),
                _encode_one_hot_entity(entity, EntityFeature.ENTITY_FEATURE__SIDE),
                _encode_one_hot_entity(entity, EntityFeature.ENTITY_FEATURE__IS_PUBLIC),
                _encode_divided_one_hot_entity(
                    entity,
                    EntityFeature.ENTITY_FEATURE__HP_RATIO,
                    MAX_RATIO_TOKEN / 32,
                ),
                _encode_one_hot_entity(entity, EntityFeature.ENTITY_FEATURE__GENDER),
                _encode_one_hot_entity(entity, EntityFeature.ENTITY_FEATURE__STATUS),
                _encode_one_hot_entity(
                    entity, EntityFeature.ENTITY_FEATURE__ITEM_EFFECT
                ),
                _encode_one_hot_entity(
                    entity, EntityFeature.ENTITY_FEATURE__BEING_CALLED_BACK
                ),
                _encode_one_hot_entity(entity, EntityFeature.ENTITY_FEATURE__TRAPPED),
                _encode_one_hot_entity(
                    entity, EntityFeature.ENTITY_FEATURE__NEWLY_SWITCHED
                ),
                _encode_one_hot_entity(
                    entity, EntityFeature.ENTITY_FEATURE__TOXIC_TURNS
                ),
                _encode_one_hot_entity(
                    entity, EntityFeature.ENTITY_FEATURE__SLEEP_TURNS
                ),
                _encode_one_hot_entity(entity, EntityFeature.ENTITY_FEATURE__FAINTED),
                _encode_one_hot_entity_boost(
                    entity, EntityFeature.ENTITY_FEATURE__BOOST_ATK_VALUE
                ),
                _encode_one_hot_entity_boost(
                    entity, EntityFeature.ENTITY_FEATURE__BOOST_DEF_VALUE
                ),
                _encode_one_hot_entity_boost(
                    entity, EntityFeature.ENTITY_FEATURE__BOOST_SPA_VALUE
                ),
                _encode_one_hot_entity_boost(
                    entity, EntityFeature.ENTITY_FEATURE__BOOST_SPD_VALUE
                ),
                _encode_one_hot_entity_boost(
                    entity, EntityFeature.ENTITY_FEATURE__BOOST_SPE_VALUE
                ),
                _encode_one_hot_entity_boost(
                    entity, EntityFeature.ENTITY_FEATURE__BOOST_EVASION_VALUE
                ),
                _encode_one_hot_entity_boost(
                    entity, EntityFeature.ENTITY_FEATURE__BOOST_ACCURACY_VALUE
                ),
            ]
        )

        move_indices = np.array(
            [
                EntityFeature.ENTITY_FEATURE__MOVEID0,
                EntityFeature.ENTITY_FEATURE__MOVEID1,
                EntityFeature.ENTITY_FEATURE__MOVEID2,
                EntityFeature.ENTITY_FEATURE__MOVEID3,
            ]
        )
        move_pp_indices = np.array(
            [
                EntityFeature.ENTITY_FEATURE__MOVEPP0,
                EntityFeature.ENTITY_FEATURE__MOVEPP1,
                EntityFeature.ENTITY_FEATURE__MOVEPP2,
                EntityFeature.ENTITY_FEATURE__MOVEPP3,
            ]
        )
        move_tokens = entity[move_indices]

        move_pp_onehot = (
            jnp.where(
                (
                    (move_tokens != MovesEnum.MOVES_ENUM___NULL)
                    | (move_tokens != MovesEnum.MOVES_ENUM___UNSPECIFIED)
                )[..., None],
                jax.nn.one_hot(move_tokens, NUM_MOVES)
                * move_pp_indices[..., None]
                / 31,
                0,
            )
            .sum(0)
            .clip(min=0, max=1)
        )

        move_encodings = jax.vmap(self._encode_move)(move_tokens)

        species_token = entity[EntityFeature.ENTITY_FEATURE__SPECIES]
        ability_token = entity[EntityFeature.ENTITY_FEATURE__ABILITY]
        item_token = entity[EntityFeature.ENTITY_FEATURE__ITEM]
        # last_move_token = entity[EntityFeature.ENTITY_FEATURE__LAST_MOVE]

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
        embedding = jnp.where(mask, embedding, 0)

        return embedding, mask

    def _encode_relative_edge(self, edge: chex.Array) -> chex.Array:
        # Encode minor arguments and side conditions using the binary encoder.
        minor_args_indices = edge[
            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MINOR_ARG0 : RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MINOR_ARG3
            + 1
        ]
        minor_args_encoding = HEX_ENCODER(
            minor_args_indices.astype(jnp.uint16)
        ).reshape(-1)

        side_condition_indices = edge[
            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SIDECONDITIONS0 : RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SIDECONDITIONS1
            + 1
        ]
        side_condition_encoding = HEX_ENCODER(
            side_condition_indices.astype(jnp.uint16)
        ).reshape(-1)

        # Aggregate embeddings for the relative edge.
        boolean_code = one_hot_concat_jax(
            [
                _encode_one_hot_relative_edge(
                    edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MAJOR_ARG,
                ),
                _encode_divided_one_hot_relative_edge(
                    edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__DAMAGE_RATIO,
                    MAX_RATIO_TOKEN / 32,
                ),
                _encode_divided_one_hot_relative_edge(
                    edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__HEAL_RATIO,
                    MAX_RATIO_TOKEN / 32,
                ),
                _encode_one_hot_relative_edge(
                    edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__STATUS_TOKEN,
                ),
                _encode_one_hot_relative_edge_boost(
                    edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_ATK_VALUE,
                ),
                _encode_one_hot_relative_edge_boost(
                    edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_DEF_VALUE,
                ),
                _encode_one_hot_relative_edge_boost(
                    edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_SPA_VALUE,
                ),
                _encode_one_hot_relative_edge_boost(
                    edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_SPD_VALUE,
                ),
                _encode_one_hot_relative_edge_boost(
                    edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_SPE_VALUE,
                ),
                _encode_one_hot_relative_edge_boost(
                    edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_EVASION_VALUE,
                ),
                _encode_one_hot_relative_edge_boost(
                    edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_ACCURACY_VALUE,
                ),
                _encode_one_hot_relative_edge(
                    edge, RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SPIKES
                ),
                _encode_one_hot_relative_edge(
                    edge,
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__TOXIC_SPIKES,
                ),
            ]
        )

        effect_from_source_indices = np.array(
            [
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN0,
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN1,
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN2,
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN3,
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN4,
            ]
        )
        effect_from_source_tokens = edge[effect_from_source_indices]
        effect_from_source_embeddings = jax.vmap(self._encode_effect_from_source)(
            effect_from_source_tokens
        )
        effect_from_source_mask = (
            effect_from_source_tokens != EffectEnum.EFFECT_ENUM___UNSPECIFIED
        ) & (effect_from_source_tokens != EffectEnum.EFFECT_ENUM___NULL)
        effect_from_source_embedding = jnp.where(
            effect_from_source_mask[..., None], effect_from_source_embeddings, 0
        ).sum(axis=0)

        ability_token = edge[RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ABILITY_TOKEN]
        item_token = edge[RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ITEM_TOKEN]
        move_token = edge[RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MOVE_TOKEN]
        edge[RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ACTION_TOKEN]

        embedding = self.relative_edge_sum(
            minor_args_encoding,
            side_condition_encoding,
            boolean_code,
            self._encode_ability(ability_token),
            self._encode_item(item_token),
            self._encode_move(move_token),
            effect_from_source_embedding,
        )

        return embedding

    def _encode_absolute_edge(self, edge: chex.Array) -> chex.Array:
        """
        Encode features of an absolute edge, including turn and request offsets.
        """
        # Compute turn and request count differences for encoding.
        edge[AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TURN_VALUE]
        request_count = edge[AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__REQUEST_COUNT]

        # Aggregate embeddings for the absolute edge.
        boolean_code = one_hot_concat_jax(
            [
                _encode_one_hot_absolute_edge(
                    edge,
                    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_ID,
                ),
                _encode_one_hot_absolute_edge(
                    edge,
                    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_MAX_DURATION,
                ),
                _encode_one_hot_absolute_edge(
                    edge,
                    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_MIN_DURATION,
                ),
                _encode_one_hot_absolute_edge(
                    edge,
                    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TERRAIN_ID,
                ),
                _encode_one_hot_absolute_edge(
                    edge,
                    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TERRAIN_MAX_DURATION,
                ),
                _encode_one_hot_absolute_edge(
                    edge,
                    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TERRAIN_MIN_DURATION,
                ),
                _encode_one_hot_absolute_edge(
                    edge,
                    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__PSEUDOWEATHER_ID,
                ),
                _encode_one_hot_absolute_edge(
                    edge,
                    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__PSEUDOWEATHER_MAX_DURATION,
                ),
                _encode_one_hot_absolute_edge(
                    edge,
                    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__PSEUDOWEATHER_MIN_DURATION,
                ),
            ]
        )

        embedding = self.absolute_edge_sum(
            boolean_code,
            _binary_scale_encoding(
                edge[AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TURN_ORDER_VALUE],
                32,
            ),
        )

        # Apply mask to filter out invalid edges.
        mask = get_edge_mask(edge)
        request_count = jnp.where(mask, request_count, 1e9)

        return embedding, mask, request_count

    # Encode each timestep's features, including nodes and edges.
    def _encode_timestep(self, history_container: HistoryContainer):
        """
        Encode features of a single timestep, including entities and edges.
        """

        # Encode entities.
        entity_embedding, _ = jax.vmap(self._encode_entity)(history_container.entities)

        # Encode relative edges.
        relative_edge_embedding = jax.vmap(self._encode_relative_edge)(
            history_container.relative_edges
        )

        # Encode absolute edges.
        absolute_edge_embedding, valid_timestep_mask, history_request_count = (
            self._encode_absolute_edge(history_container.absolute_edges)
        )

        timestep_embedding = (
            entity_embedding + relative_edge_embedding + absolute_edge_embedding[None]
        ).reshape(-1)

        timestep_embedding = self.timestep_linear(timestep_embedding)
        timestep_embedding = self.timestep_ff(timestep_embedding)

        # Apply mask to the timestep embeddings.
        timestep_embedding = jnp.where(
            valid_timestep_mask[..., None], timestep_embedding, 0
        )

        return timestep_embedding, valid_timestep_mask, history_request_count

    def _encode_entities(self, env_step: EnvStep):
        _encode_entities = jax.vmap(jax.vmap(self._encode_entity))
        private_entity_embeddings, private_entity_mask = _encode_entities(
            env_step.private_team
        )
        public_entity_embeddings, public_entity_mask = _encode_entities(
            env_step.public_team
        )

        entity_mask = jnp.concatenate(
            (private_entity_mask, public_entity_mask), axis=-1
        )
        entity_embeddings = jnp.concatenate(
            (private_entity_embeddings, public_entity_embeddings), axis=-2
        )
        entity_embeddings = jax.vmap(self.entity_ff)(entity_embeddings)
        entity_embeddings = jax.vmap(self.entity_encoder)(
            entity_embeddings, entity_mask
        )
        return entity_embeddings[:, :6], entity_mask, entity_embeddings

    def _encode_timesteps(self, history_container: HistoryContainer):
        timestep_embedding, valid_timestep_mask, history_request_count = jax.vmap(
            self._encode_timestep
        )(history_container)

        seq_len = timestep_embedding.shape[0]

        timestep_embedding = self.timestep_encoder(
            timestep_embedding,
            valid_timestep_mask,
            jnp.tril(jnp.ones((seq_len, seq_len))),
        )

        return timestep_embedding, history_request_count

    # Encode actions for the current environment step.
    def _encode_action(
        self, action: chex.Array, legal: chex.Array, entity_embedding: chex.Array
    ) -> chex.Array:
        """
        Encode features of a move, including its type, species, and action ID.
        """
        boolean_code = one_hot_concat_jax(
            [
                _encode_one_hot_action(
                    action, MovesetFeature.MOVESET_FEATURE__ACTION_TYPE
                ),
                _encode_sqrt_one_hot_action(action, MovesetFeature.MOVESET_FEATURE__PP),
                _encode_sqrt_one_hot_action(
                    action, MovesetFeature.MOVESET_FEATURE__MAXPP
                ),
                _encode_one_hot_action(action, MovesetFeature.MOVESET_FEATURE__HAS_PP),
            ]
        )

        embedding = self.action_sum(
            boolean_code,
            entity_embedding[action[MovesetFeature.MOVESET_FEATURE__ENTITY_IDX]],
            self._encode_move(action[MovesetFeature.MOVESET_FEATURE__MOVE_ID]),
        )

        return embedding

    def _encode_actions(
        self, env_step: EnvStep, entity_embeddings: chex.Array
    ) -> chex.Array:
        action_embeddings = jax.vmap(
            jax.vmap(self._encode_action, in_axes=(0, 0, None))
        )(env_step.moveset, env_step.legal.astype(int), entity_embeddings)
        action_embeddings = jax.vmap(self.action_ff)(action_embeddings)
        return jax.vmap(self.action_encoder)(action_embeddings, env_step.legal)

    def __call__(
        self, env_step: EnvStep, history_step: HistoryStep
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Forward pass of the Encoder model. Processes an environment step and outputs
        contextual embeddings for actions.
        """
        private_entity_embeddings, entity_mask, entity_embeddings = (
            self._encode_entities(env_step)
        )

        timestep_embeddings, history_request_count = self._encode_timesteps(
            history_step.major_history
        )
        timestep_mask = env_step.request_count[..., None] >= history_request_count

        action_embeddings = self._encode_actions(env_step, private_entity_embeddings)

        latent_timesteps = jax.vmap(
            self.latent_timestep_decoder, in_axes=(None, None, None, 0)
        )(self.latent_embeddings, timestep_embeddings, None, timestep_mask)

        latent_entities = jax.vmap(
            self.latent_entity_decoder, in_axes=(None, 0, None, 0)
        )(self.latent_embeddings, entity_embeddings, None, entity_mask)

        latent_actions = jax.vmap(
            self.latent_action_decoder, in_axes=(None, 0, None, 0)
        )(self.latent_embeddings, action_embeddings, None, env_step.legal)

        latent_embeddings = jax.vmap(jax.vmap(self.latent_merge))(
            latent_timesteps, latent_entities, latent_actions
        )
        contextual_latent_embeddings = jax.vmap(self.latent_encoder)(latent_embeddings)

        return contextual_latent_embeddings, action_embeddings
