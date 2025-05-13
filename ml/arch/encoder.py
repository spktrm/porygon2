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
    NUM_ABILITIES,
    NUM_ACTIONS,
    NUM_ITEMS,
    NUM_MOVES,
    NUM_SPECIES,
    RELATIVE_EDGE_MAX_VALUES,
)
from rlenv.interfaces import EnvStep, HistoryContainer, HistoryStep
from rlenv.protos.enums_pb2 import (
    AbilitiesEnum,
    ActionsEnum,
    ItemsEnum,
    MovesEnum,
    SpeciesEnum,
)
from rlenv.protos.features_pb2 import (
    AbsoluteEdgeFeature,
    ActionsFeature,
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
_encode_one_hot_action = partial(_encode_one_hot, max_values=ACTION_MAX_VALUES)
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

    @nn.compact
    def __call__(
        self, env_step: EnvStep, history_step: HistoryStep
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Forward pass of the Encoder model. Processes an environment step and outputs
        contextual embeddings for actions.
        """

        # Extract configuration parameters for embedding sizes.
        entity_size = self.cfg.entity_size

        embed_kwargs = dict(
            features=entity_size, embedding_init=nn.initializers.lecun_normal()
        )

        species_embedding = nn.Embed(
            NUM_SPECIES, name="species_embedding", **embed_kwargs
        )
        items_embedding = nn.Embed(NUM_ITEMS, name="items_embedding", **embed_kwargs)
        abilities_embedding = nn.Embed(
            NUM_ABILITIES, name="abilities_embedding", **embed_kwargs
        )
        moves_embedding = nn.Embed(NUM_MOVES, name="moves_embedding", **embed_kwargs)

        species_linear = nn.Dense(entity_size, use_bias=False, name="species_linear")
        items_linear = nn.Dense(entity_size, use_bias=False, name="items_linear")
        abilities_linear = nn.Dense(
            entity_size, use_bias=False, name="abilities_linear"
        )
        moves_linear = nn.Dense(entity_size, use_bias=False, name="moves_linear")

        def _encode_species(token: chex.Array):
            return jnp.where(
                token == SpeciesEnum.SPECIES_ENUM___UNSPECIFIED,
                0,
                species_linear(SPECIES_ONEHOT(token)),
            )

        def _encode_item(token: chex.Array):
            return jnp.where(
                token == ItemsEnum.ITEMS_ENUM___UNSPECIFIED,
                0,
                items_linear(ITEM_ONEHOT(token)),
            )

        def _encode_ability(token: chex.Array):
            return jnp.where(
                token == AbilitiesEnum.ABILITIES_ENUM___UNSPECIFIED,
                0,
                abilities_linear(ABILITY_ONEHOT(token)),
            )

        def _encode_move(token: chex.Array):
            return jnp.where(
                (token == MovesEnum.MOVES_ENUM___UNSPECIFIED)[..., None],
                0,
                moves_linear(MOVE_ONEHOT(token)),
            )

        # Initialize aggregation modules for combining feature embeddings.
        entity_combine = SumEmbeddings(entity_size, name="entity_combine")
        relative_edge_combine = SumEmbeddings(entity_size, name="relative_edge_combine")
        absolute_edge_combine = SumEmbeddings(entity_size, name="absolute_edge_combine")
        timestep_mlp = MLP((entity_size,), name="timestep_mlp")

        def _encode_entity(entity: chex.Array) -> Tuple[chex.Array, chex.Array]:
            # Encode volatile and type-change indices using the binary encoder.
            volatiles_indices = entity[
                EntityFeature.ENTITY_FEATURE__VOLATILES0 : EntityFeature.ENTITY_FEATURE__VOLATILES8
                + 1
            ]
            volatiles_encoding = HEX_ENCODER(
                volatiles_indices.astype(jnp.uint16)
            ).reshape(-1)

            typechange_indices = entity[
                EntityFeature.ENTITY_FEATURE__TYPECHANGE0 : EntityFeature.ENTITY_FEATURE__TYPECHANGE1
                + 1
            ]
            typechange_encoding = HEX_ENCODER(
                typechange_indices.astype(jnp.uint16)
            ).reshape(-1)

            boolean_code = one_hot_concat_jax(
                [
                    _encode_sqrt_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__LEVEL
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__ACTIVE
                    ),
                    _encode_one_hot_entity(entity, EntityFeature.ENTITY_FEATURE__SIDE),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__IS_PUBLIC
                    ),
                    _encode_divided_one_hot_entity(
                        entity,
                        EntityFeature.ENTITY_FEATURE__HP_RATIO,
                        MAX_RATIO_TOKEN / 32,
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__GENDER
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__STATUS
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__ITEM_EFFECT
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__BEING_CALLED_BACK
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__TRAPPED
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__NEWLY_SWITCHED
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__TOXIC_TURNS
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__SLEEP_TURNS
                    ),
                    _encode_one_hot_entity(
                        entity, EntityFeature.ENTITY_FEATURE__FAINTED
                    ),
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
            move_tokens = entity[move_indices]

            moveset_onehot = moves_embedding(move_tokens).sum(0) / entity[
                EntityFeature.ENTITY_FEATURE__NUM_MOVES
            ].clip(min=1)

            move_embeddings = _encode_move(move_tokens)
            moveset_embedding = move_embeddings.sum(0) / entity[
                EntityFeature.ENTITY_FEATURE__NUM_MOVES
            ].clip(min=1)

            species_token = entity[EntityFeature.ENTITY_FEATURE__SPECIES]
            ability_token = entity[EntityFeature.ENTITY_FEATURE__ABILITY]
            item_token = entity[EntityFeature.ENTITY_FEATURE__ITEM]
            last_move_token = entity[EntityFeature.ENTITY_FEATURE__LAST_MOVE]

            embedding = entity_combine(
                encodings=[
                    boolean_code,
                    volatiles_encoding,
                    typechange_encoding,
                ],
                embeddings=[
                    _encode_species(species_token),
                    _encode_ability(ability_token),
                    _encode_item(item_token),
                    _encode_move(last_move_token),
                    moveset_embedding,
                    species_embedding(species_token),
                    abilities_embedding(ability_token),
                    items_embedding(item_token),
                    moveset_onehot,
                ],
            )

            # Apply mask to filter out invalid entities.
            mask = get_entity_mask(entity)
            embedding = jnp.where(mask[..., None], embedding, 0)

            return embedding, mask

        def _encode_relative_edge(edge: chex.Array) -> chex.Array:
            # Encode minor arguments and side conditions using the binary encoder.

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
                        edge, RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SPIKES
                    ),
                    _encode_one_hot_relative_edge(
                        edge,
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__TOXIC_SPIKES,
                    ),
                ]
            )

            embedding = relative_edge_combine(
                encodings=[side_condition_encoding, boolean_code]
            )

            return embedding

        def _encode_absolute_edge(edge: chex.Array) -> chex.Array:
            """
            Encode features of an absolute edge, including turn and request offsets.
            """

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

            embedding = absolute_edge_combine(encodings=[boolean_code])

            # Apply mask to filter out invalid edges.
            mask = get_edge_mask(edge)

            return embedding, mask

        # Encode each timestep's features, including nodes and edges.
        def _encode_timestep(history_container: HistoryContainer):
            """
            Encode features of a single timestep, including entities and edges.
            """

            # Encode relative edges.
            relative_edge_embeddings = jax.vmap(_encode_relative_edge)(
                history_container.relative_edges
            )

            # Encode absolute edges.
            absolute_edge_embedding, edge_mask = _encode_absolute_edge(
                history_container.absolute_edges
            )

            # Combine all embeddings for the timestep.
            timestep_embedding = jnp.concatenate(
                [
                    relative_edge_embeddings[0],
                    relative_edge_embeddings[1],
                    absolute_edge_embedding,
                ],
                axis=-1,
            )

            timestep_embedding = timestep_mlp(timestep_embedding)

            # Apply mask to the timestep embeddings.
            timestep_embedding = jnp.where(edge_mask, timestep_embedding, 0)

            return timestep_embedding, edge_mask

        _encode_entities = jax.vmap(_encode_entity)
        private_entity_embeddings, private_entity_mask = _encode_entities(
            env_step.private_team
        )
        public_entity_embeddings, public_entity_mask = _encode_entities(
            env_step.public_team
        )
        timestep_embedding, _ = jax.vmap(_encode_timestep)(history_step.major_history)

        private_entity_embeddings = private_entity_embeddings + timestep_embedding
        public_entity_embeddings = public_entity_embeddings + timestep_embedding

        private_entity_embeddings = TransformerEncoder(
            **self.cfg.entity_encoder.to_dict()
        )(private_entity_embeddings, private_entity_mask)

        public_entity_embeddings = TransformerEncoder(
            **self.cfg.entity_encoder.to_dict()
        )(public_entity_embeddings, public_entity_mask)

        entity_embeddings = TransformerDecoder(**self.cfg.action_decoder.to_dict())(
            private_entity_embeddings,
            public_entity_embeddings,
            private_entity_mask,
            public_entity_mask,
        )

        entity_embeddings = TransformerDecoder(**self.cfg.action_decoder.to_dict())(
            public_entity_embeddings,
            entity_embeddings,
            public_entity_mask,
            private_entity_mask,
        )

        def _get_action_key(member_action: chex.Array):
            move_id = member_action[ActionsFeature.ACTIONS_FEATURE__MOVE_ID]
            action_index = member_action[ActionsFeature.ACTIONS_FEATURE__ACTION_INDEX]
            return (
                move_id,
                jax.nn.one_hot(action_index, 10),
                (move_id != MovesEnum.MOVES_ENUM___NULL)
                & (move_id != MovesEnum.MOVES_ENUM___UNSPECIFIED),
            )

        def _get_action_keys(member_actions: chex.Array):
            move_ids, action_indices, action_masks = jax.vmap(_get_action_key)(
                member_actions
            )
            move_embeddings = _encode_move(move_ids)
            action_indices = jnp.where(action_masks[..., None], action_indices, 0)
            action_keys = action_indices.T @ move_embeddings
            action_keys = action_keys / action_indices.sum(0)[..., None].clip(min=1)
            action_masks = action_indices.sum(0)
            return action_keys, action_masks

        my_action_embedding = self.param(
            "my_action_embedding", jax.nn.initializers.lecun_normal(), (1, entity_size)
        )
        opp_action_embedding = self.param(
            "opp_action_embedding", jax.nn.initializers.lecun_normal(), (1, entity_size)
        )
        my_action_keys, my_action_masks = jax.vmap(_get_action_keys)(
            env_step.all_my_moves
        )
        opp_action_keys, opp_action_masks = jax.vmap(_get_action_keys)(
            env_step.all_opp_moves
        )

        action_embeddings = jnp.concatenate(
            (
                my_action_keys.sum(0) + my_action_embedding,
                opp_action_keys.sum(0) + opp_action_embedding,
            ),
            axis=0,
        )

        opp_legal = opp_action_masks.any(0)
        action_embeddings = TransformerDecoder(**self.cfg.action_decoder.to_dict())(
            action_embeddings,
            entity_embeddings,
            jnp.concatenate((env_step.legal, opp_legal), axis=0),
            public_entity_mask,
        )

        my_action_embeddings, opp_action_embeddings = jnp.split(
            action_embeddings, indices_or_sections=2, axis=0
        )

        action_embeddings = TransformerDecoder(**self.cfg.action_decoder.to_dict())(
            my_action_embeddings,
            opp_action_embeddings,
            env_step.legal,
            opp_legal,
        )

        action_embeddings = TransformerEncoder(**self.cfg.entity_encoder.to_dict())(
            action_embeddings,
            env_step.legal,
        )

        return action_embeddings
