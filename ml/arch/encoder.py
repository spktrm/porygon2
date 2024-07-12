import chex
import jax
import numpy as np

import jax.numpy as jnp
import flax.linen as nn

from functools import partial
from typing import Sequence

from ml.arch.modules import ConfigurableModule
from rlenv.data import (
    NUM_GENDERS,
    NUM_HISTORY,
    NUM_HYPHEN_ARGS,
    NUM_PLAYERS,
    NUM_STATUS,
    NUM_TERRAIN,
    NUM_TYPES,
    NUM_WEATHER,
    SPIKES_TOKEN,
    TOXIC_SPIKES_TOKEN,
)
from rlenv.interfaces import EnvStep


def _onehot_encode(entity: np.ndarray, feature_idx: int, num_classes: int):
    return jax.nn.one_hot(entity[:, feature_idx], num_classes)


def _encode_divided_onehot(x: chex.Array, num_classes: int, divisor: int):
    return jax.nn.one_hot(x // divisor, num_classes // divisor)


def _encode_multi_onehot(x: chex.Array, num_classes: int):
    result = jnp.zeros(num_classes)
    result = result.at[x].set(1)
    return result


class FeatureEntity:
    SPECIES = 0
    ITEM = 1
    ABILITY = 2
    HP = 3
    ACTIVE = 4
    FAINTED = 5
    LEVEL = 6
    GENDER = 7
    BEING_CALLED_BACK = 8
    HURT_THIS_TURN = 9
    STATUS = 10
    LAST_MOVE = 11
    PUBLIC = 12
    SIDE = 13
    SLEEP_TURNS = 14
    TOXIC_TURNS = 15

    TYPE_TOKEN_START = 16
    TYPE_TOKEN_END = 18

    MOVE_PP_LEFT_START = 18
    MOVE_PP_LEFT_END = 22

    MOVE_PP_MAX_START = 22
    MOVE_PP_MAX_END = 26

    MOVE_TOKENS_START = 26
    MOVE_TOKENS_END = 30


def _binary_scale_embedding(to_encode: chex.Array, world_dim: int) -> chex.Array:
    """Encode the feature using its binary representation."""
    chex.assert_rank(to_encode, 0)
    chex.assert_type(to_encode, jnp.int32)
    num_bits = (world_dim - 1).bit_length()
    bit_mask = 1 << np.arange(num_bits)
    pos = jnp.broadcast_to(to_encode[jnp.newaxis], num_bits)
    result = jnp.not_equal(jnp.bitwise_and(pos, bit_mask), 0)
    return result.astype(jnp.float32)


class AddEncodings(ConfigurableModule):
    def __call__(self, encodings: Sequence[chex.Array]):
        return sum([nn.Dense(self.stream_size)(x) for x in encodings])


class HistoryEncoder(ConfigurableModule):
    def __call__(self, state_history: chex.Array, valid_mask: chex.Array):
        state_history = self.transformer(state_history, valid_mask)
        return self.to_vector(state_history, valid_mask)


class TeamEncoder(ConfigurableModule):
    def __call__(self, team_embeddings: chex.Array, valid_mask: chex.Array):

        team_embeddings = team_embeddings.reshape(-1, team_embeddings.shape[-1])
        valid_mask = valid_mask.reshape(-1)

        team_embeddings = self.transformer(team_embeddings, valid_mask.astype(bool))
        teams_embedding = self.to_vector(team_embeddings, valid_mask)

        return teams_embedding


class MoveEncoder(ConfigurableModule):
    def encode_moveset(self, moveset_indices: chex.Array):
        encodings = MOVE_ENCODINGS[moveset_indices]
        embedding = self.linear(encodings)
        return embedding.mean(0)

    def __call__(
        self,
        action_indices: chex.Array,
        active_entities: chex.Array,
        side_entities: chex.Array,
        history_move_index: chex.Array,
    ):
        move_encodings = MOVE_ENCODINGS[action_indices]
        move_embeddings = self.linear(move_encodings)

        _encode = jax.vmap(jax.vmap(self.encode_moveset))

        active_moveset_embeddings = _encode(
            active_entities[
                ..., FeatureEntity.MOVE_TOKENS_START : FeatureEntity.MOVE_TOKENS_END
            ]
        )
        side_moveset_embeddings = _encode(
            side_entities[
                ..., FeatureEntity.MOVE_TOKENS_START : FeatureEntity.MOVE_TOKENS_END
            ]
        )

        history_move_encoding = MOVE_ENCODINGS[history_move_index]
        history_move_embedding = self.linear(history_move_encoding)

        return (
            move_embeddings,
            active_moveset_embeddings,
            side_moveset_embeddings,
            history_move_embedding,
        )


class SideEncoder(ConfigurableModule):
    def encode(
        self,
        boosts: chex.Array,
        side_conditions: chex.Array,
        volatile_status: chex.Array,
        additional_features: chex.Array,
    ):
        boosts_onehot = jax.nn.one_hot(boosts, 13).astype(np.float32)

        side_conditions_onehot = (side_conditions > 0).astype(np.float32)

        spikes_onehot = jax.nn.one_hot(side_conditions[SPIKES_TOKEN], 4).astype(
            np.float32
        )
        toxic_spikes_onehot = jax.nn.one_hot(
            side_conditions[TOXIC_SPIKES_TOKEN], 3
        ).astype(np.float32)
        side_conditions_onehot = (side_conditions > 0).astype(np.float32)
        side_conditions_onehot = jnp.concatenate(
            (side_conditions_onehot, spikes_onehot, toxic_spikes_onehot), axis=-1
        )
        volatile_status_onehot = (volatile_status > 0).astype(np.float32)

        num_fainted = additional_features[0]
        unit_tokens = additional_features[1:]

        num_fainted_onehot = jax.nn.one_hot(num_fainted, 6)

        units_bow_onehot = self.species_onehot(unit_tokens)

        side_onehot = jnp.concatenate(
            (
                boosts_onehot.flatten(),
                side_conditions_onehot,
                volatile_status_onehot,
                num_fainted_onehot,
            ),
            axis=-1,
        )
        encodings = [side_onehot, units_bow_onehot.mean(0)]
        encodings = self.add_encodings(encodings)
        return encodings

    def __call__(
        self,
        boosts: chex.Array,
        side_conditions: chex.Array,
        volatile_status: chex.Array,
        additional_features: chex.Array,
    ):
        encodings = jax.vmap(jax.vmap(self.encode))(
            boosts, side_conditions, volatile_status, additional_features
        )
        encodings = encodings.reshape((encodings.shape[0], -1))
        encodings = self.encodings_mlp(encodings)

        return encodings


class FieldEncoder(ConfigurableModule):
    def encode(
        self,
        terrain: chex.Array,
        pseudoweather: chex.Array,
        weather: chex.Array,
        turn_context: chex.Array,
        history_move_embedding: chex.Array,
        max_turn: chex.Array,
    ):
        terrain_onehot = jax.nn.one_hot(terrain[0], NUM_TERRAIN)
        pseudoweather_onehot = (pseudoweather > 0).astype(np.float32)
        weather_onehot = jax.nn.one_hot(weather[0], NUM_WEATHER)
        weather_min_onehot = jax.nn.one_hot(weather[1], 8)
        weather_max_onehot = jax.nn.one_hot(weather[2], 8)

        turn = turn_context[3]
        relative_turn = (max_turn - turn).clip(min=0)

        move_counter_onehot = jax.nn.one_hot(turn_context[1], 4)
        switch_counter_onehot = jax.nn.one_hot(turn_context[2], 4)
        turn_onehot = jax.nn.one_hot(relative_turn, NUM_HISTORY)
        player_onehot = jax.nn.one_hot(turn_context[4], NUM_PLAYERS)

        hyphen_args = turn_context[5:-1].reshape(2, -1)
        hyphen_args_onehot = jax.vmap(
            partial(_encode_multi_onehot, num_classes=NUM_HYPHEN_ARGS)
        )(hyphen_args[..., 1:])[..., 1:]

        field_onehot = jnp.concatenate(
            (
                terrain_onehot,
                pseudoweather_onehot,
                weather_onehot,
                weather_min_onehot,
                weather_max_onehot,
                move_counter_onehot,
                switch_counter_onehot,
                turn_onehot,
                player_onehot,
                hyphen_args_onehot.reshape(-1),
                history_move_embedding,
            ),
            axis=-1,
        )

        return self.field_linear(field_onehot)

    def __call__(
        self,
        terrain: chex.Array,
        pseudoweather: chex.Array,
        weather: chex.Array,
        turn_context: chex.Array,
        history_move_embedding: chex.Array,
    ):
        valid_mask = turn_context[..., 0].astype(bool)
        max_turn = turn_context[..., 3].max()
        _encode = partial(self.encode, max_turn=max_turn)
        field_encoding = jax.vmap(_encode)(
            terrain, pseudoweather, weather, turn_context, history_move_embedding
        )
        return valid_mask, field_encoding


class EntityEncoder(ConfigurableModule):
    def encode_entity(self, entity: chex.Array, moveset_embedding):
        embeddings = [
            _binary_scale_embedding(entity[FeatureEntity.HP], 1024),
            (entity[FeatureEntity.HP] / 1023)[jnp.newaxis],
            _binary_scale_embedding(entity[FeatureEntity.LEVEL], 100),
            jax.nn.one_hot(entity[FeatureEntity.GENDER], NUM_GENDERS),
            jax.nn.one_hot(entity[FeatureEntity.STATUS], NUM_STATUS),
            _encode_multi_onehot(
                entity[FeatureEntity.TYPE_TOKEN_START : FeatureEntity.TYPE_TOKEN_END],
                NUM_TYPES,
            ),
            jax.nn.one_hot(entity[FeatureEntity.FAINTED], 2),
            jax.nn.one_hot(entity[FeatureEntity.BEING_CALLED_BACK], 2),
            jax.nn.one_hot(entity[FeatureEntity.HURT_THIS_TURN], 2),
            jax.nn.one_hot(entity[FeatureEntity.SLEEP_TURNS], 4),
            jax.nn.one_hot(entity[FeatureEntity.TOXIC_TURNS], 6),
        ]

        # Put all the encoded one-hots in a single boolean vector:
        boolean_code = jnp.concatenate(embeddings, axis=0)

        embeddings = [
            boolean_code.astype(np.float32),
            SPECIES_ENCODINGS[entity[FeatureEntity.SPECIES]],
            ABILITY_ENCODINGS[entity[FeatureEntity.ABILITY]],
            ITEM_ENCODINGS[entity[FeatureEntity.ITEM]],
        ]

        return self.add_encodings(embeddings) + moveset_embedding

    def add_extra_embeddings(self, entity: chex.Array, embedding: chex.Array):
        return (
            embedding
            + self.side_embedding(jax.nn.one_hot(entity[FeatureEntity.SIDE], 2))
            + self.public_embedding(jax.nn.one_hot(entity[FeatureEntity.PUBLIC], 2))
            + self.active_embedding(jax.nn.one_hot(entity[FeatureEntity.ACTIVE], 2))
        )

    def __call__(
        self,
        active_entities: chex.Array,
        side_entities: chex.Array,
        active_moveset_embeddings: chex.Array,
        side_moveset_embeddings: chex.Array,
    ):
        _encode = jax.vmap(jax.vmap(self.encode_entity))

        active_embeddings = _encode(active_entities, active_moveset_embeddings)
        active_embeddings = active_embeddings.reshape((active_embeddings.shape[0], -1))
        active_embeddings = self.matchup_mlp(active_embeddings)

        side_embedding = _encode(side_entities, side_moveset_embeddings)
        team_embeddings = jax.vmap(jax.vmap(self.add_extra_embeddings))(
            side_entities, side_embedding
        )
        return active_embeddings, team_embeddings, side_embedding[0]


class LegalEncoder(ConfigurableModule):
    def __call__(self, env_step: EnvStep):
        legal_moves_onehot = env_step.legal[:4][:, None] * jax.nn.one_hot(
            env_step.actions[:4], 4
        )
        return self.linear(legal_moves_onehot.sum(0))


class Encoder(ConfigurableModule):
    def __call__(self, env_step: EnvStep):
        (
            move_embeddings,
            active_moveset_embeddings,
            side_moveset_embeddings,
            history_move_embeddings,
        ) = self.move_encoder(
            env_step.actions[:4],
            env_step.active_entities,
            env_step.side_entities,
            env_step.turn_context[..., -1],
        )

        active_embeddings, team_embeddings, select_embeddings = self.entity_encoder(
            env_step.active_entities,
            env_step.side_entities,
            active_moveset_embeddings,
            side_moveset_embeddings,
        )

        valid_team_mask = jnp.ones_like(team_embeddings[..., 0])
        teams_embedding = self.team_encoder(team_embeddings, valid_team_mask)

        side_embeddings = self.side_encoder(
            env_step.boosts,
            env_step.side_conditions,
            env_step.volatile_status,
            env_step.additional_features,
        )
        valid_mask, field_embeddings = self.field_encoder(
            env_step.terrain,
            env_step.pseudoweather,
            env_step.weather,
            env_step.turn_context,
            history_move_embeddings,
        )

        history_states = jax.vmap(self.history_merge)(
            active_embeddings, side_embeddings, field_embeddings
        )

        current_state = self.history_encoder(history_states, valid_mask)

        my_active_embedding = select_embeddings[0]
        selected_unit = jnp.where(
            env_step.legal[:4].any(keepdims=True),
            my_active_embedding,
            jnp.zeros_like(my_active_embedding),
        )

        legal_moves_embedding = self.legal_encoder(env_step)

        current_state = self.vector_merge(
            current_state, teams_embedding, selected_unit, legal_moves_embedding
        )

        return (current_state, select_embeddings, move_embeddings)
