import jax
import chex
import numpy as np
import jax.numpy as jnp

from enum import Enum, auto
from functools import partial

from ml.arch.modules import ConfigurableModule
from rlenv.data import (
    NUM_ABILITIES,
    NUM_GENDERS,
    NUM_HISTORY,
    NUM_ITEM_EFFECTS,
    NUM_ITEMS,
    NUM_PLAYERS,
    NUM_MOVES,
    NUM_SPECIES,
    NUM_STATUS,
    NUM_WEATHER,
    SPIKES_TOKEN,
    TOXIC_SPIKES_TOKEN,
)
from rlenv.interfaces import EnvStep


class FeatureEntity(Enum):
    SPECIES = 0
    ITEM = auto()
    ITEM_EFFECT = auto()
    ABILITY = auto()
    GENDER = auto()
    ACTIVE = auto()
    FAINTED = auto()
    HP = auto()
    MAXHP = auto()
    STATUS = auto()
    LEVEL = auto()
    MOVEID0 = auto()
    MOVEID1 = auto()
    MOVEID2 = auto()
    MOVEID3 = auto()
    MOVEPP0 = auto()
    MOVEPP1 = auto()
    MOVEPP2 = auto()
    MOVEPP3 = auto()


class FeatureMoveset(Enum):
    MOVEID = 0
    PPLEFT = auto()
    PPMAX = auto()


class FeatureTurnContext(Enum):
    VALID = 0
    IS_MY_TURN = auto()
    ACTION = auto()
    MOVE = auto()
    SWITCH_COUNTER = auto()
    MOVE_COUNTER = auto()
    TURN = auto()


class FeatureWeather(Enum):
    WEATHER_ID = 0
    MIN_DURATION = auto()
    MAX_DURATION = auto()


def _onehot_encode(entity: np.ndarray, feature_idx: int, num_classes: int):
    return jax.nn.one_hot(entity[:, feature_idx], num_classes)


def _encode_divided_onehot(x: chex.Array, num_classes: int, divisor: int):
    return jax.nn.one_hot(x // divisor, num_classes // divisor)


def _encode_multi_onehot(x: chex.Array, num_classes: int):
    result = jnp.zeros(num_classes)
    result = result.at[x].set(1)
    return result


def _binary_scale_embedding(to_encode: chex.Array, world_dim: int) -> chex.Array:
    """Encode the feature using its binary representation."""
    chex.assert_rank(to_encode, 0)
    chex.assert_type(to_encode, jnp.int32)
    num_bits = (world_dim - 1).bit_length()
    bit_mask = 1 << np.arange(num_bits)
    pos = jnp.broadcast_to(to_encode[jnp.newaxis], num_bits)
    result = jnp.not_equal(jnp.bitwise_and(pos, bit_mask), 0)
    return result.astype(jnp.float32)


class MoveEncoder(ConfigurableModule):
    def __call__(
        self,
        moveset: chex.Array,
        history_move_index: chex.Array,
    ):
        move_embeddings = self.move_linear(
            jax.nn.one_hot(moveset[..., FeatureMoveset.MOVEID.value], NUM_MOVES)
        )

        history_move_embedding = self.move_linear(
            jax.nn.one_hot(history_move_index, NUM_MOVES)
        )

        return (move_embeddings, history_move_embedding)


class EntityEncoder(ConfigurableModule):
    def encode_entity(self, entity: chex.Array):
        embeddings = [
            _binary_scale_embedding(entity[FeatureEntity.HP.value], 1024),
            # (entity[FeatureEntity.HP.value] / entity[FeatureEntity.MAXHP.value])[
            #     jnp.newaxis
            # ],
            _binary_scale_embedding(entity[FeatureEntity.LEVEL.value], 100),
            jax.nn.one_hot(entity[FeatureEntity.GENDER.value], NUM_GENDERS),
            jax.nn.one_hot(entity[FeatureEntity.STATUS.value], NUM_STATUS),
            # _encode_multi_onehot(
            #     entity[FeatureEntity.TYPE_TOKEN_START : FeatureEntity.TYPE_TOKEN_END],
            #     NUM_TYPES,
            # ),
            jax.nn.one_hot(entity[FeatureEntity.FAINTED.value], 2),
            jax.nn.one_hot(entity[FeatureEntity.ITEM_EFFECT.value], NUM_ITEM_EFFECTS),
            # jax.nn.one_hot(entity[FeatureEntity.BEING_CALLED_BACK], 2),
            # jax.nn.one_hot(entity[FeatureEntity.HURT_THIS_TURN], 2),
            # jax.nn.one_hot(entity[FeatureEntity.SLEEP_TURNS], 4),
            # jax.nn.one_hot(entity[FeatureEntity.TOXIC_TURNS], 6),
        ]

        # Put all the encoded one-hots in a single boolean vector:
        boolean_code = jnp.concatenate(embeddings, axis=0)
        moveset_onehot = _encode_multi_onehot(
            entity[FeatureEntity.MOVEID0.value : FeatureEntity.MOVEID3.value], 4
        )

        embeddings = [
            self.onehot_linear(boolean_code.astype(np.float32)),
            self.species_linear(
                jax.nn.one_hot(entity[FeatureEntity.SPECIES.value], NUM_SPECIES)
            ),
            self.ability_linear(
                jax.nn.one_hot(entity[FeatureEntity.ABILITY.value], NUM_ABILITIES)
            ),
            self.item_linear(
                jax.nn.one_hot(entity[FeatureEntity.ITEM.value], NUM_ITEMS)
            ),
            self.moveset_linear(moveset_onehot),
        ]
        return jnp.stack(embeddings).sum(0)

    def __call__(self, active_entities: chex.Array, side_entities: chex.Array):
        _encode = jax.vmap(self.encode_entity)

        active_embeddings = jax.vmap(_encode)(active_entities)
        side_embedding = _encode(side_entities)

        return active_embeddings, side_embedding


class SideEncoder(ConfigurableModule):
    def encode(
        self,
        boosts: chex.Array,
        side_conditions: chex.Array,
        volatile_status: chex.Array,
        hyphen_args: chex.Array,
    ):
        boosts_float = (2 * boosts) ** 2 / 4
        boosts_onehot = jax.nn.one_hot(boosts + 6, 13).astype(np.float32)

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

        hyphen_args_onehot = jnp.unpackbits(hyphen_args).astype(np.float32)

        side_onehot = jnp.concatenate(
            (
                boosts_float.flatten(),
                boosts_onehot.flatten(),
                side_conditions_onehot,
                volatile_status_onehot,
                hyphen_args_onehot,
            ),
            axis=-1,
        )
        return self.linear(side_onehot)

    def __call__(
        self,
        active_embeddings: chex.Array,
        boosts: chex.Array,
        side_conditions: chex.Array,
        volatile_status: chex.Array,
        hyphen_args: chex.Array,
    ):
        _encode = jax.vmap(jax.vmap(self.encode))
        _merge = jax.vmap(jax.vmap(self.merge))

        side_embeddings = _encode(boosts, side_conditions, volatile_status, hyphen_args)
        return _merge(active_embeddings, side_embeddings)


class TeamEncoder(ConfigurableModule):
    def __call__(self, team_embeddings: chex.Array, valid_mask: chex.Array):

        team_embeddings = team_embeddings.reshape(-1, team_embeddings.shape[-1])
        valid_mask = valid_mask.reshape(-1)

        team_embeddings = self.transformer(team_embeddings, valid_mask.astype(bool))
        teams_embedding = self.to_vector(team_embeddings, valid_mask)

        return team_embeddings, teams_embedding


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
        # terrain_onehot = jax.nn.one_hot(terrain[0], NUM_TERRAIN)
        pseudoweather_onehot = (pseudoweather > 0).astype(np.float32)
        weather_onehot = jax.nn.one_hot(
            weather[FeatureWeather.WEATHER_ID.value], NUM_WEATHER
        )
        weather_min_onehot = jax.nn.one_hot(
            weather[FeatureWeather.MIN_DURATION.value], 8
        )
        weather_max_onehot = jax.nn.one_hot(
            weather[FeatureWeather.MAX_DURATION.value], 8
        )

        turn = turn_context[FeatureTurnContext.TURN.value]
        relative_turn = (max_turn - turn).clip(min=0)

        move_counter_onehot = jax.nn.one_hot(
            turn_context[FeatureTurnContext.MOVE_COUNTER.value], 4
        )
        switch_counter_onehot = jax.nn.one_hot(
            turn_context[FeatureTurnContext.SWITCH_COUNTER.value], 4
        )
        turn_onehot = jax.nn.one_hot(relative_turn, NUM_HISTORY)
        player_onehot = jax.nn.one_hot(
            turn_context[FeatureTurnContext.IS_MY_TURN.value], NUM_PLAYERS
        )

        field_onehot = jnp.concatenate(
            (
                # terrain_onehot,
                pseudoweather_onehot,
                weather_onehot,
                weather_min_onehot,
                weather_max_onehot,
                move_counter_onehot,
                switch_counter_onehot,
                turn_onehot,
                player_onehot,
                history_move_embedding,
            ),
            axis=-1,
        )

        return self.linear(field_onehot)

    def __call__(
        self,
        terrain: chex.Array,
        pseudoweather: chex.Array,
        weather: chex.Array,
        turn_context: chex.Array,
        history_move_embedding: chex.Array,
    ):
        turn = turn_context[..., FeatureTurnContext.TURN.value]
        valid_mask = turn_context[..., FeatureTurnContext.VALID.value]
        max_turn = turn.max()
        _encode = partial(self.encode, max_turn=max_turn)
        field_encoding = jax.vmap(_encode)(
            terrain, pseudoweather, weather, turn_context, history_move_embedding
        )
        return valid_mask, field_encoding


class HistoryEncoder(ConfigurableModule):
    def __call__(self, state_history: chex.Array, valid_mask: chex.Array):
        state_history = self.transformer(state_history, valid_mask)
        return self.to_vector(state_history, valid_mask)


class LegalEncoder(ConfigurableModule):
    def __call__(self, env_step: EnvStep):
        legal_moves_onehot = env_step.legal[:4][:, None] * jax.nn.one_hot(
            env_step.actions[:4], 4
        )
        return self.linear(legal_moves_onehot.sum(0))


class Encoder(ConfigurableModule):
    def __call__(self, env_step: EnvStep):
        move_embeddings, history_move_embeddings = self.move_encoder(
            env_step.moveset,
            env_step.turn_context[..., FeatureTurnContext.MOVE.value],
        )

        active_embeddings, team_embeddings = self.entity_encoder(
            env_step.active_entities, env_step.team
        )

        valid_team_mask = jnp.ones_like(team_embeddings[..., 0])
        team_embeddings, teams_embedding = self.team_encoder(
            team_embeddings[..., 1:, :], valid_team_mask[..., 1:]
        )

        side_embeddings = self.side_encoder(
            active_embeddings,
            env_step.boosts,
            env_step.side_conditions,
            env_step.volatile_status,
            env_step.hyphen_args,
        )
        valid_mask, field_embeddings = self.field_encoder(
            env_step.terrain,
            env_step.pseudoweather,
            env_step.weather,
            env_step.turn_context,
            history_move_embeddings,
        )

        history_states = jax.vmap(self.history_merge)(
            side_embeddings[:, 0], side_embeddings[:, 1], field_embeddings
        )

        current_state = self.history_encoder(history_states, valid_mask)

        my_active_embedding = team_embeddings[0]
        selected_unit = jnp.where(
            env_step.legal[:4].any(keepdims=True),
            my_active_embedding,
            jnp.zeros_like(my_active_embedding),
        )

        # legal_moves_embedding = self.legal_encoder(env_step)

        current_state = self.state_merge(
            current_state,
            teams_embedding,
            selected_unit,  # legal_moves_embedding
        )

        return current_state, team_embeddings, move_embeddings
