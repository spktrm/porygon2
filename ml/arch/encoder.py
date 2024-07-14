import jax
import chex
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from ml_collections import ConfigDict
from functools import partial

from ml.arch.modules import ToAvgVector, Transformer, VectorMerge
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
    FeatureEntity,
    FeatureMoveset,
    FeatureTurnContext,
    FeatureWeather,
)
from rlenv.interfaces import EnvStep


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


class MoveEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.move_linear = nn.Dense(features=self.cfg.entity_size)

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


class EntityEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        entity_size = self.cfg.entity_size

        self.onehot_linear = nn.Dense(features=entity_size)
        self.species_linear = nn.Dense(features=entity_size)
        self.ability_linear = nn.Dense(features=entity_size)
        self.item_linear = nn.Dense(features=entity_size)
        self.moveset_linear = nn.Dense(features=entity_size)

    def encode_entity(self, entity: chex.Array):
        hp = entity[FeatureEntity.HP.value]
        maxhp = entity[FeatureEntity.MAXHP.value].clip(min=1)

        hp_ratio = (hp / maxhp).clip(min=0, max=1)
        hp_token = (1023 * hp_ratio).astype(int)

        embeddings = [
            _binary_scale_embedding(hp_token, 1024),
            hp_ratio[jnp.newaxis],
            _binary_scale_embedding(entity[FeatureEntity.LEVEL.value], 101),
            jax.nn.one_hot(entity[FeatureEntity.GENDER.value], NUM_GENDERS),
            jax.nn.one_hot(entity[FeatureEntity.STATUS.value], NUM_STATUS),
            jax.nn.one_hot(entity[FeatureEntity.BEING_CALLED_BACK.value], 2),
            jax.nn.one_hot(entity[FeatureEntity.TRAPPED.value], 2),
            jax.nn.one_hot(entity[FeatureEntity.NEWLY_SWITCHED.value], 2),
            jax.nn.one_hot(entity[FeatureEntity.TOXIC_TURNS.value], 8),
            jax.nn.one_hot(entity[FeatureEntity.SLEEP_TURNS.value], 4),
            # _encode_multi_onehot(
            #     entity[FeatureEntity.TYPE_TOKEN_START : FeatureEntity.TYPE_TOKEN_END],
            #     NUM_TYPES,
            # ),
            jax.nn.one_hot(entity[FeatureEntity.FAINTED.value], 2),
            jax.nn.one_hot(entity[FeatureEntity.ITEM_EFFECT.value], NUM_ITEM_EFFECTS),
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


class SideEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.linear = nn.Dense(features=self.cfg.entity_size)
        self.merge = VectorMerge(**self.cfg.merge.to_dict())

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


class TeamEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.transformer = Transformer(**self.cfg.transformer.to_dict())
        self.to_vector = ToAvgVector(**self.cfg.to_vector.to_dict())

    def __call__(self, team_embeddings: chex.Array, valid_mask: chex.Array):

        team_embeddings = team_embeddings.reshape(-1, team_embeddings.shape[-1])
        valid_mask = valid_mask.reshape(-1)

        team_embeddings = self.transformer(team_embeddings, valid_mask.astype(bool))
        teams_embedding = self.to_vector(team_embeddings, valid_mask)

        return team_embeddings, teams_embedding


class FieldEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.linear = nn.Dense(features=self.cfg.vector_size)

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


class HistoryEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.transformer = Transformer(**self.cfg.transformer.to_dict())
        self.to_vector = ToAvgVector(**self.cfg.to_vector.to_dict())

    def __call__(self, state_history: chex.Array, valid_mask: chex.Array):
        state_history = self.transformer(state_history, valid_mask)
        return self.to_vector(state_history, valid_mask)


class Encoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.move_encoder = MoveEncoder(self.cfg.move_encoder)
        self.entity_encoder = EntityEncoder(self.cfg.entity_encoder)
        self.team_encoder = TeamEncoder(self.cfg.team_encoder)
        self.side_encoder = SideEncoder(self.cfg.side_encoder)
        self.field_encoder = FieldEncoder(self.cfg.field_encoder)
        self.history_encoder = HistoryEncoder(self.cfg.history_encoder)

        self.history_merge = VectorMerge(**self.cfg.history_merge.to_dict())
        self.state_merge = VectorMerge(**self.cfg.state_merge.to_dict())

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