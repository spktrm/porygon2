import jax
import chex
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from ml_collections import ConfigDict
from functools import partial

from ml.arch.modules import (
    CNNEncoder,
    MultiHeadAttention,
    Resnet,
    ToAvgVector,
    Transformer,
    VectorMerge,
)

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

from rlenv.protos.enums_pb2 import SpeciesEnum
from rlenv.protos.features_pb2 import (
    FeatureEntity,
    FeatureMoveset,
    FeatureTurnContext,
    FeatureWeather,
)


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
        self.pp_linear = nn.Dense(features=self.cfg.entity_size)

    def encode_move(self, move: chex.Array):
        pp_left = move[FeatureMoveset.PPLEFT]
        pp_max = move[FeatureMoveset.PPMAX].clip(min=1)
        pp_ratio = (pp_left / pp_max).clip(min=0, max=1)
        pp_onehot = jnp.concatenate(
            (
                pp_ratio[None],
                _binary_scale_embedding(pp_left.astype(np.int32), 64),
                _binary_scale_embedding(pp_max.astype(np.int32), 64),
            )
        )
        move_onehot = jax.nn.one_hot(move[FeatureMoveset.MOVEID], NUM_MOVES)
        return self.move_linear(move_onehot) + self.pp_linear(pp_onehot)

    def __call__(
        self,
        moveset: chex.Array,
        history_move_index: chex.Array,
    ):
        _encode = jax.vmap(self.encode_move)
        move_embeddings = _encode(moveset)

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
        hp = entity[FeatureEntity.HP]
        maxhp = entity[FeatureEntity.MAXHP].clip(min=1)

        hp_ratio = (hp / maxhp).clip(min=0, max=1)
        hp_token = (1023 * hp_ratio).astype(int)

        embeddings = [
            _binary_scale_embedding(hp_token.astype(np.int32), 1024),
            hp_ratio[jnp.newaxis],
            _binary_scale_embedding(entity[FeatureEntity.LEVEL].astype(np.int32), 101),
            jax.nn.one_hot(entity[FeatureEntity.GENDER], NUM_GENDERS),
            jax.nn.one_hot(entity[FeatureEntity.STATUS], NUM_STATUS),
            jax.nn.one_hot(entity[FeatureEntity.BEING_CALLED_BACK], 2),
            jax.nn.one_hot(entity[FeatureEntity.TRAPPED], 2),
            jax.nn.one_hot(entity[FeatureEntity.ACTIVE], 2),
            jax.nn.one_hot(entity[FeatureEntity.NEWLY_SWITCHED], 2),
            jax.nn.one_hot(entity[FeatureEntity.TOXIC_TURNS], 8),
            jax.nn.one_hot(entity[FeatureEntity.SLEEP_TURNS], 4),
            # _encode_multi_onehot(
            #     entity[FeatureEntity.TYPE_TOKEN_START : FeatureEntity.TYPE_TOKEN_END],
            #     NUM_TYPES,
            # ),
            jax.nn.one_hot(entity[FeatureEntity.FAINTED], 2),
            jax.nn.one_hot(entity[FeatureEntity.ITEM_EFFECT], NUM_ITEM_EFFECTS),
        ]

        # Put all the encoded one-hots in a single boolean vector:
        boolean_code = jnp.concatenate(embeddings, axis=0)
        moveset_onehot = _encode_multi_onehot(
            entity[FeatureEntity.MOVEID0 : FeatureEntity.MOVEID3], 4
        )

        embeddings = [
            self.onehot_linear(boolean_code.astype(np.float32)),
            self.species_linear(
                jax.nn.one_hot(entity[FeatureEntity.SPECIES], NUM_SPECIES)
            ),
            self.ability_linear(
                jax.nn.one_hot(entity[FeatureEntity.ABILITY], NUM_ABILITIES)
            ),
            self.item_linear(jax.nn.one_hot(entity[FeatureEntity.ITEM], NUM_ITEMS)),
            self.moveset_linear(moveset_onehot),
        ]
        return jnp.stack(embeddings).sum(0)

    def __call__(self, active_entities: chex.Array, side_entities: chex.Array):
        _encode = jax.vmap(jax.vmap(self.encode_entity))

        active_embeddings = _encode(active_entities)
        side_embeddings = _encode(side_entities)

        side_species_token = side_entities[..., FeatureEntity.SPECIES]
        valid_team_mask = side_species_token != SpeciesEnum.species_none
        valid_team_mask |= side_species_token != SpeciesEnum.species_pad

        return active_embeddings, side_embeddings, valid_team_mask


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
        side_embeddings = _merge(active_embeddings, side_embeddings)
        return side_embeddings.reshape(side_embeddings.shape[0], -1)


class TeamEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        # self.transformer = Transformer(**self.cfg.transteams_embeddingformer.to_dict())
        self.to_vector = ToAvgVector(**self.cfg.to_vector.to_dict())

    def __call__(self, team_embeddings: chex.Array, valid_mask: chex.Array):

        # team_embeddings = team_embeddings.reshape(-1, team_embeddings.shape[-1])
        # valid_mask = valid_mask.reshape(-1)

        # team_embeddings = self.transformerteams_embedding(team_embeddings, valid_mask.astype(bool))
        teams_embedding = jax.vmap(self.to_vector)(team_embeddings, valid_mask)

        return team_embeddings, teams_embedding.reshape(-1)


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
        weather_onehot = jax.nn.one_hot(weather[FeatureWeather.WEATHER_ID], NUM_WEATHER)
        weather_min_onehot = jax.nn.one_hot(weather[FeatureWeather.MIN_DURATION], 8)
        weather_max_onehot = jax.nn.one_hot(weather[FeatureWeather.MAX_DURATION], 8)

        turn = turn_context[FeatureTurnContext.TURN]
        relative_turn = (max_turn - turn).clip(min=0)

        move_counter_onehot = jax.nn.one_hot(
            turn_context[FeatureTurnContext.MOVE_COUNTER], 4
        )
        switch_counter_onehot = jax.nn.one_hot(
            turn_context[FeatureTurnContext.SWITCH_COUNTER], 4
        )
        turn_onehot = jax.nn.one_hot(relative_turn, NUM_HISTORY)
        player_onehot = jax.nn.one_hot(
            turn_context[FeatureTurnContext.IS_MY_TURN], NUM_PLAYERS
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
        turn = turn_context[..., FeatureTurnContext.TURN]
        valid_mask = turn_context[..., FeatureTurnContext.VALID]
        max_turn = turn.max()
        _encode = partial(self.encode, max_turn=max_turn)
        field_encoding = jax.vmap(_encode)(
            terrain, pseudoweather, weather, turn_context, history_move_embedding
        )
        return valid_mask, field_encoding


class HistoryEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        # self.transformer = Transformer(**self.cfg.transformer.to_dict())
        # self.to_vector = ToAvgVector(**self.cfg.to_vector.to_dict())
        self.encoder = CNNEncoder(
            (self.cfg.vector_size, self.cfg.entity_size),
            output_size=self.cfg.vector_size,
        )

    def __call__(self, state_history: chex.Array, valid_mask: chex.Array):
        # state_history = self.transformer(state_history, valid_mask)
        return self.encoder(state_history, valid_mask)


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

        self.state_resnet = Resnet(**self.cfg.state_resnet)

    def __call__(self, env_step: EnvStep):
        move_embeddings, history_move_embeddings = self.move_encoder(
            env_step.moveset,
            env_step.turn_context[..., FeatureTurnContext.MOVE],
        )

        active_embeddings, team_embeddings, valid_team_mask = self.entity_encoder(
            env_step.active_entities, env_step.side_entities
        )

        team_embeddings, teams_embedding = self.team_encoder(
            team_embeddings, valid_team_mask
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

        history_states = jax.vmap(self.history_merge)(side_embeddings, field_embeddings)

        current_state = self.history_encoder(history_states, valid_mask)

        # legal_moves_embedding = self.legal_encoder(env_step)

        current_state = self.state_merge(current_state, teams_embedding)
        current_state = self.state_resnet(current_state)

        return current_state, team_embeddings[0], move_embeddings
