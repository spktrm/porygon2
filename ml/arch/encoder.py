import jax
import chex
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from ml_collections import ConfigDict
from functools import partial

from ml.arch.modules import (
    CNN,
    MLP,
    GatingType,
    PretrainedEmbedding,
    Resnet,
    ToAvgVector,
    Transformer,
    UNet,
    VectorMerge,
)

from rlenv.data import (
    NUM_GENDERS,
    NUM_HISTORY,
    NUM_ITEM_EFFECTS,
    NUM_MOVES,
    NUM_STATUS,
    NUM_WEATHER,
    SPIKES_TOKEN,
    TOXIC_SPIKES_TOKEN,
)
from rlenv.interfaces import EnvStep

from rlenv.protos.enums_pb2 import SpeciesEnum
from rlenv.protos.features_pb2 import (
    FeatureAdditionalInformation,
    FeatureEntity,
    FeatureMoveset,
    FeatureTurnContext,
    FeatureWeather,
)


NUM_TYPES_INDICES = [
    FeatureAdditionalInformation.NUM_TYPES_PAD,
    FeatureAdditionalInformation.NUM_TYPES_UNK,
    FeatureAdditionalInformation.NUM_TYPES_BUG,
    FeatureAdditionalInformation.NUM_TYPES_DARK,
    FeatureAdditionalInformation.NUM_TYPES_DRAGON,
    FeatureAdditionalInformation.NUM_TYPES_ELECTRIC,
    FeatureAdditionalInformation.NUM_TYPES_FAIRY,
    FeatureAdditionalInformation.NUM_TYPES_FIGHTING,
    FeatureAdditionalInformation.NUM_TYPES_FIRE,
    FeatureAdditionalInformation.NUM_TYPES_FLYING,
    FeatureAdditionalInformation.NUM_TYPES_GHOST,
    FeatureAdditionalInformation.NUM_TYPES_GRASS,
    FeatureAdditionalInformation.NUM_TYPES_GROUND,
    FeatureAdditionalInformation.NUM_TYPES_ICE,
    FeatureAdditionalInformation.NUM_TYPES_NORMAL,
    FeatureAdditionalInformation.NUM_TYPES_POISON,
    FeatureAdditionalInformation.NUM_TYPES_PSYCHIC,
    FeatureAdditionalInformation.NUM_TYPES_ROCK,
    FeatureAdditionalInformation.NUM_TYPES_STEEL,
    FeatureAdditionalInformation.NUM_TYPES_STELLAR,
    FeatureAdditionalInformation.NUM_TYPES_WATER,
]

MEMBER_HP_INDICES = [
    FeatureAdditionalInformation.MEMBER0_HP,
    FeatureAdditionalInformation.MEMBER1_HP,
    FeatureAdditionalInformation.MEMBER2_HP,
    FeatureAdditionalInformation.MEMBER3_HP,
    FeatureAdditionalInformation.MEMBER4_HP,
    FeatureAdditionalInformation.MEMBER5_HP,
]


def _encode_multi_onehot(x: chex.Array, num_classes: int):
    result = jnp.zeros(num_classes)
    result = result.at[x].set(1)
    return result


def _binary_scale_embedding(to_encode: chex.Array, world_dim: int) -> chex.Array:
    """Encode the feature using its binary representation."""
    chex.assert_type(to_encode, jnp.int32)
    num_bits = (world_dim - 1).bit_length()
    bit_mask = 1 << np.arange(num_bits)
    pos = jnp.broadcast_to(to_encode[jnp.newaxis], num_bits)
    result = jnp.not_equal(jnp.bitwise_and(pos, bit_mask), 0)
    return result.astype(jnp.float32)


SPECIES_ONEHOT = PretrainedEmbedding("data/data/gen3/species.npy")
ABILITY_ONEHOT = PretrainedEmbedding("data/data/gen3/abilities.npy")
ITEM_ONEHOT = PretrainedEmbedding("data/data/gen3/items.npy")
MOVE_ONEHOT = PretrainedEmbedding("data/data/gen3/moves.npy")


class MoveEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.move_linear = nn.Dense(features=self.cfg.entity_size)
        self.pp_linear = nn.Dense(features=self.cfg.entity_size)

    def encode_move(self, move: chex.Array):
        pp_left = move[FeatureMoveset.PPUSED]
        move_id = move[FeatureMoveset.MOVEID]
        pp_onehot = _binary_scale_embedding(pp_left.astype(np.int32), 65)
        move_onehot = MOVE_ONEHOT(move_id)
        embedding = self.move_linear(move_onehot)
        return embedding + self.pp_linear(pp_onehot)

    def __call__(self, movesets: chex.Array):
        _encode = jax.vmap(self.encode_move)
        return _encode(movesets)


class EntityEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        entity_size = self.cfg.entity_size

        self.onehot_linear = nn.Dense(features=entity_size)
        self.species_linear = nn.Dense(features=entity_size)
        self.ability_linear = nn.Dense(features=entity_size)
        self.item_linear = nn.Dense(features=entity_size)
        self.moves_linear = nn.Dense(features=entity_size)

        self.level_linear = nn.Dense(features=entity_size)
        self.hp_linear = nn.Dense(features=entity_size)
        self.active_linear = nn.Dense(features=entity_size)

        self.merge = VectorMerge(entity_size, gating_type=GatingType.POINTWISE)

    def encode_entity(self, entity: chex.Array):
        hp = entity[FeatureEntity.HP]
        maxhp = entity[FeatureEntity.MAXHP].clip(min=1)

        hp_ratio = (hp / maxhp).clip(min=0, max=1)
        hp_token = (1023 * hp_ratio).astype(int)

        embeddings = [
            hp_ratio[jnp.newaxis],
            jax.nn.one_hot(entity[FeatureEntity.GENDER], NUM_GENDERS),
            jax.nn.one_hot(entity[FeatureEntity.STATUS], NUM_STATUS),
            jax.nn.one_hot(entity[FeatureEntity.BEING_CALLED_BACK], 2),
            jax.nn.one_hot(entity[FeatureEntity.TRAPPED], 2),
            jax.nn.one_hot(entity[FeatureEntity.NEWLY_SWITCHED], 2),
            jax.nn.one_hot(entity[FeatureEntity.TOXIC_TURNS], 8),
            jax.nn.one_hot(entity[FeatureEntity.SLEEP_TURNS], 4),
            jax.nn.one_hot(entity[FeatureEntity.FAINTED], 2),
        ]

        # Put all the encoded one-hots in a single boolean vector:
        boolean_code = jnp.concatenate(embeddings, axis=0)
        move_indices = jnp.array(
            [
                FeatureEntity.MOVEID0,
                FeatureEntity.MOVEID1,
                FeatureEntity.MOVEID2,
                FeatureEntity.MOVEID3,
            ]
        )
        pp_indices = jnp.array(
            [
                FeatureEntity.MOVEPP0,
                FeatureEntity.MOVEPP1,
                FeatureEntity.MOVEPP2,
                FeatureEntity.MOVEPP3,
            ]
        )
        moveset_onehot = jnp.concatenate(
            (
                MOVE_ONEHOT(entity[move_indices]),
                jax.vmap(partial(_binary_scale_embedding, world_dim=64))(
                    entity[pp_indices].astype(np.int32)
                ),
            ),
            axis=-1,
        )

        item_onehot = jnp.concatenate(
            (
                ITEM_ONEHOT(entity[FeatureEntity.ITEM]),
                jax.nn.one_hot(entity[FeatureEntity.ITEM_EFFECT], NUM_ITEM_EFFECTS),
            )
        )

        embedding = (
            self.hp_linear(_binary_scale_embedding(hp_token.astype(np.int32), 1024))
            + self.level_linear(
                _binary_scale_embedding(
                    entity[FeatureEntity.LEVEL].astype(np.int32), 101
                )
            )
            + self.active_linear(jax.nn.one_hot(entity[FeatureEntity.ACTIVE], 2))
            + self.onehot_linear(boolean_code.astype(np.float32))
        )
        embedding = self.merge(
            embedding,
            self.species_linear(SPECIES_ONEHOT(entity[FeatureEntity.SPECIES])),
            self.ability_linear(ABILITY_ONEHOT(entity[FeatureEntity.ABILITY])),
            self.item_linear(item_onehot),
            self.moves_linear(moveset_onehot).sum(0),
        )
        return embedding

    def __call__(self, active_entities: chex.Array, side_entities: chex.Array):
        _encode = jax.vmap(jax.vmap(self.encode_entity))

        active_embeddings = _encode(active_entities)
        side_embeddings = _encode(side_entities)

        side_species_token = side_entities[..., FeatureEntity.SPECIES]
        valid_team_mask = side_species_token != SpeciesEnum.species_none
        valid_team_mask = valid_team_mask | (
            side_species_token != SpeciesEnum.species_pad
        )

        return active_embeddings, side_embeddings, valid_team_mask


class SideEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        # self.boosts_linear = nn.Dense(features=self.cfg.entity_size)
        # self.side_conditions_linear = nn.Dense(features=self.cfg.entity_size)
        # self.volatile_status_linear = nn.Dense(features=self.cfg.entity_size)
        # self.hyphen_args_linear = nn.Dense(features=self.cfg.entity_size)
        # self.additional_information_linear = nn.Dense(features=self.cfg.entity_size)

        output_size = self.cfg.merge.output_size
        self.side_mlp = MLP((output_size,))
        self.team_mlp = MLP((output_size,))
        self.merge = VectorMerge(output_size=output_size, gating_type=GatingType.NONE)

    def encode(
        self,
        team_embedding: chex.Array,
        boosts: chex.Array,
        side_conditions: chex.Array,
        volatile_status: chex.Array,
        additional_information: chex.Array,
        hyphen_args: chex.Array,
    ):
        boosts_float = jnp.sign(boosts) * jnp.log1p(jnp.abs(boosts))
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

        boosts_encoding = jnp.concatenate(
            (boosts_float.flatten(), boosts_onehot.flatten()), axis=-1
        )

        total_pokemon = (
            additional_information[FeatureAdditionalInformation.TOTAL_POKEMON]
            .clip(min=1)
            .astype(int)
        )
        team_hp = additional_information[jnp.array(MEMBER_HP_INDICES)]
        team_hp_prob = team_hp / team_hp.sum().clip(min=1)
        team_hp_entropy = -jnp.sum(
            team_hp_prob * jnp.where(team_hp_prob > 0, jnp.log(team_hp_prob), 0)
        )
        num_types = additional_information[jnp.array(NUM_TYPES_INDICES)]
        num_types_prob = num_types / total_pokemon
        num_types_entropy = -jnp.sum(
            num_types_prob * jnp.where(num_types_prob > 0, jnp.log(num_types_prob), 0)
        )
        additional_information_encoding = jnp.concatenate(
            (
                jax.nn.one_hot(
                    additional_information[FeatureAdditionalInformation.NUM_FAINTED],
                    5,
                ),
                num_types_prob,
                num_types_entropy[None],
                additional_information[FeatureAdditionalInformation.WISHING][None],
                team_hp_entropy[None],
            )
        )

        side_encoding = jnp.concat(
            (
                boosts_encoding,
                side_conditions_onehot,
                volatile_status_onehot,
                hyphen_args_onehot,
                additional_information_encoding,
            ),
            axis=-1,
        )
        return self.merge(self.team_mlp(team_embedding), self.side_mlp(side_encoding))

        # return self.merge(
        #     self.boosts_linear(boosts_encoding),
        #     self.side_conditions_linear(side_conditions_onehot),
        #     self.volatile_status_linear(volatile_status_onehot),
        #     self.hyphen_args_linear(hyphen_args_onehot),
        #     self.additional_information_linear(additional_information_encoding),
        # )

    def __call__(
        self,
        team_embedding: chex.Array,
        boosts: chex.Array,
        side_conditions: chex.Array,
        volatile_status: chex.Array,
        additional_information: chex.Array,
        hyphen_args: chex.Array,
    ):
        _encode = jax.vmap(jax.vmap(self.encode))
        return _encode(
            team_embedding,
            boosts,
            side_conditions,
            volatile_status,
            additional_information,
            hyphen_args,
        )


class TeamEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.transformer = Transformer(**self.cfg.transformer.to_dict())
        self.to_vector = ToAvgVector(**self.cfg.to_vector.to_dict())

    def __call__(self, team_embeddings: chex.Array, valid_mask: chex.Array):
        team_embeddings = self.transformer(team_embeddings, valid_mask)
        teams_embedding = self.to_vector(team_embeddings, valid_mask)
        return teams_embedding, team_embeddings


class FieldEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.field_linear = nn.Dense(features=self.cfg.vector_size)
        self.player_linear = nn.Embed(num_embeddings=2, features=self.cfg.vector_size)
        self.turn_linear = nn.Embed(
            num_embeddings=2 * NUM_HISTORY, features=self.cfg.vector_size
        )
        self.history_move_linear = nn.Dense(features=self.cfg.vector_size)

    def encode(
        self,
        terrain: chex.Array,
        pseudoweather: chex.Array,
        weather: chex.Array,
        turn_context: chex.Array,
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
        history_move_onehot = MOVE_ONEHOT(turn_context[FeatureTurnContext.IS_MY_TURN])

        field_onehot = jnp.concatenate(
            (
                # terrain_onehot,
                pseudoweather_onehot,
                weather_onehot,
                weather_min_onehot,
                weather_max_onehot,
                move_counter_onehot,
                switch_counter_onehot,
            ),
            axis=-1,
        )

        embeddings = [
            self.field_linear(field_onehot),
            self.player_linear(turn_context[FeatureTurnContext.IS_MY_TURN]),
            self.turn_linear(relative_turn),
            self.history_move_linear(history_move_onehot),
        ]

        return jnp.stack(embeddings).sum(0)

    def __call__(
        self,
        terrain: chex.Array,
        pseudoweather: chex.Array,
        weather: chex.Array,
        turn_context: chex.Array,
    ):
        turn = turn_context[..., FeatureTurnContext.TURN]
        valid_mask = turn_context[..., FeatureTurnContext.VALID]
        max_turn = turn.max()
        _encode = partial(self.encode, max_turn=max_turn)
        field_embedding = jax.vmap(_encode)(
            terrain, pseudoweather, weather, turn_context
        )
        return valid_mask, field_embedding


class HistoryEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.transformer = Transformer(**self.cfg.transformer.to_dict())

    def __call__(self, state_history: chex.Array, valid_mask: chex.Array):
        return self.transformer(state_history, valid_mask)


def encode_team(team_embeddings: chex.Array, valid_mask: chex.Array):
    return jnp.max(jnp.where(valid_mask[..., None], team_embeddings, -1e9), axis=0)


class PublicEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.entity_encoder = EntityEncoder(self.cfg.entity_encoder)
        self.side_encoder = SideEncoder(self.cfg.side_encoder)
        self.field_encoder = FieldEncoder(self.cfg.field_encoder)
        self.history_encoder = HistoryEncoder(self.cfg.history_encoder)
        self.state_resnet = Resnet(**self.cfg.state_resnet)
        self.unet = UNet(8)
        self.compress = CNN(512)

    def __call__(self, env_step: EnvStep):
        _encode_entity = jax.vmap(jax.vmap(jax.vmap(self.entity_encoder.encode_entity)))
        team_embeddings = _encode_entity(env_step.active_entities)

        species_token = env_step.active_entities[..., FeatureEntity.SPECIES]
        invalid_mask = (species_token == SpeciesEnum.species_none) | (
            species_token == SpeciesEnum.species_pad
        )
        team_embeddings = jax.vmap(jax.vmap(encode_team))(
            team_embeddings, ~invalid_mask
        )

        side_embeddings = self.side_encoder(
            team_embeddings,
            env_step.boosts,
            env_step.side_conditions,
            env_step.volatile_status,
            env_step.additional_information,
            env_step.hyphen_args,
        )

        valid_mask, field_embeddings = self.field_encoder(
            env_step.terrain,
            env_step.pseudoweather,
            env_step.weather,
            env_step.turn_context,
        )

        field_embeddings = field_embeddings.reshape(-1, 32, 32)
        history = side_embeddings.reshape(field_embeddings.shape) + field_embeddings

        history = jnp.where(valid_mask[..., None, None], history, 0)
        state = self.unet(jnp.transpose(history, (1, 2, 0)))
        state = self.compress(state)

        return state


class SupervisedBackbone(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.public_encoder = PublicEncoder(self.cfg)
        self.next_turn = MLP((self.cfg.history_merge.output_size, 2))
        self.move_pred = MLP((self.cfg.history_merge.output_size, NUM_MOVES))
        self.action_pred = MLP((self.cfg.history_merge.output_size, 2))
        self.value_pred = MLP((self.cfg.history_merge.output_size, 1))

    def __call__(self, env_step: EnvStep):
        current_state = self.public_encoder(env_step)
        next_turn = self.next_turn(current_state)
        next_move = self.move_pred(current_state)
        next_action = self.action_pred(current_state)
        value = self.value_pred(current_state)
        return next_turn, next_move, next_action, value


class Encoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.public_encoder = PublicEncoder(self.cfg)

        self.move_encoder = MoveEncoder(self.cfg.move_encoder)
        self.state_merge = VectorMerge(**self.cfg.state_merge.to_dict())
        self.state_resnet = Resnet(**self.cfg.state_resnet)
        self.action_merge = VectorMerge(**self.cfg.action_merge.to_dict())

    def __call__(self, env_step: EnvStep):
        species_token = env_step.private_side_entities[..., FeatureEntity.SPECIES]
        invalid_team_mask = (species_token == SpeciesEnum.species_none) | (
            species_token == SpeciesEnum.species_pad
        )

        _encode_entity = jax.vmap(self.public_encoder.entity_encoder.encode_entity)
        private_side_embeddings = _encode_entity(env_step.private_side_entities)

        private_team_embedding = encode_team(
            private_side_embeddings, ~invalid_team_mask
        )

        current_state = self.public_encoder(env_step)
        current_state = self.state_merge(current_state, private_team_embedding)
        current_state = self.state_resnet(current_state)

        move_embeddings = self.move_encoder(env_step.moveset[0])

        action_entites = jnp.concatenate(
            (
                jnp.expand_dims(private_side_embeddings[0], axis=0).repeat(4, 0),
                private_side_embeddings,
            )
        )
        action_embeddings = self.action_merge(move_embeddings, action_entites)

        return current_state, action_embeddings
