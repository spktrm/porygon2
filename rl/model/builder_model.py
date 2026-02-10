import functools
from functools import partial
from pprint import pprint

import chex
import cloudpickle as pickle
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from rl.actor.agent import Agent
from rl.environment.data import (
    ITOS,
    NUM_ABILITIES,
    NUM_GENDERS,
    NUM_ITEMS,
    NUM_MOVES,
    NUM_NATURES,
    NUM_PACKED_SET_FEATURES,
    NUM_SPECIES,
    NUM_TYPECHART,
    ONEHOT_ENCODERS,
    PACKED_SET_MAX_VALUES,
)
from rl.environment.env import TeamBuilderEnvironment
from rl.environment.interfaces import (
    BuilderActorInput,
    BuilderActorOutput,
    BuilderTransition,
)
from rl.environment.protos.enums_pb2 import (
    AbilitiesEnum,
    ItemsEnum,
    MovesEnum,
    SpeciesEnum,
)
from rl.environment.protos.features_pb2 import PackedSetFeature
from rl.environment.utils import get_ex_builder_step
from rl.learner.config import get_learner_config
from rl.model.config import get_builder_model_config
from rl.model.heads import HeadParams, PolicyQKHead, RegressionValueLogitHead
from rl.model.modules import MLP, TransformerEncoder, dense_layer
from rl.model.utils import get_most_recent_file, get_num_params


def _encode_one_hot(
    entity: jax.Array,
    feature_idx: int,
    max_values: dict[int, int],
    value_offset: int = 0,
) -> tuple[int, int]:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    return entity[feature_idx] + value_offset, max_values[feature_idx] + 1


_encode_one_hot_set = partial(_encode_one_hot, max_values=PACKED_SET_MAX_VALUES)


class Porygon2BuilderModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """
        Initializes the encoder, policy head, and value head using the configuration.
        """
        entity_size = self.cfg.entity_size
        dtype = self.cfg.dtype

        dense_kwargs = dict(features=entity_size, dtype=self.cfg.dtype)

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

        self.sos_embedding = self.param(
            "sos_embedding",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, entity_size),
        )
        self.unk_embedding = self.param(
            "unk_embedding",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, entity_size),
        )

        embedding_init = nn.initializers.variance_scaling(
            1.0, "fan_in", "normal", out_axis=0
        )
        self.nature_embedding = self.param(
            "nature_embedding",
            embedding_init,
            (NUM_NATURES, entity_size),
        )
        self.gender_embedding = self.param(
            "gender_embedding",
            embedding_init,
            (NUM_GENDERS, entity_size),
        )
        self.ev_embedding = self.param(
            "ev_embedding",
            embedding_init,
            (64, entity_size),
        )
        self.typechart_embedding = self.param(
            "typechart_embedding",
            embedding_init,
            (NUM_TYPECHART, entity_size),
        )

        self.positional_embedding = nn.Embed(
            num_embeddings=6, features=entity_size, dtype=dtype
        )
        self.attribute_embedding = nn.Embed(
            num_embeddings=NUM_PACKED_SET_FEATURES, features=entity_size, dtype=dtype
        )

        self.encoder = TransformerEncoder(**self.cfg.encoder.to_dict())

        self.species_head_mlp = MLP()
        self.item_head_mlp = MLP()
        self.ability_head_mlp = MLP()
        self.move_head_mlp = MLP()
        self.ev_head_mlp = MLP()
        self.nature_head_mlp = MLP()
        self.gender_head_mlp = MLP()
        self.teratype_head_mlp = MLP()

        self.species_head = PolicyQKHead(self.cfg.species_head)
        self.item_head = PolicyQKHead(self.cfg.species_head)
        self.ability_head = PolicyQKHead(self.cfg.species_head)
        self.move_head = PolicyQKHead(self.cfg.species_head)
        self.ev_head = PolicyQKHead(self.cfg.species_head)
        self.nature_head = PolicyQKHead(self.cfg.species_head)
        self.gender_head = PolicyQKHead(self.cfg.species_head)
        self.teratype_head = PolicyQKHead(self.cfg.species_head)

        self.value_head_mlp = MLP()
        self.value_head = RegressionValueLogitHead(self.cfg.value_head)

        self.conditional_entropy_head_mlp = MLP()
        self.conditional_entropy_head = RegressionValueLogitHead(self.cfg.value_head)

    def _embed_species(self, token: jax.Array):
        mask = ~(
            (token == SpeciesEnum.SPECIES_ENUM___UNSPECIFIED)
            | (token == SpeciesEnum.SPECIES_ENUM___PAD)
        )
        _ohe_encoder = ONEHOT_ENCODERS[self.cfg.generation]["species"]
        return mask * self.species_linear(_ohe_encoder(token))

    def _embed_item(self, token: jax.Array):
        mask = ~(
            (token == ItemsEnum.ITEMS_ENUM___UNSPECIFIED)
            | (token == ItemsEnum.ITEMS_ENUM___PAD)
        )
        _ohe_encoder = ONEHOT_ENCODERS[self.cfg.generation]["items"]
        return mask * self.items_linear(_ohe_encoder(token))

    def _embed_ability(self, token: jax.Array):
        mask = ~(
            (token == AbilitiesEnum.ABILITIES_ENUM___UNSPECIFIED)
            | (token == AbilitiesEnum.ABILITIES_ENUM___PAD)
        )
        _ohe_encoder = ONEHOT_ENCODERS[self.cfg.generation]["abilities"]
        return mask * self.abilities_linear(_ohe_encoder(token))

    def _embed_move(self, token: jax.Array):
        mask = ~(
            (token == MovesEnum.MOVES_ENUM___UNSPECIFIED)
            | (token == MovesEnum.MOVES_ENUM___PAD)
        )
        _ohe_encoder = ONEHOT_ENCODERS[self.cfg.generation]["moves"]
        return mask * self.moves_linear(_ohe_encoder(token))

    def _forward_value_head(self, embedding: jax.Array):
        embedding = self.value_head_mlp(embedding)
        return self.value_head(embedding)

    def _forward_conditional_entropy_head(self, embedding: jax.Array):
        embedding = self.conditional_entropy_head_mlp(embedding)
        return self.conditional_entropy_head(embedding)

    def _encode_team(
        self,
        token: jax.Array,
        position_id: jax.Array,
        attribute_id: jax.Array,
        species_keys: jax.Array,
        ability_keys: jax.Array,
        item_keys: jax.Array,
        move_keys: jax.Array,
    ):
        species_embeddings = (
            attribute_id == PackedSetFeature.PACKED_SET_FEATURE__SPECIES
        )[..., None] * jnp.take(species_keys, token, axis=0, mode="clip")
        ability_embeddings = (
            attribute_id == PackedSetFeature.PACKED_SET_FEATURE__ABILITY
        )[..., None] * jnp.take(ability_keys, token, axis=0, mode="clip")
        item_embeddings = (attribute_id == PackedSetFeature.PACKED_SET_FEATURE__ITEM)[
            ..., None
        ] * jnp.take(item_keys, token, axis=0, mode="clip")
        move_embeddings = (
            (attribute_id >= PackedSetFeature.PACKED_SET_FEATURE__MOVE1)
            & (attribute_id <= PackedSetFeature.PACKED_SET_FEATURE__MOVE4)
        )[..., None] * jnp.take(move_keys, token, axis=0, mode="clip")
        nature_embeddings = (
            attribute_id == PackedSetFeature.PACKED_SET_FEATURE__NATURE
        )[..., None] * jnp.take(self.nature_embedding, token, axis=0, mode="clip")
        gender_embeddings = (
            attribute_id == PackedSetFeature.PACKED_SET_FEATURE__GENDER
        )[..., None] * jnp.take(self.gender_embedding, token, axis=0, mode="clip")
        ev_embeddings = (
            (attribute_id >= PackedSetFeature.PACKED_SET_FEATURE__HP_EV)
            & (attribute_id <= PackedSetFeature.PACKED_SET_FEATURE__SPE_EV)
        )[..., None] * jnp.take(self.ev_embedding, token, axis=0, mode="clip")
        hiddenpower_embeddings = (
            attribute_id >= PackedSetFeature.PACKED_SET_FEATURE__HIDDENPOWERTYPE
        )[..., None] * jnp.take(self.typechart_embedding, token, axis=0, mode="clip")
        teratype_embedding = (
            attribute_id == PackedSetFeature.PACKED_SET_FEATURE__TERATYPE
        )[..., None] * jnp.take(self.typechart_embedding, token, axis=0, mode="clip")

        position_embedding = self.positional_embedding(position_id)
        attribute_embedding = self.attribute_embedding(attribute_id)

        embeddings = (
            position_embedding
            + attribute_embedding
            + (
                species_embeddings
                + ability_embeddings
                + item_embeddings
                + move_embeddings
                + nature_embeddings
                + gender_embeddings
                + ev_embeddings
                + hiddenpower_embeddings
                + teratype_embedding
            )
        )
        embeddings = jnp.concatenate(
            (self.sos_embedding.astype(self.cfg.dtype), embeddings), axis=0
        )

        causal_mask = jnp.tril(
            jnp.ones((embeddings.shape[0], embeddings.shape[0]), dtype=jnp.bool)
        )[None]
        return self.encoder(embeddings, causal_mask)

    def _forward(
        self,
        current_embedding: jax.Array,
        species_mask: jax.Array,
        item_mask: jax.Array,
        ability_mask: jax.Array,
        move_mask: jax.Array,
        ev_mask: jax.Array,
        nature_mask: jax.Array,
        teratype_mask: jax.Array,
        gender_mask: jax.Array,
        actor_output: BuilderActorOutput,
        species_keys: jax.Array,
        ability_keys: jax.Array,
        item_keys: jax.Array,
        move_keys: jax.Array,
        head_params: HeadParams,
    ) -> BuilderActorOutput:

        value_head = self._forward_value_head(current_embedding)
        conditional_entropy_head = self._forward_conditional_entropy_head(
            current_embedding
        )

        species_head = self.species_head(
            self.species_head_mlp(current_embedding),
            species_keys,
            actor_output.species_head,
            species_mask,
            head_params=head_params,
        )
        item_head = self.item_head(
            self.item_head_mlp(current_embedding),
            item_keys,
            actor_output.item_head,
            item_mask,
            head_params=head_params,
        )
        ability_head = self.ability_head(
            self.ability_head_mlp(current_embedding),
            ability_keys,
            actor_output.ability_head,
            ability_mask,
            head_params=head_params,
        )
        move_head = self.move_head(
            self.move_head_mlp(current_embedding),
            move_keys,
            actor_output.move_head,
            move_mask,
            head_params=head_params,
        )
        ev_head = self.ev_head(
            self.ev_head_mlp(current_embedding),
            self.ev_embedding,
            actor_output.ev_head,
            ev_mask,
            head_params=head_params,
        )
        nature_head = self.nature_head(
            self.nature_head_mlp(current_embedding),
            self.nature_embedding,
            actor_output.nature_head,
            nature_mask,
            head_params=head_params,
        )
        gender_head = self.gender_head(
            self.gender_head_mlp(current_embedding),
            self.gender_embedding,
            actor_output.gender_head,
            gender_mask,
            head_params=head_params,
        )
        teratype_head = self.teratype_head(
            self.teratype_head_mlp(current_embedding),
            self.typechart_embedding,
            actor_output.teratype_head,
            teratype_mask,
            head_params=head_params,
        )

        return BuilderActorOutput(
            species_head=species_head,
            item_head=item_head,
            ability_head=ability_head,
            move_head=move_head,
            ev_head=ev_head,
            nature_head=nature_head,
            gender_head=gender_head,
            teratype_head=teratype_head,
            value_head=value_head,
            conditional_entropy_head=conditional_entropy_head,
        )

    def __call__(
        self,
        actor_input: BuilderActorInput,
        actor_output: BuilderActorOutput,
        head_params: HeadParams,
    ) -> BuilderActorOutput:
        species_keys = jax.vmap(self._embed_species)(np.arange(NUM_SPECIES))
        ability_keys = jax.vmap(self._embed_ability)(np.arange(NUM_ABILITIES))
        item_keys = jax.vmap(self._embed_item)(np.arange(NUM_ITEMS))
        move_keys = jax.vmap(self._embed_move)(np.arange(NUM_MOVES))

        team_tokens = actor_input.history.packed_team_member_tokens
        order = actor_input.history.order
        member_position = actor_input.history.member_position
        member_attribute = actor_input.history.member_attribute

        hidden_states = self._encode_team(
            jnp.take(team_tokens, order),
            member_position,
            member_attribute,
            species_keys,
            ability_keys,
            item_keys,
            move_keys,
        )
        hidden_state = jnp.take(hidden_states, actor_input.env.ts, axis=0)

        return jax.vmap(
            functools.partial(self._forward, head_params=head_params),
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None),
        )(
            hidden_state,
            actor_input.env.species_mask,
            actor_input.env.item_mask,
            actor_input.env.ability_mask,
            actor_input.env.move_mask,
            actor_input.env.ev_mask,
            actor_input.env.nature_mask,
            actor_input.env.teratype_mask,
            actor_input.env.gender_mask,
            actor_output,
            species_keys,
            ability_keys,
            item_keys,
            move_keys,
        )


def get_builder_model(config: ConfigDict = None) -> nn.Module:
    if config is None:
        config = get_builder_model_config()
    return Porygon2BuilderModel(config)


def print_packed_team_member_tokens(packed_team_member_tokens: jax.Array):
    for row in packed_team_member_tokens.reshape(-1, NUM_PACKED_SET_FEATURES):
        species = ITOS["species"].get(
            row[PackedSetFeature.PACKED_SET_FEATURE__SPECIES].item(), ""
        )
        item = ITOS["items"].get(
            row[PackedSetFeature.PACKED_SET_FEATURE__ITEM].item(), ""
        )
        ability = ITOS["abilities"].get(
            row[PackedSetFeature.PACKED_SET_FEATURE__ABILITY].item(), ""
        )
        moves = [
            ITOS["moves"].get(
                row[PackedSetFeature.PACKED_SET_FEATURE__MOVE1 + i].item(), ""
            )
            for i in range(4)
        ]
        nature = ITOS["natures"].get(
            row[PackedSetFeature.PACKED_SET_FEATURE__NATURE].item(), ""
        )
        evs = [
            str(4 * row[PackedSetFeature.PACKED_SET_FEATURE__HP_EV + i].item())
            for i in range(6)
        ]
        gender = ITOS["gendername"].get(
            row[PackedSetFeature.PACKED_SET_FEATURE__GENDER].item(), ""
        )
        ivs = ""
        shiny = ""
        level = ""
        happiness = ""
        pokeball = ""
        hiddenpowertype = ITOS["typechart"].get(
            row[PackedSetFeature.PACKED_SET_FEATURE__HIDDENPOWERTYPE].item(), ""
        )
        gigantamax = ""
        dynamaxlevel = ""
        teratype = ITOS["typechart"].get(
            row[PackedSetFeature.PACKED_SET_FEATURE__TERATYPE].item(), ""
        )

        reconstructed_set = f"|{species}|{item}|{ability}|{",".join(moves)}|{nature}|{",".join(evs)}|{gender}|{ivs}|{shiny}|{level}|{happiness},{pokeball},{hiddenpowertype},{gigantamax},{dynamaxlevel},{teratype}"
        if "_UNSPECIFIED" in reconstructed_set:
            print(f"Invalid set with unspecified features: {reconstructed_set}")
        print(reconstructed_set)


def main(debug: bool = False, generation: int = 9):
    get_learner_config()

    actor_model_config = get_builder_model_config(generation, train=False)
    actor_network = get_builder_model(actor_model_config)

    learner_model_config = get_builder_model_config(generation, train=True)
    learner_network = get_builder_model(learner_model_config)

    ex_actor_input, ex_actor_output = jax.device_put(
        jax.tree.map(lambda x: x[:, 0], get_ex_builder_step())
    )
    key = jax.random.key(42)

    latest_ckpt = get_most_recent_file(f"./ckpts/gen{generation}")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        builder_params = step["builder_state"]["params"]
    else:
        builder_params = functools.partial(
            learner_network.init, head_params=HeadParams()
        )(key, ex_actor_input, ex_actor_output)

    pprint(get_num_params(builder_params))

    agent = Agent(
        builder_apply_fn=actor_network.apply,
        # builder_head_params=HeadParams(temp=0.8, min_p=0.1),
    )

    builder_env = TeamBuilderEnvironment(generation=generation, smogon_format="ou")

    while True:

        rng_key, key = jax.random.split(key, 2)
        builder_subkeys = jax.random.split(rng_key, builder_env.length)

        build_traj = []

        with jax.disable_jit(debug):
            builder_actor_input = builder_env.reset(builder_subkeys[0])

        for builder_step_index in range(1, builder_subkeys.shape[0] + 2):
            with jax.disable_jit(debug):
                builder_agent_output = agent.step_builder(
                    builder_subkeys[builder_step_index],
                    builder_params,
                    builder_actor_input,
                )
            builder_transition = BuilderTransition(
                env_output=builder_actor_input.env,
                agent_output=builder_agent_output,
            )
            build_traj.append(builder_transition)
            if builder_actor_input.env.done.item():
                break
            with jax.disable_jit(debug):
                builder_actor_input = builder_env.step(builder_agent_output)

            print(
                builder_actor_input.history.packed_team_member_tokens.reshape(
                    -1, NUM_PACKED_SET_FEATURES
                )
            )

        builder_trajectory: BuilderTransition = jax.tree.map(
            lambda *xs: np.array(jnp.stack(xs)), *build_traj
        )

        assert np.all(
            builder_actor_input.history.packed_team_member_tokens[
                ..., PackedSetFeature.PACKED_SET_FEATURE__SPECIES
            ]
            > SpeciesEnum.SPECIES_ENUM___UNK
        ).item()

        print_packed_team_member_tokens(
            builder_actor_input.history.packed_team_member_tokens
        )
        print()


if __name__ == "__main__":
    main()
