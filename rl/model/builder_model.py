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
    BuilderEnvOutput,
    BuilderTransition,
    PolicyHeadOutput,
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
from rl.model.heads import (
    CategoricalValueLogitHead,
    HeadParams,
    PolicyQKHead,
    RegressionValueLogitHead,
)
from rl.model.modules import MLP, TransformerEncoder
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
        self.cfg.dtype

        dense_kwargs = dict(features=entity_size, dtype=self.cfg.dtype)

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

        embedding_init = nn.initializers.variance_scaling(
            1.0, "fan_in", "normal", out_axis=0
        )
        self.sos_embedding = self.param(
            "sos_embedding",
            embedding_init,
            (1, entity_size),
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

        self.positional_embedding = nn.Embed(6, entity_size, self.cfg.dtype)
        self.attribute_embedding = nn.Embed(
            NUM_PACKED_SET_FEATURES, entity_size, self.cfg.dtype
        )

        self.encoder = TransformerEncoder(**self.cfg.encoder.to_dict())
        self.norm_out = nn.RMSNorm(dtype=self.cfg.dtype)

        self.species_head_mlp = MLP()
        self.item_head_mlp = MLP()
        self.ability_head_mlp = MLP()
        self.move_head_mlp = MLP()
        self.ev_head_mlp = MLP()
        self.nature_head_mlp = MLP()
        self.gender_head_mlp = MLP()
        self.hiddenpower_head_mlp = MLP()
        self.teratype_head_mlp = MLP()

        self.species_head = PolicyQKHead(self.cfg.species_head)
        self.item_head = PolicyQKHead(self.cfg.item_head)
        self.ability_head = PolicyQKHead(self.cfg.ability_head)
        self.move_head = PolicyQKHead(self.cfg.move_head)
        self.ev_head = PolicyQKHead(self.cfg.ev_head)
        self.nature_head = PolicyQKHead(self.cfg.nature_head)
        self.gender_head = PolicyQKHead(self.cfg.gender_head)
        self.hiddenpower_head = PolicyQKHead(self.cfg.hiddenpower_head)
        self.teratype_head = PolicyQKHead(self.cfg.teratype_head)

        self.value_head_mlp = MLP()
        self.value_head = CategoricalValueLogitHead(self.cfg.value_head)

        self.conditional_entropy_head_mlp = MLP()
        self.conditional_entropy_head = RegressionValueLogitHead(self.cfg.entropy_head)

        self.niche_embedding = self.param(
            "niche_embedding", embedding_init, (self.cfg.num_niches, entity_size)
        )
        self.niche_pos_embedding = self.param(
            "niche_pos_embedding", embedding_init, (2, entity_size)
        )

        self.discriminator_head_mlp = MLP()
        self.discriminator_head = RegressionValueLogitHead(self.cfg.discriminator_head)

        self.diayn_value_head_mlp = MLP()
        self.diayn_value_head = RegressionValueLogitHead(self.cfg.diayn_value_head)

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

    def _forward_discriminator_head(self, embedding: jax.Array):
        embedding = self.discriminator_head_mlp(embedding)
        return self.discriminator_head(embedding)

    def _forward_diayn_value_head(self, embedding: jax.Array):
        embedding = self.diayn_value_head_mlp(embedding)
        return self.diayn_value_head(embedding)

    def _encode_team(
        self,
        token: jax.Array,
        position_id: jax.Array,
        attribute_id: jax.Array,
        species_keys: jax.Array,
        ability_keys: jax.Array,
        item_keys: jax.Array,
        move_keys: jax.Array,
        player_niche_id: jax.Array,
        opponent_niche_id: jax.Array,
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
        )[..., None] * jnp.take(
            self.nature_embedding.astype(self.cfg.dtype), token, axis=0, mode="clip"
        )
        gender_embeddings = (
            attribute_id == PackedSetFeature.PACKED_SET_FEATURE__GENDER
        )[..., None] * jnp.take(
            self.gender_embedding.astype(self.cfg.dtype), token, axis=0, mode="clip"
        )
        ev_embeddings = (
            (attribute_id >= PackedSetFeature.PACKED_SET_FEATURE__HP_EV)
            & (attribute_id <= PackedSetFeature.PACKED_SET_FEATURE__SPE_EV)
        )[..., None] * jnp.take(
            self.ev_embedding.astype(self.cfg.dtype), token, axis=0, mode="clip"
        )
        typechart_embeddings = self.typechart_embedding.astype(self.cfg.dtype)
        hiddenpower_embeddings = (
            attribute_id == PackedSetFeature.PACKED_SET_FEATURE__HIDDENPOWERTYPE
        )[..., None] * jnp.take(typechart_embeddings, token, axis=0, mode="clip")
        teratype_embedding = (
            attribute_id == PackedSetFeature.PACKED_SET_FEATURE__TERATYPE
        )[..., None] * jnp.take(typechart_embeddings, token, axis=0, mode="clip")

        position_embedding = self.positional_embedding(position_id)
        attribute_embedding = self.attribute_embedding(attribute_id)

        packed_set_embeddings = (
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
            ).reshape(-1, self.cfg.entity_size)
        )

        packed_set_embeddings = jnp.concatenate(
            (self.sos_embedding.astype(self.cfg.dtype), packed_set_embeddings), axis=0
        )

        # Niche-free encoding: keep SOS at both prefix positions so the discriminator
        # cannot trivially decode z from the hidden state.
        sos = self.sos_embedding.astype(self.cfg.dtype)
        packed_no_niche = jnp.concatenate(
            (sos, sos, packed_set_embeddings[1:]), axis=0
        )
        seq_len = packed_no_niche.shape[0]
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))[None]
        hidden_no_niche = self.norm_out(self.encoder(packed_no_niche, causal_mask))

        # Prepend player niche (pos 0) and opponent niche (pos 1) instead of the single SOS.
        player_niche_emb = jnp.take(
            self.niche_embedding.astype(self.cfg.dtype), player_niche_id, axis=0
        )[None]
        opponent_niche_emb = jnp.take(
            self.niche_embedding.astype(self.cfg.dtype), opponent_niche_id, axis=0
        )[None]

        niche_pos_emb = self.niche_pos_embedding.astype(self.cfg.dtype)

        packed_with_niche = jnp.concatenate(
            (
                player_niche_emb + niche_pos_emb[0],
                opponent_niche_emb + niche_pos_emb[1],
                packed_set_embeddings[1:],
            ),
            axis=0,
        )
        hidden_with_niche = self.norm_out(self.encoder(packed_with_niche, causal_mask))

        return hidden_with_niche, hidden_no_niche

    def _forward(
        self,
        hidden_state: jax.Array,
        hidden_state_no_niche: jax.Array,
        env_step: BuilderEnvOutput,
        actor_output: BuilderActorOutput,
        species_keys: jax.Array,
        ability_keys: jax.Array,
        item_keys: jax.Array,
        move_keys: jax.Array,
        head_params: HeadParams,
    ) -> BuilderActorOutput:

        value_head = self._forward_value_head(hidden_state)
        conditional_entropy_head = self._forward_conditional_entropy_head(hidden_state)
        discriminator_head = self._forward_discriminator_head(hidden_state_no_niche)
        diayn_value_head = self._forward_diayn_value_head(hidden_state_no_niche)

        species_head = self.species_head(
            self.species_head_mlp(hidden_state),
            species_keys,
            actor_output.action_head,
            env_step.species_mask,
            head_params=head_params,
        )
        item_head = self.item_head(
            self.item_head_mlp(hidden_state),
            item_keys,
            actor_output.action_head,
            env_step.item_mask,
            head_params=head_params,
        )
        ability_head = self.ability_head(
            self.ability_head_mlp(hidden_state),
            ability_keys,
            actor_output.action_head,
            env_step.ability_mask,
            head_params=head_params,
        )
        move_head = self.move_head(
            self.move_head_mlp(hidden_state),
            move_keys,
            actor_output.action_head,
            env_step.move_mask,
            head_params=head_params,
        )
        ev_head = self.ev_head(
            self.ev_head_mlp(hidden_state),
            self.ev_embedding,
            actor_output.action_head,
            env_step.ev_mask,
            head_params=head_params,
        )
        nature_head = self.nature_head(
            self.nature_head_mlp(hidden_state),
            self.nature_embedding,
            actor_output.action_head,
            env_step.nature_mask,
            head_params=head_params,
        )
        gender_head = self.gender_head(
            self.gender_head_mlp(hidden_state),
            self.gender_embedding,
            actor_output.action_head,
            env_step.gender_mask,
            head_params=head_params,
        )
        teratype_head = self.teratype_head(
            self.teratype_head_mlp(hidden_state),
            self.typechart_embedding,
            actor_output.action_head,
            env_step.teratype_mask,
            head_params=head_params,
        )

        action_indices = jnp.stack(
            (
                species_head.action_index,
                item_head.action_index,
                ability_head.action_index,
                move_head.action_index,
                ev_head.action_index,
                nature_head.action_index,
                gender_head.action_index,
                teratype_head.action_index,
            ),
        )
        log_probs = jnp.stack(
            (
                species_head.log_prob,
                item_head.log_prob,
                ability_head.log_prob,
                move_head.log_prob,
                ev_head.log_prob,
                nature_head.log_prob,
                gender_head.log_prob,
                teratype_head.log_prob,
            ),
        )
        entropies = jnp.stack(
            (
                species_head.entropy,
                item_head.entropy,
                ability_head.entropy,
                move_head.entropy,
                ev_head.entropy,
                nature_head.entropy,
                gender_head.entropy,
                teratype_head.entropy,
            )
        )
        normalized_entropies = jnp.stack(
            (
                species_head.normalized_entropy,
                item_head.normalized_entropy,
                ability_head.normalized_entropy,
                move_head.normalized_entropy,
                ev_head.normalized_entropy,
                nature_head.normalized_entropy,
                gender_head.normalized_entropy,
                teratype_head.normalized_entropy,
            )
        )
        mask = jnp.stack(
            (
                env_step.curr_attribute == PackedSetFeature.PACKED_SET_FEATURE__SPECIES,
                env_step.curr_attribute == PackedSetFeature.PACKED_SET_FEATURE__ITEM,
                env_step.curr_attribute == PackedSetFeature.PACKED_SET_FEATURE__ABILITY,
                (env_step.curr_attribute >= PackedSetFeature.PACKED_SET_FEATURE__MOVE1)
                & (
                    env_step.curr_attribute
                    <= PackedSetFeature.PACKED_SET_FEATURE__MOVE4
                ),
                (env_step.curr_attribute >= PackedSetFeature.PACKED_SET_FEATURE__HP_EV)
                & (
                    env_step.curr_attribute
                    <= PackedSetFeature.PACKED_SET_FEATURE__SPE_EV
                ),
                env_step.curr_attribute == PackedSetFeature.PACKED_SET_FEATURE__NATURE,
                env_step.curr_attribute == PackedSetFeature.PACKED_SET_FEATURE__GENDER,
                env_step.curr_attribute
                == PackedSetFeature.PACKED_SET_FEATURE__TERATYPE,
            )
        )

        action_head = PolicyHeadOutput(
            action_index=(action_indices @ mask)
            .astype(jnp.int32)
            .reshape(-1)
            .squeeze(),
            log_prob=(log_probs @ mask).astype(self.cfg.dtype).reshape(-1).squeeze(),
            entropy=(entropies @ mask).astype(self.cfg.dtype).reshape(-1).squeeze(),
            normalized_entropy=(normalized_entropies @ mask)
            .astype(self.cfg.dtype)
            .reshape(-1)
            .squeeze(),
        )

        return BuilderActorOutput(
            action_head=action_head,
            value_head=value_head,
            conditional_entropy_head=conditional_entropy_head,
            discriminator_head=discriminator_head,
            diayn_value_head=diayn_value_head,
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

        hidden_states, hidden_states_no_niche = self._encode_team(
            jnp.take(team_tokens, order, axis=0),
            actor_input.history.member_position,
            actor_input.history.member_attribute,
            species_keys,
            ability_keys,
            item_keys,
            move_keys,
            player_niche_id=jnp.squeeze(actor_input.history.niche_id),
            opponent_niche_id=jnp.squeeze(actor_input.history.opponent_niche_id),
        )

        hidden_state = jnp.take(hidden_states, actor_input.env.ts, axis=0)
        hidden_state_no_niche = jnp.take(hidden_states_no_niche, actor_input.env.ts, axis=0)

        return jax.vmap(
            functools.partial(self._forward, head_params=head_params),
            in_axes=(0, 0, 0, 0, None, None, None, None),
        )(
            hidden_state,
            hidden_state_no_niche,
            actor_input.env,
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


def get_packed_team_string(packed_team_member_tokens: jax.Array):
    # 1. Reshape flat array to (Team_Size, Features)
    #    We assume NUM_PACKED_SET_FEATURES is available globally or imported
    reshaped_tokens = packed_team_member_tokens.reshape(-1, NUM_PACKED_SET_FEATURES)

    reconstructed_sets = []

    for row in reshaped_tokens:
        # Check if the species is set (non-zero). If 0, it's an empty/padding slot.
        # if row[PackedSetFeature.PACKED_SET_FEATURE__SPECIES] == 0:
        #     continue

        species = ITOS["species"].get(
            row[PackedSetFeature.PACKED_SET_FEATURE__SPECIES].item(), ""
        )
        item = ITOS["items"].get(
            row[PackedSetFeature.PACKED_SET_FEATURE__ITEM].item(), ""
        )
        ability = ITOS["abilities"].get(
            row[PackedSetFeature.PACKED_SET_FEATURE__ABILITY].item(), ""
        )

        # 2. Fix moves iteration
        #    Using indices 0 to 3 added to base MOVE1 index
        moves = [
            ITOS["moves"].get(
                row[PackedSetFeature.PACKED_SET_FEATURE__MOVE1 + i].item(), ""
            )
            for i in range(4)
        ]

        nature = ITOS["natures"].get(
            row[PackedSetFeature.PACKED_SET_FEATURE__NATURE].item(), ""
        )

        # 3. EVs iteration (Indices 0 to 5 added to base HP_EV)
        evs = [
            str(4 * row[PackedSetFeature.PACKED_SET_FEATURE__HP_EV + i].item())
            for i in range(6)
        ]

        gender = ITOS["gendername"].get(
            row[PackedSetFeature.PACKED_SET_FEATURE__GENDER].item(), ""
        )

        # Static/Unused fields
        ivs = ""
        shiny = ""
        level = ""
        happiness = ""  # Default is usually 255
        pokeball = ""

        hiddenpowertype = ITOS["typechart"].get(
            row[PackedSetFeature.PACKED_SET_FEATURE__HIDDENPOWERTYPE].item(), ""
        )

        gigantamax = ""
        dynamaxlevel = ""

        teratype = ITOS["typechart"].get(
            row[PackedSetFeature.PACKED_SET_FEATURE__TERATYPE].item(), ""
        )

        # 4. Safer f-string formatting (using single quotes for joins)
        reconstructed_set = (
            f"|{species}|{item}|{ability}|{','.join(moves)}|{nature}|"
            f"{','.join(evs)}|{gender}|{ivs}|{shiny}|{level}|"
            f"{happiness},{pokeball},{hiddenpowertype},{gigantamax},{dynamaxlevel},{teratype}"
        )

        reconstructed_sets.append(reconstructed_set)

    return "]".join(reconstructed_sets)


def main(generation: int = 9):
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
        # builder_head_params=HeadParams(temp=0.2),
    )

    builder_env = TeamBuilderEnvironment(generation=generation, smogon_format="ou")

    i = 0
    while True:

        rng_key, key = jax.random.split(key, 2)
        builder_subkeys = jax.random.split(rng_key, builder_env.length)

        build_traj = []

        builder_actor_input = builder_env.reset(builder_subkeys[0])

        for builder_step_index in range(1, builder_subkeys.shape[0] + 1):
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

            builder_actor_input = builder_env.step(builder_agent_output)

            # print(get_packed_team_string(team_tokens))
            # print(i)

        builder_trajectory: BuilderTransition = jax.tree.map(
            lambda *xs: np.array(jnp.stack(xs)), *build_traj
        )

        team_tokens = builder_actor_input.history.packed_team_member_tokens.reshape(
            -1, NUM_PACKED_SET_FEATURES
        )
        print("\n".join(get_packed_team_string(team_tokens).split("]")))

        assert np.all(
            team_tokens[..., PackedSetFeature.PACKED_SET_FEATURE__SPECIES]
            > SpeciesEnum.SPECIES_ENUM___UNK
        ).item()

        assert np.all(
            team_tokens[
                ...,
                PackedSetFeature.PACKED_SET_FEATURE__HP_EV : PackedSetFeature.PACKED_SET_FEATURE__SPE_EV
                + 1,
            ].sum(axis=-1)
            <= 128
        ).item()

        print(i, len(build_traj))
        i += 1


if __name__ == "__main__":
    debug = True
    with jax.disable_jit(debug):
        main()
