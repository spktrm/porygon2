from dotenv import load_dotenv

load_dotenv()
import functools
import json
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
    NUM_SPECIES,
    ONEHOT_ENCODERS,
    PACKED_SET_MAX_VALUES,
    SET_MASK,
    SET_TOKENS,
)
from rl.environment.env import TeamBuilderEnvironment
from rl.environment.interfaces import (
    BuilderActorInput,
    BuilderActorOutput,
    BuilderEnvOutput,
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
from rl.model.heads import (
    HeadParams,
    PolicyLogitHeadInner,
    PolicyQKHead,
    ValueLogitHead,
)
from rl.model.modules import (
    SumEmbeddings,
    TransformerEncoder,
    create_attention_mask,
    one_hot_concat_jax,
)
from rl.model.utils import get_most_recent_file, get_num_params, legal_log_policy


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
        self.metagame_embeddings = nn.Embed(
            num_embeddings=self.cfg.metagame_vocab_size,
            features=2 * entity_size,
            dtype=self.cfg.dtype,
        )

        self.packed_set_merge = SumEmbeddings(entity_size, dtype=dtype)
        self.packed_set_query_merge = SumEmbeddings(entity_size, dtype=dtype)

        transformer_config = self.cfg.transformer.to_dict()

        self.unk_embedding = self.param(
            "unk_embedding",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, entity_size),
        )

        self.encoder = TransformerEncoder(**transformer_config)

        self.species_head = PolicyQKHead(self.cfg.species_head)
        self.packed_set_head = PolicyQKHead(self.cfg.packed_set_head)

        self.value_merge = SumEmbeddings(output_size=2 * entity_size, dtype=dtype)
        self.value_head = ValueLogitHead(self.cfg.value_head)

        self.metagame_head = PolicyLogitHeadInner(self.cfg.metagame_head)

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

    def _embed_packed_set(self, packed_set: jax.Array):
        """
        Encodes the packed set tokens into embeddings.
        """
        move_indices = np.array(
            [
                PackedSetFeature.PACKED_SET_FEATURE__MOVE1,
                PackedSetFeature.PACKED_SET_FEATURE__MOVE2,
                PackedSetFeature.PACKED_SET_FEATURE__MOVE3,
                PackedSetFeature.PACKED_SET_FEATURE__MOVE4,
            ]
        )
        move_tokens = packed_set[move_indices]
        move_encodings = jax.vmap(self._embed_move)(move_tokens)

        species_token = packed_set[PackedSetFeature.PACKED_SET_FEATURE__SPECIES]
        ability_token = packed_set[PackedSetFeature.PACKED_SET_FEATURE__ABILITY]
        item_token = packed_set[PackedSetFeature.PACKED_SET_FEATURE__ITEM]

        boolean_code = one_hot_concat_jax(
            [
                _encode_one_hot_set(
                    packed_set, PackedSetFeature.PACKED_SET_FEATURE__NATURE
                ),
                _encode_one_hot_set(
                    packed_set, PackedSetFeature.PACKED_SET_FEATURE__GENDER
                ),
                _encode_one_hot_set(
                    packed_set, PackedSetFeature.PACKED_SET_FEATURE__HIDDENPOWERTYPE
                ),
                _encode_one_hot_set(
                    packed_set, PackedSetFeature.PACKED_SET_FEATURE__TERATYPE
                ),
            ],
            dtype=self.cfg.dtype,
        )

        ev_indices = np.array(
            [
                PackedSetFeature.PACKED_SET_FEATURE__HP_EV,
                PackedSetFeature.PACKED_SET_FEATURE__ATK_EV,
                PackedSetFeature.PACKED_SET_FEATURE__DEF_EV,
                PackedSetFeature.PACKED_SET_FEATURE__SPA_EV,
                PackedSetFeature.PACKED_SET_FEATURE__SPD_EV,
                PackedSetFeature.PACKED_SET_FEATURE__SPE_EV,
            ]
        )
        iv_indices = np.array(
            [
                PackedSetFeature.PACKED_SET_FEATURE__HP_IV,
                PackedSetFeature.PACKED_SET_FEATURE__ATK_IV,
                PackedSetFeature.PACKED_SET_FEATURE__DEF_IV,
                PackedSetFeature.PACKED_SET_FEATURE__SPA_IV,
                PackedSetFeature.PACKED_SET_FEATURE__SPD_IV,
                PackedSetFeature.PACKED_SET_FEATURE__SPE_IV,
            ]
        )
        evs = packed_set[ev_indices] / 255
        ivs = packed_set[iv_indices] / 31

        embedding = self.packed_set_merge(
            boolean_code,
            self._embed_species(species_token),
            self._embed_ability(ability_token),
            self._embed_item(item_token),
            move_encodings.sum(axis=0),
            jnp.concatenate((evs, ivs), axis=-1),
        )

        return embedding

    def _forward_value_head(self, my_embedding: jax.Array, opp_embedding: jax.Array):
        hidden = self.value_merge(my_embedding, opp_embedding)
        return self.value_head(hidden)

    def _forward_metagame_head(
        self, my_embedding: jax.Array, metagame_token: jax.Array
    ):
        logits = self.metagame_head(my_embedding)
        log_probs = legal_log_policy(logits, jnp.ones_like(logits, dtype=jnp.bool))
        return jnp.take(log_probs, metagame_token, axis=-1)

    def _encode_team(
        self,
        species_tokens: jax.Array,
        packed_set_tokens: jax.Array,
        use_casual_mask: bool,
    ):
        if use_casual_mask:
            seq_len = species_tokens.shape[0]
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))
            causal_mask = jnp.expand_dims(causal_mask, axis=0)
        else:
            causal_mask = create_attention_mask(
                jnp.ones_like(species_tokens, dtype=jnp.bool)
            )

        valid_packed_sets = jnp.take(
            SET_TOKENS[self.cfg.generation]["ou_all_formats"], species_tokens, axis=0
        )
        packed_sets = jax.vmap(lambda a, idx: a[idx])(
            valid_packed_sets, packed_set_tokens
        )
        set_embeddings = jax.vmap(self._embed_packed_set)(packed_sets)
        set_embeddings = jnp.where(
            species_tokens[:, None] == SpeciesEnum.SPECIES_ENUM___UNK,
            self.unk_embedding.astype(self.cfg.dtype),
            set_embeddings,
        )

        positions = jnp.arange(set_embeddings.shape[0], dtype=jnp.int32)

        return self.encoder(
            set_embeddings,
            causal_mask,
            qkv_positions=positions,
        )

    def _forward(
        self,
        current_embedding: jax.Array,
        actor_env: BuilderEnvOutput,
        actor_output: BuilderActorOutput,
        species_keys: jax.Array,
        opp_embedding: jax.Array,
        head_params: HeadParams,
    ) -> BuilderActorOutput:

        value = self._forward_value_head(current_embedding, opp_embedding)

        metagame_log_prob = self._forward_metagame_head(
            current_embedding, actor_env.metagame_token
        )

        metagame_embedding = self.metagame_embeddings(actor_env.metagame_token)
        gamma, beta = jnp.split(metagame_embedding, 2, axis=-1)
        species_query = gamma * current_embedding + beta

        species_head = self.species_head(
            species_query,
            species_keys,
            actor_output.species_head,
            actor_env.species_mask,
            head_params=head_params,
        )

        packed_set_mask = jnp.take(
            SET_MASK[self.cfg.generation]["ou_all_formats"],
            species_head.action_index,
            axis=0,
        )

        packed_sets = SET_TOKENS[self.cfg.generation]["ou_all_formats"][
            species_head.action_index
        ]
        packed_set_keys = jax.vmap(self._embed_packed_set)(packed_sets)

        species_embedding = jnp.take(species_keys, species_head.action_index, axis=0)
        packed_set_query = self.packed_set_query_merge(species_query, species_embedding)
        packed_set_query = gamma * packed_set_query + beta

        packed_set_head = self.packed_set_head(
            packed_set_query,
            packed_set_keys,
            actor_output.packed_set_head,
            packed_set_mask,
            head_params=head_params,
        )

        return BuilderActorOutput(
            species_head=species_head,
            packed_set_head=packed_set_head,
            metagame_log_prob=metagame_log_prob,
            v=value,
        )

    def __call__(
        self,
        actor_input: BuilderActorInput,
        actor_output: BuilderActorOutput,
        head_params: HeadParams,
    ) -> BuilderActorOutput:
        species_keys = jax.vmap(self._embed_species)(np.arange(NUM_SPECIES))

        my_embeddings = self._encode_team(
            actor_input.history.species_tokens,
            actor_input.history.packed_set_tokens,
            use_casual_mask=True,
        )

        train = self.cfg.get("train", False)
        if train:
            opp_embeddings = self._encode_team(
                actor_input.hidden.species_tokens,
                actor_input.hidden.packed_set_tokens,
                use_casual_mask=False,
            )
            opp_embedding = jnp.mean(opp_embeddings, axis=0)
        else:
            opp_embedding = jnp.zeros((self.cfg.entity_size,), dtype=self.cfg.dtype)

        hidden_states = jnp.take(
            my_embeddings,
            actor_input.env.ts.reshape(-1).clip(min=0, max=my_embeddings.shape[0] - 1),
            axis=0,
        )
        return jax.vmap(
            functools.partial(self._forward, head_params=head_params),
            in_axes=(0, 0, 0, None, None),
        )(hidden_states, actor_input.env, actor_output, species_keys, opp_embedding)


def get_builder_model(config: ConfigDict = None) -> nn.Module:
    if config is None:
        config = get_builder_model_config()
    return Porygon2BuilderModel(config)


def main(debug: bool = False, generation: int = 9):
    learner_config = get_learner_config()

    actor_model_config = get_builder_model_config(
        generation,
        train=False,
        metagame_vocab_size=learner_config.metagame_vocab_size,
    )
    actor_network = get_builder_model(actor_model_config)

    learner_model_config = get_builder_model_config(
        generation,
        train=True,
        metagame_vocab_size=learner_config.metagame_vocab_size,
    )
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
        builder_params = learner_network.init(key, ex_actor_input, ex_actor_output)

    pprint(get_num_params(builder_params))

    agent = Agent(builder_apply_fn=actor_network.apply)

    builder_env = TeamBuilderEnvironment(
        generation=generation,
        smogon_format="ou_all_formats",
        metagame_vocab_size=learner_config.metagame_vocab_size,
    )

    with open(f"data/data/gen{generation}/{builder_env._smogon_format}.json", "r") as f:
        packed_sets = json.load(f)

    species_reward_bounds = (0, 0)
    teammate_reward_bounds = (0, 0)

    while True:

        rng_key, key = jax.random.split(key)
        builder_subkeys = jax.random.split(
            rng_key, builder_env._max_trajectory_length + 1
        )

        build_traj = []

        with jax.disable_jit(debug):
            builder_actor_input = builder_env.reset(3)
        for builder_step_index in range(builder_subkeys.shape[0]):
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

        builder_trajectory: BuilderTransition = jax.tree.map(
            lambda *xs: np.array(jnp.stack(xs)), *build_traj
        )

        species_reward_sum = (
            builder_trajectory.env_output.cum_species_reward[1:]
            - builder_trajectory.env_output.cum_species_reward[:-1]
        ).sum()
        new_species_reward_bounds = (
            min(species_reward_sum.item(), species_reward_bounds[0]),
            max(species_reward_sum.item(), species_reward_bounds[1]),
        )
        if new_species_reward_bounds != species_reward_bounds:
            species_reward_bounds = new_species_reward_bounds

        teammate_reward_sum = (
            builder_trajectory.env_output.cum_teammate_reward[1:]
            - builder_trajectory.env_output.cum_teammate_reward[:-1]
        ).sum()
        new_teammate_reward_bounds = (
            min(teammate_reward_sum.item(), teammate_reward_bounds[0]),
            max(teammate_reward_sum.item(), teammate_reward_bounds[1]),
        )
        if new_teammate_reward_bounds != teammate_reward_bounds:
            teammate_reward_bounds = new_teammate_reward_bounds

        # learner_output = learner_network.apply(
        #     builder_params,
        #     BuilderActorInput(env=builder_trajectory.env_output),
        #     builder_trajectory.agent_output.actor_output,
        # )

        # print(
        #     "value:", builder_trajectory.agent_output.actor_output.v.astype(jnp.float32)
        # )

        assert np.all(
            builder_actor_input.history.species_tokens > SpeciesEnum.SPECIES_ENUM___UNK
        ).item()

        for st, pst in zip(
            builder_actor_input.history.species_tokens.reshape(-1).tolist(),
            builder_actor_input.history.packed_set_tokens.reshape(-1).tolist(),
        ):
            species = ITOS["species"][st]
            packed_set = packed_sets[species][pst]

            print(species, packed_set)

        print()


if __name__ == "__main__":
    main()
