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
    CategoricalValueLogitHead,
    HeadParams,
    PolicyQKHead,
    RegressionValueLogitHead,
)
from rl.model.modules import SumEmbeddings, TransformerEncoder, one_hot_concat_jax
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

        self.packed_set_merge = SumEmbeddings(entity_size, dtype=dtype)
        self.packed_set_query_merge = SumEmbeddings(entity_size, dtype=dtype)

        self.sos_embedding = self.param(
            "sos_embedding",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, entity_size),
        )
        self.niche_embeddings = self.param(
            "niche_embeddings",
            nn.initializers.truncated_normal(stddev=0.02),
            (self.cfg.num_niches, entity_size),
        )
        self.unk_embedding = self.param(
            "unk_embedding",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, entity_size),
        )

        self.encoder = TransformerEncoder(**self.cfg.encoder.to_dict())

        self.species_head = PolicyQKHead(self.cfg.species_head)
        self.packed_set_head = PolicyQKHead(self.cfg.packed_set_head)

        self.value_head = RegressionValueLogitHead(self.cfg.value_head)
        self.discriminator_head = CategoricalValueLogitHead(self.cfg.discriminator_head)

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

    def _forward_value_head(self, embedding: jax.Array):
        return self.value_head(embedding)

    def _forward_discriminator_head(self, embedding: jax.Array):
        return self.discriminator_head(embedding)

    def _encode_team_with_niche(
        self,
        niche_id: jax.Array,
        species_tokens: jax.Array,
        packed_set_tokens: jax.Array,
        use_niche: bool = True,
    ):

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

        niche_embedding = jnp.take(self.niche_embeddings, niche_id, axis=0).astype(
            self.cfg.dtype
        )
        if not use_niche:
            niche_embedding = self.sos_embedding.astype(self.cfg.dtype)

        set_embeddings = jnp.concatenate((niche_embedding, set_embeddings), axis=0)

        seq_len = set_embeddings.shape[0]
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))
        causal_mask = jnp.expand_dims(causal_mask, axis=0)

        if self.cfg.generation < 4:
            positions = jnp.arange(set_embeddings.shape[0], dtype=jnp.int32).clip(
                min=0, max=1
            )
        else:
            positions = None

        return self.encoder(set_embeddings, causal_mask, qkv_positions=positions)

    def _forward(
        self,
        current_embedding_with_niche: jax.Array,
        current_embedding_without_niche: jax.Array,
        actor_env: BuilderEnvOutput,
        actor_output: BuilderActorOutput,
        species_keys: jax.Array,
        head_params: HeadParams,
    ) -> BuilderActorOutput:

        value_head = self._forward_value_head(current_embedding_with_niche)
        discriminator_head = self._forward_discriminator_head(
            current_embedding_without_niche
        )

        species_query = current_embedding_with_niche

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
            value_head=value_head,
            discriminator_head=discriminator_head,
        )

    def __call__(
        self,
        actor_input: BuilderActorInput,
        actor_output: BuilderActorOutput,
        head_params: HeadParams,
    ) -> BuilderActorOutput:
        species_keys = jax.vmap(self._embed_species)(np.arange(NUM_SPECIES))

        team_embeddings_with_niche = self._encode_team_with_niche(
            actor_input.history.niche_id,
            actor_input.history.species_tokens,
            actor_input.history.packed_set_tokens,
        )
        hidden_states_with_niche = jnp.take(
            team_embeddings_with_niche, actor_input.env.ts.reshape(-1), axis=0
        )

        team_embeddings_without_niche = self._encode_team_with_niche(
            actor_input.history.niche_id,
            actor_input.history.species_tokens,
            actor_input.history.packed_set_tokens,
            use_niche=False,
        )
        hidden_states_without_niche = jnp.take(
            team_embeddings_without_niche, actor_input.env.ts.reshape(-1), axis=0
        )

        return jax.vmap(
            functools.partial(self._forward, head_params=head_params),
            in_axes=(0, 0, 0, 0, None),
        )(
            hidden_states_with_niche,
            hidden_states_without_niche,
            actor_input.env,
            actor_output,
            species_keys,
        )


def get_builder_model(config: ConfigDict = None) -> nn.Module:
    if config is None:
        config = get_builder_model_config()
    return Porygon2BuilderModel(config)


def main(debug: bool = True, generation: int = 9):
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

    builder_env = TeamBuilderEnvironment(
        generation=generation, smogon_format="ou_all_formats"
    )

    with open(f"data/data/gen{generation}/{builder_env._smogon_format}.json", "r") as f:
        packed_sets = json.load(f)

    while True:

        niche_key, rng_key, key = jax.random.split(key, 3)
        builder_subkeys = jax.random.split(
            rng_key, builder_env._max_trajectory_length + 1
        )

        build_traj = []

        niche_key, key = jax.random.split(key)
        niche_id = jax.random.randint(
            niche_key, shape=(1,), minval=0, maxval=learner_model_config.num_niches
        )

        with jax.disable_jit(debug):
            builder_actor_input = builder_env.reset(niche_id)

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
