from dotenv import load_dotenv

load_dotenv()

import json
import pickle
from functools import partial
from pprint import pprint

import chex
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
    HeadOutput,
)
from rl.environment.protos.enums_pb2 import (
    AbilitiesEnum,
    ItemsEnum,
    MovesEnum,
    SpeciesEnum,
)
from rl.environment.protos.features_pb2 import PackedSetFeature
from rl.environment.utils import get_ex_builder_step
from rl.model.config import get_builder_model_config
from rl.model.heads import (
    PolicyLogitHead,
    PolicyQKHead,
    ValueLogitHead,
    sample_categorical,
)
from rl.model.modules import (
    SumEmbeddings,
    TransformerDecoder,
    TransformerEncoder,
    create_attention_mask,
    one_hot_concat_jax,
)
from rl.model.utils import (
    get_most_recent_file,
    get_num_params,
    legal_log_policy,
    legal_policy,
)


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
        self.continue_query_merge = SumEmbeddings(entity_size, dtype=dtype)
        self.selection_query_merge = SumEmbeddings(entity_size, dtype=dtype)
        self.packed_set_query_merge = SumEmbeddings(entity_size, dtype=dtype)
        self.value_merge = SumEmbeddings(entity_size, dtype=dtype)

        transformer_config = self.cfg.transformer.to_dict()

        self.encoder = TransformerEncoder(**transformer_config)
        self.decoder = TransformerDecoder(**transformer_config)

        self.continue_head = PolicyLogitHead(self.cfg.continue_head)
        self.selection_head = PolicyQKHead(self.cfg.selection_head)
        self.species_head = PolicyQKHead(self.cfg.species_head)
        self.packed_set_head = PolicyQKHead(self.cfg.packed_set_head)
        self.value_head = ValueLogitHead(self.cfg.value_head)

        self.z_logits = self.param(
            "z_logits", nn.initializers.zeros_init(), (self.cfg.num_metagame_slots,)
        )
        self.metagame_pred_head = PolicyLogitHead(self.cfg.metagame_head)

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

    def _forward_value_head(
        self,
        my_embedding: jax.Array,
        opp_embedding: jax.Array,
        selected_metagame_oh: jax.Array,
    ):
        concatenated_embedding = jnp.concatenate((my_embedding, opp_embedding), axis=-1)
        merged_embedding = self.value_merge(
            concatenated_embedding, selected_metagame_oh
        )
        return self.value_head(merged_embedding)

    def _encode_team(self, species_tokens: jax.Array, packed_set_tokens: jax.Array):
        packed_set_attn_mask = jnp.ones_like(packed_set_tokens, dtype=jnp.bool)

        valid_packed_sets = jnp.take(
            SET_TOKENS[self.cfg.generation]["ou_all_formats"], species_tokens, axis=0
        )
        packed_sets = jax.vmap(lambda a, idx: a[idx])(
            valid_packed_sets, packed_set_tokens
        )
        set_embeddings = jax.vmap(self._embed_packed_set)(packed_sets)

        positions = (jnp.arange(set_embeddings.shape[0], dtype=jnp.int32) < 1).astype(
            jnp.int32
        )
        contextual_embeddings = self.encoder(
            set_embeddings,
            create_attention_mask(packed_set_attn_mask),
            qkv_positions=positions,
        )

        pool_queries = jnp.concatenate(
            (
                contextual_embeddings.mean(axis=0, keepdims=True),
                contextual_embeddings.max(axis=0, keepdims=True),
                contextual_embeddings.min(axis=0, keepdims=True),
            ),
            axis=0,
        )
        contextual_embedding = self.decoder(
            pool_queries,
            contextual_embeddings,
            create_attention_mask(
                packed_set_attn_mask.any(axis=0, keepdims=True), packed_set_attn_mask
            ),
            kv_positions=positions,
        ).reshape(-1)

        return contextual_embeddings, contextual_embedding

    def _get_metagame_head(self, valid_mask: jax.Array, head: HeadOutput) -> HeadOutput:
        temp = self.cfg.get("temp", 1.0)
        logits = self.z_logits / temp

        log_policy = legal_log_policy(logits, valid_mask)

        entropy = ()
        train = self.cfg.get("train", False)
        if train:
            action_index = head.action_index
            policy = legal_policy(logits, valid_mask)
            entropy = -jnp.sum(policy * log_policy, axis=-1)
        else:
            action_index = sample_categorical(
                logits,
                log_policy,
                valid_mask,
                self.make_rng("sampling"),
                self.cfg.get("min_p", 0.0),
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1)
        return HeadOutput(action_index=action_index, log_prob=log_prob, entropy=entropy)

    def _forward(
        self,
        actor_env: BuilderEnvOutput,
        actor_output: BuilderActorOutput,
        species_keys: jax.Array,
        opp_embedding: jax.Array,
    ) -> BuilderActorOutput:
        my_embeddings, my_embedding = self._encode_team(
            actor_env.species_tokens, actor_env.packed_set_tokens
        )

        metagame_sele_head = self._get_metagame_head(
            actor_env.metagame_mask,
            actor_output.metagame_sele_head,
        )

        selected_metagame_oh = jax.nn.one_hot(
            metagame_sele_head.action_index,
            self.cfg.num_metagame_slots,
            dtype=self.cfg.dtype,
        )
        value = self._forward_value_head(
            my_embedding, opp_embedding, selected_metagame_oh
        )

        continue_query = self.continue_query_merge(my_embedding, selected_metagame_oh)
        continue_head = self.continue_head(
            continue_query,
            actor_env.continue_mask,
            actor_output.continue_head,
        )

        selection_head = self.selection_head(
            continue_query,
            my_embeddings,
            jnp.ones_like(my_embeddings[..., 0], dtype=jnp.bool),
            actor_output.selection_head,
        )

        selected_embedding = jnp.take(
            my_embeddings, selection_head.action_index, axis=0
        )
        selection_query = self.selection_query_merge(
            my_embedding, selected_embedding, selected_metagame_oh
        )
        species_head = self.species_head(
            selection_query,
            species_keys,
            actor_env.species_mask,
            actor_output.species_head,
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
        packed_set_query = self.packed_set_query_merge(
            selection_query, species_embedding
        )
        packed_set_head = self.packed_set_head(
            packed_set_query,
            packed_set_keys,
            packed_set_mask,
            actor_output.packed_set_head,
        )

        metagame_pred_head = self.metagame_pred_head(
            my_embedding,
            jnp.ones_like(actor_env.metagame_mask),
            actor_output.metagame_sele_head,
        )

        return BuilderActorOutput(
            metagame_sele_head=metagame_sele_head,
            metagame_pred_head=metagame_pred_head,
            continue_head=continue_head,
            selection_head=selection_head,
            species_head=species_head,
            packed_set_head=packed_set_head,
            v=value,
        )

    def __call__(
        self,
        actor_input: BuilderActorInput,
        actor_output: BuilderActorOutput = BuilderActorOutput(),
    ) -> BuilderActorOutput:
        species_keys = jax.vmap(self._embed_species)(np.arange(NUM_SPECIES))

        train = self.cfg.get("train", False)
        if train:
            _, opp_embedding = self._encode_team(
                actor_input.hidden.species_tokens, actor_input.hidden.packed_set_tokens
            )
        else:
            opp_embedding = jnp.zeros((3 * self.cfg.entity_size,), dtype=self.cfg.dtype)

        return jax.vmap(self._forward, in_axes=(0, 0, None, None))(
            actor_input.env, actor_output, species_keys, opp_embedding
        )


def get_builder_model(config: ConfigDict = None) -> nn.Module:
    if config is None:
        config = get_builder_model_config()
    return Porygon2BuilderModel(config)


def main(debug: bool = False, generation: int = 1):
    actor_network = get_builder_model(
        get_builder_model_config(generation, train=False)  # , temp=0.8, min_p=0.05)
    )
    learner_config = get_builder_model_config(generation, train=True)
    learner_network = get_builder_model(learner_config)

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
        initial_seed=42,
    )

    with open(f"data/data/gen{generation}/{builder_env.smogon_format}.json", "r") as f:
        packed_sets = json.load(f)

    metagame_counts = {i: 0 for i in range(32)}

    while True:

        rng_key, key = jax.random.split(key)
        builder_subkeys = jax.random.split(
            rng_key, builder_env.max_trajectory_length + 1
        )

        build_traj = []

        with jax.disable_jit(debug):
            builder_actor_input = builder_env.reset()
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

        metagame_counts[builder_trajectory.env_output.metagame_token[-1].item()] += 1

        # learner_output = learner_network.apply(
        #     builder_params,
        #     BuilderActorInput(env=builder_trajectory.env_output),
        #     builder_trajectory.agent_output.actor_output,
        # )

        print(
            "value:", builder_trajectory.agent_output.actor_output.v.astype(jnp.float32)
        )

        assert np.all(
            builder_actor_input.env.species_tokens > SpeciesEnum.SPECIES_ENUM___UNK
        ).item()

        for st, pst in zip(
            builder_actor_input.env.species_tokens.reshape(-1).tolist(),
            builder_actor_input.env.packed_set_tokens.reshape(-1).tolist(),
        ):
            species = ITOS["species"][st]
            packed_set = packed_sets[species][pst]

            print(species, packed_set)

        print()


if __name__ == "__main__":
    main()
