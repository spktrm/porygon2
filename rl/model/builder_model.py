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
    DEFAULT_SMOGON_FORMAT,
    NUM_SPECIES,
    ONEHOT_ENCODERS,
    PACKED_SET_MAX_VALUES,
    SET_TOKENS,
)
from rl.environment.env import TeamBuilderEnvironment
from rl.environment.interfaces import (
    BuilderActorOutput,
    BuilderAgentOutput,
    BuilderEnvOutput,
    BuilderTransition,
    SamplingConfig,
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
from rl.model.modules import (
    MLP,
    FeedForwardResidual,
    RMSNorm,
    SumEmbeddings,
    TransformerDecoder,
    TransformerEncoder,
    activation_fn,
    create_attention_mask,
    one_hot_concat_jax,
)
from rl.model.utils import (
    LARGE_NEGATIVE_BIAS,
    get_num_params,
    legal_log_policy,
    legal_policy,
)
from rl.utils import init_jax_jit_cache


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

        self.proj_species_linear = nn.Dense(name="proj_species", **dense_kwargs)
        self.proj_packed_set_linear = nn.Dense(name="proj_packed_set", **dense_kwargs)

        self.packed_set_sum = SumEmbeddings(entity_size, dtype=dtype)
        self.packed_set_ff = FeedForwardResidual(entity_size, dtype=dtype)

        self.proj_species_ln = RMSNorm(dtype=dtype)
        self.proj_packed_set_ln = RMSNorm(dtype=dtype)
        self.final_species_ln = RMSNorm(dtype=dtype)
        self.final_packed_set_ln = RMSNorm(dtype=dtype)

        transformer_config = self.cfg.transformer.to_dict()

        self.species_encoder = TransformerEncoder(**transformer_config)
        self.species_decoder = TransformerDecoder(**transformer_config)

        self.team_encoder = TransformerEncoder(**transformer_config)
        self.team_decoder = TransformerDecoder(**transformer_config)

        self.value_head = MLP((entity_size, 1), dtype=dtype)

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

    def _embed_packed_set(self, species_token: jax.Array, packed_set_token: jax.Array):
        """
        Encodes the packed set tokens into embeddings.
        """
        packed_set = SET_TOKENS[self.cfg.generation][DEFAULT_SMOGON_FORMAT][
            species_token, packed_set_token
        ]

        move_indices = np.array(
            [
                PackedSetFeature.PACKED_SET_FEATURE__MOVE1,
                PackedSetFeature.PACKED_SET_FEATURE__MOVE2,
                PackedSetFeature.PACKED_SET_FEATURE__MOVE3,
                PackedSetFeature.PACKED_SET_FEATURE__MOVE4,
            ]
        )

        is_mask = packed_set_token == -1
        move_tokens = jnp.where(
            is_mask[..., None], MovesEnum.MOVES_ENUM___UNK, packed_set[move_indices]
        )
        move_encodings = jax.vmap(self._embed_move)(move_tokens)

        ability_token = jnp.where(
            is_mask,
            AbilitiesEnum.ABILITIES_ENUM___UNK,
            packed_set[PackedSetFeature.PACKED_SET_FEATURE__ABILITY],
        )
        item_token = jnp.where(
            is_mask,
            ItemsEnum.ITEMS_ENUM___UNK,
            packed_set[PackedSetFeature.PACKED_SET_FEATURE__ITEM],
        )

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
            ]
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
        evs = jnp.where(is_mask[..., None], -1, packed_set[ev_indices] / 255)
        ivs = jnp.where(is_mask[..., None], -1, packed_set[iv_indices] / 31)

        embedding = self.packed_set_sum(
            jnp.where(is_mask[..., None], 0, boolean_code),
            self._embed_species(species_token),
            self._embed_ability(ability_token),
            self._embed_item(item_token),
            move_encodings.sum(axis=0),
            jnp.concatenate((evs, ivs), axis=-1),
        )
        return self.packed_set_ff(embedding)

    def _decode_species(self, species_token: jax.Array):
        embedding = self._embed_species(species_token)
        embedding = self.proj_species_ln(embedding)
        embedding = activation_fn(embedding)
        return self.proj_species_linear(embedding)

    def _decode_packed_set(self, species_token: jax.Array, packed_set_token: jax.Array):
        embedding = self._embed_packed_set(species_token, packed_set_token)
        embedding = self.proj_packed_set_ln(embedding)
        embedding = activation_fn(embedding)
        return self.proj_packed_set_linear(embedding)

    def _forward_species_head(self, query: jax.Array, species_mask: jax.Array):
        keys = jax.vmap(self._decode_species)(np.arange(NUM_SPECIES))
        query = self.final_species_ln(query)
        species_logits = jnp.einsum("i,ki->k", query, keys) / (
            np.sqrt(self.cfg.entity_size).astype(self.cfg.dtype)
        )
        return jnp.where(species_mask, species_logits, LARGE_NEGATIVE_BIAS)

    def _forward_packed_set_head(
        self, species_token: jax.Array, query: jax.Array, packed_set_mask: jax.Array
    ):
        packed_sets = SET_TOKENS[self.cfg.generation][DEFAULT_SMOGON_FORMAT]
        keys = jax.vmap(self._decode_packed_set, in_axes=(None, 0))(
            species_token, np.arange(packed_sets.shape[1])
        )
        query = self.final_packed_set_ln(query)
        packed_set_logits = jnp.einsum("i,ki->k", query, keys) / (
            np.sqrt(self.cfg.entity_size).astype(self.cfg.dtype)
        )
        return jnp.where(packed_set_mask, packed_set_logits, LARGE_NEGATIVE_BIAS)

    def _forward(self, env_input: BuilderEnvOutput) -> BuilderAgentOutput:
        """Autoregressively generates a team and returns (tokens, log_pi)."""
        num_tokens = env_input.species_tokens.shape[-1]
        position_indices = jnp.arange(num_tokens, dtype=jnp.int32)

        species = env_input.species_tokens
        packed_sets = env_input.packed_set_tokens

        species_attn_mask = packed_set_attn_mask = jnp.ones_like(
            position_indices, dtype=jnp.bool
        )

        species_embeddings = jax.vmap(self._embed_species)(species)
        set_embeddings = jax.vmap(self._embed_packed_set)(species, packed_sets)

        pred_species_embeddings = self.species_encoder(
            self.proj_species_ln(species_embeddings),
            create_attention_mask(species_attn_mask),
            position_indices,
        )
        pred_packed_set_embeddings = self.team_encoder(
            self.proj_packed_set_ln(set_embeddings),
            create_attention_mask(packed_set_attn_mask),
            position_indices,
        )

        pred_species_embeddings = self.species_decoder(
            pred_species_embeddings,
            pred_packed_set_embeddings,
            create_attention_mask(species_attn_mask, packed_set_attn_mask),
        )
        pred_packed_set_embeddings = self.team_decoder(
            pred_packed_set_embeddings,
            pred_species_embeddings,
            create_attention_mask(packed_set_attn_mask, species_attn_mask),
        )

        pos = env_input.pos % 6

        species_logits = self._forward_species_head(
            jnp.take(pred_species_embeddings, pos, axis=0),
            env_input.species_mask,
        )
        packed_set_logits = self._forward_packed_set_head(
            species[pos],
            jnp.take(pred_packed_set_embeddings, pos, axis=0),
            env_input.packed_set_mask,
        )

        pooled_species = pred_species_embeddings.mean(
            where=species_attn_mask[..., None], axis=0
        )
        pooled_packed_sets = pred_packed_set_embeddings.mean(
            where=packed_set_attn_mask[..., None], axis=0
        )
        value = jnp.tanh(self.value_head(pooled_species + pooled_packed_sets)).squeeze()

        return BuilderActorOutput(
            species_logits=species_logits, packed_set_logits=packed_set_logits, v=value
        )

    def __call__(self, input: BuilderEnvOutput) -> BuilderAgentOutput:
        return jax.vmap(self._forward)(input)


def get_builder_model(config: ConfigDict = None) -> nn.Module:
    if config is None:
        config = get_builder_model_config()
    return Porygon2BuilderModel(config)


def main(generation: int = 9):
    init_jax_jit_cache()

    model_config = get_builder_model_config(generation)
    network = get_builder_model(model_config)

    ex = jax.device_put(
        jax.tree.map(lambda x: x[:, 0], get_ex_builder_step(generation))
    )
    key = jax.random.key(42)

    latest_ckpt = None  # get_most_recent_file(f"./ckpts/gen{generation}")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        params = step["builder_state"]["params"]
    else:
        params = network.init(key, ex)

    pprint(get_num_params(params))

    agent = Agent(
        builder_apply_fn=jax.vmap(network.apply, in_axes=(None, 1), out_axes=1),
        builder_sampling_config=SamplingConfig(temp=1, min_p=0.05),
    )

    builder_env = TeamBuilderEnvironment(generation=generation)

    with open(
        f"data/data/gen{generation}/validated_packed_{builder_env.smogon_format}_sets.json",
        "r",
    ) as f:
        packed_sets = json.load(f)
    all_species = list(packed_sets)

    while True:

        rng_key, key = jax.random.split(key)
        builder_subkeys = jax.random.split(rng_key, builder_env.max_ts + 1)

        build_traj = []

        builder_env_output = builder_env.reset()
        for i in range(builder_subkeys.shape[0]):
            builder_agent_output = agent.step_builder(
                builder_subkeys[i], params, builder_env_output
            )
            builder_transition = BuilderTransition(
                env_output=builder_env_output, agent_output=builder_agent_output
            )
            build_traj.append(builder_transition)
            if builder_env_output.done.item():
                break
            builder_env_output = builder_env.step(builder_agent_output)

        builder_trajectory: BuilderTransition = jax.tree.map(
            lambda *xs: np.stack(xs), *build_traj
        )

        def _get_values_along_axis(
            logits: jax.Array, actions: jax.Array, axis: int = -1
        ):
            return jnp.take_along_axis(
                legal_policy(logits=logits),
                jnp.expand_dims(actions, axis=axis),
                axis=axis,
            ).reshape(-1)

        def _get_entropy(logits: jax.Array, axis: int = -1):
            pi = legal_policy(logits=logits)
            log_pi = legal_log_policy(logits=logits)
            return jnp.sum(pi * log_pi, axis=axis).reshape(-1)

        print(
            "value:", builder_trajectory.agent_output.actor_output.v.astype(jnp.float32)
        )
        print(
            "species policy:",
            _get_values_along_axis(
                logits=builder_trajectory.agent_output.actor_output.species_logits,
                actions=builder_trajectory.agent_output.species,
            )[:6].astype(jnp.float32),
        )
        print(
            "packed set policy:",
            _get_values_along_axis(
                logits=builder_trajectory.agent_output.actor_output.packed_set_logits,
                actions=builder_trajectory.agent_output.packed_set,
            )[6:-1].astype(jnp.float32),
        )

        species_entropy = _get_entropy(
            logits=builder_trajectory.agent_output.actor_output.species_logits
        )
        max_species_entropy = jnp.log(
            builder_trajectory.env_output.species_mask.sum(-1)
        )

        packed_set_entropy = _get_entropy(
            logits=builder_trajectory.agent_output.actor_output.packed_set_logits
        )
        max_packed_set_entropy = jnp.log(
            builder_trajectory.env_output.packed_set_mask.sum(-1)
        )

        print(
            "species entropy:",
            (species_entropy / max_species_entropy)[:6].astype(jnp.float32),
        )
        print(
            "packed set entropy:",
            (packed_set_entropy / max_packed_set_entropy)[6:-1].astype(jnp.float32),
        )

        for st, pst in zip(
            builder_env_output.species_tokens.reshape(-1).tolist(),
            builder_env_output.packed_set_tokens.reshape(-1).tolist(),
        ):
            print(packed_sets[all_species[st]][pst])

        print()


if __name__ == "__main__":
    main()
