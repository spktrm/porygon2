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
    PointerLogits,
    RMSNorm,
    SumEmbeddings,
    TransformerDecoder,
    TransformerEncoder,
    create_attention_mask,
    one_hot_concat_jax,
)
from rl.model.utils import BIAS_VALUE, get_most_recent_file, get_num_params
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

        embed_kwargs = dense_kwargs = dict(features=entity_size, dtype=self.cfg.dtype)

        self.species_linear = nn.Dense(name="species_linear", **dense_kwargs)
        self.items_linear = nn.Dense(name="items_linear", **dense_kwargs)
        self.abilities_linear = nn.Dense(name="abilities_linear", **dense_kwargs)
        self.moves_linear = nn.Dense(name="moves_linear", **dense_kwargs)

        self.packed_set_sum = SumEmbeddings(entity_size, dtype=dtype)
        self.packed_set_ff = FeedForwardResidual(entity_size, dtype=dtype)
        self.packed_set_ln = RMSNorm(dtype=dtype)

        transformer_config = self.cfg.transformer.to_dict()

        self.team_encoder = TransformerEncoder(**transformer_config)
        self.team_decoder = TransformerDecoder(**transformer_config)

        self.species_head = MLP((entity_size, NUM_SPECIES), dtype=dtype)
        self.packed_set_head = PointerLogits()
        self.value_head = MLP((entity_size, 1), dtype=dtype)

    def _forward_head(
        self,
        species_token: jax.Array,
        embedding: jax.Array,
        species_mask: jax.Array,
        packed_set_mask: jax.Array,
    ):
        """
        Samples a token from the embeddings using the policy head and returns the token, log probability, and entropy.
        """
        species_logits = jnp.where(
            species_mask, self.species_head(embedding), BIAS_VALUE
        )

        packed_sets = SET_TOKENS[self.cfg.generation]["ou"]

        _encode_sets = jax.vmap(self._encode_packed_set, in_axes=(None, 0))
        packed_set_embeddings = _encode_sets(
            species_token, np.arange(packed_sets.shape[1])
        )
        packed_set_logits = self.packed_set_head(embedding[None], packed_set_embeddings)

        packed_set_logits = jnp.where(
            packed_set_mask, packed_set_logits.squeeze(0), BIAS_VALUE
        )

        return species_logits, packed_set_logits

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

    def _encode_packed_set(self, species_token: jax.Array, packed_set_token: jax.Array):
        """
        Encodes the packed set tokens into embeddings.
        """
        packed_set = SET_TOKENS[self.cfg.generation]["ou"][
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

        move_tokens = jnp.where(
            packed_set_token == -1, MovesEnum.MOVES_ENUM___UNK, packed_set[move_indices]
        )
        move_encodings = jax.vmap(self._embed_move)(move_tokens)

        ability_token = jnp.where(
            packed_set_token == 1,
            AbilitiesEnum.ABILITIES_ENUM___UNK,
            packed_set[PackedSetFeature.PACKED_SET_FEATURE__ABILITY],
        )
        item_token = jnp.where(
            packed_set_token == 1,
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
        evs = jnp.where(packed_set_token == -1, -1, packed_set[ev_indices] / 255)
        ivs = jnp.where(packed_set_token == -1, -1, packed_set[iv_indices] / 31)

        embedding = self.packed_set_sum(
            jnp.where(packed_set_token == -1, 0, boolean_code),
            self._embed_species(species_token),
            self._embed_ability(ability_token),
            self._embed_item(item_token),
            move_encodings.sum(axis=0),
            jnp.concatenate((evs, ivs), axis=-1),
        )
        embedding = self.packed_set_ff(embedding)

        return self.packed_set_ln(embedding)

    def _forward(self, env_input: BuilderEnvOutput) -> BuilderAgentOutput:
        """Autoregressively generates a team and returns (tokens, log_pi)."""
        num_tokens = env_input.species_tokens.shape[-1]
        position_indices = jnp.arange(num_tokens, dtype=jnp.int32)
        attn_mask = jnp.ones_like(position_indices, dtype=jnp.bool)
        species = env_input.species_tokens

        set_embeddings = jax.vmap(self._encode_packed_set)(
            species, env_input.packed_set_tokens
        )

        pred_embeddings = self.team_encoder(
            set_embeddings, create_attention_mask(attn_mask), position_indices
        )
        pooled = self.team_decoder(
            pred_embeddings.mean(axis=0, keepdims=True),
            pred_embeddings,
            create_attention_mask(attn_mask.any(keepdims=True), attn_mask),
        )

        pos = env_input.pos % 6
        species_logits, packed_set_logits = self._forward_head(
            species[pos],
            jnp.take(pred_embeddings, pos, axis=0) + pooled.squeeze(0),
            env_input.species_mask,
            env_input.packed_set_mask,
        )

        value = jnp.tanh(self.value_head(pooled)).squeeze()

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

    latest_ckpt = get_most_recent_file(f"./ckpts/gen{generation}")
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
        # builder_sampling_config=SamplingConfig(temp=1, min_p=0.05),
    )

    builder_env = TeamBuilderEnvironment(generation=generation)

    with open(f"data/data/gen{generation}/validated_packed_ou_sets.json", "r") as f:
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

        print("value:", builder_trajectory.agent_output.actor_output.v)
        for st, pst in zip(
            builder_env_output.species_tokens, builder_env_output.packed_set_tokens
        ):
            print(packed_sets[all_species[st]][pst])
        print()


if __name__ == "__main__":
    main()
