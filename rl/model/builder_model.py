import pickle
import time
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
    ONEHOT_ENCODERS,
    PACKED_SET_MAX_VALUES,
    PACKED_SETS,
    TokenColumns,
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
from rl.environment.utils import get_ex_builder_step
from rl.model.config import get_builder_model_config
from rl.model.modules import (
    MLP,
    FeedForwardResidual,
    RMSNorm,
    SumEmbeddings,
    TransformerDecoder,
    TransformerEncoder,
    create_attention_mask,
    one_hot_concat_jax,
)
from rl.model.utils import (
    BIAS_VALUE,
    get_most_recent_file,
    get_num_params,
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

        embed_kwargs = dense_kwargs = dict(features=entity_size, dtype=self.cfg.dtype)

        sets_data = PACKED_SETS[f"gen{self.cfg.generation}"]
        num_sets = len(sets_data["sets"])
        self.num_sets = num_sets
        self.output_sets = ONEHOT_ENCODERS[self.cfg.generation]["sets"](
            np.arange(self.num_sets)
        )
        embedding_init_fn = nn.initializers.normal()

        self.species_linear = nn.Dense(name="species_linear", **dense_kwargs)
        self.items_linear = nn.Dense(name="items_linear", **dense_kwargs)
        self.abilities_linear = nn.Dense(name="abilities_linear", **dense_kwargs)
        self.moves_linear = nn.Dense(name="moves_linear", **dense_kwargs)
        self.set_linear = nn.Dense(name="set_linear", **dense_kwargs)

        self.packed_set_sum = SumEmbeddings(entity_size, dtype=dtype)
        self.packed_set_ff = FeedForwardResidual(entity_size, dtype=dtype)
        self.packed_set_ln = RMSNorm(dtype=dtype)

        transformer_config = self.cfg.transformer.to_dict()

        self.decoder = TransformerDecoder(**transformer_config)
        self.encoder = TransformerEncoder(**transformer_config)

        self.policy_head = MLP(entity_size, dtype=dtype)
        self.key_head = nn.Dense(
            name="key_head",
            kernel_init=nn.initializers.normal(),
            use_bias=False,
            **dense_kwargs,
        )
        self.policy_bias = self.param(
            "policy_bias", nn.initializers.zeros_init(), (num_sets,), dtype=dtype
        )

        self.value_head = MLP((entity_size, 1), dtype=dtype)

        self.mask_embedding = self.param(
            "mask_embedding", embedding_init_fn, (1, entity_size), dtype=dtype
        )

    def _forward_head(self, embedding: jax.Array, sample_mask: jax.Array):
        """
        Samples a token from the embeddings using the policy head and returns the token, log probability, and entropy.
        """
        query = self.policy_head(embedding)

        keys = jax.vmap(self._encode_packed_set)(self.output_sets)
        keys = self.key_head(keys)

        logits = (query @ keys.T).reshape(-1) + self.policy_bias

        return jnp.where(sample_mask, logits, BIAS_VALUE)

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

    def _encode_packed_set(self, packed_set: jax.Array):
        """
        Encodes the packed set tokens into embeddings.
        """
        move_indices = np.array(
            [
                TokenColumns.MOVE1.value,
                TokenColumns.MOVE2.value,
                TokenColumns.MOVE3.value,
                TokenColumns.MOVE4.value,
            ]
        )

        move_tokens = packed_set[move_indices]
        move_encodings = jax.vmap(self._embed_move)(move_tokens)

        species_token = packed_set[TokenColumns.SPECIES.value]
        ability_token = packed_set[TokenColumns.ABILITY.value]
        item_token = packed_set[TokenColumns.ITEM.value]

        boolean_code = one_hot_concat_jax(
            [
                _encode_one_hot_set(packed_set, TokenColumns.NATURE.value),
                _encode_one_hot_set(packed_set, TokenColumns.GENDER.value),
                _encode_one_hot_set(packed_set, TokenColumns.HIDDENPOWERTYPE.value),
                _encode_one_hot_set(packed_set, TokenColumns.TERATYPE.value),
            ]
        )

        ev_and_iv_indicies = np.array(
            [
                TokenColumns.HP_EV.value,
                TokenColumns.ATK_EV.value,
                TokenColumns.DEF_EV.value,
                TokenColumns.SPA_EV.value,
                TokenColumns.SPD_EV.value,
                TokenColumns.SPE_EV.value,
                TokenColumns.HP_IV.value,
                TokenColumns.ATK_IV.value,
                TokenColumns.DEF_IV.value,
                TokenColumns.SPA_IV.value,
                TokenColumns.SPD_IV.value,
                TokenColumns.SPE_IV.value,
            ]
        )
        evs_and_ivs = packed_set[ev_and_iv_indicies] / 255
        embedding = self.packed_set_sum(
            boolean_code,
            self._embed_species(species_token),
            self._embed_ability(ability_token),
            self._embed_item(item_token),
            move_encodings.sum(axis=0),
            evs_and_ivs,
        )
        embedding = self.packed_set_ff(embedding)

        return self.packed_set_ln(embedding)

    def _forward(self, input: BuilderEnvOutput) -> BuilderAgentOutput:
        """Autoregressively generates a team and returns (tokens, log_pi)."""
        masked = input.tokens == -1
        num_tokens = input.tokens.shape[-1]

        attn_mask = jnp.ones_like(input.tokens, dtype=jnp.bool)

        position_indices = jnp.arange(num_tokens, dtype=jnp.int32)

        set_tokens = ONEHOT_ENCODERS[self.cfg.generation]["sets"](input.tokens)
        set_embeddings = jax.vmap(self._encode_packed_set)(set_tokens)
        embeddings = jnp.where(masked[..., None], self.mask_embedding, set_embeddings)
        pred_embeddings = self.encoder(
            embeddings, create_attention_mask(attn_mask), position_indices
        )
        pooled = self.decoder(
            pred_embeddings.mean(axis=0, keepdims=True),
            pred_embeddings,
            create_attention_mask(attn_mask.any(keepdims=True), attn_mask),
        )

        logits = self._forward_head(pooled, input.mask)
        v = (2 / jnp.pi) * jnp.atan((jnp.pi / 2) * self.value_head(pooled)).squeeze()

        return BuilderActorOutput(logits=logits, v=v)

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
        jax.tree.map(lambda x: x[:, 0], get_ex_builder_step(generation=generation))
    )
    key = jax.random.key(42)

    latest_ckpt = get_most_recent_file("./ckpts")
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
        builder_sampling_config=SamplingConfig(temp=1, min_p=0.01),
    )
    sets_list = list(PACKED_SETS[f"gen{generation}"]["sets"])
    builder_env = TeamBuilderEnvironment(generation=generation)

    print("Num sets:", len(sets_list))
    time.sleep(2)

    while True:

        rng_key, key = jax.random.split(key)
        builder_subkeys = jax.random.split(rng_key, 7)

        build_traj = []

        builder_env_output = builder_env.reset()
        for i in range(7):
            builder_agent_output = agent.step_builder(
                builder_subkeys[i], params, builder_env_output
            )
            builder_transition = BuilderTransition(
                env_output=builder_env_output, agent_output=builder_agent_output
            )
            build_traj.append(builder_transition)
            if builder_env_output.done.item():
                break
            builder_env_output = builder_env.step(builder_agent_output.action.item())

        builder_trajectory: BuilderTransition = jax.tree.map(
            lambda *xs: np.stack(xs), *build_traj
        )

        tokens_buffer = builder_env_output.tokens
        print("tokens:", tokens_buffer)
        print("value:", builder_trajectory.agent_output.actor_output.v)
        print(
            "probs:",
            jnp.take_along_axis(
                legal_policy(
                    builder_trajectory.agent_output.actor_output.logits,
                    builder_trajectory.env_output.mask,
                ),
                builder_trajectory.agent_output.action[..., None],
                axis=-1,
            ).squeeze(-1),
        )
        print("\n".join(sets_list[t] for t in tokens_buffer.reshape(-1).tolist()))


if __name__ == "__main__":
    main()
