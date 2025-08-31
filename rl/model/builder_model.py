import math
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
    MASKS,
    NUM_ABILITIES,
    NUM_ITEMS,
    NUM_MOVES,
    NUM_NATURES,
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

        self.packed_set_sum = SumEmbeddings(entity_size, dtype=dtype)
        self.packed_set_ff = FeedForwardResidual(entity_size, dtype=dtype)
        self.packed_set_ln = RMSNorm(dtype=dtype)

        transformer_config = self.cfg.transformer.to_dict()

        self.encoder = TransformerEncoder(**transformer_config)
        self.decoder = TransformerDecoder(**transformer_config)

        self.species_selection_sum = SumEmbeddings(entity_size, dtype=dtype)
        self.recurrent_moveset_sum = SumEmbeddings(entity_size, dtype=dtype)
        self.species_moveset_sum = SumEmbeddings(entity_size, dtype=dtype)

        self.nature_mlp = MLP((entity_size, NUM_NATURES), dtype=dtype)
        self.ev_mlp = MLP((entity_size, 6), dtype=dtype)
        self.teratype_mlp = MLP((entity_size, NUM_TYPECHART), dtype=dtype)

        self.continue_head = MLP((entity_size, 2), dtype=dtype)
        self.selection_head = MLP((entity_size, 1), dtype=dtype)
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
        evs = packed_set[ev_indices] / 255
        ivs = packed_set[iv_indices] / 31

        embedding = self.packed_set_sum(
            boolean_code,
            self._embed_species(species_token),
            self._embed_ability(ability_token),
            self._embed_item(item_token),
            move_encodings.sum(axis=0),
            jnp.concatenate((evs, ivs), axis=-1),
        )
        embedding = self.packed_set_ff(embedding)
        return self.packed_set_ln(embedding)

    def _forward_continue_head(self, embedding: jax.Array, continue_head: HeadOutput):
        train = self.cfg.get("train", False)
        temp = self.cfg.get("temp", 1.0)
        logits = self.continue_head(embedding) / temp
        mask = jnp.ones_like(logits, dtype=jnp.bool)
        log_policy = legal_log_policy(logits, mask)

        entropy = ()
        if train:
            action_index = continue_head.action_index
            policy = legal_policy(logits, mask)
            entropy = -jnp.sum(policy * log_policy, axis=-1)
        else:
            min_p = self.cfg.get("min_p", 0.0)
            if 0.0 < min_p < 1.0:
                max_logp = log_policy.max(keepdims=True, axis=-1)
                keep = log_policy >= (max_logp + math.log(min_p))
                logits = jnp.where(keep, logits, LARGE_NEGATIVE_BIAS)
            action_index = jax.random.categorical(
                self.make_rng("sampling"), logits.astype(jnp.float32)
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1)
        return HeadOutput(action_index=action_index, log_prob=log_prob, entropy=entropy)

    def _forward_selection_head(self, queries: jax.Array, selection_head: HeadOutput):
        temp = self.cfg.get("temp", 1.0)
        logits = self.selection_head(queries).reshape(-1)
        logits = (logits - logits.mean(axis=-1)) / temp

        mask = jnp.ones_like(logits, dtype=jnp.bool)
        log_policy = legal_log_policy(logits, mask)

        entropy = ()
        train = self.cfg.get("train", False)
        if train:
            action_index = selection_head.action_index
            policy = legal_policy(logits, mask)
            entropy = -jnp.sum(policy * log_policy, axis=-1)
        else:
            min_p = self.cfg.get("min_p", 0.0)
            if 0.0 < min_p < 1.0:
                max_logp = log_policy.max(keepdims=True, axis=-1)
                keep = log_policy >= (max_logp + math.log(min_p))
                logits = jnp.where(keep, logits, LARGE_NEGATIVE_BIAS)
            action_index = jax.random.categorical(
                self.make_rng("sampling"), logits.astype(jnp.float32)
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1)
        return HeadOutput(action_index=action_index, log_prob=log_prob, entropy=entropy)

    def _forward_species_head(
        self,
        query: jax.Array,
        keys: jax.Array,
        species_mask: jax.Array,
        species_head: HeadOutput,
    ):
        temp = self.cfg.get("temp", 1.0)
        logits = jnp.einsum("j,kj->k", query, keys) / (
            temp * np.sqrt(self.cfg.entity_size).astype(query.dtype)
        )
        log_policy = legal_log_policy(logits, species_mask)

        entropy = ()
        train = self.cfg.get("train", False)
        if train:
            action_index = species_head.action_index
            policy = legal_policy(logits, species_mask)
            entropy = -jnp.sum(policy * log_policy, axis=-1)
        else:
            masked_logits = jnp.where(species_mask, logits, LARGE_NEGATIVE_BIAS)
            min_p = self.cfg.get("min_p", 0.0)
            if 0.0 < min_p < 1.0:
                max_logp = jnp.where(species_mask, log_policy, LARGE_NEGATIVE_BIAS).max(
                    keepdims=True, axis=-1
                )
                keep = log_policy >= (max_logp + math.log(min_p))
                masked_logits = jnp.where(keep, masked_logits, LARGE_NEGATIVE_BIAS)
            action_index = jax.random.categorical(
                self.make_rng("sampling"), masked_logits.astype(jnp.float32)
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1)
        return HeadOutput(action_index=action_index, log_prob=log_prob, entropy=entropy)

    def _forward_move_head(
        self,
        query: jax.Array,
        keys: jax.Array,
        mask: jax.Array,
        move_head: HeadOutput,
    ):
        temp = self.cfg.get("temp", 1.0)
        current_moveset_oh = jnp.zeros((NUM_MOVES,), dtype=query.dtype)

        log_prob_accum = 0
        entropy_accum = 0
        action_indices = []

        for i in range(4):
            current_moveset_embedding = current_moveset_oh @ keys
            contextual_query = self.recurrent_moveset_sum(
                query, current_moveset_embedding
            )
            logits = jnp.einsum("j,kj->k", contextual_query, keys) / (
                temp * np.sqrt(self.cfg.entity_size).astype(query.dtype)
            )
            log_policy = legal_log_policy(logits, mask)

            entropy = 0
            train = self.cfg.get("train", False)
            if train:
                action_index = move_head.action_index[i]
                policy = legal_policy(logits, mask)
                entropy = -jnp.sum(policy * log_policy, axis=-1)
            else:
                masked_logits = jnp.where(mask, logits, LARGE_NEGATIVE_BIAS)
                min_p = self.cfg.get("min_p", 0.0)
                if 0.0 < min_p < 1.0:
                    max_logp = jnp.where(mask, log_policy, LARGE_NEGATIVE_BIAS).max(
                        keepdims=True, axis=-1
                    )
                    keep = log_policy >= (max_logp + math.log(min_p))
                    masked_logits = jnp.where(keep, masked_logits, LARGE_NEGATIVE_BIAS)
                action_index = jax.random.categorical(
                    self.make_rng("sampling"), masked_logits.astype(jnp.float32)
                )

            selected_move_oh = jax.nn.one_hot(action_index, num_classes=NUM_MOVES)
            current_moveset_oh = current_moveset_oh + selected_move_oh
            mask = mask & ~(selected_move_oh.astype(jnp.bool))

            log_prob = jnp.take(log_policy, action_index, axis=-1)

            log_prob_accum = log_prob_accum + log_prob
            entropy_accum = entropy_accum + entropy
            action_indices.append(action_index)

        return HeadOutput(
            action_index=jnp.stack(action_indices).reshape(-1),
            log_prob=log_prob_accum,
            entropy=entropy_accum,
        )

    def _forward_sub_head(
        self,
        query: jax.Array,
        keys: jax.Array,
        mask: jax.Array,
        head: HeadOutput,
    ):
        temp = self.cfg.get("temp", 1.0)
        logits = jnp.einsum("j,kj->k", query, keys) / (
            temp * np.sqrt(self.cfg.entity_size).astype(query.dtype)
        )
        log_policy = legal_log_policy(logits, mask)

        entropy = ()
        train = self.cfg.get("train", False)
        if train:
            action_index = head.action_index
            policy = legal_policy(logits, mask)
            entropy = -jnp.sum(policy * log_policy, axis=-1)
        else:
            masked_logits = jnp.where(mask, logits, LARGE_NEGATIVE_BIAS)
            min_p = self.cfg.get("min_p", 0.0)
            if 0.0 < min_p < 1.0:
                max_logp = jnp.where(mask, log_policy, LARGE_NEGATIVE_BIAS).max(
                    keepdims=True, axis=-1
                )
                keep = log_policy >= (max_logp + math.log(min_p))
                masked_logits = jnp.where(keep, masked_logits, LARGE_NEGATIVE_BIAS)
            action_index = jax.random.categorical(
                self.make_rng("sampling"), masked_logits.astype(jnp.float32)
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1)
        return HeadOutput(action_index=action_index, log_prob=log_prob, entropy=entropy)

    def _forward_mlp_head(
        self,
        logits: jax.Array,
        mask: jax.Array,
        head: HeadOutput,
    ):
        temp = self.cfg.get("temp", 1.0)
        logits = logits / temp
        log_policy = legal_log_policy(logits, mask)

        entropy = ()
        train = self.cfg.get("train", False)
        if train:
            action_index = head.action_index
            policy = legal_policy(logits, mask)
            entropy = -jnp.sum(policy * log_policy, axis=-1)
        else:
            masked_logits = jnp.where(mask, logits, LARGE_NEGATIVE_BIAS)
            min_p = self.cfg.get("min_p", 0.0)
            if 0.0 < min_p < 1.0:
                max_logp = jnp.where(mask, log_policy, LARGE_NEGATIVE_BIAS).max(
                    keepdims=True, axis=-1
                )
                keep = log_policy >= (max_logp + math.log(min_p))
                masked_logits = jnp.where(keep, masked_logits, LARGE_NEGATIVE_BIAS)
            action_index = jax.random.categorical(
                self.make_rng("sampling"), masked_logits.astype(jnp.float32)
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1)
        return HeadOutput(action_index=action_index, log_prob=log_prob, entropy=entropy)

    def _forward_ev_head(
        self,
        logits: jax.Array,
        head: HeadOutput,
    ):
        logits = nn.softplus(logits)
        log_norm_const = jnp.sum(jax.lax.lgamma(logits), axis=-1) - jax.lax.lgamma(
            jnp.sum(logits, axis=-1)
        )

        entropy = ()
        train = self.cfg.get("train", False)
        if train:
            value = head.action_index

            sum_concentration = jnp.sum(logits, axis=-1)
            entropy = (
                log_norm_const
                + (
                    (sum_concentration - logits.shape[-1])
                    * jax.lax.digamma(sum_concentration)
                )
                - jnp.sum((logits - 1.0) * jax.lax.digamma(logits), axis=-1)
            )
        else:
            value = jax.random.dirichlet(
                self.make_rng("sampling"), logits.astype(jnp.float32), (1,)
            )

        log_prob = jnp.sum((logits - 1.0) * jnp.log(value), axis=-1) - log_norm_const

        return HeadOutput(
            action_index=value.reshape(-1), log_prob=log_prob, entropy=entropy
        )

    def _forward(
        self,
        actor_input: BuilderActorInput,
        actor_output: BuilderActorOutput,
        species_keys: jax.Array,
        ability_keys: jax.Array,
        item_keys: jax.Array,
        move_keys: jax.Array,
    ) -> BuilderActorOutput:

        packed_set_tokens = actor_input.env.packed_set_tokens

        packed_set_attn_mask = jnp.ones_like(
            packed_set_tokens[..., PackedSetFeature.PACKED_SET_FEATURE__SPECIES],
            dtype=jnp.bool,
        )

        set_embeddings = jax.vmap(self._embed_packed_set)(packed_set_tokens)

        contextual_embeddings = self.encoder(
            set_embeddings,
            create_attention_mask(packed_set_attn_mask),
        )
        contextual_embedding = self.decoder(
            set_embeddings.mean(axis=0, keepdims=True),
            set_embeddings,
            create_attention_mask(
                packed_set_attn_mask.mean(axis=0, keepdims=True), packed_set_attn_mask
            ),
        ).reshape(-1)

        continue_head = self._forward_continue_head(
            contextual_embedding, actor_output.continue_head
        )
        selection_head = self._forward_selection_head(
            contextual_embeddings, actor_output.selection_head
        )

        selected_embedding = jnp.take(
            contextual_embeddings, selection_head.action_index, axis=0
        )

        species_head = self._forward_species_head(
            selected_embedding,
            species_keys,
            actor_input.env.species_mask + (actor_input.env.species_mask.sum() == 0),
            actor_output.species_head,
        )

        learnset_mask = jnp.take(
            MASKS[self.cfg.generation]["learnset"], species_head.action_index, axis=0
        )
        item_mask = jnp.take(
            MASKS[self.cfg.generation]["items"], species_head.action_index, axis=0
        )
        ability_mask = jnp.take(
            MASKS[self.cfg.generation]["abilities"], species_head.action_index, axis=0
        )

        contextualised_selection = (
            self.species_selection_sum(
                selected_embedding,
                jnp.take(species_keys, species_head.action_index, axis=0),
            )
            + selected_embedding
        )
        moveset_head = self._forward_move_head(
            contextualised_selection,
            move_keys,
            learnset_mask,
            actor_output.moveset_head,
        )

        contextualised_selection = (
            self.species_moveset_sum(
                contextualised_selection,
                jnp.take(move_keys, moveset_head.action_index, axis=0).sum(0),
            )
            + contextualised_selection
        )
        item_head = self._forward_sub_head(
            contextualised_selection,
            item_keys,
            item_mask,
            actor_output.item_head,
        )
        ability_head = self._forward_sub_head(
            contextualised_selection,
            ability_keys,
            ability_mask,
            actor_output.ability_head,
        )

        contextualised_selection = (
            self.species_moveset_sum(
                contextualised_selection,
                jnp.take(item_keys, item_head.action_index, axis=0),
                jnp.take(ability_keys, ability_head.action_index, axis=0),
            )
            + contextualised_selection
        )
        nature_head = self._forward_mlp_head(
            self.nature_mlp(contextualised_selection),
            np.ones((NUM_NATURES), dtype=np.bool_),
            actor_output.nature_head,
        )
        ev_head = self._forward_ev_head(
            self.ev_mlp(contextualised_selection),
            actor_output.ev_head,
        )
        teratype_head = self._forward_mlp_head(
            self.teratype_mlp(contextualised_selection),
            np.ones((NUM_TYPECHART), dtype=np.bool_),
            actor_output.teratype_head,
        )

        value = jnp.tanh(self.value_head(contextual_embedding)).squeeze(-1)

        return BuilderActorOutput(
            continue_head=continue_head,
            selection_head=selection_head,
            species_head=species_head,
            moveset_head=moveset_head,
            item_head=item_head,
            ability_head=ability_head,
            ev_head=ev_head,
            nature_head=nature_head,
            teratype_head=teratype_head,
            v=value,
        )

    def __call__(
        self,
        actor_input: BuilderActorInput,
        actor_output: BuilderActorOutput = BuilderActorOutput(),
    ) -> BuilderActorOutput:
        species_keys = jax.vmap(self._embed_species)(np.arange(NUM_SPECIES))
        ability_keys = jax.vmap(self._embed_ability)(np.arange(NUM_ABILITIES))
        item_keys = jax.vmap(self._embed_item)(np.arange(NUM_ITEMS))
        move_keys = jax.vmap(self._embed_move)(np.arange(NUM_MOVES))

        return jax.vmap(self._forward, in_axes=(0, 0, None, None, None, None))(
            actor_input, actor_output, species_keys, ability_keys, item_keys, move_keys
        )


def get_builder_model(config: ConfigDict = None) -> nn.Module:
    if config is None:
        config = get_builder_model_config()
    return Porygon2BuilderModel(config)


def main(debug: bool = True, generation: int = 9):
    init_jax_jit_cache()

    actor_network = get_builder_model(
        get_builder_model_config(generation, train=False, temp=0.8, min_p=0.05)
    )
    learner_network = get_builder_model(
        get_builder_model_config(generation, train=True)
    )

    ex_actor_input, ex_actor_output = jax.device_put(
        jax.tree.map(lambda x: x[:, 0], get_ex_builder_step())
    )
    key = jax.random.key(42)

    latest_ckpt = None  # get_most_recent_file(f"./ckpts/gen{generation}")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        builder_params = step["builder_state"]["params"]
    else:
        builder_params = learner_network.init(key, ex_actor_input, ex_actor_output)

    pprint(get_num_params(builder_params))

    agent = Agent(builder_apply_fn=actor_network.apply)

    builder_env = TeamBuilderEnvironment(generation=generation, max_ts=64)

    while True:

        rng_key, key = jax.random.split(key)
        builder_subkeys = jax.random.split(rng_key, builder_env.max_ts + 1)

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

        # learner_output = learner_network.apply(
        #     builder_params,
        #     BuilderActorInput(env=builder_trajectory.env_output),
        #     builder_trajectory.agent_output.actor_output,
        # )

        print(
            "value:", builder_trajectory.agent_output.actor_output.v.astype(jnp.float32)
        )

        print()


if __name__ == "__main__":
    main()
