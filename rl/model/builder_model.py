import math
import pickle
from pprint import pprint
from typing import Callable, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from rl.environment.data import (
    NUM_ABILITIES,
    NUM_GENDERS,
    NUM_ITEMS,
    NUM_MOVES,
    NUM_NATURES,
    NUM_SPECIES,
    NUM_TEAM_SET_FEATURES,
    NUM_TYPECHART,
    STOI,
)
from rl.environment.interfaces import ActorReset
from rl.environment.protos.features_pb2 import TeamSetFeature
from rl.model.config import get_model_config
from rl.model.modules import (
    MLP,
    PretrainedEmbedding,
    SumEmbeddings,
    TransformerDecoder,
    TransformerEncoder,
    create_attention_mask,
)
from rl.model.utils import BIAS_VALUE, Params, get_most_recent_file
from rl.utils import init_jax_jit_cache

SPECIES_MASK = PretrainedEmbedding(
    fpath="data/data/gen3/OU_species_mask.npy", dtype=jnp.bool
)
LEARNSET_MASK = PretrainedEmbedding(
    fpath="data/data/gen3/OU_learnset_mask.npy", dtype=jnp.bool
)
ABILITIES_MASK = PretrainedEmbedding(
    fpath="data/data/gen3/OU_ability_mask.npy", dtype=jnp.bool
)
ITEM_MASK = PretrainedEmbedding(fpath="data/data/gen3/OU_item_mask.npy", dtype=jnp.bool)


class Porygon2BuilderModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """
        Initializes the encoder, policy head, and value head using the configuration.
        """
        entity_size = self.cfg.entity_size
        dtype = self.cfg.dtype

        self.encoder = TransformerEncoder(**self.cfg.policy_head.transformer.to_dict())
        self.decoder = TransformerDecoder(**self.cfg.policy_head.transformer.to_dict())
        self.value_head = MLP((entity_size, 1), dtype=dtype)

        self.species_head = MLP((entity_size, NUM_SPECIES), dtype=dtype)
        self.item_head = SumEmbeddings(NUM_ITEMS, dtype=dtype)
        self.ability_head = SumEmbeddings(NUM_ABILITIES, dtype=dtype)
        self.moves_head = SumEmbeddings(NUM_MOVES, dtype=dtype)
        self.nature_head = SumEmbeddings(NUM_NATURES, dtype=dtype)
        self.gender_head = SumEmbeddings(NUM_GENDERS, dtype=dtype)
        self.evs_head = SumEmbeddings(2 * 6, dtype=dtype)
        self.ivs_head = SumEmbeddings(2 * 6, dtype=dtype)
        self.level_head = SumEmbeddings(2, dtype=dtype)
        self.happiness_head = SumEmbeddings(2, dtype=dtype)
        self.hidden_power_type_head = SumEmbeddings(NUM_TYPECHART, dtype=dtype)
        self.gigantamax_head = SumEmbeddings(2, dtype=dtype)
        self.dynamax_level_head = SumEmbeddings(2, dtype=dtype)
        self.tera_type_head = SumEmbeddings(NUM_TYPECHART, dtype=dtype)

        self.set_sum = SumEmbeddings(entity_size, dtype=dtype)

    def _sample_token(
        self,
        logits: jax.Array,
        key: jax.Array,
        mask: jax.Array | None = None,
        forced_token: jax.Array | None = None,
    ) -> tuple[int, jax.Array, jax.Array]:
        if mask is not None:
            logits = jnp.where(mask, logits, BIAS_VALUE)
        log_probs = nn.log_softmax(logits, axis=-1)
        key, subkey = jax.random.split(key)
        if forced_token is None:
            token = jax.random.categorical(subkey, logits)
        else:
            token = forced_token
        log_prob = log_probs[token]
        probs = jnp.exp(log_probs)
        ohe = jax.nn.one_hot(token, logits.shape[-1], dtype=log_probs.dtype)
        pass_through = probs + jax.lax.stop_gradient(ohe - probs)
        return token, pass_through, log_prob, key

    def _sample_moves(
        self,
        logits: jax.Array,
        key: jax.Array,
        mask: jax.Array | None = None,
        forced_tokens: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Samples four distinct moves and returns (tokens, onehot, log_prob_sum, new_key)."""
        selected = jnp.zeros((NUM_MOVES,), dtype=self.cfg.dtype)
        tokens = []
        log_pi = 0.0

        for i in range(4):
            # only allow moves that haven't been chosen yet
            selection_mask = ~(selected).astype(jnp.bool)
            token, move_ohe, log_p, key = self._sample_token(
                logits,
                key,
                selection_mask & mask,
                forced_token=None if forced_tokens is None else forced_tokens[i],
            )
            tokens.append(token)
            selected = selected + move_ohe
            log_pi += log_p

        return jnp.array(tokens), selected, log_pi, key

    def _sample_stat_dist(self, logits: jax.Array, key: jax.Array) -> jax.Array:
        mu, log_var = jnp.split(logits, 2, axis=-1)
        log_std = 0.5 * log_var
        std = jnp.exp(log_std)

        key, subkey = jax.random.split(key)
        eps = jax.random.normal(subkey, mu.shape, dtype=self.cfg.dtype)
        z = mu + std * eps  # re‑parameterised sample

        log_unnormalized = -0.5 * jnp.square(eps)
        _half_log2pi = 0.5 * math.log(2 * math.pi)
        log_normalization = _half_log2pi + jnp.log(std)
        log_pi_z = (log_normalization + log_unnormalized).sum()

        return z, log_pi_z, key

    def _generate_embedding(
        self,
        species_ohe: int,
        item_ohe: int,
        ability_ohe: int,
        moves_onehot: jax.Array,
        nature_ohe: int,
        stat_features: jax.Array,
        hidden_power_ohe: jax.Array,
    ) -> jax.Array:
        """Maps discrete choices into a dense entity embedding."""
        return self.set_sum(
            species_ohe,
            item_ohe,
            ability_ohe,
            moves_onehot.astype(self.cfg.dtype),
            nature_ohe,
            stat_features,
            hidden_power_ohe,
        )

    def _build_tokens(
        self,
        species_token: int,
        item_token: int,
        ability_token: int,
        move_toks: jax.Array,
        nature_token: int,
        evs_dist: jax.Array,
        hidden_power_token: jax.Array,
        level_t: jax.Array,
        happiness_t: jax.Array,
    ) -> jax.Array:
        tokens = jnp.zeros((NUM_TEAM_SET_FEATURES,), dtype=jnp.int32)
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__SPECIES].set(species_token)
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__ITEM].set(item_token)
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__ABILITY].set(ability_token)
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__MOVEID0].set(move_toks[0])
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__MOVEID1].set(move_toks[1])
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__MOVEID2].set(move_toks[2])
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__MOVEID3].set(move_toks[3])
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__NATURE].set(nature_token)
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__EV_HP].set(evs_dist[0])
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__EV_ATK].set(evs_dist[1])
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__EV_DEF].set(evs_dist[2])
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__EV_SPA].set(evs_dist[3])
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__EV_SPD].set(evs_dist[4])
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__EV_SPE].set(evs_dist[5])
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__HIDDEN_POWER_TYPE].set(
            hidden_power_token
        )
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__LEVEL].set(level_t)
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__HAPPINESS].set(happiness_t)
        return tokens

    def __call__(
        self, init_key: jax.Array, forced_tokens: jax.Array = None
    ) -> tuple[jax.Array, jax.Array]:
        """Autoregressively generates a team and returns (tokens, log_pi)."""
        key, subkey = jax.random.split(init_key)
        team = jax.random.normal(
            subkey, (6, self.cfg.entity_size), dtype=self.cfg.dtype
        )

        log_pi_total = 0.0
        all_tokens = []
        species_mask = jnp.ones((NUM_SPECIES,), dtype=jnp.bool_)
        attn_masks = jnp.tril(jnp.ones((6, 6), dtype=jnp.bool_))

        for i in range(6):
            embeddings = self.encoder(team, create_attention_mask(attn_masks[i]))
            embed_i = embeddings[i]

            # ---- Sample species (without reuse) ---------------------------------
            species_t, species_ohe, log_p_species, key = self._sample_token(
                self.species_head(embed_i),
                key,
                species_mask & SPECIES_MASK.embeddings,
                forced_token=(
                    forced_tokens[i, TeamSetFeature.TEAM_SET_FEATURE__SPECIES]
                    if forced_tokens is not None
                    else None
                ),
            )
            log_pi_total += log_p_species
            species_mask = species_mask.at[species_t].set(False)

            # ---- Sample item -----------------------------------------------------
            item_t, item_ohe, log_p_item, key = self._sample_token(
                self.item_head(embed_i, species_ohe),
                key,
                ITEM_MASK.embeddings,
                forced_token=(
                    forced_tokens[i, TeamSetFeature.TEAM_SET_FEATURE__ITEM]
                    if forced_tokens is not None
                    else None
                ),
            )
            log_pi_total += log_p_item

            # ---- Sample ability --------------------------------------------------
            ability_t, ability_ohe, log_p_ability, key = self._sample_token(
                self.ability_head(embed_i, species_ohe),
                key,
                ABILITIES_MASK(species_t),
                forced_token=(
                    forced_tokens[i, TeamSetFeature.TEAM_SET_FEATURE__ABILITY]
                    if forced_tokens is not None
                    else None
                ),
            )
            log_pi_total += log_p_ability

            # ---- Sample four distinct moves -------------------------------------
            move_toks, moves_onehot, log_p_moves, key = self._sample_moves(
                self.moves_head(embed_i, species_ohe),
                key,
                LEARNSET_MASK(species_t),
                forced_tokens=(
                    forced_tokens[
                        i,
                        TeamSetFeature.TEAM_SET_FEATURE__MOVEID0 : TeamSetFeature.TEAM_SET_FEATURE__MOVEID3
                        + 1,
                    ]
                    if forced_tokens is not None
                    else None
                ),
            )
            log_pi_total += log_p_moves

            # ---- Sample nature --------------------------------------------------
            nature_mask = np.ones((NUM_NATURES,), dtype=np.bool_)
            nature_mask[:4] = False  # First four placeholder natures are not allowed
            nature_t, nature_ohe, log_p_nature, key = self._sample_token(
                self.nature_head(embed_i, species_ohe),
                key,
                nature_mask,
                forced_token=(
                    forced_tokens[i, TeamSetFeature.TEAM_SET_FEATURE__NATURE]
                    if forced_tokens is not None
                    else None
                ),
            )
            log_pi_total += log_p_nature

            # ---- Sample evs --------------------------------------------------
            evs_t, log_p_evs, key = self._sample_stat_dist(
                self.evs_head(embed_i, species_ohe), key
            )
            log_pi_total += log_p_evs

            # ---- Sample ivs --------------------------------------------------
            # ivs_t, log_p, key = self._sample_stat_dist(self.ivs_head(embed_i), key)
            # log_pi_total += log_p

            # ---- Sample level --------------------------------------------------
            level_t, log_p_level, key = self._sample_stat_dist(
                self.level_head(embed_i, species_ohe), key
            )
            log_pi_total += log_p_level

            # ---- Sample happiness --------------------------------------------------
            happiness_t, log_p_happiness, key = self._sample_stat_dist(
                self.happiness_head(embed_i, species_ohe), key
            )
            log_pi_total += log_p_happiness

            # ---- Sample hidden_power_type --------------------------------------------------
            hidden_power_type_t, hidden_power_type_ohe, log_p_hidden_power_type, key = (
                self._sample_token(
                    self.hidden_power_type_head(embed_i, species_ohe),
                    key,
                    forced_token=(
                        forced_tokens[
                            i, TeamSetFeature.TEAM_SET_FEATURE__HIDDEN_POWER_TYPE
                        ]
                        if forced_tokens is not None
                        else None
                    ),
                )
            )
            log_pi_total += log_p_hidden_power_type

            # ---- Build embedding & update team ----------------------------------
            evs_dist = nn.softmax(evs_t)
            level_prob = nn.sigmoid(level_t)
            happiness_prob = nn.sigmoid(happiness_t)
            team = team.at[i].set(
                self._generate_embedding(
                    species_ohe,
                    item_ohe,
                    ability_ohe,
                    moves_onehot,
                    nature_ohe,
                    jnp.concatenate(
                        (
                            evs_dist,
                            level_prob,
                            happiness_prob,
                        ),
                        axis=-1,
                    ),
                    hidden_power_type_ohe,
                )
            )

            # Accumulate tokens for the slot in the expected order.
            current = self._build_tokens(
                species_t,
                item_t,
                ability_t,
                move_toks,
                nature_t,
                jnp.floor(512 * evs_dist).astype(jnp.int32),
                hidden_power_type_t,
                jnp.floor(100 * level_prob).astype(jnp.int32).squeeze(-1),
                jnp.floor(255 * happiness_prob).astype(jnp.int32).squeeze(-1),
            )
            all_tokens.append(current)

        pooled = self.decoder(
            team.mean(axis=0, keepdims=True),
            team,
            create_attention_mask(attn_masks[-1].any(keepdims=True), attn_masks[-1]),
        )
        v = nn.tanh(self.value_head(pooled)).squeeze(-1)

        return ActorReset(
            tokens=jnp.stack(all_tokens), log_pi=log_pi_total, v=v, key=init_key
        )


def get_num_params(vars: Params, n: int = 3) -> Dict[str, Dict[str, float]]:
    def calculate_params(key: str, vars: Params) -> int:
        total = 0
        for key, value in vars.items():
            if isinstance(value, jax.Array):
                total += math.prod(value.shape)
            else:
                total += calculate_params(key, value)
        return total

    def build_param_dict(
        vars: Params, total_params: int, current_depth: int
    ) -> Dict[str, Dict[str, float]]:
        param_dict = {}
        for key, value in vars.items():
            if isinstance(value, jax.Array):
                num_params = math.prod(value.shape)
                param_dict[key] = {
                    "num_params": num_params,
                    "ratio": num_params / total_params,
                }
            else:
                nested_params = calculate_params(key, value)
                param_entry = {
                    "num_params": nested_params,
                    "ratio": nested_params / total_params,
                }
                if current_depth < n - 1:
                    param_entry["details"] = build_param_dict(
                        value, total_params, current_depth + 1
                    )
                param_dict[key] = param_entry
        return param_dict

    total_params = calculate_params("base", vars)
    return build_param_dict(vars, total_params, 0)


def get_builder_model(config: ConfigDict = None) -> nn.Module:
    if config is None:
        config = get_model_config()
    return Porygon2BuilderModel(config)


def assert_no_nan_or_inf(gradients, path=""):
    if isinstance(gradients, dict):
        for key, value in gradients.items():
            new_path = f"{path}/{key}" if path else key
            assert_no_nan_or_inf(value, new_path)
    else:
        if jnp.isnan(gradients).any() or jnp.isinf(gradients).any():
            raise ValueError(f"Gradient at {path} contains NaN or Inf values.")


def main():
    init_jax_jit_cache()
    network = get_builder_model()

    latest_ckpt = get_most_recent_file("./ckpts")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        params = step["builder_state"]["params"]
    else:
        key = jax.random.key(42)
        params = network.init(key, key)

    pprint(get_num_params(params))

    apply_fn: Callable[[Params, jax.Array, jax.Array | None], ActorReset]
    # apply_fn = jax.jit(network.apply)
    apply_fn = network.apply

    key = jax.random.key(42)

    while True:
        key, subkey = jax.random.split(key)
        output1 = apply_fn(params, subkey)
        assert jnp.all(output1.key == subkey)

        output = apply_fn(params, output1.subkey, output1.tokens)
        assert jnp.all(output.key == subkey)

        assert jnp.allclose(output.log_pi, output1.log_pi)

        for row in np.asarray(output.tokens):
            print(
                "|"
                + STOI["species"][row[TeamSetFeature.TEAM_SET_FEATURE__SPECIES]]
                + "|"
                + STOI["items"][row[TeamSetFeature.TEAM_SET_FEATURE__ITEM]]
                + "|"
                + STOI["abilities"][row[TeamSetFeature.TEAM_SET_FEATURE__ABILITY]]
                + "|"
                + STOI["moves"][row[TeamSetFeature.TEAM_SET_FEATURE__MOVEID0]]
                + ","
                + STOI["moves"][row[TeamSetFeature.TEAM_SET_FEATURE__MOVEID1]]
                + ","
                + STOI["moves"][row[TeamSetFeature.TEAM_SET_FEATURE__MOVEID2]]
                + ","
                + STOI["moves"][row[TeamSetFeature.TEAM_SET_FEATURE__MOVEID3]]
                + "|"
                + STOI["natures"][row[TeamSetFeature.TEAM_SET_FEATURE__NATURE]]
                + "|"
                + str(row[TeamSetFeature.TEAM_SET_FEATURE__EV_HP])
                + ","
                + str(row[TeamSetFeature.TEAM_SET_FEATURE__EV_ATK])
                + ","
                + str(row[TeamSetFeature.TEAM_SET_FEATURE__EV_DEF])
                + ","
                + str(row[TeamSetFeature.TEAM_SET_FEATURE__EV_SPA])
                + ","
                + str(row[TeamSetFeature.TEAM_SET_FEATURE__EV_SPD])
                + ","
                + str(row[TeamSetFeature.TEAM_SET_FEATURE__EV_SPE])
                + "|"
                + str(row[TeamSetFeature.TEAM_SET_FEATURE__LEVEL])
            )

        print()


if __name__ == "__main__":
    main()
