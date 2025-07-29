import functools
import math
import pickle
from pprint import pprint
from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
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
)
from rl.environment.interfaces import EnvStep, ModelOutput, TimeStep
from rl.environment.protos.features_pb2 import TeamSetFeature
from rl.environment.utils import get_ex_step
from rl.model.config import get_model_config
from rl.model.encoder import Encoder
from rl.model.heads import PolicyHead, ValueHead
from rl.model.modules import (
    MLP,
    SumEmbeddings,
    TransformerEncoder,
    create_attention_mask,
)
from rl.model.utils import BIAS_VALUE, Params, get_most_recent_file
from rl.utils import init_jax_jit_cache


class Porygon2BuilderModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """
        Initializes the encoder, policy head, and value head using the configuration.
        """
        entity_size = self.cfg.entity_size
        dtype = self.cfg.dtype

        self.transformer = TransformerEncoder(
            **self.cfg.policy_head.transformer.to_dict()
        )
        self.set_transformer = TransformerEncoder(
            **self.cfg.policy_head.transformer.to_dict()
        )

        self.species_head = MLP((entity_size, NUM_SPECIES), dtype=dtype)
        self.item_head = MLP((entity_size, NUM_ITEMS), dtype=dtype)
        self.ability_head = MLP((entity_size, NUM_ABILITIES), dtype=dtype)
        self.moves_head = MLP((entity_size, NUM_MOVES), dtype=dtype)
        self.nature_head = MLP((entity_size, NUM_NATURES), dtype=dtype)
        self.gender_head = MLP((entity_size, NUM_GENDERS), dtype=dtype)
        self.evs_head = MLP((entity_size, 2 * 6), dtype=dtype)
        self.ivs_head = MLP((entity_size, 2 * 6), dtype=dtype)
        self.level_head = MLP((entity_size, 2), dtype=dtype)
        self.happiness_head = MLP((entity_size, 2), dtype=dtype)
        self.hidden_power_type_head = MLP((entity_size, NUM_TYPECHART), dtype=dtype)
        self.gigantamax_head = MLP((entity_size, 2), dtype=dtype)
        self.dynamax_level_head = MLP((entity_size, 2), dtype=dtype)
        self.tera_type_head = MLP((entity_size, NUM_TYPECHART), dtype=dtype)

        self.set_sum = SumEmbeddings(entity_size, dtype=dtype)

    def _sample_token(
        self,
        logits: jax.Array,
        key: jax.Array,
        mask: jax.Array | None = None,
    ) -> tuple[int, jax.Array, jax.Array]:
        if mask is not None:
            logits = jnp.where(mask, logits, BIAS_VALUE)
        probs = nn.softmax(logits, axis=-1)
        key, subkey = jax.random.split(key)
        token = jax.random.choice(subkey, probs.shape[-1], p=probs)
        log_prob = jnp.log(probs[token])
        return token, log_prob, key

    def _sample_moves(
        self, embedding: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Samples four distinct moves and returns (tokens, onehot, log_prob_sum, new_key)."""
        selected = jnp.zeros((NUM_MOVES,), dtype=jnp.bool_)
        tokens = []
        log_pi = 0.0

        for _ in range(4):
            mask = ~selected  # only allow moves that haven't been chosen yet
            token, log_p, key = self._sample_token(
                self.moves_head(embedding), key, mask
            )
            tokens.append(token)
            selected = selected | jax.nn.one_hot(token, NUM_MOVES).astype(jnp.bool_)
            log_pi += log_p

        return jnp.array(tokens), selected, log_pi, key

    def _sample_stat_dist(self, logits: jax.Array, key: jax.Array) -> jax.Array:
        mu, log_var = jnp.split(logits, 2, axis=-1)
        std = jnp.exp(0.5 * log_var)
        key, subkey = jax.random.split(key)
        embedding = mu + std * jax.random.normal(subkey, mu.shape, dtype=self.cfg.dtype)
        log_pi_z = -0.5 * jnp.sum(jnp.square((embedding - mu) / std), axis=-1)
        return embedding, log_pi_z, key

    def _generate_embedding(
        self,
        species_token: int,
        item_token: int,
        ability_token: int,
        moves_onehot: jax.Array,
        nature_token: int,
        stat_features: jax.Array,
        hidden_power_token: jax.Array,
    ) -> jax.Array:
        """Maps discrete choices into a dense entity embedding."""
        species_oh = jax.nn.one_hot(species_token, NUM_SPECIES)
        item_oh = jax.nn.one_hot(item_token, NUM_ITEMS)
        ability_oh = jax.nn.one_hot(ability_token, NUM_ABILITIES)
        nature_oh = jax.nn.one_hot(nature_token, NUM_NATURES)
        hidden_power_oh = jax.nn.one_hot(hidden_power_token, NUM_TYPECHART)
        return self.set_sum(
            species_oh,
            item_oh,
            ability_oh,
            moves_onehot.astype(self.cfg.dtype),
            nature_oh,
            stat_features,
            hidden_power_oh,
        )

    def _build_tokens(
        self,
        species_token: int,
        item_token: int,
        ability_token: int,
        moves_onehot: jax.Array,
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
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__MOVEID0].set(
            moves_onehot[0]
        )
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__MOVEID1].set(
            moves_onehot[1]
        )
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__MOVEID2].set(
            moves_onehot[2]
        )
        tokens = tokens.at[TeamSetFeature.TEAM_SET_FEATURE__MOVEID3].set(
            moves_onehot[3]
        )
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

    def __call__(self, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Autoregressively generates a team and returns (tokens, log_pi)."""
        key, init_key = jax.random.split(key)
        team = jax.random.normal(
            init_key, (6, self.cfg.entity_size), dtype=self.cfg.dtype
        )

        log_pi_total = 0.0
        all_tokens = []
        species_mask = jnp.ones((NUM_SPECIES,), dtype=jnp.bool_)
        attn_masks = jnp.tril(jnp.ones((6, 6), dtype=jnp.bool_))

        for i in range(6):
            embeddings = self.transformer(team, create_attention_mask(attn_masks[i]))
            embed_i = embeddings[i]

            # ---- Sample species (without reuse) ---------------------------------
            species_t, log_p, key = self._sample_token(
                self.species_head(embed_i), key, species_mask
            )
            log_pi_total += log_p
            species_mask = species_mask.at[species_t].set(False)

            # ---- Sample item -----------------------------------------------------
            item_t, log_p, key = self._sample_token(self.item_head(embed_i), key)
            log_pi_total += log_p

            # ---- Sample ability --------------------------------------------------
            ability_t, log_p, key = self._sample_token(self.ability_head(embed_i), key)
            log_pi_total += log_p

            # ---- Sample four distinct moves -------------------------------------
            move_toks, moves_onehot, log_p_moves, key = self._sample_moves(embed_i, key)
            log_pi_total += log_p_moves

            # ---- Sample nature --------------------------------------------------
            nature_t, log_p, key = self._sample_token(self.nature_head(embed_i), key)
            log_pi_total += log_p

            # ---- Sample evs --------------------------------------------------
            evs_t, log_p, key = self._sample_stat_dist(self.evs_head(embed_i), key)
            log_pi_total += log_p

            # ---- Sample ivs --------------------------------------------------
            # ivs_t, log_p, key = self._sample_stat_dist(self.ivs_head(embed_i), key)
            # log_pi_total += log_p

            # ---- Sample level --------------------------------------------------
            level_t, log_p, key = self._sample_stat_dist(self.level_head(embed_i), key)
            log_pi_total += log_p

            # ---- Sample happiness --------------------------------------------------
            happiness_t, log_p, key = self._sample_stat_dist(
                self.happiness_head(embed_i), key
            )
            log_pi_total += log_p

            # ---- Sample hidden_power_type --------------------------------------------------
            hidden_power_type_t, log_p, key = self._sample_token(
                self.hidden_power_type_head(embed_i), key
            )
            log_pi_total += log_p

            # ---- Build embedding & update team ----------------------------------
            evs_dist = nn.softmax(evs_t)
            level_prob = nn.sigmoid(level_t)
            happiness_prob = nn.sigmoid(happiness_t)
            team = team.at[i].set(
                self._generate_embedding(
                    species_t,
                    item_t,
                    ability_t,
                    moves_onehot,
                    nature_t,
                    jnp.concatenate(
                        (
                            evs_dist,
                            level_prob,
                            happiness_prob,
                        ),
                        axis=-1,
                    ),
                    hidden_power_type_t,
                )
            )

            # Accumulate tokens for the slot in the expected order.
            current = self._build_tokens(
                species_t,
                item_t,
                ability_t,
                moves_onehot,
                nature_t,
                jnp.floor(512 * evs_dist).astype(jnp.int32),
                hidden_power_type_t,
                jnp.floor(100 * level_prob).astype(jnp.int32).squeeze(-1),
                jnp.floor(255 * happiness_prob).astype(jnp.int32).squeeze(-1),
            )
            all_tokens.append(current)

        return jnp.stack(all_tokens), log_pi_total


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

    key = jax.random.key(42)
    params = network.init(key, key)

    jitted_apply = jax.jit(network.apply)

    for _ in range(10):
        key, subkey = jax.random.split(key)
        output = jitted_apply(params, subkey)
    pprint(get_num_params(params))


if __name__ == "__main__":
    main()
