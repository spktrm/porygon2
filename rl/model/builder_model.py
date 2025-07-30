import math
import pickle
from pprint import pprint
from typing import Callable, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.data import PACKED_SETS
from rl.environment.interfaces import ActorReset
from rl.model.config import get_model_config
from rl.model.modules import (
    MLP,
    TransformerDecoder,
    TransformerEncoder,
    create_attention_mask,
)
from rl.model.utils import BIAS_VALUE, Params, get_most_recent_file
from rl.utils import init_jax_jit_cache

SETS_DATA = PACKED_SETS["gen3ou"]


class Porygon2BuilderModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """
        Initializes the encoder, policy head, and value head using the configuration.
        """
        entity_size = self.cfg.entity_size
        dtype = self.cfg.dtype

        num_sets = len(SETS_DATA["sets"])

        self.embeddings = nn.Embed(
            num_embeddings=num_sets, features=entity_size, dtype=dtype
        )
        transformer_config = self.cfg.policy_head.transformer.to_dict()
        self.decoder = TransformerDecoder(**transformer_config)

        transformer_config["need_pos"] = True
        self.encoder = TransformerEncoder(**transformer_config)

        self.policy_head = MLP((entity_size, num_sets), dtype=dtype)
        self.value_head = MLP((entity_size, 1), dtype=dtype)

        self.mask_embedding = self.param(
            "mask_embedding",
            nn.initializers.truncated_normal(),
            (1, entity_size),
            dtype=dtype,
        )

    def _sample_token(
        self,
        embeddings: jax.Array,
        key: jax.Array,
        sample_mask: jax.Array,
        forced_token: jax.Array = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Samples a token from the embeddings using the policy head and returns the token, log probability, and entropy.
        """
        logits = self.policy_head(embeddings)
        masked_logits = jnp.where(sample_mask, logits, BIAS_VALUE)
        log_pi = jax.nn.log_softmax(masked_logits)
        if forced_token is not None:
            token = forced_token
        else:
            token = jax.random.categorical(key, masked_logits)

        sample_mask = sample_mask & ~SETS_DATA["mask"][token]
        ent = -jnp.sum(jnp.exp(log_pi) * log_pi, axis=-1)

        return token, log_pi[token], ent, sample_mask

    def forward(
        self, init_key: jax.Array, forced_tokens: jax.Array = None
    ) -> ActorReset:
        key_splits = jax.random.split(init_key, num=6)

        attn_masks = jnp.tril(jnp.ones((6, 6), dtype=jnp.bool))

        log_pi_accum = 0.0
        ent_accum = 0.0

        tokens = jnp.ones((6,), dtype=jnp.int32) * -1
        sample_mask = jnp.ones((self.embeddings.num_embeddings,), dtype=jnp.bool)
        position_indices = jnp.arange(6, dtype=jnp.int32)

        for i, subkey in enumerate(key_splits):
            embeddings = jnp.where(
                (tokens == -1)[..., None], self.mask_embedding, self.embeddings(tokens)
            )
            pred_embeddings = self.encoder(
                embeddings, create_attention_mask(attn_masks[i]), position_indices
            )
            token, log_pi, ent, sample_mask = self._sample_token(
                pred_embeddings[i],
                subkey,
                sample_mask,
                forced_tokens[i] if forced_tokens is not None else None,
            )

            log_pi_accum += log_pi
            ent_accum += ent

            tokens = tokens.at[i].set(token)

        pred_embeddings = self.encoder(
            self.embeddings(tokens),
            create_attention_mask(attn_masks[-1]),
            position_indices,
        )

        pooled = self.decoder(
            pred_embeddings.mean(axis=0, keepdims=True),
            pred_embeddings,
            create_attention_mask(attn_masks[-1].any(keepdims=True), attn_masks[-1]),
        )
        v = nn.tanh(self.value_head(pooled)).squeeze(-1)

        return ActorReset(
            tokens=tokens.reshape(-1),
            log_pi=log_pi_accum,
            v=v,
            key=init_key,
            entropy=ent_accum,
        )

    def __call__(
        self, init_key: jax.Array, forced_tokens: jax.Array = None
    ) -> ActorReset:
        """Autoregressively generates a team and returns (tokens, log_pi)."""
        return jax.vmap(self.forward)(init_key, forced_tokens)


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
        params = network.init(key, key[None])

    pprint(get_num_params(params))

    apply_fn: Callable[[Params, jax.Array, jax.Array | None], ActorReset]
    # apply_fn = jax.jit(network.apply)
    apply_fn = network.apply

    key = jax.random.key(42)

    while True:
        key, subkey = jax.random.split(key)
        output1 = apply_fn(params, subkey[None], None)
        assert jnp.all(output1.key == subkey)

        output = apply_fn(params, output1.key, output1.tokens)
        assert jnp.all(output.key == subkey)

        assert jnp.allclose(output.log_pi, output1.log_pi)

        print()


if __name__ == "__main__":
    main()
