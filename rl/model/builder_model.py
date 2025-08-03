import math
import pickle
from pprint import pprint
from typing import Callable, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.data import PACKED_SETS
from rl.environment.interfaces import (
    BuilderActorOutput,
    BuilderAgentOutput,
    BuilderEnvOutput,
    PolicyHeadOutput,
)
from rl.environment.utils import get_ex_builder_step
from rl.model.config import get_model_config
from rl.model.modules import (
    MLP,
    TransformerDecoder,
    TransformerEncoder,
    create_attention_mask,
)
from rl.model.utils import BIAS_VALUE, Params, legal_log_policy, legal_policy
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
        transformer_config = self.cfg.action_type_head.transformer.to_dict()
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

    def _sample_token(self, embedding: jax.Array, sample_mask: jax.Array):
        """
        Samples a token from the embeddings using the policy head and returns the token, log probability, and entropy.
        """
        logits = self.policy_head(embedding)
        masked_logits = jnp.where(sample_mask, logits, BIAS_VALUE)
        pi = legal_policy(logits, sample_mask)
        log_pi = legal_log_policy(logits, sample_mask)

        return PolicyHeadOutput(masked_logits, pi, log_pi)

    def _forward(self, input: BuilderEnvOutput) -> BuilderAgentOutput:
        """Autoregressively generates a team and returns (tokens, log_pi)."""
        not_masked = input.tokens != -1
        not_masked_sum = not_masked.sum(axis=-1)
        num_tokens = input.tokens.shape[-1]

        attn_mask = jnp.ones_like(input.tokens, dtype=jnp.bool)

        position_indices = jnp.arange(num_tokens, dtype=jnp.int32)

        embeddings = jnp.where(
            not_masked[..., None], self.mask_embedding, self.embeddings(input.tokens)
        )
        pred_embeddings = self.encoder(
            embeddings, create_attention_mask(attn_mask), position_indices
        )
        head_output = self._sample_token(
            jnp.take(pred_embeddings, not_masked_sum, axis=0), input.mask
        )
        pooled = self.decoder(
            pred_embeddings.mean(axis=0, keepdims=True),
            pred_embeddings,
            create_attention_mask(attn_mask.any(keepdims=True), attn_mask),
        )
        v = nn.tanh(self.value_head(pooled)).squeeze()

        return BuilderActorOutput(head=head_output, v=v)

    def __call__(self, input: BuilderEnvOutput) -> BuilderAgentOutput:
        return jax.vmap(self._forward)(input)


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

    ex = jax.device_put(jax.tree.map(lambda x: x[:, 0], get_ex_builder_step("gen3ou")))

    latest_ckpt = None  # get_most_recent_file("./ckpts")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        params = step["builder_state"]["params"]
    else:
        key = jax.random.key(42)
        params = network.init(key, ex)

    pprint(get_num_params(params))

    apply_fn: Callable[[Params, jax.Array, jax.Array | None], BuilderAgentOutput]
    # apply_fn = jax.jit(network.apply)
    apply_fn = network.apply

    output = apply_fn(params, ex)
    print(output)


if __name__ == "__main__":
    main()
