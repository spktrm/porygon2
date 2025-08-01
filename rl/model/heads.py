from typing import NamedTuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.model.modules import (
    RMSNorm,
    TransformerDecoder,
    TransformerEncoder,
    activation_fn,
    create_attention_mask,
)
from rl.model.utils import BIAS_VALUE, legal_log_policy, legal_policy


class PolicyHeadOutput(NamedTuple):
    action: jax.Array
    log_pi: jax.Array
    entropy: jax.Array


class PolicyHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.decoder = TransformerDecoder(**self.cfg.transformer.to_dict())
        self.final_norm = RMSNorm(dtype=self.cfg.dtype)
        self.final_layer = nn.Dense(
            features=1, kernel_init=nn.initializers.normal(5e-3), dtype=self.cfg.dtype
        )

    def __call__(
        self,
        query_embeddings: jax.Array,
        key_value_embeddings: jax.Array,
        query_mask: jax.Array,
        key_value_mask: jax.Array,
        rng_key: jax.Array,
        force_action: jax.Array = None,
        temp: float = 1.0,
    ):
        query_embeddings = self.encoder(
            query_embeddings,
            create_attention_mask(query_mask),
        )
        query_embeddings = self.decoder(
            query_embeddings,
            key_value_embeddings,
            create_attention_mask(query_mask, key_value_mask),
        )

        logits = activation_fn(self.final_norm(query_embeddings))
        logits = self.final_layer(logits)
        logits = logits.reshape(-1)
        logits = (logits - logits.mean(axis=-1, keepdims=True)) / temp

        policy = legal_policy(logits, query_mask, temp)
        log_policy = legal_log_policy(logits, query_mask, temp)

        entropy = -jnp.sum(policy * log_policy, axis=-1, keepdims=True)
        if force_action is not None:
            action = force_action
        else:
            masked_logits = jnp.where(query_mask, logits, BIAS_VALUE)
            action = jax.random.categorical(rng_key, masked_logits)

        return PolicyHeadOutput(
            action=action, log_pi=log_policy[action], entropy=entropy
        )


class ScalarHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.decoder = TransformerDecoder(**self.cfg.transformer.to_dict())
        self.final_norm = RMSNorm(dtype=self.cfg.dtype)
        self.final_layer = nn.Dense(
            features=self.cfg.output_features,
            kernel_init=nn.initializers.normal(5e-3),
            dtype=self.cfg.dtype,
        )

    def __call__(self, entity_embeddings: jax.Array, entity_mask: jax.Array):
        entity_embeddings = self.encoder(
            entity_embeddings,
            create_attention_mask(entity_mask),
        )
        query = entity_embeddings.mean(
            where=entity_mask[..., None], axis=0, keepdims=True
        )
        query_mask = entity_mask.any(keepdims=True)
        pooled = self.decoder(
            query, entity_embeddings, create_attention_mask(query_mask, entity_mask)
        )
        pooled = activation_fn(self.final_norm(pooled))
        value = self.final_layer(pooled)

        return value.squeeze()
