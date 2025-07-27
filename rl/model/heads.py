import flax.linen as nn
import jax
from ml_collections import ConfigDict

from rl.model.modules import (
    RMSNorm,
    TransformerDecoder,
    TransformerEncoder,
    activation_fn,
    create_attention_mask,
)


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
        entity_embeddings: jax.Array,
        action_embeddings: jax.Array,
        entity_mask: jax.Array,
        action_mask: jax.Array,
        temp: float = 1.0,
    ):
        action_embeddings = self.encoder(
            action_embeddings,
            create_attention_mask(action_mask),
        )
        action_embeddings = self.decoder(
            action_embeddings,
            entity_embeddings,
            create_attention_mask(action_mask, entity_mask),
        )

        logits = activation_fn(self.final_norm(action_embeddings))
        logits = self.final_layer(logits)
        logits = logits.reshape(-1)
        logits = (logits - logits.mean(axis=-1, keepdims=True)) / temp

        policy = nn.softmax(logits, where=action_mask)
        log_policy = nn.log_softmax(logits, where=action_mask)

        return logits, policy, log_policy


class ValueHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.decoder = TransformerDecoder(**self.cfg.transformer.to_dict())
        self.final_norm = RMSNorm(dtype=self.cfg.dtype)
        self.final_layer = nn.Dense(
            features=1, kernel_init=nn.initializers.normal(5e-3), dtype=self.cfg.dtype
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
