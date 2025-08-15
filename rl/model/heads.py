import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.interfaces import PolicyHeadOutput
from rl.environment.protos.features_pb2 import ActionMaskFeature
from rl.model.modules import (
    MLP,
    TransformerDecoder,
    TransformerEncoder,
    create_attention_mask,
)
from rl.model.utils import BIAS_VALUE, legal_log_policy, legal_policy


class MoveHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.decoder = TransformerDecoder(**self.cfg.transformer.to_dict())
        self.final_mlp = MLP(1, dtype=self.cfg.dtype)
        self.wildcard_head = MLP(
            ActionMaskFeature.ACTION_MASK_FEATURE__CAN_TERA
            - ActionMaskFeature.ACTION_MASK_FEATURE__CAN_NORMAL
            + 1,
            dtype=self.cfg.dtype,
        )

    def __call__(
        self,
        query_embeddings: jax.Array,
        key_value_embeddings: jax.Array,
        query_mask: jax.Array,
        key_value_mask: jax.Array,
        wildcard_mask: jax.Array,
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

        logits = self.final_mlp(query_embeddings)
        logits = logits.reshape(-1) / temp

        masked_logits = jnp.where(query_mask, logits, BIAS_VALUE)
        policy = legal_policy(logits, query_mask, temp)
        log_policy = legal_log_policy(logits, query_mask, temp)

        wildcard_logits = self.wildcard_head(query_embeddings)
        wildcard_logits = wildcard_logits / temp

        wildcard_masked_logits = jnp.where(
            wildcard_mask[None], wildcard_logits, BIAS_VALUE
        )
        wildcard_policy = jax.vmap(legal_policy, in_axes=(0, None, None))(
            wildcard_logits, wildcard_mask, temp
        )
        wildcard_log_policy = jax.vmap(legal_log_policy, in_axes=(0, None, None))(
            wildcard_logits, wildcard_mask, temp
        )

        return PolicyHeadOutput(
            logits=masked_logits,
            policy=policy,
            log_policy=log_policy,
        ), PolicyHeadOutput(
            logits=wildcard_masked_logits,
            policy=wildcard_policy,
            log_policy=wildcard_log_policy,
        )


class PolicyHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.decoder = TransformerDecoder(**self.cfg.transformer.to_dict())
        self.final_mlp = MLP(1, dtype=self.cfg.dtype)

    def __call__(
        self,
        query_embeddings: jax.Array,
        key_value_embeddings: jax.Array,
        query_mask: jax.Array,
        key_value_mask: jax.Array,
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

        logits = self.final_mlp(query_embeddings)
        logits = logits.reshape(-1) / temp

        masked_logits = jnp.where(query_mask, logits, BIAS_VALUE)
        policy = legal_policy(logits, query_mask, temp)
        log_policy = legal_log_policy(logits, query_mask, temp)

        return PolicyHeadOutput(
            logits=masked_logits,
            policy=policy,
            log_policy=log_policy,
        )


class ScalarHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.decoder = TransformerDecoder(**self.cfg.transformer.to_dict())
        self.final_mlp = MLP(self.cfg.output_features, dtype=self.cfg.dtype)

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
        value = self.final_mlp(pooled)

        return value.squeeze()
