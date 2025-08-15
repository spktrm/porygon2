import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.protos.features_pb2 import ActionMaskFeature
from rl.model.modules import (
    MLP,
    TransformerDecoder,
    TransformerEncoder,
    create_attention_mask,
)
from rl.model.utils import BIAS_VALUE


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

        move_logits = self.final_mlp(query_embeddings).reshape(-1)
        masked_move_logits = jnp.where(query_mask, move_logits, BIAS_VALUE)

        wildcard_logits = self.wildcard_head(query_embeddings)
        wildcard_masked_logits = jnp.where(
            wildcard_mask[None], wildcard_logits, BIAS_VALUE
        )

        return masked_move_logits, wildcard_masked_logits


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
        logits = logits.reshape(-1)

        masked_logits = jnp.where(query_mask, logits, BIAS_VALUE)

        return masked_logits


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
