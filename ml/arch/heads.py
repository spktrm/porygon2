import chex
import numpy as np
import flax.linen as nn
import jax.numpy as jnp
from ml_collections import ConfigDict

from ml.arch.modules import MLP, Logits, TransformerDecoder, TransformerEncoder
from ml.func import legal_log_policy, legal_policy


class PolicyHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(
        self,
        action_embeddings: chex.Array,
        action_mask: chex.Array,
    ):
        action_embeddings = self.encoder(action_embeddings, jnp.ones_like(action_mask))

        logits = self.logits(action_embeddings)
        logits = logits.reshape(-1)
        logits = logits - logits.mean(where=action_mask)

        policy = legal_policy(logits, action_mask)
        log_policy = legal_log_policy(logits, action_mask)

        return logits, policy, log_policy


class ValueHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.decoder1 = TransformerDecoder(**self.cfg.transformer.to_dict())
        self.decoder2 = TransformerDecoder(**self.cfg.transformer.to_dict())
        self.logits = MLP(**self.cfg.logits.to_dict())

    def __call__(
        self,
        entity_embeddings: chex.Array,
        entity_mask: chex.Array,
        action_embeddings: chex.Array,
        action_mask: chex.Array,
    ):
        entity_context_embeddings = self.decoder1(
            entity_embeddings, action_embeddings, entity_mask, action_mask
        )
        action_context_embeddings = self.decoder2(
            action_embeddings, entity_embeddings, action_mask, entity_mask
        )

        def _pool_sequence(embeddings: chex.Array, mask: chex.Array):
            expanded_mask = mask[..., None]
            mask_sum = jnp.sum(mask).clip(min=1)
            mean_embedding = jnp.sum(embeddings * expanded_mask, axis=0) / mask_sum
            max_embedding = jnp.where(expanded_mask, embeddings, -1e30).max(0)
            min_embedding = jnp.where(expanded_mask, embeddings, 1e30).min(0)
            return jnp.concatenate(
                [mean_embedding, max_embedding, min_embedding], axis=-1
            )

        state_embedding = jnp.concatenate(
            (
                _pool_sequence(entity_context_embeddings, entity_mask),
                _pool_sequence(action_context_embeddings, action_mask),
            ),
            axis=-1,
        )

        output_logits = self.logits(state_embedding).reshape(-1)
        return jnp.tanh(output_logits)
