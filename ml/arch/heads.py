import chex
import flax.linen as nn
import jax
from ml_collections import ConfigDict

from ml.arch.modules import Logits, TransformerDecoder
from ml.func import legal_log_policy, legal_policy


class PolicyHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.decoder = TransformerDecoder(**self.cfg.transformer.to_dict())
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(
        self,
        latent_embeddings: chex.Array,
        action_embeddings: chex.Array,
        action_mask: chex.Array,
    ):
        action_embeddings = self.decoder(
            action_embeddings, latent_embeddings, action_mask, None
        )

        logits = jax.vmap(self.logits)(action_embeddings)
        logits = logits.reshape(-1)
        logits = logits - logits.mean(where=action_mask)

        policy = legal_policy(logits, action_mask)
        log_policy = legal_log_policy(logits, action_mask)

        return logits, policy, log_policy


class ValueHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.decoder = TransformerDecoder(**self.cfg.transformer.to_dict())
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(self, latent_embeddings: chex.Array):
        query = latent_embeddings.mean(axis=0, keepdims=True)

        pooled = self.decoder(query, latent_embeddings)
        value = self.logits(pooled)

        return value.reshape(-1)
