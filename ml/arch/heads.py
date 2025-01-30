import chex
import flax.linen as nn
import jax
from ml_collections import ConfigDict

from ml.arch.modules import Logits, Resnet, TransformerEncoder
from ml.func import legal_log_policy, legal_policy


import jax.numpy as jnp


class PolicyHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(self, embeddings: chex.Array, mask: chex.Array):
        embeddings = self.encoder(embeddings, mask)

        logits = jax.vmap(self.logits)(embeddings)
        logits = logits.reshape(-1)
        logits = logits - logits.mean(where=mask)

        policy = legal_policy(logits, mask)
        log_policy = legal_log_policy(logits, mask)

        return logits, policy, log_policy


class ValueHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.logits1 = Logits(**self.cfg.logits1.to_dict())
        self.logits2 = Logits(**self.cfg.logits2.to_dict())

    def __call__(self, embeddings: chex.Array, mask: chex.Array):
        embeddings = self.encoder(embeddings, mask)

        weights = jnp.where(mask[..., None], self.logits1(embeddings), -1e9)
        scores = jax.nn.softmax(weights, axis=0)

        weighted_embeddings = scores.T @ embeddings
        state_embedding = weighted_embeddings.reshape(-1)

        logits = self.logits2(state_embedding)

        return logits.reshape(-1)
