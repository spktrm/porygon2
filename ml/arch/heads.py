import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from ml.arch.modules import Logits, TransformerEncoder
from ml.func import legal_log_policy, legal_policy


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
        self.queries = self.param(
            "queries",
            nn.initializers.truncated_normal(0.02),
            (4, self.cfg.transformer.model_size),
        )
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(self, embeddings: chex.Array, mask: chex.Array):
        embeddings = self.encoder(embeddings, mask)

        attn_weights = embeddings @ self.queries.T
        attn_weights = jnp.where(mask[..., None], attn_weights, -1e9)
        attn_scores = nn.softmax(attn_weights, axis=0)

        state_embedding = attn_scores.T @ embeddings
        state_embedding = state_embedding.reshape(-1)

        logits = self.logits(state_embedding)
        # logits = softcap(logits, max_value=3)

        return logits.reshape(-1)
