import chex
import flax.linen as nn
import jax
from ml_collections import ConfigDict

from ml.arch.modules import Logits, Resnet, TransformerEncoder
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
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(self, embeddings: chex.Array, mask: chex.Array):
        embeddings = self.encoder(embeddings, mask)

        logits = jax.vmap(self.logits)(embeddings)

        return logits.reshape(-1).mean(where=mask).reshape(-1)
