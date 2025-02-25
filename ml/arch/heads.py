import chex
import flax.linen as nn
import jax.numpy as jnp
from ml_collections import ConfigDict

from ml.arch.modules import Logits, TransformerEncoder
from ml.func import legal_log_policy, legal_policy


class PolicyHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(self, action_embeddings: chex.Array, action_mask: chex.Array):
        action_embeddings = self.encoder(action_embeddings, action_mask)

        logits = self.logits(action_embeddings)
        logits = logits.reshape(-1)
        logits = logits - logits.mean(where=action_mask)

        policy = legal_policy(logits, action_mask)
        log_policy = legal_log_policy(logits, action_mask)

        return logits, policy, log_policy


class ValueHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.logits = Logits(**self.cfg.logits.to_dict())

        support = jnp.linspace(-1.5, 1.5, self.cfg.logits.num_logits + 1)
        self.centers = (support[:-1] + support[1:]) / 2

    def __call__(self, entity_embeddings: chex.Array, entity_mask: chex.Array):
        queries = self.encoder(entity_embeddings, entity_mask)
        b1, b2, b3 = jnp.split(queries, 3, axis=0)
        m1, m2, m3 = jnp.split(entity_mask[..., None], 3, axis=0)
        embedding = jnp.concatenate(
            (
                b1.mean(axis=0, where=m1),
                b2.mean(axis=0, where=m2),
                b3.mean(axis=0, where=m3),
            ),
            axis=-1,
        )
        return self.logits(embedding).reshape(-1)
