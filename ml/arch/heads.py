import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from ml.arch.modules import Logits, TransformerEncoder
from ml.func import legal_log_policy, legal_policy
from rlenv.data import NUM_ACTIONS
from rlenv.interfaces import EnvStep


class OfflinePolicyHead(nn.Module):

    def setup(self):
        self.logits = Logits(NUM_ACTIONS)

    def __call__(self, state_embedding: chex.Array):

        logits = self.logits(state_embedding)
        logits = logits.reshape(-1)
        logits = logits - logits.mean()

        legal = jnp.ones_like(logits)
        policy = legal_policy(logits, legal)
        log_policy = legal_log_policy(logits, legal)

        return logits, policy, log_policy


class PolicyHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.transformer = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(
        self,
        action_embeddings: chex.Array,
        mask: chex.Array,
    ):
        action_embeddings = self.transformer(action_embeddings, mask)
        logits = jax.vmap(self.logits)(action_embeddings)
        logits = logits.reshape(-1)
        logits = logits - logits.mean(where=mask)

        policy = legal_policy(logits, mask)
        log_policy = legal_log_policy(logits, mask)

        return logits, policy, log_policy


class ValueHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.transformer = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(
        self,
        entity_embeddings: chex.Array,
        mask: chex.Array,
    ):
        entity_embeddings = self.transformer(entity_embeddings, mask)
        logits = jax.vmap(self.logits)(entity_embeddings)
        logits = logits.reshape(-1)

        return logits.mean(where=mask, keepdims=True)
