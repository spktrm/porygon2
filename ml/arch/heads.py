import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from ml.arch.modules import Logits, NormalizeAttention, PointerLogits, Resnet
from ml.func import legal_log_policy, legal_policy


class PolicyHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.resnet = Resnet(**self.cfg.query.to_dict())
        self.logits = PointerLogits(**self.cfg.pointer_logits.to_dict())

    def __call__(
        self,
        state_embedding: chex.Array,
        action_embeddings: chex.Array,
        legal: chex.Array,
    ):
        query = self.resnet(state_embedding)
        logits = self.logits(query, action_embeddings)

        policy = legal_policy(logits, legal)
        log_policy = legal_log_policy(logits, legal)

        return logits, policy, log_policy


class ValueHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.resnet = Resnet(**self.cfg.resnet.to_dict())
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(self, x: chex.Array):
        x = self.resnet(x)
        return self.logits(x)
