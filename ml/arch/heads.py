import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from ml.arch.modules import (
    MLP,
    Logits,
    PointerLogits,
    Resnet,
    TransformerDecoder,
    TransformerEncoder,
)
from ml.func import legal_log_policy, legal_policy
from rlenv.interfaces import EnvStep


class PolicyHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.transformer = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(
        self,
        action_embeddings: chex.Array,
        env_step: EnvStep,
    ):
        legal = env_step.legal

        action_embeddings = self.transformer(
            action_embeddings,
            legal,
        )
        logits = jax.vmap(self.logits)(action_embeddings)
        logits = logits.reshape(-1)
        logits = logits - logits.mean(where=legal)

        policy = legal_policy(logits, legal)
        log_policy = legal_log_policy(logits, legal)

        return logits, policy, log_policy


class ValueHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.transformer = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(self, entity_embeddings: chex.Array, entity_mask: chex.Array):
        x = self.transformer(entity_embeddings, entity_mask)
        x = jax.vmap(self.logits)(x)
        return x.reshape(-1).mean(where=entity_mask).reshape(1)
