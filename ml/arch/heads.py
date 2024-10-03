import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from ml.arch.modules import Logits, PointerLogits
from ml.func import legal_log_policy, legal_policy


class PolicyHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.move_logits = PointerLogits(**self.cfg.pointer_logits.to_dict())
        self.switch_logits = PointerLogits(**self.cfg.pointer_logits.to_dict())

    def __call__(
        self,
        state_embedding: chex.Array,
        move_embeddings: chex.Array,
        switch_embeddings: chex.Array,
        legal: chex.Array,
    ):
        denom = jnp.array(self.cfg.key_size, dtype=jnp.float32)

        move_logits = self.move_logits(state_embedding, move_embeddings)
        switch_logits = self.switch_logits(state_embedding, switch_embeddings)

        logits = jnp.concatenate((move_logits, switch_logits), axis=-1)
        logits = jax.lax.rsqrt(denom) * logits

        policy = legal_policy(logits, legal)
        log_policy = legal_log_policy(logits, legal)

        return logits, policy, log_policy


class ValueHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(self, x: chex.Array):
        return self.logits(x)
