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
        self.action_logits = PointerLogits(**self.cfg.pointer_logits.to_dict())

    def __call__(
        self,
        state_embedding: chex.Array,
        action_embeddings: chex.Array,
        legal: chex.Array,
    ):
        denom = jnp.array(self.cfg.key_size, dtype=jnp.float32)

        action_logits = self.action_logits(state_embedding, action_embeddings)
        logits = jax.lax.rsqrt(denom) * action_logits

        policy = legal_policy(logits, legal)
        log_policy = legal_log_policy(logits, legal)

        return logits, policy, log_policy


class ValueHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(self, x: chex.Array):
        return self.logits(x)


class ValueHeadNash(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(
        self,
        my_action_embeddings: chex.Array,
        opp_action_embeddings: chex.Array,
        my_pi: chex.Array,
        opp_pi: chex.Array,
    ):
        payoff = my_action_embeddings @ opp_action_embeddings.T
        denom = jnp.array(my_action_embeddings.shape[-1], dtype=jnp.float32)
        payoff = payoff * jax.lax.rsqrt(denom)
        return (my_pi @ payoff @ opp_pi).reshape(-1)
