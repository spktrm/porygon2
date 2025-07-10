import chex
import flax.linen as nn
import jax
from ml_collections import ConfigDict

from ml.arch.modules import RMSNorm, TransformerDecoder
from ml.func import legal_log_policy, legal_policy


class PolicyHead(nn.Module):
    cfg: ConfigDict
    training: bool

    def setup(self):
        self.decoder = TransformerDecoder(**self.cfg.transformer.to_dict())
        self.final_norm = RMSNorm()
        self.final_layer = nn.Dense(
            features=1, kernel_init=nn.initializers.normal(5e-3)
        )
        self.temp = 1 if self.training else 0.1

    def __call__(
        self,
        latent_embeddings: chex.Array,
        action_embeddings: chex.Array,
        action_mask: chex.Array,
    ):
        action_embeddings = self.decoder(
            action_embeddings, latent_embeddings, action_mask, None
        )

        logits = jax.vmap(lambda x: self.final_layer(self.final_norm(x)))(
            action_embeddings
        )
        logits = logits.reshape(-1)
        logits = logits - logits.mean(where=action_mask)

        policy = legal_policy(logits, action_mask, self.temp)
        log_policy = legal_log_policy(logits, action_mask, self.temp)

        return logits, policy, log_policy


class ValueHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.decoder = TransformerDecoder(**self.cfg.transformer.to_dict())
        self.final_norm = RMSNorm()
        self.final_layer = nn.Dense(
            features=1, kernel_init=nn.initializers.normal(5e-3)
        )

    def __call__(self, latent_embeddings: chex.Array):
        query = latent_embeddings.mean(axis=0, keepdims=True)

        pooled = self.decoder(query, latent_embeddings)
        pooled = self.final_norm(pooled)
        value = self.final_layer(pooled)

        return value.reshape(-1)
