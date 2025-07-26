import chex
import flax.linen as nn
from ml_collections import ConfigDict

from rl.model.modules import RMSNorm, TransformerDecoder, activation_fn


class PolicyHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.decoder = TransformerDecoder(**self.cfg.transformer.to_dict())
        self.final_norm = RMSNorm(dtype=self.cfg.dtype)
        self.final_layer = nn.Dense(
            features=1, kernel_init=nn.initializers.normal(5e-3), dtype=self.cfg.dtype
        )

    def __call__(
        self,
        latent_embeddings: chex.Array,
        action_embeddings: chex.Array,
        action_mask: chex.Array,
        temp: float = 1.0,
    ):
        action_embeddings = self.decoder(
            action_embeddings, latent_embeddings, action_mask, None
        )

        logits = activation_fn(self.final_norm(action_embeddings))
        logits = self.final_layer(logits)
        logits = logits.reshape(-1)
        logits = (logits - logits.mean(axis=-1, keepdims=True)) / temp

        policy = nn.softmax(logits, where=action_mask)
        log_policy = nn.log_softmax(logits, where=action_mask)

        return logits, policy, log_policy


class ValueHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.decoder = TransformerDecoder(**self.cfg.transformer.to_dict())
        self.final_norm = RMSNorm(dtype=self.cfg.dtype)
        self.final_layer = nn.Dense(
            features=1, kernel_init=nn.initializers.normal(5e-3), dtype=self.cfg.dtype
        )

    def __call__(self, latent_embeddings: chex.Array):
        query = latent_embeddings.mean(axis=0, keepdims=True)

        pooled = self.decoder(query, latent_embeddings)
        pooled = activation_fn(self.final_norm(pooled))
        value = self.final_layer(pooled)

        return value.squeeze()
