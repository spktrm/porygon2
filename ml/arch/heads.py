import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from ml.arch.modules import Logits, TransformerDecoder, TransformerEncoder
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
        self.decoder = TransformerDecoder(**self.cfg.transformer.to_dict())
        self.latents = self.param(
            "latent",
            nn.initializers.truncated_normal(0.02),
            (4, 512),
        )
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(self, embeddings: chex.Array, mask: chex.Array):
        embeddings = self.encoder(embeddings, mask)
        latents = self.decoder(self.latents, embeddings, None, mask)
        latent = latents.reshape(-1)
        return self.logits(latent).reshape(-1)
