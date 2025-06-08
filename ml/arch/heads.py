import chex
import flax.linen as nn
import jax
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
        # Vmap the Logits module to handle multiple heads with independent parameters
        VmappedLogits = nn.vmap(
            Logits,
            in_axes=0,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )

        # Instantiate the vmapped Logits module.
        # Arguments are for a single Logits instance.
        # nn.vmap will ensure parameters are created for `num_heads` instances.
        self.multi_value_heads = VmappedLogits(
            **self.cfg.logits.to_dict(), name="vectorized_value_logits"
        )

    def __call__(self, latent_embeddings: chex.Array):
        latent_embeddings = self.encoder(latent_embeddings)

        reshaped_for_heads = latent_embeddings.reshape(
            self.cfg.num_heads, -1, latent_embeddings.shape[-1]
        )

        def pool_head(embedding_slice: chex.Array) -> chex.Array:
            return jnp.concatenate(
                [
                    embedding_slice.mean(0),
                    embedding_slice.max(0),
                    embedding_slice.min(0),
                ],
                axis=-1,
            )

        pooled_embeddings = jax.vmap(pool_head)(reshaped_for_heads)
        head_outputs = self.multi_value_heads(pooled_embeddings)
        head_values_tanh = jnp.tanh(head_outputs)

        return head_values_tanh.reshape(-1).mean(keepdims=True)
