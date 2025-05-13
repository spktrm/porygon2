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

    def __call__(
        self,
        action_embeddings: chex.Array,
        action_mask: chex.Array,
    ):
        action_embeddings = self.encoder(action_embeddings, action_mask)

        logits = self.logits(action_embeddings)
        logits = logits.reshape(-1)
        logits = logits - logits.mean(where=action_mask)

        policy = legal_policy(logits, action_mask)
        log_policy = legal_log_policy(logits, action_mask)

        return logits, policy, log_policy


def _init_from_embeddings(
    key: chex.PRNGKey,
    shape: tuple,
    embeddings: chex.Array,
):
    return jax.random.normal(key, shape) * embeddings.std(axis=0) + embeddings.mean(
        axis=0
    )


class ClsEmbeddings(nn.Module):
    num_cls_embeddings: int = 1

    @nn.compact
    def __call__(self, embeddings: chex.Array, mask: chex.Array):
        cls_embedding = self.param(
            "cls_embedding",
            lambda key, shape: _init_from_embeddings(key, shape, embeddings),
            (self.num_cls_embeddings, embeddings.shape[-1]),
        )
        embeddings = jnp.concatenate([cls_embedding, embeddings], axis=0)
        mask = jnp.concatenate(
            [
                jnp.ones((self.num_cls_embeddings, mask.shape[1]), dtype=mask.dtype),
                mask,
            ],
            axis=0,
        )
        return embeddings, mask


class ValueHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder1 = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.encoder2 = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(
        self,
        entity_embeddings: chex.Array,
        entity_mask: chex.Array,
        action_embeddings: chex.Array,
        action_mask: chex.Array,
    ):
        entity_embeddings = self.encoder1(entity_embeddings, entity_mask)
        entities_embedding = entity_embeddings.mean(
            axis=0, where=entity_mask[..., None]
        )

        action_embeddings = self.encoder2(action_embeddings, action_mask)
        actions_embedding = action_embeddings.mean(axis=0, where=action_mask[..., None])

        state_embedding = jnp.concatenate(
            [entities_embedding, actions_embedding], axis=-1
        )

        return self.logits(state_embedding).reshape(-1)
