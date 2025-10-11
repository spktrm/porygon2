import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.interfaces import HeadOutput
from rl.model.modules import MLP, PointerLogits, Resnet
from rl.model.utils import legal_log_policy, legal_policy


def sample_categorical(
    logits: jax.Array,
    log_policy: jax.Array,
    mask: jax.Array,
    rng_key: jax.random.PRNGKey,
    min_p: float = 0,
):
    masked_logits = jnp.where(mask, logits, -jnp.inf)
    if 0.0 < min_p < 1.0:
        masked_log_policy = jnp.where(mask, log_policy, -jnp.inf)
        max_logp = masked_log_policy.max(keepdims=True, axis=-1)
        keep = masked_log_policy >= (max_logp + math.log(min_p))
        masked_logits = jnp.where(keep & mask, masked_logits, -jnp.inf)
    return jax.random.categorical(rng_key, masked_logits.astype(jnp.float32), axis=-1)


class PolicyQKHead(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(
        self,
        query_embedding: jax.Array,
        key_embeddings: jax.Array,
        valid_mask: jax.Array,
        head: HeadOutput,
    ):
        resnet = Resnet(**self.cfg.resnet.to_dict())
        qk_logits = PointerLogits(**self.cfg.qk_logits.to_dict())

        temp = self.cfg.get("temp", 1.0)
        logits = qk_logits(resnet(query_embedding), key_embeddings)
        logits = logits / temp

        log_policy = legal_log_policy(logits, valid_mask)

        entropy = ()
        train = self.cfg.get("train", False)
        if train:
            action_index = head.action_index
            policy = legal_policy(logits, valid_mask)
            entropy = -jnp.sum(policy * log_policy, axis=-1)
        else:
            action_index = sample_categorical(
                logits,
                log_policy,
                valid_mask,
                self.make_rng("sampling"),
                self.cfg.get("min_p", 0.0),
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1)
        return HeadOutput(action_index=action_index, log_prob=log_prob, entropy=entropy)


class PolicyLogitHead(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, embedding: jax.Array, valid_mask: jax.Array, head: HeadOutput):
        resnet = Resnet(**self.cfg.resnet.to_dict())
        logits = MLP(**self.cfg.logits.to_dict())

        temp = self.cfg.get("temp", 1.0)
        embedding = resnet(embedding)
        logits = logits(embedding) / temp

        log_policy = legal_log_policy(logits, valid_mask)

        entropy = ()
        train = self.cfg.get("train", False)
        if train:
            action_index = head.action_index
            policy = legal_policy(logits, valid_mask)
            entropy = -jnp.sum(policy * log_policy, axis=-1)
        else:
            action_index = sample_categorical(
                logits,
                log_policy,
                valid_mask,
                self.make_rng("sampling"),
                self.cfg.get("min_p", 0.0),
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1)
        return HeadOutput(action_index=action_index, log_prob=log_prob, entropy=entropy)


class ValueLogitHead(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, embedding: jax.Array):
        resnet = Resnet(**self.cfg.resnet.to_dict())
        logits = MLP(**self.cfg.logits.to_dict())

        embedding = resnet(embedding)
        logits = logits(embedding)
        return logits.squeeze(-1)
