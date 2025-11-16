import math
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.interfaces import HeadOutput
from rl.model.modules import MLP, PointerLogits
from rl.model.utils import legal_log_policy, legal_policy


class HeadParams(NamedTuple):
    temp: float = 1.0
    min_p: float = 0.0


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
        head: HeadOutput,
        valid_mask: jax.Array = None,
        head_params: HeadParams = HeadParams(),
    ):
        query_embedding = MLP()(query_embedding)
        qk_logits = PointerLogits(**self.cfg.qk_logits.to_dict())

        logits = qk_logits(query_embedding, key_embeddings)
        logits = logits / head_params.temp

        if valid_mask is None:
            valid_mask = jnp.ones_like(logits, dtype=jnp.bool)

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
                min_p=head_params.min_p,
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1)
        return HeadOutput(action_index=action_index, log_prob=log_prob, entropy=entropy)


class PolicyLogitHeadInner(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array):
        return MLP(
            final_kernel_init=nn.initializers.orthogonal(1e-2),
            **self.cfg.logits.to_dict()
        )(x)


class PolicyLogitHead(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(
        self,
        embedding: jax.Array,
        head: HeadOutput,
        valid_mask: jax.Array = None,
        head_params: HeadParams = HeadParams(),
    ):
        logits = PolicyLogitHeadInner(self.cfg)(embedding)
        logits = logits / head_params.temp

        if valid_mask is None:
            valid_mask = jnp.ones_like(logits, dtype=jnp.bool)

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
                min_p=head_params.min_p,
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1)
        return HeadOutput(action_index=action_index, log_prob=log_prob, entropy=entropy)


class ValueLogitHead(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array):
        x = MLP(
            final_kernel_init=nn.initializers.orthogonal(1e-2),
            **self.cfg.logits.to_dict()
        )(x)
        return jnp.tanh(x.squeeze(-1))
