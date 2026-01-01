from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.interfaces import (
    CategoricalValueHeadOutput,
    PolicyHeadOutput,
    RegressionValueHeadOutput,
)
from rl.model.modules import MLP, PointerLogits, Resnet
from rl.model.utils import legal_log_policy, legal_policy


class HeadParams(NamedTuple):
    temp: float = 1.0
    min_p: float = 0.0


def sample_categorical(log_probs: jax.Array, rng_key: jax.Array, min_p: float = 0):
    # Fast path: no min_p adjustment, sample directly from logits.
    if min_p <= 0.0:
        return jax.random.categorical(rng_key, log_probs, axis=-1)

    # Convert to probs, clamp, renormalize, then go back to log-space for sampling.
    probs = jnp.exp(log_probs)
    probs = jnp.where(probs >= min_p * probs.max(), probs, 0)
    probs = probs / probs.sum(axis=-1, keepdims=True)

    # Avoid log(0) just in case of numerical edge cases
    log_probs = jnp.log(probs)
    log_probs = jnp.nan_to_num(log_probs, neginf=jnp.finfo(log_probs.dtype).min)

    return jax.random.categorical(rng_key, log_probs, axis=-1)


class PolicyQKHead(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(
        self,
        query_embedding: jax.Array,
        key_embeddings: jax.Array,
        head: PolicyHeadOutput,
        valid_mask: jax.Array = None,
        head_params: HeadParams = HeadParams(),
    ):
        resnet = Resnet(**self.cfg.resnet.to_dict())
        qk_logits = PointerLogits(**self.cfg.qk_logits.to_dict())

        logits = qk_logits(resnet(query_embedding)[None], key_embeddings).squeeze(0)
        logits = logits * (1 / (head_params.temp + 1e-8))

        if valid_mask is None:
            valid_mask = jnp.ones_like(logits, dtype=jnp.bool)

        log_policy = legal_log_policy(logits, valid_mask)
        policy = legal_policy(logits, valid_mask)
        entropy = -jnp.sum(policy * log_policy, axis=-1)

        train = self.cfg.get("train", False)
        if train:
            action_index = head.action_index
        else:
            action_index = sample_categorical(
                jnp.where(valid_mask, log_policy, jnp.finfo(log_policy.dtype).min),
                self.make_rng("sampling"),
                min_p=head_params.min_p,
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1)
        return PolicyHeadOutput(
            action_index=action_index,
            log_prob=log_prob,
            entropy=entropy,
            log_policy=log_policy,
        )


class PolicyLogitHeadInner(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array):
        logits = MLP(
            final_kernel_init=nn.initializers.orthogonal(1e-2),
            **self.cfg.logits.to_dict(),
        )
        return logits(x)


class PolicyLogitHead(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(
        self,
        embedding: jax.Array,
        head: PolicyHeadOutput,
        valid_mask: jax.Array = None,
        head_params: HeadParams = HeadParams(),
    ):
        logits = PolicyLogitHeadInner(self.cfg)(embedding)
        logits = logits / head_params.temp

        if valid_mask is None:
            valid_mask = jnp.ones_like(logits, dtype=jnp.bool)

        log_policy = legal_log_policy(logits, valid_mask)
        policy = legal_policy(logits, valid_mask)
        entropy = -jnp.sum(policy * log_policy, axis=-1)

        train = self.cfg.get("train", False)
        if train:
            action_index = head.action_index
        else:
            action_index = sample_categorical(
                jnp.where(valid_mask, log_policy, jnp.finfo(log_policy.dtype).min),
                self.make_rng("sampling"),
                min_p=head_params.min_p,
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1)
        return PolicyHeadOutput(
            action_index=action_index,
            log_prob=log_prob,
            entropy=entropy,
            log_policy=log_policy,
        )


class CategoricalValueLogitHead(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array):
        resnet = Resnet(**self.cfg.resnet.to_dict())
        logits = MLP(**self.cfg.logits.to_dict())

        x = resnet(x)
        x = logits(x)

        log_probs = nn.log_softmax(x, axis=-1)
        probs = jnp.exp(log_probs)
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        expectation = probs @ self.cfg.category_values

        return CategoricalValueHeadOutput(
            logits=x, log_probs=log_probs, entropy=entropy, expectation=expectation
        )


class RegressionValueLogitHead(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array):
        resnet = Resnet(**self.cfg.resnet.to_dict())
        logits = MLP(**self.cfg.logits.to_dict())

        x = resnet(x)
        x = logits(x)

        return RegressionValueHeadOutput(logits=x.squeeze(-1))
