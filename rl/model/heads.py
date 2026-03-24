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
from rl.model.modules import PointerLogits
from rl.model.utils import legal_log_policy, legal_policy


class HeadParams(NamedTuple):
    temp: float = 1.0


def sample_categorical(logits: jax.Array, rng_key: jax.Array):
    return jax.random.categorical(rng_key, logits, axis=-1)


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
        prior: jax.Array = None,
    ):
        qk_logits = PointerLogits(**self.cfg.qk_logits.to_dict())

        logits = qk_logits(query_embedding[None], key_embeddings).squeeze(0)
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
                jnp.where(valid_mask, logits, jnp.finfo(logits.dtype).min),
                self.make_rng("sampling"),
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1, mode="clip")

        valid_sum = valid_mask.sum(axis=-1)
        log_factor = 1 / jnp.log(valid_sum).astype(entropy.dtype)
        entropy_scale = jnp.where(valid_sum <= 1, 1, log_factor)

        if prior is None:
            prior = jnp.ones_like(logits) / logits.shape[-1]

        prior = prior.astype(logits.dtype)
        kl_prior = prior * (jnp.where(prior == 0, 0, jnp.log(prior)) - log_policy)
        kl_prior = kl_prior.sum(axis=-1)

        return PolicyHeadOutput(
            action_index=action_index.reshape(entropy.shape),
            log_prob=log_prob.reshape(entropy.shape),
            entropy=entropy,
            normalized_entropy=entropy * entropy_scale,
            log_policy=log_policy,
            kl_prior=kl_prior,
        )


class CategoricalValueLogitHead(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array):
        x = nn.Dense(**self.cfg.logits.to_dict(), dtype=x.dtype)(x)

        log_probs = nn.log_softmax(x, axis=-1)
        probs = jnp.exp(log_probs)
        entropy = -jnp.sum(probs * log_probs, axis=-1)

        values = self.cfg.category_values.astype(x.dtype)
        expectation = probs @ values

        return CategoricalValueHeadOutput(
            logits=x, log_probs=log_probs, entropy=entropy, expectation=expectation
        )


class RegressionValueLogitHead(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array):
        x = nn.Dense(**self.cfg.logits.to_dict(), dtype=x.dtype)(x)
        return RegressionValueHeadOutput(logits=x.squeeze(-1))
