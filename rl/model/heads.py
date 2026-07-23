from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.data import (
    FLAT_MODALITY_MASK,
    NUM_MODALITY_FEATURES,
    SRC_MODALITY_MASK,
)
from rl.environment.interfaces import (
    CategoricalValueHeadOutput,
    PolicyHeadOutput,
    RegressionValueHeadOutput,
)
from rl.model.modules import MLP, PointerLogits
from rl.model.utils import legal_log_policy, legal_policy


class HeadParams(NamedTuple):
    temp: float = 1.0


class PolicyMetrics(NamedTuple):
    policy: jax.Array
    log_policy: jax.Array
    entropy: jax.Array
    normalized_entropy: jax.Array
    magnet_kl: jax.Array


def calculate_hierarchical_prior(flat_valid_mask: jax.Array) -> jax.Array:
    """Uniform over valid modalities times uniform within each modality.

    This is the init policy of the hierarchically composed action head
    (the macro head's zero-initialised output layer and the zero square
    logits both give zero logits at init), so it is the consistent anchor
    for the magnet/exploration KLs. Supports any leading batch dims.
    """
    modality_oh = FLAT_MODALITY_MASK[:, None] == jnp.arange(NUM_MODALITY_FEATURES)
    valid_per_modality = flat_valid_mask[..., :, None] & modality_oh
    modality_counts = valid_per_modality.sum(axis=-2)
    num_valid_modalities = jnp.maximum(
        (modality_counts > 0).sum(axis=-1, keepdims=True), 1
    )
    counts_per_cell = jnp.maximum(modality_counts, 1)[..., FLAT_MODALITY_MASK]
    prior = 1.0 / (num_valid_modalities * counts_per_cell)
    return jnp.where(flat_valid_mask, prior, 0.0)


def compute_policy_metrics(
    logits: jax.Array, valid_mask: jax.Array, prior: jax.Array = None
):
    """
    Computes standard policy distributions, entropy, normalized entropy,
    and the KL divergence (exploration magnet) penalty.
    """
    # 1. Distill policy distributions
    log_policy = legal_log_policy(logits, valid_mask)
    policy = legal_policy(logits, valid_mask)

    # 2. Base Entropy
    entropy = -jnp.sum(policy * log_policy, axis=-1)

    # 3. Normalized Entropy
    valid_sum = valid_mask.sum(axis=-1)
    safe_log_sum = jnp.maximum(valid_sum, 2)
    log_factor = 1.0 / jnp.log(safe_log_sum).astype(entropy.dtype)
    entropy_scale = jnp.where(valid_sum <= 1, 1.0, log_factor)
    normalized_entropy = entropy * entropy_scale

    # 4. Exploration KL (magnet_kl)
    if prior is None:
        valid_sum_expanded = jnp.maximum(valid_sum[..., None], 1)
        prior = jnp.where(valid_mask, 1.0 / valid_sum_expanded, 0.0)
    prior = prior.astype(log_policy.dtype)

    # Safe log calculation
    safe_prior = jnp.where(valid_mask, prior, 1e-9)
    log_prior = jnp.where(valid_mask, jnp.log(safe_prior), 0.0)

    # D_KL(Policy || Prior)
    magnet_kl = policy * (log_policy - log_prior)
    magnet_kl = jnp.where(valid_mask, magnet_kl, 0.0).sum(axis=-1)

    return PolicyMetrics(
        policy=policy,
        log_policy=log_policy,
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        magnet_kl=magnet_kl,
    )


def sample_categorical(logits: jax.Array, rng_key: jax.Array):
    return jax.random.categorical(rng_key, logits, axis=-1)


class MacroHead(nn.Module):
    """Modality-level (macro) logits from per-modality pooled src slots.

    One learned query per modality attention-pools that modality's live
    src-slot embeddings, then a shared MLP with a zero-initialised output
    layer maps each pooled vector to a scalar. Owning the modality contest
    with dedicated parameters keeps the (per-modality shift-invariant)
    micro gradient from moving the macro decision through gram-logit
    magnitude, which the old mean-pool of the square logits allowed. Zero
    output init keeps macro logits exactly zero at init, so
    calculate_hierarchical_prior remains the exact init-policy anchor.
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(self, src_embeddings: jax.Array, src_valid: jax.Array):
        queries = self.param(
            "modality_queries",
            nn.initializers.lecun_normal(),
            (NUM_MODALITY_FEATURES, src_embeddings.shape[-1]),
        ).astype(src_embeddings.dtype)

        attn_logits = PointerLogits(**self.cfg.qk_logits.to_dict())(
            queries, src_embeddings
        ).squeeze(-1)

        valid_src_per_modality = (
            SRC_MODALITY_MASK[None, :] == jnp.arange(NUM_MODALITY_FEATURES)[:, None]
        ) & src_valid[None, :]
        attn = jax.nn.softmax(
            jnp.where(valid_src_per_modality, attn_logits, -1e9), axis=-1
        )
        pooled = attn @ src_embeddings

        hidden = MLP(**self.cfg.mlp.to_dict())(pooled)
        logits = nn.Dense(1, kernel_init=nn.initializers.zeros, dtype=hidden.dtype)(
            hidden
        )
        # Modalities with no live src pool an arbitrary mixture; callers
        # must mask them out via legal_log_policy(macro_logits, valid).
        return logits.squeeze(-1)


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

        logits = qk_logits(query_embedding[None], key_embeddings).reshape(
            key_embeddings.shape[0]
        )
        logits = logits * (1 / (head_params.temp + 1e-8))

        if valid_mask is None:
            valid_mask = jnp.ones_like(logits, dtype=jnp.bool)

        policy_metrics = compute_policy_metrics(
            logits=logits, valid_mask=valid_mask, prior=prior
        )

        train = self.cfg.get("train", False)
        if train:
            action_index = head.action_index
        else:
            action_index = sample_categorical(
                jnp.where(valid_mask, logits, jnp.finfo(logits.dtype).min),
                self.make_rng("sampling"),
            )

        log_prob = jnp.take(
            policy_metrics.log_policy, action_index, axis=-1, mode="clip"
        )

        return PolicyHeadOutput(
            action_index=action_index.reshape(policy_metrics.entropy.shape),
            log_prob=log_prob.reshape(policy_metrics.entropy.shape),
            entropy=policy_metrics.entropy,
            normalized_entropy=policy_metrics.normalized_entropy,
            log_policy=policy_metrics.log_policy,
            magnet_kl=policy_metrics.magnet_kl,
        )


class CategoricalValueLogitHead(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, embedding: jax.Array):
        logits = MLP(**self.cfg.mlp.to_dict())(embedding)

        log_probs = nn.log_softmax(logits, axis=-1)
        probs = jnp.exp(log_probs)
        entropy = -jnp.sum(probs * log_probs, axis=-1)

        values = self.cfg.category_values.astype(logits.dtype)
        expectation = probs @ values

        mean_logit = jnp.mean(logits, axis=-1, keepdims=True)
        l2_norm = jnp.linalg.norm(logits - mean_logit, axis=-1)

        return CategoricalValueHeadOutput(
            logits=logits,
            log_probs=log_probs,
            entropy=entropy,
            expectation=expectation,
            l2_norm=l2_norm,
        )


class RegressionValueLogitHead(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array):
        x = MLP(**self.cfg.mlp.to_dict())(x)
        if getattr(self.cfg, "output_activation", None) is not None:
            x = self.cfg.output_activation(x)
        return RegressionValueHeadOutput(logits=x.squeeze(-1))
