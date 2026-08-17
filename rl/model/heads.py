from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.data import (
    FLAT_MODALITY_MASK,
    FLAT_SRC_GROUP_MASK,
    NUM_ACTION_SLOT_GROUPS,
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


class MicroHead(nn.Module):
    """Parameter-less within-modality (micro) readout over the typed
    trunk streams (2026-08-17, replacing PerModalityPolicyHead).

    The role and modality depth that previously lived in per-modality
    src/tgt MLP stacks now lives in the round trunk: move / switch /
    target slots travel every round as separate residual streams with
    their own gates and out-norms, so by the time embeddings reach this
    head the src and tgt halves of a grid cell are (structural diagonal
    cells aside) vectors from different typed spaces — no slot does query
    and key duty through one shared projection, the pathology that
    retired the original parameter-free gram head. Diagonal cells (pass /
    default) score a squared norm, which is harmless: they are OTHER-
    modality singletons whose within-modality softmax is degenerate, so
    the modality head alone decides them.

    The readout is a scaled dot-product grid times a per-slot-group
    ZERO-INIT scale. Zero init keeps every micro logit exactly 0 at model
    init, preserving calculate_hierarchical_prior as the exact
    init-policy anchor (the job the old modality_scale's ones-init did),
    and each group owns its sharpness through its scalar plus its
    stream's residual magnitudes — the per-modality logit-scale probe
    showed distinct temperatures emerge when allowed (wildcard > move >
    switch). Deliberately NO layer norm on either side: norming the dot
    would freeze the logit scale the typed streams are supposed to own.
    The downstream within-modality logsumexp removes per-modality shifts,
    so these scales control sharpness only — the modality contest belongs
    to MacroHead.
    """

    @nn.compact
    def __call__(self, action_embeddings: jax.Array) -> jax.Array:
        inv_sqrt_d = action_embeddings.shape[-1] ** -0.5
        logits = (action_embeddings @ action_embeddings.T) * inv_sqrt_d
        type_scale = self.param(
            "type_scale", nn.initializers.zeros_init(), (NUM_ACTION_SLOT_GROUPS,)
        ).astype(logits.dtype)
        return logits.reshape(-1) * type_scale[FLAT_SRC_GROUP_MASK]


class MacroHead(nn.Module):
    """Modality-level (macro) logits from per-modality pooled src slots.

    One learned query per modality attention-pools that modality's live
    src-slot embeddings — each modality's srcs live in exactly one typed
    trunk stream (move+wildcard → move, switch → switch, other → target),
    so the pools read typed spaces — then a shared MLP with a
    zero-initialised output layer maps each pooled vector to a scalar.
    Owning the modality contest with dedicated parameters keeps the
    (per-modality shift-invariant) micro gradient from moving the macro
    decision through dot-logit magnitude. Zero output init keeps macro
    logits exactly zero at init, so calculate_hierarchical_prior remains
    the exact init-policy anchor.
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


class SlotConditioning(nn.Module):
    """Injects slot 1's chosen action into the action embeddings for
    slot 2's head pass (doubles).

    The trunk runs ONCE per turn; conditioning the second active mon's
    decision on the first happens here at head level: the chosen action's
    src/tgt embeddings (already computed by the same trunk forward) map
    through a zero-initialised projection onto every action embedding, so
    at init slot 2's policy is exactly the unconditioned one and the
    conditioning pathway is learned from zero. This replaces the old
    two-request scheme where the second decision re-encoded the full state
    with prev-action tokens (a second trunk pass per doubles turn).
    """

    @nn.compact
    def __call__(
        self, action_embeddings: jax.Array, src_index: jax.Array, tgt_index: jax.Array
    ) -> jax.Array:
        chosen = jnp.concatenate(
            [action_embeddings[src_index], action_embeddings[tgt_index]], axis=-1
        )
        delta = nn.Dense(
            action_embeddings.shape[-1],
            kernel_init=nn.initializers.zeros,
            dtype=action_embeddings.dtype,
        )(chosen)
        return action_embeddings + delta[None, :]


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


class QValueHead(nn.Module):
    """All-action categorical Q readout, conditioned on a pooled value
    embedding (docs/q-critic-plan.md).

    Reads the SAME action-slot embeddings the policy heads read, plus a
    pooled per-state conditioning vector that sets the head's information
    set: the ONE bound instance is called twice — with the privileged
    value_all embedding (Q_all, feeds the Retrace recursion) and with the
    private value embedding (Q_private, the policy's information set) —
    so all params are shared across rungs, the same calibration-coupling
    trick as v_head's private readout. The conditioning enters by concat
    into each role's residual MLP; scores every src x tgt grid cell with
    one logit per CAT_VF_SUPPORT bin via a PointerLogits whose num_heads
    is the bin count. Learner-only — gated in
    Porygon2PlayerModel.__call__; never sampled from, so it has no
    interaction with acting or replay. Handles arbitrary leading batch
    dims (PointerLogits/MLP are einsum-based).
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(self, action_embeddings: jax.Array, cond: jax.Array) -> jax.Array:
        src = action_embeddings
        tgt = action_embeddings
        entity_size = action_embeddings.shape[-1]
        c = nn.Dense(entity_size, name="cond_proj", dtype=cond.dtype)(cond)
        c = nn.LayerNorm(name="cond_norm", dtype=c.dtype)(c)
        cb = jnp.broadcast_to(c[..., None, :], (*src.shape[:-1], entity_size))
        for block in range(self.cfg.num_blocks):
            src = src + MLP(**self.cfg.src_mlp.to_dict(), name=f"src_mlp_b{block}")(
                jnp.concatenate([src, cb], axis=-1)
            )
            tgt = tgt + MLP(**self.cfg.tgt_mlp.to_dict(), name=f"tgt_mlp_b{block}")(
                jnp.concatenate([tgt, cb], axis=-1)
            )
        # (..., N, N, n_bins) -> (..., N * N, n_bins): flat cell order
        # matches the policy head's flat action indexing.
        logits = PointerLogits(**self.cfg.qk_logits.to_dict())(src, tgt)
        return logits.reshape(*logits.shape[:-3], -1, logits.shape[-1])


# Alive-mon differential support: margins -6..+6, matching the offline
# critic's 13-bin distributional target.
NUM_MARGIN_BINS = 13


def margin_win_mass(logits: jax.Array) -> jax.Array:
    """P(win) − P(loss) from 13-bin margin logits (last axis)."""
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    half = NUM_MARGIN_BINS // 2
    return probs[..., half + 1 :].sum(-1) - probs[..., :half].sum(-1)


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


class MultiLambdaValueLogitHead(nn.Module):
    """Learner-only categorical value logits for K auxiliary lambdas.

    Output (..., K, n_bins): row k is trained by CE against the gamma=1
    v-trace distribution target built with player_aux_lambdas[k]
    (rl/online/targets.py). With terminal-only reward every row estimates
    the same win probability from a different bias/variance target
    construction — lambda=1 is the Monte Carlo anchor (a gamma spectrum
    would degenerate here: gamma^45 kills the signal). Representation
    shaping only — the policy's advantages read the main v_head; these
    rows never feed the actor loss.
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(self, embedding: jax.Array):
        logits = MLP(**self.cfg.mlp.to_dict())(embedding)
        return logits.reshape(*logits.shape[:-1], self.cfg.num_heads, -1)


class RegressionValueLogitHead(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array):
        x = MLP(**self.cfg.mlp.to_dict())(x)
        if getattr(self.cfg, "output_activation", None) is not None:
            x = self.cfg.output_activation(x)
        return RegressionValueHeadOutput(logits=x.squeeze(-1))
