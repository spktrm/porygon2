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
    SRC_GROUP_MASK,
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
    """Within-modality (micro) readout over the flat src x tgt grid, with
    NO parameter shared between slot groups (2026-08-25).

    Every parameter here is indexed by the SRC slot's group (move / switch /
    target — SRC_GROUP_MASK, the same partition the encoder's typed residual
    streams use). A cell's group is a function of its src half alone, so
    "this cell's group" is always well defined, and `local_tgt` is read under
    the CELL's group: the same target token scores differently depending on
    which modality is choosing it ("how good is this target for a move" vs
    "for a switch"), which is the point of separating them.

    What this replaces: one parameter-free dot grid (policy) or one shared
    PointerLogits (advantage), with the three groups distinguished ONLY by a
    scalar. That scalar was the group's entire parameterisation, and on the
    2026-08-25 read of the 84.9k-step run the target group's scalar was still
    BITWISE ZERO on both families — the group had no trained readout at all
    (docs/qva-redesign-step0-reference.md). LESSONS.md section 13 already
    recorded that the Nov-2025 per-modality head beat the flat gram head and
    that flattening it was a known regression; this restores the separation.

    Implementation note: G disjoint projections are one Dense with G*qk
    outputs — attention heads own disjoint output coordinates, so there is
    genuinely no weight sharing between groups, and the per-group select is
    a one-hot contraction over the group axis.

    Two structural properties are preserved exactly:

    FLAT-AT-INIT. `type_scale` is zero-init, so every micro logit is exactly
    0 at init and calculate_hierarchical_prior stays the exact init-policy
    anchor.

    SINGLE-FACTOR GRADIENT. A learned grid behind a zero-init scale is a
    gate x grid PRODUCT, and that product is what stalled: the gate's
    gradient is a random grid's correlation with the residual, the grid's is
    proportional to the gate, and neither moved in 60k steps (gate 0.03-0.06,
    q/k kernels still at lecun init — docs/critic-weakness-analysis.md Step-3
    post-mortem). So each group also carries zero-init `micro_local_src` /
    `micro_local_tgt` routes: ONE zero-init factor over a LIVE input, whose
    gradient is consistent from step 0. The grid supplies interaction terms
    behind the scale; the local routes carry the early signal.

    The per-group rms normalisation (2026-08-25, 473ba77) stays and matters
    MORE with learned kernels, not less: without it the loss pins the PRODUCT
    scale.grid at the band, so growing kernel norms crush the scale (0.026 ->
    1e-4 was measured) while the raw grid grows, the gradient to the scale
    (= the grid) explodes, and the micro route dies mid-run. Dividing by the
    group's stop-grad rms gauges that away, leaving type_scale as the one
    live factor with an O(w) gradient.
    """

    num_logits: int = 1
    qk_size: int = None
    use_bias: bool = True
    qk_layer_norm: bool = True

    @nn.compact
    def __call__(self, action_embeddings: jax.Array) -> jax.Array:
        g = NUM_ACTION_SLOT_GROUPS
        k = self.num_logits
        lead = action_embeddings.shape[:-2]
        n = action_embeddings.shape[-2]
        # (N, G) one-hot of each SRC slot's group: the per-group select.
        group_oh = jax.nn.one_hot(
            jnp.asarray(SRC_GROUP_MASK), g, dtype=jnp.float32
        )

        # Per-group q/k projections -> (..., N, N, G * K) -> (..., N, N, G, K)
        grid = PointerLogits(
            qk_size=self.qk_size,
            num_heads=g * k,
            use_bias=self.use_bias,
            qk_layer_norm=self.qk_layer_norm,
            name="micro_qk",
        )(action_embeddings, action_embeddings)
        grid = grid.reshape(*lead, n, n, g, k).astype(jnp.float32)
        # Select each cell's group by its SRC slot.
        flat = jnp.einsum("...ijgk,ig->...ijk", grid, group_oh)
        flat = flat.reshape(*lead, n * n, k)

        # Per-group rms over the grid cells (f32, eps-guarded, all cells:
        # legality lives downstream), stop-grad so the normaliser is a
        # gauge, not a trainable route.
        flat_group_oh = jax.nn.one_hot(
            jnp.asarray(FLAT_SRC_GROUP_MASK), g, dtype=jnp.float32
        )
        sq = jnp.square(flat)
        group_ms = jnp.einsum("...ck,cg->...gk", sq, flat_group_oh) / jnp.maximum(
            flat_group_oh.sum(axis=0), 1.0
        )[:, None]
        group_rms = jax.lax.stop_gradient(
            jnp.sqrt(group_ms + 1e-12)[..., FLAT_SRC_GROUP_MASK, :]
        )
        normed = flat / group_rms

        type_scale = self.param(
            "type_scale", nn.initializers.zeros_init(), (g, k)
        ).astype(jnp.float32)
        out = normed * type_scale[FLAT_SRC_GROUP_MASK, :]

        # Single-factor per-group local routes over live inputs.
        local_src = (
            nn.Dense(
                g * k,
                use_bias=False,
                kernel_init=nn.initializers.zeros,
                name="micro_local_src",
            )(action_embeddings)
            .reshape(*lead, n, g, k)
            .astype(jnp.float32)
        )
        local_tgt = (
            nn.Dense(
                g * k,
                use_bias=False,
                kernel_init=nn.initializers.zeros,
                name="micro_local_tgt",
            )(action_embeddings)
            .reshape(*lead, n, g, k)
            .astype(jnp.float32)
        )
        src_term = jnp.einsum("...igk,ig->...ik", local_src, group_oh)
        # The TGT token read under the CELL's group, i.e. the src's.
        tgt_term = jnp.einsum("...jgk,ig->...ijk", local_tgt, group_oh)
        local = (src_term[..., :, None, :] + tgt_term).reshape(*lead, n * n, k)

        out = out + local
        return out.squeeze(-1) if k == 1 else out


class ActionAdapter(nn.Module):
    """Owned residual MLP between the trunk's action embeddings and one
    head family's MacroMicroHead (2026-08-20).

    The policy's micro readout is a parameter-free dot grid straight off
    the typed trunk streams, so before this adapter existed any OTHER
    head training on the same embeddings (the Q head's CE) reshaped the
    live policy geometry directly. Each head family now reads the trunk
    through its own instance — the policy's plain, the Q head's with the
    projected value conditioning concatenated in (the rung's information
    set has to reach every CELL, not just the macro level, or Q_all and
    Q_private could only differ per modality) — so the trunk still
    receives every head's gradient but the head-specific geometry is
    decoupled.

    ZERO-INIT output layer: the adapter is an exact identity at init, so
    calculate_hierarchical_prior stays the exact init-policy anchor and a
    fresh-initted adapter perturbs nothing — it learns from zero.
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array, cond: jax.Array | None = None) -> jax.Array:
        hidden = x
        if cond is not None:
            hidden = jnp.concatenate(
                [
                    x,
                    jnp.broadcast_to(
                        cond[..., None, :], (*x.shape[:-1], cond.shape[-1])
                    ),
                ],
                axis=-1,
            )
        hidden = MLP(**self.cfg.mlp.to_dict())(hidden)
        return x + nn.Dense(
            x.shape[-1], kernel_init=nn.initializers.zeros, dtype=x.dtype
        )(hidden)


class MacroHead(nn.Module):
    """Modality-level (macro) readout from per-modality pooled src slots.

    One learned query per modality attention-pools that modality's live
    src-slot embeddings — each modality's srcs live in exactly one typed
    trunk stream (move+wildcard → move, switch → switch, other → target),
    so the pools read typed spaces — then that modality's OWN MLP and OWN
    zero-initialised output layer map the pooled vector to out_features
    outputs (default 1: the modality logits, squeezed).

    No parameter is shared between modalities (2026-08-25). Before that the
    MLP and out layer were shared and only the pooling query differed, so
    every modality's level was read by one function of a pooled vector.
    Owning the modality contest with dedicated parameters keeps the
    (per-modality shift-invariant) micro gradient from moving the macro
    decision through dot-logit magnitude. Zero output init keeps macro
    outputs exactly zero at init, so calculate_hierarchical_prior remains
    the exact init-policy anchor (and a fresh-initted Q macro starts as a
    pure level-free micro readout).

    Optional cond: a pooled per-state conditioning vector, broadcast and
    concatenated into the MLP input (the Q rungs' information set); the
    policy instantiation passes none, so its param tree and math are
    unchanged. Handles arbitrary leading batch dims — the policy calls it
    per timestep under vmap, the Q head calls it with T leading.
    """

    cfg: ConfigDict
    out_features: int = 1

    @nn.compact
    def __call__(
        self,
        src_embeddings: jax.Array,
        src_valid: jax.Array,
        cond: jax.Array | None = None,
    ):
        queries = self.param(
            "modality_queries",
            nn.initializers.lecun_normal(),
            (NUM_MODALITY_FEATURES, src_embeddings.shape[-1]),
        ).astype(src_embeddings.dtype)
        queries = jnp.broadcast_to(
            queries, (*src_embeddings.shape[:-2], *queries.shape)
        )

        attn_logits = PointerLogits(**self.cfg.qk_logits.to_dict())(
            queries, src_embeddings
        ).squeeze(-1)

        valid_src_per_modality = (
            SRC_MODALITY_MASK[None, :] == jnp.arange(NUM_MODALITY_FEATURES)[:, None]
        ) & src_valid[..., None, :]
        attn = jax.nn.softmax(
            jnp.where(valid_src_per_modality, attn_logits, -1e9), axis=-1
        )
        # A modality with no live src would otherwise softmax a uniform
        # row over every slot (invalid ones included); pool nothing.
        attn = jnp.where(valid_src_per_modality.any(axis=-1, keepdims=True), attn, 0.0)
        pooled = attn @ src_embeddings

        if cond is not None:
            pooled = jnp.concatenate(
                [
                    pooled,
                    jnp.broadcast_to(
                        cond[..., None, :], (*pooled.shape[:-1], cond.shape[-1])
                    ),
                ],
                axis=-1,
            )
        # PER-MODALITY MLP and output layer (2026-08-25): each modality gets
        # its own parameters, not a shared MLP distinguished only by the
        # learned query above. M is 5, so an explicit loop is clearer than a
        # vmap over a size-5 axis and costs the same FLOPs (each MLP now runs
        # on one row instead of one MLP running on five).
        per_modality = [
            nn.Dense(
                self.out_features,
                kernel_init=nn.initializers.zeros,
                dtype=pooled.dtype,
                name=f"out_{m}",
            )(MLP(**self.cfg.mlp.to_dict(), name=f"mlp_{m}")(pooled[..., m, :]))
            for m in range(NUM_MODALITY_FEATURES)
        ]
        logits = jnp.stack(per_modality, axis=-2)
        # Modalities with no live src pool zeros and read the out bias;
        # callers still mask them (legal_log_policy for the policy, the
        # flat action mask for the Q grid).
        return logits.squeeze(-1) if self.out_features == 1 else logits


class MacroMicroHead(nn.Module):
    """The shared two-level action-grid readout (2026-08-20): one module
    owns the macro/micro scoring of the flat src x tgt grid for BOTH the
    policy and the Q critic — the composition the Nov-2025 competition
    result and the 2026-08-17 policy head paid for, now applied to any
    head family that scores actions.

    cfg.num_logits is the output width PER CELL/MODALITY (1 = the
    policy's scalar logits; the Q critic passes the categorical bin
    count). micro scores every grid cell through MicroHead — per-slot-group
    q/k projections, per-group zero-init scale and per-group zero-init
    local routes (2026-08-25: the 'dot' / 'pointer' split is gone, both
    families use the one readout). macro scores each modality via
    MacroHead over per-modality pooled src slots at the same width, with
    that modality's own MLP and zero-init out layer; optional cond threads
    to those MLPs (the critic's information set). Every output is EXACTLY
    0 at init, so each instantiation starts flat — the policy at its
    hierarchical prior, the advantage at 0 everywhere.

    Returns the (macro, micro) pair RAW — composition belongs to the
    caller, because that is where the two families genuinely differ: the
    policy multiplies softmaxes in log space (macro owns the modality
    contest, micro is per-modality shift-invariant), the Q readout adds
    macro onto the legality-centred micro (compose_action_grid: dueling
    identifiability, absolute magnitudes).
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(
        self,
        action_embeddings: jax.Array,
        src_valid: jax.Array,
        cond: jax.Array | None = None,
    ):
        num_logits = self.cfg.get("num_logits", 1)
        micro = MicroHead(
            num_logits=num_logits,
            **self.cfg.get("micro_qk", ConfigDict()).to_dict(),
            name="micro",
        )(action_embeddings)
        macro = MacroHead(self.cfg.macro, out_features=num_logits, name="macro")(
            action_embeddings, src_valid, cond=cond
        )
        return macro, micro


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


class ActionScores(NamedTuple):
    """One action-axis readout's outputs.

    `logits` is the composed src x tgt grid. `macro` / `micro` are the FREE
    per-level logits before composition — NeuRD differentiates against these,
    because the composed grid is already a normalised log-policy and
    differentiating through the normalisations adds a pi-prefactored
    cross-term (LESSONS.md section 3, tests/test_neurd_loss.py pins it).
    """

    logits: jax.Array
    macro: jax.Array
    micro: jax.Array
    flat_valid: jax.Array


class ActionScoreHead(nn.Module):
    """THE action-axis readout: adapter -> macro/micro -> grid composition.

    One module for both families, differing only in `reduce` — the policy and
    the advantage score the same grid and disagree only about how macro and
    micro combine:

      reduce="logsumexp"  the hierarchical log-policy. micro becomes a
                          within-modality log-softmax (per-modality
                          shift-invariant) and macro, passed as a legal
                          log-softmax over modalities, owns the modality
                          contest — so micro confidence can never move the
                          modality decision through logit magnitude.
      reduce="mean"       the modality-centred advantage. micro is centred
                          over each modality's LEGAL cells, so it carries
                          only within-modality shape and each modality's
                          LEVEL must flow through the low-dimensional macro
                          route ("is switching better here").

    Before 2026-08-25 this sequence was open-coded three times — the singles
    policy, the doubles per-stage scorer and the Q head each rebuilt
    adapter -> src_valid -> macro/micro -> compose by hand, and drifted.

    `cond` is an optional per-state conditioning vector (the critic's value
    embedding): projected and normed here, then threaded BOTH into the
    adapter and into the macro MLP, so the information set reaches every
    CELL rather than only the modality level.
    """

    cfg: ConfigDict
    reduce: str

    @nn.compact
    def __call__(
        self,
        action_embeddings: jax.Array,
        valid_mask: jax.Array,
        cond: jax.Array | None = None,
        temp: float = 1.0,
    ) -> ActionScores:
        flat_valid = valid_mask.reshape(*valid_mask.shape[:-2], -1)
        # A src slot is actionable iff its row has any valid tgt cell.
        src_valid = valid_mask.any(axis=-1)

        c = None
        if cond is not None:
            # Projection/norm in f32 (flax default), cast back so the bf16
            # grid tensors stay bf16.
            c = nn.LayerNorm(name="cond_norm")(
                nn.Dense(action_embeddings.shape[-1], name="cond_proj")(cond)
            ).astype(action_embeddings.dtype)

        adapted = ActionAdapter(self.cfg.adapter, name="adapter")(
            action_embeddings, cond=c
        )
        macro, micro = MacroMicroHead(self.cfg.macro_micro, name="macro_micro")(
            adapted, src_valid, cond=c
        )
        # num_logits 1: MacroHead already squeezes its single output; the
        # pointer grid keeps its head axis — drop it so both micro kinds
        # ride the one scalar composition path.
        if micro.ndim == flat_valid.ndim + 1:
            micro = micro.squeeze(-1)
        if macro.ndim == flat_valid.ndim + 1:
            macro = macro.squeeze(-1)

        macro = macro.astype(jnp.float32) / temp
        micro = micro.astype(jnp.float32) / temp

        composed_macro = macro
        if self.reduce == "logsumexp":
            modality_oh = FLAT_MODALITY_MASK[:, None] == jnp.arange(
                NUM_MODALITY_FEATURES
            )
            counts = (flat_valid[..., None] & modality_oh).sum(axis=-2)
            composed_macro = legal_log_policy(macro, counts > 0)

        logits = compose_action_grid(
            composed_macro, micro, flat_valid, reduce=self.reduce
        )
        return ActionScores(
            logits=logits, macro=macro, micro=micro, flat_valid=flat_valid
        )


def compose_q(
    value: jax.Array,
    advantage_raw: jax.Array,
    log_policy: jax.Array,
    legal_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """The Q = V + A identity, written once.

        A(s, a) = A_raw(s, a) - E_sg(pi)[A_raw(s, .)]
        Q(s, a) = sg(V(s)) + A(s, a)

    Returns (advantage, q) over legal cells, zero elsewhere, all f32.

    Two stop-gradients, both load-bearing:

    sg(V) closes the STATE route. Taken-cell supervision is satisfiable by a
    state-only function (the Step-6 verdict in docs/critic-weakness-analysis.md
    — the head fit the label to the label-entropy floor in <800 steps while
    within-state action variance FELL 5x), so every state-level degree of
    freedom must sit in V where this loss cannot reach it. What is left for
    the advantage head to explain is the action axis alone.

    sg(pi) stops the Q loss steering the POLICY through the centring term.

    The pi-weighting here is required, and is deliberately NOT the uniform
    centring compose_action_grid applies within each modality: pi-weighting is
    what makes E_pi[Q] = V exactly, which is what makes V the correct NeuRD
    baseline. Doing it at the micro tier instead would put a mass prefactor
    back on starved cells, which docs/entropy-gradient-pressure.md shows can
    never restore a dead modality.

    Note what the centring can and cannot do: it conditions the head and pins
    the level, but it cannot remove a MODALITY offset (a -0.1 on switches at
    pi(switch) = 0.01 sums to ~0). Unclipped — the Huber loss sees the raw
    sum, the policy clips to the reward support.
    """
    adv = advantage_raw.astype(jnp.float32)
    pi = jax.lax.stop_gradient(jnp.exp(log_policy.astype(jnp.float32))) * legal_mask
    pi = pi / jnp.maximum(pi.sum(axis=-1, keepdims=True), 1e-8)
    baseline = (pi * jnp.where(legal_mask, adv, 0.0)).sum(axis=-1, keepdims=True)
    advantage = jnp.where(legal_mask, adv - baseline, 0.0)
    v = jax.lax.stop_gradient(value.astype(jnp.float32))
    q = jnp.where(legal_mask, v[..., None] + advantage, 0.0)
    return advantage, q


def compose_action_grid(
    macro: jax.Array, micro: jax.Array, flat_valid_mask: jax.Array, reduce: str
) -> jax.Array:
    """The one macro/micro grid composition both head families share:

      out(cell) = macro[modality(cell)]
                  + micro(cell) − reduce_modality(micro over LEGAL cells)

    reduce="logsumexp" is the policy's hierarchical log-policy: the
    per-modality logsumexp turns micro into a within-modality log-softmax
    (shift-invariant per modality), and macro — passed as the legal
    log-softmax over modalities, micro pre-divided by temp — owns the
    modality contest. Callers mask invalid cells afterwards.

    reduce="mean" is the value composition (per output bin): centring
    micro over each modality's LEGAL cells makes the decomposition
    identifiable (dueling-style) — micro carries only within-modality
    shape, so the modality's level must flow through the macro readout, a
    low-dimensional path that generalises across states, i.e. an explicit
    parameter route for "is switching better here" (the flat-grid
    predecessor made the head express the switch/move contest cell-by-cell
    through one shared bilinear). Legal cells only, because invalid cells
    never receive CE gradient and their drift must not leak into live
    values.

    micro is (..., N*N) scalar (policy) or (..., N*N, K) per-bin (Q), with
    macro (..., M) / (..., M, K) to match; handles arbitrary leading batch
    dims.
    """
    scalar = micro.ndim == flat_valid_mask.ndim
    m = micro[..., None] if scalar else micro
    mac = macro[..., None] if scalar else macro
    modality_oh = FLAT_MODALITY_MASK[:, None] == jnp.arange(NUM_MODALITY_FEATURES)
    valid_per_modality = flat_valid_mask[..., None] & modality_oh
    if reduce == "logsumexp":
        stat = nn.logsumexp(
            jnp.where(valid_per_modality[..., None], m[..., :, None, :], -1e9),
            axis=-3,
        )
        # An empty modality's stat is -1e9; subtracting it would leave
        # +1e9 in cells the caller masks anyway -- keep the intermediate
        # finite-small instead of one mask-drop from poisoning a softmax.
        stat = jnp.where(valid_per_modality.any(axis=-2)[..., None], stat, 0.0)
    elif reduce == "mean":
        weights = valid_per_modality.astype(m.dtype)
        counts = jnp.maximum(weights.sum(axis=-2), 1.0)
        stat = jnp.einsum("...cm,...ck->...mk", weights, m) / counts[..., None]
    else:
        raise ValueError(f"unknown reduce: {reduce}")
    out = m - stat[..., FLAT_MODALITY_MASK, :] + mac[..., FLAT_MODALITY_MASK, :]
    return out.squeeze(-1) if scalar else out


# Alive-mon differential support: margins -6..+6, matching the offline
# critic's 13-bin distributional target.
NUM_MARGIN_BINS = 13


class CategoricalValueLogitHead(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, embedding: jax.Array):
        # f32 from the head outwards (2026-08-24): a handful of bins, and
        # the main critic's CE, the v-trace bootstrap probs and the
        # expectation all read them -- the 1.0-weighted head was the one
        # rung still paying bf16 while the ladder heads were cast f32.
        logits = MLP(**self.cfg.mlp.to_dict())(embedding).astype(jnp.float32)

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
        # f32 out: scalar regression targets (reg value, builder ev /
        # conditional entropy) are all f32-MSE consumers.
        x = MLP(**self.cfg.mlp.to_dict())(x).astype(jnp.float32)
        if getattr(self.cfg, "output_activation", None) is not None:
            x = self.cfg.output_activation(x)
        return RegressionValueHeadOutput(logits=x.squeeze(-1))
