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

    The readout is a dot-product grid, rms-NORMALISED per slot group
    (stop-gradient), times a per-slot-group ZERO-INIT scale. Zero init
    keeps every micro logit exactly 0 at model init, preserving
    calculate_hierarchical_prior as the exact init-policy anchor, and
    each group owns its sharpness through its scalar ALONE. The earlier
    design left the raw dot un-normalised ("the typed streams own the
    logit scale") — refuted 2026-08-25: the loss pins the PRODUCT
    scale.gram at the band, so growing stream norms forced type_scale
    0.026 -> 1e-4 while the raw gram grew to rms ~6e3, the gradient to
    type_scale (= the gram) hit 7e7 in the Jacobian probe (THE macro-head
    grad-runaway amplifier), and the upstream micro gradient (= the
    scale) died — the micro-gate stall re-created dynamically, mid-run.
    Dividing by the group's stop-grad rms gauges the stream scale away:
    type_scale is the one live factor, its gradient is O(w), and
    stream-norm drift can neither starve nor explode the route. The
    downstream within-modality logsumexp removes per-modality shifts, so
    these scales control sharpness only — the modality contest belongs
    to MacroHead.
    """

    @nn.compact
    def __call__(self, action_embeddings: jax.Array) -> jax.Array:
        inv_sqrt_d = action_embeddings.shape[-1] ** -0.5
        logits = (
            jnp.einsum("...id,...jd->...ij", action_embeddings, action_embeddings)
            * inv_sqrt_d
        )
        type_scale = self.param(
            "type_scale", nn.initializers.zeros_init(), (NUM_ACTION_SLOT_GROUPS,)
        ).astype(jnp.float32)
        flat = logits.reshape(*logits.shape[:-2], -1)
        # Per-group rms over the grid cells (f32, eps-guarded, all cells:
        # legality lives downstream), stop-grad so the normaliser is a
        # gauge, not a trainable route.
        group_oh = jax.nn.one_hot(
            FLAT_SRC_GROUP_MASK, NUM_ACTION_SLOT_GROUPS, dtype=jnp.float32
        )
        sq = jnp.square(flat.astype(jnp.float32))
        group_ms = (sq @ group_oh) / jnp.maximum(group_oh.sum(axis=0), 1.0)
        group_rms = jax.lax.stop_gradient(
            jnp.sqrt(group_ms + 1e-12)[..., FLAT_SRC_GROUP_MASK]
        )
        normed = flat.astype(jnp.float32) / group_rms
        return (normed * type_scale[FLAT_SRC_GROUP_MASK]).astype(logits.dtype)


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
    so the pools read typed spaces — then a shared MLP with a
    zero-initialised output layer maps each pooled vector to out_features
    outputs (default 1: the policy's modality logits, squeezed; the Q
    instantiation passes the categorical bin count).
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
        hidden = MLP(**self.cfg.mlp.to_dict())(pooled)
        logits = nn.Dense(
            self.out_features, kernel_init=nn.initializers.zeros, dtype=hidden.dtype
        )(hidden)
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
    count). micro scores every grid cell; cfg.micro_kind picks the
    readout: 'dot' is the policy's parameter-free scaled dot grid
    (MicroHead — the typed trunk streams plus the caller's ActionAdapter
    own the geometry, per-group zero-init scales keep the init anchor;
    scalar-only); 'pointer' is a PointerLogits grid with num_logits
    heads, returned flat as (..., N*N, num_logits) matching the policy's
    flat action indexing, behind its own per-group zero-init scale. Both
    kinds therefore obey one contract: every output is EXACTLY 0 at
    init, so each instantiation starts flat/unbiased (the policy at its
    hierarchical prior, the Q readout at uniform bins => E[Q] = 0
    everywhere). macro scores each modality via MacroHead over
    per-modality pooled src slots at the same width (zero-init out, same
    contract); optional cond threads to its MLP (the Q rungs'
    information set).

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
        if self.cfg.micro_kind == "dot":
            assert num_logits == 1, "the dot micro grid is scalar per cell"
            micro = MicroHead(name="micro")(action_embeddings)
        else:
            grid = PointerLogits(
                **self.cfg.micro_qk.to_dict(), num_heads=num_logits, name="micro"
            )(action_embeddings, action_embeddings)
            micro = grid.reshape(*grid.shape[:-3], -1, grid.shape[-1])
            # Per-slot-group ZERO-INIT scale — the exact trick MicroHead's
            # type_scale plays for the dot grid, so both micro kinds obey
            # the same contract: every micro output is exactly 0 at init.
            # With the macro out layer and the caller's adapter also
            # zero-init, the whole composed readout starts perfectly flat
            # (uniform bins => E[Q] = 0 for every cell) — no lecun noise
            # posing as action preferences for the CE to unlearn or for
            # downstream consumers (COMA/boost read Q̄ from step 0) to
            # mistake for signal. Gradient reaches the scale immediately
            # (the grid behind it is live), unfreezing the pointer.
            micro_scale = self.param(
                "micro_scale",
                nn.initializers.zeros_init(),
                (NUM_ACTION_SLOT_GROUPS,),
            ).astype(micro.dtype)
            micro = micro * micro_scale[FLAT_SRC_GROUP_MASK][..., None]
            # Single-factor within-modality routes (2026-08-24,
            # docs/critic-weakness-analysis.md Step 3 post-mortem). The
            # gated pointer is a scalar-gate x random-grid PRODUCT: the
            # gate's gradient is a random grid's correlation with the
            # residual, the grid's is proportional to the gate, and on
            # the live run neither moved in 60k steps (gate ~0.03-0.06,
            # q/k kernels at lecun init). A zero-init Dense on the src
            # token and one on the tgt token are each ONE zero-init
            # factor over a live input, so their gradient is consistent
            # from step 0. Singles: moves share a tgt column and resolve
            # through local_src, switches share a src row and resolve
            # through local_tgt; the pointer keeps interaction terms.
            # Composed Q is still exactly 0 at init (flat-at-init
            # contract); the modality centring in compose_action_grid
            # removes any per-modality offset these add.
            local_src = nn.Dense(
                num_logits,
                use_bias=False,
                kernel_init=nn.initializers.zeros,
                name="micro_local_src",
            )(action_embeddings)
            local_tgt = nn.Dense(
                num_logits,
                use_bias=False,
                kernel_init=nn.initializers.zeros,
                name="micro_local_tgt",
            )(action_embeddings)
            local = local_src[..., :, None, :] + local_tgt[..., None, :, :]
            micro = micro + local.reshape(*local.shape[:-3], -1, local.shape[-1])
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
