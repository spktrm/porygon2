import math
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.data import NUM_ACTION_CELLS
from rl.environment.interfaces import (
    CategoricalValueHeadOutput,
    PairValueHeadOutput,
    PolicyHeadOutput,
    RegressionValueHeadOutput,
)
from rl.model.constants import ALLY_TARGET_ROWS, CELL_BANK_SRC, CELL_BANK_TGT
from rl.model.modules import MLP, PointerLogits
from rl.model.utils import legal_log_policy, legal_policy


class HeadParams(NamedTuple):
    """Per-CALL sampling knobs, traced (a new value never recompiles).

    `prune_threshold` (2026-09-09) is DeepNash's FineTuning threshold: legal
    cells whose probability is below it are removed from the SAMPLED
    distribution and the rest renormalised (rl/model/utils.py
    prune_log_policy). 0.0, the training actors' value, is bit-identical
    to sampling the policy as trained; only the `thresholded` eval slot
    sets it. The policy's metrics always read the untouched policy."""

    temp: float = 1.0
    prune_threshold: float = 0.0


class PolicyMetrics(NamedTuple):
    policy: jax.Array
    log_policy: jax.Array
    entropy: jax.Array
    normalized_entropy: jax.Array
    magnet_kl: jax.Array


def compute_policy_metrics(
    logits: jax.Array, valid_mask: jax.Array, prior: jax.Array = None
):
    log_policy = legal_log_policy(logits, valid_mask)
    policy = legal_policy(logits, valid_mask)
    entropy = -jnp.sum(policy * log_policy, axis=-1)

    valid_sum = valid_mask.sum(axis=-1)
    safe_log_sum = jnp.maximum(valid_sum, 2)
    log_factor = 1.0 / jnp.log(safe_log_sum).astype(entropy.dtype)
    entropy_scale = jnp.where(valid_sum <= 1, 1.0, log_factor)
    normalized_entropy = entropy * entropy_scale

    if prior is None:
        valid_sum_expanded = jnp.maximum(valid_sum[..., None], 1)
        prior = jnp.where(valid_mask, 1.0 / valid_sum_expanded, 0.0)
    prior = prior.astype(log_policy.dtype)

    # 1e-9 rather than the true 0 on illegal cells: the log is taken before
    # the mask is reapplied, so a literal 0 would make it -inf.
    safe_prior = jnp.where(valid_mask, prior, 1e-9)
    log_prior = jnp.where(valid_mask, jnp.log(safe_prior), 0.0)

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


# Alive-mon differential support: margins -6..+6, matching the offline
# critic's 13-bin distributional target.
NUM_MARGIN_BINS = 13


def chosen_bank_rows(
    private_rows: jax.Array,
    move_rows: jax.Array,
    target_rows: jax.Array,
    action_cell: jax.Array,
):
    """The readout rows that produced a block cell's logit.

    The bank is private(6) | move(16) | target(17), and `CELL_BANK_SRC` /
    `CELL_BANK_TGT` name a cell's (source, target) row in it: a switch cell
    its private row and the ALLY_1_TARGET row it switches into, a move cell
    its move row and its target row, a standalone cell its target row twice.
    Written once (2026-09-03) for `SlotConditioning` and (2026-09-11) the
    previous action, which both condition on a taken action through its
    OWN rows rather than a cell index.
    """
    bank = jnp.concatenate((private_rows, move_rows, target_rows), axis=0)
    src_row = jnp.take(jnp.asarray(CELL_BANK_SRC), action_cell)
    tgt_row = jnp.take(jnp.asarray(CELL_BANK_TGT), action_cell)
    return jnp.take(bank, src_row, axis=0), jnp.take(bank, tgt_row, axis=0)


class SlotConditioning(nn.Module):
    """Doubles: condition slot 2's rows on the cell slot 1 chose.

    Zero-init output, so slot 2 starts as an exact copy of slot 1's readout
    and the conditioning has to earn its way in. Keyed by the chosen BLOCK
    CELL since 2026-08-31: `CELL_BANK_SRC`/`CELL_BANK_TGT` name the readout
    input rows that produced the cell's logit, so a switch cell now gathers
    its private row and the ally row it replaces, where the grid era's
    ALLY_i_SWITCH pseudo-slot gathered zeros.

    NOTE this keeps the MODEL side of doubles reachable and nothing more. The
    plumbing outside it -- per-slot masks in requests, two stored action
    indices, the (2, NUM_ACTION_CELLS) log_policy the learner would need, and
    the slot-alignment defect in the service -- is the known-open
    doubles workstream.
    """

    @nn.compact
    def __call__(
        self,
        sequence_rows: tuple[jax.Array, jax.Array, jax.Array],
        action_cell: jax.Array,
    ):
        private_rows, move_rows, target_rows = sequence_rows
        width = private_rows.shape[-1]
        chosen = jnp.concatenate(
            chosen_bank_rows(private_rows, move_rows, target_rows, action_cell),
            axis=-1,
        )
        delta = nn.Dense(
            width,
            kernel_init=nn.initializers.zeros_init(),
            use_bias=False,
            dtype=private_rows.dtype,
            name="condition",
        )(chosen)
        return (
            private_rows + delta,
            move_rows + delta,
            target_rows + delta,
        )


def zero_scalar(rows: jax.Array, name: str) -> jax.Array:
    """A zero-init single-factor scalar over live rows, (..., rows, 1): no
    stall mode (one zero factor over a live input moves at step 1)."""
    return nn.Dense(
        1,
        kernel_init=nn.initializers.zeros_init(),
        use_bias=False,
        dtype=rows.dtype,
        name=name,
    )(rows)


def bilinear_pair(
    src_rows: jax.Array,
    tgt_rows: jax.Array,
    *,
    qk_size: int,
    query_name: str,
    key_name: str,
    src_name: str | None = None,
    tgt_name: str | None = None,
    query_init: nn.initializers.Initializer = nn.initializers.zeros_init(),
) -> jax.Array:
    """THE pair form, written once (hoisted from FlatActionReadout
    2026-09-12 so the pairwise critic is a call to it, not a copy): a
    bilinear between every source row and every target row, (..., S, T) in
    the rows' dtype, plus -- when named -- a scalar on each side. The
    source-side `query` is the zero factor: the product is exactly 0 at
    init, d/d query is a rank-1 outer product of LIVE inputs so it moves at
    step 1, and `key` (gradient proportional to query) unfreezes at step 2
    (FlatActionReadout's docstring carries the argument). Submodules
    register on the calling compact module under the given names, so the
    parameter tree is the caller's. `query_init` is the zero factor's init;
    a caller whose product ALREADY multiplies a zero-init term passes a
    live one, or the two zero factors stall each other."""
    dtype = src_rows.dtype
    query = nn.Dense(
        qk_size, kernel_init=query_init, use_bias=False, dtype=dtype, name=query_name
    )(src_rows)
    key = nn.Dense(qk_size, use_bias=False, dtype=dtype, name=key_name)(tgt_rows)
    logits = jnp.einsum("...sq,...tq->...st", query, key) / math.sqrt(qk_size)
    if src_name is not None:
        logits = logits + zero_scalar(src_rows, src_name)
    if tgt_name is not None:
        logits = logits + zero_scalar(tgt_rows, tgt_name)[..., 0][..., None, :]
    return logits


class FlatActionReadout(nn.Module):
    """The whole action readout: three small heads over named trunk rows.

    The three heads are the three blocks of the action space (the flattening
    of ActionMask's fields -- proto/service.proto `Action`), emitted directly
    since 2026-08-31; the 41x41 scatter they used to land in is gone:

      switch   sheet rows x the ALLY row of the active slot being replaced
               (2026-09-11): the same pair form as moves x targets, so a
               candidate's logit reads what that row attended to -- the
               field, the opponent's active -- where a scalar per sheet row
               could only read the candidate. One block serves the battle
               switch and the team-preview lead alike; `kind` only matters
               to the service's decoder.
      move     16 candidate move rows against the 17 target rows, four of
               which carry the actual mon they would hit.
      other    one logit per target row for the standalone actions -- pass,
               default.

    INIT CONTRACT. Every logit is exactly 0 at init, so the policy starts
    UNIFORM over legal cells and `compute_policy_metrics(prior=None)` -- which
    already defaults to uniform-over-legal -- is the consistent anchor for the
    magnet and for the zero-avoiding KL.

    Getting to exact zero WITHOUT re-creating the two-factor stall is the
    whole subtlety here:

      * Each pair's `query` (source side) is zero-init and its `key` is
        not. The bilinear is exactly 0 at init, and d/d query is a rank-1
        outer product of LIVE inputs, so query moves at step 1 and key --
        whose gradient is proportional to query -- unfreezes at step 2. That is one zero factor over a live input, not a
        scalar multiplying a random grid.
      * NO layer-norm on the query heads. Its input is identically zero at
        init and its Jacobian goes as 1/sqrt(eps), i.e. ~1e3, straight into
        the zero-init kernel. The trunk's final pre-norm conditions these rows
        already.
      * `local_src` / `local_tgt` and the two scalar heads are zero-init
        single-factor routes over live rows, which have no stall mode.
        `local_src` is also where a per-MODALITY force can live: modality is a
        function of the src half alone, so it is the flat design's answer to
        the macro head's dedicated per-modality parameter, and it is the
        pre-decided place to add depth if the macro entropy floor cannot hold.
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(
        self,
        private_rows: jax.Array,
        move_rows: jax.Array,
        target_rows: jax.Array,
        temp: float = 1.0,
        decision_slot: int = 0,
    ) -> jax.Array:
        qk_size = self.cfg.qk_size

        # The switch block is sheet rows x the ALLY row of the active slot a
        # switch replaces (2026-09-11). `switch` keeps its name as the
        # sheet-side scalar, so a merge carries it; `switch_local_tgt`, a
        # scalar on the ally row, is the whether-to-switch level -- one that
        # reads the state, replacing the context-free `switch_bias`.
        ally_row = jnp.take(
            target_rows, ALLY_TARGET_ROWS[decision_slot : decision_slot + 1], axis=-2
        )
        switch_logit = bilinear_pair(
            private_rows,
            ally_row,
            qk_size=qk_size,
            query_name="switch_query",
            key_name="switch_key",
            src_name="switch",
            tgt_name="switch_local_tgt",
        )[..., 0]
        move_target = bilinear_pair(
            move_rows,
            target_rows,
            qk_size=qk_size,
            query_name="query",
            key_name="key",
            src_name="local_src",
            tgt_name="local_tgt",
        )
        other_logit = zero_scalar(target_rows, "other")[..., 0]

        cells = jnp.concatenate(
            (
                switch_logit,
                move_target.reshape(*move_target.shape[:-2], -1),
                other_logit,
            ),
            axis=-1,
        )
        assert cells.shape[-1] == NUM_ACTION_CELLS

        # f32 once, before the masked log-softmax: bf16 normalisation holds
        # only to ~3e-3 and every policy-loss term reads this.
        # A PLAIN ARRAY, not a NamedTuple: flax's capture_intermediates skips
        # non-array returns, which would silently exempt this module from
        # tests/test_dtype_policy.py -- and this cast is the one place in the
        # forward that is deliberately f32.
        return cells.astype(jnp.float32) / temp


class CategoricalValueLogitHead(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, embedding: jax.Array):
        # f32 from the head outwards: a handful of bins, and the main
        # critic's CE, the v-trace bootstrap probs and the expectation all
        # read them.
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
        # conditional entropy, the PBRS potential channel) are all f32-MSE
        # consumers.
        mlp_config = self.cfg.mlp.to_dict()
        if getattr(self.cfg, "zero_init_output", False):
            # A head whose output must start at a known point: zero last
            # kernel, and Dense biases start at zero, so it reads exactly 0.
            mlp_config["final_kernel_init"] = nn.initializers.zeros
        x = MLP(**mlp_config)(x).astype(jnp.float32)
        if getattr(self.cfg, "output_activation", None) is not None:
            x = self.cfg.output_activation(x)
        return RegressionValueHeadOutput(logits=x.squeeze(-1))


# The signed parts of a PairValueHead's value, in `partials` order.
PAIR_VALUE_PARTIALS = (
    "unary_mine",
    "unary_theirs",
    "cross",
    "synergy_mine",
    "synergy_theirs",
)


def masked_softmax(scores: jax.Array, mask: jax.Array) -> jax.Array:
    """Softmax over the last two axes restricted to `mask`: exact zeros off
    the mask, exact 1/n at equal scores, and all zeros (not NaN) when the
    mask is empty. f32 in, f32 out."""
    scores = jnp.where(mask, scores, -1e9)
    scores = scores - jnp.max(scores, axis=(-2, -1), keepdims=True)
    weights = jnp.where(mask, jnp.exp(scores), 0.0)
    return weights / jnp.maximum(weights.sum(axis=(-2, -1), keepdims=True), 1e-9)


def centred_over(values: jax.Array, mask: jax.Array) -> jax.Array:
    """`values` minus their mean over `mask` (the last two axes), exact zeros
    off the mask. A pair term this passes through is zero-mean over the
    live pairs of a state by construction: a state-wide offset cannot
    live in it, only in the unary, so whatever it carries differs across
    pairs -- which is also the only gradient its softmax weights can
    receive (d/d alpha_ij is the pair's deviation from the weighted mean)."""
    count = jnp.maximum(mask.sum(axis=(-2, -1), keepdims=True), 1)
    mean = jnp.where(mask, values, 0.0).sum(axis=(-2, -1), keepdims=True) / count
    return jnp.where(mask, values - mean, 0.0)


class PairValueHead(nn.Module):
    """The pairwise entity critic (2026-09-12): a generalised additive model
    over 12 entity rows, my side A = rows[:6], theirs B = rows[6:].

        V = sum_A v_i u_i - sum_B v_j u_j
            + sum_{i in A, j in B} alpha_ij m_ij
            + sum_{i,i' in A} beta_ii' s_ii' - sum_{j,j' in B} beta_jj' s_jj'

    u_i = u(x_i): the unary term, single-mon strength, so the pair terms
        are interactions and not main effects in disguise. The rows are
        post-trunk (both heads, 2026-09-12 user call), so whatever field,
        history or request context a term needs has already been routed
        into its row by the trunk -- the head carries no context of its own.
    m_ij = tanh(g(i, j) - g(j, i)), centred over the alive cross pairs of
        the state (`centred_over`): the cross-side pair, antisymmetric by
        construction. Strength of i over j and pressure of j on i are this
        one number. g is ONE bilinear (`cross_query` x `cross_key`) over
        all 12 rows, read in both orientations. The centring is a
        structural restriction, not a penalty (2026-09-13): the label is
        one scalar per state, so a pair term competing with a contextual
        unary for the same residual found the state-wide offset first and
        repeated it across all 36 pairs, and its weights, whose only
        gradient is a pair's deviation from the weighted mean, never left
        uniform. Zero-mean pairs cannot carry that offset.
    alpha = softmax over ALIVE cross pairs of h(i, j) + h(j, i), a second
        bilinear symmetrised; a fainted or absent mon's pairs weigh exactly
        0. Concentrating on the decisive matchup is the intended reading.
    s_ii' = tanh(g_s(i, i') + g_s(i', i)): the same-side pair (synergy),
        symmetric by construction, one bilinear shared by both sides, with
        its own symmetric softmax weights beta per side; centred over each
        side's alive pairs for the same reason as m.

    Because u, g, h, g_s are each ONE function shared across sides, swapping
    the two sides (rows and flags) negates the HEAD exactly
    (tests/test_pair_value_head.py); whether the trunk rows themselves are
    mirror-consistent is a property of the trunk, not of this head. At init
    V == 0 (the unary's last kernel and the pair queries are zero, tanh 0 =
    0); the weight queries are LIVE at init, a contrast over the rows,
    because a centred term under uniform weights is two zero factors.
    The pair queries and the unary's last layer move at step 1, keys and
    the weight scores from step 2 (their gradients are proportional to the
    pair terms, which are 0 for one step). The pair parts
    are convex combinations of centred numbers in [-2, 2], so each is
    bounded by two units and the unary terms carry the scale. bf16 through the
    projections, one f32 cast before the tanh / softmax / sums.
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(
        self,
        rows: jax.Array,
        valid: jax.Array,
        alive: jax.Array,
    ) -> PairValueHeadOutput:
        per_side = rows.shape[0] // 2
        zeros = nn.initializers.zeros_init()
        qk_size = self.cfg.qk_size

        unary = MLP(
            layer_sizes=(self.cfg.unary_hidden, 1),
            final_kernel_init=zeros,
            name="unary",
        )(rows)[..., 0].astype(jnp.float32)
        unary = jnp.where(valid, unary, 0.0)

        def pair(name: str, query_init=zeros) -> jax.Array:
            return bilinear_pair(
                rows,
                rows,
                qk_size=qk_size,
                query_name=f"{name}_query",
                key_name=f"{name}_key",
                query_init=query_init,
            ).astype(jnp.float32)

        # The weight scores multiply a centred, zero-init pair term, so they
        # are the live factor: uniform weights over a centred term give the
        # term no gradient (d/dm_ij = alpha_ij - 1/n = 0) and vice versa.
        live = nn.initializers.lecun_normal()

        mine = slice(0, per_side)
        theirs = slice(per_side, 2 * per_side)
        present = valid & alive
        cross_mask = present[mine][:, None] & present[theirs][None, :]
        off_diagonal = ~jnp.eye(per_side, dtype=jnp.bool_)
        synergy_mask = (
            jnp.stack(
                (
                    present[mine][:, None] & present[mine][None, :],
                    present[theirs][:, None] & present[theirs][None, :],
                )
            )
            & off_diagonal
        )

        cross_scores = pair("cross")
        cross = jnp.tanh(cross_scores[mine, theirs] - cross_scores[theirs, mine].T)
        cross = centred_over(cross, cross_mask)
        weight_scores = pair("cross_weight", live)
        cross_weight = masked_softmax(
            weight_scores[mine, theirs] + weight_scores[theirs, mine].T, cross_mask
        )

        synergy_scores = pair("synergy")
        synergy_blocks = jnp.stack(
            (synergy_scores[mine, mine], synergy_scores[theirs, theirs])
        )
        synergy = jnp.tanh(synergy_blocks + jnp.swapaxes(synergy_blocks, -2, -1))
        synergy = centred_over(synergy, synergy_mask)
        synergy_weight_scores = pair("synergy_weight", live)
        synergy_weight_blocks = jnp.stack(
            (synergy_weight_scores[mine, mine], synergy_weight_scores[theirs, theirs])
        )
        synergy_weight = masked_softmax(
            synergy_weight_blocks + jnp.swapaxes(synergy_weight_blocks, -2, -1),
            synergy_mask,
        )

        synergy_sums = jnp.sum(synergy_weight * synergy, axis=(-2, -1))
        partials = jnp.stack(
            (
                jnp.sum(unary[mine]),
                -jnp.sum(unary[theirs]),
                jnp.sum(cross_weight * cross),
                synergy_sums[0],
                -synergy_sums[1],
            )
        )
        return PairValueHeadOutput(
            value=jnp.sum(partials),
            unary=unary,
            cross=cross,
            cross_weight=cross_weight,
            synergy=synergy,
            synergy_weight=synergy_weight,
            partials=partials,
        )
