import functools
import math
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.data import NUM_ACTION_CELLS
from rl.environment.interfaces import (
    CategoricalValueHeadOutput,
    PolicyHeadOutput,
    RegressionValueHeadOutput,
)
from rl.model.constants import CELL_BANK_SRC, CELL_BANK_TGT
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


def compute_policy_metrics(
    logits: jax.Array, valid_mask: jax.Array, prior: jax.Array = None
):
    """
    Computes standard policy distributions, entropy, normalized entropy,
    and the KL divergence (exploration magnet) penalty.
    """
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
    its private row twice, a move cell its move row and its target row, a
    standalone cell its target row twice. Written once (2026-09-03) for
    `SlotConditioning` and the dynamics head, which both condition on the
    taken action through its OWN rows rather than a cell index.
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
    its private row where the grid era's ALLY_i_SWITCH pseudo-slot gathered
    zeros.

    NOTE this keeps the MODEL side of doubles reachable and nothing more. The
    plumbing outside it -- per-slot masks in requests, two stored action
    indices, the (2, NUM_ACTION_CELLS) log_policy the learner would need, and
    the ~75% slot-alignment defect in the service -- is the known-open
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


class FlatActionReadout(nn.Module):
    """The whole action readout: three small heads over named trunk rows.

    Replaces the hierarchical stack -- `MacroMicroHead` = per-modality
    queries, five MLPs and five zero-init output layers, over a per-slot-group
    `PointerLogits` grid with a stop-grad RMS gauge and three zero-init local
    routes -- which was instantiated twice, once for the policy and once for
    an advantage head the policy did not read. 2.65M parameters became 0.13M.

    The three heads are the three blocks of the action space (the flattening
    of ActionMask's fields -- proto/service.proto `Action`), emitted directly
    since 2026-08-31; the 41x41 scatter they used to land in is gone:

      switch   one logit per SHEET ROW, "may this mon come in and should it".
               One block serves the battle switch and the team-preview lead
               alike; `kind` only matters to the service's decoder.
      move     THE ONLY BILINEAR: 16 candidate move rows against the 17 target
               rows, four of which carry the actual mon they would hit.
      other    one logit per target row for the standalone actions -- pass,
               default.

    INIT CONTRACT. Every logit is exactly 0 at init, so the policy starts
    UNIFORM over legal cells and `compute_policy_metrics(prior=None)` -- which
    already defaults to uniform-over-legal -- is the consistent anchor for the
    magnet and for the zero-avoiding KL. `calculate_hierarchical_prior` was
    that anchor while the head was hierarchical and retires with it.

    Getting to exact zero WITHOUT re-creating the two-factor stall (CLAUDE.md
    13: a learned grid behind a zero-init scale sat at lecun init for 60k
    steps) is the whole subtlety here:

      * `query` is zero-init and `key` is not. The bilinear is exactly 0 at
        init, and d/d query is a rank-1 outer product of LIVE inputs, so query
        moves at step 1 and key -- whose gradient is proportional to query --
        unfreezes at step 2. That is one zero factor over a live input, not a
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
    ) -> jax.Array:
        dtype = private_rows.dtype
        zeros = nn.initializers.zeros_init()
        scalar_head = functools.partial(
            nn.Dense, features=1, kernel_init=zeros, use_bias=False, dtype=dtype
        )

        switch_logit = scalar_head(name="switch")(private_rows)[..., 0]
        other_logit = scalar_head(name="other")(target_rows)[..., 0]

        qk_size = self.cfg.qk_size
        query = nn.Dense(
            qk_size, kernel_init=zeros, use_bias=False, dtype=dtype, name="query"
        )(move_rows)
        key = nn.Dense(qk_size, use_bias=False, dtype=dtype, name="key")(target_rows)
        move_target = jnp.einsum("...mq,...tq->...mt", query, key) / math.sqrt(qk_size)
        move_target = (
            move_target
            + scalar_head(name="local_src")(move_rows)
            + scalar_head(name="local_tgt")(target_rows)[..., 0][..., None, :]
        )

        # The whether-to-switch baseline: one scalar over the whole switch
        # block. The grid era's `ally_switch_bias` (2, 1) is folded to one
        # value -- in singles only row 0 ever trained (the mask cleared the
        # other ally half), and at team preview a uniform shift over an
        # all-switch legal set is softmax-invariant, so behaviour is
        # unchanged. A per-active-slot bias rejoins with the doubles
        # workstream, which needs per-slot masks anyway.
        switch_bias = self.param("switch_bias", zeros, (1,)).astype(dtype)

        cells = jnp.concatenate(
            (
                switch_logit + switch_bias,
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
