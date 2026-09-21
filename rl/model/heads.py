import math
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.data import NUM_ACTION_CELLS, NUM_SWITCH_CELLS, OTHER_CELL_OFFSET
from rl.environment.interfaces import (
    CategoricalValueHeadOutput,
    PolicyHeadOutput,
    RegressionValueHeadOutput,
)
from rl.model.constants import (
    CELL_BANK_SRC,
    CELL_BANK_TGT,
    MY_ACTIVE_PUBLIC_ROWS,
    OPP_ACTIVE_PUBLIC_ROWS,
)
from rl.model.modules import MLP, PointerLogits
from rl.model.utils import legal_log_policy


class ReadoutRows(NamedTuple):
    """The trunk rows the action readout scores, sliced by name once
    (`PlayerModel.readout_rows`). `public` is every public entity row, mine
    then theirs, actives first on each side (`MY_ACTIVE_PUBLIC_ROWS`,
    `OPP_ACTIVE_PUBLIC_ROWS`); `public_alive` whether each exists and is
    alive and `public_active` whether it is on the field -- the entity's own
    state, never the action mask."""

    private: jax.Array
    move: jax.Array
    target: jax.Array
    public: jax.Array
    public_alive: jax.Array
    public_active: jax.Array

    @property
    def opponent(self) -> jax.Array:
        return self.public[..., OPP_ACTIVE_PUBLIC_ROWS[0] :, :]

    @property
    def opponent_alive(self) -> jax.Array:
        return self.public_alive[..., OPP_ACTIVE_PUBLIC_ROWS[0] :]

    def my_active(self, slot: int) -> tuple[jax.Array, jax.Array]:
        """My active in `slot` (a (..., 1, width) row) and whether one is
        there: on the field and alive."""
        row = MY_ACTIVE_PUBLIC_ROWS[slot]
        present = self.public_active[..., row] & self.public_alive[..., row]
        return self.public[..., row : row + 1, :], present


class HeadParams(NamedTuple):
    """Per-CALL sampling knobs, traced (a new value never recompiles).

    `greedy` plays the player policy's most likely legal cell instead of
    sampling it (rl/model/utils.py sampling_log_policy). False, the
    training actors' value, is bit-identical to sampling the policy as
    trained; only the `argmax` eval slot sets it. The policy's metrics
    always read the untouched policy, and the builder heads read `temp`
    alone."""

    temp: float = 1.0
    greedy: bool = False


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
    policy = jnp.where(valid_mask, jnp.exp(log_policy), 0.0)
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
    def __call__(self, rows: ReadoutRows, action_cell: jax.Array) -> ReadoutRows:
        private_rows, move_rows, target_rows = rows.private, rows.move, rows.target
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
        return rows._replace(
            private=private_rows + delta,
            move=move_rows + delta,
            target=target_rows + delta,
        )


def row_score(rows: jax.Array, name: str) -> jax.Array:
    """A zero-init linear score of each row alone, (..., rows, 1): no stall
    mode (one zero factor over a live input moves at step 1)."""
    return nn.Dense(
        1,
        kernel_init=nn.initializers.zeros_init(),
        use_bias=False,
        dtype=rows.dtype,
        name=name,
    )(rows)


def bilinear_pair(
    src_rows: jax.Array, tgt_rows: jax.Array, *, qk_size: int, block: str
) -> jax.Array:
    """(..., src, tgt): the pair logits plus a zero-init score of each source
    row and of each target row. `block` names the four projections:
    `{block}_query`, `{block}_key`, `{block}_score`, `{block}_target_score`.
    """
    logits = pair_logits(
        src_rows, tgt_rows, qk_size=qk_size, query=f"{block}_query", key=f"{block}_key"
    )
    logits = logits + row_score(src_rows, f"{block}_score")
    return logits + row_score(tgt_rows, f"{block}_target_score")[..., 0][..., None, :]


def pair_logits(
    src_rows: jax.Array, tgt_rows: jax.Array, *, qk_size: int, query: str, key: str
) -> jax.Array:
    """(..., src, tgt): q(src)·k(tgt)/√d with a zero-init query over live
    keys, the one bilinear every pair in the readout is built from."""
    dtype = src_rows.dtype
    queries = nn.Dense(
        qk_size,
        kernel_init=nn.initializers.zeros_init(),
        use_bias=False,
        dtype=dtype,
        name=query,
    )(src_rows)
    keys = nn.Dense(qk_size, use_bias=False, dtype=dtype, name=key)(tgt_rows)
    return jnp.einsum("...sq,...tq->...st", queries, keys) / math.sqrt(qk_size)


class OpponentTeamPairing(nn.Module):
    """A candidate's expected pair score against the opponent's TEAM, under
    the readout's own belief about who stands opposite next turn:

        term[c]        = sum_j  belief[c, j] * matchup[c, j]
        matchup[c, j]  = q_block(c) · k(their_j) / √d  +  score(their_j)
        belief[c, ·]   = softmax over their ALIVE rows of
                         q_belief_block(c) · k_belief(their_j) / √d

    j runs over every opponent public row, revealed or not; a fainted or
    absent row is masked out of the belief and so out of the term. One
    instance serves every block: the keys `opponent_key`, `belief_key` and
    the row score `opponent_score` are shared, so the move block's
    every-turn gradient trains the factors the sparse switch block reads
    through; the two queries are per block and zero-init, so the term is
    exactly 0 at init with the belief uniform over the alive rows.
    """

    qk_size: int

    @nn.compact
    def __call__(
        self,
        candidates: jax.Array,
        opponent: jax.Array,
        opponent_alive: jax.Array,
        *,
        block: str,
    ) -> jax.Array:
        matchup = pair_logits(
            candidates,
            opponent,
            qk_size=self.qk_size,
            query=f"{block}_opponent_query",
            key="opponent_key",
        )
        matchup = matchup + row_score(opponent, "opponent_score")[..., 0][..., None, :]

        belief_logits = pair_logits(
            candidates,
            opponent,
            qk_size=self.qk_size,
            query=f"{block}_belief_query",
            key="belief_key",
        )
        alive = opponent_alive[..., None, :]
        belief = jax.nn.softmax(jnp.where(alive, belief_logits, -1e9), axis=-1)
        belief = jnp.where(alive, belief, 0.0)

        return jnp.sum(belief * matchup, axis=-1)


class FlatActionReadout(nn.Module):
    """The whole action readout: three small heads over named trunk rows.

    The three heads are the three blocks of the action space (the flattening
    of ActionMask's fields -- proto/service.proto `Action`), emitted directly
    since 2026-08-31; the 41x41 scatter they used to land in is gone:

      switch   the candidate's sheet row paired with every mon on the field,
               all from the PUBLIC rows (2026-09-18): my active it replaces
               (what it gives up), my other active that stays (doubles; the
               term masks itself out in singles), and the opponent's TEAM --
               every alive public row of theirs, revealed or not, weighted
               by a learned belief about who is opposite next turn
               (`OpponentTeamPairing`). Presence is the entity's own state,
               not the action mask, so every term is live on a forced switch
               (no legal move, hence no valid enemy TARGET row) and the
               opponent term at team preview (six enemy rows, no actives).
               One block serves the battle switch and the team-preview lead
               alike; `kind` only matters to the service's decoder.
      move     16 candidate move rows against the 17 target rows, four of
               which carry the actual mon they would hit, plus the same
               opponent-team term per move (who the move actually lands on
               after their response) -- and the move block's every-turn
               gradient is what trains the shared opponent key and belief
               key the switch block reads through.
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
      * The `*_score` heads are zero-init single-factor routes over live
        rows, which have no stall mode. `move_score` / `switch_score` are
        also where a per-MODALITY force can live: modality is a function of
        the source row alone, so they are the flat design's answer to the
        macro head's dedicated per-modality parameter, and the pre-decided
        place to add depth if the macro entropy floor cannot hold.
    """

    cfg: ConfigDict

    def chosen_pair_features(self, rows: ReadoutRows, action_cell: jax.Array):
        source, target = chosen_bank_rows(
            rows.private, rows.move, rows.target, action_cell
        )
        leaving, _ = rows.my_active(0)

        def features(block, target_row):
            def projection(row, role):
                params = self.get_variable("params", f"{block}_{role}")
                return nn.Dense(
                    params["kernel"].shape[-1],
                    use_bias=False,
                    dtype=row.dtype,
                    parent=None,
                ).apply({"params": params}, row)

            interaction = projection(source, "query") * projection(target_row, "key")
            return jnp.concatenate(
                (
                    interaction / math.sqrt(self.cfg.qk_size),
                    projection(source, "score"),
                    projection(target_row, "target_score"),
                )
            )

        pair = jnp.where(
            action_cell < NUM_SWITCH_CELLS,
            features("switch", leaving[0]),
            features("move", target),
        )
        return jnp.where(action_cell < OTHER_CELL_OFFSET, pair, 0)

    @nn.compact
    def __call__(
        self, rows: ReadoutRows, temp: float = 1.0, decision_slot: int = 0
    ) -> jax.Array:
        private_rows, move_rows, target_rows = rows.private, rows.move, rows.target
        qk_size = self.cfg.qk_size
        opponent_team = OpponentTeamPairing(qk_size, name="opponent_team")

        def against_their_team(candidates: jax.Array, block: str) -> jax.Array:
            return opponent_team(
                candidates, rows.opponent, rows.opponent_alive, block=block
            )

        # switch(c) = pair(c, my active it replaces)
        #           + [partner present] pair(c, my active that stays)
        #           + E_belief[pair(c, their team)]
        leaving_row, _ = rows.my_active(decision_slot)
        partner_row, partner_present = rows.my_active(1 - decision_slot)
        leaving = bilinear_pair(
            private_rows, leaving_row, qk_size=qk_size, block="switch"
        )
        partner = pair_logits(
            private_rows,
            partner_row,
            qk_size=qk_size,
            query="partner_query",
            key="partner_key",
        )
        partner = (
            partner + row_score(partner_row, "partner_score")[..., 0][..., None, :]
        )
        partner = jnp.where(partner_present[..., None, None], partner, 0.0)
        switch_logit = (
            leaving[..., 0]
            + partner[..., 0]
            + against_their_team(private_rows, "switch")
        )

        # move(m, t) = pair(m, target t) + E_belief[pair(m, their team)]
        move_target = bilinear_pair(
            move_rows, target_rows, qk_size=qk_size, block="move"
        )
        move_target = move_target + against_their_team(move_rows, "move")[..., None]

        other_logit = row_score(target_rows, "other_score")[..., 0]

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
        logits = MLP(**self.cfg.mlp.to_dict(), name="mlp")(embedding).astype(
            jnp.float32
        )

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
        x = MLP(**mlp_config, name="mlp")(x).astype(jnp.float32)
        if getattr(self.cfg, "output_activation", None) is not None:
            x = self.cfg.output_activation(x)
        return RegressionValueHeadOutput(logits=x.squeeze(-1))
