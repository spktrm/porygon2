"""The latent transition model (2026-09-05; latent actions 2026-09-07):
g(h_t, u, z) -> h_{t+1}, unrolled K steps, with nothing masked past the
root encoding.

`h_t` is the trunk's post-trunk POLICY-READABLE sequence (73 rows), exactly
what the heads read, so the imagined `h_{t+1}` keeps the sequence layout
and `V`, this model itself and the heads below apply to it unchanged. Two
latents condition one step:

- `u`, the LATENT ACTION: one categorical of `action_classes` codes. At an
  observed state the action encoder q(u | h, a) reads the taken cell's
  readout rows against the whole sequence (one cross-attention) -- my
  action IS observed, so unlike LAPO/Genie there is no inverse-dynamics
  inference. At an imagined node the candidate generator rho(u | h) is
  the policy over the latent alphabet: an autoregressive decoder that
  draws J distinct codes WITHOUT replacement inside its own support set,
  so a rollout never enumerates concrete cells past the root and never
  reads a future request. The generator is distilled from the base policy
  at observed states (an imitation proposal, not search distillation).
- `z`, the chance code (`code_groups` categoricals of `code_classes`):
  the opponent decides, the engine rolls and information is revealed,
  none of it observed as a choice (memory: the opponent's action is
  NEVER a label); inferred from the real next rows by the posterior and
  predicted from (h, u) by the prior.

Both enter g as TOKENS beside the 73 rows (a learned slot embedding on
the rows so a row the trunk zeroed is still addressable), through
`dynamics_blocks` under an all-True mask: validity is content the rows
carry, legality is the generator's mass. The real target rows (zero
where the trunk zeroed them at t+1) teach absent rows -> zero and
appearing rows -> content; the 2026-09-05 form masked g with the CURRENT
validity and could never write a row that appears (LESSONS 2026-09-07).

Readers on an imagined node: grounding (per row, the CHANGE in the
DYNAMICS_TARGET_ROWS' pre-trunk content, zero-init at the copy
predictor), the request kind + done, and the conditional TERMINAL
OUTCOME (loss / draw / win given that the node ends the game -- V is
unconditional, so the fractional-continuation backup in rl/model/search.py
needs this reader, not V, at a node that may be terminal). The shared
`v_head` is applied in the parent model.

Init contract: `dynamics_out_proj` is the ONE zero factor -- g is exactly
the copy predictor at step 0 and its gradient is the outer product of the
live block output with the residual, so it moves at step 1 and everything
behind it unfreezes at step 2 (the readout's query/key rule, not the
two-factor stall). Only the policy-readable rows exist here, so nothing a
rollout sees is privileged (`tests/test_transition_model.py` pins it with
the posterior as the positive control).
"""

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.data import CAT_VF_SUPPORT
from rl.model.categoricals import (
    NEGATIVE_INFINITY_LOGIT,
    draw_or_mode,
    gumbel_top_k,
    masked_log_softmax,
    straight_through_sample,
    support_set,
    unimix_probs,
)
from rl.model.constants import (
    CLS_ROW,
    DYNAMICS_TARGET_ROWS,
    MOVE_ROWS,
    NUM_POLICY_READABLE_ROWS,
    PRIVATE_ROWS,
    TARGET_ROWS,
)
from rl.model.heads import chosen_bank_rows
from rl.model.modules import FFWMLP, MLP, MultiHeadAttention, RMSNorm
from rl.model.trunk import Trunk


class RealState(NamedTuple):
    """One observed state's legal set through the action encoder: the
    static enumeration of its legal cells (`cells` padded with cell 0,
    `cell_valid` the occupancy, `taken_index` the taken cell's slot,
    `overflow` when the enumeration cannot represent the step -- more
    legal cells than the width, or the taken cell not among them),
    the encoder's logits per enumerated cell (live), the base policy's
    latent-action distribution p(u | h) = sum_a pi(a) q(u | h, a) (sg),
    its teacher support set and the Gumbel-top-J teacher order, and the
    taken cell's own code distribution (sg, the alignment target)."""

    taken_cell: jax.Array
    cells: jax.Array
    cell_valid: jax.Array
    taken_index: jax.Array
    overflow: jax.Array
    logits: jax.Array
    code_target: jax.Array
    support_mask: jax.Array
    teacher_codes: jax.Array
    taken_code_probs: jax.Array


class Candidates(NamedTuple):
    """The generator's draw at one node: J code indices, which slots are
    occupied (inside the node's support set), the log of each draw's
    conditional (over the support minus the prefix -- NOT a marginal
    prior and NOT an inclusion probability), the first-conditional prior
    rho at each code, the support mask and the prior mass the occupied
    draws retain."""

    codes: jax.Array
    occupied: jax.Array
    log_draw_conditionals: jax.Array
    rho_at_codes: jax.Array
    support_mask: jax.Array
    retained_mass: jax.Array


class RootOutputs(NamedTuple):
    """The OBSERVED root state h_t's legal set through the action encoder.
    One per start step: leading axis T."""

    action_logits: jax.Array
    action_cells: jax.Array
    action_cell_valid: jax.Array
    action_taken_index: jax.Array
    action_overflow: jax.Array


class NodeOutputs(NamedTuple):
    """The readers at every node hhat_0 .. hhat_K: leading axis K + 1."""

    generator_logits: jax.Array
    teacher_codes: jax.Array
    generator_target: jax.Array
    support_mask: jax.Array
    node_overflow: jax.Array
    align_logits: jax.Array
    align_target: jax.Array


class StepOutputs(NamedTuple):
    """The transitions hhat_k -> hhat_{k+1}, each paired with the real step
    t + k + 1: leading axis K."""

    action_one_hot: jax.Array
    pred: jax.Array
    prior_logits: jax.Array
    post_logits: jax.Array
    post_one_hot: jax.Array
    kind_logits: jax.Array
    done_logit: jax.Array
    terminal_logits: jax.Array


class FirstStepOutputs(NamedTuple):
    """The first transition only: the prior-MODE decode (no gradient) and
    the grounding reads off it. No offset axis."""

    pred_prior: jax.Array
    ground: jax.Array
    ground_prior: jax.Array


# The two leaves that never leave the model: the parent reads the imagined
# STATES for the value head, the consistency error and the rms panel, and
# exports what it derives from them, not the states themselves.
INTERNAL_LEAVES = frozenset({"pred", "pred_prior"})


class TransitionOutput(NamedTuple):
    """The K-step unroll's outputs, in the four groups that share a leading
    axis. After the parent's vmap every leaf also carries the trajectory
    axis T, at the position `VMAP_AXES` gives for its group -- so a node or
    step leaf reads (offset, T, ...) and a root leaf (T, ...)."""

    root: RootOutputs
    nodes: NodeOutputs
    steps: StepOutputs
    first: FirstStepOutputs

    def exported(self) -> dict[str, jax.Array]:
        """The leaves that leave the model as `PlayerActorOutput`'s
        `transition_*` fields, with the prefix applied in ONE place --
        `player_model._forward_transition` adds only what it derives. Pinned
        against the dataclass by tests/test_transition_model.py."""
        return {
            f"transition_{name}": leaf
            for group in self
            for name, leaf in zip(type(group)._fields, group)
            if name not in INTERNAL_LEAVES
        }


# The vmap axis the trajectory axis T takes in each group, written once
# here instead of once per field: a group is exactly the set of leaves
# that share one.
VMAP_AXES = TransitionOutput(root=0, nodes=1, steps=1, first=0)


class StepLatents(NamedTuple):
    """One imagined step's latents and the state g wrote under them.
    `pred_prior` is the prior-MODE decode, present on the first transition
    only (`with_prior_decode`) and None elsewhere -- it never leaves
    `_unroll`, which reads it into `FirstStepOutputs`."""

    pred: jax.Array
    pred_prior: jax.Array | None
    prior_logits: jax.Array
    post_logits: jax.Array
    post_one_hot: jax.Array


class NodeReadout(NamedTuple):
    kind_logits: jax.Array
    done_logit: jax.Array
    terminal_logits: jax.Array


def legal_enumeration(legal: jax.Array, taken_cell: jax.Array, max_cells: int):
    """The static enumeration of one legal set: `cells` the first
    `max_cells` legal cell indices (padded with 0), `cell_valid` their
    occupancy, `taken_index` the taken cell's slot (0 when absent) and
    `overflow` -- the enumeration cannot represent this step: more legal
    cells than the width, or the taken cell not among the enumerated ones
    (a padded row's empty legal set included). Overflow is counted and
    masks every action-set loss, never silently renormalised."""
    cells = jnp.nonzero(legal, size=max_cells, fill_value=0)[0]
    num_legal = legal.sum()
    cell_valid = jnp.arange(max_cells) < num_legal
    is_taken = (cells == taken_cell) & cell_valid
    taken_index = jnp.argmax(is_taken)
    overflow = (num_legal > max_cells) | jnp.logical_not(is_taken.any())
    return cells, cell_valid, taken_index, overflow


def _stack(entries: list) -> NamedTuple:
    """Stack a list of same-typed NamedTuples leafwise onto a new leading
    offset axis, keeping the type."""
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *entries)


def _gather_offsets(leaf: jax.Array, offsets: list[jax.Array]) -> jax.Array:
    """(T, ...) -> (K + 1, T, ...): the real step each offset reads, one
    take per offset. The last step reads itself; the loss masks the
    out-of-range offsets."""
    return jnp.stack([jnp.take(leaf, index, axis=0) for index in offsets])


class RowRead(nn.Module):
    """ONE bias-free Dense(D -> width) applied to every row and flattened in
    row order, so a reader sees WHICH row carries what. The mean pool it
    replaced cancelled a row's identity and diluted a one-row change by
    the row count (irqeetfg 1266k-1312k: kl_long < kl_short). No validity
    argument and no bias: a row the trunk zeroed reads exactly 0, and an
    imagined node -- which has no validity to hand it -- reads whatever
    content its rows carry."""

    width: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, rows: jax.Array) -> jax.Array:
        read = nn.Dense(self.width, use_bias=False, dtype=self.dtype, name="read")(rows)
        return read.reshape(-1)


class ActionEncoder(nn.Module):
    """q(u | h, a): the taken cell's readout rows form ONE query that
    attends over the whole sequence (the cell's consequence lives in the
    opponent's active row, my boosts and the field, not in its own two
    rows), then an MLP to the `action_classes` logits. Applied at observed
    states and at the root only -- the one place a concrete legal set is
    legitimate."""

    cfg: ConfigDict
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, rows: jax.Array, src_row: jax.Array, tgt_row: jax.Array):
        model_size = rows.shape[-1]
        query = nn.Dense(model_size, dtype=self.dtype, name="query_proj")(
            jnp.concatenate((src_row, tgt_row), axis=-1)
        )
        read = MultiHeadAttention(
            name="read",
            num_heads=self.cfg.num_heads,
            qk_size=self.cfg.qk_size,
            v_size=self.cfg.v_size,
            model_size=model_size,
            qk_layer_norm=self.cfg.qk_layer_norm,
            use_bias=self.cfg.use_bias,
            dtype=self.dtype,
        )
        attended = read(
            q=RMSNorm()(query)[None],
            kv=RMSNorm()(rows),
            mask=jnp.ones((1, rows.shape[0]), bool),
        )[0]
        return MLP(**self.cfg.mlp.to_dict())(
            jnp.concatenate((query, attended), axis=-1)
        )


class CandidateDecoderBlock(nn.Module):
    """One decoder block over the candidate tokens: causal self-attention
    over the tokens, cross-attention over the state rows, SwiGLU FFW, all
    pre-RMSNorm with plain residuals. A `TrunkBlock` over [rows ; tokens]
    with a block mask would compute the same function while re-encoding
    73 state rows on every one of the J sequential draws (5-9x the
    search-time FLOPs) -- this block's queries are the tokens only."""

    cfg: ConfigDict

    @nn.compact
    def __call__(self, carry, masks):
        tokens, rows = carry
        causal_mask, cross_mask = masks
        attention = dict(
            num_heads=self.cfg.num_heads,
            qk_size=self.cfg.qk_size,
            v_size=self.cfg.v_size,
            model_size=self.cfg.model_size,
            qk_layer_norm=self.cfg.qk_layer_norm,
            use_bias=self.cfg.use_bias,
            dtype=tokens.dtype,
        )
        normed = RMSNorm()(tokens)
        tokens = tokens + MultiHeadAttention(name="self_attention", **attention)(
            q=normed, kv=normed, mask=causal_mask
        )
        tokens = tokens + MultiHeadAttention(name="cross_attention", **attention)(
            q=RMSNorm()(tokens), kv=RMSNorm()(rows), mask=cross_mask
        )
        tokens = tokens + FFWMLP(
            hidden_size=self.cfg.hidden_size, use_bias=self.cfg.use_bias, name="ffw"
        )(RMSNorm()(tokens))
        return (tokens, rows), None


class CandidateGenerator(nn.Module):
    """rho(u | h) and the ordered conditionals: token j carries the code
    drawn at slot j - 1 (slot 0 a learned start token) plus a slot
    embedding; `num_blocks` decoder blocks scanned as the trunk is; one
    Dense head to the alphabet. Teacher-forced: all J conditionals in one
    pass through the causal mask."""

    cfg: ConfigDict
    num_candidates: int
    num_classes: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, rows: jax.Array, code_embeddings: jax.Array) -> jax.Array:
        """`code_embeddings` (J - 1, D): the embeddings of the codes drawn
        at slots 0 .. J - 2, in order. Returns (J, C) logits."""
        model_size = rows.shape[-1]
        init = nn.initializers.variance_scaling(1.0, "fan_in", "normal")
        start = self.param("start_embedding", init, (1, model_size))
        positions = self.param(
            "position_embedding", init, (self.num_candidates, model_size)
        )
        tokens = jnp.concatenate(
            (start.astype(self.dtype), code_embeddings.astype(self.dtype)), axis=0
        )
        tokens = tokens + positions.astype(self.dtype)
        num_tokens = self.num_candidates
        causal_mask = jnp.tril(jnp.ones((num_tokens, num_tokens), bool))
        cross_mask = jnp.ones((num_tokens, rows.shape[0]), bool)
        block = nn.remat(
            CandidateDecoderBlock, policy=jax.checkpoint_policies.nothing_saveable
        )
        (tokens, _), _ = nn.scan(
            block,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=nn.broadcast,
            length=self.cfg.num_blocks,
        )(self.cfg, name="blocks")((tokens, rows), (causal_mask, cross_mask))
        return nn.Dense(self.num_classes, dtype=self.dtype, name="head")(
            RMSNorm()(tokens)
        )


class TransitionModel(nn.Module):
    cfg: ConfigDict
    dtype: jnp.dtype

    @property
    def has_code(self) -> bool:
        return self.cfg.code_groups > 0

    @property
    def unroll_steps(self) -> int:
        return self.cfg.unroll_steps

    def setup(self):
        model_size = self.cfg.block.model_size
        init = nn.initializers.variance_scaling(1.0, "fan_in", "normal")
        self.slot_embedding = self.param(
            "slot_embedding", init, (NUM_POLICY_READABLE_ROWS, model_size)
        )
        self.action_table = self.param(
            "action_table", init, (self.cfg.action_classes, model_size)
        )
        self.condition_type_embedding = self.param(
            "condition_type_embedding", nn.initializers.zeros_init(), (2, model_size)
        )
        self.dynamics_blocks = Trunk(self.cfg.block, name="dynamics_blocks")
        self.dynamics_out_proj = nn.Dense(
            model_size,
            kernel_init=nn.initializers.zeros_init(),
            use_bias=False,
            dtype=self.dtype,
            name="dynamics_out_proj",
        )
        if self.has_code:
            code_groups = self.cfg.code_groups
            assert model_size % code_groups == 0
            self.code_table = self.param(
                "code_table",
                nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0),
                (code_groups, self.cfg.code_classes, model_size // code_groups),
            )
            self.chance_token_proj = nn.Dense(
                model_size, dtype=self.dtype, name="chance_token_proj"
            )
            self.row_read = RowRead(
                self.cfg.row_read_width, self.dtype, name="row_read"
            )
            # `_latent_net`: the nets that read the cell rows were renamed
            # so the by-path checkpoint merge inits them fresh -- their
            # first kernel's width changed and the later layers would
            # otherwise resume on features that no longer exist.
            self.prior_latent_net = MLP(**self.cfg.prior.mlp.to_dict())
            self.posterior_latent_net = MLP(**self.cfg.posterior.mlp.to_dict())
        self.action_encoder = ActionEncoder(
            self.cfg.action_encoder, self.dtype, name="action_encoder"
        )
        self.candidate_generator = CandidateGenerator(
            self.cfg.generator,
            self.cfg.num_candidates,
            self.cfg.action_classes,
            self.dtype,
            name="candidate_generator",
        )
        self.ground_delta_head = MLP(
            **self.cfg.ground.mlp.to_dict(), final_kernel_init=nn.initializers.zeros
        )
        self.cls_head = MLP(**self.cfg.cls_head.mlp.to_dict())
        self.terminal_outcome_head = MLP(**self.cfg.terminal_outcome.mlp.to_dict())

    def code_embedding(self, code_one_hot: jax.Array) -> jax.Array:
        """(G, K) one-hot -> the concatenated code-table vector (D,)."""
        table = self.code_table.astype(self.dtype)
        return jnp.einsum("gk,gkd->gd", code_one_hot.astype(self.dtype), table).reshape(
            -1
        )

    def action_embedding(self, action_one_hot: jax.Array) -> jax.Array:
        return action_one_hot.astype(self.dtype) @ self.action_table.astype(self.dtype)

    def code_logits(self, net: MLP, features: jax.Array) -> jax.Array:
        return (
            net(features)
            .astype(jnp.float32)
            .reshape(self.cfg.code_groups, self.cfg.code_classes)
        )

    def imagine(
        self,
        rows: jax.Array,
        action_one_hot: jax.Array,
        code_one_hot: jax.Array | None,
    ) -> jax.Array:
        """One step of g over one (73, D) sequence: the rows plus their
        slot embedding, an action token and (with a code) a chance token,
        through the dynamics blocks under an all-True mask; the residual
        uses the ORIGINAL rows, so with `dynamics_out_proj` at zero the
        output IS the input. No validity, legality or future input."""
        types = self.condition_type_embedding.astype(self.dtype)
        tokens = [
            rows + self.slot_embedding.astype(self.dtype),
            (self.action_embedding(action_one_hot) + types[0])[None],
        ]
        if code_one_hot is not None:
            chance = self.chance_token_proj(self.code_embedding(code_one_hot))
            tokens.append((chance + types[1])[None])
        sequence = jnp.concatenate(tokens, axis=0)
        num_tokens = sequence.shape[0]
        hidden = self.dynamics_blocks(
            sequence,
            jnp.ones((num_tokens,), bool),
            jnp.ones((num_tokens, num_tokens), bool),
        )
        return rows + self.dynamics_out_proj(hidden[: rows.shape[0]])

    def action_rows(self, rows: jax.Array, action_cell: jax.Array):
        return chosen_bank_rows(
            rows[PRIVATE_ROWS],
            rows[MOVE_ROWS],
            rows[TARGET_ROWS],
            action_cell.reshape(()),
        )

    def action_logits(self, rows: jax.Array, action_cell: jax.Array) -> jax.Array:
        """q(u | h, a) as (C,) f32 logits."""
        src_row, tgt_row = self.action_rows(rows, action_cell)
        return self.action_encoder(rows, src_row, tgt_row).astype(jnp.float32)

    def prior_features(self, rows: jax.Array, action_one_hot: jax.Array):
        return jnp.concatenate(
            (self.row_read(rows), self.action_embedding(action_one_hot)), axis=-1
        )

    def prior(self, rows: jax.Array, action_one_hot: jax.Array) -> jax.Array:
        """The rollout-side chance distribution, (G, K) f32 logits."""
        return self.code_logits(
            self.prior_latent_net, self.prior_features(rows, action_one_hot)
        )

    def posterior(
        self, rows: jax.Array, action_one_hot: jax.Array, next_rows: jax.Array
    ) -> jax.Array:
        """LEARNER-ONLY: reads the real next rows (stop-gradient at the
        call site) as the per-row CHANGE from `rows` -- an appearing row
        is a change like any other -- through the read the prior applies
        to the state."""
        features = jnp.concatenate(
            (
                self.prior_features(rows, action_one_hot),
                self.row_read(next_rows - rows),
            ),
            axis=-1,
        )
        return self.code_logits(self.posterior_latent_net, features)

    def candidate_logits(self, rows: jax.Array, codes: jax.Array) -> jax.Array:
        """Teacher-forced generator logits (J, C) f32 given the codes at
        slots 0 .. J - 1 (slot j reads the codes before it)."""
        embeddings = self.action_table.astype(self.dtype)[codes[:-1]]
        return self.candidate_generator(rows, embeddings).astype(jnp.float32)

    def generate(
        self, rows: jax.Array, rng: jax.Array | None, mass_threshold: float
    ) -> Candidates:
        """J distinct codes drawn autoregressively without replacement
        inside the node's own support set (the smallest rho prefix
        holding `mass_threshold`); with a key each draw is a categorical
        over the support minus the prefix, without one the argmax. Slots
        past the support count are unoccupied (and still distinct)."""
        num_draws = self.cfg.num_candidates
        num_codes = self.cfg.action_classes
        codes = jnp.zeros((num_draws,), jnp.int32)
        drawn = jnp.zeros((num_codes,), bool)
        rho = jax.nn.softmax(self.candidate_logits(rows, codes)[0], axis=-1)
        support = support_set(rho, mass_threshold)
        keys = [None] * num_draws
        if rng is not None:
            keys = list(jax.random.split(rng, num_draws))
        log_draws = []
        for slot in range(num_draws):
            logits = self.candidate_logits(rows, codes)[slot]
            allowed = support & jnp.logical_not(drawn)
            allowed = jnp.where(allowed.any(), allowed, jnp.logical_not(drawn))
            log_conditional = masked_log_softmax(logits, allowed)
            index = draw_or_mode(log_conditional, keys[slot])
            codes = codes.at[slot].set(index)
            drawn = drawn.at[index].set(True)
            log_draws.append(log_conditional[index])
        occupied = jnp.arange(num_draws) < support.sum()
        rho_at_codes = rho[codes]
        return Candidates(
            codes=codes,
            occupied=occupied,
            log_draw_conditionals=jnp.stack(log_draws),
            rho_at_codes=rho_at_codes,
            support_mask=support,
            retained_mass=jnp.sum(rho_at_codes, where=occupied),
        )

    def node_readout(self, rows: jax.Array) -> NodeReadout:
        """The imagined node's own readers off its CLS row: request kind
        + done logits, and the conditional terminal-outcome logits."""
        cls_logits = self.cls_head(rows[CLS_ROW]).astype(jnp.float32)
        terminal = self.terminal_outcome_head(rows[CLS_ROW]).astype(jnp.float32)
        return NodeReadout(
            kind_logits=cls_logits[:-1],
            done_logit=cls_logits[-1],
            terminal_logits=terminal,
        )

    def continue_prob(self, rows: jax.Array) -> jax.Array:
        """P(the game continues past this node), from the done reader."""
        return jax.nn.sigmoid(-self.node_readout(rows).done_logit)

    def expected_terminal_outcome(self, rows: jax.Array) -> jax.Array:
        """E[outcome | this node ends the game] on the value support."""
        probs = jax.nn.softmax(self.node_readout(rows).terminal_logits, axis=-1)
        return probs @ jnp.asarray(CAT_VF_SUPPORT, jnp.float32)

    def real_state(
        self,
        rows: jax.Array,
        taken_cell: jax.Array,
        legal: jax.Array,
        log_policy: jax.Array,
        rng: jax.Array | None,
    ) -> RealState:
        """One observed state's legal set through the encoder (the only
        place a concrete legal set enters), the base policy's latent
        target and its teacher order."""
        cells, cell_valid, taken_index, overflow = legal_enumeration(
            legal, taken_cell, self.cfg.max_cells
        )
        logits = jax.vmap(self.action_logits, in_axes=(None, 0))(rows, cells)
        probs = unimix_probs(logits)
        log_pi = jnp.where(
            cell_valid, log_policy[cells].astype(jnp.float32), NEGATIVE_INFINITY_LOGIT
        )
        pi = jnp.exp(jax.nn.log_softmax(log_pi))
        target = jax.lax.stop_gradient(pi @ probs)
        uniform = jnp.full_like(target, 1.0 / target.shape[0])
        target = jnp.where(cell_valid.any(), target, uniform)
        support = support_set(target, self.cfg.mass_threshold)
        teacher_log = jnp.where(support, jnp.log(target), -jnp.inf)
        teacher_codes = gumbel_top_k(teacher_log, self.cfg.num_candidates, rng)
        return RealState(
            taken_cell=taken_cell,
            cells=cells,
            cell_valid=cell_valid,
            taken_index=taken_index,
            overflow=overflow,
            logits=logits,
            code_target=target,
            support_mask=support,
            teacher_codes=teacher_codes,
            taken_code_probs=jax.lax.stop_gradient(probs[taken_index]),
        )

    def _node(self, state: jax.Array, real: RealState, rng, is_root: bool):
        """The readers at one node that need the real step's action set:
        the generator teacher-forced on the real teacher order, the
        encoder on the taken cell (the root reuses the real-state logits;
        an imagined node recomputes them -- the alignment read) and the
        latent action drawn from it for the next transition."""
        generator_logits = self.candidate_logits(state, real.teacher_codes)
        if is_root:
            taken_logits = real.logits[real.taken_index]
        else:
            taken_logits = self.action_logits(state, real.taken_cell)
        action_one_hot = straight_through_sample(unimix_probs(taken_logits), rng)
        return generator_logits, taken_logits, action_one_hot

    def _transition(
        self,
        state: jax.Array,
        action_one_hot: jax.Array,
        next_rows: jax.Array,
        rng: jax.Array | None,
        with_prior_decode: bool,
    ) -> StepLatents:
        """One imagined step off `state` along `action_one_hot`: the chance
        code inferred by the posterior from the real next rows, and the
        state g writes under it. Without a code the latents read zero and g
        runs on the action token alone."""
        code_shape = (self.cfg.code_groups, self.cfg.code_classes)
        pred_prior = None
        if self.has_code:
            prior_logits = self.prior(state, action_one_hot)
            post_logits = self.posterior(state, action_one_hot, next_rows)
            post_one_hot = straight_through_sample(unimix_probs(post_logits), rng)
            pred = self.imagine(state, action_one_hot, post_one_hot)
            if with_prior_decode:
                prior_mode = jax.nn.one_hot(
                    jnp.argmax(prior_logits, axis=-1),
                    self.cfg.code_classes,
                    dtype=jnp.float32,
                )
                pred_prior = jax.lax.stop_gradient(
                    self.imagine(
                        state, jax.lax.stop_gradient(action_one_hot), prior_mode
                    )
                )
        else:
            prior_logits = jnp.zeros(code_shape, jnp.float32)
            post_logits = jnp.zeros(code_shape, jnp.float32)
            post_one_hot = jnp.zeros(code_shape, jnp.float32)
            pred = self.imagine(state, action_one_hot, None)
            if with_prior_decode:
                pred_prior = jax.lax.stop_gradient(pred)
        return StepLatents(
            pred=pred,
            pred_prior=pred_prior,
            prior_logits=prior_logits,
            post_logits=post_logits,
            post_one_hot=post_one_hot,
        )

    def _unroll(self, real_rows, reals: RealState, node_keys, transition_keys):
        """One start step: hhat_0 = h_t, K transitions along the recorded
        actions, the readers at every node. Leading axis K + 1 on the real
        inputs (the real steps t .. t + K), gathered by the caller."""
        num_steps = self.unroll_steps
        nodes = []
        steps = []
        first = None
        state = real_rows[0]
        for offset in range(num_steps + 1):
            real = jax.tree.map(lambda leaf: leaf[offset], reals)
            generator_logits, align_logits, action_one_hot = self._node(
                state, real, node_keys[offset], is_root=offset == 0
            )
            nodes.append(
                NodeOutputs(
                    generator_logits=generator_logits,
                    teacher_codes=real.teacher_codes,
                    generator_target=real.code_target,
                    support_mask=real.support_mask,
                    node_overflow=real.overflow,
                    align_logits=align_logits,
                    align_target=real.taken_code_probs,
                )
            )
            if offset == num_steps:
                break
            # MuZero's 0.5 scale on the gradient into the unrolled state;
            # the forward value is unchanged.
            state_in = state
            if offset > 0:
                state_in = 0.5 * state + 0.5 * jax.lax.stop_gradient(state)
            latents = self._transition(
                state_in,
                action_one_hot,
                real_rows[offset + 1],
                transition_keys[offset],
                with_prior_decode=offset == 0,
            )
            readout = self.node_readout(latents.pred)
            if offset == 0:
                first = FirstStepOutputs(
                    pred_prior=latents.pred_prior,
                    ground=self.ground_delta_head(latents.pred[DYNAMICS_TARGET_ROWS]),
                    ground_prior=jax.lax.stop_gradient(
                        self.ground_delta_head(latents.pred_prior[DYNAMICS_TARGET_ROWS])
                    ),
                )
            steps.append(
                StepOutputs(
                    action_one_hot=action_one_hot,
                    pred=latents.pred,
                    prior_logits=latents.prior_logits,
                    post_logits=latents.post_logits,
                    post_one_hot=latents.post_one_hot,
                    kind_logits=readout.kind_logits,
                    done_logit=readout.done_logit,
                    terminal_logits=readout.terminal_logits,
                )
            )
            state = latents.pred
        assert first is not None, "unroll_steps >= 1 guarantees a first transition"
        root = jax.tree.map(lambda leaf: leaf[0], reals)
        return TransitionOutput(
            root=RootOutputs(
                action_logits=root.logits,
                action_cells=root.cells,
                action_cell_valid=root.cell_valid,
                action_taken_index=root.taken_index,
                action_overflow=root.overflow,
            ),
            nodes=_stack(nodes),
            steps=_stack(steps),
            first=first,
        )

    def __call__(
        self,
        rows: jax.Array,
        action_cell: jax.Array,
        legal: jax.Array,
        log_policy: jax.Array,
    ) -> TransitionOutput:
        """Leading axis T on every input: rows (T, 73, D), action_cell
        (T,), legal (T, 295) bool, log_policy (T, 295) (the base policy,
        sg'd by the caller). Each start step is unrolled `unroll_steps`
        transitions along the recorded actions, with the real steps
        gathered by their POSITIONAL successors (the last step to itself;
        the loss masks the out-of-range offsets). A "sampling" rng, when
        supplied, draws the teacher orders, the action codes and the
        posterior codes; without it (init, probes, the offline harness)
        every draw is its mode / top-J."""
        num_rows = rows.shape[0]
        num_steps = self.unroll_steps
        assert num_steps >= 1, "unroll_steps: 1 is the single-step model"
        # Without keys the pytrees carry no leaves, so the vmaps pass them
        # through unchanged and every draw below sees `None`.
        teacher_keys = None
        node_keys = [None] * (num_steps + 1)
        transition_keys = [None] * num_steps
        if self.has_rng("sampling"):
            key = self.make_rng("sampling")
            teacher_key, node_key, transition_key = jax.random.split(key, 3)
            teacher_keys = jax.random.split(teacher_key, num_rows)
            node_keys = jax.random.split(node_key, (num_rows, num_steps + 1))
            transition_keys = jax.random.split(transition_key, (num_rows, num_steps))
        reals = jax.vmap(self.real_state)(
            rows, action_cell, legal, log_policy, teacher_keys
        )
        offsets = [
            jnp.minimum(jnp.arange(num_rows) + offset, num_rows - 1)
            for offset in range(num_steps + 1)
        ]
        gathered_rows = _gather_offsets(rows, offsets)
        gathered_reals = jax.tree.map(
            lambda leaf: _gather_offsets(leaf, offsets), reals
        )
        return jax.vmap(self._unroll, in_axes=(1, 1, 0, 0), out_axes=VMAP_AXES)(
            gathered_rows, gathered_reals, node_keys, transition_keys
        )
