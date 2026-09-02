"""Recurrent per-Pokemon history encoder over battle history edges.

Replaces the (disabled) transformer-over-history path. Instead of attending
over up to NUM_HISTORY timesteps, a bank of 12 recurrent states -- one per
public Pokemon slot -- is scanned once along the history axis. Each history
step scatters its edge embeddings into the slots named by
ENTITY_EDGE_FEATURE__ENTITY_IDX, so a slot's state only advances when
something happened to that Pokemon. Carry is O(12 * entity_size) regardless
of history length.

The per-request states are residual-injected into the encoder's public
entity tokens; the per-request states are trained end-to-end by the
task gradients alone.
"""

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike
from ml_collections import ConfigDict

from rl.environment.protos.enums_pb2 import BattlemajorargsEnum
from rl.environment.protos.features_pb2 import EntityEdgeFeature, FieldFeature
from rl.model.constants import NUM_PUBLIC_SLOTS
from rl.model.modules import MultiHeadAttention, create_attention_mask, layer_norm

# The scan is latency-bound (hundreds of tiny sequential kernels), not
# FLOP-bound; unrolling fuses steps per launch. Re-tuned 8 -> 32 for the
# slimmer 2026-09-01 step (see __call__'s precompute note): with fewer ops
# per step, more steps fit a launch -- measured 13.82 -> 13.22ms on the
# H=512 scan.
SCAN_UNROLL = 32

# The field history carries THREE states, mirroring the env-step field triple
# that _embed_field already produces (2026-08-28). Hazards are side-differenced
# — spikes on my side and spikes on theirs are opposite facts — and collapsing
# them into one vector with a Dense meant the recurrent field memory could only
# hold their mixture. Row order matches _embed_field's stack.
NUM_FIELD_ROWS = 3
FIELD_ROW_GLOBAL, FIELD_ROW_MINE, FIELD_ROW_THEIRS = 0, 1, 2
# ENTITY_PUBLIC_NODE_FEATURE__SIDE == 1 is mine (service isMySide).
SIDE_MINE = 1

# All EIGHT columns the service writes (state.ts maxRelevant = 8). Until
# 2026-09-01 this listed only IDX0..3, so any step touching more than four
# entities -- spread moves, hazard cascades -- had rows 5-8 silently dropped
# before the scatter ever saw them.
_RELEVANT_ENTITY_FEATURES = np.array(
    [
        FieldFeature.Value(f"FIELD_FEATURE__RELEVANT_ENTITY_IDX{index}")
        for index in range(8)
    ]
)


def major_arg_step_mask(history_field: jax.Array, edge_cache: jax.Array) -> jax.Array:
    """(H,) bool: history steps that carry at least one battle major arg.

    Mirrors the relevant-edge gather of PerSlotHistoryEncoder.__call__: a
    step's edges are the cache rows named by its RELEVANT_ENTITY_IDX
    columns, capped by NUM_RELEVANT. A step counts as major when any such
    edge's MAJOR_ARG is a real protocol arg (anything past the
    UNSPECIFIED/NULL/PAD sentinels — moves, switches, faints, cant, ...).
    These are the integrated history critic's supervision points, matching
    the offline critic's convention of scoring at decision-bearing events
    rather than every residual/chip line.
    """
    relevant = history_field[:, _RELEVANT_ENTITY_FEATURES]  # (H, K)
    num_relevant = history_field[:, FieldFeature.FIELD_FEATURE__NUM_RELEVANT]
    edge_mask = np.arange(relevant.shape[1])[None] < num_relevant[:, None]
    major = jnp.take(
        edge_cache[:, EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG],
        relevant,
        axis=0,
    )  # (H, K)
    is_real = major > BattlemajorargsEnum.BATTLEMAJORARGS_ENUM___PAD
    return (edge_mask & is_real).any(axis=-1)


@chex.dataclass
class PerSlotHistoryOutput:
    # Per-history-step snapshots: (H, 12, D) / (H, D).
    slot_snapshots: ArrayLike = ()
    field_snapshots: ArrayLike = ()
    # Latest raw node embedding per slot as of each step (H, 12, D): the
    # entity's current snapshot, unmixed by GRU gating — what a hand
    # evaluator reads. Parameter-free carry.
    node_snapshots: ArrayLike = ()
    step_valid: ArrayLike = ()
    step_request_count: ArrayLike = ()


class HistoryAttentionPool(nn.Module):
    """Cross-attention pooling of the recurrent history states into a fixed
    bank of learned latent summaries.

    A set of num_latents learned queries attends over the 15 history tokens
    (12 slot states + the 3 field states), yielding (num_latents, D) latents.
    Shared module code, separately trained instances: the offline critic
    reads the flattened latents through its linear probe; the RL trunk
    reads its own instance's latents as extra history-context tokens,
    trained from scratch by RL gradients.
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(
        self, tokens: jax.Array, token_mask: jax.Array | None = None
    ) -> jax.Array:
        """(S, D) history tokens -> (num_latents, D) latent summaries.
        token_mask (S,) restricts which tokens are readable — e.g. the
        offline critic pools my-side and opponent-side slots separately
        (shared params) for its antisymmetric outcome readout."""
        pcfg = self.cfg.history_pool
        if token_mask is None:
            token_mask = jnp.ones(tokens.shape[0], dtype=jnp.bool_)
        queries = self.param(
            "latent_queries",
            nn.initializers.normal(0.02),
            (pcfg.num_latents, self.cfg.entity_size),
        ).astype(tokens.dtype)
        attended = MultiHeadAttention(
            name="latent_cross",
            num_heads=pcfg.num_heads,
            qk_size=pcfg.qk_size,
            v_size=pcfg.qk_size,
            model_size=self.cfg.entity_size,
            use_bias=pcfg.use_bias,
            dtype=tokens.dtype,
        )(
            q=layer_norm(queries),
            kv=layer_norm(tokens),
            mask=create_attention_mask(
                jnp.ones(queries.shape[0], dtype=jnp.bool_),
                token_mask,
            ),
        )
        return queries + attended


class NodeHistoryRead(nn.Module):
    """Residual cross-read of the diaries by the photos.

    Each slot's current snapshot (node state) queries the recurrent slot
    states + field state — the same current-obs-reads-history pattern as
    the RL trunk's gated history_cross rounds. The residual gate is
    zero-init, so at initialization the output IS the raw snapshots
    (hand-rule parity is the floor) and history context blends in only as
    training finds it useful.
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(
        self,
        node_states: jax.Array,
        slot_states: jax.Array,
        field_state: jax.Array,
    ) -> jax.Array:
        """(12, D) snapshots, (12, D) slot states, (3, D) field -> (12, D)."""
        pcfg = self.cfg.history_pool
        kv = jnp.concatenate((slot_states, field_state), axis=0)
        gate = self.param("gate", nn.initializers.zeros_init(), (1,)).astype(
            node_states.dtype
        )
        attended = MultiHeadAttention(
            name="diary_cross",
            num_heads=pcfg.num_heads,
            qk_size=pcfg.qk_size,
            v_size=pcfg.qk_size,
            model_size=self.cfg.entity_size,
            use_bias=pcfg.use_bias,
            dtype=node_states.dtype,
        )(
            q=layer_norm(node_states),
            kv=layer_norm(kv),
            mask=create_attention_mask(
                jnp.ones(node_states.shape[0], dtype=jnp.bool_),
                jnp.ones(kv.shape[0], dtype=jnp.bool_),
            ),
        )
        return node_states + gate * attended


class _GateParams(nn.Module):
    """Raw kernel(+bias) with EXACTLY nn.Dense's param layout and init, so a
    module tree built from these loads a checkpoint written by flax's
    GRUCell (whose gates are Dense children named ir/iz/in/hr/hz/hn)."""

    features: int
    in_features: int
    use_bias: bool
    kernel_init: nn.initializers.Initializer

    @nn.compact
    def __call__(self) -> tuple[jax.Array, jax.Array | None]:
        kernel = self.param(
            "kernel", self.kernel_init, (self.in_features, self.features)
        )
        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros_init(), (self.features,))
        else:
            bias = None
        return kernel, bias


class SplitGRUCell(nn.Module):
    """flax.linen.GRUCell with the input-side gate GEMMs SPLITTABLE.

    Same math, same param tree (children ir/iz/in with bias + lecun init,
    hr/hz/hn orthogonal recurrent init, hn biased -- mirrored from the
    installed flax source), but the input projections can be applied to a
    ROW-SLICE of each kernel. That is what lets the scan hoist the
    precomputable part of its input over the whole history as one batched
    GEMM and keep only the carry-dependent tail inside the serial chain --
    the standard cuDNN-RNN restructure, in pure JAX (2026-09-01; table in
    PerSlotHistoryEncoder.__call__).
    """

    features: int
    input_features: int
    dtype: jnp.dtype

    def setup(self):
        lecun = nn.initializers.lecun_normal()
        orthogonal = nn.initializers.orthogonal()

        def input_gate(name):
            return _GateParams(
                features=self.features,
                in_features=self.input_features,
                use_bias=True,
                kernel_init=lecun,
                name=name,
            )

        def hidden_gate(name, use_bias):
            return _GateParams(
                features=self.features,
                in_features=self.features,
                use_bias=use_bias,
                kernel_init=orthogonal,
                name=name,
            )

        self.gate_ir = input_gate("ir")
        self.gate_iz = input_gate("iz")
        self.gate_in = input_gate("in")
        self.gate_hr = hidden_gate("hr", use_bias=False)
        self.gate_hz = hidden_gate("hz", use_bias=False)
        self.gate_hn = hidden_gate("hn", use_bias=True)

    def _input_slice(self, xs, start, with_bias):
        """(i_r, i_z, i_n) contributions of input columns [start:start+W),
        where W = xs.shape[-1]. Bias is added iff `with_bias` -- exactly
        once across the slices of a full input."""
        width = xs.shape[-1]
        xs = xs.astype(self.dtype)
        outs = []
        for gate in (self.gate_ir, self.gate_iz, self.gate_in):
            kernel, bias = gate()
            part = xs @ kernel[start : start + width].astype(self.dtype)
            if with_bias:
                part = part + bias.astype(self.dtype)
            outs.append(part)
        return tuple(outs)

    def project_inputs(self, xs):
        """The hoisted half: gate contributions of the leading input columns
        (all of them if xs is full width), bias included."""
        return self._input_slice(xs, 0, with_bias=True)

    def project_carry_inputs(self, xs, start):
        """The serial half: gate contributions of the carry-dependent input
        columns starting at `start`. No bias -- project_inputs carried it."""
        return self._input_slice(xs, start, with_bias=False)

    def advance(self, h, input_gates):
        """The recurrent half, given precomputed input-side gates."""
        i_r, i_z, i_n = input_gates
        h = h.astype(self.dtype)

        def hidden(gate):
            kernel, bias = gate()
            out = h @ kernel.astype(self.dtype)
            if bias is not None:
                out = out + bias.astype(self.dtype)
            return out

        r = nn.sigmoid(i_r + hidden(self.gate_hr))
        z = nn.sigmoid(i_z + hidden(self.gate_hz))
        n = nn.tanh(i_n + r * hidden(self.gate_hn))
        return (1.0 - z) * n + z * h


class PerSlotHistoryEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        entity_size = self.cfg.entity_size
        init = nn.initializers.normal(0.02)
        # Slots are keyed by revelation order across BOTH sides, so a slot's
        # side is dynamic — one shared initial state; side information enters
        # through the node snapshots in the messages.
        self.initial_slot_state = self.param(
            "initial_slot_state", init, (1, entity_size)
        )
        self.initial_field_state = self.param(
            "initial_field_state", init, (NUM_FIELD_ROWS, entity_size)
        )
        # Projects [node ; edge ; field ; side ; src_node ; src_edge ;
        # is_src] into a directed slot message (the last three blocks are
        # the 2026-09-01 TGN source half).
        self.message_projection = nn.Dense(
            features=entity_size,
            use_bias=False,
            dtype=self.cfg.dtype,
            name="message_projection",
        )
        # SplitGRUCell, not nn.GRUCell: same math, same param tree, but the
        # input-side gate GEMMs can be hoisted out of the scan (see
        # __call__). The slot input is [messages ; field_vec ; flat_field]
        # = 5D wide, of which the first 2D are precomputable per step and
        # the trailing 3D (the three field states) depend on the carry. A
        # mean over the other slots' states used to ride here too (the
        # "gestalt"); it was redundant with flat_field -- which is fed the
        # SUM of every message -- and with the trunk's read-time attention
        # over the HISTORY_ENTITY rows, and it was on the serial tail
        # (deleted 2026-09-02).
        self.slot_cell = SplitGRUCell(
            entity_size,
            input_features=5 * entity_size,
            dtype=self.cfg.dtype,
            name="slot_cell",
        )
        self.field_cell = SplitGRUCell(
            entity_size,
            input_features=2 * entity_size,
            dtype=self.cfg.dtype,
            name="field_cell",
        )

    def initial_state(self) -> tuple[jax.Array, jax.Array]:
        h_slots = jnp.repeat(self.initial_slot_state, NUM_PUBLIC_SLOTS, axis=0).astype(
            self.cfg.dtype
        )
        h_field = self.initial_field_state.astype(self.cfg.dtype)
        return h_slots, h_field

    def _scan_step(self, carry, xs):
        """One history step -- ONLY the carry-dependent work.

        Everything else that used to live here (the per-step segment_sums
        into slot messages, the latest-node bookkeeping, the input-side gate
        GEMMs) is precomputed over the whole history in __call__ as batched
        ops: ~35 GFLOP delivered as one GEMM stream instead of hundreds of
        dependent 12-row kernels. What remains per step is irreducibly
        serial: the carry-dependent slice of the slot input (flat_field,
        shared by all 12 slots, so ONE 3D-wide row), the recurrent gate
        GEMMs, and the gating.
        """
        h_slots, h_field = carry
        slot_pre, field_gates, touched, valid = xs

        # The carry-dependent tail of the slot input: the three field
        # states, identical for every slot -- projected once as a single
        # row and broadcast, where the old cell multiplied the same values
        # through the kernel 12 times.
        flat_field = h_field.reshape(-1)[None]
        carry_gates = self.slot_cell.project_carry_inputs(
            flat_field, start=2 * self.cfg.entity_size
        )
        slot_gates = tuple(pre + tail for pre, tail in zip(slot_pre, carry_gates))

        new_slots = self.slot_cell.advance(h_slots, slot_gates)
        slot_gate = (touched * valid)[..., None]
        h_slots = slot_gate * new_slots + (1 - slot_gate) * h_slots

        new_field = self.field_cell.advance(h_field, field_gates)
        h_field = valid * new_field + (1 - valid) * h_field

        carry = (h_slots, h_field)
        return carry, carry

    def __call__(
        self,
        history_field: jax.Array,
        node_embedding_cache: jax.Array,
        edge_embedding_cache: jax.Array,
        edge_slot_ids: jax.Array,
        edge_major_args: jax.Array,
        node_sides: jax.Array,
        field_step_embeddings: jax.Array,
        field_row_embeddings: jax.Array,
        step_request_count: jax.Array,
        step_valid: jax.Array,
    ) -> PerSlotHistoryOutput:
        """Scan the slot bank along the history axis.

        Args:
            history_field: (H, NUM_FIELD_FEATURES) raw int history rows.
            node_embedding_cache: (P, D) embedded public-entity cache rows.
            edge_embedding_cache: (P, D) embedded edge cache rows.
            edge_slot_ids: (P,) ENTITY_EDGE_FEATURE__ENTITY_IDX per cache row.
            edge_major_args: (P,) ENTITY_EDGE_FEATURE__MAJOR_ARG per cache
                row -- what identifies a step's SOURCE rows (the mover).
            node_sides: (P,) relative side (1 = mine) per cache row.
            field_step_embeddings: (H, D) pooled field embedding per step —
                the message/slot-input view.
            field_row_embeddings: (H, 3, D) the (global, mine, theirs) field
                token triple per step, one input per field state.
            step_request_count: (H,) request count of each history step.
            step_valid: (H,) bool.
        """
        relevant = history_field[:, _RELEVANT_ENTITY_FEATURES]  # (H, K)
        num_relevant = history_field[:, FieldFeature.FIELD_FEATURE__NUM_RELEVANT]
        edge_mask = jnp.arange(relevant.shape[1])[None] < num_relevant[:, None]
        step_valid = step_valid & edge_mask.any(axis=-1)

        node_embeddings = jnp.take(node_embedding_cache, relevant, axis=0)  # (H, K, D)
        edge_embeddings = jnp.take(edge_embedding_cache, relevant, axis=0)  # (H, K, D)
        slot_ids = jnp.take(edge_slot_ids, relevant, axis=0).clip(
            0, NUM_PUBLIC_SLOTS - 1
        )  # (H, K)

        # Perspective is otherwise a whisper in these inputs (a single
        # additive bias inside the node embeddings — edges and most field
        # rows are perspective-blind). The outcome is inherently a
        # side-differenced quantity, so hand each message an explicit
        # "mine / theirs" tag instead of making the model excavate it.
        edge_sides = jnp.take(node_sides, relevant, axis=0)  # (H, K)
        side_onehot = jax.nn.one_hot(
            edge_sides, 2, dtype=node_embeddings.dtype
        )  # (H, K, 2)
        edge_is_mine = edge_sides == SIDE_MINE

        # ---- the directed edge (TGN message, 2026-09-01) ------------------
        # A step's SOURCE rows are the ones carrying a real major arg (the
        # mover of a move/switch/faint/cant); their masked-mean node+edge
        # latents ride EVERY message of the step, so mover and target
        # finally coexist in one vector -- the relation "move X did N to Y"
        # the per-slot scatter used to destroy. TGN proper uses the source's
        # MEMORY here; that is carry-dependent and hence serial, so against
        # the ~26us/step dependency-latency bound the source's raw cache
        # embedding (its revealed state at event time) stands in, keeping
        # the whole message batch precomputable.
        edge_majors = jnp.take(edge_major_args, relevant, axis=0)  # (H, K)
        is_src = (
            edge_majors > BattlemajorargsEnum.BATTLEMAJORARGS_ENUM___PAD
        ) & edge_mask
        src_weights = is_src.astype(node_embeddings.dtype)[..., None]
        src_denom = src_weights.sum(axis=-2).clip(min=1)
        src_node = (node_embeddings * src_weights).sum(axis=-2) / src_denom
        src_edge = (edge_embeddings * src_weights).sum(axis=-2) / src_denom

        messages = self.message_projection(
            jnp.concatenate(
                (
                    node_embeddings,
                    edge_embeddings,
                    jnp.broadcast_to(
                        field_step_embeddings[:, None], node_embeddings.shape
                    ),
                    side_onehot,
                    jnp.broadcast_to(src_node[:, None], node_embeddings.shape),
                    jnp.broadcast_to(src_edge[:, None], edge_embeddings.shape),
                    src_weights,
                ),
                axis=-1,
            )
        )

        # ---- batched precompute (2026-09-01) -----------------------------
        # Everything the old per-step _observe_step did that does not read
        # the carry, done once over the whole history: the edge scatter
        # (vmapped segment_sums), the
        # latest-node stream (a last-touched-value recurrence, solved in
        # parallel by a cummax over touched step indices + one gather), and
        # the input-side GRU gate GEMMs (the hoist -- one (H*rows, width)
        # GEMM per gate instead of hundreds of 12-row kernels in the serial
        # chain). Measured on the standalone bench: -33% to -38% fwd+bwd on
        # the scan at H 256-512; scan share of the actor forward was 34-56%
        # at the common buckets.
        valid_f = step_valid.astype(messages.dtype)
        seg = jnp.where(edge_mask & step_valid[:, None], slot_ids, NUM_PUBLIC_SLOTS)

        def scatter_step(step_messages, step_nodes, step_seg):
            slot_sum = jax.ops.segment_sum(
                step_messages, step_seg, num_segments=NUM_PUBLIC_SLOTS + 1
            )[:-1]
            counts = jax.ops.segment_sum(
                jnp.ones(step_seg.shape, jnp.int32),
                step_seg,
                num_segments=NUM_PUBLIC_SLOTS + 1,
            )[:-1]
            node_means = jax.ops.segment_sum(
                step_nodes, step_seg, num_segments=NUM_PUBLIC_SLOTS + 1
            )[:-1] / counts.clip(min=1)[..., None].astype(step_nodes.dtype)
            return slot_sum, counts, node_means

        slot_messages, counts, node_means = jax.vmap(scatter_step)(
            messages, node_embeddings, seg
        )
        touched = counts > 0  # (H, 12)

        # Latest-node stream: node_means at the last touched step <= t, or 0
        # before any touch -- exactly the old scan's latest_nodes carry,
        # without the carry.
        step_index = jnp.arange(touched.shape[0])[:, None]
        last_touched = jax.lax.cummax(
            jnp.where(touched, step_index, -1), axis=0
        )  # (H, 12)
        gathered = jnp.take_along_axis(
            node_means, last_touched.clip(min=0)[..., None], axis=0
        )
        node_snapshots = jnp.where((last_touched >= 0)[..., None], gathered, 0).astype(
            self.cfg.dtype
        )

        # (all, mine, theirs) message sums for the three field states.
        live = (edge_mask & step_valid[:, None]).astype(messages.dtype)[..., None]
        mine = live * edge_is_mine.astype(messages.dtype)[..., None]
        side_messages = jnp.stack(
            (
                (messages * live).sum(axis=-2),
                (messages * mine).sum(axis=-2),
                (messages * (live - mine)).sum(axis=-2),
            ),
            axis=-2,
        )  # (H, 3, D)

        # The hoisted input-side gate GEMMs. Slot input layout is
        # [messages ; field_vec ; flat_field]; the first 2D columns are
        # precomputable (projected here, bias included), the carry tail is
        # projected inside the step. The field input is FULLY precomputable.
        def per_slot_stream(x):
            return jnp.broadcast_to(x[:, None, :], slot_messages.shape)

        slot_pre_inputs = jnp.concatenate(
            (slot_messages, per_slot_stream(field_step_embeddings)), axis=-1
        )  # (H, 12, 2D)
        slot_pre_gates = self.slot_cell.project_inputs(slot_pre_inputs)
        field_inputs = jnp.concatenate(
            (field_row_embeddings.astype(self.cfg.dtype), side_messages), axis=-1
        )  # (H, 3, 2D)
        field_gates = self.field_cell.project_inputs(field_inputs)

        observe = nn.scan(
            type(self)._scan_step,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
            unroll=SCAN_UNROLL,
        )
        h0_slots, h0_field = self.initial_state()
        _, (slot_snapshots, field_snapshots) = observe(
            self,
            (h0_slots, h0_field),
            (
                slot_pre_gates,
                field_gates,
                touched.astype(h0_slots.dtype),
                valid_f,
            ),
        )

        return PerSlotHistoryOutput(
            slot_snapshots=slot_snapshots,
            field_snapshots=field_snapshots,
            node_snapshots=node_snapshots,
            step_valid=step_valid,
            step_request_count=step_request_count,
        )

    def state_at_requests(
        self, history_output: PerSlotHistoryOutput, request_counts: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """For each request, gather the state after the last history step whose
        request_count <= the request's.
        (T,) -> ((T, 12, D) slot states, (T, D) field state,
        (T, 12, D) latest node snapshots)."""
        h0_slots, h0_field = self.initial_state()
        step_indices = jnp.arange(history_output.step_valid.shape[0])

        def gather_one(request_count: jax.Array):
            ok = history_output.step_valid & (
                history_output.step_request_count <= request_count
            )
            idx = jnp.where(ok, step_indices, -1).max()
            has_history = idx >= 0
            safe_idx = jnp.maximum(idx, 0)
            slots = jnp.where(
                has_history, history_output.slot_snapshots[safe_idx], h0_slots
            )
            field = jnp.where(
                has_history, history_output.field_snapshots[safe_idx], h0_field
            )
            nodes = jnp.where(
                has_history,
                history_output.node_snapshots[safe_idx],
                jnp.zeros_like(h0_slots),
            )
            return slots, field, nodes

        return jax.vmap(gather_one)(request_counts)
