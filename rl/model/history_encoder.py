"""Two input-gated associative scans over the entity, field and register
memories with the step attention between them."""

from collections.abc import Callable

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike
from ml_collections import ConfigDict

from rl.environment.interfaces import HistoryCarry
from rl.environment.protos.enums_pb2 import BattlemajorargsEnum
from rl.environment.protos.features_pb2 import EntityEdgeFeature, FieldFeature
from rl.model.constants import (
    HISTORY_ACTIVE_STATE_ROWS,
    HISTORY_EVENT_STATE_ROWS,
    HISTORY_FIELD_STATE_ROWS,
    HISTORY_REGISTER_STATE_ROWS,
    HISTORY_SLOT_STATE_ROWS,
    HISTORY_STATE_GROUP_IDS,
    NUM_ACTIVE_SLOTS,
    NUM_FIELD_ROWS,
    NUM_HISTORY_REGISTERS,
    NUM_HISTORY_STATE_GROUPS,
    NUM_HISTORY_STATE_ROWS,
    NUM_PUBLIC_SLOTS,
    RELEVANT_ENTITY_FEATURES,
)


def relevant_edges(history_field: jax.Array) -> tuple[jax.Array, jax.Array]:
    """A step's edges are the cache rows named by its RELEVANT_ENTITY_IDX
    columns, capped by NUM_RELEVANT: (H, K) row indices and the (H, K)
    bool mask of the live ones. Written once -- the encoder's gather and
    the wire-side telemetry must agree on it."""
    relevant = history_field[:, RELEVANT_ENTITY_FEATURES]  # (H, K)
    num_relevant = history_field[:, FieldFeature.FIELD_FEATURE__NUM_RELEVANT]
    edge_mask = jnp.arange(relevant.shape[1])[None] < num_relevant[:, None]
    return relevant, edge_mask


def source_rows(edge_major_args: jax.Array, edge_mask: jax.Array) -> jax.Array:
    """(H, K) bool: the step's SOURCE rows -- the mover of a move, switch,
    faint or cant -- i.e. the live rows whose MAJOR_ARG is a real protocol
    arg (anything past the UNSPECIFIED/NULL/PAD sentinels). A
    self-targeting move's row is both source and affected."""
    is_real = edge_major_args > BattlemajorargsEnum.BATTLEMAJORARGS_ENUM___PAD
    return is_real & edge_mask


def major_arg_step_mask(history_field: jax.Array, edge_cache: jax.Array) -> jax.Array:
    """(H,) bool: history steps that carry at least one battle major arg.
    These are the integrated history critic's supervision points, matching
    the offline critic's convention of scoring at decision-bearing events
    rather than every residual/chip line.
    """
    relevant, edge_mask = relevant_edges(history_field)
    major = jnp.take(
        edge_cache[:, EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG],
        relevant,
        axis=0,
    )  # (H, K)
    return source_rows(major, edge_mask).any(axis=-1)


@chex.dataclass
class PerSlotHistoryOutput:
    slot_snapshots: ArrayLike = ()
    field_snapshots: ArrayLike = ()
    active_snapshots: ArrayLike = ()
    register_snapshots: ArrayLike = ()
    # Latest raw node embedding per slot as of each step (H, 12, D): the
    # entity's current snapshot, unmixed by the recurrence — what a hand
    # evaluator reads. Parameter-free carry.
    node_snapshots: ArrayLike = ()
    # The packed cache row that snapshot came from, (H, 12), -1 never
    # touched: the per-event public state reads the slot's raw features
    # (side, position, fainted) off that row.
    node_row_index: ArrayLike = ()
    # Carry stores f32 recurrence outputs, before the snapshot casts.
    final_slot_state: ArrayLike = ()
    final_field_state: ArrayLike = ()
    final_active_state: ArrayLike = ()
    final_register_state: ArrayLike = ()
    # Layer-1 memory, carry-only: the trunk reads layer 2.
    final_inner_state: ArrayLike = ()
    step_valid: ArrayLike = ()
    step_request_count: ArrayLike = ()
    # The step GAT's read, for telemetry: (H, heads, K, K) attention
    # probabilities (zero on padded keys) and the (H, K) source-row mask
    # they are read against.
    step_attention_probs: ArrayLike = ()
    step_source_rows: ArrayLike = ()
    step_slot_gate: ArrayLike = ()


def invalid_history_carry(width: int) -> HistoryCarry:
    """The actor's full-window request: leaves PRESENT, so a batch of
    requests stacks whether or not each one resumes, and `valid` False, so
    the encoder starts from its learned h0 -- the from-scratch forward."""
    return HistoryCarry(
        slot_states=np.zeros((NUM_PUBLIC_SLOTS, width), np.float32),
        field_states=np.zeros((NUM_FIELD_ROWS, width), np.float32),
        active_states=np.zeros((NUM_ACTIVE_SLOTS, width), np.float32),
        register_states=np.zeros((NUM_HISTORY_REGISTERS, width), np.float32),
        inner_states=np.zeros((HISTORY_EVENT_STATE_ROWS.stop, width), np.float32),
        node_snapshots=np.zeros((NUM_PUBLIC_SLOTS, width), np.float32),
        valid=np.zeros((), np.bool_),
    )


def invalid_history_carry_like(carry: HistoryCarry) -> HistoryCarry:
    """Zeros in `carry`'s shapes with `valid` False: the fill for a request
    that does not resume, batched beside ones that do."""
    return jax.tree.map(np.zeros_like, carry)


def history_carry_from(output: PerSlotHistoryOutput) -> HistoryCarry:
    """The state after the window: what the next request's suffix resumes
    from. Post-window regardless of the request-aligned gather -- edges are
    stamped with the request count they were ingested under, so at request
    N every window step has count <= N and the gather selects the last
    valid step anyway."""
    return HistoryCarry(
        slot_states=output.final_slot_state,
        field_states=output.final_field_state,
        active_states=output.final_active_state,
        register_states=output.final_register_state,
        inner_states=output.final_inner_state,
        node_snapshots=output.node_snapshots[-1],
        valid=jnp.ones((), dtype=jnp.bool_),
    )


def _masked_mean(values: jax.Array, weight: jax.Array) -> jax.Array:
    """Mean of values under a broadcastable bool weight; 0.0 when empty."""
    weight = jnp.broadcast_to(weight, values.shape).astype(jnp.float32)
    return (values.astype(jnp.float32) * weight).sum() / weight.sum().clip(min=1.0)


def history_step_stats(
    output: PerSlotHistoryOutput, key_mask: np.ndarray
) -> dict[str, jax.Array]:
    """Per-trajectory scalars for the History panels, from the step GAT's
    probabilities and the backbone's write gate.

    step_attn_entropy: attention entropy per live query row, normalised by
    log(live rows), over steps with >= 2 live rows -- 1.0 = the GAT reads
    every row uniformly, i.e. it has not learned to select.
    step_attn_to_src: the mass a NON-source row places on the step's source
    rows, over steps carrying both; beside step_attn_to_src_uniform (the
    source rows' share of live rows -- what uniform attention would
    place). Above uniform = "who did this to me" is being read.
    gate_mean: the slot write gate over all valid (step, slot) pairs;
    pinned at 0 (nothing written) or 1 (memory overwritten every step) is
    the collapse shape.
    """
    probs = output.step_attention_probs.astype(jnp.float32)  # (H, heads, K, K)
    # (H, K): the rows a valid step's attention could read.
    row_mask = jnp.asarray(key_mask)[None] & output.step_valid[:, None]
    num_live = row_mask.sum(-1)  # (H,)
    # Padded keys carry exactly 0 mass, so the clip only guards the log.
    entropy = -(probs * jnp.log(probs.clip(min=1e-9))).sum(-1)  # (H, heads, K)
    normalised_entropy = entropy / jnp.log(num_live.clip(min=2))[:, None, None]
    entropy_weight = (row_mask & (num_live >= 2)[:, None])[:, None, :]
    source = output.step_source_rows & row_mask  # (H, K)
    to_src = (probs * source[:, None, None, :]).sum(-1)  # (H, heads, K)
    src_weight = (row_mask & ~source & source.any(-1)[:, None])[:, None, :]
    src_share = source.sum(-1) / num_live.clip(min=1)  # (H,)
    gate_weight = jnp.broadcast_to(
        output.step_valid[:, None], output.step_slot_gate.shape
    )
    return {
        "step_attn_entropy": _masked_mean(normalised_entropy, entropy_weight),
        "step_attn_to_src": _masked_mean(to_src, src_weight),
        "step_attn_to_src_uniform": _masked_mean(
            jnp.broadcast_to(src_share[:, None, None], to_src.shape), src_weight
        ),
        "gate_mean": _masked_mean(output.step_slot_gate, gate_weight),
    }


class StepAttention(nn.Module):
    """Masked self-attention over the rows of each history step."""

    num_heads: int
    qk_size: int
    features: int
    dtype: jnp.dtype
    output_init: Callable = nn.initializers.zeros_init()

    @nn.compact
    def __call__(
        self, rows: jax.Array, row_mask: jax.Array, *, value_rows: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Address rows include identities; value rows contain only content."""
        num_steps, num_rows = row_mask.shape

        def project(inputs, name, width, init):
            return nn.Dense(
                features=self.num_heads * width,
                use_bias=False,
                dtype=self.dtype,
                kernel_init=init,
                name=name,
            )(inputs).reshape(num_steps, num_rows, self.num_heads, width)

        lecun = nn.initializers.lecun_normal()
        query = project(rows, "query", self.qk_size, lecun)
        key = project(rows, "key", self.qk_size, lecun)
        value = project(value_rows, "value", self.features // self.num_heads, lecun)
        logits = jnp.einsum("hiad,hjad->haij", query, key) / jnp.sqrt(
            jnp.asarray(self.qk_size, self.dtype)
        )
        key_mask = row_mask[:, None, None, :]  # (H, 1, 1, K)
        logits = jnp.where(key_mask, logits, -1e9)
        probs = jax.nn.softmax(logits, axis=-1)
        probs = jnp.where(key_mask, probs, 0)
        attended = jnp.einsum("haij,hjad->hiad", probs, value).reshape(
            num_steps, num_rows, -1
        )
        out = nn.Dense(
            features=self.features,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=self.output_init,
            name="out_proj",
        )(attended)
        return out, probs


# Which of the 19 rows a step's attention may READ: the register rows carry
# no state at the read (their content is zero, their row is the learned
# identity alone), so they are queries only -- the 2026-09-13 assessment
# declined register keys as a static sink.
STEP_KEY_MASK = np.ones(NUM_HISTORY_STATE_ROWS, dtype=bool)
STEP_KEY_MASK[HISTORY_REGISTER_STATE_ROWS] = False


class GatedLinearCell(nn.Module):
    """minGRU (Feng et al. 2024): h_t = (1 - z_t) h_{t-1} + z_t c_t with
    z_t = sigmoid(W_z x_t + b_z) and c_t = W_c x_t + b_c. Gate and candidate
    read the INPUT only, never h_{t-1}, so each step is a per-channel affine
    map of the carry and the sequence is an associative scan of depth
    O(log H) (`gated_linear_scan`). Selectivity is kept: z_t is input-
    dependent, so a row writes what its own event says to write."""

    features: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, xs: jax.Array) -> tuple[jax.Array, jax.Array]:
        gate = nn.sigmoid(
            nn.Dense(features=self.features, dtype=self.dtype, name="gate")(xs)
        )
        candidate = nn.Dense(
            features=self.features, dtype=self.dtype, name="candidate"
        )(xs)
        return gate, candidate


def gated_linear_scan(
    gate: jax.Array, candidate: jax.Array, write: jax.Array, initial: jax.Array
) -> jax.Array:
    """h_t = A_t * h_0 + B_t over (H, N, D) gate/candidate with an (H, N)
    write mask -- 0 where step t leaves unit n (an untouched slot, an
    invalid step) folds in as the identity (a, b) = (1, 0), so a never-
    written unit holds `initial` EXACTLY. Two steps compose as
    ((a1, b1), (a2, b2)) -> (a1 a2, a2 b1 + b2), which is associative. Runs
    in f32: the coefficient products compound bf16 reassociation across
    log2(H) levels (the precision ledger's value-recursion rule)."""
    write = write.astype(jnp.float32)[..., None]
    decay = 1.0 - write * gate.astype(jnp.float32)
    drive = write * gate.astype(jnp.float32) * candidate.astype(jnp.float32)

    def compose(earlier, later):
        decay_earlier, drive_earlier = earlier
        decay_later, drive_later = later
        return decay_earlier * decay_later, decay_later * drive_earlier + drive_later

    cum_decay, cum_drive = jax.lax.associative_scan(compose, (decay, drive), axis=0)
    return cum_decay * initial.astype(jnp.float32)[None] + cum_drive


class HistorySequenceStep(nn.Module):
    """The recurrence over the history rows (2026-09-18): 12 entity slots,
    the field triple, the 4 active slots and the registers. Layer 1 is an
    input-gated scan over the event rows (all but the registers); the step
    attention then reads layer 1's states (which already integrate every
    event up to this step) for all steps at once; layer 2 is an input-gated
    scan over event rows plus what attention returned, over every row. No attention reads the
    memory of its own layer, so both scans are associative and there is no
    loop to contract (the 2026-09-13 memory-in-the-loop GRU needed a retain
    bias of 4.0 against its own chaos; LESSONS 09-18 has the ablation). The
    register rows have no event input, so their attention row is the
    learned register identity -- a fixed query whose read layer 2
    accumulates: a latent history-summary token.
    """

    cfg: ConfigDict

    def setup(self):
        width = self.cfg.entity_size
        self.input_norm = nn.RMSNorm(dtype=self.cfg.dtype, name="input_norm")
        self.group_identity = self.param(
            "group_identity",
            nn.initializers.normal(0.02),
            (NUM_HISTORY_STATE_GROUPS, width),
        )
        self.register_identity = self.param(
            "register_identity",
            nn.initializers.normal(0.02),
            (NUM_HISTORY_REGISTERS, width),
        )
        self.attention = StepAttention(
            num_heads=self.cfg.history_step.num_heads,
            qk_size=self.cfg.history_step.qk_size,
            features=width,
            dtype=self.cfg.dtype,
            output_init=nn.initializers.lecun_normal(),
            name="attention",
        )
        self.inner_cells = {
            token_type: GatedLinearCell(
                width, dtype=self.cfg.dtype, name=f"{token_type}_inner_cell"
            )
            for token_type in ("entity", "field", "active")
        }
        self.cells = {
            token_type: GatedLinearCell(
                width, dtype=self.cfg.dtype, name=f"{token_type}_cell"
            )
            for token_type in ("entity", "field", "active", "register")
        }

    def read(
        self, content: jax.Array, identities: jax.Array, key_mask: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """The step attention over (S, rows, D) content rows: identities and
        the group / register identities address it, normalised content is
        what it reads (values carry no identity)."""
        dtype = self.cfg.dtype
        normalised = self.input_norm(content.astype(dtype))
        rows = (
            normalised
            + identities
            + self.group_identity.astype(dtype)[jnp.asarray(HISTORY_STATE_GROUP_IDS)]
        )
        rows = rows.at[:, HISTORY_REGISTER_STATE_ROWS].add(
            self.register_identity.astype(dtype)
        )
        return self.attention(
            rows, jnp.broadcast_to(key_mask, rows.shape[:2]), value_rows=normalised
        )

    def _per_group(self, cells, rows: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Apply each token type's cell to its rows; (S, N, D) -> gate,
        candidate (S, N, D) in row order."""
        gates, candidates = [], []
        for token_type, state_rows in (
            ("entity", HISTORY_SLOT_STATE_ROWS),
            ("field", HISTORY_FIELD_STATE_ROWS),
            ("active", HISTORY_ACTIVE_STATE_ROWS),
            ("register", HISTORY_REGISTER_STATE_ROWS),
        ):
            if token_type not in cells:
                continue
            gate, candidate = cells[token_type](rows[:, state_rows])
            gates.append(gate)
            candidates.append(candidate)
        return jnp.concatenate(gates, axis=1), jnp.concatenate(candidates, axis=1)

    def __call__(
        self,
        event_rows: jax.Array,
        attention_identities: jax.Array,
        touched: jax.Array,
        active_touched: jax.Array,
        step_valid: jax.Array,
        initial_inner: jax.Array,
        initial_memory: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """All H steps at once: (H, rows, D) memory states, (H, event rows, D)
        layer-1 states, (H, heads, rows, rows) probabilities, (H, 12) slot
        write gates."""
        num_steps = event_rows.shape[0]
        event_inputs = event_rows[:, HISTORY_EVENT_STATE_ROWS]
        gate_inner, candidate_inner = self._per_group(self.inner_cells, event_inputs)
        write_inner = (
            jnp.concatenate(
                (
                    touched,
                    jnp.ones((num_steps, NUM_FIELD_ROWS), jnp.bool_),
                    active_touched,
                ),
                axis=1,
            )
            & step_valid[:, None]
        )
        inner = gated_linear_scan(
            gate_inner, candidate_inner, write_inner, initial_inner
        )
        content = jnp.concatenate(
            (
                inner.astype(self.cfg.dtype),
                jnp.zeros(
                    (num_steps, NUM_HISTORY_REGISTERS, self.cfg.entity_size),
                    self.cfg.dtype,
                ),
            ),
            axis=1,
        )
        attended, probabilities = self.read(
            content, attention_identities, jnp.asarray(STEP_KEY_MASK)
        )
        gate, candidate = self._per_group(self.cells, event_rows + attended)
        write = jnp.broadcast_to(step_valid[:, None], gate.shape[:2])
        states = gated_linear_scan(gate, candidate, write, initial_memory)
        return states, inner, probabilities, gate[:, HISTORY_SLOT_STATE_ROWS].mean(-1)


class PerSlotHistoryEncoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.initial_memory = self.param(
            "initial_memory",
            nn.initializers.normal(0.02),
            (NUM_HISTORY_STATE_ROWS, self.cfg.entity_size),
        )
        self.event_projection = nn.Dense(
            self.cfg.entity_size,
            use_bias=False,
            dtype=self.cfg.dtype,
            name="event_projection",
        )
        self.initial_inner_memory = self.param(
            "initial_inner_memory",
            nn.initializers.normal(0.02),
            (HISTORY_EVENT_STATE_ROWS.stop, self.cfg.entity_size),
        )
        self.sequence_step = HistorySequenceStep(self.cfg, name="sequence_step")

    def resolve_initial(
        self, carry: HistoryCarry
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """(rows, D) f32 memory, (event rows, D) f32 layer-1 memory and (12, D)
        node snapshots the window starts from: the learned h0, or the carry
        where it is valid."""
        node0 = jnp.zeros((NUM_PUBLIC_SLOTS, self.cfg.entity_size), self.cfg.dtype)
        inner0 = self.initial_inner_memory
        if isinstance(carry.valid, tuple):
            return self.initial_memory, inner0, node0
        carried = jnp.concatenate(
            (
                carry.slot_states,
                carry.field_states,
                carry.active_states,
                carry.register_states,
            ),
            axis=0,
        ).astype(jnp.float32)
        return (
            jnp.where(carry.valid, carried, self.initial_memory),
            jnp.where(carry.valid, carry.inner_states.astype(jnp.float32), inner0),
            jnp.where(carry.valid, carry.node_snapshots.astype(self.cfg.dtype), node0),
        )

    def _recur(
        self,
        event_rows: jax.Array,
        attention_identities: jax.Array,
        touched: jax.Array,
        active_touched: jax.Array,
        step_valid: jax.Array,
        initial_memory: jax.Array,
        initial_inner: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """(H, rows, D) states in the compute dtype, (H, heads, rows, rows)
        probabilities, (H, 12) slot write gates, the final (rows, D) f32
        memory and the final (event rows, D) f32 layer-1 memory."""
        states, inner, probabilities, gates = self.sequence_step(
            event_rows,
            attention_identities,
            touched,
            active_touched,
            step_valid,
            initial_inner.astype(jnp.float32),
            initial_memory.astype(jnp.float32),
        )
        # Invalid tail steps are the identity, so the last row is the state
        # after the last valid step.
        return (
            states.astype(self.cfg.dtype),
            probabilities,
            gates,
            states[-1],
            inner[-1],
        )

    def __call__(
        self,
        history_field: jax.Array,
        node_embedding_cache: jax.Array,
        node_identity_cache: jax.Array,
        active_state_cache: jax.Array,
        active_slot_ids: jax.Array,
        active_identities: jax.Array,
        edge_embedding_cache: jax.Array,
        edge_slot_ids: jax.Array,
        edge_major_args: jax.Array,
        field_row_embeddings: jax.Array,
        field_identities: jax.Array,
        step_request_count: jax.Array,
        step_valid: jax.Array,
        carry: HistoryCarry = HistoryCarry(),
    ) -> PerSlotHistoryOutput:
        initial_memory, initial_inner, node0 = self.resolve_initial(carry)
        relevant, edge_mask = relevant_edges(history_field)
        node_embeddings = jnp.take(node_embedding_cache, relevant, axis=0)
        edge_embeddings = jnp.take(edge_embedding_cache, relevant, axis=0)
        slot_ids = jnp.take(edge_slot_ids, relevant, axis=0).clip(
            0, NUM_PUBLIC_SLOTS - 1
        )
        is_source = source_rows(jnp.take(edge_major_args, relevant, axis=0), edge_mask)
        row_inputs = jnp.concatenate(
            (
                node_embeddings,
                edge_embeddings,
                is_source.astype(self.cfg.dtype)[..., None],
            ),
            axis=-1,
        )
        messages = self.event_projection(row_inputs)
        segments = jnp.where(
            edge_mask & step_valid[:, None], slot_ids, NUM_PUBLIC_SLOTS
        )

        def scatter_step(
            step_messages, step_nodes, step_source, step_segments, step_rows
        ):
            counts = jax.ops.segment_sum(
                jnp.ones(step_segments.shape, jnp.int32),
                step_segments,
                num_segments=NUM_PUBLIC_SLOTS + 1,
            )[:-1]
            latest_row = jax.ops.segment_max(
                step_rows, step_segments, num_segments=NUM_PUBLIC_SLOTS + 1
            )[:-1]
            latest_row = jnp.where(counts > 0, latest_row, -1)
            summed = jax.ops.segment_sum(
                step_messages, step_segments, num_segments=NUM_PUBLIC_SLOTS + 1
            )[:-1]
            node_means = jax.ops.segment_sum(
                step_nodes, step_segments, num_segments=NUM_PUBLIC_SLOTS + 1
            )[:-1] / counts.clip(min=1)[..., None].astype(step_nodes.dtype)
            sources = (
                jax.ops.segment_sum(
                    step_source.astype(jnp.int32),
                    step_segments,
                    num_segments=NUM_PUBLIC_SLOTS + 1,
                )[:-1]
                > 0
            )
            return summed, counts, node_means, sources, latest_row

        # One scatter carries the node content and its identity side by side.
        slot_messages, counts, node_means, slot_sources, step_rows = jax.vmap(
            scatter_step
        )(
            messages,
            jnp.concatenate(
                (
                    jnp.take(node_embedding_cache, relevant, axis=0),
                    jnp.take(node_identity_cache, relevant, axis=0),
                ),
                axis=-1,
            ),
            is_source,
            segments,
            relevant.astype(jnp.int32),
        )
        node_means, slot_identities = jnp.split(node_means, 2, axis=-1)
        touched = counts > 0
        # The active slots' inputs: the state token of whichever entity an
        # event touched while it held the slot, averaged over the step's
        # events. Benched entities fall in the dump segment.
        active_ids = jnp.take(active_slot_ids, relevant, axis=0)
        active_segments = jnp.where(
            edge_mask & step_valid[:, None], active_ids, NUM_ACTIVE_SLOTS
        )

        def scatter_active(step_tokens, step_segments):
            counts = jax.ops.segment_sum(
                jnp.ones(step_segments.shape, jnp.int32),
                step_segments,
                num_segments=NUM_ACTIVE_SLOTS + 1,
            )[:-1]
            summed = jax.ops.segment_sum(
                step_tokens, step_segments, num_segments=NUM_ACTIVE_SLOTS + 1
            )[:-1]
            return summed / counts.clip(min=1)[..., None].astype(summed.dtype), counts

        active_inputs, active_counts = jax.vmap(scatter_active)(
            jnp.take(active_state_cache, relevant, axis=0), active_segments
        )
        active_touched = active_counts > 0
        # Packed rows are appended in step order, so the running maximum is
        # each slot's latest row as of every step.
        node_row_index = jax.lax.cummax(step_rows, axis=0)
        # Snapshot support remains solely for the standalone offline critic.
        step_index = jnp.arange(touched.shape[0])[:, None]
        last_touched = jax.lax.cummax(jnp.where(touched, step_index, -1), axis=0)
        gathered = jnp.take_along_axis(
            node_means, last_touched.clip(min=0)[..., None], axis=0
        )
        node_snapshots = jnp.where(
            (last_touched >= 0)[..., None], gathered, node0[None]
        ).astype(self.cfg.dtype)
        event_rows = jnp.concatenate(
            (
                slot_messages,
                field_row_embeddings.astype(self.cfg.dtype),
                active_inputs.astype(self.cfg.dtype),
                jnp.zeros(
                    (step_valid.shape[0], NUM_HISTORY_REGISTERS, self.cfg.entity_size),
                    self.cfg.dtype,
                ),
            ),
            axis=1,
        )
        attention_identities = jnp.zeros_like(event_rows)
        attention_identities = attention_identities.at[:, HISTORY_SLOT_STATE_ROWS].set(
            slot_identities
        )
        attention_identities = attention_identities.at[:, HISTORY_FIELD_STATE_ROWS].set(
            field_identities
        )
        attention_identities = attention_identities.at[
            :, HISTORY_ACTIVE_STATE_ROWS
        ].set(active_identities.astype(self.cfg.dtype))
        states, probabilities, gates, final_memory, final_inner = self._recur(
            event_rows,
            attention_identities,
            touched,
            active_touched,
            step_valid,
            initial_memory,
            initial_inner,
        )
        source_mask = (
            jnp.zeros((step_valid.shape[0], NUM_HISTORY_STATE_ROWS), jnp.bool_)
            .at[:, HISTORY_SLOT_STATE_ROWS]
            .set(slot_sources)
        )
        return PerSlotHistoryOutput(
            slot_snapshots=states[:, HISTORY_SLOT_STATE_ROWS],
            field_snapshots=states[:, HISTORY_FIELD_STATE_ROWS],
            active_snapshots=states[:, HISTORY_ACTIVE_STATE_ROWS],
            register_snapshots=states[:, HISTORY_REGISTER_STATE_ROWS],
            node_snapshots=node_snapshots,
            node_row_index=node_row_index,
            final_slot_state=final_memory[HISTORY_SLOT_STATE_ROWS],
            final_field_state=final_memory[HISTORY_FIELD_STATE_ROWS],
            final_active_state=final_memory[HISTORY_ACTIVE_STATE_ROWS],
            final_register_state=final_memory[HISTORY_REGISTER_STATE_ROWS],
            final_inner_state=final_inner,
            step_valid=step_valid,
            step_request_count=step_request_count,
            step_attention_probs=probabilities,
            step_source_rows=source_mask,
            step_slot_gate=gates[:, HISTORY_SLOT_STATE_ROWS],
        )

    def state_at_requests(
        self,
        history_output: PerSlotHistoryOutput,
        request_counts: jax.Array,
        carry: HistoryCarry = HistoryCarry(),
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """For each request, gather the state after the last history step whose
        request_count <= the request's; with no such step, what the window
        started from (`resolve_initial` of the same carry the scan ran on --
        a zero-new-steps suffix returns the carry itself).
        (T,) -> ((T, 12, D) slot states, (T, 3, D) field states,
        (T, 12, D) latest node snapshots, (T, 4, D) global registers,
        (T, 4, D) active-slot states)."""
        initial_memory, _, node0 = self.resolve_initial(carry)
        initial_memory = initial_memory.astype(self.cfg.dtype)
        h0_slots = initial_memory[HISTORY_SLOT_STATE_ROWS]
        h0_field = initial_memory[HISTORY_FIELD_STATE_ROWS]
        h0_registers = initial_memory[HISTORY_REGISTER_STATE_ROWS]
        h0_actives = initial_memory[HISTORY_ACTIVE_STATE_ROWS]
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
                has_history, history_output.node_snapshots[safe_idx], node0
            )
            registers = jnp.where(
                has_history, history_output.register_snapshots[safe_idx], h0_registers
            )
            actives = jnp.where(
                has_history, history_output.active_snapshots[safe_idx], h0_actives
            )
            return slots, field, nodes, registers, actives

        return jax.vmap(gather_one)(request_counts)
