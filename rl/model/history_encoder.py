"""One recurrent attention sequence over entity, field and register memories."""

import functools
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
    HISTORY_FIELD_STATE_ROWS,
    HISTORY_REGISTER_STATE_ROWS,
    HISTORY_SLOT_STATE_ROWS,
    HISTORY_STATE_GROUP_IDS,
    NUM_HISTORY_REGISTERS,
    NUM_HISTORY_STATE_ROWS,
    NUM_PUBLIC_SLOTS,
    RELEVANT_ENTITY_FEATURES,
)
from rl.model.modules import MultiHeadAttention, create_attention_mask, layer_norm

# Measured on the carry-vs-full-window divergence: 1.0 (the published LSTM
# forget-bias) still decorrelates, 3.0 is the first value under the bound, and
# this carries the margin. LESSONS 2026-09-13 has the sweep.
HISTORY_RETAIN_BIAS = 4.0

# The field history carries THREE states, mirroring the env-step field triple
# that _embed_field already produces (2026-08-28). Hazards are side-differenced
# — spikes on my side and spikes on theirs are opposite facts — and collapsing
# them into one vector with a Dense meant the recurrent field memory could only
# hold their mixture. Row order matches _embed_field's stack.
NUM_FIELD_ROWS = 3
FIELD_ROW_GLOBAL, FIELD_ROW_MINE, FIELD_ROW_THEIRS = 0, 1, 2
# ENTITY_PUBLIC_NODE_FEATURE__SIDE == 1 is mine (service isMySide).
SIDE_MINE = 1


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
    register_snapshots: ArrayLike = ()
    # Latest raw node embedding per slot as of each step (H, 12, D): the
    # entity's current snapshot, unmixed by GRU gating — what a hand
    # evaluator reads. Parameter-free carry.
    node_snapshots: ArrayLike = ()
    # Carry stores f32 recurrence outputs, before the snapshot casts.
    final_slot_state: ArrayLike = ()
    final_field_state: ArrayLike = ()
    final_register_state: ArrayLike = ()
    step_valid: ArrayLike = ()
    step_request_count: ArrayLike = ()
    # The step GAT's read, for telemetry: (H, heads, K, K) attention
    # probabilities (zero on padded keys), the (H, K) live-row mask and
    # the (H, K) source-row mask they are read against.
    step_attention_probs: ArrayLike = ()
    step_row_mask: ArrayLike = ()
    step_source_rows: ArrayLike = ()
    step_slot_gate: ArrayLike = ()
    step_touched: ArrayLike = ()


def invalid_history_carry(width: int) -> HistoryCarry:
    """The actor's full-window request: leaves PRESENT, so a batch of
    requests stacks whether or not each one resumes, and `valid` False, so
    the encoder starts from its learned h0 -- the from-scratch forward."""
    return HistoryCarry(
        slot_states=np.zeros((NUM_PUBLIC_SLOTS, width), np.float32),
        field_states=np.zeros((NUM_FIELD_ROWS, width), np.float32),
        register_states=np.zeros((NUM_HISTORY_REGISTERS, width), np.float32),
        node_snapshots=np.zeros((NUM_PUBLIC_SLOTS, width), np.float32),
        valid=np.zeros((), np.bool_),
    )


def history_carry_from(output: PerSlotHistoryOutput) -> HistoryCarry:
    """The state after the window: what the next request's suffix resumes
    from. Post-window regardless of the request-aligned gather -- edges are
    stamped with the request count they were ingested under, so at request
    N every window step has count <= N and the gather selects the last
    valid step anyway."""
    return HistoryCarry(
        slot_states=output.final_slot_state,
        field_states=output.final_field_state,
        register_states=output.final_register_state,
        node_snapshots=output.node_snapshots[-1],
        valid=jnp.ones((), dtype=jnp.bool_),
    )


def _masked_mean(values: jax.Array, weight: jax.Array) -> jax.Array:
    """Mean of values under a broadcastable bool weight; 0.0 when empty."""
    weight = jnp.broadcast_to(weight, values.shape).astype(jnp.float32)
    return (values.astype(jnp.float32) * weight).sum() / weight.sum().clip(min=1.0)


def history_step_stats(output: PerSlotHistoryOutput) -> dict[str, jax.Array]:
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
    row_mask = output.step_row_mask & output.step_valid[:, None]  # (H, K)
    num_live = row_mask.sum(-1)  # (H,)
    # Padded keys carry exactly 0 mass, so the clip only guards the log.
    entropy = -(probs * jnp.log(probs.clip(min=1e-9))).sum(-1)  # (H, heads, K)
    normalised_entropy = entropy / jnp.log(num_live.clip(min=2))[:, None, None]
    entropy_weight = (row_mask & (num_live >= 2)[:, None])[:, None, :]
    source = output.step_source_rows & row_mask  # (H, K)
    to_src = (probs * source[:, None, None, :]).sum(-1)  # (H, heads, K)
    src_weight = (row_mask & ~source & source.any(-1)[:, None])[:, None, :]
    src_share = source.sum(-1) / num_live.clip(min=1)  # (H,)
    gate_weight = (
        output.step_row_mask[:, HISTORY_SLOT_STATE_ROWS] & output.step_valid[:, None]
    )  # (H, 12)
    return {
        "step_attn_entropy": _masked_mean(normalised_entropy, entropy_weight),
        "step_attn_to_src": _masked_mean(to_src, src_weight),
        "step_attn_to_src_uniform": _masked_mean(
            jnp.broadcast_to(src_share[:, None, None], to_src.shape), src_weight
        ),
        "gate_mean": _masked_mean(output.step_slot_gate, gate_weight),
    }


class HistoryAttentionPool(nn.Module):
    """Cross-attention pooling of the recurrent history states into a fixed
    bank of learned latent summaries.

    A set of num_latents learned queries attends over the 15 history tokens
    (12 slot states + the 3 field states), yielding (num_latents, D) latents.
    The offline critic reads the flattened latents through its linear probe.
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
    states + field state. The residual gate is zero-init, so at
    initialisation the output IS the raw snapshots (hand-rule parity is the
    floor) and history context blends in only as training finds it useful.
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
            name="attn_out",
        )(attended)
        return out, probs


class HistoryGRUCell(nn.Module):
    """Flax's reset-after GRU equations with f32 activations and memory mixing."""

    features: int
    dtype: jnp.dtype
    # sigmoid(retain_bias) is how much memory survives one event. 0 is Flax's
    # init and leaves the recurrence CHAOTIC once attention feeds memory back:
    # the loop expands, so the actor's carry and the learner's full window
    # separate from a rounding difference into different policies within one
    # game. Contraction is what bounds that, and it has to beat the attention
    # branch's gain -- the published forget-bias 1.0 does not (LESSONS
    # 2026-09-13). Deviates from HistoryGRUCell's Flax-reference default,
    # which is why it is set here and not in the cell.
    retain_bias: float = 0.0

    @nn.compact
    def __call__(
        self, memory: jax.Array, inputs: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        input_dense = functools.partial(
            nn.Dense,
            features=self.features,
            use_bias=True,
            dtype=self.dtype,
            kernel_init=nn.initializers.lecun_normal(),
        )
        recurrent_dense = functools.partial(
            nn.Dense,
            features=self.features,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.orthogonal(),
        )
        reset_gate = nn.sigmoid(
            input_dense(name="ir")(inputs).astype(jnp.float32)
            + recurrent_dense(name="hr")(memory).astype(jnp.float32)
        )
        retain_gate = nn.sigmoid(
            input_dense(
                name="iz", bias_init=nn.initializers.constant(self.retain_bias)
            )(inputs).astype(jnp.float32)
            + recurrent_dense(name="hz")(memory).astype(jnp.float32)
        )
        candidate = nn.tanh(
            input_dense(name="in")(inputs).astype(jnp.float32)
            + reset_gate
            * recurrent_dense(name="hn", use_bias=True)(memory).astype(jnp.float32)
        )
        write_gate = 1 - retain_gate
        updated = write_gate * candidate + retain_gate * memory.astype(jnp.float32)
        return updated, write_gate


class HistorySequenceStep(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(
        self,
        memory: jax.Array,
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        event_rows, attention_identities, valid = inputs
        content = nn.RMSNorm(dtype=self.cfg.dtype, name="input_norm")(
            memory.astype(self.cfg.dtype) + event_rows
        )
        group_identity = self.param(
            "group_identity", nn.initializers.normal(0.02), (3, self.cfg.entity_size)
        )
        register_identity = self.param(
            "register_identity",
            nn.initializers.normal(0.02),
            (NUM_HISTORY_REGISTERS, self.cfg.entity_size),
        )
        rows = (
            content
            + attention_identities
            + group_identity.astype(self.cfg.dtype)[
                jnp.asarray(HISTORY_STATE_GROUP_IDS)
            ]
        )
        rows = rows.at[HISTORY_REGISTER_STATE_ROWS].add(
            register_identity.astype(self.cfg.dtype)
        )
        attended, probabilities = StepAttention(
            num_heads=self.cfg.history_step.num_heads,
            qk_size=self.cfg.history_step.qk_size,
            features=self.cfg.entity_size,
            dtype=self.cfg.dtype,
            output_init=nn.initializers.lecun_normal(),
            name="attention",
        )(
            rows[None],
            jnp.ones((1, NUM_HISTORY_STATE_ROWS), jnp.bool_),
            value_rows=content[None],
        )
        # Normalisation and row identities address attention; the GRU also
        # reads raw event features and its own unnormalised previous memory.
        gru_inputs = event_rows + attended[0]
        updated_groups = []
        gate_groups = []
        for token_type, state_rows in (
            ("entity", HISTORY_SLOT_STATE_ROWS),
            ("field", HISTORY_FIELD_STATE_ROWS),
            ("register", HISTORY_REGISTER_STATE_ROWS),
        ):
            updated_group, group_gate = HistoryGRUCell(
                self.cfg.entity_size,
                dtype=self.cfg.dtype,
                retain_bias=HISTORY_RETAIN_BIAS,
                name=f"{token_type}_gru",
            )(memory[state_rows], gru_inputs[state_rows])
            updated_groups.append(updated_group)
            gate_groups.append(group_gate)
        updated = jnp.concatenate(updated_groups, axis=0)
        write_gate = jnp.concatenate(gate_groups, axis=0)
        memory = jnp.where(valid, updated, memory)
        return memory, (
            memory.astype(self.cfg.dtype),
            probabilities[0],
            write_gate.mean(-1),
        )


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
        # Cross-row reads of previous memory require a chronological scan.
        step = nn.remat(
            HistorySequenceStep, policy=jax.checkpoint_policies.nothing_saveable
        )
        self.sequence_step = nn.scan(
            step,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )(self.cfg, name="sequence_step")

    def resolve_initial(self, carry: HistoryCarry) -> tuple[jax.Array, jax.Array]:
        node0 = jnp.zeros((NUM_PUBLIC_SLOTS, self.cfg.entity_size), self.cfg.dtype)
        if isinstance(carry.valid, tuple):
            return self.initial_memory, node0
        if isinstance(carry.register_states, tuple):
            registers = self.initial_memory[HISTORY_REGISTER_STATE_ROWS]
        else:
            registers = carry.register_states
        carried = jnp.concatenate(
            (carry.slot_states, carry.field_states, registers), axis=0
        ).astype(jnp.float32)
        return (
            jnp.where(carry.valid, carried, self.initial_memory),
            jnp.where(carry.valid, carry.node_snapshots.astype(self.cfg.dtype), node0),
        )

    def _recur(
        self,
        event_rows: jax.Array,
        step_valid: jax.Array,
        initial_memory: jax.Array,
        attention_identities: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        if attention_identities is None:
            attention_identities = jnp.zeros_like(event_rows)
        final_memory, (states, probabilities, gates) = self.sequence_step(
            initial_memory.astype(jnp.float32),
            (event_rows, attention_identities, step_valid),
        )
        return states, probabilities, gates, final_memory

    def __call__(
        self,
        history_field: jax.Array,
        node_embedding_cache: jax.Array,
        node_content_cache: jax.Array,
        edge_embedding_cache: jax.Array,
        edge_slot_ids: jax.Array,
        edge_major_args: jax.Array,
        field_row_embeddings: jax.Array,
        step_request_count: jax.Array,
        step_valid: jax.Array,
        carry: HistoryCarry = HistoryCarry(),
        node_identity_cache: jax.Array | None = None,
        field_identities: jax.Array | None = None,
    ) -> PerSlotHistoryOutput:
        initial_memory, node0 = self.resolve_initial(carry)
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

        def scatter_step(step_messages, step_nodes, step_source, step_segments):
            counts = jax.ops.segment_sum(
                jnp.ones(step_segments.shape, jnp.int32),
                step_segments,
                num_segments=NUM_PUBLIC_SLOTS + 1,
            )[:-1]
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
            return summed, counts, node_means, sources

        slot_messages, counts, node_means, slot_sources = jax.vmap(scatter_step)(
            messages,
            jnp.take(node_content_cache, relevant, axis=0),
            is_source,
            segments,
        )
        touched = counts > 0
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
                jnp.zeros(
                    (step_valid.shape[0], NUM_HISTORY_REGISTERS, self.cfg.entity_size),
                    self.cfg.dtype,
                ),
            ),
            axis=1,
        )
        if node_identity_cache is None:
            slot_identities = jnp.zeros_like(slot_messages)
        else:
            _, _, slot_identities, _ = jax.vmap(scatter_step)(
                messages,
                jnp.take(node_identity_cache, relevant, axis=0),
                is_source,
                segments,
            )
        attention_identities = jnp.zeros_like(event_rows)
        attention_identities = attention_identities.at[:, HISTORY_SLOT_STATE_ROWS].set(
            slot_identities
        )
        if field_identities is not None:
            attention_identities = attention_identities.at[
                :, HISTORY_FIELD_STATE_ROWS
            ].set(field_identities)
        states, probabilities, gates, final_memory = self._recur(
            event_rows, step_valid, initial_memory, attention_identities
        )
        source_mask = (
            jnp.zeros((step_valid.shape[0], NUM_HISTORY_STATE_ROWS), jnp.bool_)
            .at[:, HISTORY_SLOT_STATE_ROWS]
            .set(slot_sources)
        )
        return PerSlotHistoryOutput(
            slot_snapshots=states[:, HISTORY_SLOT_STATE_ROWS],
            field_snapshots=states[:, HISTORY_FIELD_STATE_ROWS],
            register_snapshots=states[:, HISTORY_REGISTER_STATE_ROWS],
            node_snapshots=node_snapshots,
            final_slot_state=final_memory[HISTORY_SLOT_STATE_ROWS],
            final_field_state=final_memory[HISTORY_FIELD_STATE_ROWS],
            final_register_state=final_memory[HISTORY_REGISTER_STATE_ROWS],
            step_valid=step_valid,
            step_request_count=step_request_count,
            step_attention_probs=probabilities,
            step_row_mask=jnp.ones(
                (step_valid.shape[0], NUM_HISTORY_STATE_ROWS), jnp.bool_
            ),
            step_source_rows=source_mask,
            step_slot_gate=gates[:, HISTORY_SLOT_STATE_ROWS],
            step_touched=touched,
        )

    def state_at_requests(
        self,
        history_output: PerSlotHistoryOutput,
        request_counts: jax.Array,
        carry: HistoryCarry = HistoryCarry(),
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """For each request, gather the state after the last history step whose
        request_count <= the request's; with no such step, what the window
        started from (`resolve_initial` of the same carry the scan ran on --
        a zero-new-steps suffix returns the carry itself).
        (T,) -> ((T, 12, D) slot states, (T, 3, D) field states,
        (T, 12, D) latest node snapshots, (T, 4, D) global registers)."""
        initial_memory, node0 = self.resolve_initial(carry)
        initial_memory = initial_memory.astype(self.cfg.dtype)
        h0_slots = initial_memory[HISTORY_SLOT_STATE_ROWS]
        h0_field = initial_memory[HISTORY_FIELD_STATE_ROWS]
        h0_registers = initial_memory[HISTORY_REGISTER_STATE_ROWS]
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
            return slots, field, nodes, registers

        return jax.vmap(gather_one)(request_counts)
