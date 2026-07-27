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

from rl.environment.protos.features_pb2 import FieldFeature
from rl.model.modules import MultiHeadAttention, create_attention_mask, layer_norm

NUM_PUBLIC_SLOTS = 12

# The scan is latency-bound (hundreds of tiny sequential kernels), not
# FLOP-bound; unrolling fuses steps per launch.
SCAN_UNROLL = 8

_RELEVANT_ENTITY_FEATURES = np.array(
    [
        FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX0,
        FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX1,
        FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX2,
        FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX3,
    ]
)


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

    A set of num_latents learned queries attends over the 13 history tokens
    (12 slot states + field state), yielding (num_latents, D) latents.
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
        """(12, D) snapshots, (12, D) slot states, (D,) field -> (12, D)."""
        pcfg = self.cfg.history_pool
        kv = jnp.concatenate((slot_states, field_state[None]), axis=0)
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
            "initial_field_state", init, (entity_size,)
        )
        # Projects [node snapshot ; edge ; field] into a slot message.
        self.message_projection = nn.Dense(
            features=entity_size,
            use_bias=False,
            dtype=self.cfg.dtype,
            name="message_projection",
        )
        self.slot_cell = nn.GRUCell(entity_size, dtype=self.cfg.dtype, name="slot_cell")
        self.field_cell = nn.GRUCell(
            entity_size, dtype=self.cfg.dtype, name="field_cell"
        )

    def initial_state(self) -> tuple[jax.Array, jax.Array]:
        h_slots = jnp.repeat(self.initial_slot_state, NUM_PUBLIC_SLOTS, axis=0).astype(
            self.cfg.dtype
        )
        h_field = self.initial_field_state.astype(self.cfg.dtype)
        return h_slots, h_field

    def _advance(
        self,
        h_slots: jax.Array,
        h_field: jax.Array,
        slot_messages: jax.Array,
        touched: jax.Array,
        field_vec: jax.Array,
        valid: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Advance the slot bank one step.

        Args:
            h_slots: (..., 12, D), h_field: (..., D).
            slot_messages: (..., 12, D) per-slot message (zero if untouched).
            touched: (..., 12) float gates in [0, 1].
            field_vec: (..., D) field embedding for this step.
            valid: (...,) float step-validity gate.
        """
        # Battle gestalt so a slot update can condition on the other slots,
        # plus the persistent field state for global memory.
        ctx = h_slots.mean(axis=-2)

        def per_slot(x: jax.Array) -> jax.Array:
            return jnp.broadcast_to(x[..., None, :], h_slots.shape)

        slot_inputs = jnp.concatenate(
            (slot_messages, per_slot(field_vec), per_slot(ctx), per_slot(h_field)),
            axis=-1,
        )
        # GRUCell applies its Dense layers to the last axis, so the 12 slots
        # (and any leading batch axes) share one set of cell weights.
        new_slots, _ = self.slot_cell(h_slots, slot_inputs)
        slot_gate = (touched * valid[..., None])[..., None]
        h_slots = slot_gate * new_slots + (1 - slot_gate) * h_slots

        new_field, _ = self.field_cell(
            h_field,
            jnp.concatenate((field_vec, slot_messages.sum(axis=-2)), axis=-1),
        )
        field_gate = valid[..., None]
        h_field = field_gate * new_field + (1 - field_gate) * h_field
        return h_slots, h_field

    def _observe_step(self, carry, xs):
        """One real history step: scatter edges into the slot bank."""
        h_slots, h_field, latest_nodes = carry
        field_vec, messages, node_embs, slot_ids, edge_mask, valid = xs

        # Padded / invalid edges scatter into a 13th bin that is dropped.
        seg = jnp.where(edge_mask & valid, slot_ids, NUM_PUBLIC_SLOTS)
        slot_messages = jax.ops.segment_sum(
            messages, seg, num_segments=NUM_PUBLIC_SLOTS + 1
        )[:-1]
        counts = jax.ops.segment_sum(
            (edge_mask & valid).astype(jnp.int32),
            seg,
            num_segments=NUM_PUBLIC_SLOTS + 1,
        )[:-1]
        touched = counts > 0

        # Keep each touched slot's LATEST node snapshot verbatim. The GRU
        # state integrates history; this preserves the entity's current
        # state (hp/fainted/status) unmixed — empirically the GRU-only
        # readout loses it (a raw hand rule over snapshots beat the model
        # on late-game states).
        node_means = jax.ops.segment_sum(
            node_embs, seg, num_segments=NUM_PUBLIC_SLOTS + 1
        )[:-1] / counts.clip(min=1)[..., None].astype(node_embs.dtype)
        latest_nodes = jnp.where(
            touched[..., None], node_means.astype(latest_nodes.dtype), latest_nodes
        )

        h_slots, h_field = self._advance(
            h_slots,
            h_field,
            slot_messages,
            touched.astype(h_slots.dtype),
            field_vec,
            valid.astype(h_slots.dtype),
        )
        carry = (h_slots, h_field, latest_nodes)
        return carry, carry

    def __call__(
        self,
        history_field: jax.Array,
        node_embedding_cache: jax.Array,
        edge_embedding_cache: jax.Array,
        edge_slot_ids: jax.Array,
        node_sides: jax.Array,
        field_step_embeddings: jax.Array,
        step_request_count: jax.Array,
        step_valid: jax.Array,
    ) -> PerSlotHistoryOutput:
        """Scan the slot bank along the history axis.

        Args:
            history_field: (H, NUM_FIELD_FEATURES) raw int history rows.
            node_embedding_cache: (P, D) embedded public-entity cache rows.
            edge_embedding_cache: (P, D) embedded edge cache rows.
            edge_slot_ids: (P,) ENTITY_EDGE_FEATURE__ENTITY_IDX per cache row.
            node_sides: (P,) relative side (1 = mine) per cache row.
            field_step_embeddings: (H, D) pooled field embedding per step.
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
        side_onehot = jax.nn.one_hot(
            jnp.take(node_sides, relevant, axis=0), 2, dtype=node_embeddings.dtype
        )  # (H, K, 2)

        messages = self.message_projection(
            jnp.concatenate(
                (
                    node_embeddings,
                    edge_embeddings,
                    jnp.broadcast_to(
                        field_step_embeddings[:, None], node_embeddings.shape
                    ),
                    side_onehot,
                ),
                axis=-1,
            )
        )

        observe = nn.scan(
            type(self)._observe_step,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
            unroll=SCAN_UNROLL,
        )
        h0_slots, h0_field = self.initial_state()
        latest0 = jnp.zeros_like(h0_slots)
        _, (slot_snapshots, field_snapshots, node_snapshots) = observe(
            self,
            (h0_slots, h0_field, latest0),
            (
                field_step_embeddings,
                messages,
                node_embeddings,
                slot_ids,
                edge_mask,
                step_valid,
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
            ok = history_output.step_valid & (history_output.step_request_count <= request_count)
            idx = jnp.where(ok, step_indices, -1).max()
            has_history = idx >= 0
            safe_idx = jnp.maximum(idx, 0)
            slots = jnp.where(has_history, history_output.slot_snapshots[safe_idx], h0_slots)
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