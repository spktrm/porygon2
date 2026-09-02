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

from rl.environment.interfaces import HistoryCarry
from rl.environment.protos.enums_pb2 import BattlemajorargsEnum
from rl.environment.protos.features_pb2 import EntityEdgeFeature, FieldFeature
from rl.model.constants import NUM_PUBLIC_SLOTS
from rl.model.modules import MultiHeadAttention, create_attention_mask, layer_norm

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


def relevant_edges(history_field: jax.Array) -> tuple[jax.Array, jax.Array]:
    """A step's edges are the cache rows named by its RELEVANT_ENTITY_IDX
    columns, capped by NUM_RELEVANT: (H, K) row indices and the (H, K)
    bool mask of the live ones. Written once -- the encoder's gather and
    the wire-side telemetry must agree on it."""
    relevant = history_field[:, _RELEVANT_ENTITY_FEATURES]  # (H, K)
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
    # Per-history-step snapshots: (H, 12, D) / (H, D).
    slot_snapshots: ArrayLike = ()
    field_snapshots: ArrayLike = ()
    # Latest raw node embedding per slot as of each step (H, 12, D): the
    # entity's current snapshot, unmixed by GRU gating — what a hand
    # evaluator reads. Parameter-free carry.
    node_snapshots: ArrayLike = ()
    # The two recursions' states after the window's last step, in f32
    # (before the compute-dtype cast the snapshots go through): (12, D)
    # and (3, D). Padding is trailing and invalid steps compose as the
    # identity, so these ARE the state after the last valid step -- the
    # actor's carry (`history_carry_from`).
    final_slot_state: ArrayLike = ()
    final_field_state: ArrayLike = ()
    step_valid: ArrayLike = ()
    step_request_count: ArrayLike = ()
    # The step GAT's read, for telemetry: (H, heads, K, K) attention
    # probabilities (zero on padded keys), the (H, K) live-row mask and
    # the (H, K) source-row mask they are read against.
    step_attention_probs: ArrayLike = ()
    step_row_mask: ArrayLike = ()
    step_source_rows: ArrayLike = ()
    # The backbone's write gate, mean over units: (H, 12), read against
    # the (H, 12) touched mask -- only a touched slot's gate is applied.
    step_slot_gate: ArrayLike = ()
    step_touched: ArrayLike = ()


def history_carry_from(output: PerSlotHistoryOutput) -> HistoryCarry:
    """The state after the window: what the next request's suffix resumes
    from. Post-window regardless of the request-aligned gather -- edges are
    stamped with the request count they were ingested under, so at request
    N every window step has count <= N and the gather selects the last
    valid step anyway."""
    return HistoryCarry(
        slot_states=output.final_slot_state,
        field_states=output.final_field_state,
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
    gate_mean: the slot write gate over touched, valid (step, slot) pairs;
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
    gate_weight = output.step_touched & output.step_valid[:, None]  # (H, 12)
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


class StepAttention(nn.Module):
    """One GAT-style attention layer over the rows of a single history
    step -- every live row attends to every live row, itself included.

    A step is one major log line and the minor lines until the next: up to
    K = 8 rows, one per mon touched, each carrying its own node snapshot,
    its own edge (what happened to it), its side and whether it was the
    MOVER (a self-targeting move is one row on both counts). The relation
    "X did N to Y" lives across two of those rows, and this is the layer
    where they meet, weighted by content: the source mean it replaces was
    invertible in singles (2-source steps are 2-row steps) and lossy in
    doubles (a spread move: 2 movers, 3 targets, one average for all).

    Not modules.MultiHeadAttention: that carries trunk-sized plumbing (qk
    layer norm, rope, a query-side validity mask) for a 61-row sequence;
    this is 8 rows. Padded rows get a -1e9 floor AND their probabilities
    re-masked, so a 1-row step is exactly its own value. The output
    projection is zeros-init -- one zero factor over live inputs, the
    FlatActionReadout argument -- so the messages are identity at step 0
    and the projection moves at step 1.
    """

    num_heads: int
    qk_size: int
    features: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self, rows: jax.Array, row_mask: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """rows (H, K, W), row_mask (H, K) bool -> (H, K, features) and
        the (H, heads, K, K) probabilities."""
        num_steps, num_rows = row_mask.shape

        def project(name, width, init):
            return nn.Dense(
                features=self.num_heads * width,
                use_bias=False,
                dtype=self.dtype,
                kernel_init=init,
                name=name,
            )(rows).reshape(num_steps, num_rows, self.num_heads, width)

        lecun = nn.initializers.lecun_normal()
        query = project("query", self.qk_size, lecun)
        key = project("key", self.qk_size, lecun)
        value = project("value", self.features // self.num_heads, lecun)
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
            kernel_init=nn.initializers.zeros_init(),
            name="attn_out",
        )(attended)
        return out, probs


class GatedLinearCell(nn.Module):
    """The minGRU recurrence (Feng et al. 2024): h_t = (1 - z_t) * h_{t-1}
    + z_t * c_t with z_t = sigmoid(W_z x_t + b_z) and c_t = W_c x_t + b_c.

    Gate and candidate read the INPUT only -- never h_{t-1} -- so each
    step is a per-channel affine map of the carry and the whole sequence
    is an associative scan (gated_linear_scan) of depth O(log H) instead
    of a serial chain of H dependent kernels. That is what the GRU it
    replaces (2026-09-02) could not offer: its scan sat on a ~26us/step
    dependency-latency floor that hoisting (-8.6%) and unrolling could
    not move. The price is the candidate's blindness to the carry,
    accepted for windows of 256-512 steps; the RG-LRU form (a learned
    per-channel decay in place of 1 - z_t) is the recorded fallback.
    Selectivity is kept: z_t is input-dependent, so a slot writes what
    its own message says to write. Coefficients are emitted in the
    compute dtype and the scan runs them in f32.
    """

    features: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, xs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """xs (..., W) -> (write gate z, candidate c), each (..., features)."""
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
    """States after each step of h_t = a_t * h_{t-1} + b_t, in parallel.

    gate/candidate (H, N, D) from GatedLinearCell; write (H, N) -- 1 where
    step t writes unit n, 0 where it leaves it (an untouched slot, an
    invalid step), which folds in as the identity map (a, b) = (1, 0) so a
    never-written unit holds `initial` EXACTLY; initial (N, D). Two
    steps compose as ((a1, b1), (a2, b2)) -> (a1 * a2, a2 * b1 + b2), which
    is associative, so jax.lax.associative_scan gives every prefix
    (A_t, B_t) and h_t = A_t * h_0 + B_t. Runs in f32: the coefficient
    products compound bf16 reassociation across log2(H) levels (the
    precision ledger's value-recursion rule); returns f32 (H, N, D).
    """
    write = write.astype(jnp.float32)[..., None]
    decay = 1.0 - write * gate.astype(jnp.float32)
    drive = write * gate.astype(jnp.float32) * candidate.astype(jnp.float32)

    def compose(earlier, later):
        decay_earlier, drive_earlier = earlier
        decay_later, drive_later = later
        return decay_earlier * decay_later, decay_later * drive_earlier + drive_later

    cum_decay, cum_drive = jax.lax.associative_scan(compose, (decay, drive), axis=0)
    return cum_decay * initial.astype(jnp.float32)[None] + cum_drive


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
        # Projects [node ; edge ; side ; is_src ; field] into a slot message;
        # the step GAT below adds what the OTHER rows of the step carry.
        self.message_projection = nn.Dense(
            features=entity_size,
            use_bias=False,
            dtype=self.cfg.dtype,
            name="message_projection",
        )
        step_cfg = self.cfg.history_step
        self.step_attention = StepAttention(
            num_heads=step_cfg.num_heads,
            qk_size=step_cfg.qk_size,
            features=entity_size,
            dtype=self.cfg.dtype,
            name="step_attention",
        )
        # Two gated linear cells, two parallel scans, zero serial work
        # (2026-09-02; see _recur). The slot input is [messages ;
        # field_vec ; flat_field_{t-1}] = 5D wide -- the three field states
        # after the PREVIOUS step, exactly the carry the GRU read, now an
        # input column because the field scan runs first. A mean over the
        # other slots' states used to ride here too (the "gestalt"); it
        # was redundant with flat_field -- which is fed the SUM of every
        # message -- and with the trunk's read-time attention over the
        # HISTORY_ENTITY rows (deleted 2026-09-02).
        self.slot_cell = GatedLinearCell(
            entity_size, dtype=self.cfg.dtype, name="slot_cell"
        )
        self.field_cell = GatedLinearCell(
            entity_size, dtype=self.cfg.dtype, name="field_cell"
        )

    def initial_state(self) -> tuple[jax.Array, jax.Array]:
        h_slots = jnp.repeat(self.initial_slot_state, NUM_PUBLIC_SLOTS, axis=0).astype(
            self.cfg.dtype
        )
        h_field = self.initial_field_state.astype(self.cfg.dtype)
        return h_slots, h_field

    def resolve_initial(
        self, carry: HistoryCarry
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """What the window starts from: ((12, D) f32 slot states, (3, D)
        f32 field states, (12, D) node snapshots in the compute dtype).

        With no carry leaves (the default everywhere but the actor) this is
        the learned h0 and an all-zero snapshot, with no select in the
        trace -- the from-scratch function, bit for bit. With leaves
        present, `valid` selects per call between the carried state and
        that same h0, so the actor's full-window recompute is the same
        function too.
        """
        h0_slots, h0_field = self.initial_state()
        h0_slots = h0_slots.astype(jnp.float32)
        h0_field = h0_field.astype(jnp.float32)
        node0 = jnp.zeros(h0_slots.shape, self.cfg.dtype)
        if isinstance(carry.valid, tuple):
            return h0_slots, h0_field, node0
        return (
            jnp.where(carry.valid, carry.slot_states.astype(jnp.float32), h0_slots),
            jnp.where(carry.valid, carry.field_states.astype(jnp.float32), h0_field),
            jnp.where(carry.valid, carry.node_snapshots.astype(self.cfg.dtype), node0),
        )

    def _recur(
        self,
        slot_inputs: jax.Array,
        field_inputs: jax.Array,
        touched: jax.Array,
        step_valid: jax.Array,
        h0_slots: jax.Array,
        h0_field: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """The recurrent half: (H, 12, 2D) precomputed slot inputs
        [messages ; field_vec], (H, 3, 2D) field inputs, (H, 12) bool
        touched, (H,) bool step_valid, the f32 (12, D) / (3, D) states the
        window starts from -> ((H, 12, D), (H, 3, D)) states after each
        step, in the compute dtype, the (H, 12) mean slot write gate
        (telemetry: pinned at 0 or 1 is the collapse shape), and the two
        post-window states in f32 (the carry).

        Field scan first -- its inputs are all precomputed -- then its
        states, shifted one step back (the state BEFORE step t, i.e. the
        GRU's carry), become the trailing 3D columns of the slot input,
        and the slot scan follows. A slot reads its own state, its own
        input and the field states, never another slot's: the scan is a
        per-unit affine map, so that is by construction, and the tests
        pin it with controls. Invalid steps and untouched slots write
        nothing (the identity coefficient).
        """
        field_gate, field_candidate = self.field_cell(field_inputs)
        field_write = jnp.broadcast_to(step_valid[:, None], field_gate.shape[:2])
        field_states = gated_linear_scan(
            field_gate, field_candidate, field_write, h0_field
        )  # (H, 3, D) f32
        # (H, 3D): the field states BEFORE each step -- the GRU's carry.
        previous_field = jnp.concatenate(
            (h0_field[None], field_states[:-1]), axis=0
        ).reshape(field_states.shape[0], -1)
        slot_gate, slot_candidate = self.slot_cell(
            jnp.concatenate(
                (
                    slot_inputs,
                    jnp.broadcast_to(
                        previous_field.astype(slot_inputs.dtype)[:, None],
                        slot_inputs.shape[:2] + previous_field.shape[-1:],
                    ),
                ),
                axis=-1,
            )
        )
        slot_states = gated_linear_scan(
            slot_gate, slot_candidate, touched & step_valid[:, None], h0_slots
        )  # (H, 12, D) f32
        return (
            slot_states.astype(self.cfg.dtype),
            field_states.astype(self.cfg.dtype),
            slot_gate.mean(-1),
            slot_states[-1],
            field_states[-1],
        )

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
        carry: HistoryCarry = HistoryCarry(),
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
            carry: the state the window starts from (`resolve_initial`);
                the default is the learned h0.
        """
        h0_slots, h0_field, node0 = self.resolve_initial(carry)
        relevant, edge_mask = relevant_edges(history_field)  # (H, K)
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

        # ---- the directed message (TGN, 2026-09-01; step GAT 2026-09-02) --
        # A step's SOURCE rows are the ones carrying a real major arg (the
        # mover of a move/switch/faint/cant). Each row's message is its own
        # [node ; edge ; side ; is_src] plus what the step GAT reads off the
        # OTHER rows of the step -- so mover and target coexist in one
        # vector, the relation "move X did N to Y" the per-slot scatter
        # used to destroy, without the masked source mean that conflated
        # a doubles spread move's two movers. TGN proper would use the
        # source's MEMORY here; that is carry-dependent and hence serial,
        # so the rows' raw cache embeddings (their revealed state at event
        # time) stand in, keeping the whole message batch precomputable.
        edge_majors = jnp.take(edge_major_args, relevant, axis=0)  # (H, K)
        is_src = source_rows(edge_majors, edge_mask)
        row_inputs = jnp.concatenate(
            (
                node_embeddings,
                edge_embeddings,
                side_onehot,
                is_src.astype(node_embeddings.dtype)[..., None],
            ),
            axis=-1,
        )  # (H, K, 2D + 3)
        attended, step_attention_probs = self.step_attention(row_inputs, edge_mask)
        messages = (
            self.message_projection(
                jnp.concatenate(
                    (
                        row_inputs,
                        jnp.broadcast_to(
                            field_step_embeddings[:, None], node_embeddings.shape
                        ),
                    ),
                    axis=-1,
                )
            )
            + attended
        )

        # ---- batched precompute (2026-09-01) -----------------------------
        # Everything that does not read a carry, done once over the whole
        # history as batched ops: the edge scatter (vmapped segment_sums)
        # and the latest-node stream (a last-touched-value recurrence,
        # solved in parallel by a cummax over touched step indices + one
        # gather). The recurrences themselves are parallel scans (_recur).
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

        # Latest-node stream: node_means at the last touched step <= t, or
        # the window's starting snapshot (0 from scratch, the carried one
        # on a suffix) before any touch -- exactly the old scan's
        # latest_nodes carry, without the carry.
        step_index = jnp.arange(touched.shape[0])[:, None]
        last_touched = jax.lax.cummax(
            jnp.where(touched, step_index, -1), axis=0
        )  # (H, 12)
        gathered = jnp.take_along_axis(
            node_means, last_touched.clip(min=0)[..., None], axis=0
        )
        node_snapshots = jnp.where(
            (last_touched >= 0)[..., None], gathered, node0[None]
        ).astype(self.cfg.dtype)

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

        slot_inputs = jnp.concatenate(
            (
                slot_messages,
                jnp.broadcast_to(
                    field_step_embeddings[:, None, :], slot_messages.shape
                ),
            ),
            axis=-1,
        )  # (H, 12, 2D)
        field_inputs = jnp.concatenate(
            (field_row_embeddings.astype(self.cfg.dtype), side_messages), axis=-1
        )  # (H, 3, 2D)
        (
            slot_snapshots,
            field_snapshots,
            step_slot_gate,
            final_slot_state,
            final_field_state,
        ) = self._recur(
            slot_inputs, field_inputs, touched, step_valid, h0_slots, h0_field
        )

        return PerSlotHistoryOutput(
            slot_snapshots=slot_snapshots,
            field_snapshots=field_snapshots,
            node_snapshots=node_snapshots,
            final_slot_state=final_slot_state,
            final_field_state=final_field_state,
            step_valid=step_valid,
            step_request_count=step_request_count,
            step_attention_probs=step_attention_probs,
            step_row_mask=edge_mask,
            step_source_rows=is_src,
            step_slot_gate=step_slot_gate,
            step_touched=touched,
        )

    def state_at_requests(
        self,
        history_output: PerSlotHistoryOutput,
        request_counts: jax.Array,
        carry: HistoryCarry = HistoryCarry(),
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """For each request, gather the state after the last history step whose
        request_count <= the request's; with no such step, what the window
        started from (`resolve_initial` of the same carry the scan ran on --
        a zero-new-steps suffix returns the carry itself).
        (T,) -> ((T, 12, D) slot states, (T, 3, D) field states,
        (T, 12, D) latest node snapshots)."""
        h0_slots, h0_field, node0 = self.resolve_initial(carry)
        h0_slots = h0_slots.astype(self.cfg.dtype)
        h0_field = h0_field.astype(self.cfg.dtype)
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
            return slots, field, nodes

        return jax.vmap(gather_one)(request_counts)
