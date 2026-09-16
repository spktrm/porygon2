"""The event world model over the trunk's public output rows.

State: the 55-row public sequence after the trunk at one history step
(PUBLIC_SEQUENCE_ROWS). Step: one service edge. Two learned parts, both
observers of the state (the state arrives under stop_gradient):

- `MajorArgDecoder`: the discrete grammar of the NEXT event -- KIND,
  ACTOR, MOVE, TARGET, TOUCHED (+ a new-turn bit) -- decoded
  autoregressively, cross-attending the state rows. On own-side events it
  is our public action model, on the opponent's it is theirs. The own
  declared action enters as one key-only token that the MOVE / TARGET /
  TOUCHED queries may read only when the actor is our own side: the
  opponent's choice is simultaneous and cannot read our declaration.
- `LatentFlow`: a rectified flow over the DIFFERENCE between consecutive
  states, scaled per public group, endpoint-parameterised behind a
  zero-init projection so every sample is the copy predictor at init, and
  SPARSE: only the rows the decoded touched set names (plus the summary
  rows every event reaches) are denoised; every other row is an exact copy.
  `MeanStep` is the matched control: the same blocks under the plain
  delta MSE.
"""

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike
from ml_collections import ConfigDict

from rl.environment.data import NUM_MOVES
from rl.environment.protos.enums_pb2 import MovesEnum
from rl.model.constants import (
    ALLY_TARGET_ROWS,
    ENEMY_TARGET_ROWS,
    HISTORY_ENTITY_ROWS,
    NUM_PUBLIC_SEQUENCE_ROWS,
    NUM_PUBLIC_SLOTS,
    PUBLIC_CLS_LOCAL_ROW,
    PUBLIC_ROWS,
    PUBLIC_SEQUENCE_ROWS,
    SEQUENCE_GROUP_IDS,
    SEQUENCE_SLICES,
    TARGET_ROWS,
    SequenceGroup,
)
from rl.model.modules import FFWMLP, MLP, MultiHeadAttention, RMSNorm
from rl.model.trunk import Trunk
from rl.offline.event_labels import (
    NUM_DECLARED_KINDS,
    NUM_EVENT_KINDS,
    DeclaredKind,
    EventKind,
)

NUM_SLOT_CLASSES = NUM_PUBLIC_SLOTS + 1  # + NO_SLOT
NUM_TOUCHED_BITS = NUM_PUBLIC_SLOTS + 1  # + the field bit
FIELD_TOUCHED_BIT = NUM_PUBLIC_SLOTS
REAL_MOVE_FLOOR = MovesEnum.MOVES_ENUM___SWITCH_IN
# Decoder token positions; DECLARED is key-only.
DECLARED, KIND, ACTOR, MOVE, TARGET, TOUCHED = range(6)
NUM_DECODER_TOKENS = 6
NUM_EVENT_TOKENS = 4  # kind, actor, move, target enter the flow as tokens

# Local (0..8) group id per public-sequence row; the flow's scale is per group.
_public_group_ids = SEQUENCE_GROUP_IDS[PUBLIC_SEQUENCE_ROWS]
PUBLIC_GROUPS, LOCAL_GROUP_IDS = np.unique(_public_group_ids, return_inverse=True)
NUM_PUBLIC_GROUPS = len(PUBLIC_GROUPS)
LOCAL_GROUP_NAMES = [SequenceGroup(int(group)).name.lower() for group in PUBLIC_GROUPS]

_summary = np.zeros(NUM_PUBLIC_SEQUENCE_ROWS, dtype=bool)
for _group in (
    SequenceGroup.INFO,
    SequenceGroup.HISTORY_REGISTER,
    SequenceGroup.PUBLIC_REGISTER,
):
    _summary[SEQUENCE_SLICES[_group]] = True
_summary[PUBLIC_CLS_LOCAL_ROW] = True
SUMMARY_ROWS = _summary
FIELD_UPDATE_ROWS = np.zeros(NUM_PUBLIC_SEQUENCE_ROWS, dtype=bool)
FIELD_UPDATE_ROWS[SEQUENCE_SLICES[SequenceGroup.FIELD]] = True
FIELD_UPDATE_ROWS[SEQUENCE_SLICES[SequenceGroup.HISTORY_FIELD]] = True
ALLY_TARGET_SEQUENCE_ROWS = TARGET_ROWS.start + ALLY_TARGET_ROWS
ENEMY_TARGET_SEQUENCE_ROWS = TARGET_ROWS.start + ENEMY_TARGET_ROWS


def update_rows(
    touched: jax.Array,
    field_touched: jax.Array,
    ally_active_slots: jax.Array,
    enemy_active_slots: jax.Array,
) -> jax.Array:
    """(55,) bool: the rows one event may change. A touched slot's entity
    and history rows, the field triples when the field bit is set, the
    target rows whose gathered active entity is touched (derived, not
    predicted; -1 = no such active), and always the summary rows."""
    rows = jnp.asarray(SUMMARY_ROWS)
    slot_rows = jnp.zeros(NUM_PUBLIC_SEQUENCE_ROWS, jnp.bool_)
    slot_rows = slot_rows.at[PUBLIC_ROWS].set(touched)
    slot_rows = slot_rows.at[HISTORY_ENTITY_ROWS].set(touched)
    rows = rows | slot_rows
    rows = rows | (jnp.asarray(FIELD_UPDATE_ROWS) & field_touched)

    def active_touched(slots):
        return (slots >= 0) & touched[slots.clip(0, NUM_PUBLIC_SLOTS - 1)]

    rows = rows.at[jnp.asarray(ALLY_TARGET_SEQUENCE_ROWS)].set(
        rows[jnp.asarray(ALLY_TARGET_SEQUENCE_ROWS)] | active_touched(ally_active_slots)
    )
    rows = rows.at[jnp.asarray(ENEMY_TARGET_SEQUENCE_ROWS)].set(
        rows[jnp.asarray(ENEMY_TARGET_SEQUENCE_ROWS)]
        | active_touched(enemy_active_slots)
    )
    return rows


def actor_mask(kind: jax.Array, num_revealed: jax.Array) -> jax.Array:
    """(13,) legal ACTOR classes given KIND: a revealed slot, the next
    unrevealed slot when a switch/drag reveals it, NO_SLOT only for the
    kinds that have no actor."""
    slots = jnp.arange(NUM_PUBLIC_SLOTS)
    revealed = slots < num_revealed
    reveals = (kind == EventKind.SWITCH) | (kind == EventKind.DRAG)
    next_slot = slots == num_revealed
    no_actor = (kind == EventKind.RESIDUAL) | (kind == EventKind.END)
    slot_ok = (revealed | (reveals & next_slot)) & ~no_actor
    return jnp.concatenate([slot_ok, no_actor[None]])


def move_mask(kind: jax.Array) -> jax.Array:
    """(NUM_MOVES,) legal MOVE ids: a real move for MOVE, a real move or
    the no-move id 0 for CANT; the position is off the loss otherwise."""
    ids = jnp.arange(NUM_MOVES)
    real = ids >= REAL_MOVE_FLOOR
    return jnp.where(kind == EventKind.CANT, real | (ids == 0), real)


def target_mask(num_revealed: jax.Array) -> jax.Array:
    slots = jnp.arange(NUM_PUBLIC_SLOTS)
    return jnp.concatenate([slots < num_revealed, jnp.ones(1, jnp.bool_)])


def move_position_valid(kind: jax.Array) -> jax.Array:
    return (kind == EventKind.MOVE) | (kind == EventKind.CANT)


def target_position_valid(kind: jax.Array) -> jax.Array:
    return kind == EventKind.MOVE


@chex.dataclass
class EventTokens:
    kind: ArrayLike = ()
    actor: ArrayLike = ()
    move: ArrayLike = ()
    target: ArrayLike = ()


@chex.dataclass
class DecoderLogits:
    kind: ArrayLike = ()  # (7,)
    new_turn: ArrayLike = ()  # ()
    actor: ArrayLike = ()  # (13,)
    move: ArrayLike = ()  # (NUM_MOVES,)
    target: ArrayLike = ()  # (13,)
    touched: ArrayLike = ()  # (13,)


class CandidateDecoderBlock(nn.Module):
    """One decoder block over the grammar tokens: masked self-attention
    over the tokens, cross-attention over the state rows, SwiGLU FFW, all
    pre-RMSNorm with plain residuals; the queries are the tokens only."""

    cfg: ConfigDict

    @nn.compact
    def __call__(self, carry, masks):
        tokens, rows = carry
        self_mask, cross_mask = masks
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
            q=normed, kv=normed, mask=self_mask
        )
        tokens = tokens + MultiHeadAttention(name="cross_attention", **attention)(
            q=RMSNorm()(tokens), kv=RMSNorm()(rows), mask=cross_mask
        )
        tokens = tokens + FFWMLP(
            hidden_size=self.cfg.hidden_size, use_bias=self.cfg.use_bias, name="ffw"
        )(RMSNorm()(tokens))
        return (tokens, rows), None


def decoder_self_mask(actor_is_mine: jax.Array) -> jax.Array:
    """(6, 6) query x key: causal over the grammar positions; the DECLARED
    key is readable by KIND and ACTOR always and by MOVE / TARGET / TOUCHED
    only for an own-side actor."""
    causal = jnp.tril(jnp.ones((NUM_DECODER_TOKENS, NUM_DECODER_TOKENS), jnp.bool_))
    reads_declared = jnp.array([True, True, True, False, False, False])
    reads_declared = reads_declared | actor_is_mine
    return causal.at[:, DECLARED].set(reads_declared)


class MajorArgDecoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        width = self.cfg.model_size
        self.position_embeddings = self.param(
            "position_embeddings",
            nn.initializers.normal(0.02),
            (NUM_DECODER_TOKENS, width),
        )
        block = nn.remat(
            CandidateDecoderBlock, policy=jax.checkpoint_policies.nothing_saveable
        )
        self.blocks = nn.scan(
            block,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=nn.broadcast,
            length=self.cfg.num_blocks,
        )(self.cfg, name="blocks")
        self.output_norm = RMSNorm()
        self.kind_head = nn.Dense(NUM_EVENT_KINDS + 1, name="kind_head")
        self.actor_head = nn.Dense(NUM_SLOT_CLASSES, name="actor_head")
        self.move_head = nn.Dense(NUM_MOVES, name="move_head")
        self.target_head = nn.Dense(NUM_SLOT_CLASSES, name="target_head")
        self.touched_head = nn.Dense(NUM_TOUCHED_BITS, name="touched_head")

    def __call__(
        self,
        rows: jax.Array,
        row_valid: jax.Array,
        token_inputs: jax.Array,
        actor_is_mine: jax.Array,
    ) -> DecoderLogits:
        """`token_inputs` (6, D): the DECLARED token and, per grammar
        position, the embedding of the PREVIOUS position's value (teacher
        forced in training, the sampled value at search time)."""
        dtype = rows.dtype
        tokens = token_inputs.astype(dtype) + self.position_embeddings.astype(dtype)
        self_mask = decoder_self_mask(actor_is_mine)
        cross_mask = jnp.broadcast_to(
            row_valid[None, :], (NUM_DECODER_TOKENS, row_valid.shape[0])
        )
        (tokens, _), _ = self.blocks(
            (tokens, rows.astype(dtype)), (self_mask, cross_mask)
        )
        tokens = self.output_norm(tokens).astype(jnp.float32)
        kind_logits = self.kind_head(tokens[KIND])
        return DecoderLogits(
            kind=kind_logits[:-1],
            new_turn=kind_logits[-1],
            actor=self.actor_head(tokens[ACTOR]),
            move=self.move_head(tokens[MOVE]),
            target=self.target_head(tokens[TARGET]),
            touched=self.touched_head(tokens[TOUCHED]),
        )


def sinusoidal_time(t: jax.Array, width: int) -> jax.Array:
    half = width // 2
    frequencies = jnp.exp(-jnp.arange(half) * (np.log(10000.0) / max(half - 1, 1)))
    angles = t * frequencies
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)])


class LatentFlow(nn.Module):
    """Endpoint-parameterised rectified flow over the scaled difference."""

    cfg: ConfigDict

    def setup(self):
        width = self.cfg.block.model_size
        self.slot_embedding = self.param(
            "slot_embedding",
            nn.initializers.normal(0.02),
            (NUM_PUBLIC_SEQUENCE_ROWS, width),
        )
        self.token_type_embedding = self.param(
            "token_type_embedding",
            nn.initializers.normal(0.02),
            (NUM_EVENT_TOKENS, width),
        )
        self.noise_projection = nn.Dense(width, use_bias=False, name="noise_projection")
        self.time_projection = nn.Dense(width, name="time_projection")
        self.blocks = Trunk(self.cfg.block, name="blocks")
        self.out_proj = nn.Dense(
            width, kernel_init=nn.initializers.zeros, use_bias=False, name="out_proj"
        )

    def endpoint(
        self,
        rows: jax.Array,
        event_tokens: jax.Array,
        x_t: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        """x̂1: the predicted scaled difference from the noisy point x_t at
        time t, (55, D), zero at init."""
        dtype = rows.dtype
        time = self.time_projection(sinusoidal_time(t, rows.shape[-1]).astype(dtype))
        state_rows = (
            rows.astype(dtype)
            + self.slot_embedding.astype(dtype)
            + self.noise_projection(x_t.astype(dtype))
            + time[None]
        )
        tokens = event_tokens.astype(dtype) + self.token_type_embedding.astype(dtype)
        sequence = jnp.concatenate([state_rows, tokens], axis=0)
        valid = jnp.ones(sequence.shape[0], jnp.bool_)
        mask = jnp.ones((sequence.shape[0], sequence.shape[0]), jnp.bool_)
        hidden = self.blocks(sequence, valid, mask)
        return self.out_proj(hidden[:NUM_PUBLIC_SEQUENCE_ROWS]).astype(jnp.float32)

    def sample(
        self,
        rows: jax.Array,
        event_tokens: jax.Array,
        row_mask: jax.Array,
        scale: jax.Array,
        rng: jax.Array,
        num_steps: int,
    ) -> jax.Array:
        """Euler integration of v = (x̂1 - x_t) / (1 - t) from noise on the
        masked rows; the final point IS the last endpoint prediction, so a
        zero-init flow returns the copy predictor bit for bit."""
        width = rows.shape[-1]
        noise = jax.random.normal(rng, (NUM_PUBLIC_SEQUENCE_ROWS, width), jnp.float32)
        x = jnp.where(row_mask[:, None], noise, 0.0)

        # A lifted scan (params broadcast), not lax.scan: the endpoint net's
        # parameters are created on its first call, which under a raw scan
        # happens inside the traced body.
        def euler(module, carry, index):
            x, rows, event_tokens, row_mask = carry
            t = index.astype(jnp.float32) / num_steps
            x1 = module.endpoint(rows, event_tokens, x, t)
            velocity = (x1 - x) / (1.0 - t)
            x = jnp.where(row_mask[:, None], x + velocity / num_steps, 0.0)
            return (x, rows, event_tokens, row_mask), None

        scanned = nn.scan(
            euler,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )
        (x, _, _, _), _ = scanned(
            self, (x, rows, event_tokens, row_mask), jnp.arange(num_steps)
        )
        delta = x * scale[jnp.asarray(LOCAL_GROUP_IDS)][:, None]
        return rows + jnp.where(row_mask[:, None], delta, 0.0).astype(rows.dtype)


class MeanStep(nn.Module):
    """The matched control: the flow's blocks under the plain delta MSE."""

    cfg: ConfigDict

    def setup(self):
        width = self.cfg.block.model_size
        self.slot_embedding = self.param(
            "slot_embedding",
            nn.initializers.normal(0.02),
            (NUM_PUBLIC_SEQUENCE_ROWS, width),
        )
        self.token_type_embedding = self.param(
            "token_type_embedding",
            nn.initializers.normal(0.02),
            (NUM_EVENT_TOKENS, width),
        )
        self.blocks = Trunk(self.cfg.block, name="blocks")
        self.out_proj = nn.Dense(
            width, kernel_init=nn.initializers.zeros, use_bias=False, name="out_proj"
        )

    def __call__(self, rows: jax.Array, event_tokens: jax.Array) -> jax.Array:
        dtype = rows.dtype
        state_rows = rows.astype(dtype) + self.slot_embedding.astype(dtype)
        tokens = event_tokens.astype(dtype) + self.token_type_embedding.astype(dtype)
        sequence = jnp.concatenate([state_rows, tokens], axis=0)
        valid = jnp.ones(sequence.shape[0], jnp.bool_)
        mask = jnp.ones((sequence.shape[0], sequence.shape[0]), jnp.bool_)
        hidden = self.blocks(sequence, valid, mask)
        return self.out_proj(hidden[:NUM_PUBLIC_SEQUENCE_ROWS]).astype(jnp.float32)


@chex.dataclass
class StepTerms:
    """Per-step raw terms; the trainer pools them over the batch."""

    logits: DecoderLogits = None
    grammar_valid: ArrayLike = ()  # (5,) which positions carry a loss
    flow_error: ArrayLike = ()  # (55,) |x̂1 - x1|² per row
    mean_error: ArrayLike = ()  # (55,) |δ̂ - x1|² per row, the control
    update_mask: ArrayLike = ()  # (55,)
    delta_energy: ArrayLike = ()  # (55,) |δ|² per row, unscaled
    terminal_logits: ArrayLike = ()  # (3,)


class EventWorldModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        width = self.cfg.model_size
        init = nn.initializers.normal(0.02)
        self.kind_table = self.param("kind_table", init, (NUM_EVENT_KINDS, width))
        self.slot_table = self.param("slot_table", init, (NUM_SLOT_CLASSES, width))
        self.move_table = self.param("move_table", init, (NUM_MOVES, width))
        self.declared_kind_table = self.param(
            "declared_kind_table", init, (NUM_DECLARED_KINDS, width)
        )
        self.start_token = self.param("start_token", init, (width,))
        # The per-group RMS difference the flow is scaled by: a parameter
        # in name only (the optimiser never touches it) so the actor's
        # params view carries it; the trainer writes its EMA here.
        self.delta_scale = self.param(
            "delta_scale", nn.initializers.ones, (NUM_PUBLIC_GROUPS,)
        )
        self.decoder = MajorArgDecoder(self.cfg.decoder, name="decoder")
        self.flow = LatentFlow(self.cfg.flow, name="flow")
        self.mean_step = MeanStep(self.cfg.flow, name="mean_step")
        self.terminal_outcome_head = MLP(
            layer_sizes=(width, 3), use_bias=True, name="terminal_outcome_head"
        )

    def declared_token(self, declared_kind: jax.Array, declared_arg: jax.Array):
        arg = jnp.where(
            declared_kind == DeclaredKind.MOVE,
            self.move_table[declared_arg.clip(0, NUM_MOVES - 1)],
            self.slot_table[declared_arg.clip(0, NUM_SLOT_CLASSES - 1)],
        )
        arg = jnp.where(declared_kind == DeclaredKind.UNKNOWN, 0.0, arg)
        return self.declared_kind_table[declared_kind] + arg

    def event_token_embeddings(self, tokens: EventTokens) -> jax.Array:
        """(4, D): the event as the flow sees it."""
        return jnp.stack(
            [
                self.kind_table[tokens.kind],
                self.slot_table[tokens.actor],
                self.move_table[tokens.move.clip(0, NUM_MOVES - 1)],
                self.slot_table[tokens.target],
            ]
        )

    def decoder_inputs(
        self, declared_kind: jax.Array, declared_arg: jax.Array, tokens: EventTokens
    ) -> jax.Array:
        """(6, D): DECLARED, then each position's input = the previous
        position's value (KIND reads the start token)."""
        event = self.event_token_embeddings(tokens)
        return jnp.stack(
            [
                self.declared_token(declared_kind, declared_arg),
                self.start_token,
                event[0],
                event[1],
                event[2],
                event[3],
            ]
        )

    def decode(
        self,
        rows: jax.Array,
        row_valid: jax.Array,
        declared_kind: jax.Array,
        declared_arg: jax.Array,
        tokens: EventTokens,
        actor_is_mine: jax.Array,
    ) -> DecoderLogits:
        inputs = self.decoder_inputs(declared_kind, declared_arg, tokens)
        return self.decoder(rows, row_valid, inputs, actor_is_mine)

    def imagine(
        self,
        rows: jax.Array,
        tokens: EventTokens,
        row_mask: jax.Array,
        scale: jax.Array,
        rng: jax.Array,
    ) -> jax.Array:
        return self.flow.sample(
            rows,
            self.event_token_embeddings(tokens),
            row_mask,
            scale,
            rng,
            self.cfg.flow_steps,
        )

    def terminal_outcome(self, rows: jax.Array) -> jax.Array:
        return self.terminal_outcome_head(rows[PUBLIC_CLS_LOCAL_ROW]).astype(
            jnp.float32
        )

    def step_terms(
        self,
        rows: jax.Array,
        row_valid: jax.Array,
        next_rows: jax.Array,
        declared_kind: jax.Array,
        declared_arg: jax.Array,
        tokens: EventTokens,
        actor_is_mine: jax.Array,
        row_mask: jax.Array,
        scale: jax.Array,
        noise: jax.Array,
        time: jax.Array,
    ) -> StepTerms:
        """Teacher-forced terms for ONE (state, next state) pair: the
        grammar logits, the flow's endpoint error at one (noise, time)
        draw, the mean control's error, and the raw delta energy the
        trainer pools into the per-group scale."""
        logits = self.decode(
            rows, row_valid, declared_kind, declared_arg, tokens, actor_is_mine
        )
        event = self.event_token_embeddings(tokens)
        delta = (next_rows - rows).astype(jnp.float32)
        x1 = delta / scale[jnp.asarray(LOCAL_GROUP_IDS)][:, None]
        x1 = jnp.where(row_mask[:, None], x1, 0.0)
        x_t = (1.0 - time) * jnp.where(row_mask[:, None], noise, 0.0) + time * x1
        endpoint = self.flow.endpoint(rows, event, x_t, time)
        flow_error = jnp.square(endpoint - x1).sum(-1)
        mean_error = jnp.square(self.mean_step(rows, event) - x1).sum(-1)
        grammar_valid = jnp.stack(
            [
                jnp.ones((), jnp.bool_),
                tokens.kind != EventKind.END,
                move_position_valid(tokens.kind),
                target_position_valid(tokens.kind),
                tokens.kind != EventKind.END,
            ]
        )
        return StepTerms(
            logits=logits,
            grammar_valid=grammar_valid,
            flow_error=flow_error,
            mean_error=mean_error,
            update_mask=row_mask,
            delta_energy=jnp.square(delta).sum(-1),
            terminal_logits=self.terminal_outcome(next_rows),
        )


def pooled_group_loss(
    error: jax.Array, energy: jax.Array, mask: jax.Array, floor: float
) -> tuple[jax.Array, jax.Array]:
    """The normalised difference loss and its per-group values: within each
    public group, the masked error sum over the masked energy sum (floored),
    then the mean over groups. Copy scores 1, the exact difference 0, the
    negated difference 4, whatever the rows' magnitudes. `error` and
    `energy` are (..., 55), `mask` (..., 55) bool."""
    group_ids = jnp.asarray(LOCAL_GROUP_IDS)
    weight = mask.astype(jnp.float32)
    per_group_error = jax.ops.segment_sum(
        (error * weight).reshape(-1, NUM_PUBLIC_SEQUENCE_ROWS).sum(0),
        group_ids,
        num_segments=NUM_PUBLIC_GROUPS,
    )
    per_group_energy = jax.ops.segment_sum(
        (energy * weight).reshape(-1, NUM_PUBLIC_SEQUENCE_ROWS).sum(0),
        group_ids,
        num_segments=NUM_PUBLIC_GROUPS,
    )
    per_group = per_group_error / jnp.maximum(per_group_energy, floor)
    return per_group.mean(), per_group


def group_delta_scale(
    delta_energy: jax.Array, mask: jax.Array, floor: float
) -> jax.Array:
    """(9,) the RMS difference per public group over the masked rows: the
    unit the flow works in. Floored so an all-static group never divides by
    zero."""
    group_ids = jnp.asarray(LOCAL_GROUP_IDS)
    weight = mask.astype(jnp.float32).reshape(-1, NUM_PUBLIC_SEQUENCE_ROWS)
    energy = (delta_energy.reshape(-1, NUM_PUBLIC_SEQUENCE_ROWS) * weight).sum(0)
    total = jax.ops.segment_sum(energy, group_ids, num_segments=NUM_PUBLIC_GROUPS)
    count = jax.ops.segment_sum(
        weight.sum(0), group_ids, num_segments=NUM_PUBLIC_GROUPS
    )
    return jnp.sqrt(jnp.maximum(total / jnp.maximum(count, 1.0), floor))


def masked_log_softmax(logits: jax.Array, legal: jax.Array) -> jax.Array:
    masked = jnp.where(legal, logits, -1e9)
    return jax.nn.log_softmax(masked, axis=-1)


def grammar_nll(
    logits: DecoderLogits,
    tokens: EventTokens,
    touched: jax.Array,
    new_turn: jax.Array,
    num_revealed: jax.Array,
) -> jax.Array:
    """(5,) negative log-likelihood per grammar position under the legality
    masks (KIND, ACTOR, MOVE, TARGET, TOUCHED), TOUCHED being the summed
    Bernoulli NLL of its 13 bits, KIND carrying the new-turn bit too."""
    kind_nll = -masked_log_softmax(logits.kind, jnp.ones(NUM_EVENT_KINDS, jnp.bool_))[
        tokens.kind
    ]
    new_turn_nll = -jax.nn.log_sigmoid(
        jnp.where(new_turn, logits.new_turn, -logits.new_turn)
    )
    actor_nll = -masked_log_softmax(
        logits.actor, actor_mask(tokens.kind, num_revealed)
    )[tokens.actor]
    move_nll = -masked_log_softmax(logits.move, move_mask(tokens.kind))[tokens.move]
    target_nll = -masked_log_softmax(logits.target, target_mask(num_revealed))[
        tokens.target
    ]
    touched_nll = -jax.nn.log_sigmoid(
        jnp.where(touched, logits.touched, -logits.touched)
    ).sum()
    return jnp.stack(
        [kind_nll + new_turn_nll, actor_nll, move_nll, target_nll, touched_nll]
    )
