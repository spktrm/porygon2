import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike
from ml_collections import ConfigDict

from rl.environment.data import NUM_MOVES
from rl.environment.interfaces import PlayerActorInput
from rl.environment.protos.features_pb2 import InfoFeature
from rl.model.config import get_player_model_config
from rl.model.encoder import Encoder
from rl.model.heads import CategoricalValueHeadOutput
from rl.model.modules import (
    FFWMLP,
    MultiHeadAttention,
    create_attention_mask,
    layer_norm,
)
from rl.offline.dataset import MAX_MARGIN, NUM_ACTION_CLASSES, NUM_SURVIVAL_BINS

# Bin values as normalized potentials: expected margin in [-1, 1].
MARGIN_SUPPORT = np.arange(-MAX_MARGIN, MAX_MARGIN + 1) / MAX_MARGIN

# Gated relational rounds over the 13 history tokens before side-pooling.
# Fixed (not config) so RL-side consumers (rl/offline/artifact.py) rebuild
# the exact param tree from the module alone.
NUM_RELATIONAL_ROUNDS = 2

# Elo conditioning: bucket 0 = unknown/unrated (live self-play, rating
# dropout, pre-rating shards), buckets 1..11 = (-inf, 1100), [1100, 1200),
# ..., [2000, inf).
NUM_RATING_BUCKETS = 12


def rating_bucket(rating: jax.Array) -> jax.Array:
    bucket = jnp.clip((rating - 1000) // 100, 0, 10) + 1
    return jnp.where(rating > 0, bucket, 0)


class InteractionOutcomeProbe(nn.Module):
    """Antisymmetrised interaction readout with mirror-antisymmetry by
    construction.

    Six margin scores z = W·(my − opp) + g(my, opp) − g(opp, my) build the
    13-bin logits [-z_6 .. -z_1, tie_bias, z_1 .. z_6] over final
    alive-mon margins -6..+6 (g is a small MLP over the concatenated
    pooled banks; the linear term keeps the old difference probe as an
    exact subspace). Mirroring the perspective swaps the pooled banks,
    negates every term, and exactly REVERSES the margin distribution —
    Φ(mirror(s)) = −Φ(s) for every parameter setting, so SGD cannot
    satisfy the loss by memorizing game identity: only side-differenced
    structure reduces it. (Empirically necessary: a free-form probe
    memorizes pairs and generalizes at chance.) Unlike the pure linear
    probe, g admits my-side × opp-side interaction terms — "their sweeper
    is a threat BECAUSE my remaining mons can't check it" — which a linear
    function of the difference cannot represent.
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(self, my: jax.Array, opp: jax.Array) -> CategoricalValueHeadOutput:
        # my/opp: (T, num_latents * D) flattened side-pooled latents.
        z = nn.Dense(MAX_MARGIN, use_bias=False, name="margin_scores")(my - opp)
        hidden = nn.Dense(self.cfg.entity_size, name="interaction_hidden")
        scores = nn.Dense(MAX_MARGIN, use_bias=False, name="interaction_scores")

        def g(a: jax.Array, b: jax.Array) -> jax.Array:
            return scores(nn.gelu(hidden(jnp.concatenate((a, b), axis=-1))))

        z = z + g(my, opp) - g(opp, my)
        tie_bias = self.param("tie_bias", nn.initializers.zeros_init(), (1,))
        logits = jnp.concatenate(
            (
                -z[..., ::-1],
                jnp.broadcast_to(tie_bias, z.shape[:-1] + (1,)).astype(z.dtype),
                z,
            ),
            axis=-1,
        )
        log_probs = nn.log_softmax(logits, axis=-1)
        probs = jnp.exp(log_probs)
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        values = jnp.asarray(MARGIN_SUPPORT, dtype=logits.dtype)
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


class PerSlotLogitsHead(nn.Module):
    """Small nonlinear per-slot readout shared by every training-time aux
    head: (T, num_slots, D) tokens -> (T, num_slots, num_outputs) logits.
    A small head keeps the pressure on the encoder features rather than on
    a clever readout. All aux heads are local and side-agnostic: they never
    touch the antisymmetric margin readout, and the RL consumption path
    (__call__) never calls them.

    Computes in ``dtype`` (bf16 like the encoder): without it, bf16 tokens
    against f32 params silently promote, and the vocab-sized logit tensors
    become the run's dominant activation cost. Losses upcast to f32 where
    the softmax/log-sum accumulations genuinely need it."""

    num_outputs: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, tokens: jax.Array) -> jax.Array:
        hidden = nn.gelu(
            nn.Dense(tokens.shape[-1], dtype=self.dtype, name="hidden")(tokens)
        )
        return nn.Dense(self.num_outputs, dtype=self.dtype, name="logits")(hidden)


class RelationalRounds(nn.Module):
    """Gated self-attention rounds over the 13 history tokens (12 slot
    tokens + field) before side-pooling.

    The side-pooled readout can only use cross-side structure that already
    exists in the tokens; upstream, sides only mix through the GRU's
    mean-pooled context and one gated cross-read. These rounds let
    "their sweeper vs my remaining bench" be computed relationally. Every
    residual write is behind a zero-init scalar gate, so at initialization
    the rounds are an exact no-op (hand-rule parity stays the floor, and
    checkpoints resumed from pre-round artifacts start unchanged); the
    rounds are permutation-equivariant over tokens, so the mirror swap
    still permutes the token bank exactly and Φ(mirror) = −Φ is untouched.
    """

    cfg: ConfigDict  # the encoder ConfigDict (entity_size, round, ...)
    num_rounds: int = NUM_RELATIONAL_ROUNDS

    @nn.compact
    def __call__(self, tokens: jax.Array, token_valid: jax.Array) -> jax.Array:
        # tokens: (S, D); token_valid: (S,) — invalid slots (never revealed)
        # are masked out of attention and hard-zeroed on the way out.
        rcfg = self.cfg.round
        mask = create_attention_mask(token_valid, token_valid)
        for i in range(self.num_rounds):
            attn = MultiHeadAttention(
                name=f"self_attn_{i}",
                num_heads=rcfg.num_heads,
                qk_size=rcfg.qk_size,
                v_size=rcfg.v_size,
                model_size=self.cfg.entity_size,
                use_bias=rcfg.use_bias,
                dtype=tokens.dtype,
            )(q=layer_norm(tokens), kv=layer_norm(tokens), mask=mask)
            attn_gate = self.param(
                f"attn_gate_{i}", nn.initializers.zeros_init(), (1,)
            ).astype(tokens.dtype)
            tokens = tokens + attn_gate * attn
            ffw = FFWMLP(
                hidden_size=rcfg.hidden_size, use_bias=rcfg.use_bias, name=f"ffw_{i}"
            )(layer_norm(tokens))
            ffw_gate = self.param(
                f"ffw_gate_{i}", nn.initializers.zeros_init(), (1,)
            ).astype(tokens.dtype)
            tokens = tokens + ffw_gate * ffw
        return jnp.where(token_valid[..., None], tokens, 0)


@chex.dataclass
class OfflineAuxOutput:
    """Training-time aux head logits, all per (T, num_slots, ...); dead
    weight at RL consumption time."""

    # Discounted time-to-next-faint bins (imminent doom).
    survival: ArrayLike = ()
    # Next executed move: full move vocab + a "never acts again" class.
    # Predicting this before the move is revealed requires a set posterior.
    next_action: ArrayLike = ()
    # Discounted time until this mon next uses a move unrevealed as of t —
    # the longer-horizon "an unseen move is coming" hazard.
    unseen: ArrayLike = ()
    # Eventually-revealed move set (positive-unlabelled multi-label).
    revealed_set: ArrayLike = ()


class Porygon2OfflineCritic(nn.Module):
    """Antisymmetrised outcome readout over the recurrent history pathway
    only.

    Pathway: Encoder.encode_history (GRU slot states + latest node
    snapshots) -> Encoder.read_history_into_nodes (snapshots make a
    zero-init-gated cross-read of the recurrent states, mirroring the RL
    trunk's history reads) -> RelationalRounds (gated self-attention over
    the 13 tokens, letting cross-side threat structure form before sides
    are pooled apart) -> Encoder.pool_history twice with side masks
    (shared pool params): my-side tokens + field, opponent tokens + field.
    The two latent banks feed InteractionOutcomeProbe: a linear difference
    term plus an antisymmetrised interaction MLP g(my, opp) − g(opp, my).

    Public-only by construction: the pathway reads packed public
    entity/edge caches, field history, and request counts — private team,
    own moveset, and action masks are architecturally unreachable, and
    replay-exported inputs match live inputs exactly.

    This model is a standalone artifact: it shares Encoder code with the
    RL model for convenience, but its trained params never enter the RL
    network. The RL learner consumes it only as a frozen, stop-gradient
    potential Φ (offline_critic_ckpt_path), so RL trains fully from
    scratch with no frozen or warm-started subtrees.

    Training additionally supervises per-slot aux heads (with_aux) on the
    shared tokens — feature-shaping losses that are dead weight at
    consumption time:
    - survival: discounted time-to-next-faint (imminent doom);
    - next_action: the mon's next executed move (a set posterior is
      required to score well before the move is revealed);
    - unseen: discounted time until the mon next uses a move unrevealed
      as of the current step (latent-threat hazard, longer horizon);
    - revealed_set: the mon's eventually-revealed move set
      (positive-unlabelled).
    """

    cfg: ConfigDict
    # Elo conditioning: embed each side's pre-game rating bucket as one
    # extra token in that side's pooled read. False reproduces the
    # pre-rating architecture exactly (no rating params created), which is
    # how checkpoints trained before the feature stay loadable.
    rating_conditioning: bool = True

    def setup(self):
        self.encoder = Encoder(self.cfg.encoder)
        self.relational_rounds = RelationalRounds(self.cfg.encoder)
        # The outcome probe stays f32: its tensors are tiny (13 bins) and
        # the margin logits are the artifact's entire product.
        self.outcome_head = InteractionOutcomeProbe(self.cfg)
        dtype = self.cfg.dtype
        self.survival_head = PerSlotLogitsHead(NUM_SURVIVAL_BINS, dtype=dtype)
        self.next_action_head = PerSlotLogitsHead(NUM_ACTION_CLASSES, dtype=dtype)
        self.unseen_head = PerSlotLogitsHead(NUM_SURVIVAL_BINS, dtype=dtype)
        self.set_head = PerSlotLogitsHead(NUM_MOVES, dtype=dtype)
        self.rating_embed = nn.Embed(
            NUM_RATING_BUCKETS, self.cfg.encoder.entity_size, name="rating_embed"
        )

    def _token_valid(self, actor_input: PlayerActorInput) -> jax.Array:
        # Slots never occupied in this game are masked out of the
        # relational rounds; the field token is always live.
        slot_sides = self.encoder.history_slot_sides(actor_input.packed_history)
        return jnp.concatenate(
            ((slot_sides == 0) | (slot_sides == 1), jnp.ones(1, dtype=jnp.bool_))
        )

    def _tokens_from_states(
        self,
        node_states: jax.Array,
        slot_states: jax.Array,
        field_state: jax.Array,
        token_valid: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        # Photos attend to diaries, trunk-style: each slot's current
        # snapshot makes a zero-init-gated cross-read of the recurrent
        # states, so at init the tokens ARE the raw snapshots (hand-rule
        # parity is the floor — GRU-only tokens lost current hp/faint info
        # to gating and were beaten by a raw hand rule late-game) and
        # history context blends in only where it helps.
        tokens = self.encoder.read_history_into_nodes(
            node_states, slot_states, field_state
        )
        # Relational stage: gated self-attention over [12 slots | field],
        # so cross-side matchup structure exists in the tokens BEFORE the
        # side-pooled readout separates the sides.
        mixed = jax.vmap(self.relational_rounds, in_axes=(0, None))(
            jnp.concatenate((tokens, field_state[:, None, :]), axis=-2), token_valid
        )
        return mixed[:, :-1, :], mixed[:, -1, :]

    def _history_tokens(self, actor_input: PlayerActorInput):
        slot_states, field_state, node_states = self.encoder.encode_history(
            actor_input.env, actor_input.packed_history, actor_input.history
        )
        return self._tokens_from_states(
            node_states, slot_states, field_state, self._token_valid(actor_input)
        )

    def _history_tokens_with_announced(self, actor_input: PlayerActorInput):
        """Realised AND announced token banks, per request. The announced
        bank (Φ_ann) runs the SAME modules end to end — cross-read,
        relational rounds, side-pooling, antisymmetric readout — over the
        announced recurrent state (pre-turn state + outcome-masked turn
        messages) and pre-turn node snapshots, so mirror antisymmetry of
        Φ_ann follows automatically and no new parameters exist."""
        states, announced = self.encoder.encode_history_with_announced(
            actor_input.env, actor_input.packed_history, actor_input.history
        )
        slot_states, field_state, node_states = states
        ann_slot_states, ann_field_state, pre_node_states = announced
        token_valid = self._token_valid(actor_input)
        return (
            self._tokens_from_states(
                node_states, slot_states, field_state, token_valid
            ),
            self._tokens_from_states(
                pre_node_states, ann_slot_states, ann_field_state, token_valid
            ),
        )

    def _outcome(
        self, actor_input: PlayerActorInput, tokens: jax.Array, field_state: jax.Array
    ) -> CategoricalValueHeadOutput:
        slot_sides = self.encoder.history_slot_sides(actor_input.packed_history)
        field_token = jnp.ones(1, dtype=jnp.bool_)
        if self.rating_conditioning:
            # One embedded rating token per side, pooled with that side's
            # slots. Mirroring swaps the perspectives — and with them the
            # rating features and side masks — so the pooled difference
            # still negates exactly and Φ(mirror) = −Φ is untouched.
            # Ratings are constant per game; read them off the first step.
            info = actor_input.env.info
            buckets = rating_bucket(
                jnp.stack(
                    (
                        info[0, InfoFeature.INFO_FEATURE__MY_RATING],
                        info[0, InfoFeature.INFO_FEATURE__OPP_RATING],
                    )
                )
            )
            rating_tokens = jnp.broadcast_to(
                self.rating_embed(buckets)[None],
                (tokens.shape[0], 2, tokens.shape[-1]),
            ).astype(tokens.dtype)
            tokens = jnp.concatenate((tokens, rating_tokens), axis=-2)
            my_extra = jnp.array([True, False])
            opp_extra = jnp.array([False, True])
        else:
            my_extra = jnp.zeros(0, dtype=jnp.bool_)
            opp_extra = jnp.zeros(0, dtype=jnp.bool_)
        # pool_history appends the field token last: mask order is
        # [12 slots, (my_rt, opp_rt), field].
        my_mask = jnp.concatenate((slot_sides == 1, my_extra, field_token))
        opp_mask = jnp.concatenate((slot_sides == 0, opp_extra, field_token))
        my_latents = self.encoder.pool_history(tokens, field_state, my_mask)
        opp_latents = self.encoder.pool_history(tokens, field_state, opp_mask)
        num_steps = tokens.shape[0]
        return self.outcome_head(
            my_latents.reshape(num_steps, -1), opp_latents.reshape(num_steps, -1)
        )

    def __call__(self, actor_input: PlayerActorInput) -> CategoricalValueHeadOutput:
        tokens, field_state = self._history_tokens(actor_input)
        return self._outcome(actor_input, tokens, field_state)

    def with_aux(
        self, actor_input: PlayerActorInput
    ) -> tuple[CategoricalValueHeadOutput, OfflineAuxOutput]:
        """Training entry point: the margin head plus the per-slot aux
        logits (T, 12, ...) off the same tokens. Init with this method so
        the aux params exist; consumers keep calling __call__, which never
        touches the aux heads (their params ride along in the artifact,
        unread)."""
        tokens, field_state = self._history_tokens(actor_input)
        return (
            self._outcome(actor_input, tokens, field_state),
            OfflineAuxOutput(
                survival=self.survival_head(tokens),
                next_action=self.next_action_head(tokens),
                unseen=self.unseen_head(tokens),
                revealed_set=self.set_head(tokens),
            ),
        )

    def announced(
        self, actor_input: PlayerActorInput
    ) -> tuple[CategoricalValueHeadOutput, CategoricalValueHeadOutput]:
        """Consumption entry point for the skill/luck decomposition and
        dice-excised shaping: (Φ, Φ_ann) per step, both through the SAME
        antisymmetric readout. Φ_ann(t) is the margin belief given the
        pre-turn state plus both players' announced choices for the turn
        leading into state t, chance unresolved — so per turn:
        decision = Φ_ann(t+1) − Φ(t), dice = Φ(t+1) − Φ_ann(t+1).
        Adds no parameters: any artifact computes it, but only artifacts
        trained at announced points (manifest announced_states) produce
        calibrated values."""
        (tokens, field_state), (ann_tokens, ann_field_state) = (
            self._history_tokens_with_announced(actor_input)
        )
        return (
            self._outcome(actor_input, tokens, field_state),
            self._outcome(actor_input, ann_tokens, ann_field_state),
        )

    def with_aux_and_announced(self, actor_input: PlayerActorInput) -> tuple[
        CategoricalValueHeadOutput,
        OfflineAuxOutput,
        CategoricalValueHeadOutput,
    ]:
        """Training entry point with announced states: with_aux plus the
        Φ_ann margin head. Announced states are extra supervision points
        for the same trajectory margin label (deep supervision through the
        shared readout); aux heads stay realised-state-only."""
        (tokens, field_state), (ann_tokens, ann_field_state) = (
            self._history_tokens_with_announced(actor_input)
        )
        return (
            self._outcome(actor_input, tokens, field_state),
            OfflineAuxOutput(
                survival=self.survival_head(tokens),
                next_action=self.next_action_head(tokens),
                unseen=self.unseen_head(tokens),
                revealed_set=self.set_head(tokens),
            ),
            self._outcome(actor_input, ann_tokens, ann_field_state),
        )


def get_offline_critic(generation: int, rating_conditioning: bool = True) -> nn.Module:
    return Porygon2OfflineCritic(
        get_player_model_config(generation, train=False),
        rating_conditioning=rating_conditioning,
    )
