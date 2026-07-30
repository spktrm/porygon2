import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from rl.environment.interfaces import PlayerActorInput
from rl.environment.protos.features_pb2 import InfoFeature
from rl.model.config import get_player_model_config
from rl.model.encoder import Encoder
from rl.model.heads import CategoricalValueHeadOutput
from rl.offline.dataset import MAX_MARGIN, NUM_SURVIVAL_BINS

# Bin values as normalized potentials: expected margin in [-1, 1].
MARGIN_SUPPORT = np.arange(-MAX_MARGIN, MAX_MARGIN + 1) / MAX_MARGIN

# Elo conditioning: bucket 0 = unknown/unrated (live self-play, rating
# dropout, pre-rating shards), buckets 1..11 = (-inf, 1100), [1100, 1200),
# ..., [2000, inf).
NUM_RATING_BUCKETS = 12


def rating_bucket(rating: jax.Array) -> jax.Array:
    bucket = jnp.clip((rating - 1000) // 100, 0, 10) + 1
    return jnp.where(rating > 0, bucket, 0)


class AntisymmetricOutcomeProbe(nn.Module):
    """Linear margin probe with mirror-antisymmetry by construction.

    Six margin scores z = W · (my − opp) build the 13-bin logits
    [-z_6 .. -z_1, tie_bias, z_1 .. z_6] over final alive-mon margins
    -6..+6. Mirroring the perspective swaps the pooled summaries, negates
    the difference, negates z, and exactly REVERSES the margin
    distribution — Φ(mirror(s)) = −Φ(s) for every parameter setting, so
    SGD cannot satisfy the loss by memorizing game identity: only
    side-differenced structure reduces it. (Empirically necessary: a
    free-form probe memorizes pairs and generalizes at chance.)
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(self, diff: jax.Array) -> CategoricalValueHeadOutput:
        # diff: (T, num_latents * D) = flattened (my − opp) pooled latents.
        z = nn.Dense(MAX_MARGIN, use_bias=False, name="margin_scores")(diff)
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


class PerSlotSurvivalHead(nn.Module):
    """Auxiliary per-mon discounted-survival readout, training-time only.

    Predicts a categorical over NUM_SURVIVAL_BINS value bins of
    y = discount**(steps to this slot's next faint) from the slot's own
    history token. Purpose: give the encoder a dense, temporally-local,
    per-entity gradient that forces "this mon is doomed soon" into the
    latents BEFORE the faint bit flips — the final-margin label broadcasts
    one distribution across every step, so it carries no timing signal of
    its own. Local and side-agnostic: it never touches the antisymmetric
    margin readout, and the RL consumption path (__call__) never calls it.
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(self, tokens: jax.Array) -> jax.Array:
        # tokens: (T, num_slots, D) -> logits (T, num_slots, bins). A small
        # nonlinear head keeps the pressure on the encoder features rather
        # than on a clever readout.
        hidden = nn.gelu(nn.Dense(tokens.shape[-1], name="hidden")(tokens))
        return nn.Dense(NUM_SURVIVAL_BINS, name="logits")(hidden)


class Porygon2OfflineCritic(nn.Module):
    """Antisymmetric linear outcome probe over the recurrent history
    pathway only.

    Pathway: Encoder.encode_history (GRU slot states + latest node
    snapshots) -> Encoder.read_history_into_nodes (snapshots make a
    zero-init-gated cross-read of the recurrent states, mirroring the RL
    trunk's history reads) -> Encoder.pool_history twice with side masks
    (shared pool params): my-side tokens + field, opponent tokens + field.
    The flattened difference of the two latent banks feeds
    AntisymmetricOutcomeProbe; the probe is a single weight vector.

    Public-only by construction: the pathway reads packed public
    entity/edge caches, field history, and request counts — private team,
    own moveset, and action masks are architecturally unreachable, and
    replay-exported inputs match live inputs exactly.

    This model is a standalone artifact: it shares Encoder code with the
    RL model for convenience, but its trained params never enter the RL
    network. The RL learner consumes it only as a frozen, stop-gradient
    potential Φ (offline_critic_ckpt_path), so RL trains fully from
    scratch with no frozen or warm-started subtrees.

    Training additionally supervises a per-slot discounted-survival aux
    head (with_aux) on the shared tokens — a feature-shaping loss that
    teaches the encoder to see imminent faints before they land, so the
    linear margin probe has something to read earlier than the faint bit.
    The aux head is dead weight at consumption time.
    """

    cfg: ConfigDict
    # Elo conditioning: embed each side's pre-game rating bucket as one
    # extra token in that side's pooled read. False reproduces the
    # pre-rating architecture exactly (no rating params created), which is
    # how checkpoints trained before the feature stay loadable.
    rating_conditioning: bool = True

    def setup(self):
        self.encoder = Encoder(self.cfg.encoder)
        self.outcome_head = AntisymmetricOutcomeProbe(self.cfg)
        self.survival_head = PerSlotSurvivalHead(self.cfg)
        self.rating_embed = nn.Embed(
            NUM_RATING_BUCKETS, self.cfg.encoder.entity_size, name="rating_embed"
        )

    def _history_tokens(self, actor_input: PlayerActorInput):
        slot_states, field_state, node_states = self.encoder.encode_history(
            actor_input.env, actor_input.packed_history, actor_input.history
        )
        # Photos attend to diaries, trunk-style: each slot's current
        # snapshot makes a zero-init-gated cross-read of the recurrent
        # states, so at init the tokens ARE the raw snapshots (hand-rule
        # parity is the floor — GRU-only tokens lost current hp/faint info
        # to gating and were beaten by a raw hand rule late-game) and
        # history context blends in only where it helps.
        tokens = self.encoder.read_history_into_nodes(
            node_states, slot_states, field_state
        )
        return tokens, field_state

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
        diff = (my_latents - opp_latents).reshape(tokens.shape[0], -1)
        return self.outcome_head(diff)

    def __call__(self, actor_input: PlayerActorInput) -> CategoricalValueHeadOutput:
        tokens, field_state = self._history_tokens(actor_input)
        return self._outcome(actor_input, tokens, field_state)

    def with_aux(
        self, actor_input: PlayerActorInput
    ) -> tuple[CategoricalValueHeadOutput, jax.Array]:
        """Training entry point: the margin head plus per-slot survival
        logits (T, 12, NUM_SURVIVAL_BINS) off the same tokens. Init with
        this method so the aux params exist; consumers keep calling
        __call__, which never touches the aux head (its params ride along
        in the artifact, unread)."""
        tokens, field_state = self._history_tokens(actor_input)
        return (
            self._outcome(actor_input, tokens, field_state),
            self.survival_head(tokens),
        )


def get_offline_critic(generation: int, rating_conditioning: bool = True) -> nn.Module:
    return Porygon2OfflineCritic(
        get_player_model_config(generation, train=False),
        rating_conditioning=rating_conditioning,
    )
