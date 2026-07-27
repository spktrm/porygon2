import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.interfaces import PlayerActorInput
from rl.model.config import get_player_model_config
from rl.model.encoder import Encoder
from rl.model.heads import CategoricalValueHeadOutput


class AntisymmetricOutcomeProbe(nn.Module):
    """Linear outcome probe with mirror-antisymmetry by construction.

    win_logit = w · (my − opp), loss_logit = −win_logit, tie a lone bias.
    Mirroring the perspective swaps the pooled summaries, negates the
    difference, and exactly swaps the win/loss logits — so
    Φ(mirror(s)) = −Φ(s) holds for every parameter setting, and SGD cannot
    satisfy the loss by memorizing game identity: only side-differenced
    structure reduces it. (Empirically necessary: with a free-form probe
    the critic memorizes pairs and generalizes at chance.)
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(self, diff: jax.Array) -> CategoricalValueHeadOutput:
        # diff: (T, num_latents * D) = flattened (my − opp) pooled latents.
        z = nn.Dense(1, use_bias=False, name="win_score")(diff)[..., 0]
        tie_bias = self.param("tie_bias", nn.initializers.zeros_init(), ())
        logits = jnp.stack(
            (-z, jnp.broadcast_to(tie_bias, z.shape).astype(z.dtype), z),
            axis=-1,
        )
        log_probs = nn.log_softmax(logits, axis=-1)
        probs = jnp.exp(log_probs)
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        values = self.cfg.v_head.category_values.astype(logits.dtype)
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
    """

    cfg: ConfigDict

    def setup(self):
        self.encoder = Encoder(self.cfg.encoder)
        self.outcome_head = AntisymmetricOutcomeProbe(self.cfg)

    def __call__(self, actor_input: PlayerActorInput) -> CategoricalValueHeadOutput:
        slot_states, field_state, node_states = self.encoder.encode_history(
            actor_input.env, actor_input.packed_history, actor_input.history
        )
        slot_sides = self.encoder.history_slot_sides(actor_input.packed_history)
        field_token = jnp.ones(1, dtype=jnp.bool_)
        my_mask = jnp.concatenate((slot_sides == 1, field_token))
        opp_mask = jnp.concatenate((slot_sides == 0, field_token))
        # Photos attend to diaries, trunk-style: each slot's current
        # snapshot makes a zero-init-gated cross-read of the recurrent
        # states, so at init the tokens ARE the raw snapshots (hand-rule
        # parity is the floor — GRU-only tokens lost current hp/faint info
        # to gating and were beaten by a raw hand rule late-game) and
        # history context blends in only where it helps.
        tokens = self.encoder.read_history_into_nodes(
            node_states, slot_states, field_state
        )
        my_latents = self.encoder.pool_history(tokens, field_state, my_mask)
        opp_latents = self.encoder.pool_history(tokens, field_state, opp_mask)
        diff = (my_latents - opp_latents).reshape(slot_states.shape[0], -1)
        return self.outcome_head(diff)


def get_offline_critic(generation: int) -> nn.Module:
    return Porygon2OfflineCritic(get_player_model_config(generation, train=False))