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

    Pathway: Encoder.encode_history -> PerSlotHistoryEncoder ->
    Encoder.pool_history twice with side masks (shared pool params): once
    over my-side slots + field, once over opponent-side slots + field. The
    flattened difference of the two latent banks feeds
    AntisymmetricOutcomeProbe. All learnable capacity sits in the shared
    history pathway (encoder + pool), which the RL trunk reuses and
    warm-starts; the probe is a single weight vector.

    Public-only by construction: the pathway reads packed public
    entity/edge caches, field history, and request counts — private team,
    own moveset, and action masks are architecturally unreachable, and
    replay-exported inputs match live inputs exactly.

    The "encoder" attribute matches Porygon2PlayerModel, so trained
    history-pathway subtrees (history_encoder, history_pool, public
    embedders) merge into an RL init via load_from_params; the
    offline-only probe has no RL counterpart and is dropped by the merge.
    """

    cfg: ConfigDict

    def setup(self):
        self.encoder = Encoder(self.cfg.encoder)
        self.outcome_head = AntisymmetricOutcomeProbe(self.cfg)

    def __call__(
        self, actor_input: PlayerActorInput
    ) -> tuple[CategoricalValueHeadOutput, jax.Array]:
        slot_states, field_state, aux_state_loss = self.encoder.encode_history(
            actor_input.env, actor_input.packed_history, actor_input.history
        )
        slot_sides = self.encoder.history_slot_sides(actor_input.packed_history)
        field_token = jnp.ones(1, dtype=jnp.bool_)
        my_mask = jnp.concatenate((slot_sides == 1, field_token))
        opp_mask = jnp.concatenate((slot_sides == 0, field_token))
        my_latents = self.encoder.pool_history(slot_states, field_state, my_mask)
        opp_latents = self.encoder.pool_history(slot_states, field_state, opp_mask)
        diff = (my_latents - opp_latents).reshape(slot_states.shape[0], -1)
        return self.outcome_head(diff), aux_state_loss


def get_offline_critic(generation: int) -> nn.Module:
    return Porygon2OfflineCritic(get_player_model_config(generation, train=False))
