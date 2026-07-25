import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.interfaces import PlayerActorInput
from rl.model.config import get_player_model_config
from rl.model.encoder import Encoder
from rl.model.heads import CategoricalValueHeadOutput, CategoricalValueLogitHead


class Porygon2OfflineCritic(nn.Module):
    """Outcome classifier over the recurrent history pathway only.

    The critic reads nothing but the public battle-event stream:
    Encoder.encode_history (packed public entity/edge caches + field
    history + request counts) feeding the PerSlotHistoryEncoder, whose
    per-request slot/field states are pooled into a 3-way outcome
    classifier. Private team, own moveset, action masks, and the rest of
    the trunk never enter the computation, so:

    - public-only is guaranteed by construction (no projection needed),
    - replay-exported inputs and live inputs are the SAME distribution
      (history caches are built from the same protocol events), so the
      frozen Φ carries no train/serve bias into RL training.

    The "encoder" attribute matches Porygon2PlayerModel, so the trained
    history-pathway subtrees (history_encoder + public entity/edge/field
    embedders) merge into an RL init via load_from_params; the
    offline-only outcome_head has no RL counterpart and is dropped by the
    merge.
    """

    cfg: ConfigDict

    def setup(self):
        self.encoder = Encoder(self.cfg.encoder)
        self.outcome_head = CategoricalValueLogitHead(self.cfg.v_head)

    def __call__(self, actor_input: PlayerActorInput) -> CategoricalValueHeadOutput:
        slot_states, field_state = self.encoder.encode_history(
            actor_input.env, actor_input.packed_history, actor_input.history
        )
        # (T, NUM_PUBLIC_SLOTS, D) + (T, D) -> (T, 2D): battle gestalt over
        # the slot bank plus the persistent field memory.
        features = jnp.concatenate((slot_states.mean(axis=-2), field_state), axis=-1)
        return jax.vmap(self.outcome_head)(features)


def get_offline_critic(generation: int) -> nn.Module:
    return Porygon2OfflineCritic(get_player_model_config(generation, train=False))
