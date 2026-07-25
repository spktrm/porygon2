import flax.linen as nn
import jax
from ml_collections import ConfigDict

from rl.environment.interfaces import PlayerActorInput
from rl.model.config import get_player_model_config
from rl.model.encoder import Encoder
from rl.model.heads import CategoricalValueHeadOutput, CategoricalValueLogitHead


class Porygon2OfflineCritic(nn.Module):
    """Linear outcome probe over the recurrent history pathway only.

    The critic reads nothing but the public battle-event stream:
    Encoder.encode_history (packed public entity/edge caches + field
    history + request counts) feeding the PerSlotHistoryEncoder, then
    Encoder.pool_history attention-pools the slot/field states into
    num_latents learned latents whose flattened concat feeds a LINEAR
    3-way outcome probe. The probe is deliberately capacity-free: all
    learning lands in the history encoder + attention pool, which the RL
    trunk shares (it reads the same latents as history-context tokens),
    so offline-trained capacity is exactly what warm-starts RL. Private
    team, own moveset, action masks, and the rest of the trunk never
    enter the computation, so:

    - public-only is guaranteed by construction (no projection needed),
    - replay-exported inputs and live inputs are the SAME distribution
      (history caches are built from the same protocol events), so the
      frozen Φ carries no train/serve bias into RL training.

    The "encoder" attribute matches Porygon2PlayerModel, so the trained
    history-pathway subtrees (history_encoder, history_pool, public
    entity/edge/field embedders) merge into an RL init via
    load_from_params; the offline-only outcome_head has no RL counterpart
    and is dropped by the merge.
    """

    cfg: ConfigDict

    def setup(self):
        self.encoder = Encoder(self.cfg.encoder)
        self.outcome_head = CategoricalValueLogitHead(self.cfg.v_head)

    def __call__(self, actor_input: PlayerActorInput) -> CategoricalValueHeadOutput:
        slot_states, field_state = self.encoder.encode_history(
            actor_input.env, actor_input.packed_history, actor_input.history
        )
        latents = self.encoder.pool_history(slot_states, field_state)
        # (T, num_latents, D) -> (T, num_latents * D) for the linear probe.
        features = latents.reshape(latents.shape[0], -1)
        return jax.vmap(self.outcome_head)(features)


def get_offline_critic(generation: int) -> nn.Module:
    return Porygon2OfflineCritic(get_player_model_config(generation, train=False))
