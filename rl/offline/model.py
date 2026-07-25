import flax.linen as nn
import jax
from ml_collections import ConfigDict

from rl.environment.interfaces import PlayerActorInput
from rl.model.config import get_player_model_config
from rl.model.encoder import Encoder
from rl.model.heads import CategoricalValueHeadOutput, CategoricalValueLogitHead


class Porygon2OfflineCritic(nn.Module):
    """Encoder + categorical value head of the player model, without the
    policy heads.

    Submodule attribute names deliberately match Porygon2PlayerModel
    ("encoder", "v_head") so the trained param subtrees merge directly into
    a fresh RL init via rl.learner.config.load_from_params/merge_params;
    pi_head/macro_head simply keep their fresh initialization.
    """

    cfg: ConfigDict

    def setup(self):
        self.encoder = Encoder(self.cfg.encoder)
        self.v_head = CategoricalValueLogitHead(self.cfg.v_head)

    def __call__(self, actor_input: PlayerActorInput) -> CategoricalValueHeadOutput:
        _, value_embeddings = self.encoder(
            actor_input.env, actor_input.packed_history, actor_input.history
        )
        # value_embeddings: (T, 4 * entity_size) — map the head over time,
        # mirroring Porygon2PlayerModel.__call__.
        return jax.vmap(self.v_head)(value_embeddings)


def get_offline_critic(generation: int) -> nn.Module:
    return Porygon2OfflineCritic(get_player_model_config(generation, train=False))
