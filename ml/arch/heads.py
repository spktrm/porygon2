import chex
import flax.linen as nn
import jax
from ml_collections import ConfigDict

from ml.arch.modules import (
    Logits,
    SequenceToVector,
    TransformerDecoder,
    TransformerEncoder,
)
from ml.func import legal_log_policy, legal_policy
from rlenv.interfaces import EnvStep


class PolicyHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(self, action_embeddings: chex.Array, legal_action_mask: EnvStep):
        action_embeddings = self.encoder(action_embeddings, legal_action_mask)

        logits = jax.vmap(self.logits)(action_embeddings)
        logits = logits.reshape(-1)
        logits = logits - logits.mean(where=legal_action_mask)

        policy = legal_policy(logits, legal_action_mask)
        log_policy = legal_log_policy(logits, legal_action_mask)

        return logits, policy, log_policy


class ValueHead(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def __call__(self, entity_embeddings: chex.Array, valid_entity_mask: chex.Array):
        state_value = SequenceToVector(self.cfg.seq2vec)(
            entity_embeddings, valid_entity_mask
        )
        return state_value.reshape(-1)
