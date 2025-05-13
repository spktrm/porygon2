import chex
import flax.linen as nn
from ml_collections import ConfigDict

from ml.arch.modules import Logits, TransformerEncoder
from ml.func import legal_log_policy, legal_policy


class PolicyHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(
        self,
        action_embeddings: chex.Array,
        action_mask: chex.Array,
    ):
        action_embeddings = self.encoder(action_embeddings, action_mask)

        logits = self.logits(action_embeddings)
        logits = logits.reshape(-1)
        logits = logits - logits.mean(where=action_mask)

        policy = legal_policy(logits, action_mask)
        log_policy = legal_log_policy(logits, action_mask)

        return logits, policy, log_policy


class ValueHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(
        self,
        action_embeddings: chex.Array,
        action_mask: chex.Array,
    ):
        action_embeddings = self.encoder(action_embeddings, action_mask)
        actions_embedding = action_embeddings.mean(axis=0, where=action_mask[..., None])
        return self.logits(actions_embedding).reshape(-1)
