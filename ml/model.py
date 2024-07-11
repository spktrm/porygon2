import flax.linen as nn

from ml.config import RNaDConfig
from rlenv.interfaces import EnvStep


class Model(nn.Module):
    @nn.compact
    def __call__(self, env_step: EnvStep):
        pi = env_step.legal / env_step.legal.sum(-1, keepdims=True)
        logit = log_pi = pi
        v = env_step.legal.sum(keepdims=True).astype(float)
        return pi, v, log_pi, logit


def get_model(config: RNaDConfig) -> nn.Module:
    return Model()
