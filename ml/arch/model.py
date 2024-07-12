import math
import pprint
import chex
import jax
import flax.linen as nn

from ml.arch.config import get_model_cfg
from ml.arch.interfaces import ModuleConfigDict
from ml.arch.modules import ConfigurableModule
from ml.utils import Params

from rlenv.env import get_ex_step
from rlenv.interfaces import EnvStep


class Model(ConfigurableModule):
    def __call__(self, env_step: EnvStep):
        current_state, select_embeddings, move_embeddings = self.encoder(env_step)
        logit, pi, log_pi = self.policy_head(
            current_state, select_embeddings, move_embeddings, env_step.legal
        )
        v = self.value_head(current_state)
        return pi, v, log_pi, logit


def get_num_params(vars: Params):
    total = 0
    for _, value in vars.items():
        if isinstance(value, chex.Array):
            total += math.prod(value.shape)
        else:
            total += get_num_params(value)
    return total


def get_model(config: ModuleConfigDict) -> nn.Module:
    return Model(config)


def main():
    config = get_model_cfg()
    network = get_model(config)
    ex = get_ex_step()

    params = network.init(jax.random.key(0), ex)
    output = network.apply(params, ex)

    pprint.pprint(output)

    print("{:,}".format(get_num_params(params)))


if __name__ == "__main__":
    main()
