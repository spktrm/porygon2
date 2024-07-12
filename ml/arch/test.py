import jax

from ml.arch.model import get_model
from ml.config import RNaDConfig
from rlenv.env import get_ex_step


def main():
    config = RNaDConfig()
    network = get_model(config)
    ex = get_ex_step()

    params = network.init(jax.random.key(0), ex)
    return network.apply(params, ex)


if __name__ == "__main__":
    main()
