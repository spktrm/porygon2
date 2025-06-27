import functools
import pickle

import jax

from ml.arch.config import get_model_cfg
from ml.arch.model import get_model
from rlenv.env import get_ex_step


def main():
    config = get_model_cfg()
    network = get_model(config)
    ex, hx = get_ex_step()

    latest_ckpt = None  # get_most_recent_file("./ckpts")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        params = step["params"]
    else:
        key = jax.random.key(42)
        params = network.init(key, ex, hx)

    f = functools.partial(network.apply, params)

    z = jax.jit(f).lower(ex, hx)

    print(z.compile().cost_analysis())


if __name__ == "__main__":
    main()
