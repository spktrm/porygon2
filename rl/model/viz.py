import functools

import cloudpickle as pickle
import jax

from rl.environment.utils import get_ex_player_step
from rl.model.config import get_player_model_config
from rl.model.player_model import get_player_model


def main():
    config = get_player_model_config()
    network = get_player_model(config)
    ex, hx = get_ex_player_step()

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
