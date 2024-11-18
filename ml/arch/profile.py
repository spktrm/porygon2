import pickle

import jax

from ml.arch.config import get_model_cfg
from ml.arch.model import get_model
from ml.utils import get_most_recent_file
from rlenv.env import get_ex_step


def block_all(xs):
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), xs)
    return xs


def main():
    config = get_model_cfg()
    network = get_model(config)
    ex_step = get_ex_step()

    latest_ckpt = get_most_recent_file("./ckpts")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        params = step["params"]
    else:
        key = jax.random.key(42)
        params = network.init(key, ex_step)

    apply_fn = network.apply
    output = apply_fn(params, ex_step)

    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        # Run the operations to be profiled
        output = apply_fn(params, ex_step)
        block_all(output)


if __name__ == "__main__":
    main()