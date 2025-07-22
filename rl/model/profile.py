import pickle

import jax

from rl.environment.utils import get_ex_step
from rl.model.model import get_model
from rl.model.utils import get_most_recent_file


def block_all(xs):
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), xs)
    return xs


def main():
    network = get_model()
    ts = jax.tree.map(lambda x: x[:, 0], get_ex_step())

    latest_ckpt = get_most_recent_file("./ckpts")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        params = step["params"]
    else:
        key = jax.random.key(42)
        params = network.init(key, ts)

    @jax.jit
    def call_network(ts):
        return network.apply(params, ts)

    compiled = call_network.lower(ts).compile()
    flops = compiled.cost_analysis()["flops"]

    print(flops)


if __name__ == "__main__":
    main()
