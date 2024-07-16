import jax

import numpy as np
import flax.linen as nn

from tqdm import tqdm

from ml.config import RNaDConfig
from rlenv.client import BatchCollector
from rlenv.data import EVALUATION_SOCKET_PATH, TRAINING_SOCKET_PATH
from rlenv.env import EnvStep, get_ex_step


class Model(nn.Module):
    @nn.compact
    def __call__(self, env_step: EnvStep):
        pi = env_step.legal / env_step.legal.sum(-1, keepdims=True)
        logit = log_pi = pi
        v = env_step.legal.sum()
        return pi, v, log_pi, logit


def main():
    pbar1 = tqdm(desc="num_batches")
    pbar2 = tqdm(desc="num_steps")
    pbar3 = tqdm(desc="num_games")

    network = Model()
    config = RNaDConfig()
    collector1 = BatchCollector(
        network,
        path=TRAINING_SOCKET_PATH,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    collector2 = BatchCollector(
        network,
        path=EVALUATION_SOCKET_PATH,
        batch_size=1,
        seed=config.seed,
    )

    ex = get_ex_step()
    params = network.init(jax.random.key(0), ex)
    network.apply(params, ex)

    for _ in range(100000):
        for collector in [collector1]:
            trajectory = collector.collect_batch_trajectory(params)

            turns_increasing = trajectory.env.turn[1:] >= trajectory.env.turn[:-1]

            if not np.all(turns_increasing):
                raise ValueError()

            pbar1.update(1)
            pbar2.update(trajectory.env.valid.sum())
            pbar3.update(config.batch_size)

    for collector in [collector1, collector2]:
        collector.game.close()
    print("done")


if __name__ == "__main__":
    main()
