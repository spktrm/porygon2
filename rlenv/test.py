import numpy as np

from tqdm import tqdm

from ml.config import RNaDConfig
from rlenv.client import BatchCollector


def main():
    pbar1 = tqdm(desc="num_batches")
    pbar2 = tqdm(desc="num_steps")
    pbar3 = tqdm(desc="num_games")

    collector = BatchCollector(config=RNaDConfig())

    for _ in range(100000):
        trajectory = collector.collect_batch_trajectory()

        turns_increasing = trajectory.env.turn[1:] >= trajectory.env.turn[:-1]

        if not np.all(turns_increasing):
            raise ValueError()

        pbar1.update(1)
        pbar2.update(trajectory.env.valid.sum())
        pbar3.update(collector.config.batch_size)

    collector.game.close()
    print("done")


if __name__ == "__main__":
    main()
