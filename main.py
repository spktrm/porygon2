from tqdm import tqdm

from ml.config import RNaDConfig
from rlenv.client import BatchCollector


def main():
    collector = BatchCollector(config=RNaDConfig())

    for _ in range(100000):
        trajectory = collector.collect_batch_trajectory()

    collector.game.close()


if __name__ == "__main__":
    main()
