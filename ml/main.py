import pprint

from tqdm import trange

from ml.learner import Learner
from ml.config import RNaDConfig
from ml.model import get_model

from rlenv.client import BatchCollector


def main():

    config = RNaDConfig()
    network = get_model(config)
    collector = BatchCollector(network, batch_size=config.batch_size)
    learner = Learner(network, config=config)

    for _ in trange(0, config.num_steps):
        batch = collector.collect_batch_trajectory(learner.params)
        logs = learner.step(batch)
        pprint.pprint(logs)

    collector.game.close()
    print("done")


if __name__ == "__main__":
    main()
