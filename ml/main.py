import wandb

from tqdm import trange

from ml.learner import Learner
from ml.config import RNaDConfig
from ml.arch.config import get_model_cfg
from ml.arch.model import get_model, get_num_params

from rlenv.client import BatchCollector


def main():
    learner_config = RNaDConfig()
    model_config = get_model_cfg()
    network = get_model(model_config)
    collector = BatchCollector(network, batch_size=learner_config.batch_size)
    learner = Learner(network, config=learner_config)

    wandb.init(
        project="pokemon-rl",
        config={
            "num_params": get_num_params(learner.params),
            "learner_config": learner_config,
            "model_config": model_config,
        },
    )

    for _ in trange(0, learner_config.num_steps):
        batch = collector.collect_batch_trajectory(learner.params)
        logs = learner.step(batch)
        wandb.log(logs)

    collector.game.close()
    print("done")


if __name__ == "__main__":
    main()
