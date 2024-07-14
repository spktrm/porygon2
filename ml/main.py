import json
import wandb

from tqdm import trange

from ml.learner import Learner
from ml.config import RNaDConfig
from ml.arch.config import get_model_cfg
from ml.arch.model import get_model, get_num_params
from ml.utils import Params

from rlenv.client import BatchCollector
from rlenv.data import EVALUATION_SOCKET_PATH, TRAINING_SOCKET_PATH


def evaluate(params: Params, collector: BatchCollector, num_eval_games: int = 200):
    rewards_sum = 0
    for _ in trange(0, num_eval_games):
        batch = collector.collect_batch_trajectory(params)
        rewards_sum += (batch.actor.rewards[..., 0] * batch.env.valid).sum(0)
    winrates = (rewards_sum / num_eval_games + 1) / 2
    wandb.log({f"wr{i}": wr for i, wr in enumerate(winrates.tolist())})


def main():
    learner_config = RNaDConfig()
    model_config = get_model_cfg()
    network = get_model(model_config)

    training_collector = BatchCollector(
        network, TRAINING_SOCKET_PATH, batch_size=learner_config.batch_size
    )
    evaluation_collector = BatchCollector(network, EVALUATION_SOCKET_PATH, batch_size=2)
    learner = Learner(network, config=learner_config)

    wandb.init(
        project="pokemon-rl",
        config={
            "num_params": get_num_params(learner.params),
            "learner_config": learner_config,
            "model_config": json.loads(model_config.to_json_best_effort()),
        },
    )

    eval_freq = 1000
    save_freq = 1000

    for _ in trange(0, learner_config.num_steps):
        if learner.learner_steps % save_freq == 0:
            learner.save()

        if learner.learner_steps % eval_freq == 0:
            evaluate(learner.params, evaluation_collector)

        batch = training_collector.collect_batch_trajectory(learner.params)
        logs = learner.step(batch)

        wandb.log(logs)

    training_collector.game.close()
    print("done")


if __name__ == "__main__":
    main()