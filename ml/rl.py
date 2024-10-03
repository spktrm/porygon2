import json
import pickle
from pprint import pprint

import wandb

wandb.require("core")

from tqdm import trange

from ml.arch.config import get_model_cfg
from ml.arch.model import get_model, get_num_params
from ml.config import VtraceConfig
from ml.learner import Learner
from ml.utils import Params, get_most_recent_file
from rlenv.client import BatchCollector
from rlenv.data import EVALUATION_SOCKET_PATH, TRAINING_SOCKET_PATH


def evaluate(params: Params, collector: BatchCollector, num_eval_games: int = 200):
    win_rewards_sum = 0
    hp_rewards_sum = 0
    fainted_rewards_sum = 0
    for _ in trange(0, num_eval_games):
        batch = collector.collect_batch_trajectory(params)
        win_rewards_sum += (batch.actor.win_rewards[..., 0] * batch.env.valid).sum(0)
        hp_rewards_sum += (batch.actor.hp_rewards[..., 0] * batch.env.valid).sum(0)
        fainted_rewards_sum += (
            batch.actor.fainted_rewards[..., 0] * batch.env.valid
        ).sum(0)
    # winrates = (rewards_sum / num_eval_games + 1) / 2
    winrates = win_rewards_sum / num_eval_games
    hp_diff = hp_rewards_sum / num_eval_games
    fainted_reward = fainted_rewards_sum / num_eval_games
    winrates = {f"wr{i}": wr for i, wr in enumerate(winrates.tolist())}
    hp_diff = {f"hp{i}": wr for i, wr in enumerate(hp_diff.tolist())}
    fainted_reward = {f"f{i}": wr for i, wr in enumerate(fainted_reward.tolist())}
    wandb.log({**winrates, **hp_diff, **fainted_reward})


def main():
    learner_config = VtraceConfig()
    model_config = get_model_cfg()
    pprint(learner_config)

    network = get_model(model_config)

    training_collector = BatchCollector(
        network, TRAINING_SOCKET_PATH, batch_size=learner_config.batch_size
    )
    evaluation_collector = BatchCollector(
        network, EVALUATION_SOCKET_PATH, batch_size=13
    )
    learner = Learner(network, config=learner_config)

    latest_ckpt = get_most_recent_file("./ckpts")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)

        # for key, value in step.items():
        #     setattr(learner, key, value)

        learner.params = step["params"]
        learner.params_target = step["params"]
        learner.params_prev = step["params"]
        learner.params_prev_ = step["params"]

        # for key, value in step.items():
        #     setattr(learner, key, value)

    wandb.init(
        project="pokemon-rl",
        config={
            "num_params": get_num_params(learner.params),
            "learner_config": learner_config,
            "model_config": json.loads(model_config.to_json_best_effort()),
        },
    )

    eval_freq = 5000
    save_freq = 1000

    for _ in trange(0, learner_config.num_steps):
        if learner.learner_steps % save_freq == 0 and learner.learner_steps > 0:
            learner.save()

        if learner.learner_steps % eval_freq == 0 and learner_config.do_eval:
            evaluate(learner.params, evaluation_collector)

        batch = training_collector.collect_batch_trajectory(learner.params)
        logs = learner.step(batch)

        wandb.log(logs)

    training_collector.game.close()
    print("done")


if __name__ == "__main__":
    main()