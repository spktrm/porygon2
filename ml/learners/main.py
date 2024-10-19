import json
import os
import pickle
from pprint import pprint

import jax
from tqdm import trange

import wandb
from ml.arch.config import get_model_cfg
from ml.arch.model import get_model, get_num_params
from ml.config import VtraceConfig
from ml.learners import vtrace as learner
from ml.utils import Params, get_most_recent_file
from rlenv.client import BatchCollector
from rlenv.data import EVALUATION_SOCKET_PATH, TRAINING_SOCKET_PATH


def evaluate(params: Params, collector: BatchCollector, num_eval_games: int = 200):
    win_rewards, hp_rewards, fainted_rewards = 0, 0, 0

    for _ in trange(num_eval_games):
        batch = collector.collect_batch_trajectory(params)
        valid_mask = batch.env.valid

        win_rewards += (batch.actor.win_rewards[..., 0] * valid_mask).sum(0)
        hp_rewards += (batch.actor.hp_rewards[..., 0] * valid_mask).sum(0)
        fainted_rewards += (batch.actor.fainted_rewards[..., 0] * valid_mask).sum(0)

    winrates = {
        f"wr{i}": wr for i, wr in enumerate((win_rewards / num_eval_games).tolist())
    }
    hp_diff = {
        f"hp{i}": hp for i, hp in enumerate((hp_rewards / num_eval_games).tolist())
    }
    fainted_reward = {
        f"f{i}": f for i, f in enumerate((fainted_rewards / num_eval_games).tolist())
    }

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
    state = learner.create_train_state(network, jax.random.PRNGKey(42), learner_config)

    latest_ckpt = get_most_recent_file("./ckpts")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)

        state = state.replace(
            params=step["params"],
            params_target=step["params"],
        )

        if getattr(state, "params_prev", None):
            state = state.replace(
                params_prev=step["params"],
            )

        if getattr(state, "params_prev_", None):
            state = state.replace(
                params_prev_=step["params"],
            )

    wandb.init(
        project="pokemon-rl",
        config={
            "num_params": get_num_params(state.params),
            "learner_config": learner_config,
            "model_config": json.loads(model_config.to_json_best_effort()),
        },
    )

    eval_freq = 5000
    save_freq = 1000

    for _ in trange(0, learner_config.num_steps):
        if state.learner_steps % save_freq == 0 and state.learner_steps > 0:
            with open(
                os.path.abspath(f"ckpts/ckpt_{state.learner_steps:08}"), "wb"
            ) as f:
                pickle.dump(
                    dict(
                        params=state.params,
                        params_target=state.params_target,
                    ),
                    f,
                )

        if state.learner_steps % eval_freq == 0 and learner_config.do_eval:
            evaluate(state.params, evaluation_collector)

        batch = training_collector.collect_batch_trajectory(state.params)

        state, logs = learner.train_step(state, batch, learner_config)

        wandb.log(logs)

    training_collector.game.close()
    print("done")


if __name__ == "__main__":
    main()
