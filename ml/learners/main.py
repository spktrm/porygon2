import json
from pprint import pprint

import jax
import numpy as np
from tqdm import trange

import wandb
from ml.arch.config import get_model_cfg
from ml.arch.model import get_model, get_num_params
from ml.learners import rnad as learner
from ml.utils import get_most_recent_file
from rlenvv2.main import BatchCollectorV2, BatchSinglePlayerEnvironment


def main():
    learner_config = learner.get_config()
    model_config = get_model_cfg()
    pprint(learner_config)

    network = get_model(model_config)

    training_collector = BatchCollectorV2(network, learner_config.batch_size)
    evaluation_collector = BatchCollectorV2(network, 4, BatchSinglePlayerEnvironment)

    state = learner.create_train_state(network, jax.random.PRNGKey(42), learner_config)

    latest_ckpt = get_most_recent_file("./ckpts")
    if latest_ckpt:
        state = learner.load(state, latest_ckpt)

    wandb.init(
        project="pokemon-rl",
        config={
            "num_params": get_num_params(state.params),
            "learner_config": learner_config,
            "model_config": json.loads(model_config.to_json_best_effort()),
        },
    )

    eval_freq = 5000
    save_freq = 10000
    tau = 0.1

    # initialise average returns
    avg_reward = np.zeros(evaluation_collector.game.num_envs)
    if learner_config.do_eval:
        num_inits = 50
        for _ in trange(0, num_inits, desc="init winrates..."):
            batch = evaluation_collector.collect_batch_trajectory(state.params)
            avg_reward += np.sign(
                batch.actor.win_rewards[..., 0] * batch.env.valid
            ).sum(0)
        avg_reward /= num_inits

    for step_idx in trange(0, learner_config.num_steps, desc="training"):
        if state.learner_steps % save_freq == 0 and state.learner_steps > 0:
            learner.save(state)

        winrates = {}
        if (
            state.learner_steps % (eval_freq // learner_config.num_eval_games) == 0
            and learner_config.do_eval
        ):
            batch = evaluation_collector.collect_batch_trajectory(state.params)
            win_rewards = np.sign(
                batch.actor.win_rewards[..., 0] * batch.env.valid
            ).sum(0)
            avg_reward = avg_reward * (1 - tau) + win_rewards * tau
            winrates = {f"wr{i}": wr for i, wr in enumerate(avg_reward)}

        batch = training_collector.collect_batch_trajectory(state.params)

        state, logs = learner.train_step(state, batch, learner_config)

        logs["step_idx"] = step_idx
        wandb.log({**logs, **winrates})

    print("done")


if __name__ == "__main__":
    main()