import json
import math
from pprint import pprint
from typing import Iterator

import jax
import numpy as np
from flax.training import train_state
from tqdm import tqdm

import wandb
from ml.arch.config import get_model_cfg
from ml.arch.model import get_model, get_num_params
from ml.learners import vtrace as learner
from ml.learners.func import collect_batch_telemetry_data, collect_nn_telemetry_data
from ml.utils import get_most_recent_file
from rlenv.interfaces import TimeStep
from rlenv.main import EvalBatchCollector, SingleTrajectoryTrainingBatchCollector


def iterate(batch: TimeStep, minibatch_size: int = 4) -> Iterator[TimeStep]:
    _, batch_size, *__ = batch.env.valid.shape
    indices = np.arange(batch_size)
    np.random.shuffle(indices)
    for batch_index in range(math.ceil(batch_size / minibatch_size)):
        minibatch_indices = indices[
            batch_index * minibatch_size : (batch_index + 1) * minibatch_size
        ]
        yield jax.tree.map(lambda x: x[:, minibatch_indices], batch)


def evaluate(evaluation_collector: EvalBatchCollector, state: train_state.TrainState):
    eval_batch = evaluation_collector.collect_batch_trajectory(state.params)

    win_rewards = np.sign(
        (
            eval_batch.actor.rewards.win_rewards[..., 0]
            * eval_batch.env.valid.squeeze()
        ).sum(0)
    )

    fainted_rewards = (
        eval_batch.actor.rewards.fainted_rewards[..., 0]
        * eval_batch.env.valid.squeeze()
    ).sum(0)

    winrates = {f"wr{i}": wr for i, wr in enumerate(win_rewards)}
    winrates.update({f"hp{i}": f for i, f in enumerate(fainted_rewards)})

    return winrates


def main():
    learner_config = learner.get_config()
    model_config = get_model_cfg()
    pprint(learner_config)

    network = get_model(model_config)

    training_collector = SingleTrajectoryTrainingBatchCollector(
        network, learner_config.batch_size
    )
    evaluation_collector = EvalBatchCollector(network, 4)

    state = learner.create_train_state(network, jax.random.PRNGKey(42), learner_config)

    latest_ckpt = None  # get_most_recent_file("./ckpts")
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
    save_freq = 1000

    train_progress = tqdm(desc="training")

    for step_idx in range(learner_config.num_steps):
        logs: dict
        winrates = {}

        time_to_eval = state.step % (eval_freq // learner_config.num_eval_games) == 0
        if time_to_eval and learner_config.do_eval:
            winrates = evaluate(evaluation_collector, state)

        batch = training_collector.collect_batch_trajectory(state.params)
        state, logs = learner.train_step(state, batch, learner_config)

        logs.update(collect_nn_telemetry_data(state))
        logs.update(collect_batch_telemetry_data(batch))
        # logs.update(collect_action_prob_telemetry_data(minibatch))

        logs["step_idx"] = step_idx
        wandb.log({**logs, **winrates})
        train_progress.update(1)

        if state.step % save_freq == 0 and state.step > 0:
            learner.save(state)

    print("done")


if __name__ == "__main__":
    main()
