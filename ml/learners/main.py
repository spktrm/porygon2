import json
from pprint import pprint

import jax
import numpy as np
from tqdm import trange

import wandb
from ml.arch.config import get_model_cfg
from ml.arch.model import get_model, get_num_params
from ml.learners import vtrace as learner
from ml.learners.func import collect_batch_telemetry_data, collect_nn_telemetry_data
from ml.utils import get_most_recent_file
from rlenv.main import EvalBatchCollector, SingleTrajectoryTrainingBatchCollector


def main():
    learner_config = learner.get_config()
    model_config = get_model_cfg()
    pprint(learner_config)

    network = get_model(model_config)

    training_collector = SingleTrajectoryTrainingBatchCollector(
        network, learner_config.minibatch_size
    )
    evaluation_collector = EvalBatchCollector(network, 4)

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
    save_freq = 1000

    # buffer = ReplayBuffer()
    # buffer_fill = tqdm(desc="Filling Buffer...", total=buffer.max_buffer_size)
    # while len(buffer) < 50:
    #     trajectories = training_collector.collect_batch_trajectory(state.params)
    #     buffer.add_trajectories(trajectories)
    #     buffer_fill.update(training_collector.batch_size)

    for step_idx in trange(0, learner_config.num_steps, desc="training"):
        if state.learner_steps % save_freq == 0 and state.learner_steps > 0:
            learner.save(state)

        winrates = {}
        if (
            state.learner_steps % (eval_freq // learner_config.num_eval_games) == 0
            and learner_config.do_eval
        ):
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

        batch = training_collector.collect_batch_trajectory(state.params)
        # buffer.add_trajectories(trajectories)

        # batch = buffer.sample(learner_config.minibatch_size)

        logs: dict
        state, logs = learner.train_step(state, batch, learner_config)

        logs.update(collect_nn_telemetry_data(state))
        logs.update(collect_batch_telemetry_data(batch))

        logs["step_idx"] = step_idx
        wandb.log({**logs, **winrates})

    print("done")


if __name__ == "__main__":
    main()
