import os

from rl.model.utils import get_most_recent_file

# Can cause memory issues if set to True
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# gemm_any=True is ~10% speed up in this setup
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True " "--xla_gpu_enable_latency_hiding_scheduler=true "
)

import json
import queue
import threading
import traceback
from pprint import pprint

import jax
import numpy as np
import wandb.wandb_run

import wandb
from rl.actor.actor import Actor
from rl.actor.agent import Agent
from rl.concurrency.lock import FairLock
from rl.environment.env import SinglePlayerSyncEnvironment
from rl.environment.interfaces import Trajectory
from rl.learner.buffer import ReplayBuffer, ReplayRatioController
from rl.learner.config import create_train_state, get_learner_config, load_train_state
from rl.learner.learner import Learner
from rl.model.builder_model import get_builder_model
from rl.model.config import get_model_config
from rl.model.player_model import get_num_params, get_player_model
from rl.utils import init_jax_jit_cache


def run_training_actor(
    actor: Actor, stop_signal: list[bool], controller: ReplayRatioController
):
    """Runs an actor to produce trajectories, checking the ratio each time."""

    while not stop_signal[0]:
        try:
            controller.actor_wait()
            step_count, player_params, builder_params = actor.pull_params()
            actor.unroll_and_push(step_count, player_params, builder_params)
        except Exception as e:
            traceback.print_exc()
            raise e


def run_eval_actor(
    actor: Actor, wandb_run: wandb.wandb_run.Run, stop_signal: list[bool]
):
    """Runs an actor to produce num_trajectories trajectories."""

    old_step_count, player_params, builder_params = actor.pull_params()
    session_id = actor._player_env.username
    win_reward_sum = {old_step_count: (0, 0)}

    while not stop_signal[0]:
        try:
            step_count, player_params, builder_params = actor.pull_params()
            assert step_count >= old_step_count, (
                f"Actor {session_id} tried to pull params with frame count "
                f"{step_count} but expected at least {old_step_count}."
            )
            if step_count not in win_reward_sum:
                win_reward_sum[step_count] = (0, 0)

            if step_count > old_step_count:
                reward_count, reward_sum = win_reward_sum.pop(old_step_count)
                wandb_run.log(
                    {
                        "Step": old_step_count,
                        f"wr-{session_id}": reward_sum / max(1, reward_count),
                    }
                )
                old_step_count = step_count

            player_params = jax.device_put(player_params)
            builder_params = jax.device_put(builder_params)
            subkey = actor.split_rng()
            eval_trajectory = actor.unroll(
                subkey, step_count, player_params, builder_params
            )

            win_rewards = np.sign(
                eval_trajectory.player_transitions.env_output.win_reward[-1]
            )
            # Update the win reward sum for this step count.
            reward_count, reward_sum = win_reward_sum[step_count]
            win_reward_sum[step_count] = (reward_count + 1, reward_sum + win_rewards)

            # Log the win reward mean for this step count if we have enough data.
        except Exception:
            traceback.print_exc()
            continue


def host_to_device_worker(
    trajectory_queue: queue.Queue[Trajectory],
    stop_signal: list[bool],
    replay_buffer: ReplayBuffer,
    controller: ReplayRatioController,
):
    """Elementary data pipeline."""
    while not stop_signal[0]:
        try:
            trajectory = trajectory_queue.get(timeout=10)
        except queue.Empty:
            continue

        replay_buffer.add(trajectory)

        controller.signal_learner()


def main():
    """Main function to run the RL learner."""
    init_jax_jit_cache()

    learner_config = get_learner_config()
    model_config = get_model_config()
    pprint(learner_config)

    player_network = get_player_model(model_config)
    builder_network = get_builder_model(model_config)

    actor_threads: list[threading.Thread] = []
    stop_signal = [False]
    num_samples = [0]

    num_eval_actors = 5
    trajectory_queue: queue.Queue[Trajectory] = queue.Queue(
        maxsize=2 * learner_config.num_actors
    )

    player_state, builder_state = create_train_state(
        player_network, builder_network, jax.random.key(42), learner_config
    )

    gpu_lock = FairLock()  # threading.Lock()
    agent = Agent(player_state.apply_fn, builder_state.apply_fn, gpu_lock)

    replay_buffer = ReplayBuffer(
        capacity=max(
            learner_config.replay_buffer_capacity, learner_config.batch_size * 2
        )
    )

    latest_ckpt = get_most_recent_file("./ckpts")
    if latest_ckpt:
        player_state, builder_state = load_train_state(
            player_state, builder_state, latest_ckpt
        )

    controller = ReplayRatioController(
        replay_buffer, lambda: num_samples[0], learner_config
    )

    wandb_run = wandb.init(
        project="pokemon-rl",
        config={
            "num_params": get_num_params(player_state.params),
            "learner_config": learner_config,
            "model_config": json.loads(model_config.to_json_best_effort()),
        },
    )

    learner = Learner(
        player_state=player_state,
        builder_state=builder_state,
        learner_config=learner_config,
        replay_buffer=replay_buffer,
        controller=controller,
        wandb_run=wandb_run,
        gpu_lock=gpu_lock,
        num_samples=num_samples,
    )

    for game_id in range(learner_config.num_actors // 2):
        for player_id in range(2):
            actor = Actor(
                agent=agent,
                env=SinglePlayerSyncEnvironment(f"train-{game_id:02d}{player_id:02d}"),
                unroll_length=learner_config.unroll_length,
                queue=trajectory_queue,
                learner=learner,
                rng_seed=len(actor_threads),
            )
            args = (actor, stop_signal, controller)
            actor_threads.append(
                threading.Thread(
                    target=run_training_actor,
                    args=args,
                    name=f"Actor-{game_id}-{player_id}",
                )
            )

    for eval_id in range(num_eval_actors):
        actor = Actor(
            agent=agent,
            env=SinglePlayerSyncEnvironment(f"eval-{eval_id:04d}"),
            unroll_length=learner_config.unroll_length,
            learner=learner,
            rng_seed=len(actor_threads),
        )
        args = (actor, wandb_run, stop_signal)
        actor_threads.append(
            threading.Thread(
                target=run_eval_actor, args=args, name=f"EvalActor-{eval_id}"
            )
        )

    # Start the actors and learner.
    for t in actor_threads:
        t.start()

    transfer_thread = threading.Thread(
        target=host_to_device_worker,
        args=(trajectory_queue, stop_signal, replay_buffer, controller),
    )
    transfer_thread.start()

    learner.train()

    stop_signal[0] = True
    for t in actor_threads:
        t.join()

    print("done")


if __name__ == "__main__":
    main()
