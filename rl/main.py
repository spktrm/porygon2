from dotenv import load_dotenv

load_dotenv()
import json
import threading
import time
import traceback
from pprint import pprint

import jax
import numpy as np
import wandb.wandb_run

import wandb
from rl.actor.actor import Actor
from rl.actor.agent import Agent
from rl.environment.env import SinglePlayerSyncEnvironment
from rl.learner.config import create_train_state, get_learner_config, load_train_state
from rl.learner.learner import Learner
from rl.model.builder_model import get_builder_model
from rl.model.config import get_builder_model_config, get_player_model_config
from rl.model.player_model import get_num_params, get_player_model


def run_training_actor(actor: Actor, stop_signal: list[bool]):
    """Runs an actor to produce trajectories, checking the ratio each time."""

    while not stop_signal[0]:
        try:
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

        time.sleep(2)


def main():
    """Main function to run the RL learner."""

    learner_config = get_learner_config()
    pprint(learner_config)

    learner_player_model_config = get_player_model_config(
        learner_config.generation, train=True
    )
    learner_builder_model_config = get_builder_model_config(
        learner_config.generation, train=True
    )
    actor_player_model_config = get_player_model_config(
        learner_config.generation, train=False
    )
    actor_builder_model_config = get_builder_model_config(
        learner_config.generation, train=False
    )

    learner_player_network = get_player_model(learner_player_model_config)
    learner_builder_network = get_builder_model(learner_builder_model_config)
    actor_player_network = get_player_model(actor_player_model_config)
    actor_builder_network = get_builder_model(actor_builder_model_config)

    actor_threads: list[threading.Thread] = []
    stop_signal = [False]

    player_state, builder_state = create_train_state(
        learner_player_network,
        learner_builder_network,
        jax.random.key(42),
        learner_config,
    )

    agent = Agent(actor_player_network.apply, actor_builder_network.apply)

    player_state, builder_state = load_train_state(
        learner_config, player_state, builder_state
    )

    wandb_run = wandb.init(
        project="pokemon-rl",
        config={
            "num_player_params": get_num_params(player_state.params),
            "num_builder_params": get_num_params(builder_state.params),
            "learner_config": learner_config,
            "player_model_config": json.loads(
                learner_player_model_config.to_json_best_effort()
            ),
            "builder_model_config": json.loads(
                learner_builder_model_config.to_json_best_effort()
            ),
        },
    )

    learner = Learner(
        player_state=player_state,
        builder_state=builder_state,
        learner_config=learner_config,
        wandb_run=wandb_run,
    )

    for game_id in range(learner_config.num_actors // 2):
        for player_id in range(2):
            actor = Actor(
                agent=agent,
                env=SinglePlayerSyncEnvironment(
                    f"train-{game_id:02d}{player_id:02d}",
                    generation=learner_config.generation,
                ),
                unroll_length=learner_config.unroll_length,
                learner=learner,
                rng_seed=len(actor_threads),
            )
            args = (actor, stop_signal)
            actor_threads.append(
                threading.Thread(
                    target=run_training_actor,
                    args=args,
                    name=f"Actor-{game_id}-{player_id}",
                )
            )

    for eval_id in range(learner_config.num_eval_actors):
        actor = Actor(
            agent=agent,
            env=SinglePlayerSyncEnvironment(
                f"eval-{eval_id:04d}", generation=learner_config.generation
            ),
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

    learner.train()

    stop_signal[0] = True
    for t in actor_threads:
        t.join()

    print("done")


if __name__ == "__main__":
    main()
