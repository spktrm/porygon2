from dotenv import load_dotenv

load_dotenv()
import concurrent.futures
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
from rl.model.heads import HeadParams
from rl.model.player_model import get_num_params, get_player_model


def run_training_actor_pair(
    player: Actor,
    opponent: Actor,
    executor: concurrent.futures.ThreadPoolExecutor,
    stop_signal: list[bool],
):
    """Runs an actor to produce trajectories"""

    while not stop_signal[0]:
        try:
            player_params = player.pull_main_player()
            opponent_params, is_trainable = player.get_match()

            player_ckpt = np.array(player_params.step_count).item()
            opponent_ckpt = np.array(opponent_params.step_count).item()

            player.set_current_ckpt(player_ckpt)
            player.set_opponent_ckpt(opponent_ckpt)

            opponent.set_current_ckpt(opponent_ckpt)
            opponent.set_opponent_ckpt(player_ckpt)

            # Grab the result from either self play or playing historical opponents
            future1 = executor.submit(player.unroll_and_push, player_params)

            # Will only push if is_trainable is True
            future2 = executor.submit(
                opponent.unroll_and_push, opponent_params, is_trainable
            )
            trajectory = future1.result()
            future2.result()

            if not is_trainable:
                player.update_player_league_stats(
                    player_params, opponent_params, trajectory
                )
        except Exception as e:
            traceback.print_exc()
            raise e


def run_eval_heuristic(
    actor: Actor,
    executor: concurrent.futures.ThreadPoolExecutor,
    stop_signal: list[bool],
    wandb_run: wandb.wandb_run.Run,
):
    """Runs an actor to produce num_trajectories trajectories."""
    learner = actor._learner

    step_count = np.array(learner.player_state.step_count).item()

    session_id = actor._player_env.username

    while not stop_signal[0]:
        try:
            new_step_count = np.array(learner.player_state.step_count).item()
            if new_step_count > step_count:
                step_count = new_step_count

                player = actor.pull_main_player()

                future1 = executor.submit(actor.unroll_and_push, player)
                eval_trajectory = future1.result()

                payoff = eval_trajectory.player_transitions.env_output.win_reward[-1]

                wandb_run.log({"training_step": step_count, f"wr-{session_id}": payoff})

        except Exception:
            traceback.print_exc()
            # Dont let bad evaluation crash the whole training loop
            continue

        time.sleep(5)


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

    gpu_lock = threading.Lock()
    learning_agent = Agent(
        actor_player_network.apply,
        actor_builder_network.apply,
        gpu_lock=gpu_lock,
    )
    eval_agent = Agent(
        actor_player_network.apply,
        actor_builder_network.apply,
        gpu_lock=gpu_lock,
        player_head_params=HeadParams(temp=0.8, min_p=0.1),
        builder_head_params=HeadParams(temp=0.8, min_p=0.1),
    )

    player_state, builder_state, league = load_train_state(
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
        config=learner_config,
        league=league,
        wandb_run=wandb_run,
        gpu_lock=gpu_lock,
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=(learner_config.num_actors // 2 + learner_config.num_eval_actors)
    ) as executor:

        for game_id in range(learner_config.num_actors // 2):
            actors = [
                Actor(
                    agent=learning_agent,
                    env=SinglePlayerSyncEnvironment(
                        f"train:p{player_id}g{game_id:02d}",
                        generation=learner_config.generation,
                    ),
                    unroll_length=learner_config.unroll_length,
                    learner=learner,
                    rng_seed=len(actor_threads),
                )
                for player_id in range(2)
            ]
            args = (*actors, executor, stop_signal)
            actor_threads.append(
                threading.Thread(
                    target=run_training_actor_pair,
                    args=args,
                    name=f"Selfplay-{game_id}",
                )
            )

        for eval_id in range(learner_config.num_eval_actors):
            actor = Actor(
                agent=eval_agent,
                env=SinglePlayerSyncEnvironment(
                    f"eval-heuristic:{eval_id:04d}",
                    generation=learner_config.generation,
                ),
                unroll_length=learner_config.unroll_length,
                learner=learner,
                rng_seed=len(actor_threads),
            )
            args = (actor, executor, stop_signal, wandb_run)
            actor_threads.append(
                threading.Thread(
                    target=run_eval_heuristic, args=args, name=f"EvalActor-{eval_id}"
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
