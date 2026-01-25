import pickle

from dotenv import load_dotenv

from rl.model.utils import ParamsContainer

load_dotenv()

import concurrent.futures
import json
import logging
import threading
import time
from typing import List

import numpy as np

import wandb
from rl.actor.actor import Actor
from rl.actor.agent import Agent
from rl.environment.env import SinglePlayerSyncEnvironment
from rl.learner.config import get_learner_config
from rl.learner.learner import LearnerManager
from rl.model.builder_model import get_builder_model
from rl.model.config import get_builder_model_config, get_player_model_config
from rl.model.heads import HeadParams
from rl.model.player_model import get_player_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def run_training_worker(actor1: Actor, actor2: Actor, stop_event: threading.Event):
    """
    Runs a training loop for a pair of actors.
    Uses a private, local ThreadPool to handle the synchronous game execution.
    """
    worker_id = f"{actor1._player_env.username}"
    logger.info(f"Worker {worker_id} started.")

    league = actor1._learner_manager.league

    # We create a persistent pool of 2 threads specifically for this game pair.
    # This avoids the overhead of creating/destroying threads every loop.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as game_executor:

        while not stop_event.is_set():
            try:
                # 1. Sync Weights
                player = actor1._learner_manager.get_current_player()
                opponent_id, _ = player.get_match()

                is_trainable = player.player_id == opponent_id

                actor1.set_current_ckpt(player.player_id)
                actor1.set_opponent_ckpt(opponent_id)
                actor2.set_current_ckpt(opponent_id)
                actor2.set_opponent_ckpt(player.player_id)

                # 2. Run Rollouts Simultaneously
                # We submit both tasks to the local executor so they run at the exact same time
                # allowing them to handshake/step-sync within the environment.
                future_p = game_executor.submit(
                    actor1.unroll_and_push, player.get_weights()
                )

                opponent = league.get_player(opponent_id)
                future_o = game_executor.submit(
                    actor2.unroll_and_push, opponent.get_weights(), is_trainable
                )

                # 3. Wait for both to finish
                trajectory = future_p.result()
                future_o.result()

                # 4. Update Stats
                if not is_trainable:
                    player._payoff.update(player.player_id, opponent_id, trajectory)

            except Exception as e:
                logger.exception(f"Error in worker {worker_id}: {e}")


def run_eval_worker(actor: Actor, stop_event: threading.Event, wandb_run: wandb.Run):
    """
    Runs the evaluation loop. Eval is usually single-player or against a bot,
    so it runs sequentially on this thread.
    """
    learner_manager = actor._learner_manager
    session_id = actor._player_env.username
    last_step = -1

    while not stop_event.is_set():
        try:
            current_step = np.array(
                learner_manager.current_learner.player_state.step_count
            ).item()

            if current_step > last_step:
                last_step = current_step

                player_params = (
                    actor._learner_manager.get_current_player().get_weights()
                )

                # Eval is typically simpler, so we run it directly.
                # If Eval is also 2-player sync, you would need a ThreadPool here too.
                eval_trajectory = actor.unroll_and_push(player_params, do_push=False)

                payoff = eval_trajectory.player_transitions.env_output.win_reward[-1]
                wandb_run.log(
                    {"training_step": current_step, f"wr-{session_id}": payoff}
                )

            time.sleep(5)

        except Exception as e:
            logger.exception(f"Error in eval {session_id}: {e}")
            time.sleep(5)


def main():
    config = get_learner_config()
    logger.info(f"Loaded config:\n{config}")

    learner_player_conf = get_player_model_config(config.generation, train=True)
    learner_build_conf = get_builder_model_config(config.generation, train=True)
    actor_player_conf = get_player_model_config(config.generation, train=False)
    actor_build_conf = get_builder_model_config(config.generation, train=False)

    actor_player_net = get_player_model(actor_player_conf)
    actor_build_net = get_builder_model(actor_build_conf)

    wandb_run = wandb.init(
        project="pokemon-rl",
        config={
            "learner_config": config,
            "player_config": json.loads(learner_player_conf.to_json_best_effort()),
            "builder_config": json.loads(learner_build_conf.to_json_best_effort()),
        },
    )

    gpu_lock = threading.Lock()

    learner_manager = LearnerManager(config, wandb_run)

    resume_from_ckpt = True
    if resume_from_ckpt:
        with open("ckpts/gen9/ckpt_00400000", "rb") as f:
            datum = pickle.load(f)

        for learner in learner_manager.learners.values():
            learner.player_state = learner.player_state.replace(
                **datum["player_state"],
            )
            learner.builder_state = learner.builder_state.replace(
                **datum["builder_state"],
            )

        for weight_id in learner.league.weight_store._weights.keys():
            container = learner.league.weight_store.get(weight_id)
            learner.league.weight_store._weights[weight_id] = ParamsContainer(
                player_type=container.player_type,
                parent=container.player_type,
                step_count=container.player_type,
                player_frame_count=container.player_type,
                builder_frame_count=container.player_type,
                player_params=datum["player_state"]["params"],
                builder_params=datum["builder_state"]["params"],
            )

    learning_agent = Agent(
        actor_player_net.apply, actor_build_net.apply, gpu_lock=gpu_lock
    )
    eval_agent = Agent(
        actor_player_net.apply,
        actor_build_net.apply,
        gpu_lock=gpu_lock,
        player_head_params=HeadParams(temp=0.8, min_p=0.1),
        builder_head_params=HeadParams(temp=0.8, min_p=0.1),
    )

    # --- 4. Main Threading Loop ---
    stop_event = threading.Event()
    threads: List[threading.Thread] = []

    # Spawn Training Threads
    for game_id in range(config.num_actors // 2):
        actors = [
            Actor(
                agent=learning_agent,
                env=SinglePlayerSyncEnvironment(
                    f"train:p{pid}g{game_id:02d}", generation=config.generation
                ),
                unroll_length=config.unroll_length,
                learner_manager=learner_manager,
                rng_seed=len(threads),
            )
            for pid in range(2)
        ]

        # We spawn ONE thread per Game Pair.
        # Inside that thread, it uses a local threadpool to handle the 2-player sync.
        t = threading.Thread(
            target=run_training_worker,
            args=(*actors, stop_event),
            name=f"Trainer-{game_id}",
        )
        t.start()
        threads.append(t)

    # Spawn Eval Threads
    for eval_id in range(config.num_eval_actors):
        actor = Actor(
            agent=eval_agent,
            env=SinglePlayerSyncEnvironment(
                f"eval-heuristic:{eval_id:04d}", generation=config.generation
            ),
            unroll_length=config.unroll_length,
            learner_manager=learner_manager,
            rng_seed=len(threads),
        )

        t = threading.Thread(
            target=run_eval_worker,
            args=(actor, stop_event, wandb_run),
            name=f"Eval-{eval_id}",
        )
        t.start()
        threads.append(t)

    learner_thread = threading.Thread(target=learner_manager.train)
    learner_thread.start()

    # --- 5. Run & Cleanup ---
    logger.info(f"System running with {len(threads)} manager threads.")
    try:
        learner_thread.join()
    except KeyboardInterrupt:
        logger.warning("Keyboard Interrupt detected. Shutting down...")
    finally:
        stop_event.set()
        for t in threads:
            t.join()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
