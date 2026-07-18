from dotenv import load_dotenv

load_dotenv()
import argparse
import concurrent.futures
import functools
import json
import logging
import os
import threading
import time

import jax
import numpy as np
import wandb.wandb_run

import wandb
from rl.actor.agent import Agent
from rl.actor.builder_actor import BuilderActor
from rl.actor.player_actor import PlayerActor
from rl.environment.env import SinglePlayerSyncEnvironment
from rl.learner.config import create_train_state, get_learner_config, load_train_state
from rl.learner.learner import CAT_VF_SUPPORT, Learner
from rl.model.builder_model import get_builder_model
from rl.model.config import get_builder_model_config, get_player_model_config
from rl.model.heads import HeadParams
from rl.model.player_model import get_num_params, get_player_model
from rl.model.utils import ParamsContainer

logger = logging.getLogger(__name__)


def run_training_actor_pair(
    player: PlayerActor,
    opponent: PlayerActor,
    executor: concurrent.futures.ThreadPoolExecutor,
    stop_signal: list[bool],
):
    """Runs an actor to produce trajectories"""

    worker_id = threading.current_thread().name

    while not stop_signal[0]:
        try:
            player_params = player.pull_main_player()
            opponent_params, is_trainable = player.get_match()

            player_ckpt = np.array(player_params.step_count).item()
            opponent_ckpt = np.array(opponent_params.step_count).item()

            game_id = f"{worker_id}-p{player_ckpt}-v-p{opponent_ckpt}"
            for actor in (player, opponent):
                actor.set_game_id(game_id)

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
            logger.error(f"Error in {worker_id}: {e}", exc_info=True)
            raise e


def run_eval_heuristic(
    actor: PlayerActor,
    executor: concurrent.futures.ThreadPoolExecutor,
    stop_signal: list[bool],
    wandb_run: wandb.wandb_run.Run,
):
    """Runs an actor to produce num_trajectories trajectories."""
    learner = actor._learner

    with learner.gpu_lock:
        step_count = np.array(learner.player_state.step_count).item()

    # Metric identity comes from the eval thread's name (set at spawn:
    # EvalActor-full-0, ...), not the env username, so renaming or reusing
    # envs never renames the wandb series.
    session_id = threading.current_thread().name
    swap = True

    while not stop_signal[0]:
        try:
            with learner.gpu_lock:
                new_step_count = np.array(learner.player_state.step_count).item()
            if new_step_count > step_count:
                step_count = new_step_count

                # Snapshot params to host under the lock: the learner's train
                # step donates its state buffers, so holding live device
                # references across an unroll would read deleted arrays.
                if swap:
                    prefix = "main"
                    with learner.gpu_lock:
                        player_params = jax.device_get(learner.player_state.params)
                        builder_params = jax.device_get(learner.builder_state.params)
                else:
                    prefix = "ema"
                    with learner.gpu_lock:
                        player_params = jax.device_get(
                            learner.player_state.target_params
                        )
                        builder_params = jax.device_get(
                            learner.builder_state.target_params
                        )

                player = ParamsContainer(
                    step_count=step_count,
                    player_frame_count=0,
                    builder_frame_count=0,
                    player_params=player_params,
                    builder_params=builder_params,
                )

                swap = not swap

                future1 = executor.submit(actor.unroll_and_push, player)
                eval_trajectory = future1.result()

                payoff = (
                    eval_trajectory.player_transitions.env_output.win_reward[-1]
                    @ CAT_VF_SUPPORT
                )

                wandb_run.log(
                    {
                        "training_step": step_count,
                        f"{prefix}-payoff-{session_id}": payoff,
                        f"{prefix}-wr-{session_id}": payoff > 0,
                    }
                )

        except Exception:
            logger.error("Error running eval heuristic", exc_info=True)
            # Dont let bad evaluation crash the whole training loop
            continue

        time.sleep(5)


def run_eval_search_h2h(
    search_actor: PlayerActor,
    policy_actor: PlayerActor,
    executor: concurrent.futures.ThreadPoolExecutor,
    stop_signal: list[bool],
    wandb_run: wandb.wandb_run.Run,
):
    """Head-to-head decision-time-search vs plain-policy on identical params.

    Both sides pull the current main params each game; one steps through
    search_step, the other through the plain policy. Same weights, same
    opponent, so the payoff series is the most direct live estimate of the
    search improvement (0 = no edge; the two series vs the heuristic bot
    measure the same thing with an extra confound). Trajectories never
    enter replay (both actors are is_eval).
    """
    learner = search_actor._learner

    with learner.gpu_lock:
        step_count = np.array(learner.player_state.step_count).item()

    session_id = threading.current_thread().name

    while not stop_signal[0]:
        try:
            with learner.gpu_lock:
                new_step_count = np.array(learner.player_state.step_count).item()
            if new_step_count > step_count:
                step_count = new_step_count

                with learner.gpu_lock:
                    player_params = jax.device_get(learner.player_state.params)
                    builder_params = jax.device_get(learner.builder_state.params)

                container = ParamsContainer(
                    step_count=step_count,
                    player_frame_count=0,
                    builder_frame_count=0,
                    player_params=player_params,
                    builder_params=builder_params,
                )

                game_id = f"{session_id}-{step_count}"
                for actor in (search_actor, policy_actor):
                    actor.set_game_id(game_id)

                future1 = executor.submit(search_actor.unroll_and_push, container)
                future2 = executor.submit(policy_actor.unroll_and_push, container)
                search_trajectory = future1.result()
                future2.result()

                payoff = (
                    search_trajectory.player_transitions.env_output.win_reward[-1]
                    @ CAT_VF_SUPPORT
                )

                wandb_run.log(
                    {
                        "training_step": step_count,
                        f"search-h2h-payoff-{session_id}": payoff,
                        f"search-h2h-wr-{session_id}": payoff > 0,
                    }
                )

        except Exception:
            logger.error("Error running search h2h eval", exc_info=True)
            continue

        time.sleep(5)


def run_builder_actor(actor: BuilderActor, stop_signal: list[bool]):
    while not stop_signal[0]:
        try:
            param_container = actor.pull_main_player()
            new_key = actor.split_rng()
            actor.unroll(new_key, param_container.builder_params)
        except Exception as e:
            logger.error("Error running builder actor", exc_info=True)
            raise e


def main(args: argparse.Namespace):
    """Main function to run the RL learner."""
    debug = args.debug
    if debug:
        os.environ["WANDB_MODE"] = "disabled"

    salt = int(time.time())

    learner_config = get_learner_config()
    logger.info(f"Learner Config: {learner_config}")

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
        player_head_params=HeadParams(temp=0.5),
        builder_head_params=HeadParams(temp=1.0),
    )
    # Same network, but stepping through search_step: decision-time
    # equilibrium search over the search heads (rl/model/search.py).
    # Depth/branching come from the actor model config's cfg.search.
    search_eval_agent = None
    if actor_player_model_config.search.enabled:
        search_eval_agent = Agent(
            functools.partial(actor_player_network.apply, method="search_step"),
            actor_builder_network.apply,
            gpu_lock=gpu_lock,
            player_head_params=HeadParams(temp=0.5),
            builder_head_params=HeadParams(temp=1.0),
        )
    logger.info("Loading train state...")
    # LOAD_STATE_MODE=params merges the latest checkpoint's params into the
    # fresh init (new modules keep their init, trained modules keep their
    # weights; opt_state/league reset) — use for the first launch after an
    # architecture change instead of restarting from scratch.
    load_mode = os.environ.get("LOAD_STATE_MODE", "checkpoint")
    player_state, builder_state, league = load_train_state(
        learner_config, player_state, builder_state, mode=load_mode
    )

    player_state = jax.device_put(player_state)
    builder_state = jax.device_put(builder_state)

    logger.info("Initializing WandB...")
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
    logger.info(f"WandB serialized run: {wandb_run.name}")

    learner = Learner(
        player_state=player_state,
        builder_state=builder_state,
        config=learner_config,
        league=league,
        wandb_run=wandb_run,
        gpu_lock=gpu_lock,
        debug=debug,
    )

    env_func = functools.partial(
        SinglePlayerSyncEnvironment,
        generation=learner_config.generation,
        smogon_format=learner_config.smogon_format,
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=(
            learner_config.num_player_actors
            + 2 * learner_config.num_eval_actors
            # The search-vs-policy head-to-head pair submits two unrolls.
            + 2
        )
    ) as executor:
        if "randombattle" not in learner_config.smogon_format:
            logger.info(
                f"Initializing {learner_config.num_builder_actors} builder actors..."
            )
            for builder_id in range(learner_config.num_builder_actors):
                actor = BuilderActor(
                    agent=learning_agent,
                    learner=learner,
                    rng_seed=len(actor_threads) + salt,
                )
                args = (actor, stop_signal)
                actor_threads.append(
                    threading.Thread(
                        target=run_builder_actor,
                        args=args,
                        name=f"BuilderActor-{builder_id}",
                    )
                )

        logger.info(
            f"Initializing {learner_config.num_player_actors} player actors (self-play)..."
        )
        for game_id in range(learner_config.num_player_actors // 2):
            actors = [
                PlayerActor(
                    agent=learning_agent,
                    env=env_func(f"train:p{player_id}g{game_id:02d}"),
                    unroll_length=learner_config.unroll_length,
                    learner=learner,
                    rng_seed=len(actor_threads) + salt,
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

        # Thread names carry the wandb series identity (run_eval_heuristic
        # reads the current thread's name).
        eval_variants = (("full", eval_agent),)
        if search_eval_agent is not None:
            # The gap between the search and full series (same params, same
            # heuristic opponent) is the live search-improvement signal.
            eval_variants = (*eval_variants, ("search", search_eval_agent))
        logger.info(
            f"Initializing {learner_config.num_eval_actors} evaluation actor pairs "
            f"({' + '.join(name for name, _ in eval_variants)})..."
        )
        for eval_id in range(learner_config.num_eval_actors):
            for variant_name, agent in eval_variants:
                actor = PlayerActor(
                    agent=agent,
                    # The username MUST start with "eval-heuristic": the
                    # service routes clients into games against the heuristic
                    # bot by that prefix (service/src/server/utils.ts).
                    env=env_func(f"eval-heuristic-{variant_name}:{eval_id:04d}"),
                    unroll_length=learner_config.unroll_length,
                    learner=learner,
                    rng_seed=len(actor_threads) + salt,
                    is_eval=True,
                )
                actor_threads.append(
                    threading.Thread(
                        target=run_eval_heuristic,
                        args=(actor, executor, stop_signal, wandb_run),
                        name=f"EvalActor-{variant_name}-{eval_id}",
                    )
                )

        if search_eval_agent is not None:
            # Search-vs-policy head-to-head on identical params: the two
            # actors share a game id each episode, so the service pairs
            # them against each other (matchmaking is by game id).
            logger.info("Initializing search-vs-policy head-to-head eval pair...")
            h2h_actors = [
                PlayerActor(
                    agent=agent,
                    env=env_func(f"h2h-{name}:{0:04d}"),
                    unroll_length=learner_config.unroll_length,
                    learner=learner,
                    rng_seed=len(actor_threads) + salt,
                    is_eval=True,
                )
                for name, agent in (
                    ("search", search_eval_agent),
                    ("policy", eval_agent),
                )
            ]
            actor_threads.append(
                threading.Thread(
                    target=run_eval_search_h2h,
                    args=(*h2h_actors, executor, stop_signal, wandb_run),
                    name="EvalActor-h2h-0",
                )
            )

        # Start the actors and learner.
        for t in actor_threads:
            t.start()

        try:
            learner.train()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down gracefully...")
        finally:
            stop_signal[0] = True
            for t in actor_threads:
                t.join(timeout=30)
            try:
                wandb_run.finish()
            except Exception:
                logger.warning(
                    "wandb_run.finish() failed during shutdown", exc_info=True
                )

    logger.info("Training run complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Run the RL learner.")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode", default=False
    )
    args = parser.parse_args()
    main(args)
