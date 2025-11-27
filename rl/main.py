from dotenv import load_dotenv

from rl.environment.interfaces import PlayerTrajectory, TokenizedTeam
from rl.learner.buffer import BuilderMetadata, BuilderReplayBuffer
from rl.model.utils import ParamsContainer

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
from rl.actor.actor import BuilderActor, PlayerActor
from rl.actor.agent import BuilderAgent, PlayerAgent
from rl.environment.env import SinglePlayerSyncEnvironment, TeamBuilderEnvironment
from rl.learner.config import (
    LearnerConfig,
    create_builder_train_state,
    create_player_train_state,
    get_learner_config,
    load_train_state,
    save_train_state,
)
from rl.learner.league import MAIN_KEY, PlayerLeague
from rl.learner.learner import BuilderLearner, PlayerLearner
from rl.model.builder_model import get_builder_model
from rl.model.config import get_builder_model_config, get_player_model_config
from rl.model.heads import HeadParams
from rl.model.player_model import get_num_params, get_player_model


def run_builder_actor(
    league: PlayerLeague,
    learner: BuilderLearner,
    actor: BuilderActor,
    executor: concurrent.futures.ThreadPoolExecutor,
    stop_signal: list[bool],
):
    """Runs an actor to produce trajectories"""

    while not stop_signal[0]:
        try:
            main_params = league.get_main_player()
            future = executor.submit(actor.unroll, main_params)
            trajectory = future.result()
            learner.enqueue_traj(trajectory)

        except Exception as e:
            traceback.print_exc()
            raise e


def run_player_actor_pair(
    league: PlayerLeague,
    learner: PlayerLearner,
    actor1: PlayerActor,
    actor2: PlayerActor,
    builder_actor: BuilderActor,
    builder_replay_buffer: BuilderReplayBuffer,
    executor: concurrent.futures.ThreadPoolExecutor,
    stop_signal: list[bool],
):
    while not stop_signal[0]:
        try:
            main_params = league.get_main_player()
            opponent_params, is_trainable = league.get_opponent(main_params)

            main_ckpt = np.array(main_params.step_count).item()
            opponent_ckpt = np.array(opponent_params.step_count).item()

            actor1.set_current_ckpt(main_ckpt)
            actor1.set_opponent_ckpt(opponent_ckpt)

            actor2.set_current_ckpt(opponent_ckpt)
            actor2.set_opponent_ckpt(main_ckpt)

            futures: list[
                tuple[concurrent.futures.Future[PlayerTrajectory], int, BuilderMetadata]
            ] = []
            for actor, param_container, should_sample_team in zip(
                [actor1, actor2], [main_params, opponent_params], [True, is_trainable]
            ):
                builder_key, builder_metadata = None, None
                if should_sample_team:
                    builder_key, builder_trajectory, builder_metadata = (
                        builder_replay_buffer.sample_for_player()
                    )
                else:
                    bulilder_future = executor.submit(
                        builder_actor.unroll, param_container
                    )
                    builder_trajectory = bulilder_future.result()

                tokenized_team = TokenizedTeam(
                    builder_trajectory.history.species_tokens.reshape(-1).tolist(),
                    builder_trajectory.history.packed_set_tokens.reshape(-1).tolist(),
                )
                future = executor.submit(actor.unroll, param_container, tokenized_team)

                futures.append((future, builder_key, builder_metadata))

            for future, builder_key, builder_metadata in futures:
                traj = future.result()
                payoff = traj.transitions.env_output.win_reward[-1].item()

                if builder_key is not None:
                    update_ema = 1e-2
                    new_avg_reward = (
                        update_ema * payoff
                        + (1 - update_ema) * builder_metadata.avg_reward
                    )
                    new_metadata = BuilderMetadata(
                        n_sampled=builder_metadata.n_sampled + 1,
                        avg_reward=new_avg_reward,
                    )
                    builder_replay_buffer.update(builder_key, new_metadata)
                    learner.enqueue_traj(traj)
                else:
                    league.update_payoff(main_params, opponent_params, payoff)

        except Exception as e:
            traceback.print_exc()
            raise e


def run_eval_heuristic(
    league: PlayerLeague,
    learner: PlayerLearner,
    actor: PlayerActor,
    executor: concurrent.futures.ThreadPoolExecutor,
    stop_signal: list[bool],
    wandb_run: wandb.wandb_run.Run,
):
    """Runs an actor to produce num_trajectories trajectories."""
    step_count = np.array(learner._state.step_count).item()

    session_id = actor._env.username

    while not stop_signal[0]:
        try:
            new_step_count = np.array(learner._state.step_count).item()
            if new_step_count > step_count:
                step_count = new_step_count

                player = league.get_main_player()

                future1 = executor.submit(actor.unroll, player)
                eval_trajectory = future1.result()

                payoff = eval_trajectory.transitions.env_output.win_reward[-1]

                wandb_run.log({"Step": step_count, f"wr-{session_id}": payoff})

                time.sleep(5)

            else:
                time.sleep(1)

        except Exception:
            traceback.print_exc()
            # Dont let bad evaluation crash the whole training loop
            continue


def run_learner(learner: PlayerLearner | BuilderLearner):
    learner.train()


def run_league(
    league: PlayerLeague,
    player_learner: PlayerLearner,
    builder_learner: BuilderLearner,
    learner_config: LearnerConfig,
):
    """Runs the league to manage players and opponents."""

    current_time = time.time()

    last_model_update_time = current_time
    last_add_player_time = current_time
    last_save_time = current_time

    while True:
        try:
            current_time = time.time()

            step_count = np.array(player_learner._state.step_count).item()
            frame_count = np.array(player_learner._state.frame_count).item()

            if (
                current_time - last_model_update_time
            ) > learner_config.model_update_interval_secs:
                league.update_main_player(
                    ParamsContainer(
                        frame_count=frame_count,
                        step_count=MAIN_KEY,
                        player_params=player_learner._state.params,
                        builder_params=builder_learner._state.params,
                    )
                )
                last_model_update_time = current_time

            if (
                current_time - last_add_player_time
            ) > learner_config.add_player_interval_secs:
                latest_player = league.get_latest_player()

                steps_passed = frame_count - latest_player.frame_count
                if steps_passed < learner_config.add_player_min_frames:
                    return False

                historical_players = [
                    v for k, v in league.players.items() if k != MAIN_KEY
                ]
                win_rates = league.get_winrate(
                    (league.players[MAIN_KEY], historical_players)
                )

                should_add_player = (win_rates.min() > 0.7) | (
                    steps_passed >= learner_config.add_player_max_frames
                )

                if should_add_player:
                    print("Adding new player to league @ step", step_count)
                    league.add_player(
                        ParamsContainer(
                            frame_count=frame_count,
                            step_count=step_count,
                            player_params=player_learner._state.params,
                            builder_params=builder_learner._state.params,
                        )
                    )

            if (current_time - last_save_time) > learner_config.save_interval_secs:
                print("Saving Training ckpt at step", step_count)
                save_train_state(
                    learner_config,
                    player_learner._state,
                    builder_learner._state,
                    league,
                )
                last_save_time = current_time

            time.sleep(1)

        except Exception:
            traceback.print_exc()
            # Dont let bad league management crash the whole training loop
            continue


def main(from_scratch: bool = True):
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

    learner_threads: list[threading.Thread] = []
    actor_threads: list[threading.Thread] = []

    stop_signal = [False]

    gpu_lock = threading.Lock()

    player_agent = PlayerAgent(actor_player_network.apply, gpu_lock=gpu_lock)
    builder_agent = BuilderAgent(actor_builder_network.apply, gpu_lock=gpu_lock)
    eval_player_agent = PlayerAgent(
        actor_player_network.apply,
        gpu_lock=gpu_lock,
        head_params=HeadParams(temp=0.8, min_p=0.1),
    )

    init_key = jax.random.key(42)
    player_state = create_player_train_state(
        learner_player_network, init_key, learner_config.player_config
    )
    builder_state = create_builder_train_state(
        learner_builder_network, init_key, learner_config.builder_config
    )
    league = PlayerLeague(
        main_player=ParamsContainer(
            frame_count=0,
            step_count=MAIN_KEY,
            player_params=player_state.params,
            builder_params=builder_state.params,
        ),
        players=[
            ParamsContainer(
                frame_count=0,
                step_count=0,
                player_params=player_state.params,
                builder_params=builder_state.params,
            )
        ],
        league_size=learner_config.league_size,
    )
    if not from_scratch:
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

    player_config = learner_config.player_config
    builder_config = learner_config.builder_config
    player_learner = PlayerLearner(
        state=player_state,
        config=learner_config.player_config,
        wandb_run=wandb_run,
        gpu_lock=gpu_lock,
    )
    builder_learner = BuilderLearner(
        state=builder_state,
        config=learner_config.builder_config,
        wandb_run=wandb_run,
        gpu_lock=gpu_lock,
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=(
            player_config.num_actors // 2
            + player_config.num_eval_actors
            + builder_config.num_actors
        )
    ) as executor:
        for builder_actor_id in range(builder_config.num_actors):
            actor = BuilderActor(
                agent=builder_agent,
                env=TeamBuilderEnvironment(
                    generation=learner_config.generation, smogon_format="ou_all_formats"
                ),
                unroll_length=builder_config.unroll_length,
                rng_seed=len(actor_threads),
            )
            args = (league, builder_learner, actor, executor, stop_signal)
            actor_threads.append(
                threading.Thread(
                    target=run_builder_actor,
                    args=args,
                    name=f"BuilderActor-{builder_actor_id}",
                )
            )

        historical_builder_actor = BuilderActor(
            agent=builder_agent,
            env=TeamBuilderEnvironment(
                generation=learner_config.generation, smogon_format="ou_all_formats"
            ),
            unroll_length=builder_config.unroll_length,
            rng_seed=len(actor_threads),
        )
        for player_game_id in range(player_config.num_actors // 2):
            actors = [
                PlayerActor(
                    agent=player_agent,
                    env=SinglePlayerSyncEnvironment(
                        f"train:p{player_id}g{player_game_id:02d}",
                        generation=learner_config.generation,
                    ),
                    unroll_length=player_config.unroll_length,
                    rng_seed=len(actor_threads),
                )
                for player_id in range(2)
            ]
            args = (
                league,
                player_learner,
                *actors,
                historical_builder_actor,
                builder_learner._replay_buffer,
                executor,
                stop_signal,
            )
            actor_threads.append(
                threading.Thread(
                    target=run_player_actor_pair,
                    args=args,
                    name=f"PlayerActorPair-{player_game_id}",
                )
            )

        for builder_actor_id in range(player_config.num_eval_actors):
            actor = PlayerActor(
                agent=eval_player_agent,
                env=SinglePlayerSyncEnvironment(
                    f"eval-heuristic:{builder_actor_id:04d}",
                    generation=learner_config.generation,
                ),
                unroll_length=player_config.unroll_length,
                rng_seed=len(actor_threads),
            )
            args = (actor, executor, stop_signal, wandb_run)
            actor_threads.append(
                threading.Thread(
                    target=run_eval_heuristic,
                    args=args,
                    name=f"EvalActor-{builder_actor_id}",
                )
            )

        for learner in [player_learner, builder_learner]:
            learner_threads.append(
                threading.Thread(
                    target=run_learner,
                    args=(learner,),
                    name=f"{type(learner).__name__}-Thread",
                )
            )

        learner_threads.append(
            threading.Thread(
                target=run_league,
                args=(league, player_learner, builder_learner, learner_config),
                name="League-Thread",
            )
        )

        # Start the actors and learner.
        for t in actor_threads:
            t.start()

        for t in learner_threads:
            t.start()

        for t in learner_threads:
            t.join()

        stop_signal[0] = True
        for t in actor_threads:
            t.join()

    print("done")


if __name__ == "__main__":
    main()
