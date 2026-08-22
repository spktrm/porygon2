from dotenv import load_dotenv

load_dotenv()
import argparse
import concurrent.futures
import functools
import json
import logging
import os
import signal
import sys
import threading
import time

import jax
import numpy as np
import wandb.wandb_run
from tqdm import tqdm

import wandb
from rl.environment.data import CAT_VF_SUPPORT
from rl.environment.env import SinglePlayerSyncEnvironment
from rl.environment.protos.features_pb2 import EntityPublicNodeFeature
from rl.model.builder_model import get_builder_model
from rl.model.config import get_builder_model_config, get_player_model_config
from rl.model.heads import HeadParams
from rl.model.player_model import get_num_params, get_player_model
from rl.model.utils import ParamsContainer
from rl.online.agent import Agent
from rl.online.artifact import (
    create_train_state,
    load_train_state,
    load_wandb_run_info,
    save_wandb_run_info,
)
from rl.online.builder_actor import BuilderActor
from rl.online.config import Porygon2LearnerConfig, get_learner_config
from rl.online.inference import InferenceServer
from rl.online.player_actor import ActorStopped, PlayerActor
from rl.online.training import Learner, OOMGuardTriggered

logger = logging.getLogger(__name__)


class TqdmLoggingHandler(logging.Handler):
    """Routes log records through tqdm.write() instead of a raw stream
    write. tqdm.write() clears whatever progress bars are currently
    rendered, prints the line cleanly above them, then redraws the bars —
    the default StreamHandler writes straight to stderr with no knowledge
    of tqdm's cursor position, which is what corrupted terminal output
    into garbled interleaved text once multiple bars were running
    concurrently (producer/consumer/batches)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


class JaxCacheNoiseFilter(logging.Filter):
    """.env's JAX_EXPLAIN_CACHE_MISSES=1 logs a MISS line plus a "why not
    persisted" line for every compile under
    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS — by design (see the .env
    comment), and the overwhelming majority of what fires at startup, since
    the small utility jits (jit__normal, jit_multiply, ...) are all
    sub-2s and re-trigger on every restart. Drop that harmless pair only;
    keep the rarer host-callback/process-id "not writing" reasons visible —
    those are exactly what EXPLAIN_CACHE_MISSES was turned on to catch
    (a real silent-miss regression looks like one of those, not this)."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if msg.startswith("PERSISTENT COMPILATION CACHE MISS"):
            return False
        if "because it took <" in msg:
            return False
        return True


# Wandb metric-key names for the service's evalActionMapping indices
# (service/src/server/eval.ts).
EVAL_BASELINE_NAMES = {0: "random", 1: "default", 2: "simpleheuristic"}


def run_training_actor_pair(
    player: PlayerActor,
    opponent: PlayerActor,
    executor: concurrent.futures.ThreadPoolExecutor,
    stop_signal: list[bool],
):
    """Runs an actor to produce trajectories"""

    worker_id = threading.current_thread().name

    while not stop_signal[0]:
        # Actor gate: timeout keeps the stop_signal check live while
        # gated, and the fresh dict lookup survives a rebuild replacing
        # the RunState object.
        run_state = player._learner.run_state
        if not run_state.run_gate.wait(timeout=1.0):
            continue
        try:
            player_params = player.pull_own_player()
            opponent_params, is_trainable = player.get_match()

            player_ckpt = np.array(player_params.step_count).item()
            opponent_ckpt = np.array(opponent_params.step_count).item()

            # Population-prefixed: game_id must be unique in the TS game
            # server's pendingGames map (service/src/server/worker.ts).
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
        except ActorStopped:
            # Clean shutdown unwind (Ctrl-C) — training stopped while
            # this actor was blocked inside an
            # unroll. Not an error; just stop producing.
            break
        except Exception as e:
            logger.error(f"Error in {worker_id}: {e}", exc_info=True)
            raise e


def run_eval_heuristic(
    actor: PlayerActor,
    executor: concurrent.futures.ThreadPoolExecutor,
    stop_signal: list[bool],
    wandb_run: wandb.wandb_run.Run,
    learner_config: Porygon2LearnerConfig,
):
    """Runs an actor to produce num_trajectories trajectories."""
    learner = actor._learner
    main_run_state = learner.run_state

    with learner.gpu_lock:
        step_count = np.array(main_run_state.player_state.step_count).item()

    # Metric identity comes from the eval thread's name (set at spawn:
    # EvalActor-simpleheuristic-0, ...), not the env username, so renaming
    # or reusing envs never renames the wandb series.
    session_id = threading.current_thread().name

    games = 0
    # Bias-corrected exponential smoothing over the EMA-params series: the
    # raw per-game values are 0/1 (win) and -6..6 (margin), far too noisy
    # to read per checkpoint.
    smooth_decay = 0.5 ** (1.0 / max(learner_config.eval_smoothing_halflife, 1))
    smooth_wr = 0.0
    smooth_margin = 0.0
    smooth_weight = 0.0

    while not stop_signal[0]:
        if not main_run_state.run_gate.wait(timeout=1.0):
            continue
        try:
            with learner.gpu_lock:
                new_step_count = np.array(main_run_state.player_state.step_count).item()
            if new_step_count > step_count:
                step_count = new_step_count
                games += 1

                # Snapshot params to host under the lock: the learner's train
                # step donates its state buffers, so holding live device
                # references across an unroll would read deleted arrays.
                # EMA target params by default — the deployment/league
                # params — with an occasional main-params game as a
                # divergence check (the two lag by only ~1/ema_rate steps).
                use_main = (
                    learner_config.eval_main_params_every > 0
                    and games % learner_config.eval_main_params_every == 0
                )
                if use_main:
                    prefix = "main"
                    with learner.gpu_lock:
                        player_params = jax.device_get(
                            main_run_state.player_state.params
                        )
                        builder_params = jax.device_get(
                            main_run_state.builder_state.params
                        )
                else:
                    prefix = "ema"
                    with learner.gpu_lock:
                        player_params = jax.device_get(
                            main_run_state.player_state.target_params
                        )
                        builder_params = jax.device_get(
                            main_run_state.builder_state.target_params
                        )

                player = ParamsContainer(
                    step_count=step_count,
                    player_frame_count=0,
                    builder_frame_count=0,
                    player_params=player_params,
                    builder_params=builder_params,
                )

                future1 = executor.submit(actor.unroll_and_push, player)
                eval_trajectory = future1.result()

                payoff = (
                    eval_trajectory.player_transitions.env_output.win_reward[-1]
                    @ CAT_VF_SUPPORT
                )

                # Final alive-mon differential, a continuous signal that
                # moves before winrate does. Rows 0-5 of public_team are the
                # agent's side, 6-11 the opponent's (state.ts getPublicTeam
                # orders [playerIndex, 1 - playerIndex]); unrevealed
                # opponents count as alive (FAINTED defaults to 0).
                final_public = np.asarray(
                    eval_trajectory.player_transitions.env_output.public_team[-1]
                )
                fainted = final_public[
                    :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED
                ]
                margin = float(fainted[6:].sum() - fainted[:6].sum())

                logs = {
                    "training_step": step_count,
                    f"{prefix}-payoff-{session_id}": float(payoff),
                    # float, not bool: the wandb UI renders boolean series
                    # as NaN in line plots.
                    f"{prefix}-wr-{session_id}": float(payoff > 0),
                    f"{prefix}-margin-{session_id}": margin,
                    f"games-{session_id}": games,
                }
                if not use_main:
                    smooth_wr = smooth_decay * smooth_wr + float(payoff > 0)
                    smooth_margin = smooth_decay * smooth_margin + margin
                    smooth_weight = smooth_decay * smooth_weight + 1.0
                    logs[f"smoothed-wr-{session_id}"] = smooth_wr / smooth_weight
                    logs[f"smoothed-margin-{session_id}"] = (
                        smooth_margin / smooth_weight
                    )
                wandb_run.log(logs)

        except ActorStopped:
            break
        except Exception:
            logger.error("Error running eval heuristic", exc_info=True)
            # Dont let bad evaluation crash the whole training loop
            continue

        time.sleep(5)


def run_builder_actor(actor: BuilderActor, stop_signal: list[bool]):
    while not stop_signal[0]:
        # Same actor gating as run_training_actor_pair.
        run_state = actor._learner.run_state
        if not run_state.run_gate.wait(timeout=1.0):
            continue
        try:
            param_container = actor.pull_own_player()
            new_key = actor.split_rng()
            # pull_own_player() returns host numpy (league.get_live() ->
            # jax.device_get()'d params) — device_put it ONCE here, same as
            # PlayerActor.unroll_and_push does for player_params. Without
            # this, jax.jit implicitly re-transfers the same host array to
            # device on every one of unroll()'s ~102 steps instead of once,
            # across every concurrent builder actor thread, continuously
            # for the process lifetime — a real memory/throughput cost.
            builder_params = jax.device_put(param_container.builder_params)
            actor.unroll(new_key, builder_params)
        except Exception as e:
            logger.error("Error running builder actor", exc_info=True)
            raise e


def _stop_stale_wandb_runs(
    project: str = "pokemon-rl", skip_ids: set[str] | None = None
):
    """Stop any run this project still shows as "running" from a previous
    process. start.sh's `tmux kill-session` SIGKILLs the old python
    process without giving wandb.finish() a chance to run, so those runs
    otherwise sit "Running" in the dashboard until W&B's own heartbeat
    timeout eventually flips them to Crashed. Called once, before this
    process's own wandb.init() calls, so every "running" run found here is
    necessarily stale. Assumes single-box, single-training-process usage
    — stopping every "running" run in the
    project would be wrong if two training processes were ever live at
    once."""
    try:
        api = wandb.Api()
        runs = list(
            api.runs(f"{api.default_entity}/{project}", filters={"state": "running"})
        )
    except Exception:
        logger.warning(
            "Could not query wandb for stale runs — skipping.", exc_info=True
        )
        return
    for run in runs:
        if skip_ids and run.id in skip_ids:
            # This process is about to wandb.init(id=..., resume=...) this
            # exact run — resuming flips it back to running by itself, and
            # racing a server-side stop against that handoff buys nothing.
            logger.info("Leaving stale wandb run %s — resuming it below.", run.name)
            continue
        try:
            logger.info("Stopping stale wandb run %s (state=running)", run.name)
            run.stop()
        except Exception:
            # Best-effort only — e.g. wandb.Api().Run.stop() doesn't exist
            # before some SDK version (AttributeError on 0.27.2, present by
            # 0.28.1). A stale run this misses just falls back to the
            # pre-existing behavior: sitting "Running" until W&B's own
            # heartbeat timeout marks it Crashed. Never worth blocking this
            # process's own startup over.
            logger.warning(
                "Failed to stop stale wandb run %s — leaving it for W&B's "
                "own timeout to resolve.",
                run.name,
                exc_info=True,
            )


def main(args: argparse.Namespace):
    """Launches one persistent process: the actor pool, the inference
    server, and the learner. Everything after initial setup is driven by a
    single call to Learner.train(), which runs for the life of the
    process."""
    learner_config = get_learner_config()
    debug = args.debug
    if debug:
        os.environ["WANDB_MODE"] = "disabled"

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

    player_state, builder_state = create_train_state(
        learner_player_network,
        learner_builder_network,
        jax.random.key(42),
        learner_config,
    )

    # Shared by the learner and the actors — same architecture, so one
    # Agent/gpu_lock serves everyone. Constructing a separate Agent per
    # extra network would trigger redundant jax.jit traces of the identical
    # apply_fn for no reason (rl/online/agent.py's Agent is already fully
    # stateless w.r.t. "which model": params are a per-call argument).
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
    # One batched-inference server for ALL training PlayerActors
    # (rl/online/inference.py). Same apply_fn and
    # default HeadParams as learning_agent — eval actors stay on
    # eval_agent's direct path (different sampling temperature, and 3
    # low-volume threads don't warrant a second server). Builder actors
    # also stay direct: builder inference is one team-build per game vs.
    # ~35 player steps, negligible traffic.
    # Always on: b219d84 deleted the inference_* config fields (the
    # batch-1 fallback path had no remaining user) but left these reads
    # behind, which raised AttributeError at the first launch after it.
    # The server's constructor defaults are the values the fields held.
    inference_server = InferenceServer(actor_player_network.apply, gpu_lock=gpu_lock)
    inference_server.start()

    logger.info("Loading train state...")
    mode = os.environ.get("LOAD_STATE_MODE", "checkpoint")
    player_state, builder_state, league, controller_bytes = load_train_state(
        learner_config, player_state, builder_state, mode=mode
    )
    player_state = jax.device_put(player_state)
    builder_state = jax.device_put(builder_state)

    # A checkpoint-mode restart reuses the previous session's group and
    # run id (persisted next to the checkpoints), so the run stays
    # continuous across restarts instead of starting a new one per
    # process. load_wandb_run_info returns None whenever the
    # resume didn't actually happen (params/scratch mode, or checkpoint
    # mode falling back to scratch because no checkpoint exists) — those
    # are new lineages and get a fresh session.
    wandb_resume = load_wandb_run_info(learner_config) if mode == "checkpoint" else None
    if wandb_resume is not None:
        wandb_group = wandb_resume["group"]
        logger.info("Resuming previous wandb session %s", wandb_group)
    else:
        wandb_group = f"session-{int(time.time())}"
    resume_run_ids: dict[str, str] = (wandb_resume or {}).get("runs", {})
    model_config_payload = {
        "num_player_params": get_num_params(player_state.params),
        "num_builder_params": get_num_params(builder_state.params),
        "learner_config": learner_config,
        "player_model_config": json.loads(
            learner_player_model_config.to_json_best_effort()
        ),
        "builder_model_config": json.loads(
            learner_builder_model_config.to_json_best_effort()
        ),
    }

    _stop_stale_wandb_runs(skip_ids=set(resume_run_ids.values()))

    logger.info("Initializing WandB...")
    run_id = resume_run_ids.get("main")
    wandb_run = wandb.init(
        project="pokemon-rl",
        group=wandb_group,
        job_type="main",
        name=f"{wandb_group}-main",
        id=run_id,
        # "allow", not "must": resume the run when it still exists
        # server-side, otherwise recreate it under the same id — a
        # wandb-side deletion should never block a training restart.
        resume="allow" if run_id else None,
        tags=["main"],
        config=model_config_payload,
    )
    # Default x-axis = the monotonic lifetime_step (logged with every
    # learner metric; carried across resumes — see RunState.
    # lifetime_step). Without this, charts plot against _step (log-call
    # count) and every resume draws a sawtooth or paints over earlier
    # x-ranges.
    wandb_run.define_metric("*", step_metric="lifetime_step")
    logger.info(
        "WandB serialized run: %s (id=%s, resumed=%s)",
        wandb_run.name,
        wandb_run.id,
        wandb_run.resumed,
    )

    # Written every session (fresh or resumed): the next checkpoint-mode
    # restart reads this to resume this exact run. Kept as a dict keyed by
    # "main" so a runtime file written before the single-population
    # collapse still loads.
    save_wandb_run_info(learner_config, wandb_group, {"main": wandb_run.id})

    env_func = functools.partial(
        SinglePlayerSyncEnvironment,
        generation=learner_config.generation,
        smogon_format=learner_config.smogon_format,
    )
    # Two workers per player actor (both sides of a game step
    # concurrently), plus the eval actors.
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=(
            2 * learner_config.num_player_actors
            + 2 * len(learner_config.eval_baselines)
        )
    )

    learner: Learner | None = None

    def spawn_actor_pool() -> None:
        """Learner's spawn_actor_pool callback (learner.py can't import
        PlayerActor/BuilderActor itself — both already import Learner, so
        constructing them here in main.py and registering the threads back
        on the Learner avoids a circular import). Fires once during initial
        setup below."""
        num_player_actors = learner_config.num_player_actors
        num_builder_actors = learner_config.num_builder_actors
        stop_signal = learner.run_state.stop_signal
        salt = time.time_ns()
        new_threads: list[threading.Thread] = []

        if "randombattle" not in learner_config.smogon_format:
            logger.info("Initializing %d builder actors...", num_builder_actors)
            for builder_id in range(num_builder_actors):
                actor = BuilderActor(
                    agent=learning_agent,
                    learner=learner,
                    rng_seed=len(new_threads) + salt,
                )
                new_threads.append(
                    threading.Thread(
                        target=run_builder_actor,
                        args=(actor, stop_signal),
                        name=f"BuilderActor-{builder_id}",
                        daemon=True,
                    )
                )

        logger.info("Initializing %d player actors (self-play)...", num_player_actors)
        for game_id in range(num_player_actors // 2):
            actors = []
            for player_id in range(2):
                slot = game_id * 2 + player_id
                actors.append(
                    PlayerActor(
                        agent=learning_agent,
                        env=env_func(f"main:p{player_id}g{game_id:02d}"),
                        unroll_length=learner_config.unroll_length,
                        learner=learner,
                        rng_seed=len(new_threads) + salt + slot,
                        inference_client=inference_server,
                    )
                )
            new_threads.append(
                threading.Thread(
                    target=run_training_actor_pair,
                    args=(*actors, executor, stop_signal),
                    name=f"Selfplay-{game_id}",
                    daemon=True,
                )
            )

        logger.info(
            "Initializing %d evaluation actors (baseline indices: %s)...",
            len(learner_config.eval_baselines),
            learner_config.eval_baselines,
        )
        for eval_id, baseline_index in enumerate(learner_config.eval_baselines):
            baseline_name = EVAL_BASELINE_NAMES[baseline_index]
            actor = PlayerActor(
                agent=eval_agent,
                # The username MUST start with "eval-heuristic" (the
                # service routes such clients into games against a
                # baseline bot by that prefix,
                # service/src/server/utils.ts) and its ":<n>" suffix
                # selects which baseline: the service parses the
                # trailing number into an evalActionMapping index
                # (service/src/server/runner.ts).
                env=env_func(
                    f"eval-heuristic-{baseline_name}-{eval_id}:{baseline_index:04d}"
                ),
                unroll_length=learner_config.unroll_length,
                learner=learner,
                rng_seed=len(new_threads) + salt,
                is_eval=True,
            )
            new_threads.append(
                threading.Thread(
                    target=run_eval_heuristic,
                    args=(
                        actor,
                        executor,
                        stop_signal,
                        wandb_run,
                        learner_config,
                    ),
                    name=f"EvalActor-{baseline_name}-{eval_id}",
                    daemon=True,
                )
            )

        for t in new_threads:
            t.start()
        learner.register_actor_threads(new_threads)

    learner = Learner(
        config=learner_config,
        league=league,
        player_state=player_state,
        builder_state=builder_state,
        main_wandb_run=wandb_run,
        gpu_lock=gpu_lock,
        debug=debug,
        controller_bytes=controller_bytes,
        spawn_actor_pool=spawn_actor_pool,
    )
    spawn_actor_pool()

    crashed = False
    try:
        learner.train()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Learner has saved main's checkpoint.")
    except OOMGuardTriggered as e:
        logger.warning(
            "Stopped due to low available memory (checkpoint saved at %s). "
            "Relaunch this process to resume — a fresh process is what "
            "actually reclaims OS memory, so don't just retry in-place.",
            e.checkpoint_path,
        )
    except Exception:
        # Learner.train() already logged the full traceback; this handler
        # exists so the finish() below can mark the wandb runs FAILED.
        # Letting the exception fly past an unconditional finish() left
        # session 1786537634's OOM crash showing as three cleanly-
        # "finished" runs, which sent the postmortem down the wrong path.
        crashed = True
    finally:
        # From here the process is exiting as fast as it safely can — a
        # second Ctrl-C would abort this cleanup midway, skipping the
        # remaining wandb finishes and leaving those runs "Running" until
        # W&B's heartbeat timeout flips them to Crashed. Ignore further
        # SIGINT (main thread only, which is where this finally runs).
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logger.info(
            "Shutting down: finishing the wandb run (further Ctrl-C is "
            "ignored — this takes a few seconds)..."
        )
        executor.shutdown(wait=False, cancel_futures=True)
        try:
            wandb_run.finish(exit_code=1 if crashed else 0)
        except Exception:
            logger.warning("wandb_run.finish() failed during shutdown", exc_info=True)

    if crashed:
        logger.error("Training run crashed — see traceback above.")
        sys.exit(1)
    logger.info("Training run complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[TqdmLoggingHandler()],
    )
    logging.getLogger("jax._src.compiler").addFilter(JaxCacheNoiseFilter())
    parser = argparse.ArgumentParser(description="Run the RL learner.")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode", default=False
    )
    args = parser.parse_args()
    main(args)
