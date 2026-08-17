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
from rl.online.learner import (
    CAT_VF_SUPPORT,
    POPULATION_NAMES,
    Learner,
    OOMGuardTriggered,
    PopulationName,
)
from rl.online.player_actor import ActorStopped, PlayerActor

logger = logging.getLogger(__name__)


class TqdmLoggingHandler(logging.Handler):
    """Routes log records through tqdm.write() instead of a raw stream
    write. tqdm.write() clears whatever progress bars are currently
    rendered, prints the line cleanly above them, then redraws the bars —
    the default StreamHandler writes straight to stderr with no knowledge
    of tqdm's cursor position, which is what corrupted terminal output
    into garbled interleaved text once multiple bars were running
    concurrently (one per population per producer/consumer/batches)."""

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
        # Block-sequential actor gating (Learner._set_active): only the
        # active population's actors play, so the whole actor budget
        # serves whoever's block it is. Timeout keeps the stop_signal
        # check live while gated; the fresh dict lookup survives a
        # population reset replacing the PopulationState object.
        pop = player._learner.populations[player.population]
        if not pop.run_gate.wait(timeout=1.0):
            continue
        try:
            player_params = player.pull_own_player()
            opponent_params, is_trainable = player.get_match()

            player_ckpt = np.array(player_params.step_count).item()
            opponent_ckpt = np.array(opponent_params.step_count).item()

            # Population-prefixed: with all three populations' actors now
            # running concurrently (docs/exploiter-phase-plan.md's
            # three-population redesign), each independently step-counting
            # from its own creation/reset point, two DIFFERENT populations
            # can otherwise produce an identical game_id — impossible
            # under the old design (only one population was ever live at
            # once), now a real collision risk in the TS game server's
            # pendingGames map (service/src/server/worker.ts).
            game_id = (
                f"{player.population}-{worker_id}-p{player_ckpt}-v-p{opponent_ckpt}"
            )
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

            # Tempered games are excluded from the payoff table: PFSP
            # weights, verification picks and promotion bars should read
            # the base policy's strength, not a temp-2 explorer's.
            if not is_trainable and not bool(np.asarray(trajectory.explore).item()):
                player.update_player_league_stats(
                    player_params, opponent_params, trajectory
                )
        except ActorStopped:
            # Clean shutdown unwind (Ctrl-C / population reset) — the
            # population stopped while this actor was blocked inside an
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
    """Runs an actor to produce num_trajectories trajectories. main only —
    win-rate vs. the scripted baselines is a real strength signal for main,
    pure diagnostic noise for an exploiter (docs/exploiter-phase-plan.md);
    exploiter populations never get eval actors at all."""
    learner = actor._learner
    main_pop = learner.populations["main"]

    with learner.gpu_lock:
        step_count = np.array(main_pop.player_state.step_count).item()

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
        # Eval measures main; during another population's block main is
        # (mostly) frozen, so idle at main's own block gate rather than
        # burning shared inference on a frozen policy.
        if not main_pop.run_gate.wait(timeout=1.0):
            continue
        try:
            with learner.gpu_lock:
                new_step_count = np.array(main_pop.player_state.step_count).item()
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
                        player_params = jax.device_get(main_pop.player_state.params)
                        builder_params = jax.device_get(main_pop.builder_state.params)
                else:
                    prefix = "ema"
                    with learner.gpu_lock:
                        player_params = jax.device_get(
                            main_pop.player_state.target_params
                        )
                        builder_params = jax.device_get(
                            main_pop.builder_state.target_params
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
        # Same block gating as run_training_actor_pair.
        pop = actor._learner.populations[actor.population]
        if not pop.run_gate.wait(timeout=1.0):
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
    (docs/exploiter-phase-plan.md) — stopping every "running" run in the
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
    """Launches one persistent process holding three live, continuously-
    training populations — MainPlayer, MainExploiter, LeagueExploiter
    (docs/exploiter-phase-plan.md's 2026-08-12 three-population redesign,
    superseding the old discrete-phases-in-one-process orchestration this
    function used to run). Everything after initial setup is driven by a
    single call to Learner.train(), which runs for the life of the
    process; population creation/reset happens in-process (Learner._reset_
    population), not via a fork-a-new-phase loop here.
    """
    learner_config = get_learner_config()
    debug = args.debug
    if debug:
        os.environ["WANDB_MODE"] = "disabled"

    logger.info(f"Learner Config: {learner_config}")

    learner_player_model_config = get_player_model_config(
        learner_config.generation,
        train=True,
        q_head_enabled=learner_config.player_q_enabled,
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

    # Shared across all three populations — same architecture, so one
    # Agent/gpu_lock serves everyone. Constructing a separate Agent per
    # population would trigger redundant jax.jit traces of the identical
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
    # No separate exploration Agents: head_params is a per-call traced
    # argument of Agent.step_player now, so ladder actors share
    # learning_agent and pass their per-game sampled temperature
    # themselves (see PlayerActor). They bypass the batched
    # InferenceServer (which serves everyone at the base temperature) the
    # same way eval actors do.

    # One batched-inference server for ALL training PlayerActors across
    # the three populations (rl/online/inference.py). Same apply_fn and
    # default HeadParams as learning_agent — eval actors stay on
    # eval_agent's direct path (different sampling temperature, and 3
    # low-volume threads don't warrant a second server). Builder actors
    # also stay direct: builder inference is one team-build per game vs.
    # ~35 player steps, negligible traffic.
    inference_server = None
    if learner_config.inference_server_enabled:
        inference_server = InferenceServer(
            actor_player_network.apply,
            gpu_lock=gpu_lock,
            max_batch=learner_config.inference_max_batch,
            params_cache_size=learner_config.inference_params_cache_size,
        )
        inference_server.start()

    logger.info("Loading main's train state...")
    mode = os.environ.get("LOAD_STATE_MODE", "checkpoint")
    player_state, builder_state, league, controller_bytes, resume_state = (
        load_train_state(learner_config, player_state, builder_state, mode=mode)
    )
    player_state = jax.device_put(player_state)
    builder_state = jax.device_put(builder_state)

    # One wandb_group ties all three populations' runs together under
    # wandb's Group view as one session — replaces the old meaning ("one
    # episode's phases") with "one process's 3 populations", same
    # underlying convention.
    #
    # A checkpoint-mode restart reuses the previous session's group and
    # per-population run ids (persisted next to the checkpoints), so each
    # population keeps one continuous wandb run across restarts instead of
    # a new trio per process. load_wandb_run_info returns None whenever the
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

    logger.info("Initializing WandB (3 persistent runs, one per population)...")
    wandb_runs: dict[PopulationName, wandb.wandb_run.Run] = {}
    for pop_name in POPULATION_NAMES:
        run_id = resume_run_ids.get(pop_name)
        wandb_runs[pop_name] = wandb.init(
            project="pokemon-rl",
            group=wandb_group,
            job_type=pop_name,
            name=f"{wandb_group}-{pop_name}",
            id=run_id,
            # "allow", not "must": resume the run when it still exists
            # server-side, otherwise recreate it under the same id — a
            # wandb-side deletion should never block a training restart.
            resume="allow" if run_id else None,
            # Keeps exploiter runs visually distinguishable on the
            # dashboard so they never get mistaken for "the next real main
            # lineage" (docs/exploiter-phase-plan.md open question 3).
            tags=[pop_name] + (["exploiter"] if pop_name != "main" else []),
            config=model_config_payload,
            # Required for 3 genuinely-concurrent runs from one process.
            # wandb.init()'s default reinit mode ("default" -> "return_
            # previous" outside notebooks) means every call AFTER the
            # first just hands back the already-active run instead of
            # creating a new one — confirmed the hard way: without this,
            # main_exploiter/league_exploiter silently received main's own
            # Run object, so only one run ever showed up per session
            # group. "create_new" makes each call independently create a
            # real run; its only cost (not updating the wandb.run global /
            # top-level wandb.log) is a non-issue here since every caller
            # already logs through the specific Run object handed back
            # (wandb_runs[...]/pop.wandb_run), never the bare wandb.log().
            reinit="create_new",
        )
        # Default x-axis = the population's own monotonic lifetime_step
        # (logged with every learner metric; carried across resumes and
        # attempt re-forks — see PopulationState.lifetime_step). Without
        # this, charts plot against _step (log-call count) and every
        # resume/re-fork draws a sawtooth or paints over earlier x-ranges.
        wandb_runs[pop_name].define_metric("*", step_metric="lifetime_step")
        logger.info(
            "WandB serialized run (%s): %s (id=%s, resumed=%s)",
            pop_name,
            wandb_runs[pop_name].name,
            wandb_runs[pop_name].id,
            wandb_runs[pop_name].resumed,
        )

    # Written every session (fresh or resumed): the next checkpoint-mode
    # restart reads this to resume these exact runs.
    save_wandb_run_info(
        learner_config,
        wandb_group,
        {pop_name: run.id for pop_name, run in wandb_runs.items()},
    )

    env_func = functools.partial(
        SinglePlayerSyncEnvironment,
        generation=learner_config.generation,
        smogon_format=learner_config.smogon_format,
    )
    # Every population's pool is main-sized now, but the run_gate means at
    # most one pool plays at a time (plus a brief overlap while a
    # just-gated-off pool finishes its in-flight games at a block switch,
    # plus the eval actors) — hence 2x one pool, not 3x.
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=(
            2 * learner_config.num_player_actors
            + 2 * len(learner_config.eval_baselines)
        )
    )

    learner: Learner | None = None

    def spawn_actor_pool(population: PopulationName) -> None:
        """Learner's spawn_actor_pool callback (learner.py can't import
        PlayerActor/BuilderActor itself — both already import Learner,
        so constructing them here in main.py and registering the threads
        back onto the population avoids a circular import). Called
        synchronously from Learner._reset_population the instant a
        population's live params are set — for "main" this fires once
        during initial setup below; for the two exploiter populations it
        fires every creation AND every reset, since their actor pools
        must be rebuilt against the freshly-forked params each time.

        Every population gets the SAME full-size pool: the per-population
        run_gate (Learner._set_active) means only the block owner's pool
        actually plays, so uniform sizing is what gives whoever is
        training the full actor budget — the old smaller exploiter pools
        were sized for all pools running concurrently, which gating made
        obsolete."""
        num_player_actors = learner_config.num_player_actors
        num_builder_actors = learner_config.num_builder_actors
        stop_signal = learner.populations[population].stop_signal
        salt = time.time_ns()
        new_threads: list[threading.Thread] = []

        if "randombattle" not in learner_config.smogon_format:
            logger.info(
                "[%s] Initializing %d builder actors...", population, num_builder_actors
            )
            for builder_id in range(num_builder_actors):
                actor = BuilderActor(
                    agent=learning_agent,
                    learner=learner,
                    rng_seed=len(new_threads) + salt,
                    population=population,
                )
                new_threads.append(
                    threading.Thread(
                        target=run_builder_actor,
                        args=(actor, stop_signal),
                        name=f"BuilderActor-{population}-{builder_id}",
                        daemon=True,
                    )
                )

        logger.info(
            "[%s] Initializing %d player actors (self-play)...",
            population,
            num_player_actors,
        )
        # Exploration ladder: every actor independently draws a per-game
        # explore coin (explore_game_prob) and, on explore games, a fresh
        # log-uniform temperature — no dedicated ladder slots. Dedicated
        # slots bypassed the InferenceServer full-time and out-produced
        # the server-queued base pairs ~4x (44% row share instead of the
        # intended ~17%); a per-game coin makes the trajectory share equal
        # the probability by construction, and tempered play is spread
        # across the whole matchmaking mix. The untempered side of a
        # mixed game still pushes ordinary PG/value rows — played against
        # an exploring opponent, which is exactly the opponent-switch-
        # pressure coverage mirror self-play stopped producing.
        for game_id in range(num_player_actors // 2):
            actors = []
            for player_id in range(2):
                slot = game_id * 2 + player_id
                actors.append(
                    PlayerActor(
                        agent=learning_agent,
                        env=env_func(f"{population}:p{player_id}g{game_id:02d}"),
                        unroll_length=learner_config.unroll_length,
                        learner=learner,
                        rng_seed=len(new_threads) + salt + slot,
                        population=population,
                        inference_client=inference_server,
                        explore_game_prob=learner_config.explore_game_prob,
                        explore_temp_range=learner_config.explore_temp_range,
                    )
                )
            new_threads.append(
                threading.Thread(
                    target=run_training_actor_pair,
                    args=(*actors, executor, stop_signal),
                    name=f"Selfplay-{population}-{game_id}",
                    daemon=True,
                )
            )

        if population == "main":
            logger.info(
                "[main] Initializing %d evaluation actors (baseline indices: %s)...",
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
                    population="main",
                )
                new_threads.append(
                    threading.Thread(
                        target=run_eval_heuristic,
                        args=(
                            actor,
                            executor,
                            stop_signal,
                            wandb_runs["main"],
                            learner_config,
                        ),
                        name=f"EvalActor-{baseline_name}-{eval_id}",
                        daemon=True,
                    )
                )

        for t in new_threads:
            t.start()
        learner.register_actor_threads(population, new_threads)

    learner = Learner(
        config=learner_config,
        league=league,
        player_state=player_state,
        builder_state=builder_state,
        main_wandb_run=wandb_runs["main"],
        main_exploiter_wandb_run=wandb_runs["main_exploiter"],
        league_exploiter_wandb_run=wandb_runs["league_exploiter"],
        gpu_lock=gpu_lock,
        player_network=learner_player_network,
        debug=debug,
        controller_bytes=controller_bytes,
        spawn_actor_pool=spawn_actor_pool,
    )
    # main always exists from process start — its actor pool (and eval
    # actors) spin up now. The two exploiter populations' pools spin up
    # lazily, via the same spawn_actor_pool callback, the instant
    # Learner._reset_population first creates them.
    spawn_actor_pool("main")
    # A checkpoint-mode restart that stopped mid-exploiter-block resumes
    # that block (restored populations get their pools via the same
    # callback — which needs `learner` bound, hence after construction).
    learner.restore_populations(resume_state)

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
            "Shutting down: finishing %d wandb runs (further Ctrl-C is "
            "ignored — this takes a few seconds)...",
            len(wandb_runs),
        )
        executor.shutdown(wait=False, cancel_futures=True)
        for wandb_run in wandb_runs.values():
            try:
                wandb_run.finish(exit_code=1 if crashed else 0)
            except Exception:
                logger.warning(
                    "wandb_run.finish() failed during shutdown", exc_info=True
                )

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
