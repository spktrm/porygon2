from dotenv import load_dotenv

load_dotenv()
import argparse
import concurrent.futures
import dataclasses
import functools
import gc
import json
import logging
import os
import shutil
import threading
import time
from typing import Literal

import jax
import numpy as np
import wandb.wandb_run

import wandb
from rl import checkpoint
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
    merge_pending_exploiter_promotions,
)
from rl.online.builder_actor import BuilderActor
from rl.online.config import Porygon2LearnerConfig, get_learner_config
from rl.online.league import MAIN_KEY, League
from rl.online.learner import (
    CAT_VF_SUPPORT,
    ExploiterBudgetExhausted,
    ExploiterPhaseRequested,
    ExploiterPromoted,
    Learner,
)
from rl.online.player_actor import PlayerActor

logger = logging.getLogger(__name__)

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
    learner_config: Porygon2LearnerConfig,
):
    """Runs an actor to produce num_trajectories trajectories."""
    learner = actor._learner

    with learner.gpu_lock:
        step_count = np.array(learner.player_state.step_count).item()

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
        try:
            with learner.gpu_lock:
                new_step_count = np.array(learner.player_state.step_count).item()
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

        except Exception:
            logger.error("Error running eval heuristic", exc_info=True)
            # Dont let bad evaluation crash the whole training loop
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


@dataclasses.dataclass
class PhaseOutcome:
    """What a single call to _run_one_phase ended with.

    kind:
        "exhausted"           — config.num_steps reached (essentially never
                                 happens at num_steps=5,000,000; here for
                                 completeness).
        "exploiter_requested" — main paused itself for a PSRO exploiter
                                 phase; checkpoint_path is a freshly written
                                 full checkpoint to fork from.
        "promoted"            — a running exploiter cleared the promotion
                                 bar and wrote a promoted snapshot.
        "failed"              — a running exploiter exhausted its frame
                                 budget without clearing the bar; nothing
                                 was written, its state is simply discarded.
        "interrupted"         — Ctrl-C. Deliberately NOT re-raised past
                                 this point: the orchestration loop below
                                 only continues on "exploiter_requested", so
                                 returning this (rather than letting
                                 KeyboardInterrupt propagate all the way to
                                 the top uncaught) is what stops the whole
                                 loop cleanly instead of dumping a traceback
                                 after a perfectly normal, intentional stop.
    """

    kind: Literal[
        "exhausted", "exploiter_requested", "promoted", "failed", "interrupted"
    ]
    checkpoint_path: str | None = None
    promoted_snapshot_dir: str | None = None


def _run_one_phase(
    learner_config: Porygon2LearnerConfig,
    debug: bool,
    mode: str,
    fork_from_ckpt: str | None,
    run_subdir: str | None,
    reset_plasticity_overdue: bool,
    wandb_group: str | None,
) -> PhaseOutcome:
    """Runs actors + the learner for one phase — main, or a single
    exploiter attempt — until config.num_steps, a crash, Ctrl-C, or an
    exploiter-phase transition. Exactly one phase is ever live at a time:
    this hardware has one GPU and can't run distributed/concurrent
    training, and the design (docs/exploiter-phase-plan.md) is strictly
    sequential regardless — pause/fork/train/promote-or-discard/resume,
    never two learners at once. main()'s orchestration loop calls this
    repeatedly from the same process for that reason: no subprocess
    spawning, no separate launch commands, one continuous GPU context.

    Each call is its own wandb run (wandb.init() below) — a phase
    transition always starts a fresh run, it doesn't rename the previous
    one. wandb_group ties every phase of one orchestrated episode (main's
    initial segment, every exploiter attempt, every subsequent main
    resume) together under wandb's Group view, so "the run" stays
    findable as a unit even though it's several run names underneath.
    """
    if debug:
        os.environ["WANDB_MODE"] = "disabled"

    salt = time.time_ns()

    logger.info(f"Learner Config: {learner_config}")
    if learner_config.pin_opponent_steps and not run_subdir:
        raise ValueError(
            "config.pin_opponent_steps is set (this is an exploiter-phase "
            "run) but run_subdir is not — this would namespace this run's "
            "checkpoints/snapshots into main's own checkpoint tree instead "
            "of ckpts/gen{N}/exploiters/{run_subdir}/, eventually colliding "
            "with main's ckpt_{step:08}/players/p_{step:08} directories at "
            "the same step counts once main resumes."
        )

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
    logger.info("Loading train state...")
    player_state, builder_state, league, controller_bytes = load_train_state(
        learner_config,
        player_state,
        builder_state,
        mode=mode,
        fork_from_ckpt=fork_from_ckpt,
        run_subdir=run_subdir,
    )

    # A promoted exploiter (rl/online/promote_exploiter.py, or the
    # in-process auto path in rl/online/learner.py) only reaches a live
    # league here, at the start of a fresh phase — see
    # docs/exploiter-phase-plan.md piece 5. Harmless to call for an
    # exploiter phase too: already-merged step counts are skipped, and a
    # freshly forked exploiter's league already carries whatever main had
    # merged as of the fork point.
    merged = merge_pending_exploiter_promotions(learner_config, league)
    if merged:
        logger.info("Merged %d promoted exploiter snapshot(s) into the league.", merged)

    player_state = jax.device_put(player_state)
    builder_state = jax.device_put(builder_state)

    logger.info("Initializing WandB...")
    wandb_run = wandb.init(
        project="pokemon-rl",
        # Ties every phase of one orchestrated episode together in wandb's
        # Group view — without this each phase transition (pause, fork,
        # resume) would be an unrelated-looking run name with nothing
        # connecting it to the episode it's actually part of.
        group=wandb_group,
        # Open question 3 in the exploiter-phase doc: keep exploiter runs
        # visually distinguishable on the dashboard so they never get
        # mistaken for "the next real main lineage".
        tags=["exploiter"] if learner_config.pin_opponent_steps else None,
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
        player_network=learner_player_network,
        debug=debug,
        controller_bytes=controller_bytes,
        run_subdir=run_subdir,
    )
    if reset_plasticity_overdue:
        learner.plasticity.acknowledge_exploiter_episode()

    env_func = functools.partial(
        SinglePlayerSyncEnvironment,
        generation=learner_config.generation,
        smogon_format=learner_config.smogon_format,
    )

    outcome = PhaseOutcome(kind="exhausted")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=(
            learner_config.num_player_actors + 2 * len(learner_config.eval_baselines)
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
        # reads the current thread's name), including the baseline being
        # played, so metric keys stay meaningful if the eval allocation
        # changes.
        logger.info(
            f"Initializing {len(learner_config.eval_baselines)} evaluation actors "
            f"(baseline indices: {learner_config.eval_baselines})..."
        )
        for eval_id, baseline_index in enumerate(learner_config.eval_baselines):
            baseline_name = EVAL_BASELINE_NAMES[baseline_index]
            actor = PlayerActor(
                agent=eval_agent,
                # The username MUST start with "eval-heuristic" (the service
                # routes such clients into games against a baseline bot by
                # that prefix, service/src/server/utils.ts) and its ":<n>"
                # suffix selects which baseline: the service parses the
                # trailing number into an evalActionMapping index
                # (service/src/server/runner.ts).
                env=env_func(
                    f"eval-heuristic-{baseline_name}-{eval_id}:{baseline_index:04d}"
                ),
                unroll_length=learner_config.unroll_length,
                learner=learner,
                rng_seed=len(actor_threads) + salt,
                is_eval=True,
            )
            actor_threads.append(
                threading.Thread(
                    target=run_eval_heuristic,
                    args=(actor, executor, stop_signal, wandb_run, learner_config),
                    name=f"EvalActor-{baseline_name}-{eval_id}",
                )
            )

        # Start the actors and learner.
        for t in actor_threads:
            t.start()

        try:
            learner.train()
        except KeyboardInterrupt:
            # Deliberately not re-raised: Learner.train() has already saved
            # a checkpoint on its way out (its own KeyboardInterrupt
            # handler) and this method's finally block below still joins
            # every actor thread and finishes wandb cleanly either way.
            # Returning "interrupted" is what actually stops the
            # orchestration loop in main() (it only continues on
            # "exploiter_requested") — re-raising here just sent a
            # perfectly normal, intentional Ctrl-C stop past main()
            # uncaught, printing a traceback that looks exactly like a
            # crash for something that wasn't one.
            logger.info("Keyboard interrupt received. Shutting down gracefully...")
            outcome = PhaseOutcome(kind="interrupted")
        except ExploiterPhaseRequested as e:
            logger.info("Main paused for an exploiter phase @ %s", e.checkpoint_path)
            outcome = PhaseOutcome(
                kind="exploiter_requested", checkpoint_path=e.checkpoint_path
            )
        except ExploiterPromoted as e:
            logger.info("Exploiter promoted -> %s", e.snapshot_dir)
            outcome = PhaseOutcome(
                kind="promoted", promoted_snapshot_dir=e.snapshot_dir
            )
        except ExploiterBudgetExhausted:
            logger.info(
                "Exploiter exhausted its budget without promoting — discarding."
            )
            outcome = PhaseOutcome(kind="failed")
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

    return outcome


def _pick_pin_opponent_steps(
    checkpoint_path: str, k: int, min_games: float
) -> tuple[int, ...]:
    """The k opponents main currently struggles with MOST, as of
    checkpoint_path — ranked by the same win-rate table
    Learner._measure_exploitability / _should_add_new_player already read,
    not by recency. Recency was the doc's original proxy for "resembles
    current main," but it's a weak one: a persistent shared blind spot can
    survive across many "recent" snapshots, while the league's own
    win/loss counts say directly which opponent is actually exposing it.

    Snapshots with fewer than min_games effective games against main are
    excluded from the ranking — the same reliability bar as
    exploit_ctrl_min_games_per_opponent/bandit_min_games_per_opponent,
    applied here for the identical reason: a freshly-added or
    lightly-played snapshot reads near 0.5 by construction (main vs. a
    near-identical recent self), which looks exactly like a real hole
    otherwise (1338's exploit_ctrl false-positive, in the doc's session
    log, was exactly this). Excluded snapshots are only used as a
    recency-ordered filler if there aren't enough reliably-measured hard
    opponents to reach k.
    """
    league_bytes = checkpoint.load_league_bytes(checkpoint_path)
    if league_bytes is None:
        return ()
    league = League.deserialize(league_bytes)
    historical = [s for s in league.players if s != MAIN_KEY]
    if not historical:
        return ()

    def _games_vs_main(step: int) -> float:
        return league.games.get((MAIN_KEY, step), 0.0) + league.games.get(
            (step, MAIN_KEY), 0.0
        )

    rateable = [s for s in historical if _games_vs_main(s) >= min_games]
    hardest_first = sorted(
        rateable, key=lambda s: league._win_rate_by_steps(MAIN_KEY, s)
    )
    picked = hardest_first[:k]

    if len(picked) < k:
        already_picked = set(picked)
        fallback_by_recency = sorted(
            (s for s in historical if s not in already_picked), reverse=True
        )
        picked += fallback_by_recency[: k - len(picked)]

    return tuple(picked)


def _cleanup_exploiter_run(generation: int, run_id: str) -> None:
    """Deletes an exploiter attempt's own working checkpoint tree once its
    outcome is known and already acted on. Safe for both terminal
    outcomes: a promotion already copied everything that matters into
    exploiters/promoted/ (write_promoted_snapshot), and a clean
    budget-exhausted failure leaves nothing worth keeping — this is
    disposable search scratch in the fully-automated pipeline, not unique
    work product (the wandb run for the attempt is the permanent record of
    what happened). Only ever called after _run_one_phase returns a clean
    PhaseOutcome — a genuine crash propagates past this call entirely, so
    a real bug still has a checkpoint tree to inspect.
    """
    run_root = f"./ckpts/gen{generation}/exploiters/{run_id}"
    shutil.rmtree(run_root, ignore_errors=True)
    logger.info("Cleaned up discarded exploiter checkpoints at %s", run_root)


def main(args: argparse.Namespace):
    """Orchestrates one continuous session: main training, with automatic
    PSRO exploiter phases (docs/exploiter-phase-plan.md) interleaved in the
    SAME process when config.auto_exploiter_enabled is set. One command,
    no manual promote_exploiter.py invocation, no separate launch per
    phase — everything after the initial launch is driven by this loop.
    """
    learner_config = get_learner_config()
    debug = args.debug

    # Only the very first phase in this process honours these — every
    # phase after that is an internal, deterministic transition (fork a
    # fresh exploiter, or resume main from the checkpoint it just paused
    # at), never "start from scratch" or "merge params across an
    # architecture change".
    initial_mode = os.environ.get("LOAD_STATE_MODE", "checkpoint")
    initial_fork_from_ckpt = os.environ.get("FORK_FROM_CKPT")
    initial_run_subdir = os.environ.get("EXPLOITER_RUN_ID")

    # One group per orchestration-loop invocation (i.e. per process launch,
    # not per phase) — every phase this process runs, from main's initial
    # segment through every exploiter attempt to every later main resume,
    # shows up under this same group in wandb, so the whole episode stays
    # findable as a unit despite being several run names underneath.
    wandb_group = f"session-{int(time.time())}"

    outcome = _run_one_phase(
        learner_config,
        debug,
        mode=initial_mode,
        fork_from_ckpt=initial_fork_from_ckpt,
        run_subdir=initial_run_subdir,
        reset_plasticity_overdue=False,
        wandb_group=wandb_group,
    )
    gc.collect()

    while (
        outcome.kind == "exploiter_requested" and learner_config.auto_exploiter_enabled
    ):
        main_checkpoint = outcome.checkpoint_path
        promoted = False

        for k in learner_config.auto_exploiter_ladder:
            pin_opponent_steps = _pick_pin_opponent_steps(
                main_checkpoint, k, learner_config.exploit_ctrl_min_games_per_opponent
            )
            if not pin_opponent_steps:
                logger.warning(
                    "No historical league members yet to pin an exploiter "
                    "against — skipping this stagnation episode entirely."
                )
                break

            run_id = f"auto-{int(time.time())}-k{k}"
            logger.info(
                "Launching exploiter phase: k=%d pin=%s run_id=%s fork=%s",
                k,
                pin_opponent_steps,
                run_id,
                main_checkpoint,
            )
            exploiter_outcome = _run_one_phase(
                learner_config.replace(
                    pin_opponent_steps=pin_opponent_steps,
                    auto_exploiter_enabled=True,
                ),
                debug,
                mode="checkpoint",
                fork_from_ckpt=main_checkpoint,
                run_subdir=run_id,
                reset_plasticity_overdue=False,
                wandb_group=wandb_group,
            )
            gc.collect()

            # Disposable scratch either way: a promotion already copied
            # what matters into exploiters/promoted/, and a clean failure
            # has nothing worth keeping — the wandb run is the permanent
            # record. Only skipped on a genuine crash, which propagates
            # past this call entirely rather than returning a PhaseOutcome.
            if exploiter_outcome.kind in ("promoted", "failed"):
                _cleanup_exploiter_run(learner_config.generation, run_id)

            if exploiter_outcome.kind == "promoted":
                promoted = True
                break
            logger.info(
                "Exploiter attempt k=%d did not clear the promotion bar "
                "within its budget — trying the next ladder rung.",
                k,
            )

        if not promoted:
            logger.info(
                "Exhausted auto_exploiter_ladder=%s without a promotion "
                "this episode. Resuming main unchanged.",
                learner_config.auto_exploiter_ladder,
            )

        logger.info("Resuming main.")
        outcome = _run_one_phase(
            learner_config,
            debug,
            mode="checkpoint",
            fork_from_ckpt=None,
            run_subdir=None,
            reset_plasticity_overdue=True,
            wandb_group=wandb_group,
        )
        gc.collect()

    logger.info("Training run complete (%s).", outcome.kind)


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
