from dotenv import load_dotenv

load_dotenv()
import argparse
import concurrent.futures
import faulthandler
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
from rl import checkpoint
from rl.environment.actor_stats import ActorStats
from rl.environment.data import CAT_VF_SUPPORT
from rl.environment.env import BattleError, SinglePlayerSyncEnvironment
from rl.environment.protos.features_pb2 import EntityPublicNodeFeature
from rl.model.builder_model import get_builder_model
from rl.model.config import get_builder_model_config, get_player_model_config
from rl.model.heads import HeadParams
from rl.model.player_model import get_num_params, get_player_model
from rl.model.utils import ParamsContainer
from rl.online.agent import Agent
from rl.online.artifact import (
    ckpt_root,
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
from rl.online.training.league_ops import import_br_snapshots, register_br_target

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
            # nolock: tqdm's write lock is not signal-safe — a Ctrl-C that
            # lands inside a bar update leaves the main thread holding it
            # forever, and every later log line (the shutdown path's
            # included) then deadlocks behind the logging handler lock. The
            # lock only guards cosmetic bar/line interleaving.
            tqdm.write(self.format(record), nolock=True)
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
            # Both sides must have returned before either actor is reused:
            # one side's failure must not submit a second unroll onto an
            # actor whose first is still running.
            concurrent.futures.wait([future1, future2])
            errors = [f.exception() for f in (future1, future2)]
            if any(isinstance(e, ActorStopped) for e in errors):
                raise ActorStopped("training stopped")
            battle_errors = [e for e in errors if isinstance(e, BattleError)]
            if battle_errors:
                raise battle_errors[0]
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
        except BattleError as e:
            # The service aborted this battle (its step/reset watchdog,
            # or a server-side throw it now reports instead of swallowing).
            # The game is gone on the server; whatever the surviving side
            # pushed before the abort is ordinary truncated-game data.
            # Start another rather than kill this pair's thread — a dead
            # Selfplay thread silently costs the run two actor slots.
            logger.warning(
                "%s: battle aborted by the service, starting a new game — %s",
                worker_id,
                str(e).splitlines()[0] if str(e) else e,
            )
            for actor in (player, opponent):
                actor.reset_game_id()
            continue
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


def resolve_run_setup(
    learner_config: Porygon2LearnerConfig, args: argparse.Namespace
) -> tuple[Porygon2LearnerConfig, str, str | None, str]:
    """CLI → (config, load mode, explicit init ckpt, wandb job name).

    A --br-target run derives everything itself: its own checkpoint
    subtree under br/<tag>, checkpoint-mode resume when that subtree
    already holds a checkpoint, otherwise params-mode init from the
    target (fresh optimiser, zeroed counters, fresh league) shaped by
    --br-init (scratch-mode when that says so)."""
    if args.num_steps is not None:
        learner_config = learner_config.replace(num_steps=args.num_steps)

    if args.br_target is None:
        if args.br_init != "target" or args.br_perturb_frac is not None:
            raise SystemExit(
                "--br-init/--br-perturb-frac only apply to a --br-target run"
            )
        mode = args.load_mode or os.environ.get("LOAD_STATE_MODE", "checkpoint")
        return learner_config, mode, args.init_ckpt, "main"

    if args.load_mode or args.init_ckpt:
        raise SystemExit(
            "--br-target derives its own load mode and init source; "
            "--load-mode/--init-ckpt cannot be combined with it"
        )
    target = os.path.abspath(args.br_target)
    if not os.path.isdir(target):
        raise SystemExit(f"--br-target {target!r} does not exist")
    if args.run_tag is not None:
        run_tag = args.run_tag
    else:
        run_tag = os.path.basename(os.path.normpath(target))
    # Stop condition: an explicit --br-winrate always applies; with
    # NEITHER budget flag given, "train until the target is reliably
    # beaten" (0.7) is the default, with config.num_steps (5e6) as the
    # effectively-unbounded backstop. An explicit --num-steps alone keeps
    # the pure step-budget behaviour.
    if args.br_winrate is not None:
        stop_winrate = args.br_winrate
    elif args.num_steps is None:
        stop_winrate = 0.7
    else:
        stop_winrate = 0.0
    if args.br_perturb_frac is not None and args.br_init != "shrink-perturb":
        raise SystemExit("--br-perturb-frac only applies to --br-init shrink-perturb")
    if args.br_perturb_frac is not None:
        perturb_frac = args.br_perturb_frac
    else:
        perturb_frac = 0.5
    learner_config = learner_config.replace(
        br_target_ckpt=target,
        ckpt_subdir=os.path.join("br", run_tag),
        br_stop_winrate=stop_winrate,
        br_init=args.br_init,
        br_perturb_frac=perturb_frac,
    )
    root = ckpt_root(learner_config)
    os.makedirs(root, exist_ok=True)
    if checkpoint.most_recent_ckpt_dir(root) is not None:
        if args.br_init != "target":
            # Resuming re-runs the SAME command (start_br.sh's contract),
            # so the init flag rides along on resumes — it applied at
            # first launch and does nothing now. Say so rather than let a
            # DIFFERENT intended init silently resume the old lineage.
            logger.warning(
                "BR subtree %s already holds a checkpoint — resuming it; "
                "--br-init only shapes a FIRST launch (use a fresh "
                "--run-tag for a new init)",
                root,
            )
        return learner_config, "checkpoint", None, f"br-{run_tag}"
    if args.br_init == "scratch":
        return learner_config, "scratch", None, f"br-{run_tag}"
    return learner_config, "params", target, f"br-{run_tag}"


def main(args: argparse.Namespace):
    """Launches one persistent process: the actor pool, the inference
    server, and the learner. Everything after initial setup is driven by a
    single call to Learner.train(), which runs for the life of the
    process."""
    learner_config, load_mode, init_ckpt, job_name = resolve_run_setup(
        get_learner_config(), args
    )
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
    # Every PlayerActor (training and eval) carries history across a game's
    # requests when the flag is on; None is the full-window path.
    history_carry_width = None
    if learner_config.player_actor_history_carry:
        history_carry_width = actor_player_model_config.entity_size

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
    # The CROSS-LINEAGE eval arm (2026-08-29). temp=0.5 is not comparable
    # across the flat-readout rewrite: the hierarchical head divided BOTH
    # levels by temp, so eval sharpened the modality marginal as well as the
    # within-modality choice, and the flat head's single division does not.
    # The two differ by a per-modality reweighting that relatively
    # down-weights the concentrated switch modality -- so the same policy
    # reads as switching more under the new head, which would look like a
    # free improvement and is pure parameterisation. temp=1.0 is identical
    # under both, so it is the arm to compare across the boundary; temp=0.5
    # stays as the within-lineage trend.
    eval_agent_untempered = Agent(
        actor_player_network.apply,
        actor_builder_network.apply,
        gpu_lock=gpu_lock,
        player_head_params=HeadParams(temp=1.0),
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
    # One timing sink shared by every training actor, its env and the
    # server; the learner drains it (actor_stats_log_steps).
    actor_stats = ActorStats()
    inference_server = InferenceServer(
        actor_player_network.apply, gpu_lock=gpu_lock, stats=actor_stats
    )
    inference_server.start()

    logger.info("Loading train state...")
    mode = load_mode
    player_state, builder_state, league, controller_bytes = load_train_state(
        learner_config, player_state, builder_state, mode=mode, ckpt_path=init_ckpt
    )
    player_state = jax.device_put(player_state)
    builder_state = jax.device_put(builder_state)

    pinned_opponent: ParamsContainer | None = None
    if learner_config.br_target_ckpt is not None:
        # BR run: the frozen target becomes this run's sole league member
        # (payoff rows silently drop unregistered participants) and every
        # game is pinned against it via the actor-pool short-circuit.
        pinned_opponent = register_br_target(league, learner_config)
        logger.info(
            "BR run: all games pinned against %s (target step %d)",
            learner_config.br_target_ckpt,
            pinned_opponent.step_count,
        )
    elif mode == "checkpoint":
        # Parent resume: pick up snapshots published by child BR runs
        # since the league was last serialized.
        for imported in import_br_snapshots(league, learner_config):
            logger.info("Imported BR snapshot %s into the league", imported)

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
    elif learner_config.br_target_ckpt is not None:
        # Fresh BR child: join the PARENT's wandb group so its curves sit
        # beside main's (read-only — a child never writes the parent's
        # session file); a parent without a session falls through to a
        # fresh group.
        parent_info = load_wandb_run_info(learner_config.replace(ckpt_subdir=None))
        if parent_info is not None:
            wandb_group = parent_info["group"]
        else:
            wandb_group = f"session-{int(time.time())}"
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
    if learner_config.br_target_ckpt is not None:
        job_type = "br"
    else:
        job_type = "main"
    run_id = resume_run_ids.get(job_name)
    wandb_run = wandb.init(
        project="pokemon-rl",
        group=wandb_group,
        job_type=job_type,
        name=f"{wandb_group}-{job_name}",
        id=run_id,
        # "allow", not "must": resume the run when it still exists
        # server-side, otherwise recreate it under the same id — a
        # wandb-side deletion should never block a training restart.
        resume="allow" if run_id else None,
        tags=[job_type],
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
    # restart reads this to resume this exact run. Keyed by job name and
    # written under THIS run's ckpt root, so a BR child's session file
    # lives in its own subtree and can never clobber the parent's.
    save_wandb_run_info(learner_config, wandb_group, {job_name: wandb_run.id})

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
                        history_carry_width=history_carry_width,
                        pinned_opponent=pinned_opponent,
                        stats=actor_stats,
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
            # The last slot runs untempered, and says so in its thread name --
            # metric identity comes from that name, so the two arms land on
            # separate wandb series.
            untempered = eval_id == len(learner_config.eval_baselines) - 1
            if untempered:
                slot_agent = eval_agent_untempered
                slot_suffix = "-t1"
            else:
                slot_agent = eval_agent
                slot_suffix = ""
            actor = PlayerActor(
                agent=slot_agent,
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
                history_carry_width=history_carry_width,
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
                    name=f"EvalActor-{baseline_name}{slot_suffix}-{eval_id}",
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
        actor_stats=actor_stats,
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
        inference_server.stop()
        _finish_wandb_bounded(wandb_run, exit_code=1 if crashed else 0)

    if crashed:
        logger.error("Training run crashed — see traceback above.")
    else:
        logger.info("Training run complete.")
    _hard_exit(1 if crashed else 0)


def _finish_wandb_bounded(
    wandb_run: wandb.wandb_run.Run, exit_code: int, budget: float = 120.0
) -> None:
    """wandb_run.finish() on a helper thread with a wall-clock budget.
    finish() blocks on the wandb-core service's final sync; when that
    service is gone (a terminal Ctrl-C hits the whole foreground process
    group, core included) the wait has nothing to return to, and the
    main thread sat on it indefinitely on 2026-08-23. A bounded wait
    keeps the exit path reachable; the run then shows Crashed on W&B's
    heartbeat timeout, which is accurate."""

    def _finish() -> None:
        try:
            wandb_run.finish(exit_code=exit_code)
        except Exception:
            logger.warning("wandb_run.finish() failed during shutdown", exc_info=True)

    t = threading.Thread(target=_finish, name="wandb-finish", daemon=True)
    t.start()
    t.join(timeout=budget)
    if t.is_alive():
        logger.warning(
            "wandb_run.finish() did not return within %.0fs — exiting without it",
            budget,
        )


def _hard_exit(code: int, budget: float = 30.0) -> None:
    """Bounded join of every non-daemon thread, then os._exit. The
    interpreter's own exit joins ThreadPoolExecutor workers (non-daemon)
    with no timeout, so one actor thread parked on an un-timed wait held
    the whole process hostage after its checkpoint had landed. Stragglers
    are named in the log so the next such hang is diagnosable; SIGUSR1
    (faulthandler, registered at startup) dumps every thread's stack."""
    deadline = time.monotonic() + budget
    stragglers = []
    for t in threading.enumerate():
        if t is threading.current_thread() or t.daemon:
            continue
        t.join(timeout=max(0.0, deadline - time.monotonic()))
        if t.is_alive():
            stragglers.append(t.name)
    if stragglers:
        logger.warning(
            "%d non-daemon thread(s) still alive at exit: %s — hard-exiting",
            len(stragglers),
            stragglers,
        )
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    # kill -USR1 <pid> prints every thread's stack — the only stack dump
    # that works under yama ptrace_scope=1 without root. To a FILE, not
    # stderr: the 2026-08-23 hangs were exactly the case where stderr's
    # pipe had lost its reader, so a dump to stderr vanished.
    os.makedirs("runtime", exist_ok=True)
    _stacks = open(f"runtime/stacks_{os.getpid()}.log", "a")
    faulthandler.register(signal.SIGUSR1, file=_stacks, all_threads=True, chain=False)
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
    parser.add_argument(
        "--load-mode",
        choices=["scratch", "checkpoint", "params"],
        default=None,
        help="Train-state init mode; default is the LOAD_STATE_MODE env "
        "var, then 'checkpoint'.",
    )
    parser.add_argument(
        "--init-ckpt",
        default=None,
        help="Explicit source checkpoint for checkpoint/params mode "
        "(default: most recent under this run's root). A missing explicit "
        "path fails loudly instead of falling back to scratch.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override config.num_steps: the run's total TRAIN-step budget "
        "(host_step, which resumes carry forward) — idle ticks don't "
        "count, and resuming a completed run needs a larger value.",
    )
    parser.add_argument(
        "--br-target",
        default=None,
        help="Best-response mode: train against this frozen checkpoint — "
        "own checkpoint subtree under br/<tag>, params-mode init from the "
        "target (fresh optimiser) on first launch and a normal resume "
        "after, every game pinned against the target, latest params "
        "published into the parent's players/ dir on every stop.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Name for the BR child dir and wandb run "
        "(default: the target checkpoint's basename).",
    )
    parser.add_argument(
        "--br-init",
        choices=["target", "head-reset", "shrink-perturb", "scratch"],
        default="target",
        help="BR param init on FIRST launch (a resume keeps its subtree): "
        "'target' inherits the frozen target verbatim; 'head-reset' "
        "grafts a fresh action readout onto the inherited trunk (uniform "
        "policy over legal cells at step 0); 'shrink-perturb' "
        "interpolates every player param toward fresh init by "
        "--br-perturb-frac; 'scratch' ignores the target's params.",
    )
    parser.add_argument(
        "--br-perturb-frac",
        type=float,
        default=None,
        help="shrink-perturb interpolation toward fresh init: 0.0 = pure "
        "inherit, 1.0 = pure fresh init (default 0.5).",
    )
    parser.add_argument(
        "--br-winrate",
        type=float,
        default=None,
        help="BR stop condition: end the run once the winrate against the "
        "target clears this (with the min-games reliability floor). "
        "Defaults to 0.7 when --num-steps is not given; whichever of the "
        "two conditions fires first stops the run.",
    )
    args = parser.parse_args()
    main(args)
