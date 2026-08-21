"""The Learner: owns the League, the gpu_lock and the training loop."""

import collections
import gc
import json
import logging
import os
import pickle
import queue
import random
import sys
import threading
import time
from _thread import LockType
from contextlib import nullcontext
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
import wandb.wandb_run
from tqdm import tqdm

import wandb
from rl import checkpoint

from rl.environment.data import (
    STOI,
)
from rl.environment.interfaces import (
    Batch,
    Trajectory,
)
from rl.environment.utils import (
    close_tqdm_bar,
    next_tqdm_position,
)
from rl.model.utils import ParamsContainer
from rl.online.artifact import (
    Porygon2BuilderTrainState,
    Porygon2PlayerTrainState,
    write_checkpoint_components,
)
from rl.online.buffer import BuilderTrajectoryStore, PlayerTrajectoryStore
from rl.online.config import Porygon2LearnerConfig
from rl.online.controllers import PILogController
from rl.online.league import (
    LIVE_KEYS,
    MAIN_KEY,
    League,
    PlayerRef,
)
from rl.online.training.batching import stack_batch
from rl.online.training.run_state import RunState
from rl.online.training.train_step import TRAIN_STEP_JIT, train_step

logger = logging.getLogger(__name__)

# Why a snapshot was added to the league. "dominant" is the healthy path
# (the agent beat its own history); "overdue" means only the frame budget

AddReason = Literal["initial", "dominant", "overdue"]

class OOMGuardTriggered(Exception):
    """Raised by Learner._check_oom_guard when available system RAM drops
    below config.oom_guard_min_available_fraction — a self-monitoring
    safety valve, not a leak fix in itself. A full checkpoint has already
    been written by the time this is raised. rl.online.main stops the
    whole process on this (same as a Ctrl-C interrupt) rather than
    continuing in the same process: freeing Python objects doesn't
    guarantee the OS actually reclaims that memory, so the only way to get
    back to a genuinely clean memory state is a fresh process — the user
    (or whatever supervises this box) needs to relaunch, which will resume
    from the checkpoint this exception carries."""

    def __init__(self, checkpoint_path: str):
        super().__init__(checkpoint_path)
        self.checkpoint_path = checkpoint_path


class Learner:
    """Owns the League, the gpu_lock, the compiled train_step and the one
    live RunState. The MainExploiter/LeagueExploiter populations were
    removed 2026-08-21 — see LESSONS.md 9 for the design and why it never
    ran on this box."""

    def __init__(
        self,
        config: Porygon2LearnerConfig,
        league: League,
        player_state: Porygon2PlayerTrainState,
        builder_state: Porygon2BuilderTrainState,
        main_wandb_run: wandb.wandb_run.Run,
        gpu_lock: LockType | None = None,
        debug: bool = False,
        controller_bytes: bytes | None = None,
        spawn_actor_pool: "Callable[[], None] | None" = None,
    ):
        self.config = config
        self.league = league
        self.gpu_lock = gpu_lock or nullcontext()
        self.debug = debug
        # Lets main.py spin up the actor pool once the run state exists
        # (main.py owns PlayerActor/BuilderActor construction — Learner
        # can't import those without a circular import). None is fine for
        # standalone construction (tests, debug scripts): the run just
        # never gets actors, matching the "nothing passed in means don't
        # wire it up" convention elsewhere in this file.
        self._spawn_actor_pool = spawn_actor_pool

        # train_step's config is a static jit arg, so every field is part
        # of the compile cache key. One Learner, constructed once, holds
        # one config for the life of the process — nothing varies it. A
        # value that DOES vary during a run must not live in it at all; it
        # needs its own traced pytree argument, because retained
        # executables per distinct static value OOM-killed run 1326
        # (LESSONS.md 1).

        self._train_step_jit = train_step if debug else TRAIN_STEP_JIT
        # Shape-lattice fail-fast: every combo compiles at the FIRST batch
        # (_precompile_lattice) so no variant can arrive as a surprise
        # compile mid-run. Process-local by design.
        self._shape_lattice_compiled: bool = False

        self.run_state = self._build_run_state(
            player_state,
            builder_state,
            main_wandb_run,
            controller_bytes=controller_bytes,
        )

        self.done = False

    # --- run-state construction ----------------------------------------------

    def _build_run_state(
        self,
        player_state: Porygon2PlayerTrainState,
        builder_state: Porygon2BuilderTrainState,
        wandb_run: wandb.wandb_run.Run,
        controller_bytes: bytes | None = None,
    ) -> RunState:
        """Builds a fresh RunState around an already-constructed
        player_state/builder_state. Controllers and replay are always
        fresh here; restore_controller_state (below) reinstates their EMAs
        from the checkpoint when there is one."""
        config = self.config
        is_not_randoms = config.smogon_format != "randombattle"
        run_state = RunState(
            wandb_run=wandb_run,
            player_replay=PlayerTrajectoryStore(
                max_size=config.player_replay_buffer_capacity,
                max_reuses=config.player_replay_ratio,
                need_tracking=is_not_randoms,
                name="player",
            ),
            builder_replay=BuilderTrajectoryStore(
                max_size=config.builder_replay_buffer_capacity,
                max_reuses=config.builder_replay_ratio,
                name="builder",
            ),
            player_state=player_state,
            builder_state=builder_state,
            created_at_frame=int(jax.device_get(player_state.frame_count)),
            # Seeded from the state's own (restored) step_count, NOT 0: the
            # league keys snapshots by host_step and get_latest_player picks
            # "newest" as max(key) — a session-local counter restarting at 0
            # made every post-restart add key smaller than the restored
            # league's, so the stale pre-restart ref stayed "latest" forever,
            # frames_passed never reset, and "overdue" fired on every
            # league-management tick (the 2026-08-14 10:15 add storm; also
            # the p_{step:08} snapshot-dir overwrite hazard once the counter
            # caught up).
            host_step=int(jax.device_get(player_state.step_count)),
            replay_pi=PILogController(
                initial_log=float(np.log(config.player_replay_ratio)),
                log_min=float(np.log(config.player_replay_ratio_min)),
                log_max=float(np.log(config.player_replay_ratio_max)),
                kp=config.player_replay_ctrl_kp,
                ki=config.player_replay_ctrl_ki,
            ),
            replay_kl_target=float(config.player_replay_kl_target),
            consumer_progress=tqdm(
                desc="consumer", smoothing=0.1, position=next_tqdm_position()
            ),
            train_progress=tqdm(
                desc="batches", smoothing=0.1, position=next_tqdm_position()
            ),
        )
        run_state.run_gate.set()
        self._restore_controller_state(run_state, controller_bytes)
        self.league.update_live(MAIN_KEY, self._create_params_container(run_state))
        return run_state

    # --- trajectory intake ---------------------------------------------------

    def enqueue_traj(self, traj: Trajectory):
        """Called by actors to push data into the run's
        replay buffer."""
        run_state = self.run_state
        add_cond = run_state.player_replay._add_cv
        with add_cond:
            add_cond.wait_for(lambda: run_state.done or run_state.player_replay.ready_to_add())
            if run_state.done:
                return
            run_state.player_replay.add(traj)

        sample_cond = run_state.player_replay._sample_cv
        with sample_cond:
            sample_cond.notify_all()

    # --- background workers --------------------------------------------------

    def host_to_device_worker(self, run_state: RunState):
        """Background thread to batch data and push to the run's
        own GPU queue."""
        max_burst = 8
        batch_size = self.config.batch_size

        sample_cond = run_state.player_replay._sample_cv
        with sample_cond:
            sample_cond.wait_for(
                lambda: run_state.done
                or run_state.player_replay.is_min_fill_fraction_reached(
                    self.config.replay_buffer_min_fill_fraction
                )
            )

        init_key = jax.random.PRNGKey(random.randint(0, 2**16 - 1))
        while not run_state.done:
            for _ in range(max_burst):
                if run_state.done:
                    break

                sample_cond = run_state.player_replay._sample_cv
                with sample_cond:
                    sample_cond.wait_for(
                        lambda: run_state.done
                        or run_state.player_replay.ready_to_sample(batch_size)
                    )
                    if run_state.done:
                        break
                    batch = run_state.player_replay.sample(batch_size)

                # Normalise the exploration-ladder tag every trajectory
                # carries (explore actors mark theirs explore=True at
                # construction — see PlayerActor; train_step keeps those
                # rows out of the league/builder signals only).
                # Trajectories from before the field was populated stack
                # as False, so the shared train_step jit always sees one
                # pytree structure across batches.
                batch = [
                    t.replace(
                        explore=(
                            np.array([False])
                            if isinstance(t.explore, tuple)
                            else np.asarray(t.explore).reshape(1)
                        )
                    )
                    for t in batch
                ]

                add_cond = run_state.player_replay._add_cv
                with add_cond:
                    add_cond.notify_all()

                run_state.consumer_progress.update(batch_size)

                init_key, batch_key = jax.random.split(init_key)
                stacked = stack_batch(
                    batch,
                    rng_key=batch_key,
                    lattice=self.config.player_shape_lattice,
                )
                while not run_state.done:
                    try:
                        run_state.device_q.put(stacked, timeout=1.0)
                        break
                    except queue.Full:
                        continue

        logger.info("host_to_device_worker exiting.")

    def _wandb_log_worker(self, run_state: RunState):
        """Background thread: drains log dicts for the run,
        paying the device->host transfer and wandb serialization here so
        the train loop never has to synchronize with the GPU per step. A
        single consumer preserves wandb step ordering. Also hosts the replay-ratio controller, which
        needs exactly the host-side per-step logs this thread already
        produces."""
        while True:
            logs = run_state.log_q.get()
            if logs is None:
                break
            try:
                host_logs = jax.device_get(logs)
                self._update_replay_controller(run_state, host_logs)
                run_state.wandb_run.log(host_logs)
            except Exception:
                logger.exception("wandb logging failed")

    def _checkpoint_writer_worker(self, run_state: RunState):
        """Background thread: does the actual checkpoint disk I/O so the
        training loop never blocks on it. Payloads are already fully
        host-side and pre-serialized by
        the time they're queued (see _handle_periodic_tasks) — this thread
        never touches a live device buffer or mutates self.league
        directly, only writes what it was handed."""
        while True:
            payload = run_state.ckpt_q.get()
            if payload is None:
                break
            try:
                write_checkpoint_components(
                    payload["save_path"],
                    payload["learner_config"],
                    payload["player_components"],
                    payload["builder_components"],
                    payload["league_bytes"],
                    payload["controller_bytes"],
                    step_count=payload["step_count"],
                    frame_count=payload["frame_count"],
                )
            except Exception:
                logger.exception(
                    "Background checkpoint write failed @ "
                    "step %s — the next periodic checkpoint will simply try "
                    "again.",
                    payload.get("step_count"),
                )

    # --- controller state ----------------------------------------------------

    def controller_state_bytes(self, run_state: RunState) -> bytes:
        """Host-side training dynamics for the checkpoint. Every adaptive
        controller this project built has since been removed (LESSONS.md
        §10), so what is left is the monotonic x-axis counter — but the
        section-wise shape is kept: it is what lets a checkpoint written by
        a superseded revision resume without failing."""
        state = {
            # Monotonic per-run x-axis counter (see RunState.
            # lifetime_step) — restored so charts never rewind at a resume.
            "lifetime_step": run_state.lifetime_step,
        }
        return pickle.dumps(state)

    def _restore_controller_state(
        self, run_state: RunState, data: bytes | None
    ) -> None:
        """Counterpart to controller_state_bytes. Missing sections (older
        checkpoints, or a controller since removed) are simply skipped.

        Never fatal: this state only saves a controller some re-warmup, so
        a blob written by a superseded revision must not be able to fail a
        resume."""
        if not data:
            return
        try:
            state = pickle.loads(data)
        except Exception:
            logger.exception("controller state unreadable — starting fresh")
            return
        # Pre-lifetime_step checkpoints fall back to host_step, which is
        # exact: the counter only ever advances with training.
        run_state.lifetime_step = int(state.get("lifetime_step", run_state.host_step))
        # Checkpoints written before the controller removals (entropy_ctrl
        # 2026-08-13; lambda_ctrl/exploit_ctrl 2026-08-14) carry those
        # sections — simply never read, same as any other extra section.

    # No _update_hyper_controllers anymore: the magnet KL coef became a
    # fixed config scalar when the AdaptivityController was removed
    # (2026-08-13), and the advantage lambda went with the
    # LambdaGapController (2026-08-14) — UPGO's per-step cut plus the
    # fixed player_lambda replaced it (see targets.py). The replay
    # reuse-cap controller below is the one remaining per-log-tick loop.

    def _update_replay_controller(self, run_state: RunState, host_logs: dict) -> None:
        """Velocity-form PI loop holding the replayed-batch actor KL at
        run_state.replay_kl_target by adjusting the reuse
        cap."""
        config = self.config
        if not config.player_replay_ctrl_enabled:
            return
        # The _own variant excludes tempered explore rows (which train the
        # policy since 2026-08-17 but would inflate the mean KL and make
        # the controller silently cut the reuse cap).
        kl = host_logs.get("player_learner_actor_forward_kl_own")
        if kl is not None and np.isfinite(kl):
            run_state.replay_ctrl_kl_sum += float(kl)
            run_state.replay_ctrl_kl_count += 1

        if run_state.replay_ctrl_kl_count >= config.player_replay_ctrl_interval:
            kl_mean = run_state.replay_ctrl_kl_sum / run_state.replay_ctrl_kl_count
            run_state.replay_ctrl_kl_sum = 0.0
            run_state.replay_ctrl_kl_count = 0

            err = (run_state.replay_kl_target - kl_mean) / run_state.replay_kl_target
            run_state.replay_pi.step(err)

            cap = int(round(np.exp(run_state.replay_pi.log)))
            if cap != run_state.player_replay.max_reuses:
                run_state.player_replay.set_max_reuses(cap)

            adds = run_state.player_replay.total_adds
            samples = run_state.player_replay.total_samples
            delta_adds = adds - run_state.replay_ctrl_prev_adds
            delta_samples = samples - run_state.replay_ctrl_prev_samples
            run_state.replay_ctrl_prev_adds = adds
            run_state.replay_ctrl_prev_samples = samples
            if delta_adds > 0:
                run_state.replay_realised_ratio = delta_samples / delta_adds

        host_logs["player_replay_max_reuses"] = float(run_state.player_replay.max_reuses)
        host_logs["player_replay_realised_ratio"] = run_state.replay_realised_ratio

    # --- scheduler -----------------------------------------------------------

    def _ready_run_state(self) -> RunState | None:
        """The run state if it is ready to train this tick, else None:
        warm-enough replay buffer, a batch already on device.

        The .empty() peek is race-free for our purposes: this (the train
        loop) is the sole consumer of the device_q, so an observed
        non-empty queue can't be emptied by anyone else before train()
        collects it."""
        run_state = self.run_state
        if (
            run_state is not None
            and run_state.player_state is not None
            and run_state.player_replay.is_min_fill_fraction_reached(
                self.config.replay_buffer_min_fill_fraction
            )
            and not run_state.device_q.empty()
        ):
            return run_state
        return None

    def train(self):
        """Training loop. Each tick: check readiness (_ready_run_state),
        pull one batch from the device_q, train via the compiled train_step
        under the gpu_lock, run the periodic tasks. The actor pool runs
        continuously and independently."""
        for run_state in (self.run_state,):
            self._start_workers(run_state)

        try:
            for _ in range(self.config.num_steps):
                if self.done:
                    break
                run_state = self._ready_run_state()
                if run_state is None:
                    # Nothing has a warm-enough replay buffer yet (e.g. at
                    # process start, before main's own buffer fills), or
                    # the device_q is momentarily
                    # empty — brief wait rather than a busy spin.
                    threading.Event().wait(timeout=0.1)
                    continue

                try:
                    # Never blocks: _ready_run_state only returns after
                    # observing a batch, and this thread
                    # is the sole consumer of every device_q.
                    batch = run_state.device_q.get_nowait()
                except queue.Empty:
                    continue
                with self.gpu_lock:
                    batch = jax.device_put(batch)
                    logs = self._train_step(run_state, batch)

                run_state.host_step += 1
                run_state.lifetime_step += 1
                run_state.frames_trained_total = (
                    int(jax.device_get(run_state.player_state.frame_count))
                    - run_state.created_at_frame
                )
                self._handle_periodic_tasks(run_state, run_state.host_step, logs)

        except KeyboardInterrupt:
            # One synchronous full save so a deliberate restart loses
            # nothing since the last periodic checkpoint.
            logger.info("Keyboard interrupt received. Saving checkpoint...")
            run_state = self.run_state
            try:
                self._write_checkpoint(run_state, synchronous=True)
            except RuntimeError:
                logger.exception(
                    "Skipping interrupt checkpoint: train state was donated "
                    "mid-step. Latest periodic checkpoint is unaffected."
                )
            raise
        except Exception:
            # logger.exception, NOT traceback.print_exc(): the logging
            # handler routes through tqdm.write(), so the traceback prints
            # cleanly above the progress bars — print_exc() wrote raw to
            # stderr and got shredded line-by-line into the concurrent bar
            # redraws (session 1786537634's OOM traceback was near-
            # unreadable in the captured console for exactly this reason).
            logger.exception("Learner training crashed")
            raise
        finally:
            self.done = True
            for run_state in (self.run_state,):
                # strict=False: process is exiting — a straggler here is
                # tolerable (daemon threads die with the process), and
                # raising would mask the real outcome, turning e.g. a
                # clean Ctrl-C into a crash. Resets keep strict=True.
                self._stop_workers(run_state, strict=False)
            tqdm.write("Training Finished.")

    def register_actor_threads(
        self, threads: list[threading.Thread]
    ) -> None:
        """Called by main.py right after it constructs and starts a
        run's PlayerActor/BuilderActor pool (in response to the
        spawn_actor_pool callback, on creation, or after a reset) — Learner
        can't spawn these itself without a circular import. Registering
        them here means a shutdown waits for (and
        straggler-checks) them exactly like the 3 internal workers,
        instead of silently leaving them running against now-stale state."""
        self.run_state.actor_threads.extend(threads)

    def _start_workers(self, run_state: RunState) -> None:
        transfer_thread = threading.Thread(
            target=self.host_to_device_worker,
            args=(run_state,),
            daemon=True,
            name="transfer",
        )
        transfer_thread.start()
        log_thread = threading.Thread(
            target=self._wandb_log_worker,
            args=(run_state,),
            daemon=True,
            name="log",
        )
        log_thread.start()
        ckpt_thread = threading.Thread(
            target=self._checkpoint_writer_worker,
            args=(run_state,),
            daemon=True,
            name="ckpt",
        )
        ckpt_thread.start()
        run_state.worker_threads.extend([transfer_thread, log_thread, ckpt_thread])

    def _stop_workers(
        self, run_state: RunState, strict: bool = True
    ) -> None:
        run_state.done = True
        run_state.stop_signal[0] = True
        # Wake actors idling at the block gate so they observe stop_signal
        # immediately instead of on their next wait() timeout.
        run_state.run_gate.set()
        try:
            run_state.device_q.get_nowait()
        except queue.Empty:
            pass
        for cond in (
            run_state.player_replay._add_cv,
            run_state.player_replay._sample_cv,
            run_state.builder_replay._add_cv,
            run_state.builder_replay._sample_cv,
        ):
            with cond:
                cond.notify_all()

        for t in run_state.worker_threads:
            if t.name.startswith("transfer-"):
                t.join(timeout=10)
        run_state.log_q.put(None)
        for t in run_state.worker_threads:
            if t.name.startswith("log-"):
                t.join(timeout=30)
        run_state.ckpt_q.put(None)
        for t in run_state.worker_threads:
            if t.name.startswith("ckpt-"):
                t.join(timeout=60)
        # External actor threads (main.py's PlayerActor/BuilderActor pool,
        # registered via register_actor_threads): already signalled via
        # run_state.stop_signal[0] above — just wait for them here.
        for t in run_state.actor_threads:
            t.join(timeout=30)

        all_threads = run_state.worker_threads + run_state.actor_threads
        stragglers = [t for t in all_threads if t.is_alive()]
        if stragglers:
            logger.warning(
                "%d worker thread(s) did not stop within "
                "their join timeout: %s — giving a 30s grace period before "
                "treating this as a hung shutdown.",
                len(stragglers),
                [t.name for t in stragglers],
            )
            for t in stragglers:
                t.join(timeout=30)
            stragglers = [t for t in stragglers if t.is_alive()]
        if stragglers:
            if strict:
                raise RuntimeError(
                    f"{len(stragglers)} worker thread(s) never stopped: "
                    f"{[t.name for t in stragglers]}. Refusing to proceed with "
                    "training state still reachable from a live thread."
                )
            # strict=False is the whole-PROCESS shutdown path (train()'s
            # finally, incl. Ctrl-C): the straggler raise exists to stop a
            # rebuild from starting on top of state a leaked
            # thread still holds (the 2026-08-11 RAM/VRAM leak) — at
            # process exit there is no next phase to protect, every
            # thread is a daemon that dies with the process, and raising
            # here would convert a clean Ctrl-C into a "crashed" outcome
            # (an actor blocked on the game-server websocket mid-game is
            # normal at this point, not a leak).
            logger.warning(
                "%d thread(s) still alive at process "
                "shutdown: %s — proceeding; they are daemons and exit "
                "with the process.",
                len(stragglers),
                [t.name for t in stragglers],
            )

        # Return the 4 progress-bar rows to the shared pool
        # (close_tqdm_bar) so the replacement fork reuses the same rows —
        # without this, every rebuild leaked 4 dead rows and
        # pushed all live bars one screen-row further down, unboundedly,
        # for the life of the process. Closing is safe against any update
        # racing in from a straggler: tqdm's close() flips .disable, which
        # every update() checks first.
        for bar in (
            run_state.consumer_progress,
            run_state.train_progress,
            run_state.player_replay._progress,
            run_state.builder_replay._progress,
        ):
            close_tqdm_bar(bar)

    # Known python-thread name prefixes, for the census below — anything
    # unrecognized lands in "other".
    _THREAD_NAME_BUCKETS = (
        "Selfplay-",
        "BuilderActor-",
        "EvalActor-",
        "transfer-",
        "log-",
        "ckpt-",
        "inference-server",
        "ThreadPoolExecutor",
    )

    def _log_memory_diagnostics(self, logs: dict) -> None:
        """Process-wide RAM attribution, riding main's periodic wandb logs
        every memory_diag_interval steps.

        Motivated by session 1786537634: RSS climbed 5.9GB -> 17GB while
        the OS thread count grew 478 -> 775 with no obvious owner, and
        none of it was attributable from wandb alone. The bounded-by-
        design consumers (replay buffers, league opponent cache) get
        exact byte counts here; the thread census separates python
        threads (named, bucketed below) from native ones — if
        diag_os_threads far exceeds diag_py_threads, the growth lives in
        native pools (XLA/CUDA/websocket internals), not python code."""
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        logs["diag_rss_mb"] = int(line.split()[1]) / 1024.0
                    elif line.startswith("Threads:"):
                        logs["diag_os_threads"] = int(line.split()[1])
        except Exception:
            pass  # non-Linux — skip, same posture as _available_memory_fraction

        py_threads = threading.enumerate()
        logs["diag_py_threads"] = len(py_threads)
        buckets = dict.fromkeys(self._THREAD_NAME_BUCKETS, 0)
        buckets["other"] = 0
        for t in py_threads:
            for prefix in self._THREAD_NAME_BUCKETS:
                if t.name.startswith(prefix):
                    buckets[prefix] += 1
                    break
            else:
                buckets["other"] += 1
        for prefix, count in buckets.items():
            key = prefix.rstrip("-").lower().replace("-", "_")
            logs[f"diag_py_threads_{key}"] = count

        # Heap census: attributes host RSS the byte-exact counters below
        # (replay buffers, league cache) don't cover — e.g. the ~3GB the
        # 2026-08-18 fork jump left unexplained by thread counts
        # and league cache alone. sys.getsizeof is shallow (a dict/list's
        # own overhead, not its contents), but that's exactly what surfaces
        # a genuine culprit: a huge COUNT of one type (numpy arrays, proto
        # objects, EnvironmentState instances) dominating aggregate bytes.
        # Logged to the console, not wandb — dynamic top-N, not a stable
        # scalar key. gc.get_objects() walks the whole heap, so this rides
        # the same 5000-step cadence as the rest of this function.
        try:
            counts = collections.Counter()
            sizes = collections.Counter()
            for obj in gc.get_objects():
                t = type(obj).__name__
                counts[t] += 1
                sizes[t] += sys.getsizeof(obj)
            top = sizes.most_common(15)
            logger.info(
                "Heap census (top-15 by approx shallow size): %s",
                ", ".join(f"{t}={counts[t]}objs/{sz / 2**20:.1f}MB" for t, sz in top),
            )
        except Exception:
            logger.exception("Heap census failed")

        run_state = self.run_state
        logs["diag_player_replay_mb"] = run_state.player_replay.nbytes() / 2**20
        logs["diag_builder_replay_mb"] = run_state.builder_replay.nbytes() / 2**20
        try:
            with open("runtime/service_memory.json") as f:
                node_stats = json.load(f)
            # Service writes every 10s; 60s is a generous staleness bound
            # in case the file is left over from a service that's since died.
            if time.time() - node_stats["ts"] < 60:
                for key in (
                    "rss_mb",
                    "heap_used_mb",
                    "num_workers",
                    "worker_heap_used_mb",
                    "workers_reported",
                ):
                    logs[f"diag_node_{key}"] = node_stats[key]
        except Exception:
            pass  # service not up, stats file stale/absent, or race on the rename

        entries, cache_bytes = self.league.cache_stats()
        logs["diag_league_cache_entries"] = entries
        logs["diag_league_cache_mb"] = cache_bytes / 2**20

    def _precompile_lattice(self, run_state: RunState, batch: Batch) -> None:
        """Fail-fast compilation of EVERY lattice combo at the first
        batch, so a shape variant can never arrive as a surprise compile
        mid-run (the exact mechanism that OOM'd the geometric-bucket
        sessions ~20min in: the first top-bucket batch). Each combo is
        exercised through the real jit with a resized copy of the first
        real batch and a COPY of the train states (the jit donates its
        state args; outputs are discarded), so the dispatch cache is
        warm and any compile-time OOM happens at launch, before hours of
        training are at stake. Runs under the caller's gpu_lock.

        Resizing pads the T axis by repeating each chunk's final row —
        the actor's own padding convention, so no all-invalid mask rows
        are fabricated — and pads/slices the H axes with zeros (zero
        history rows are ordinary invalid steps)."""
        lattice = tuple(self.config.player_shape_lattice)
        full = (self.config.player_chunk_length, self.config.player_history_length)
        assert lattice[-1] == full, (lattice, full)
        assert all(
            a[0] <= b[0] and a[1] <= b[1] for a, b in zip(lattice, lattice[1:])
        ), f"player_shape_lattice must be an ascending chain: {lattice}"
        assert len(lattice) <= 4, f"lattice too large (memory risk): {lattice}"
        if len(lattice) <= 1:
            return

        def resize_time(x, target):
            if x.shape[0] == target:
                return x
            if x.shape[0] > target:
                return x[:target]
            pad = jnp.repeat(x[-1:], target - x.shape[0], axis=0)
            return jnp.concatenate([x, pad], axis=0)

        def resize_zeros(x, target):
            if x.shape[0] == target:
                return x
            if x.shape[0] > target:
                return x[:target]
            widths = [(0, target - x.shape[0])] + [(0, 0)] * (x.ndim - 1)
            return jnp.pad(x, widths)

        current = (
            batch.player_transitions.env_output.done.shape[0],
            batch.player_history.field.shape[0],
        )
        for t_c, h_c in lattice:
            if (t_c, h_c) == current:
                continue  # the real call right after this compiles it
            resized = batch.replace(
                player_transitions=jax.tree.map(
                    lambda x: resize_time(x, t_c), batch.player_transitions
                ),
                player_history=jax.tree.map(
                    lambda x: resize_zeros(x, h_c), batch.player_history
                ),
                player_packed_history=jax.tree.map(
                    lambda x: resize_zeros(x, 2 * h_c), batch.player_packed_history
                ),
            )
            logger.info("Precompiling train_step shape combo (%d, %d)…", t_c, h_c)
            start = time.time()

            # True buffer copies with identical avals: the jit donates its
            # state args, so passing the live states would free them, and
            # a leaf-type-changing copy (e.g. jnp.copy on a weak-typed
            # python scalar) would trace as yet another variant.
            def copy_state(tree):
                return jax.tree.map(
                    lambda x: (
                        jnp.array(x, copy=True) if isinstance(x, jax.Array) else x
                    ),
                    tree,
                )

            self._train_step_jit(
                copy_state(run_state.player_state),
                copy_state(run_state.builder_state),
                resized,
                self.config,
            )
            logger.info(
                "Compiled (%d, %d) in %.1fs.", t_c, h_c, time.time() - start
            )

    def _train_step(self, run_state: RunState, batch: Batch) -> dict:
        """Runs the JAX update, rebinding the result onto run_state."""
        if not self._shape_lattice_compiled:
            self._precompile_lattice(run_state, batch)
            self._shape_lattice_compiled = True
        run_state.player_state, run_state.builder_state, logs = self._train_step_jit(
            run_state.player_state,
            run_state.builder_state,
            batch,
            self.config,
        )
        return logs

    def _handle_periodic_tasks(self, run_state: RunState, step: int, logs: dict):
        """Handles logging, progress bars, and checkpointing for run_state."""
        run_state.train_progress.update(1)

        if (
            self.config.smogon_format != "randombattle"
            and step % self.config.save_interval_steps == 0
        ):
            logs.update(self._get_usage_counts(run_state))

        if step % self.config.league_winrate_log_steps == 0:
            logs.update(self._get_league_winrates(run_state))
            logs.update(self._get_league_winrate_heatmap(run_state))

        if (
            self.config.memory_diag_interval > 0
            and step % self.config.memory_diag_interval == 0
        ):
            self._log_memory_diagnostics(logs)

        # The default x-axis for every metric on the run
        # (wandb.define_metric in main.py): monotonic across resumes AND
        # attempt re-forks, unlike host_step/frames.
        logs["lifetime_step"] = run_state.lifetime_step
        run_state.log_q.put(logs)

        # PlayerActor.pull_own_player() reads league.get_live(MAIN_KEY),
        # so this is what makes the actors play the CURRENT policy rather
        # than the one they were started with.
        if step % self.config.main_player_update_steps == 0:
            self.league.update_live(MAIN_KEY, self._create_params_container(run_state))

        if step % self.config.save_interval_steps == 0:
            self._write_checkpoint(run_state)

        if step % self.config.manage_league_interval == 0:
            self._manage_league(run_state, step)

        self._check_oom_guard(run_state, step)

    def _write_checkpoint(self, run_state: RunState, synchronous: bool = False) -> str:
        """Writes the full resumable state: params, target_params,
        opt_state, host counters, the serialized League and the controller
        blob, keyed to the run's own step_count.

        Everything host-side/fast happens synchronously here (device
        pulls, small in-memory serializations); only the actual disk
        write goes to run_state's own background writer, via a payload that's
        already fully host-side (plain dicts/bytes/ints, never a live
        TrainState or the live League object itself). synchronous=True
        (Ctrl-C/OOM-guard path) writes inline instead, since there may be
        no time left for the background writer to run."""
        host_player_state = jax.device_get(run_state.player_state)
        host_builder_state = jax.device_get(run_state.builder_state)
        player_components = dict(
            params=host_player_state.params,
            target_params=host_player_state.target_params,
            opt_state=host_player_state.opt_state,
            scalars=dict(
                step_count=host_player_state.step_count,
                frame_count=host_player_state.frame_count,
            ),
        )
        builder_components = dict(
            params=host_builder_state.params,
            target_params=host_builder_state.target_params,
            opt_state=host_builder_state.opt_state,
            scalars=dict(
                step_count=host_builder_state.step_count,
                frame_count=host_builder_state.frame_count,
            ),
        )
        save_path = os.path.abspath(
            os.path.join(
                f"./ckpts/gen{self.config.generation}",
                f"ckpt_{int(np.asarray(host_player_state.step_count)):08}",
            )
        )
        payload = dict(
            save_path=save_path,
            learner_config=self.config,
            player_components=player_components,
            builder_components=builder_components,
            league_bytes=self.league.serialize(),
            controller_bytes=self.controller_state_bytes(run_state),
            step_count=int(np.asarray(host_player_state.step_count)),
            frame_count=int(np.asarray(host_player_state.frame_count)),
        )
        if synchronous:
            return write_checkpoint_components(
                payload["save_path"],
                payload["learner_config"],
                payload["player_components"],
                payload["builder_components"],
                payload["league_bytes"],
                payload["controller_bytes"],
                step_count=payload["step_count"],
                frame_count=payload["frame_count"],
            )
        run_state.ckpt_q.put(payload)
        return save_path

    def _manage_league(self, run_state: RunState, step: int):
        """Checks whether a new snapshot should be added to the league."""
        reason = self._should_add_new_player(run_state)
        if reason is not None:
            tqdm.write(f"Adding new player to league @ {step} ({reason})")
            self._add_player_to_league(run_state, step, origin="main")
            run_state.player_replay.reset_usage_counts()

    def _available_memory_fraction() -> float | None:
        """Fraction of total system RAM currently available (reclaimable
        caches counted as available, matching what actually predicts an
        OOM kill), or None if it can't be determined (non-Linux, or
        /proc/meminfo unreadable) — the caller treats None as "skip the
        check", the same defensive posture as this codebase's other
        optional-environment guards (e.g. the matplotlib import)."""
        try:
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    key, value = line.split(":", 1)
                    meminfo[key] = int(value.strip().split()[0])  # kB
            return meminfo["MemAvailable"] / meminfo["MemTotal"]
        except Exception:
            return None

    def _check_oom_guard(self, run_state: RunState, step: int) -> None:
        """Self-monitoring safety valve, not a leak fix: if available RAM
        drops below config.oom_guard_min_available_fraction, save a
        full checkpoint now and raise OOMGuardTriggered — better to stop on
        our own terms with a guaranteed-complete checkpoint than let the
        kernel's OOM killer pick an arbitrary moment (possibly mid-write)
        to SIGKILL this process."""
        if (
            not self.config.oom_guard_enabled
            or step % self.config.oom_guard_check_interval != 0
        ):
            return
        available_fraction = self._available_memory_fraction()
        if (
            available_fraction is not None
            and available_fraction < self.config.oom_guard_min_available_fraction
        ):
            logger.warning(
                "Available memory fraction %.3f < oom_guard_min_available_fraction "
                "%.3f @ step %d — saving a checkpoint and "
                "stopping before the kernel OOM-kills this process.",
                available_fraction,
                self.config.oom_guard_min_available_fraction,
                step,
            )
            save_path = self._write_checkpoint(
                self.run_state, synchronous=True
            )
            raise OOMGuardTriggered(save_path)

    # (_measure_exploitability/_update_exploit_controller/_apply_exploit_
    # scale removed 2026-08-14 with the ExploitabilityController — the
    # worst-matchup win-rate signal still exists in _should_add_new_player's
    # "dominant" gate; it just doesn't actuate anything anymore.)

    def _should_add_new_player(self, run_state: RunState) -> AddReason | None:
        """Returns why a snapshot should join the league, or None to skip.
        main only."""
        # Pacing is measured against main's OWN last checkpoint (AlphaStar
        # MainPlayer.ready_to_checkpoint: steps since self._checkpoint_step),
        # not the league's newest entry — a foreign-origin publication
        # would otherwise become "latest" permanently (its offset key wins
        # max()) with a frame count that never advances, firing an overdue
        # add on every league-management tick.
        latest = self.league.get_latest_player(origin="main")
        current = self.league.get_live(MAIN_KEY)

        latest_frames = latest.player_frame_count if latest is not None else 0
        frames_passed = int(current.player_frame_count - latest_frames)

        if frames_passed < self.config.add_player_min_frames:
            return None

        historical_players = [
            v for k, v in self.league.players.items() if k not in LIVE_KEYS
        ]

        if not historical_players:
            if (
                int(run_state.player_state.step_count)
                > self.config.minimum_historical_player_steps
            ):
                return "initial"
            return None

        win_rates = self.league.get_winrate((current, historical_players))

        if win_rates.min() > 0.7:
            return "dominant"
        if frames_passed >= self.config.add_player_max_frames:
            return "overdue"
        return None

    def _create_params_container(self, run_state: RunState) -> ParamsContainer:
        return ParamsContainer(
            player_frame_count=jax.device_get(run_state.player_state.frame_count).item(),
            builder_frame_count=jax.device_get(run_state.builder_state.frame_count).item(),
            step_count=MAIN_KEY,
            player_params=jax.device_get(run_state.player_state.params),
            builder_params=jax.device_get(run_state.builder_state.params),
        )

    def _add_player_to_league(
        self, run_state: RunState, step: int, origin: str = "main"
    ):
        """Persist the current params as an opponent snapshot and register
        a ref. Only the params files are written (no optimiser state); the
        league holds the lightweight ref and materialises the params
        lazily when this player is actually drawn as an opponent."""
        league_step = step
        players_root = f"./ckpts/gen{self.config.generation}/players"
        snapshot_dir = os.path.abspath(f"{players_root}/p_{league_step:08}")
        checkpoint.save_param_snapshot(
            snapshot_dir,
            player_components=dict(
                params=jax.device_get(run_state.player_state.params),
                target_params=jax.device_get(run_state.player_state.target_params),
            ),
            builder_components=dict(
                params=jax.device_get(run_state.builder_state.params),
                target_params=jax.device_get(run_state.builder_state.target_params),
            ),
        )
        self.league.add_player(
            PlayerRef(
                step_count=league_step,
                snapshot_dir=snapshot_dir,
                player_frame_count=jax.device_get(run_state.player_state.frame_count).item(),
                builder_frame_count=jax.device_get(
                    run_state.builder_state.frame_count
                ).item(),
                player_key="params",
                builder_key="params",
                origin=origin,
            )
        )

    def _get_usage_counts(self, run_state: RunState):
        result = {}
        for key, counts in [
            ("species", run_state.player_replay._species_counts),
            ("items", run_state.player_replay._item_counts),
            ("abilities", run_state.player_replay._ability_counts),
            ("moves", run_state.player_replay._move_counts),
        ]:
            names = list(STOI[key])
            table = wandb.Table(columns=[key, "usage"])
            for name, count in zip(names, counts):
                table.add_data(name, count)
            result[f"{key}_usage"] = table
        return result

    def _winrate_tracked_opponents(self) -> list[PlayerRef]:
        """Every historical league member."""
        return [v for k, v in self.league.players.items() if k not in LIVE_KEYS]

    @staticmethod
    def _ref_label(ref: PlayerRef) -> str:
        """Payoff-table label: the snapshot's own step count."""
        return f"{ref.step_count}"

    def _get_league_winrates(self, run_state: RunState):
        current = self.league.get_live(MAIN_KEY)
        others = self._winrate_tracked_opponents()
        if not others:
            return {}
        win_rates = self.league.get_winrate((current, others))
        # Origin-labelled keys ("league_main_v_ME-1834_winrate") still
        # match scripts/wandb_views.py's ^league_main_v_.*_winrate$ panel
        # regex.
        return {
            f"league_main_v_{self._ref_label(others[i])}_winrate": wr
            for i, wr in enumerate(win_rates)
        }

    def _get_league_winrate_heatmap(self, run_state: RunState):
        """Full pairwise win-rate matrix over the whole shared payoff
        table: live main and every historical snapshot (when they
        exist), and every historical snapshot with an origin-labelled
        row — logged through a custom Vega-Lite chart preset
        (jtwin/league-payoff-heatmap-v10, registered once via
        scripts/register_wandb_charts.py) instead of hijacking wandb's
        confusion-matrix preset: proper axis titles (player/opponent, not
        Actual/Predicted), a red/gold/green win-rate colour band per
        cell, and a text label per cell. The colour is a chain of
        condition/value tests on winrate with NO field bound directly to
        the colour channel — every version that bound colour to a table
        field (scale.range, scale.scheme+domain+clamp, a literal
        per-cell hex column with scale: null) rendered as either an
        unrelated colour or one flat colour for every cell in wandb's
        actual custom-chart panel, confirmed via wandb's own GraphQL API
        (spec stored correctly) and a neutral Vega-Lite renderer (spec
        renders correctly outside wandb) — so wandb's Vega2 runtime does
        not honour a field-bound colour channel here. Condition/value
        (no field) is the one pattern proven to render correctly (the
        text mark's black/white choice used exactly this pattern the
        whole time). Interactive (hover shows exact values), no
        matplotlib figure render on the train-loop thread, no image
        upload per log. row_idx/col_idx carry insertion order so the
        chart's ordinal axes sort by league structure rather than
        wandb's default alphabetical sort. A pair that has never actually
        played just shows the table's prior."""
        current = self.league.get_live(MAIN_KEY)
        others = self._winrate_tracked_opponents()
        if not others:
            return {}

        all_players = [current] + others
        labels = ["main (live)"] + [self._ref_label(p) for p in others]
        matrix = np.asarray(self.league.get_winrate((all_players, all_players)))

        table = wandb.Table(
            columns=["row", "row_idx", "col", "col_idx", "winrate"],
            data=[
                [row, i, col, j, float(matrix[i, j])]
                for i, row in enumerate(labels)
                for j, col in enumerate(labels)
            ],
        )
        chart = wandb.plot_table(
            "jtwin/league-payoff-heatmap-v10",
            table,
            fields={
                "row": "row",
                "row_idx": "row_idx",
                "col": "col",
                "col_idx": "col_idx",
                "winrate": "winrate",
            },
            string_fields={
                "title": "league payoff table (row beats column)"
            },
        )
        return {"league_winrate_heatmap": chart}
