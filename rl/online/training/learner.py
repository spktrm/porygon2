"""The Learner: owns the League, the gpu_lock and the training loop."""

import logging
import os
import pickle
import queue
import threading
import time
from _thread import LockType
from contextlib import nullcontext
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import wandb.wandb_run
from tqdm import tqdm

import wandb

from rl.environment.interfaces import (
    Batch,
    Trajectory,
)
from rl.environment.utils import (
    next_tqdm_position,
)
from rl.online.artifact import (
    Porygon2BuilderTrainState,
    Porygon2PlayerTrainState,
    write_checkpoint_components,
)
from rl.online.buffer import BuilderTrajectoryStore, PlayerTrajectoryStore
from rl.online.config import Porygon2LearnerConfig
from rl.online.training.controllers import PILogController
from rl.online.league import (
    MAIN_KEY,
    League,
)
from rl.online.training.diagnostics import (
    available_memory_fraction,
    log_memory_diagnostics,
)
from rl.online.training.league_ops import (
    add_player_to_league,
    create_params_container,
    get_league_winrate_heatmap,
    get_league_winrates,
    get_usage_counts,
    should_add_new_player,
)
from rl.online.training.run_state import RunState
from rl.online.training.train_step import TRAIN_STEP_JIT, train_step
from rl.online.training.workers import start_workers, stop_workers

logger = logging.getLogger(__name__)


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
        self.league.update_live(MAIN_KEY, create_params_container(run_state))
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
            start_workers(run_state, self.config)

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
                stop_workers(run_state, strict=False)
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
            logs.update(get_usage_counts(run_state))

        if step % self.config.league_winrate_log_steps == 0:
            logs.update(get_league_winrates(self.league))
            logs.update(get_league_winrate_heatmap(self.league))

        if (
            self.config.memory_diag_interval > 0
            and step % self.config.memory_diag_interval == 0
        ):
            log_memory_diagnostics(run_state, self.league, logs)

        # The default x-axis for every metric on the run
        # (wandb.define_metric in main.py): monotonic across resumes AND
        # attempt re-forks, unlike host_step/frames.
        logs["lifetime_step"] = run_state.lifetime_step
        run_state.log_q.put(logs)

        # PlayerActor.pull_own_player() reads league.get_live(MAIN_KEY),
        # so this is what makes the actors play the CURRENT policy rather
        # than the one they were started with.
        if step % self.config.main_player_update_steps == 0:
            self.league.update_live(MAIN_KEY, create_params_container(run_state))

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
            reg_params=host_player_state.reg_params,
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
        reason = should_add_new_player(run_state, self.league, self.config)
        if reason is not None:
            tqdm.write(f"Adding new player to league @ {step} ({reason})")
            add_player_to_league(run_state, self.league, self.config, step)
            run_state.player_replay.reset_usage_counts()


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
        available_fraction = available_memory_fraction()
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

