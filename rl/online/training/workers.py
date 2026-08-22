"""The run's three background threads, and the replay-reuse controller.

Free functions over RunState rather than Learner methods: a worker needs
the run's queues and the config, nothing else the Learner owns. That is
also what makes the reuse controller testable without standing up a
Learner.
"""

import logging
import queue
import random
import threading
import time

import jax
import numpy as np

from rl.environment.utils import close_tqdm_bar
from rl.online.artifact import write_checkpoint_components
from rl.online.config import Porygon2LearnerConfig
from rl.online.training.batching import stack_batch
from rl.online.training.run_state import RunState

logger = logging.getLogger(__name__)


def host_to_device_worker(run_state: RunState, config: Porygon2LearnerConfig):
    """Background thread to batch data and push to the run's
    own GPU queue."""
    max_burst = 8
    batch_size = config.batch_size

    sample_cond = run_state.player_replay._sample_cv
    with sample_cond:
        sample_cond.wait_for(
            lambda: run_state.done
            or run_state.player_replay.is_min_fill_fraction_reached(
                config.replay_buffer_min_fill_fraction
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
                lattice=config.player_shape_lattice,
            )
            while not run_state.done:
                try:
                    run_state.device_q.put(stacked, timeout=1.0)
                    break
                except queue.Full:
                    continue

    logger.info("host_to_device_worker exiting.")

def wandb_log_worker(run_state: RunState, config: Porygon2LearnerConfig):
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
            update_replay_controller(run_state, config, host_logs)
            run_state.wandb_run.log(host_logs)
        except Exception:
            logger.exception("wandb logging failed")

def checkpoint_writer_worker(run_state: RunState):
    """Background thread: does the actual checkpoint disk I/O so the
    training loop never blocks on it. Payloads are already fully
    host-side and pre-serialized by
    the time they're queued (see _handle_periodic_tasks) — this thread
    never touches a live device buffer or mutates the League
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
def update_replay_controller(
    run_state: RunState, config: Porygon2LearnerConfig, host_logs: dict
) -> None:
    """Velocity-form PI loop holding the replayed-batch actor KL at
    run_state.replay_kl_target by adjusting the reuse
    cap."""
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
def start_workers(run_state: RunState, config: Porygon2LearnerConfig) -> None:
    transfer_thread = threading.Thread(
        target=host_to_device_worker,
        args=(run_state, config),
        daemon=True,
        name="transfer",
    )
    transfer_thread.start()
    log_thread = threading.Thread(
        target=wandb_log_worker,
        args=(run_state, config),
        daemon=True,
        name="log",
    )
    log_thread.start()
    ckpt_thread = threading.Thread(
        target=checkpoint_writer_worker,
        args=(run_state,),
        daemon=True,
        name="ckpt",
    )
    ckpt_thread.start()
    run_state.worker_threads.extend([transfer_thread, log_thread, ckpt_thread])

def stop_workers(run_state: RunState, strict: bool = True) -> None:
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

    # Every wait below is a SHARED wall-clock budget, not a per-thread
    # timeout: threads that are hung are hung together (they were all
    # signalled at the same instant), so charging each one its own
    # timeout made a shutdown cost N x timeout of pure waiting.
    def join_all(threads, budget: float) -> None:
        deadline = time.monotonic() + budget
        for t in threads:
            t.join(timeout=max(0.0, deadline - time.monotonic()))

    def named(prefix: str) -> list:
        # Names carry no run suffix ("transfer", not "transfer-main");
        # matching on "transfer-" quietly joined nothing and dumped all
        # three workers into the straggler path every single shutdown.
        return [t for t in run_state.worker_threads if t.name == prefix]

    join_all(named("transfer"), 10)
    run_state.log_q.put(None)
    join_all(named("log"), 30)
    run_state.ckpt_q.put(None)
    # The ckpt worker may be mid-write of a full checkpoint — the one
    # wait here that is genuinely about work in flight, not liveness.
    join_all(named("ckpt"), 60)
    # External actor threads (main.py's PlayerActor/BuilderActor pool,
    # registered via register_actor_threads): already signalled via
    # run_state.stop_signal[0] above — just wait for them here. They
    # unwind on their next step or receive poll (~1s), so the budget is
    # short.
    join_all(run_state.actor_threads, 15)

    all_threads = run_state.worker_threads + run_state.actor_threads
    stragglers = [t for t in all_threads if t.is_alive()]
    if stragglers:
        logger.warning(
            "%d worker thread(s) did not stop within "
            "their join timeout: %s — giving a 15s grace period before "
            "treating this as a hung shutdown.",
            len(stragglers),
            [t.name for t in stragglers],
        )
        join_all(stragglers, 15)
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
