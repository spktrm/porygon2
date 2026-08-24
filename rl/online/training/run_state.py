"""The training run's mutable state."""

import dataclasses
import logging
import queue
import threading
from typing import Literal

import wandb.wandb_run

import wandb
from rl.online.artifact import Porygon2BuilderTrainState, Porygon2PlayerTrainState
from rl.online.buffer import BuilderTrajectoryStore, PlayerTrajectoryStore
from rl.online.training.controllers import PILogController

logger = logging.getLogger(__name__)

# Why a snapshot was added to the league. "dominant" is the healthy path
# (the agent beat its own history); "overdue" means only the frame budget
# expired, which is the plateau signature.
AddReason = Literal["initial", "dominant", "overdue"]


@dataclasses.dataclass
class RunState:
    """The training run's mutable state.

    Kept as a container rather than flattened onto Learner because it
    draws a clean line: everything here is per-lineage and mutable (train
    state, replay stores, controller EMAs, queues, worker threads), while
    Learner owns the process-wide singletons (the League, the gpu_lock,
    the compiled train_step). Was PopulationState until 2026-08-21, when
    the MainExploiter/LeagueExploiter populations were removed and the
    dict-of-one plus its single-inhabitant Literal went with them.
    """

    wandb_run: wandb.wandb_run.Run
    player_replay: PlayerTrajectoryStore
    builder_replay: BuilderTrajectoryStore
    player_state: Porygon2PlayerTrainState | None = None
    builder_state: Porygon2BuilderTrainState | None = None
    created_at_frame: int | None = None
    host_step: int = 0
    device_q: "queue.Queue" = None
    log_q: "queue.Queue" = None
    ckpt_q: "queue.Queue" = None
    # The 3 internal background workers (host_to_device/
    # log/checkpoint) — owned and joined entirely within this file.
    worker_threads: list = dataclasses.field(default_factory=list)
    # The PlayerActor/BuilderActor game-playing threads —
    # constructed and started by main.py (Learner can't import
    # player_actor.py/builder_actor.py without a circular import, since
    # both already import Learner), registered here via
    # Learner.register_actor_threads so a shutdown or reset waits for them
    # too, not just the 3 internal workers.
    actor_threads: list = dataclasses.field(default_factory=list)
    stop_signal: list = dataclasses.field(default_factory=lambda: [False])
    done: bool = False
    # Actor gate: set = the actor threads may play games. Held open for
    # the whole run; kept because the actor threads wait on it between
    # games and shutdown relies on that same wait.
    run_gate: "threading.Event" = None
    replay_pi: PILogController | None = None
    # Fixed at config.player_replay_kl_target — the ExploitabilityController
    # that used to scale it was removed 2026-08-14 (last of the adaptive
    # hyperparameter loops; see rl/online/controllers.py's module docstring).
    replay_kl_target: float = 0.045
    replay_ctrl_kl_sum: float = 0.0
    replay_ctrl_kl_count: int = 0
    replay_ctrl_prev_adds: int = 0
    replay_ctrl_prev_samples: int = 0
    replay_realised_ratio: float = float("nan")
    # Cumulative frames trained since process start. Telemetry only.
    frames_trained_total: int = 0
    # Monotonic train-tick counter over the WHOLE wandb-run
    # lifetime: restored from the checkpoint's host blob, so it never
    # rewinds or resets across a resume. Logged as "lifetime_step" with every
    # metric and set as the run's default x-axis (wandb.define_metric in
    # main.py) — charts read as cumulative training progress instead of
    # the sawtooth/overdraw that _step (log-call count) and the
    # per-attempt counters produce across resumes and re-forks.
    lifetime_step: int = 0
    consumer_progress: object = None
    train_progress: object = None

    def __post_init__(self):
        if self.device_q is None:
            self.device_q = queue.Queue(maxsize=1)
        if self.log_q is None:
            self.log_q = queue.Queue(maxsize=64)
        if self.ckpt_q is None:
            self.ckpt_q = queue.Queue(maxsize=2)
        if self.run_gate is None:
            self.run_gate = threading.Event()
