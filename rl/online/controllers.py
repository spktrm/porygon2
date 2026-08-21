"""Feedback controllers for training hyperparameters.

All of them follow the replay-ratio controller's design language
(velocity-form PI on a log-scale actuator, EMA-smoothed sensors,
anti-windup via clamping the actuator itself) and those that drive
train_step actuate RUNTIME scalars — never static config, which
recompiles (and whose retained executables OOM-killed run 1326). That
shared shape lives in ``PILogController`` (the actuator: step, bump,
clamp) and ``EmaSensor`` (the sensor: EMA + tick-gating) below; each
hyperparameter controller is then just its error function, bounds, and
whatever hard overrides it needs — the replay reuse-cap loop in
learner.py builds on ``PILogController`` the same way but keeps its own
windowed-mean sensor, since averaging the actor-KL over a fixed window
rather than decaying it is a deliberate, different smoothing choice.

What survives (2026-08-14): exactly ONE continuous controller — the
replay reuse-cap loop in learner.py (actor-KL -> reuse cap), which
actuates a data-pipeline resource, never learning dynamics — plus the
event-driven PlasticityController (learner.py), a discrete state
machine over countable events (N overdue league additions -> one
shrink-and-perturb -> tracked recovery -> cooldown), which is the only
mechanism that ever touches main's weights. This module now holds just
the two shared primitives (PILogController, EmaSensor).

Three controllers were deliberately removed, for the same reason — their
effects proved hard to tune and predict:

- AdaptivityController (magnet KL coef; removed 2026-08-13): the
  commitment-covariance PI caused three separate bugs (unreachable
  target pinning pressure at the ceiling in 1338/1339,
  divide-by-near-zero in 1341, an exploit_ctrl target-scaling bug), and
  its stacked event bumps held main at ~6x baseline pressure for a whole
  run (session 1786537634) with zero floor breaches. The coefficient is
  now exactly config.player_magnet_kl_coef, always. Its entropy sensors
  are still logged; modality collapse (the 1330 failure mode) is
  watched, not auto-corrected.
- LambdaGapController (advantage lambda; removed 2026-08-14): replaced
  by AlphaStar's actual recipe — fixed TD(lambda=0.8) value targets,
  unparameterised v-trace policy advantages, and a UPGO term whose
  per-step outcome-conditional cut supplies locally what the
  controller's single global lambda approximated (see targets.py). Its
  one genuinely-useful behaviour, forcing pure Monte Carlo while a
  freshly-perturbed critic is untrustworthy, survives as "upgo_coef=0
  during plasticity recovery" — itself removed 2026-08-21 with the
  PlasticityController (LESSONS.md 10).
- ExploitabilityController (caution scale; removed 2026-08-14): built to
  scale three other controllers' targets, it outlived all three — by the
  end its only action was a bounded nudge on the replay KL target,
  driven by a slow, prior-dominated worst-matchup win-rate sensor (which
  false-positived in 1338 from lightly-played snapshots reading ~0.5).
  AlphaStar has no analogue. The sensor's signal survives as
  _should_add_new_player's "dominant" gate and the
  league_main_winrate_min auditor; the replay KL target is fixed at
  config.player_replay_kl_target.
"""

import numpy as np


class PILogController:
    """Velocity-form PI control of a scalar actuator held in log space,
    with clamping to [log_min, log_max] as anti-windup: the integral
    term cannot accumulate past the actuator's own bounds.

    This is the mechanism every controller in this module (and the
    replay reuse-cap loop in learner.py) shares. Callers own the sensor,
    the error normalisation (by convention err > 0 must mean "push the
    actuator up"), and the log<->value transform — this class only knows
    about the actuator.
    """

    def __init__(
        self, initial_log: float, log_min: float, log_max: float, kp: float, ki: float
    ):
        self.log_min = log_min
        self.log_max = log_max
        self.kp = kp
        self.ki = ki
        self.log = float(np.clip(initial_log, log_min, log_max))
        self.prev_err = 0.0

    def step(self, err: float) -> None:
        """One PI update from a precomputed, already-normalised error."""
        self.log += self.kp * (err - self.prev_err) + self.ki * err
        self.prev_err = err
        self.log = float(np.clip(self.log, self.log_min, self.log_max))

    def bump(self, delta: float) -> None:
        """Discrete step for a known event (perturbation, new opponent),
        bypassing the PI recurrence entirely."""
        self.log = float(np.clip(self.log + delta, self.log_min, self.log_max))

    def set_log(self, value: float) -> None:
        """Hard set — recovery-to-ceiling, or a checkpoint restore.
        Re-clips, since bounds may not match whoever produced ``value``."""
        self.log = float(np.clip(value, self.log_min, self.log_max))


class EmaSensor:
    """EMA-smoothed sensor with tick-gating: ``observe`` feeds one raw
    reading (ignoring None/non-finite ones), ``ready`` reports whether
    ``interval`` readings have accumulated since the last ``consume``.
    """

    def __init__(self, alpha: float, interval: int):
        self.alpha = alpha
        self.interval = interval
        self.ema: float | None = None
        self.ticks = 0

    def observe(self, reading: float | None) -> None:
        if reading is not None and np.isfinite(reading):
            self.ema = (
                reading
                if self.ema is None
                else (1 - self.alpha) * self.ema + self.alpha * reading
            )
            self.ticks += 1

    def ready(self) -> bool:
        return self.ticks >= self.interval and self.ema is not None

    def consume(self) -> None:
        self.ticks = 0

    def reset(self) -> None:
        self.ticks = 0
