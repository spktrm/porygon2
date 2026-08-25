"""Feedback controllers for training hyperparameters.

All of them follow the replay-ratio controller's design language
(velocity-form PI on a log-scale actuator, EMA-smoothed sensors,
anti-windup via clamping the actuator itself) and those that drive
train_step actuate RUNTIME scalars — never static config, which
recompiles (and whose retained executables OOM-killed run 1326) — the
RuntimeScalars pytree that carried them was removed 2026-08-21 once
nothing varied them, so a NEW controller has to bring its own back. That
shared shape lives in ``PILogController`` (the actuator: step, bump,
clamp) below; a hyperparameter controller is then just its error
function, bounds, and whatever hard overrides it needs — the replay
reuse-cap loop in
learner.py builds on ``PILogController`` the same way but keeps its own
windowed-mean sensor, since averaging the actor-KL over a fixed window
rather than decaying it is a deliberate, different smoothing choice.

What survives (2026-08-21): exactly ONE controller — the replay
reuse-cap loop in learner.py (actor-KL -> reuse cap), which actuates a
data-pipeline resource, never learning dynamics. This module holds the
one shared primitive it uses (PILogController).

Every other controller this project built has been removed, all for the
same reason — their effects proved harder to tune and predict than the
thing they controlled:

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
  per-step outcome-conditional cut supplied locally what the
  controller's single global lambda approximated. Its one
  genuinely-useful behaviour, forcing pure Monte Carlo while a
  freshly-perturbed critic is untrustworthy, survived as "upgo_coef=0
  during plasticity recovery" — and both of those went in the
  2026-08-21 pass (CLAUDE.md 3 and 10).
- PlasticityController (shrink-and-perturb on league stagnation; removed
  2026-08-21): rarely-fired machinery with a large blast radius. See
  CLAUDE.md 10 — the evidence pulls both ways and is recorded there.
- ExploitabilityController (caution scale; removed 2026-08-14): built to
  scale three other controllers' targets, it outlived all three — by the
  end its only action was a bounded nudge on the replay KL target,
  driven by a slow, prior-dominated worst-matchup win-rate sensor (which
  false-positived in 1338 from lightly-played snapshots reading ~0.5).
  AlphaStar has no analogue. The sensor's signal survives as
  _should_add_new_player's "dominant" gate; the replay KL target is
  fixed at config.player_replay_kl_target. (The BT-rating auditors it
  also fed went 2026-08-21 with rl/online/ratings.py — a rating needs
  hundreds of games per point, so it never became actionable.)
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
