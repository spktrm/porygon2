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

Timescale separation across the loops: replay controller (fast,
actor-KL -> reuse cap), lambda controller (slow, calibration gap ->
advantage lambda), plasticity (episodic; pushes lambda to its ceiling
during recovery), exploitability controller (slowest — league win-rate
-> a shared caution scale applied to the lambda and replay controllers'
targets, not a runtime scalar of its own).

There is deliberately no controller on the magnet KL coefficient
anymore. The AdaptivityController (commitment-covariance PI, later
floor-only escalation on entropy-floor breaches) was removed entirely
2026-08-13: across several runs its effect proved hard to tune and
predict — the PI action caused three separate bugs (unreachable target
pinning pressure at the ceiling in 1338/1339, divide-by-near-zero in
1341, an exploit_ctrl target-scaling bug), and the stacked event bumps
held main at ~6x baseline pressure for a whole run (session 1786537634)
with zero floor breaches. The coefficient is now exactly
config.player_magnet_kl_coef, always. The entropy signals it watched
are still logged from train_step for the dashboard; nothing acts on
them — modality collapse (the 1330 failure mode) is watched, not
auto-corrected.
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


class LambdaGapController:
    """Adapts the ADVANTAGE lambda by holding the critic's measured
    bootstrap bias at a target.

    Sensor: player_bootstrap_gap — the per-batch mean |main head value −
    lambda=1.0 Monte Carlo anchor value|. Gap under target means the
    critic is trustworthy, so lambda drifts down and the actor harvests
    variance reduction; gap over target means bootstrap bias is leaking
    into the policy gradient, so lambda backs off toward outcomes.

    Works in log(1 - lambda) (horizon space): a step is a multiplicative
    change to the bootstrap horizon, so moves are equally sized at 0.99
    and 0.6.

    Plasticity coupling: while the plasticity controller is in recovery
    the perturbed critic cannot be trusted, so lambda is driven to its
    ceiling and the PI state is reset; annealing restarts from the top
    as the critic re-fits — the transient reversion the 1328 postmortem
    showed a schedule cannot provide.

    Known blind spot: the gap reads zero where a shared-trunk error
    fools both readouts, so lambda_min is a hard floor, not a tunable
    convenience.
    """

    def __init__(
        self,
        initial_lambda: float,
        gap_target: float,
        kp: float,
        ki: float,
        interval: int,
        lambda_min: float,
        lambda_max: float,
        sensor_ema: float,
    ):
        self.gap_target = gap_target
        log_h_min = float(np.log(1.0 - lambda_max + 1e-6))
        log_h_max = float(np.log(1.0 - lambda_min))
        initial_log_h = float(np.log(1.0 - initial_lambda + 1e-6))

        self._pi = PILogController(initial_log_h, log_h_min, log_h_max, kp, ki)
        self._sensor = EmaSensor(sensor_ema, interval)

    @property
    def value(self) -> float:
        return float(1.0 - np.exp(self._pi.log))

    def update(self, gap: float | None, recovering: bool) -> dict[str, float]:
        if recovering:
            self._pi.set_log(self._pi.log_min)
            self._pi.prev_err = 0.0
            self._sensor.reset()
            return self.logs()

        self._sensor.observe(gap)
        if self._sensor.ready():
            self._sensor.consume()
            # err > 0: gap under target -> grow the horizon (lambda down).
            err = (self.gap_target - self._sensor.ema) / self.gap_target
            self._pi.step(err)
        return self.logs()

    def logs(self) -> dict[str, float]:
        out = {"lambda_ctrl_lambda": self.value}
        if self._sensor.ema is not None:
            out["lambda_ctrl_gap_ema"] = float(self._sensor.ema)
        return out

    def state_dict(self) -> dict:
        return dict(
            log_h=self._pi.log,
            gap_ema=self._sensor.ema,
            prev_err=self._pi.prev_err,
            ticks=self._sensor.ticks,
        )

    def load_state_dict(self, state: dict) -> None:
        # Tolerant by key: a checkpoint written by an older controller
        # revision must never be able to fail a resume — the state is an
        # optimisation (skip the re-anneal), not a correctness input.
        # Re-clip too: the band moves when lambda_ctrl_min/max change,
        # and a restored actuator outside it would be stuck until the
        # integral walked it back.
        if "log_h" in state:
            self._pi.set_log(state["log_h"])
        self._sensor.ema = state.get("gap_ema")
        self._pi.prev_err = float(state.get("prev_err", 0.0))
        self._sensor.ticks = int(state.get("ticks", 0))


class ExploitabilityController:
    """Adapts a shared CAUTION SCALE by holding league exploitability —
    1 minus main's win-rate against its worst historical snapshot — at a
    target, then applies that scale to the lambda and replay
    controllers' targets (lambda_ctrl_gap_target, the replay KL target)
    rather than driving a runtime scalar of its own. Callers own
    applying the scale — see learner._apply_exploit_scale — since the
    sign of the adjustment differs per target (shrink some, grow others)
    depending on which direction means "more cautious" for that
    controller. Used to also scale the adaptivity controller's commit
    target, but that controller no longer exists at all (removed
    2026-08-13 — see the module docstring), so this touches exactly two
    targets.

    Sign convention is flipped relative to the other controllers: the
    actuator (log(scale)) rises when the sensor is ABOVE target (more
    exploitable than tolerated), not below, since exploitability is a
    quantity to hold DOWN rather than a trust quantity to hold up.
    exploit_ctrl_target=0.3 mirrors the existing "dominant" league-
    addition threshold (win-rate > 0.7 there == exploitability < 0.3
    here) — scale grows once the worst-case win-rate drifts below that
    same bar.

    Deliberately not a bandit: LambdaBandit's discounted-UCB paid an
    exploration tax (it must sometimes hold an arm it suspects is worse,
    to keep the uncertainty estimate honest) on top of the unavoidable
    latency of the win-rate signal itself (games have to be played). A
    PI loop pays only the unavoidable part — it reacts to every reading
    proportionally, with no explore/exploit phase of its own.

    Sensor: measured only every manage_league_interval call (win-rate
    against historical snapshots is not a per-batch quantity), gated by
    exploit_ctrl_min_historical so a lone freshly-added snapshot (whose
    win-rate is still Bayesian-prior-dominated, see League._win_rate)
    cannot swing the scale. Deliberately the raw win-rate table, not the
    slower BT-rating auditors in bandit.py — those need hundreds of games
    per point and stay logged-only for exactly that reason.
    """

    def __init__(
        self,
        target: float,
        kp: float,
        ki: float,
        interval: int,
        min_scale: float,
        max_scale: float,
        sensor_ema: float,
    ):
        self.target = target
        log_min = float(np.log(min_scale))
        log_max = float(np.log(max_scale))

        self._pi = PILogController(0.0, log_min, log_max, kp, ki)
        self._sensor = EmaSensor(sensor_ema, interval)

    @property
    def value(self) -> float:
        return float(np.exp(self._pi.log))

    def update(self, exploitability: float | None) -> dict[str, float]:
        self._sensor.observe(exploitability)
        if self._sensor.ready():
            self._sensor.consume()
            # err > 0: more exploitable than tolerated -> raise caution.
            err = (self._sensor.ema - self.target) / max(self.target, 1e-6)
            self._pi.step(err)
        return self.logs()

    def logs(self) -> dict[str, float]:
        out = {"exploit_ctrl_scale": self.value}
        if self._sensor.ema is not None:
            out["exploit_ctrl_exploitability_ema"] = float(self._sensor.ema)
        return out

    def state_dict(self) -> dict:
        return dict(
            log_scale=self._pi.log,
            exploitability_ema=self._sensor.ema,
            prev_err=self._pi.prev_err,
            ticks=self._sensor.ticks,
        )

    def load_state_dict(self, state: dict) -> None:
        if "log_scale" in state:
            self._pi.set_log(state["log_scale"])
        self._sensor.ema = state.get("exploitability_ema")
        self._pi.prev_err = float(state.get("prev_err", 0.0))
        self._sensor.ticks = int(state.get("ticks", 0))
