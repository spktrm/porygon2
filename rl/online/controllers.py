"""Feedback controllers for training hyperparameters.

All of them follow the replay-ratio controller's design language
(velocity-form PI on a log-scale actuator, EMA-smoothed sensors,
anti-windup via clamping the actuator itself) and the three that drive
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
actor-KL -> reuse cap), adaptivity controller (commitment covariance ->
magnet KL coef, with entropy floors as hard backstops), lambda
controller (slow, calibration gap -> advantage lambda), plasticity
(episodic; pushes lambda to its ceiling during recovery and bumps the
magnet coef), exploitability controller (slowest — league win-rate ->
a shared caution scale applied to the OTHER controllers' targets, not a
runtime scalar of its own).

The two per-batch feedback controllers are deliberately symmetric: one
measures whether the CRITIC deserves trust (bootstrap gap -> how much to
bootstrap), the other whether the POLICY's commitments deserve trust
(commitment covariance -> how much to let it commit). The exploitability
controller measures what both of those are ultimately proxies for —
whether the LEAGUE still finds a hole — but only every
manage_league_interval call, since win-rate data cannot arrive any
faster than games are played; it is a slow outer loop over the two fast
inner ones' setpoints, not a replacement for them (see run 1330: a
modality collapse that a coarse win-rate signal would see only after
the damage was already done).
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

    def __init__(self, initial_log: float, log_min: float, log_max: float, kp: float, ki: float):
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


class AdaptivityController:
    """Adapts the magnet KL coefficient by holding the policy's measured
    COMMITMENT VALIDATION at a target — the entropy counterpart of the
    lambda controller.

    Sensor: player_commit_cov, the batch CORRELATION between log pi(a)
    of the action taken and the advantage it earned (bounded [-1, 1];
    normalised so it does not drift with entropy or advantage scale —
    the raw covariance would partly measure the entropy this controller
    actuates, coupling sensor to actuator). High means the policy's
    confidence is being validated, so diversity pressure can decay and
    the policy is allowed to sharpen. Near zero (or negative) means it is
    confidently choosing actions that are not paying, so pressure rises
    and the policy holds its options open.

    Why feedback and not an anneal: the covariance falls exactly when the
    world changes under the policy — a new league snapshot whose style
    breaks habitual lines, or a shrink-and-perturb that scrambles
    preferences — and recovers as the policy re-adapts. A monotone
    schedule cannot see either event. Discrete shocks additionally call
    ``bump()`` so pressure rises the instant the event is known rather
    than a controller tick later.

    Backstops, not mechanism: the covariance can only see actions the
    policy actually takes, so a modality going extinct is invisible to it
    (run 1330 lost switching at a healthy-looking covariance). The
    entropy floors below are hard overrides for that failure mode.

    Actuator is log(coef), clamped to
    [baseline*min_scale, baseline*max_scale] — pressure can decay BELOW
    the configured baseline once commitment is well validated, which is
    how the loop expresses an anneal endogenously; clamping is
    anti-windup.
    """

    def __init__(
        self,
        baseline_coef: float,
        commit_target: float,
        kp: float,
        ki: float,
        interval: int,
        max_scale: float,
        min_scale: float,
        sensor_ema: float,
        action_floor: float,
        modality_floor: float,
        floor_gain: float,
        event_bump: float,
    ):
        baseline_log = float(np.log(baseline_coef))
        # Lower bound is BELOW baseline: with a stationary uniform magnet
        # a fixed coefficient c converges to the QRE of the regularised
        # game, which is O(c) away from the unregularised equilibrium, so
        # a well-validated policy should be allowed to shed pressure and
        # commit (Ataraxos anneals its KL coef downward for the same
        # reason). Bounded well above zero: c -> 0 with a fixed magnet
        # loses the stable fixed point and invites cycling.
        log_min = baseline_log + float(np.log(min_scale))
        log_max = baseline_log + float(np.log(max_scale))

        self.commit_target = commit_target
        self.action_floor = action_floor
        self.modality_floor = modality_floor
        self.floor_gain = floor_gain
        self.event_bump = event_bump

        self._pi = PILogController(baseline_log, log_min, log_max, kp, ki)
        self._sensor = EmaSensor(sensor_ema, interval)

    @property
    def value(self) -> float:
        return float(np.exp(self._pi.log))

    def bump(self, scale: float | None = None) -> None:
        """Immediate pressure step for a known shock (perturbation, new
        league opponent). Feedback decays it once the policy re-validates."""
        self._pi.bump(self.event_bump if scale is None else scale)

    def update(
        self,
        commit_cov: float | None,
        action_entropy: float | None,
        modality_entropy: float | None,
    ) -> dict[str, float]:
        self._sensor.observe(commit_cov)

        # Hard floors first: a collapsed modality is invisible to the
        # covariance, so these override the PI action entirely.
        breach = 0.0
        if action_entropy is not None and np.isfinite(action_entropy):
            breach = max(breach, self.action_floor - action_entropy)
        if modality_entropy is not None and np.isfinite(modality_entropy):
            breach = max(breach, self.modality_floor - modality_entropy)

        if breach > 0.0:
            self._pi.bump(self.floor_gain * breach)
            self._pi.prev_err = 0.0
            self._sensor.reset()
        elif self._sensor.ready():
            self._sensor.consume()
            # err > 0: commitment under-validated -> raise pressure.
            err = (self.commit_target - self._sensor.ema) / max(
                abs(self.commit_target), 1e-6
            )
            self._pi.step(err)

        return self.logs(breach)

    def logs(self, breach: float = 0.0) -> dict[str, float]:
        out = {
            "adapt_ctrl_coef": self.value,
            "adapt_ctrl_floor_breach": float(breach),
        }
        if self._sensor.ema is not None:
            out["adapt_ctrl_commit_ema"] = float(self._sensor.ema)
        return out

    def state_dict(self) -> dict:
        return dict(
            log_coef=self._pi.log,
            cov_ema=self._sensor.ema,
            prev_err=self._pi.prev_err,
            ticks=self._sensor.ticks,
        )

    def load_state_dict(self, state: dict) -> None:
        # Tolerant by key: checkpoints written by the superseded
        # EntropyRateController carry {fast, slow, log_coef}. The
        # actuator (log_coef) means the same thing in both and is worth
        # keeping; the old entropy EMAs have no counterpart here, so the
        # commitment sensor simply starts fresh. Re-clip because
        # baseline/min_scale/max_scale move when the config changes.
        if "log_coef" in state:
            self._pi.set_log(state["log_coef"])
        self._sensor.ema = state.get("cov_ema")
        self._pi.prev_err = float(state.get("prev_err", 0.0))
        self._sensor.ticks = int(state.get("ticks", 0))


class ExploitabilityController:
    """Adapts a shared CAUTION SCALE by holding league exploitability —
    1 minus main's win-rate against its worst historical snapshot — at a
    target, then applies that scale to the OTHER controllers' targets
    (lambda_ctrl_gap_target, adapt_ctrl_commit_target, the replay KL
    target) rather than driving a runtime scalar of its own. Callers own
    applying the scale — see learner._update_exploit_controller — since
    the sign of the adjustment differs per target (shrink some, grow
    others) depending on which direction means "more cautious" for that
    controller.

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
