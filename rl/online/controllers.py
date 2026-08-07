"""Feedback controllers for training hyperparameters.

Both follow the replay-ratio controller's design language (velocity-form
PI on a log-scale actuator, EMA-smoothed sensors, anti-windup via
clamping the actuator itself) and both actuate RUNTIME scalars passed
into the jitted train_step — never static config, which recompiles (and
whose retained executables OOM-killed run 1326).

Timescale separation across the loops: replay controller (fast,
actor-KL -> reuse cap), adaptivity controller (commitment covariance ->
magnet KL coef, with entropy floors as hard backstops), lambda
controller (slow, calibration gap -> advantage lambda), plasticity
(episodic; pushes lambda to its ceiling during recovery and bumps the
magnet coef).

The two feedback controllers are deliberately symmetric: one measures
whether the CRITIC deserves trust (bootstrap gap -> how much to
bootstrap), the other whether the POLICY's commitments deserve trust
(commitment covariance -> how much to let it commit).
"""

import numpy as np


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
    and 0.6. Clamping log(1 - lambda) doubles as anti-windup.

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
        self.kp = kp
        self.ki = ki
        self.interval = interval
        self.log_h_min = float(np.log(1.0 - lambda_max + 1e-6))
        self.log_h_max = float(np.log(1.0 - lambda_min))
        self.sensor_ema = sensor_ema

        self._log_h = float(np.log(1.0 - initial_lambda + 1e-6))
        self._gap_ema: float | None = None
        self._prev_err = 0.0
        self._ticks = 0

    @property
    def value(self) -> float:
        return float(1.0 - np.exp(self._log_h))

    def update(self, gap: float | None, recovering: bool) -> dict[str, float]:
        if recovering:
            self._log_h = self.log_h_min
            self._prev_err = 0.0
            self._ticks = 0
            return self.logs()

        if gap is not None and np.isfinite(gap):
            self._gap_ema = (
                gap
                if self._gap_ema is None
                else (1 - self.sensor_ema) * self._gap_ema + self.sensor_ema * gap
            )
            self._ticks += 1

        if self._ticks >= self.interval and self._gap_ema is not None:
            self._ticks = 0
            # err > 0: gap under target -> grow the horizon (lambda down).
            err = (self.gap_target - self._gap_ema) / self.gap_target
            self._log_h += self.kp * (err - self._prev_err) + self.ki * err
            self._prev_err = err
            self._log_h = float(np.clip(self._log_h, self.log_h_min, self.log_h_max))
        return self.logs()

    def logs(self) -> dict[str, float]:
        out = {"lambda_ctrl_lambda": self.value}
        if self._gap_ema is not None:
            out["lambda_ctrl_gap_ema"] = float(self._gap_ema)
        return out

    def state_dict(self) -> dict:
        return dict(
            log_h=self._log_h,
            gap_ema=self._gap_ema,
            prev_err=self._prev_err,
            ticks=self._ticks,
        )

    def load_state_dict(self, state: dict) -> None:
        # Tolerant by key: a checkpoint written by an older controller
        # revision must never be able to fail a resume — the state is an
        # optimisation (skip the re-anneal), not a correctness input.
        # Re-clip too: the band moves when lambda_ctrl_min/max change,
        # and a restored actuator outside it would be stuck until the
        # integral walked it back.
        if "log_h" in state:
            self._log_h = float(
                np.clip(state["log_h"], self.log_h_min, self.log_h_max)
            )
        self._gap_ema = state.get("gap_ema")
        self._prev_err = float(state.get("prev_err", 0.0))
        self._ticks = int(state.get("ticks", 0))


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
        self.baseline_log = float(np.log(baseline_coef))
        self.max_log = self.baseline_log + float(np.log(max_scale))
        # Lower bound is BELOW baseline: with a stationary uniform magnet
        # a fixed coefficient c converges to the QRE of the regularised
        # game, which is O(c) away from the unregularised equilibrium, so
        # a well-validated policy should be allowed to shed pressure and
        # commit (Ataraxos anneals its KL coef downward for the same
        # reason). Bounded well above zero: c -> 0 with a fixed magnet
        # loses the stable fixed point and invites cycling.
        self.min_log = self.baseline_log + float(np.log(min_scale))
        self.commit_target = commit_target
        self.kp = kp
        self.ki = ki
        self.interval = interval
        self.sensor_ema = sensor_ema
        self.action_floor = action_floor
        self.modality_floor = modality_floor
        self.floor_gain = floor_gain
        self.event_bump = event_bump

        self._log_coef = self.baseline_log
        self._cov_ema: float | None = None
        self._prev_err = 0.0
        self._ticks = 0

    @property
    def value(self) -> float:
        return float(np.exp(self._log_coef))

    def bump(self, scale: float | None = None) -> None:
        """Immediate pressure step for a known shock (perturbation, new
        league opponent). Feedback decays it once the policy re-validates."""
        self._log_coef = float(
            np.clip(
                self._log_coef + (self.event_bump if scale is None else scale),
                self.min_log,
                self.max_log,
            )
        )

    def update(
        self,
        commit_cov: float | None,
        action_entropy: float | None,
        modality_entropy: float | None,
    ) -> dict[str, float]:
        if commit_cov is not None and np.isfinite(commit_cov):
            self._cov_ema = (
                commit_cov
                if self._cov_ema is None
                else (1 - self.sensor_ema) * self._cov_ema
                + self.sensor_ema * commit_cov
            )
            self._ticks += 1

        # Hard floors first: a collapsed modality is invisible to the
        # covariance, so these override the PI action entirely.
        breach = 0.0
        if action_entropy is not None and np.isfinite(action_entropy):
            breach = max(breach, self.action_floor - action_entropy)
        if modality_entropy is not None and np.isfinite(modality_entropy):
            breach = max(breach, self.modality_floor - modality_entropy)

        if breach > 0.0:
            self._log_coef += self.floor_gain * breach
            self._prev_err = 0.0
            self._ticks = 0
        elif self._ticks >= self.interval and self._cov_ema is not None:
            self._ticks = 0
            # err > 0: commitment under-validated -> raise pressure.
            err = (self.commit_target - self._cov_ema) / max(
                abs(self.commit_target), 1e-6
            )
            self._log_coef += self.kp * (err - self._prev_err) + self.ki * err
            self._prev_err = err

        self._log_coef = float(np.clip(self._log_coef, self.min_log, self.max_log))
        return self.logs(breach)

    def logs(self, breach: float = 0.0) -> dict[str, float]:
        out = {
            "adapt_ctrl_coef": self.value,
            "adapt_ctrl_floor_breach": float(breach),
        }
        if self._cov_ema is not None:
            out["adapt_ctrl_commit_ema"] = float(self._cov_ema)
        return out

    def state_dict(self) -> dict:
        return dict(
            log_coef=self._log_coef,
            cov_ema=self._cov_ema,
            prev_err=self._prev_err,
            ticks=self._ticks,
        )

    def load_state_dict(self, state: dict) -> None:
        # Tolerant by key: checkpoints written by the superseded
        # EntropyRateController carry {fast, slow, log_coef}. The
        # actuator (log_coef) means the same thing in both and is worth
        # keeping; the old entropy EMAs have no counterpart here, so the
        # commitment sensor simply starts fresh. Re-clip because
        # baseline/min_scale/max_scale move when the config changes.
        if "log_coef" in state:
            self._log_coef = float(
                np.clip(state["log_coef"], self.min_log, self.max_log)
            )
        self._cov_ema = state.get("cov_ema")
        self._prev_err = float(state.get("prev_err", 0.0))
        self._ticks = int(state.get("ticks", 0))
