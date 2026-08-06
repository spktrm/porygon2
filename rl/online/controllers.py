"""Feedback controllers for training hyperparameters.

Both follow the replay-ratio controller's design language (velocity-form
PI on a log-scale actuator, EMA-smoothed sensors, anti-windup via
clamping the actuator itself) and both actuate RUNTIME scalars passed
into the jitted train_step — never static config, which recompiles (and
whose retained executables OOM-killed run 1326).

Timescale separation across the four loops: replay controller (fast,
actor-KL -> reuse cap), entropy rate limiter (fast safety valve,
entropy -> magnet coef), lambda controller (slow, calibration gap ->
advantage lambda), plasticity (episodic; pushes lambda to its ceiling
during recovery).
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
        # Re-clip: the band moves when lambda_ctrl_min/max change between
        # runs, and a restored actuator outside the new band would be
        # stuck until the integral walked it back.
        self._log_h = float(
            np.clip(state["log_h"], self.log_h_min, self.log_h_max)
        )
        self._gap_ema = state["gap_ema"]
        self._prev_err = state["prev_err"]
        self._ticks = int(state["ticks"])


class EntropyRateController:
    """Caps the RATE of policy-entropy decline by scaling the magnet KL
    coefficient — a rate limiter, deliberately not a setpoint: entropy
    should fall over training, it just must not cliff (run 1328 lost
    0.18 normalised entropy in one 20k-step window under lambda=0.5).

    Sensor: fast-vs-slow EMA gap of normalised action entropy — positive
    when declining, with the slow EMA's horizon defining "recent".
    Asymmetric actuation on log(coef): decline beyond the allowed rate
    (or entropy under the hard floor) scales the magnet coefficient up;
    otherwise the coefficient decays back toward its baseline, so the
    controller only ever ADDS diversity pressure and its quiescent state
    is exactly the config value.
    """

    def __init__(
        self,
        baseline_coef: float,
        max_decline: float,
        entropy_floor: float,
        gain: float,
        decay: float,
        max_scale: float,
        fast_ema: float,
        slow_ema: float,
    ):
        self.baseline_log = float(np.log(baseline_coef))
        self.max_decline = max_decline
        self.entropy_floor = entropy_floor
        self.gain = gain
        self.decay = decay
        self.max_log = self.baseline_log + float(np.log(max_scale))
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema

        self._fast: float | None = None
        self._slow: float | None = None
        self._log_coef = self.baseline_log

    @property
    def value(self) -> float:
        return float(np.exp(self._log_coef))

    def update(self, entropy: float | None) -> dict[str, float]:
        if entropy is not None and np.isfinite(entropy):
            self._fast = (
                entropy
                if self._fast is None
                else (1 - self.fast_ema) * self._fast + self.fast_ema * entropy
            )
            self._slow = (
                entropy
                if self._slow is None
                else (1 - self.slow_ema) * self._slow + self.slow_ema * entropy
            )

        if self._fast is None or self._slow is None:
            return self.logs()

        decline = self._slow - self._fast  # positive while entropy falls
        breach = max(decline - self.max_decline, 0.0) / self.max_decline
        if self._fast < self.entropy_floor:
            breach = max(breach, 1.0)

        if breach > 0.0:
            self._log_coef += self.gain * breach
        else:
            self._log_coef -= self.decay
        self._log_coef = float(np.clip(self._log_coef, self.baseline_log, self.max_log))
        return self.logs()

    def logs(self) -> dict[str, float]:
        out = {"entropy_ctrl_coef": self.value}
        if self._fast is not None and self._slow is not None:
            out["entropy_ctrl_decline"] = float(self._slow - self._fast)
        return out

    def state_dict(self) -> dict:
        return dict(fast=self._fast, slow=self._slow, log_coef=self._log_coef)

    def load_state_dict(self, state: dict) -> None:
        self._fast = state["fast"]
        self._slow = state["slow"]
        # Re-clip against the current baseline/max_scale, which move when
        # player_magnet_kl_coef or entropy_ctrl_max_scale change.
        self._log_coef = float(
            np.clip(state["log_coef"], self.baseline_log, self.max_log)
        )
