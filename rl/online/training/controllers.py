"""The one surviving host-side controller: the replay-reuse PI loop.

``PILogController`` is the actuator — a PI update in log space with clipped
bounds. Every other controller this project built (lambda, adaptivity, magnet
watchdog, plasticity) has been removed; CLAUDE.md 10 records what each one
measured and why it went, including the evidence that pulls both ways.
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
