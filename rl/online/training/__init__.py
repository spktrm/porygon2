"""Learner-side training: the update, its batch assembly, and the loop.

Split out of a single 2,600-line rl/online/learner.py (2026-08-21). The
seam is what each piece needs to run: train_step needs only tensors and
config, batching needs only trajectories, and the Learner needs the
process-wide singletons. loss/targets/controllers/telemetry live here too
— nothing outside training/ imports them. What stays flat in rl/online/ is
what the ACTORS also touch: buffer, guards, agent, inference, league,
config. Everything the actors and the entrypoint import comes through
here.
"""

from rl.online.training.batching import stack_batch
from rl.online.training.learner import Learner, OOMGuardTriggered
from rl.online.training.run_state import AddReason, RunState
from rl.online.training.train_step import train_step

__all__ = [
    "AddReason",
    "Learner",
    "OOMGuardTriggered",
    "RunState",
    "stack_batch",
    "train_step",
]
