"""Config primitives shared by the RL learner and the offline replay trainer.

Experiment configs compose from ``BaseTrainingConfig`` so the format and
logging plumbing lives in one place, while RL self-play and offline training
keep separate, independently tunable configs.
"""

from typing import Literal

import chex


@chex.dataclass(frozen=True)
class AdamWConfig:
    """Adam optimizer related params."""

    b1: float
    b2: float
    eps: float
    weight_decay: float


GenT = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9]
SmogonFormatT = Literal["ou", "uu", "ru", "nu", "pu", "ubers", "randombattle"]


@chex.dataclass(frozen=True)
class BaseTrainingConfig:
    """Fields common to every training pipeline (RL self-play and offline)."""

    # Smogon Generation
    generation: GenT = 9
    smogon_format: SmogonFormatT = "randombattle"

    # Logging params
    log_artifacts_online: bool = False

    @property
    def format_id(self) -> str:
        return f"gen{self.generation}{self.smogon_format}"
