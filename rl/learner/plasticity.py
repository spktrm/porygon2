"""Shrink-and-perturb plasticity updates triggered by league stagnation.

Detection is bias-free: it relies only on how snapshots get added to the
league. Healthy training adds players because the main agent dominates its
own history ("dominant"); a stagnant run only adds them because the frame
budget expired ("overdue"). Consecutive overdue-only additions therefore
signal a plateau and trigger a shrink-and-perturb update (Ash & Adams,
https://arxiv.org/abs/1910.08475): params are interpolated toward a fresh
init draw, restoring plasticity, while the untouched EMA target params act
as a self-distillation anchor through the KL loss so pre-perturbation
behaviour is relearned rather than lost.
"""

from typing import Literal

import jax

from rl.learner.learner import Porygon2PlayerTrainState

AddReason = Literal["initial", "dominant", "overdue"]


def _shrink_coefficient(path, shrink_by_module: dict, default_shrink: float) -> float:
    """Resolve the shrink factor for a leaf from its top-level module name.

    Param trees look like {"params": {"encoder": ..., "v_head": ...}}; leaves
    outside the "params" collection (e.g. batch stats) are left untouched via
    a coefficient of 1.0.
    """
    keys = [getattr(p, "key", None) for p in path]
    if not keys or keys[0] != "params":
        return 1.0
    module = keys[1] if len(keys) > 1 else None
    return shrink_by_module.get(module, default_shrink)


def shrink_and_perturb_player_state(
    player_state: Porygon2PlayerTrainState,
    rng: jax.Array,
    default_shrink: float,
    module_shrink: tuple[tuple[str, float], ...] = (),
):
    """Interpolate params toward a fresh init draw and reset optimizer state.

    ``new = lam * old + (1 - lam) * fresh_init`` per leaf, with ``lam`` chosen
    by top-level module. Sampling the perturbation from the init distribution
    respects per-layer init scales, unlike fixed-sigma Gaussian noise.
    ``target_params`` and ``anchor_params`` are deliberately not perturbed so
    the KL terms pull the perturbed policy back toward pre-perturbation
    behaviour.
    """
    fresh_params = player_state.init_fn(rng)
    shrink_by_module = dict(module_shrink)

    def _interp(path, old, new):
        lam = _shrink_coefficient(path, shrink_by_module, default_shrink)
        return (lam * old + (1.0 - lam) * new).astype(old.dtype)

    new_params = jax.tree_util.tree_map_with_path(
        _interp, player_state.params, fresh_params
    )
    return player_state.replace(
        params=new_params,
        # Adam moments are stale for the perturbed weights; the entropy jump
        # after perturbation also invalidates the alpha controller's moments.
        opt_state=player_state.tx.init(new_params),
        alpha_opt_state=player_state.alpha_tx.init(player_state.alpha_params),
    )


class PlasticityController:
    """Tracks league-addition reasons and gates shrink-and-perturb updates.

    A perturbation fires after ``overdue_trigger`` consecutive overdue-only
    league additions, then the controller enters recovery: no further
    perturbations (and no overdue counting) until the main player's winrate
    against the pre-perturbation snapshot reaches ``recovery_winrate`` and
    ``cooldown_frames`` have elapsed since the perturbation.
    """

    def __init__(
        self,
        enabled: bool,
        overdue_trigger: int,
        recovery_winrate: float,
        cooldown_frames: int,
    ):
        self.enabled = enabled
        self.overdue_trigger = overdue_trigger
        self.recovery_winrate = recovery_winrate
        self.cooldown_frames = cooldown_frames

        self.consecutive_overdue = 0
        self.perturbation_count = 0
        self.recovering = False
        self.recovery_ref_step: int | None = None
        self.last_perturb_frame: int | None = None
        self.last_recovery_winrate = 0.0

    def on_player_added(self, reason: AddReason):
        if reason == "dominant":
            self.consecutive_overdue = 0
        elif reason == "overdue" and not self.recovering:
            self.consecutive_overdue += 1

    def should_perturb(self, current_frames: int) -> bool:
        if not self.enabled or self.recovering:
            return False
        if self.consecutive_overdue < self.overdue_trigger:
            return False
        if (
            self.last_perturb_frame is not None
            and current_frames - self.last_perturb_frame < self.cooldown_frames
        ):
            return False
        return True

    def on_perturbation(self, recovery_ref_step: int, current_frames: int):
        self.perturbation_count += 1
        self.consecutive_overdue = 0
        self.recovering = True
        self.recovery_ref_step = recovery_ref_step
        self.last_perturb_frame = current_frames

    def check_recovery(self, winrate_vs_ref: float, current_frames: int):
        self.last_recovery_winrate = winrate_vs_ref
        if not self.recovering:
            return
        cooled_down = (
            self.last_perturb_frame is None
            or current_frames - self.last_perturb_frame >= self.cooldown_frames
        )
        if cooled_down and winrate_vs_ref >= self.recovery_winrate:
            self.recovering = False
            self.recovery_ref_step = None

    def logs(self) -> dict:
        return {
            "plasticity_consecutive_overdue": self.consecutive_overdue,
            "plasticity_perturbation_count": self.perturbation_count,
            "plasticity_recovering": int(self.recovering),
            "plasticity_recovery_winrate": self.last_recovery_winrate,
        }
