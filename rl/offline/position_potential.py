"""Experimental public-state potential for singles and multiple-active formats.

Coefficients are from the 2026-09-11 singles human replay outcome fit. The
multiple-active aggregation is a structural extension, not doubles calibration.
This module is an offline reference; the learner does not consume its rewards.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

TypeChart = Mapping[str, Mapping[str, float]]
_FIT = json.loads(Path(__file__).with_name("position_potential_fit.json").read_text())


@dataclass(frozen=True)
class PublicPokemon:
    """A revealed mon's last public state; unrevealed roster slots use None.

    Offensive types include original STAB and any additional revealed tera
    STAB. Defensive types describe the current revealed typing. The caller
    resolves these from public events, never the completed replay's future.
    """

    offensive_types: tuple[str, ...]
    defensive_types: tuple[str, ...]
    hp_fraction: float
    active: bool = False

    def __post_init__(self) -> None:
        if not self.offensive_types or not self.defensive_types:
            raise ValueError("Revealed Pokémon require offensive and defensive types")
        if not math.isfinite(self.hp_fraction) or not 0 <= self.hp_fraction <= 1:
            raise ValueError("HP fraction must be finite and between zero and one")


@dataclass(frozen=True)
class PositionFeatures:
    hp_balance: float
    alive_balance: float
    matchup: float


@dataclass(frozen=True)
class PotentialFit:
    """Full precision train-only win-logit coefficients, not shaping strength.

    reference_team_size records the six-mon units of the singles fit. Scaling
    roster fractions to those units preserves scale across bring-four/six;
    predictive calibration across formats is not established by this choice.
    """

    hp_weight: float = _FIT["hp_weight"]
    alive_weight: float = _FIT["alive_weight"]
    matchup_weight: float = _FIT["matchup_weight"]
    reference_team_size: int = _FIT["reference_team_size"]

    def __post_init__(self) -> None:
        if self.reference_team_size <= 0:
            raise ValueError("Reference team size must be positive")
        for weight in (self.hp_weight, self.alive_weight, self.matchup_weight):
            if not math.isfinite(weight):
                raise ValueError("Potential weights must be finite")


def pair_pressure(
    attacker: PublicPokemon, defender: PublicPokemon, type_chart: TypeChart
) -> float:
    """Log2 best species-STAB multiplier; immunity shares the quarter floor."""
    best_multiplier = 0.0
    for offensive_type in attacker.offensive_types:
        multipliers = type_chart[offensive_type]
        multiplier = 1.0
        for defensive_type in defender.defensive_types:
            factor = multipliers[defensive_type]
            if not math.isfinite(factor) or not 0 <= factor <= 2:
                raise ValueError("Type-chart factors must be finite and in [0, 2]")
            multiplier *= factor
        best_multiplier = max(best_multiplier, multiplier)
    # The upper cap also bounds caller-supplied multi-type states.
    return math.log2(min(4.0, max(0.25, best_multiplier)))


def field_matchup(
    own_actives: Sequence[PublicPokemon],
    opponent_actives: Sequence[PublicPokemon],
    type_chart: TypeChart,
) -> float:
    """Compare each defender's worst incoming type threat across the field.

    A dangerous opponent is not diluted by a harmless one. This deliberately
    does not model targeting allocation, double targets, spread damage or
    support moves. Empty active fields have no matchup estimate.
    """
    own_alive = [mon for mon in own_actives if mon.hp_fraction > 0]
    opponent_alive = [mon for mon in opponent_actives if mon.hp_fraction > 0]
    if not own_alive or not opponent_alive:
        return 0.0

    def exposure(
        defenders: Sequence[PublicPokemon], attackers: Sequence[PublicPokemon]
    ) -> float:
        threats = [
            max(pair_pressure(attacker, defender, type_chart) for attacker in attackers)
            for defender in defenders
        ]
        return math.fsum(threats) / len(threats)

    return exposure(opponent_alive, own_alive) - exposure(own_alive, opponent_alive)


def position_features(
    own_team: Sequence[PublicPokemon | None],
    opponent_team: Sequence[PublicPokemon | None],
    type_chart: TypeChart,
    *,
    fit: PotentialFit = PotentialFit(),
) -> PositionFeatures:
    """Team fractions in reference-roster units plus the active-field matchup.

    Each team contains only its battle roster, not unselected preview options.
    None represents an unrevealed member of that roster (last public HP=1).
    Fainted mons remain in the roster with HP=0; do not shrink the denominator.
    """

    def team_summary(team: Sequence[PublicPokemon | None]):
        if not team:
            raise ValueError("Battle roster size must be known and positive")
        hp_fractions = []
        alive_count = 0
        actives = []
        for mon in team:
            if mon is None:
                hp_fractions.append(1.0)
                alive_count += 1
            else:
                hp_fractions.append(mon.hp_fraction)
                if mon.hp_fraction > 0:
                    alive_count += 1
                    if mon.active:
                        actives.append(mon)
        normalisation = fit.reference_team_size / len(team)
        return (
            math.fsum(hp_fractions) * normalisation,
            alive_count * normalisation,
            actives,
        )

    own_hp, own_alive, own_actives = team_summary(own_team)
    opponent_hp, opponent_alive, opponent_actives = team_summary(opponent_team)
    return PositionFeatures(
        hp_balance=own_hp - opponent_hp,
        alive_balance=own_alive - opponent_alive,
        matchup=field_matchup(own_actives, opponent_actives, type_chart),
    )


def position_potential(
    features: PositionFeatures,
    *,
    strength: float,
    fit: PotentialFit = PotentialFit(),
) -> float:
    """Return raw Phi = strength * tanh(fitted win logit / 2).

    Features are HP balance, alive balance and active-field matchup balance.
    Positive Phi means a favourable estimated position. Phi is not an additive
    reward: centre its endpoints and take their difference with shaping_reward.
    """
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("Potential strength must be finite and nonnegative")
    feature_values = (features.hp_balance, features.alive_balance, features.matchup)
    if not all(math.isfinite(value) for value in feature_values):
        raise ValueError("Position features must be finite")
    win_logit = (
        fit.hp_weight * features.hp_balance
        + fit.alive_weight * features.alive_balance
        + fit.matchup_weight * features.matchup
    )
    return strength * math.tanh(win_logit / 2)


def centred_potential(value: float, initial_value: float, *, terminal: bool) -> float:
    """Return Psi = raw Phi - game-start Phi, or zero at the true terminal.

    value and initial_value must use the same fitted potential and strength.
    initial_value is fixed for the whole battle, including both doubles slots
    and every replay chunk. Both the initial and terminal Psi are zero.
    """
    if not math.isfinite(value) or not math.isfinite(initial_value):
        raise ValueError("Potential endpoints must be finite")
    if terminal:
        return 0.0
    return value - initial_value


def shaping_reward(current: float, following: float, *, gamma: float) -> float:
    """Return F = gamma * following - current, to add to the original reward.

    current and following are consecutive CENTRED potentials Psi, as returned
    by centred_potential, not raw Phi scores. following must be zero at a real
    game termination; it must not be reset at a chunk boundary. gamma must match
    the learner discount. With initial and terminal Psi zero, discounted F sums
    to zero; at the current gamma=1 the ordinary sum is also exactly zero.

    For gamma=1, nonterminal F equals raw Phi(next) - raw Phi(current), while
    terminal F equals raw Phi(game start) - raw Phi(current). An unchanged-field
    decision microstep gets zero. Never duplicate a resolved field reward across
    the two active-slot choices. The caller retains the original reward as well.
    """
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("Gamma must be finite and between zero and one")
    if not math.isfinite(current) or not math.isfinite(following):
        raise ValueError("Potential endpoints must be finite")
    return gamma * following - current
