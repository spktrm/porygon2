"""Offline potential contracts; no model initialisation or simulator required."""

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest

from rl.offline.position_potential import (
    PositionFeatures,
    PotentialFit,
    PublicPokemon,
    centred_potential,
    field_matchup,
    pair_pressure,
    position_features,
    position_potential,
    shaping_reward,
)

TYPES = ("Fire", "Water", "Grass", "Electric", "Ground", "Normal", "Ghost")
CHART = {attack: {defence: 1.0 for defence in TYPES} for attack in TYPES}
for attack, defence in (
    ("Fire", "Grass"),
    ("Water", "Fire"),
    ("Grass", "Water"),
    ("Electric", "Water"),
    ("Ground", "Electric"),
):
    CHART[attack][defence] = 2.0
for attack, defence in (("Fire", "Water"), ("Water", "Grass"), ("Grass", "Fire")):
    CHART[attack][defence] = 0.5
CHART["Electric"]["Ground"] = 0.0
CHART["Normal"]["Ghost"] = 0.0


def mon(kind, hp=1.0, *, active=True):
    return PublicPokemon((kind,), (kind,), hp, active)


def test_single_active_reduces_to_original_pair_and_six_mon_units():
    own = mon("Fire", 0.5)
    opponent = mon("Grass")
    features = position_features(
        [own, None, None, None, None, None], [opponent] + [None] * 5, CHART
    )
    assert features == PositionFeatures(-0.5, 0.0, 2.0)
    assert features.matchup == pair_pressure(own, opponent, CHART) - pair_pressure(
        opponent, own, CHART
    )
    fitted = PotentialFit()
    expected = math.tanh((-0.5 * fitted.hp_weight + 2 * fitted.matchup_weight) / 2)
    assert position_potential(features, strength=1.0) == expected


def test_doubles_slot_permutation_and_player_swap_with_live_control():
    own = [mon("Fire", 0.5), mon("Water"), None, None]
    opponent = [mon("Grass"), mon("Electric"), None, None]
    baseline = position_features(own, opponent, CHART)
    for own_order in itertools.permutations(own[:2]):
        for opponent_order in itertools.permutations(opponent[:2]):
            assert (
                position_features(
                    list(own_order) + own[2:],
                    list(opponent_order) + opponent[2:],
                    CHART,
                )
                == baseline
            )
    reverse = position_features(opponent, own, CHART)
    assert reverse == PositionFeatures(
        -baseline.hp_balance, -baseline.alive_balance, -baseline.matchup
    )
    assert position_potential(reverse, strength=0.3) == -position_potential(
        baseline, strength=0.3
    )
    changed = position_features(
        [mon("Ghost", 0.5), own[1], None, None], opponent, CHART
    )
    assert changed.matchup != baseline.matchup


def test_dangerous_second_opponent_is_not_diluted_by_a_harmless_one():
    water = mon("Water")
    electric = mon("Electric")
    ghost = mon("Ghost")
    assert field_matchup([water], [electric, ghost], CHART) == -1.0
    assert field_matchup([water], [ghost], CHART) == 0.0
    # Averaging all four pair operands would dilute the electric threat.
    pair_average = (
        pair_pressure(water, electric, CHART)
        + pair_pressure(water, ghost, CHART)
        - pair_pressure(electric, water, CHART)
        - pair_pressure(ghost, water, CHART)
    ) / 2
    assert pair_average == -0.5


def test_roster_fraction_scale_and_fainted_slot_handling():
    four = position_features([mon("Fire", 0), None, None, None], [None] * 4, CHART)
    six = position_features(
        [mon("Fire", 0), mon("Water", 0), None, None, None, None], [None] * 6, CHART
    )
    assert four.hp_balance == four.alive_balance == -1.5
    assert six.hp_balance == six.alive_balance == -2.0
    four_equal = position_features([mon("Fire", 0)] * 2 + [None] * 2, [None] * 4, CHART)
    six_equal = position_features([mon("Fire", 0)] * 3 + [None] * 3, [None] * 6, CHART)
    assert four_equal == six_equal
    assert four.matchup == 0.0
    with pytest.raises(ValueError, match="roster"):
        position_features([], [None] * 4, CHART)


def test_bench_is_material_only_and_empty_active_slot_is_not_a_target():
    own = [mon("Water"), mon("Fire", active=False)]
    opponent = [mon("Electric"), mon("Ghost", 0)]
    assert position_features(own, opponent, CHART).matchup == -1.0
    changed_bench = [own[0], mon("Ground", active=False)]
    assert position_features(own, opponent, CHART) == position_features(
        changed_bench, opponent, CHART
    )
    brought_in = [replace(own[0], active=False), mon("Ground")]
    assert position_features(brought_in, opponent, CHART).matchup == 3.0


def test_joint_choice_microsteps_do_not_duplicate_shaping_and_total_is_zero():
    raw = [0.2, 0.2, 0.5, 0.5, -0.1]
    endpoints = [
        centred_potential(value, raw[0], terminal=index == len(raw) - 1)
        for index, value in enumerate(raw)
    ]
    shaping = [
        shaping_reward(current, following, gamma=1.0)
        for current, following in zip(endpoints[:-1], endpoints[1:])
    ]
    np.testing.assert_allclose(shaping, [0, 0.3, 0, -0.3])
    assert math.isclose(sum(shaping), 0, abs_tol=1e-12)
    # Paying the nonzero field transition twice would break zero-return PBRS.
    assert not math.isclose(sum(shaping) + shaping[1], 0, abs_tol=1e-12)


@pytest.mark.parametrize("gamma", [1.0, 0.97])
def test_corrected_value_overrides_potential_and_discounted_sum_is_zero(gamma):
    endpoints = np.asarray([0.0, 0.2, -0.3, 0.0])
    values = np.asarray([0.1, -0.2, 0.8, 0.0])
    rewards = np.asarray([0.0, 0.0, 1.0])
    correction = values - endpoints
    shaping = np.asarray(
        [
            shaping_reward(current, following, gamma=gamma)
            for current, following in zip(endpoints[:-1], endpoints[1:])
        ]
    )
    np.testing.assert_allclose(
        rewards + shaping + gamma * correction[1:] - correction[:-1],
        rewards + gamma * values[1:] - values[:-1],
    )
    assert abs(np.dot(gamma ** np.arange(len(shaping)), shaping)) < 1e-12


def test_bounded_potential_off_mode_and_invalid_inputs():
    assert position_potential(PositionFeatures(6, 6, 4), strength=0) == 0
    assert abs(position_potential(PositionFeatures(6, 6, 4), strength=0.3)) <= 0.3
    with pytest.raises(ValueError):
        mon("Fire", float("nan"))
    with pytest.raises(ValueError):
        position_potential(PositionFeatures(0, 0, 0), strength=-1)
    with pytest.raises(ValueError):
        shaping_reward(0, 1, gamma=1.1)
