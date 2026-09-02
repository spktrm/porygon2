"""The actor's carry loop, plain python (no model): which path a request
takes and why, and how the inference server stacks a mixed group."""

import jax
import numpy as np
import pytest

from rl.environment.actor_stats import ActorStats
from rl.environment.interfaces import HistoryCarry
from rl.environment.protos.features_pb2 import FieldFeature
from rl.environment.utils import (
    ACTOR_HISTORY_MIN_LENGTH,
    get_ex_player_step,
    packed_valid_rows,
)
from rl.model.history_encoder import invalid_history_carry
from rl.online.inference import _stack_history_carries
from rl.online.player_actor import PlayerActor, _last_step_index

WIDTH = 8


class _StubEnv:
    def __init__(self):
        self.history_rewrite_count = 0


def _actor(stats: ActorStats | None) -> PlayerActor:
    return PlayerActor(
        agent=None,
        env=_StubEnv(),
        unroll_length=1,
        learner=None,
        stats=stats,
        history_carry_width=WIDTH,
    )


def _valid_carry() -> HistoryCarry:
    carry = invalid_history_carry(WIDTH)
    return carry.replace(
        slot_states=carry.slot_states + 1.0, valid=np.ones((), dtype=bool)
    )


@pytest.fixture(scope="module")
def full_window():
    actor_input, _ = jax.tree.map(lambda x: np.asarray(x[:, 0]), get_ex_player_step())
    return actor_input


def test_last_step_index_reads_the_last_valid_step(full_window):
    field = np.asarray(full_window.history.field)
    valid_steps = int(field[:, FieldFeature.FIELD_FEATURE__VALID].sum())
    assert _last_step_index(full_window) == int(
        field[valid_steps - 1, FieldFeature.FIELD_FEATURE__INDEX]
    )
    empty = field.copy()
    empty[:, FieldFeature.FIELD_FEATURE__VALID] = 0
    assert (
        _last_step_index(
            full_window.replace(history=full_window.history.replace(field=empty))
        )
        == -1
    )


def test_game_start_is_the_full_window_with_an_invalid_carry(full_window):
    stats = ActorStats()
    actor = _actor(stats)
    request = actor._carry_history(full_window, None, -1, 0)
    full = actor.clip_actor_history(full_window)
    np.testing.assert_array_equal(request.history.field, full.history.field)
    np.testing.assert_array_equal(
        request.packed_history.revealed_cache, full.packed_history.revealed_cache
    )
    assert not bool(request.history_carry.valid)
    assert request.history_carry.slot_states.shape == (12, WIDTH)
    means = stats.drain()
    assert means["actor_history_recompute_frac"] == 1.0
    assert means["actor_history_recompute_game_start"] == 1.0
    assert means["actor_history_recompute_rewrite"] == 0.0
    assert means["actor_history_recompute_gap"] == 0.0
    assert "actor_history_suffix_steps" not in means


def test_continuing_window_is_the_suffix_with_the_carry(full_window):
    stats = ActorStats()
    actor = _actor(stats)
    carry = _valid_carry()
    last = _last_step_index(full_window)
    # Everything consumed: a zero-step suffix at the smallest bucket.
    request = actor._carry_history(full_window, carry, last, 0)
    assert request.history.field.shape[0] == ACTOR_HISTORY_MIN_LENGTH
    assert not request.history.field[:, FieldFeature.FIELD_FEATURE__VALID].any()
    assert request.history_carry is carry
    means = stats.drain()
    assert means["actor_history_recompute_frac"] == 0.0
    assert means["actor_history_suffix_steps"] == 0.0
    assert means["actor_history_suffix_rows"] == 0.0
    # Three steps behind: exactly those three, and their packed rows.
    request = actor._carry_history(full_window, carry, last - 3, 0)
    assert int(request.history.field[:, FieldFeature.FIELD_FEATURE__VALID].sum()) == 3
    assert request.history_carry is carry
    means = stats.drain()
    assert means["actor_history_suffix_steps"] == 3.0
    assert means["actor_history_suffix_rows"] == packed_valid_rows(
        request.packed_history
    )


def test_rewrite_and_gap_recompute_from_scratch(full_window):
    stats = ActorStats()
    actor = _actor(stats)
    carry = _valid_carry()
    last = _last_step_index(full_window)
    actor._env.history_rewrite_count = 1
    request = actor._carry_history(full_window, carry, last, 0)
    assert not bool(request.history_carry.valid)
    means = stats.drain()
    assert means["actor_history_recompute_rewrite"] == 1.0
    assert means["actor_history_recompute_gap"] == 0.0
    # The count the carry was taken at: no rewrite since, suffix path.
    request = actor._carry_history(full_window, carry, last, 1)
    assert request.history_carry is carry
    stats.drain()
    # A carried step the window no longer continues from.
    request = actor._carry_history(full_window, carry, last + 5, 1)
    assert not bool(request.history_carry.valid)
    means = stats.drain()
    assert means["actor_history_recompute_gap"] == 1.0
    assert means["actor_history_recompute_rewrite"] == 0.0


def test_no_carry_width_never_records_carry_stats(full_window):
    stats = ActorStats()
    actor = PlayerActor(
        agent=None, env=_StubEnv(), unroll_length=1, learner=None, stats=stats
    )
    assert actor._history_carry_width is None
    assert "actor_history_recompute_frac" not in stats.drain()


def test_mixed_group_stacks_with_an_invalid_fill():
    carry = _valid_carry()
    stacked = _stack_history_carries([HistoryCarry(), carry, HistoryCarry()])
    assert stacked.slot_states.shape == (3, 12, WIDTH)
    np.testing.assert_array_equal(stacked.valid, [False, True, False])
    np.testing.assert_array_equal(stacked.slot_states[1], carry.slot_states)
    assert not stacked.slot_states[0].any()
    # No carry anywhere: the empty carry, i.e. the encoder's static branch.
    empty = _stack_history_carries([HistoryCarry(), HistoryCarry()])
    assert isinstance(empty.valid, tuple)
