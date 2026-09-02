"""`clip_history_suffix` (2026-09-02): the actor's incremental window is the
full window's steps after the carried point, token for token, with the
packed rows they name and the index columns rebased so every reference
still lands on the same row."""

import jax
import numpy as np
import pytest

from rl.environment.protos.features_pb2 import FieldFeature
from rl.environment.utils import (
    _ALL_RELEVANT_IDX_COLUMNS,
    ACTOR_HISTORY_MIN_LENGTH,
    _bucket_level,
    _bucket_value,
    _packed_valid_rows,
    clip_history_suffix,
    get_ex_player_step,
)

VALID = FieldFeature.FIELD_FEATURE__VALID
INDEX = FieldFeature.FIELD_FEATURE__INDEX
NUM_RELEVANT = FieldFeature.FIELD_FEATURE__NUM_RELEVANT


@pytest.fixture(scope="module")
def full_window():
    actor_input, _ = get_ex_player_step()
    return jax.tree.map(lambda x: np.asarray(x[:, 0]), actor_input)


def _valid_steps(actor_input) -> int:
    return int(np.asarray(actor_input.history.field)[:, VALID].sum())


def test_nothing_consumed_is_the_full_window(full_window):
    valid_steps = _valid_steps(full_window)
    assert valid_steps > 0
    suffix, new_steps = clip_history_suffix(full_window, last_step_index=-1)
    assert new_steps == valid_steps
    field_full = np.asarray(full_window.history.field)
    field_cut = np.asarray(suffix.history.field)
    assert np.array_equal(field_cut[:valid_steps], field_full[:valid_steps])
    assert not field_cut[valid_steps:].any()
    packed_valid = _packed_valid_rows(full_window.packed_history)
    for name in ("revealed_cache", "public_cache", "edge_cache"):
        full = np.asarray(getattr(full_window.packed_history, name))
        cut = np.asarray(getattr(suffix.packed_history, name))
        assert np.array_equal(cut[:packed_valid], full[:packed_valid]), name
        assert not cut[packed_valid:].any(), name


@pytest.mark.parametrize("consumed", [1, 2, 37, 100, 160])
def test_suffix_is_the_tail_token_for_token(full_window, consumed):
    field_full = np.asarray(full_window.history.field)
    valid_steps = _valid_steps(full_window)
    assert consumed < valid_steps
    last = int(field_full[consumed - 1, INDEX])
    suffix, new_steps = clip_history_suffix(full_window, last_step_index=last)
    assert new_steps == valid_steps - consumed

    field_cut = np.asarray(suffix.history.field)
    start_row = int(
        field_full[consumed, FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX0]
    )
    expected = field_full[consumed:valid_steps].copy()
    rebase_at = np.ix_(np.arange(new_steps), _ALL_RELEVANT_IDX_COLUMNS)
    packed_rows = suffix.packed_history.revealed_cache.shape[0]
    expected[rebase_at] = np.clip(expected[rebase_at] - start_row, 0, packed_rows - 1)
    assert np.array_equal(field_cut[:new_steps], expected)
    assert not field_cut[new_steps:].any()

    # Every live reference resolves to the same packed row it did before.
    packed_valid = _packed_valid_rows(full_window.packed_history)
    for name in ("revealed_cache", "public_cache", "edge_cache"):
        full = np.asarray(getattr(full_window.packed_history, name))
        cut = np.asarray(getattr(suffix.packed_history, name))
        assert np.array_equal(
            cut[: packed_valid - start_row], full[start_row:packed_valid]
        )
        assert not cut[packed_valid - start_row :].any(), name
        for step in range(new_steps):
            num_relevant = int(field_full[consumed + step, NUM_RELEVANT])
            old_rows = field_full[
                consumed + step, _ALL_RELEVANT_IDX_COLUMNS[:num_relevant]
            ]
            new_rows = field_cut[step, _ALL_RELEVANT_IDX_COLUMNS[:num_relevant]]
            assert np.array_equal(cut[new_rows], full[old_rows]), (name, step)

    # Shapes are the actor bucket values of what was kept, not of the window.
    assert field_cut.shape[0] == _bucket_value(
        _bucket_level(new_steps, ACTOR_HISTORY_MIN_LENGTH),
        ACTOR_HISTORY_MIN_LENGTH,
        field_full.shape[0],
    )
    assert packed_rows == _bucket_value(
        _bucket_level(packed_valid - start_row, ACTOR_HISTORY_MIN_LENGTH),
        ACTOR_HISTORY_MIN_LENGTH,
        full_window.packed_history.revealed_cache.shape[0],
    )


def test_zero_new_steps_is_an_empty_window(full_window):
    field_full = np.asarray(full_window.history.field)
    last = int(field_full[_valid_steps(full_window) - 1, INDEX])
    suffix, new_steps = clip_history_suffix(full_window, last_step_index=last)
    assert new_steps == 0
    assert not np.asarray(suffix.history.field).any()
    assert suffix.history.field.shape[0] == ACTOR_HISTORY_MIN_LENGTH
    assert not any(
        np.asarray(leaf).any() for leaf in jax.tree.leaves(suffix.packed_history)
    )
    assert suffix.packed_history.revealed_cache.shape[0] == ACTOR_HISTORY_MIN_LENGTH


def test_an_empty_window_resumes_only_from_nothing(full_window):
    field = np.asarray(full_window.history.field).copy()
    field[:, VALID] = 0
    empty = full_window.replace(history=full_window.history.replace(field=field))
    suffix, new_steps = clip_history_suffix(empty, last_step_index=-1)
    assert suffix is not None and new_steps == 0
    assert clip_history_suffix(empty, last_step_index=3) == (None, 0)


def test_a_gap_refuses_to_resume(full_window):
    field_full = np.asarray(full_window.history.field)
    valid_steps = _valid_steps(full_window)
    # The service dropped the window's oldest 10 steps: a carry ending
    # before them cannot be resumed, one ending exactly at the cut can.
    dropped = 10
    field = np.zeros_like(field_full)
    field[: valid_steps - dropped] = field_full[dropped:valid_steps]
    shifted = full_window.replace(history=full_window.history.replace(field=field))
    boundary = int(field_full[dropped - 1, INDEX])
    assert clip_history_suffix(shifted, last_step_index=boundary - 1) == (None, 0)
    assert (
        clip_history_suffix(shifted, last_step_index=boundary)[1]
        == valid_steps - dropped
    )
    # A carry from beyond the window's end is not in it either.
    beyond = int(field_full[valid_steps - 1, INDEX]) + 5
    assert clip_history_suffix(full_window, last_step_index=beyond) == (None, 0)
