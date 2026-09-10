"""Unilateral interval boundaries and game-level leakage controls."""

from types import SimpleNamespace

import numpy as np
import pytest

from rl.environment.protos.features_pb2 import FieldFeature, InfoFeature
from rl.offline.interval_data import game_split, iter_intervals


def make_chunk(
    requests: tuple[int, ...] = (0, 1, 2, 2), terminal: int | None = 2, offset: int = 0
) -> SimpleNamespace:
    info = np.zeros((len(requests), len(InfoFeature.keys())), dtype=np.int32)
    info[:, InfoFeature.INFO_FEATURE__REQUEST_COUNT] = requests
    info[:, InfoFeature.INFO_FEATURE__NUM_ACTIVE] = 1
    done = np.zeros(len(requests), dtype=bool)
    if terminal is not None:
        done[terminal] = True
    history = np.zeros((5, len(FieldFeature.keys())), dtype=np.int32)
    history[:4, FieldFeature.FIELD_FEATURE__VALID] = 1
    history[:, FieldFeature.FIELD_FEATURE__REQUEST_COUNT] = [0, 1, 1, 2, 99]
    return SimpleNamespace(
        player_transitions=SimpleNamespace(
            env_output=SimpleNamespace(
                info=info,
                done=done,
                action_mask=np.ones((len(requests), 3), dtype=bool),
                opp_private_team=np.array([987654]),
            ),
            agent_output=SimpleNamespace(
                actor_output=SimpleNamespace(
                    action_head=SimpleNamespace(
                        action_index=np.arange(len(requests)) % 3
                    )
                )
            ),
        ),
        player_history=SimpleNamespace(field=history),
        game_step_offset=np.array([offset]),
        game_length=np.array([offset + len(requests)]),
    )


def test_terminal_padding_and_future_history_are_excluded() -> None:
    intervals = list(iter_intervals(make_chunk()))
    assert [(entry.source_row, entry.successor_row) for entry in intervals] == [
        (0, 1),
        (1, 2),
    ]
    assert intervals[0].history_indices == (1, 2)
    assert intervals[1].history_indices == (3,)
    assert [entry.terminal for entry in intervals] == [False, True]
    assert [entry.action_index for entry in intervals] == [0, 1]


def test_bootstrap_overlap_produces_unique_game_steps() -> None:
    first = list(iter_intervals(make_chunk((0, 1, 2), terminal=None)))
    second = list(iter_intervals(make_chunk((2, 3, 4), terminal=2, offset=2)))
    assert [entry.game_step for entry in first + second] == [0, 1, 2, 3]


def test_doubles_previous_choice_microstep_is_not_dropped() -> None:
    chunk = make_chunk((0, 0, 1), terminal=2)
    info = chunk.player_transitions.env_output.info
    info[:, InfoFeature.INFO_FEATURE__NUM_ACTIVE] = 2
    info[1, InfoFeature.INFO_FEATURE__HAS_PREV_ACTION] = 1
    first, second = iter_intervals(chunk)
    assert first.source_request == first.successor_request
    assert first.history_indices == ()
    assert first.successor_has_previous_action
    assert second.has_previous_action and second.num_active == 2


def test_visibility_and_legal_action_positive_controls() -> None:
    chunk = make_chunk()
    before = list(iter_intervals(chunk))
    chunk.player_transitions.env_output.opp_private_team[:] = -1
    assert list(iter_intervals(chunk)) == before
    chunk.player_history.field[1, FieldFeature.FIELD_FEATURE__VALID] = 0
    assert list(iter_intervals(chunk))[0].history_indices == (2,)
    chunk.player_transitions.env_output.action_mask[0, 0] = False
    with pytest.raises(ValueError, match="Illegal submitted"):
        list(iter_intervals(chunk))


def test_missing_history_coverage_is_explicit() -> None:
    chunk = make_chunk()
    chunk.player_history.field[0, FieldFeature.FIELD_FEATURE__VALID] = 0
    assert not list(iter_intervals(chunk))[0].history_prefix_retained


def test_split_is_shared_by_game_and_order_independent() -> None:
    identities = [f"collection:game-{index}" for index in range(100)]
    original = {identity: game_split(identity) for identity in identities}
    assert {
        identity: game_split(identity) for identity in reversed(identities)
    } == original
    assert set(original.values()) == {"train", "heldout"}
    assert game_split("same-game", heldout_fraction=0) == "train"
    assert game_split("same-game", heldout_fraction=1) == "heldout"
    with pytest.raises(ValueError):
        game_split("")


@pytest.mark.parametrize("request_type", [1, 2, 3])
def test_request_kinds_retain_unilateral_action_contract(request_type: int) -> None:
    chunk = make_chunk()
    chunk.player_transitions.env_output.info[
        :, InfoFeature.INFO_FEATURE__REQUEST_TYPE
    ] = request_type
    intervals = list(iter_intervals(chunk))
    assert len(intervals) == 2
    assert all(entry.request_type == request_type for entry in intervals)


def test_game_length_excludes_nonterminal_padding_and_rejects_reverse_requests() -> (
    None
):
    chunk = make_chunk(terminal=None)
    chunk.game_length = np.array([2])
    assert len(list(iter_intervals(chunk))) == 1
    chunk.player_transitions.env_output.info[
        1, InfoFeature.INFO_FEATURE__REQUEST_COUNT
    ] = -1
    with pytest.raises(ValueError, match="must not decrease"):
        list(iter_intervals(chunk))


def test_evaluation_partitions_reserve_final_games_and_exclude_unused_training() -> (
    None
):
    from rl.offline.interval_data import evaluation_partitions

    arrays = {
        "game": np.array(["train", "train", "validation", "test", "test", "unused"]),
        "heldout": np.array([False, False, True, True, True, False]),
        "final_test": np.array([False, False, False, True, True, False]),
        "train_eligible": np.array([True, True, False, False, False, False]),
    }
    partitions = evaluation_partitions(arrays)
    for actual, expected in zip(partitions, [[0, 1], [2], [3, 4]]):
        np.testing.assert_array_equal(actual, expected)
    arrays["train_eligible"][3] = True
    with pytest.raises(ValueError, match="training must be disjoint"):
        evaluation_partitions(arrays)
    arrays["train_eligible"][3] = False
    arrays["final_test"][4] = False
    with pytest.raises(ValueError, match="Game crosses"):
        evaluation_partitions(arrays)


def test_evaluation_partitions_preserve_original_split() -> None:
    from rl.offline.interval_data import evaluation_partitions

    arrays = {"game": np.array(["train", "test"]), "heldout": np.array([False, True])}
    train, validation, final = evaluation_partitions(arrays)
    np.testing.assert_array_equal(train, [0])
    np.testing.assert_array_equal(validation, [1])
    assert not len(final)
