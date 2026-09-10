"""The eval-leak gate: no eval trajectory ever reaches replay."""

from rl.online.guards import EVAL_USERNAME_PREFIX, should_push_trajectory


def test_eval_actor_never_pushes() -> None:
    assert not should_push_trajectory(True, True, "player-3")


def test_unpushed_never_pushes() -> None:
    assert not should_push_trajectory(False, False, "player-3")


def test_eval_username_never_pushes_even_without_the_flag() -> None:
    # The `thresholded` slot samples a distribution no training actor
    # uses; its env username alone must keep it out of replay.
    username = f"{EVAL_USERNAME_PREFIX}-simpleheuristic-1:0002"
    assert not should_push_trajectory(False, True, username)


def test_training_actor_pushes() -> None:
    # Positive control: the gate can open.
    assert should_push_trajectory(False, True, "player-3")
