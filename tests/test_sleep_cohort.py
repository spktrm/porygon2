"""The hindsight sleep labels, pinned on logs the service's own simulator
(@pkmn/sim, gen9customgame) wrote for each scenario."""

from rl.probes.sleep_turns import (
    OwnAction,
    SleepTurn,
    sleep_decisions,
    sleep_summary,
    sleep_usable_moves,
)

SLEEP_TALK_UNTIL_WAKING = """
|switch|p1a: Sleeper|Snorlax|524/524
|switch|p2a: Wall|Blissey|714/714
|turn|1
|move|p2a: Wall|Seismic Toss|p1a: Sleeper
|move|p1a: Sleeper|Rest|p1a: Sleeper
|-status|p1a: Sleeper|slp|[from] move: Rest
|turn|2
|move|p2a: Wall|Seismic Toss|p1a: Sleeper
|cant|p1a: Sleeper|slp
|move|p1a: Sleeper|Sleep Talk|p1a: Sleeper
|move|p1a: Sleeper|Rest||[from] move: Sleep Talk|[still]
|-fail|p1a: Sleeper
|turn|3
|move|p2a: Wall|Seismic Toss|p1a: Sleeper
|cant|p1a: Sleeper|slp
|move|p1a: Sleeper|Sleep Talk|p1a: Sleeper
|move|p1a: Sleeper|Body Slam|p2a: Wall|[from] move: Sleep Talk
|turn|4
|move|p2a: Wall|Seismic Toss|p1a: Sleeper
|-curestatus|p1a: Sleeper|slp|[msg]
|move|p1a: Sleeper|Sleep Talk||[still]
|-fail|p1a: Sleeper
|turn|5
|move|p2a: Wall|Seismic Toss|p1a: Sleeper
|move|p1a: Sleeper|Body Slam|p2a: Wall
|turn|6
""".strip().splitlines()

ORDINARY_MOVES_UNTIL_WAKING = """
|switch|p1a: Sleeper|Snorlax|524/524
|switch|p2a: Wall|Blissey|714/714
|turn|1
|move|p2a: Wall|Seismic Toss|p1a: Sleeper
|move|p1a: Sleeper|Rest|p1a: Sleeper
|-status|p1a: Sleeper|slp|[from] move: Rest
|turn|2
|move|p2a: Wall|Seismic Toss|p1a: Sleeper
|cant|p1a: Sleeper|slp
|turn|3
|move|p2a: Wall|Seismic Toss|p1a: Sleeper
|cant|p1a: Sleeper|slp
|turn|4
|move|p2a: Wall|Seismic Toss|p1a: Sleeper
|-curestatus|p1a: Sleeper|slp|[msg]
|move|p1a: Sleeper|Body Slam|p2a: Wall
|turn|5
""".strip().splitlines()

NOTHING_CALLABLE = """
|switch|p1a: Sleeper|Snorlax|524/524
|switch|p2a: Spore|Breloom|261/261
|turn|1
|move|p2a: Spore|Spore|p1a: Sleeper
|-status|p1a: Sleeper|slp|[from] move: Spore
|cant|p1a: Sleeper|slp
|move|p1a: Sleeper|Sleep Talk||[still]
|-fail|p1a: Sleeper
|turn|2
|move|p2a: Spore|Splash|p2a: Spore
|cant|p1a: Sleeper|slp
|move|p1a: Sleeper|Sleep Talk||[still]
|-fail|p1a: Sleeper
|turn|3
""".strip().splitlines()

SWITCHES_OUT_ASLEEP = """
|switch|p1a: Sleeper|Snorlax|524/524
|switch|p2a: Wall|Blissey|714/714
|turn|1
|move|p2a: Wall|Seismic Toss|p1a: Sleeper
|move|p1a: Sleeper|Rest|p1a: Sleeper
|-status|p1a: Sleeper|slp|[from] move: Rest
|turn|2
|switch|p1a: Back|Snorlax|524/524
|move|p2a: Wall|Seismic Toss|p1a: Back
|turn|3
""".strip().splitlines()

FAINTS_BEFORE_MOVING = """
|switch|p1a: Sleeper|Shedinja|1/1
|switch|p2a: Spore|Breloom|261/261
|turn|1
|move|p2a: Spore|Spore|p1a: Sleeper
|-status|p1a: Sleeper|slp|[from] move: Spore
|cant|p1a: Sleeper|slp
|turn|2
|move|p2a: Spore|Rock Tomb|p1a: Sleeper
|faint|p1a: Sleeper
|switch|p1a: Back|Snorlax|524/524
|turn|3
""".strip().splitlines()


def _labels(lines):
    return [(item.turn, item.sleep, item.action) for item in sleep_decisions(lines)]


def test_sleep_usable_set_comes_from_the_move_data():
    assert sleep_usable_moves() == {"Sleep Talk", "Snore"}


def test_successful_sleep_talk_is_not_wasted_despite_its_cant():
    executed = sleep_decisions(SLEEP_TALK_UNTIL_WAKING)[1]
    assert executed.turn == 3
    assert executed.sleep == SleepTurn.STAYED_ASLEEP
    assert executed.action == OwnAction.SLEEP_USABLE_EXECUTED
    assert not executed.wasted
    # The control: "any sleep cant means a wasted turn" calls this one wasted.
    assert executed.logged_sleep_cant


def test_sleep_talk_outcomes_across_one_rest():
    assert _labels(SLEEP_TALK_UNTIL_WAKING) == [
        (2, SleepTurn.STAYED_ASLEEP, OwnAction.CALLED_MOVE_FAILED),
        (3, SleepTurn.STAYED_ASLEEP, OwnAction.SLEEP_USABLE_EXECUTED),
        (4, SleepTurn.WOKE, OwnAction.WAKING_FAILURE),
    ]


def test_ordinary_moves_are_wasted_until_the_cure_event():
    decisions = sleep_decisions(ORDINARY_MOVES_UNTIL_WAKING)
    assert _labels(ORDINARY_MOVES_UNTIL_WAKING) == [
        (2, SleepTurn.STAYED_ASLEEP, OwnAction.ORDINARY_WASTED),
        (3, SleepTurn.STAYED_ASLEEP, OwnAction.ORDINARY_WASTED),
        (4, SleepTurn.WOKE, OwnAction.ORDINARY_EXECUTED),
    ]
    assert [item.wasted for item in decisions] == [True, True, False]


def test_sleep_talk_failing_while_asleep_is_not_a_waking():
    # Turn 1 is absent: the Pokemon was awake when that decision was made.
    assert _labels(NOTHING_CALLABLE) == [
        (2, SleepTurn.STAYED_ASLEEP, OwnAction.ASLEEP_FAILURE),
    ]
    assert sleep_decisions(NOTHING_CALLABLE)[0].wasted


def test_turns_with_no_move_attempt_are_ambiguous():
    for lines in (SWITCHES_OUT_ASLEEP, FAINTS_BEFORE_MOVING):
        assert _labels(lines) == [
            (2, SleepTurn.AMBIGUOUS, OwnAction.NO_ATTEMPT),
        ]


def test_a_turn_the_log_never_closed_is_dropped():
    unfinished = NOTHING_CALLABLE + ["|move|p2a: Spore|Splash|p2a: Spore"]
    assert [item.turn for item in sleep_decisions(unfinished)] == [2]


def test_an_illusion_reveal_is_not_a_switch():
    revealed = list(ORDINARY_MOVES_UNTIL_WAKING)
    revealed.insert(
        revealed.index("|turn|3") + 1, "|replace|p1a: Sleeper|Zoroark, L83, M"
    )
    assert _labels(revealed) == _labels(ORDINARY_MOVES_UNTIL_WAKING)


def test_summary_rates_attempted_turns_only():
    decisions = sleep_decisions(ORDINARY_MOVES_UNTIL_WAKING) + sleep_decisions(
        FAINTS_BEFORE_MOVING
    )
    summary = sleep_summary(decisions)
    assert summary["decisions"] == 4
    assert summary["attempted"] == 3
    assert summary["wasted_rate"] == 2 / 3
    assert sleep_summary([])["wasted_rate"] is None
