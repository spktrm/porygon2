"""Hindsight labels for turns a side's active Pokemon began asleep, read
from the simulator's protocol log (the service writes one per game under
BATTLE_LOG_DIR).

Diagnostic only: whether the Pokemon woke is known after the turn, never at
the decision, so nothing here may select a runtime rule.

The labels follow `data/ps/sim/data/conditions.ts` (`slp.onBeforeMove`): the
sleep counter ticks at the move attempt before anything else and whatever
the move, so waking does not depend on the action chosen; `cant|...|slp` is
written BEFORE the `sleepUsable` check, so a successful Sleep Talk logs it
too; and Sleep Talk fails while still asleep when no move is callable
(`moves.ts`, `sleeptalk.onHit`). A sleep `cant` therefore says nothing about
whether the turn was wasted.
"""

import enum
import functools
import re
from dataclasses import dataclass
from pathlib import Path

MOVES_PATH = Path(__file__).resolve().parents[2] / "data/ps/sim/data/moves.ts"
_NAME = re.compile(r'^\s*name: "(.+)",\s*$')
# `replace` (Illusion ending) is absent on purpose: it renames the Pokemon
# already in play, carries no condition field and changes no status.
_SWITCH_COMMANDS = ("switch", "drag")
_END_COMMANDS = ("win", "tie")


@functools.cache
def sleep_usable_moves() -> frozenset[str]:
    """Display names of the moves flagged `sleepUsable` in the vendored move
    data; every other move is an ordinary move."""
    names = set()
    name = None
    for line in MOVES_PATH.read_text().splitlines():
        matched = _NAME.match(line)
        if matched:
            name = matched.group(1)
        elif "sleepUsable: true" in line and name is not None:
            names.add(name)
    if not names:
        raise ValueError(f"no sleepUsable move found in {MOVES_PATH}")
    return frozenset(names)


class SleepTurn(enum.Enum):
    STAYED_ASLEEP = "stayed_asleep"
    WOKE = "woke"
    # No own move attempt (fainted first, switched or dragged out, cured by
    # an item or ability, game over): the counter never ticked.
    AMBIGUOUS = "ambiguous"


class OwnAction(enum.Enum):
    ORDINARY_WASTED = "ordinary_wasted"
    ORDINARY_EXECUTED = "ordinary_executed"
    SLEEP_USABLE_EXECUTED = "sleep_usable_executed"
    # Sleep Talk worked and the move it drew failed (a called Rest while
    # asleep): chance, not the decision, so it is never counted as wasted.
    CALLED_MOVE_FAILED = "called_move_failed"
    ASLEEP_FAILURE = "asleep_failure"
    WAKING_FAILURE = "waking_failure"
    INTERRUPTED = "interrupted"
    NO_ATTEMPT = "no_attempt"


WASTED_ACTIONS = frozenset(
    {OwnAction.ORDINARY_WASTED, OwnAction.ASLEEP_FAILURE, OwnAction.WAKING_FAILURE}
)


@dataclass(frozen=True)
class SleepDecision:
    turn: int
    sleep: SleepTurn
    action: OwnAction
    move: str | None
    logged_sleep_cant: bool

    @property
    def wasted(self) -> bool:
        return self.action in WASTED_ACTIONS


@dataclass
class _TurnEvents:
    turn: int
    sleep_cant: bool = False
    woke: bool = False
    move: str | None = None
    failed: bool = False
    called_failed: bool = False
    closed: bool = False


def _asleep_in(condition: str) -> bool:
    return "slp" in condition.split()[1:]


def _classify(events: _TurnEvents) -> SleepDecision:
    usable = events.move in sleep_usable_moves()
    if events.sleep_cant:
        sleep = SleepTurn.STAYED_ASLEEP
        if events.move is None:
            action = OwnAction.ORDINARY_WASTED
        elif events.failed:
            action = OwnAction.ASLEEP_FAILURE
        elif events.called_failed:
            action = OwnAction.CALLED_MOVE_FAILED
        elif usable:
            action = OwnAction.SLEEP_USABLE_EXECUTED
        else:
            action = OwnAction.INTERRUPTED
    elif events.woke:
        sleep = SleepTurn.WOKE
        if events.move is None:
            action = OwnAction.INTERRUPTED
        elif usable and events.failed:
            action = OwnAction.WAKING_FAILURE
        elif usable:
            action = OwnAction.SLEEP_USABLE_EXECUTED
        else:
            action = OwnAction.ORDINARY_EXECUTED
    else:
        sleep = SleepTurn.AMBIGUOUS
        action = OwnAction.NO_ATTEMPT
    return SleepDecision(
        turn=events.turn,
        sleep=sleep,
        action=action,
        move=events.move,
        logged_sleep_cant=events.sleep_cant,
    )


def sleep_decisions(lines: list[str], side: str = "p1") -> list[SleepDecision]:
    """One record per turn `side`'s active Pokemon began asleep (singles)."""
    active = f"{side}a"
    decisions = []
    asleep = False
    events = None
    previous_command = None
    last_own_move = None

    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        command = parts[1]
        own = len(parts) > 2 and parts[2].startswith(active)

        if command == "turn" or command in _END_COMMANDS:
            if events is not None:
                decisions.append(_classify(events))
            events = None
            if command == "turn" and asleep:
                events = _TurnEvents(turn=int(parts[2]))
            last_own_move = None
        elif command in _SWITCH_COMMANDS:
            if own:
                asleep = _asleep_in(parts[4])
                if events is not None:
                    events.closed = True
            last_own_move = None
        elif command == "faint":
            if own:
                asleep = False
                if events is not None:
                    events.closed = True
        elif command == "-status" and own:
            asleep = parts[3] == "slp"
        elif command == "-curestatus" and own and parts[3] == "slp":
            asleep = False
            tagged = any(part.startswith("[from]") for part in parts[4:])
            natural = not tagged and previous_command != "-enditem"
            if events is not None and not events.closed:
                if natural and events.move is None:
                    events.woke = True
                else:
                    events.closed = True
        elif command == "cant":
            if own and events is not None and not events.closed:
                if parts[3] == "slp":
                    events.sleep_cant = True
            last_own_move = None
        elif command == "move":
            last_own_move = None
            if own and events is not None and not events.closed:
                called = any(part.startswith("[from]") for part in parts[4:])
                if called:
                    last_own_move = "called"
                elif events.move is None:
                    events.move = parts[3]
                    last_own_move = "declared"
        elif command == "-fail":
            if events is not None and not events.closed:
                if last_own_move == "declared":
                    events.failed = True
                elif last_own_move == "called":
                    events.called_failed = True
        previous_command = command

    # A turn the log never closed (no next `turn`, no `win`/`tie`) is an
    # incomplete observation, not an AMBIGUOUS one: dropped.
    return decisions


def read_sleep_decisions(path: Path, side: str = "p1") -> list[SleepDecision]:
    return sleep_decisions(path.read_text().splitlines(), side)


def sleep_summary(decisions: list[SleepDecision]) -> dict:
    """Counts per label, and the wasted-turn rate over the decisions that
    reached a move attempt (AMBIGUOUS turns are counted, never rated)."""
    attempted = [item for item in decisions if item.sleep != SleepTurn.AMBIGUOUS]
    summary = dict(
        decisions=len(decisions),
        attempted=len(attempted),
        by_sleep={
            label.value: sum(item.sleep == label for item in decisions)
            for label in SleepTurn
        },
        by_action={
            label.value: sum(item.action == label for item in decisions)
            for label in OwnAction
        },
        wasted_rate=None,
    )
    if attempted:
        summary["wasted_rate"] = sum(item.wasted for item in attempted) / len(attempted)
    return summary
