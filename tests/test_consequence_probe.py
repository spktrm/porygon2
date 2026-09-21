"""The consequence probe's labels, on hand-built event intervals."""

from types import SimpleNamespace

import numpy as np

from rl.environment.event_labels import SIDE_MINE, EventKind
from rl.environment.protos.features_pb2 import EntityPublicNodeFeature, InfoFeature
from rl.model.constants import NUM_PUBLIC_SLOTS, OPP_ACTIVE_PUBLIC_ROWS
from rl.probes.consequence_probe import interval_labels

_THEIRS = 1 - SIDE_MINE
_ORDER_0 = InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0
_HP = EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO
_OPP_ROW = int(OPP_ACTIVE_PUBLIC_ROWS[0])


def _events(pairs):
    return SimpleNamespace(
        kind=np.asarray([kind for kind, _ in pairs]),
        actor_side=np.asarray([side for _, side in pairs]),
    )


def _env(order_next, hp_next):
    num_info = _ORDER_0 + NUM_PUBLIC_SLOTS
    info = np.zeros((2, 1, num_info), np.int32)
    info[0, 0, _ORDER_0:] = np.arange(NUM_PUBLIC_SLOTS)
    info[1, 0, _ORDER_0:] = order_next
    public_team = np.zeros((2, 1, NUM_PUBLIC_SLOTS, _HP + 1), np.int32)
    public_team[0, 0, _OPP_ROW, _HP] = 100
    public_team[1, 0, :, _HP] = hp_next
    return SimpleNamespace(info=info, public_team=public_team)


def _labels(pairs, order_next=None, hp_next=100):
    if order_next is None:
        order_next = np.arange(NUM_PUBLIC_SLOTS)
    events = _events(pairs)
    interval = np.ones(len(pairs), bool)
    return interval_labels(events, interval, _env(order_next, hp_next), 0, 0)


def test_a_cant_alone_is_not_an_executed_move():
    blocked = _labels([(EventKind.MOVE, _THEIRS), (EventKind.CANT, SIDE_MINE)])
    assert blocked["own_move_executed"] == 0.0
    assert blocked["moved_first"] == 0.0
    # The control: Sleep Talk logs the CANT and the move, and did execute.
    talked = _labels([(EventKind.CANT, SIDE_MINE), (EventKind.MOVE, SIDE_MINE)])
    assert talked["own_move_executed"] == 1.0


def test_labels_are_undefined_without_the_events_that_define_them():
    switched = _labels([(EventKind.SWITCH, SIDE_MINE), (EventKind.MOVE, _THEIRS)])
    assert np.isnan(switched["own_move_executed"])
    assert np.isnan(switched["moved_first"])
    assert switched["any_faint"] == 0.0


def test_hp_is_followed_through_a_reordering_of_the_public_rows():
    # The opponent's active Pokemon moves to the back row after it switches
    # out; its hp is read where its identity went, not at the active row.
    order_next = np.arange(NUM_PUBLIC_SLOTS)
    order_next[[_OPP_ROW, NUM_PUBLIC_SLOTS - 1]] = order_next[
        [NUM_PUBLIC_SLOTS - 1, _OPP_ROW]
    ]
    hp_next = np.full(NUM_PUBLIC_SLOTS, 40)
    hp_next[NUM_PUBLIC_SLOTS - 1] = 100
    moved = _labels([(EventKind.MOVE, SIDE_MINE)], order_next, hp_next)
    assert moved["opp_active_hp_moved"] == 0.0
    hp_next[NUM_PUBLIC_SLOTS - 1] = 55
    assert (
        _labels([(EventKind.MOVE, SIDE_MINE)], order_next, hp_next)[
            "opp_active_hp_moved"
        ]
        == 1.0
    )
