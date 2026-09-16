import jax.numpy as jnp
import numpy as np

from rl.environment import event_labels as labels
from rl.environment.protos.enums_pb2 import BattlemajorargsEnum, MovesEnum
from rl.environment.protos.features_pb2 import (
    EntityEdgeFeature,
    EntityPublicNodeFeature,
    EntityRevealedNodeFeature,
    FieldFeature,
)
from rl.environment.utils import get_ex_trajectory
from rl.model import history_encoder
from rl.model.constants import NUM_PUBLIC_SLOTS, RELEVANT_ENTITY_FEATURES

MOVE = BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__MOVE
SWITCH = BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__SWITCH
DRAG = BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__DRAG
CANT = BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__CANT
FAINT = BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__FAINT
PAD = BattlemajorargsEnum.BATTLEMAJORARGS_ENUM___PAD
NUM_EDGE = len(EntityEdgeFeature.keys())
NUM_PUBLIC = len(EntityPublicNodeFeature.keys())
NUM_REVEALED = len(EntityRevealedNodeFeature.keys())
NUM_FIELD = len(FieldFeature.keys())
MINE, THEIRS = 1, 0
MOVE_A, MOVE_B = 50, 70


class Stream:
    """Builds a packed history by hand: one call per step, one row per
    touched entity, in the service's layout."""

    def __init__(self):
        self.field, self.edge, self.public, self.revealed = [], [], [], []
        self.turn = 0

    def step(self, rows, *, new_turn=False, weather=0, request_count=0):
        if new_turn:
            self.turn += 1
        field = np.zeros(NUM_FIELD, np.int32)
        field[FieldFeature.FIELD_FEATURE__VALID] = 1
        field[FieldFeature.FIELD_FEATURE__TURN_VALUE] = self.turn
        # never 0 on the wire; the label must not read it
        field[FieldFeature.FIELD_FEATURE__TURN_ORDER_VALUE] = 2
        field[FieldFeature.FIELD_FEATURE__WEATHER_ID] = weather
        field[FieldFeature.FIELD_FEATURE__REQUEST_COUNT] = request_count
        field[FieldFeature.FIELD_FEATURE__NUM_RELEVANT] = len(rows)
        for k, (slot, side, major, move, known) in enumerate(rows):
            field[RELEVANT_ENTITY_FEATURES[k]] = len(self.edge)
            edge = np.full(NUM_EDGE, PAD, np.int32)
            edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG] = major
            edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__ENTITY_IDX] = slot
            edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN] = move
            public = np.zeros(NUM_PUBLIC, np.int32)
            public[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE] = side
            revealed = np.zeros(NUM_REVEALED, np.int32)
            for j, move_id in enumerate(known):
                revealed[
                    EntityRevealedNodeFeature.Value(
                        f"ENTITY_REVEALED_NODE_FEATURE__MOVEID{j}"
                    )
                ] = move_id
            self.edge.append(edge)
            self.public.append(public)
            self.revealed.append(revealed)
        self.field.append(field)

    def arrays(self, pad_steps=2):
        field = np.stack(self.field + [np.zeros(NUM_FIELD, np.int32)] * pad_steps)
        return (
            field,
            np.stack(self.edge),
            np.stack(self.public),
            np.stack(self.revealed),
        )


def hand_stream() -> Stream:
    s = Stream()
    none = MovesEnum.MOVES_ENUM___PAD
    s.step([(0, MINE, SWITCH, none, ())], new_turn=True)  # my lead
    s.step([(1, THEIRS, SWITCH, none, ())])  # their lead
    # turn 1: they move first, I flinch (cant carries the move id)
    s.step(
        [(1, THEIRS, MOVE, MOVE_A, (MOVE_A,)), (0, MINE, PAD, none, ())], new_turn=True
    )
    s.step([(0, MINE, CANT, MOVE_B, (MOVE_B,))])
    # turn 2: I move (previously revealed by the cant), weather starts, I faint, I switch in slot 2
    s.step(
        [(0, MINE, MOVE, MOVE_B, (MOVE_B,)), (1, THEIRS, PAD, none, (MOVE_A,))],
        new_turn=True,
        weather=3,
    )
    s.step([(0, MINE, FAINT, none, (MOVE_B,))], weather=3)
    s.step([(2, MINE, SWITCH, none, ())], weather=3)
    # turn 3: a drag is never my declaration; a cant without a move is UNKNOWN
    s.step([(2, MINE, CANT, none, ())], new_turn=True, weather=3)
    s.step([(3, MINE, DRAG, none, ())], weather=3)
    return s


def test_hand_stream_labels():
    out = labels.event_labels(*hand_stream().arrays())
    K = labels.EventKind
    kinds = [K.SWITCH, K.MOVE, K.CANT, K.MOVE, K.FAINT, K.SWITCH, K.CANT, K.DRAG, K.END]
    assert out.kind[:9].tolist() == [int(k) for k in kinds]
    assert out.valid[:9].all() and not out.valid[9:].any()
    assert out.terminal[8] and out.terminal.sum() == 1
    # event 2 (their move) from state 1: actor slot 1, their side, target my slot 0
    assert out.actor[1] == 1 and out.actor_side[1] == THEIRS and out.target[1] == 0
    assert out.touched[1].tolist() == [True, True] + [False] * 10
    assert out.move[1] == MOVE_A and out.move_valid[1]
    assert not out.move_previously_revealed[1]  # slot 1 had no move on file
    # event 4 (my move B) was revealed by the cant at event 3
    assert out.move_previously_revealed[3]
    assert out.field_touched[3] and not out.field_touched[4]
    assert out.new_turn[1] and out.new_turn[3] and out.new_turn[6]
    # declarations governing each next event: leads = my switch to 0,
    # turn 1 = move B (cant carries it), turn 2 = move B, then the forced
    # switch to 2 after my faint, turn 3 = UNKNOWN (cant without a move,
    # and the drag never counts)
    D = labels.DeclaredKind
    assert out.declared_kind[0] == D.SWITCH and out.declared_arg[0] == 0
    assert out.declared_kind[1] == D.MOVE and out.declared_arg[1] == MOVE_B
    assert out.declared_kind[3] == D.MOVE and out.declared_arg[3] == MOVE_B
    # the faint (event 5) still happens under the move declaration; the
    # forced switch governs event 6
    assert out.declared_kind[4] == D.MOVE and out.declared_arg[4] == MOVE_B
    assert out.declared_kind[5] == D.SWITCH and out.declared_arg[5] == 2
    assert out.declared_kind[6] == D.UNKNOWN and out.declared_kind[7] == D.UNKNOWN
    # boundaries: before each new turn, and after my faint (answered by a switch)
    assert out.boundary[:9].tolist() == [
        False,
        True,
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]
    assert out.num_revealed[:9].tolist() == [1, 2, 2, 2, 2, 2, 3, 3, 4]


def test_unanswered_faint_is_unknown_and_not_a_boundary():
    s = Stream()
    none = MovesEnum.MOVES_ENUM___PAD
    s.step([(0, MINE, MOVE, MOVE_A, ())], new_turn=True)
    s.step([(0, MINE, FAINT, none, ())])
    s.step([(1, THEIRS, MOVE, MOVE_B, ())])
    out = labels.event_labels(*s.arrays())
    assert out.declared_kind[1] == labels.DeclaredKind.UNKNOWN
    assert not out.boundary[1]


def test_numpy_relevant_edges_matches_the_encoder_on_the_fixture():
    trajectory = get_ex_trajectory()
    field = np.asarray(trajectory.history.field)
    ours = labels.relevant_edges(field)
    theirs = history_encoder.relevant_edges(jnp.asarray(field))
    np.testing.assert_array_equal(ours[0], np.asarray(theirs[0]))
    np.testing.assert_array_equal(ours[1], np.asarray(theirs[1]))
    out = labels.event_labels(
        field,
        np.asarray(trajectory.packed_history.edge_cache),
        np.asarray(trajectory.packed_history.public_cache),
        np.asarray(trajectory.packed_history.revealed_cache),
    )
    assert out.valid.sum() == (field[:, FieldFeature.FIELD_FEATURE__VALID] > 0).sum()
    assert out.terminal.sum() == 1 and out.kind[out.terminal][0] == labels.EventKind.END
    assert (out.num_revealed <= NUM_PUBLIC_SLOTS).all()
