"""The depth-1 event rollout operator on stubs (rl/model/event_search.py):
own-action forcing, the decision-boundary rule, Q ranking by the stub
value, padding, and the value-blind arm consuming the same rollouts."""

import jax
import jax.numpy as jnp
import numpy as np

from rl.environment.data import NUM_MOVES
from rl.environment.event_labels import NO_SLOT, DeclaredKind, EventKind
from rl.model import event_search as es
from rl.model import world_model as wm
from rl.model.constants import NUM_PUBLIC_SEQUENCE_ROWS

WIDTH = 8
MY_SLOT, THEIR_SLOT = 0, 6


def _root(has_reserve=True, num_revealed=8):
    sides = jnp.asarray([1] * 6 + [0] * 6)
    return es.RootInfo(
        slot_sides=sides,
        num_revealed=jnp.asarray(num_revealed),
        ally_active_slots=jnp.asarray([MY_SLOT, -1]),
        enemy_active_slots=jnp.asarray([THEIR_SLOT, -1]),
        has_reserve=jnp.asarray(has_reserve),
    )


def _logits(kind_of, actor_of, new_turn_of):
    """A decoder stub that puts all mass on a scripted event: `kind_of`,
    `actor_of` and `new_turn_of` read the rows' first channel (the event
    counter the imagine stub advances) so the script can depend on how
    far the rollout has gone."""

    def decode_fn(rows, declared_kind, declared_arg, tokens, actor_is_mine):
        counter = rows[0, 0]
        kind = kind_of(counter)
        actor = actor_of(counter)

        def peaked(size, index):
            return jnp.where(jnp.arange(size) == index, 30.0, 0.0)

        return wm.DecoderLogits(
            kind=peaked(wm.NUM_EVENT_KINDS, kind),
            new_turn=jnp.where(new_turn_of(counter), 30.0, -30.0),
            actor=peaked(wm.NUM_SLOT_CLASSES, actor),
            move=peaked(NUM_MOVES, 100),
            target=peaked(wm.NUM_SLOT_CLASSES, THEIR_SLOT),
            touched=jnp.full(wm.NUM_TOUCHED_BITS, -30.0).at[actor].set(30.0),
        )

    return decode_fn


def _imagine_fn(rows, tokens, row_mask, rng):
    # Advance the event counter in channel 0 and stamp the move played into
    # channel 1 so a test can read what was executed.
    rows = rows.at[0, 0].add(1.0)
    return rows.at[0, 1].set(tokens.move.astype(rows.dtype))


def _fns(decode_fn, value_fn=None):
    if value_fn is None:

        def value_fn(rows):
            return rows[0, 0] * 0.1

    return es.EventSearchFns(
        decode_fn=decode_fn,
        imagine_fn=_imagine_fn,
        value_fn=value_fn,
        terminal_fn=lambda rows: jnp.asarray(-1.0),
    )


def _rows():
    return jnp.zeros((NUM_PUBLIC_SEQUENCE_ROWS, WIDTH))


def test_own_execution_carries_the_declared_move_once() -> None:
    # Their move first, then mine, then a new turn opens after event 3.
    decode = _logits(
        kind_of=lambda c: EventKind.MOVE,
        actor_of=lambda c: jnp.where(c == 0, THEIR_SLOT, MY_SLOT),
        new_turn_of=lambda c: c >= 2,
    )
    final = es.rollout(
        _rows(),
        jnp.asarray(DeclaredKind.MOVE),
        jnp.asarray(77),
        _root(),
        _fns(decode),
        es.RolloutBudget(max_events=6),
        jax.random.key(0),
    )
    # Event 1 (theirs) played the prior's move 100, event 2 (mine) the
    # declared 77, event 3 (mine again, already executed) the prior's 100.
    assert int(final.num_events) == 3
    assert float(final.rows[0, 1]) == 100.0
    assert bool(final.own_executed)
    # Re-run stopping after my execution to read the forced move.
    decode_two = _logits(
        kind_of=lambda c: EventKind.MOVE,
        actor_of=lambda c: jnp.where(c == 0, THEIR_SLOT, MY_SLOT),
        new_turn_of=lambda c: c >= 1,
    )
    final = es.rollout(
        _rows(),
        jnp.asarray(DeclaredKind.MOVE),
        jnp.asarray(77),
        _root(),
        _fns(decode_two),
        es.RolloutBudget(max_events=6),
        jax.random.key(0),
    )
    assert int(final.num_events) == 2 and float(final.rows[0, 1]) == 77.0


def test_boundary_rule() -> None:
    def run(kind_of, actor_of, has_reserve=True, new_turn_of=lambda c: False):
        return es.rollout(
            _rows(),
            jnp.asarray(DeclaredKind.UNKNOWN),
            jnp.asarray(0),
            _root(has_reserve=has_reserve),
            _fns(_logits(kind_of, actor_of, new_turn_of)),
            es.RolloutBudget(max_events=5),
            jax.random.key(1),
        )

    # An own faint with a reserve stops the rollout; without one it does not.
    faint_mine = run(lambda c: EventKind.FAINT, lambda c: MY_SLOT)
    assert int(faint_mine.num_events) == 1 and not bool(faint_mine.ended)
    no_reserve = run(lambda c: EventKind.FAINT, lambda c: MY_SLOT, has_reserve=False)
    assert int(no_reserve.num_events) == 5
    # Their faint is not our decision.
    faint_theirs = run(lambda c: EventKind.FAINT, lambda c: THEIR_SLOT)
    assert int(faint_theirs.num_events) == 5
    # END stops and marks the leaf terminal.
    ended = run(lambda c: EventKind.END, lambda c: NO_SLOT)
    assert int(ended.num_events) == 1 and bool(ended.ended)
    # Padded steps after the stop carry the state unchanged.
    assert float(faint_mine.rows[0, 0]) == 1.0
    # A new turn after the second event stops there.
    turn = run(
        lambda c: EventKind.MOVE, lambda c: THEIR_SLOT, new_turn_of=lambda c: c >= 1
    )
    assert int(turn.num_events) == 2


def test_q_ranks_cells_by_leaf_value_and_pads_cleanly() -> None:
    # The stub value is the event counter; a declared MOVE makes my event
    # execute (counter advances), a declared UNKNOWN too -- so make the
    # value depend on the declared arg through the stamped move.
    decode = _logits(
        kind_of=lambda c: EventKind.MOVE,
        actor_of=lambda c: MY_SLOT,
        new_turn_of=lambda c: c >= 0,
    )
    fns = _fns(decode, value_fn=lambda rows: rows[0, 1] / 100.0)
    declared_kinds = jnp.asarray(
        [DeclaredKind.MOVE, DeclaredKind.MOVE, DeclaredKind.MOVE]
    )
    declared_args = jnp.asarray([30, 90, 60])
    cell_valid = jnp.asarray([True, True, False])
    q, finals = es.q_values(
        _rows(),
        declared_kinds,
        declared_args,
        cell_valid,
        _root(),
        fns,
        es.RolloutBudget(max_events=3),
        4,
        jax.random.key(2),
    )
    np.testing.assert_allclose(np.asarray(q), [0.3, 0.9, 0.0], atol=1e-6)
    assert int(finals.num_events[0, 0]) == 1
    bonus = es.search_bonus(q, cell_valid, temp=0.5, value_blind=False)
    np.testing.assert_allclose(np.asarray(bonus), [0.6, 1.8, 0.0], atol=1e-6)
    blind = es.search_bonus(q, cell_valid, temp=0.5, value_blind=True)
    assert not np.asarray(blind).any()
    diagnostics = es.search_diagnostics(jnp.zeros(3), bonus, cell_valid)
    assert float(diagnostics["search_root_kl"]) > 0
    zero = es.search_diagnostics(jnp.zeros(3), blind, cell_valid)
    np.testing.assert_allclose(float(zero["search_root_kl"]), 0.0, atol=1e-6)


def test_new_slot_actor_is_ours_only_for_our_unplayed_switch() -> None:
    root = _root(num_revealed=8)
    decode = _logits(
        kind_of=lambda c: EventKind.SWITCH,
        actor_of=lambda c: 8,  # the next unrevealed slot
        new_turn_of=lambda c: c >= 0,
    )
    tokens, touched, new_turn, mine = es.sample_event(
        _rows(),
        es.RolloutState(
            _rows(),
            jnp.asarray(8),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(0),
        ),
        jnp.asarray(DeclaredKind.SWITCH),
        jnp.asarray(8),
        root,
        _fns(decode),
        jax.random.key(3),
        1.0,
    )
    assert int(tokens.actor) == 8 and bool(mine)
    tokens, touched, new_turn, theirs = es.sample_event(
        _rows(),
        es.RolloutState(
            _rows(),
            jnp.asarray(8),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(0),
        ),
        jnp.asarray(DeclaredKind.MOVE),
        jnp.asarray(50),
        root,
        _fns(decode),
        jax.random.key(3),
        1.0,
    )
    assert int(tokens.actor) == 8 and not bool(theirs)
