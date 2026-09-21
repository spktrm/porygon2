"""Contracts of the flat action readout and the one-sequence trunk.

Each test carries the control that proves it could fail: zero-init paths and
masked routes make invariance tests pass vacuously, which is the trap the
2026-08-25 privileged-critic work paid for twice.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import ConfigDict

from rl.environment.data import (
    MOVE_CELL_OFFSET,
    MOVE_INDICES,
    NUM_ACTION_CELLS,
    NUM_SWITCH_CELLS,
    NUM_TARGET_SLOTS,
    OTHER_CELL_OFFSET,
    RESERVE_ENTITY_INDICES,
    TARGET_SLOT_INDICES,
)
from rl.model.constants import (
    CLS_ROW,
    MOVE_ROWS,
    NUM_PUBLIC_SLOTS,
    NUM_SEQUENCE_ROWS,
    PRIVATE_ROWS,
    SEQUENCE_GROUP_IDS,
    SEQUENCE_READ_MASK,
    SEQUENCE_SLICES,
    TARGET_ROWS,
    SequenceGroup,
)
from rl.model.heads import FlatActionReadout, ReadoutRows
from rl.model.trunk import Trunk

READ_MASK = jnp.asarray(SEQUENCE_READ_MASK)

WIDTH = 32


def _readout() -> FlatActionReadout:
    return FlatActionReadout(ConfigDict(dict(qk_size=WIDTH)))


# Twelve public rows, mine then theirs, actives first. Singles: one active a
# side. Their slots 4 and 5: one fainted, one unrevealed-and-absent.
PUBLIC_ALIVE = jnp.array([True] * 6 + [True, True, True, True, False, False])
PUBLIC_ACTIVE = jnp.array([True] + [False] * 5 + [True] + [False] * 5)
# Doubles: both of my slots and both of theirs on the field.
DOUBLES_ACTIVE = jnp.array([True, True] + [False] * 4 + [True, True] + [False] * 4)


def _rows(key: jax.Array) -> ReadoutRows:
    keys = jax.random.split(key, 4)
    return ReadoutRows(
        private=jax.random.normal(keys[0], (len(RESERVE_ENTITY_INDICES), WIDTH)),
        move=jax.random.normal(keys[1], (len(MOVE_INDICES), WIDTH)),
        target=jax.random.normal(keys[2], (len(TARGET_SLOT_INDICES), WIDTH)),
        public=jax.random.normal(keys[3], (NUM_PUBLIC_SLOTS, WIDTH)),
        public_alive=PUBLIC_ALIVE,
        public_active=PUBLIC_ACTIVE,
    )


def _without_their_team(rows: ReadoutRows) -> ReadoutRows:
    return rows._replace(public_alive=rows.public_alive.at[6:].set(False))


def _init() -> tuple[FlatActionReadout, dict, ReadoutRows]:
    head = _readout()
    rows = _rows(jax.random.key(0))
    params = head.init(jax.random.key(1), rows)
    return head, params, rows


def test_sequence_layout_is_derived_and_contiguous() -> None:
    assert NUM_SEQUENCE_ROWS == 99
    assert len(SEQUENCE_GROUP_IDS) == NUM_SEQUENCE_ROWS
    covered = []
    for group, sl in SEQUENCE_SLICES.items():
        covered.extend(range(sl.start, sl.stop))
        assert (SEQUENCE_GROUP_IDS[sl] == int(group)).all()
    assert covered == list(range(NUM_SEQUENCE_ROWS))
    assert SEQUENCE_SLICES[SequenceGroup.CLS].start == CLS_ROW
    # The three slices the readout owns are disjoint -- an off-by-one would
    # hand a head someone else's rows and nothing else would notice. They
    # are no longer adjacent (TARGET_SLOT sits in the public tier), so the
    # check is disjointness itself.
    owned = [
        set(range(*sl.indices(NUM_SEQUENCE_ROWS)))
        for sl in (PRIVATE_ROWS, MOVE_ROWS, TARGET_ROWS)
    ]
    assert sum(len(rows) for rows in owned) == len(set().union(*owned))


def test_every_logit_is_exactly_zero_at_init() -> None:
    """The policy starts UNIFORM over legal cells, so
    compute_policy_metrics(prior=None) is the consistent anchor and the PG
    has no lecun noise posing as an action preference to unlearn."""
    head, params, rows = _init()
    logits = head.apply(params, rows)
    assert logits.shape[-1] == NUM_ACTION_CELLS
    np.testing.assert_array_equal(np.asarray(logits), 0.0)


def test_zero_init_query_gets_live_gradient_and_the_key_unfreezes() -> None:
    """The two-factor stall guard (LESSONS.md 13: a learned grid behind a
    zero-init scale sat at lecun init for 60k steps).

    `query` is the zero factor, but its gradient is a rank-1 outer product
    of LIVE rows, so it moves at step 1. `key`'s gradient is proportional to
    query and is therefore exactly zero for one step -- and the control
    below is that nudging query off zero unfreezes it, i.e. this is a
    one-step unfreeze and not a stalled product.
    """
    head, params, rows = _init()

    def total(p: dict) -> jax.Array:
        return jnp.sum(head.apply(p, rows))

    grads = jax.grad(total)(params)["params"]
    assert np.abs(np.asarray(grads["move_query"]["kernel"])).max() > 0
    assert np.abs(np.asarray(grads["move_score"]["kernel"])).max() > 0
    assert np.abs(np.asarray(grads["move_target_score"]["kernel"])).max() > 0
    assert np.abs(np.asarray(grads["switch_score"]["kernel"])).max() > 0
    assert np.abs(np.asarray(grads["switch_query"]["kernel"])).max() > 0
    assert np.abs(np.asarray(grads["switch_target_score"]["kernel"])).max() > 0
    np.testing.assert_array_equal(np.asarray(grads["move_key"]["kernel"]), 0.0)
    np.testing.assert_array_equal(np.asarray(grads["switch_key"]["kernel"]), 0.0)

    nudged = jax.tree.map(lambda x: x, params)
    nudged["params"]["move_query"]["kernel"] = (
        nudged["params"]["move_query"]["kernel"] + 1e-2
    )
    key_grad = jax.grad(total)(nudged)["params"]["move_key"]["kernel"]
    assert np.abs(np.asarray(key_grad)).max() > 0

    nudged["params"]["switch_query"]["kernel"] = (
        nudged["params"]["switch_query"]["kernel"] + 1e-2
    )
    switch_key_grad = jax.grad(total)(nudged)["params"]["switch_key"]["kernel"]
    assert np.abs(np.asarray(switch_key_grad)).max() > 0


def test_the_pointer_is_not_symmetric() -> None:
    """A move-src/target-tgt cell is not a target-src/move-tgt cell, so
    query and key must not share one projection."""
    head, params, rows = _init()
    p = jax.tree.map(lambda x: x, params)
    p["params"]["move_query"]["kernel"] = jax.random.normal(
        jax.random.key(3), p["params"]["move_query"]["kernel"].shape
    )
    logits = np.asarray(head.apply(p, rows))
    block = logits[MOVE_CELL_OFFSET:OTHER_CELL_OFFSET].reshape(
        len(MOVE_INDICES), NUM_TARGET_SLOTS
    )
    square = min(block.shape)
    assert not np.allclose(
        block[:square, :square], block[:square, :square].T, atol=1e-6
    )


def _open(params: dict, seed: int = 5) -> dict:
    """Non-zero every zero-init leaf, so the grid actually varies."""
    keys = iter(jax.random.split(jax.random.key(seed), 64))
    return jax.tree.map(lambda x: x + jax.random.normal(next(keys), x.shape), params)


@pytest.mark.parametrize("reserve", [0, 3])
def test_a_sheet_row_moves_only_its_own_switch_cell(reserve: int) -> None:
    head, params, rows = _init()
    params = _open(params)
    bumped = rows._replace(private=rows.private.at[reserve].add(1.0))
    base = np.asarray(head.apply(params, rows))
    moved = np.asarray(head.apply(params, bumped))

    changed = ~np.isclose(base, moved, atol=1e-6)
    expected = np.zeros_like(changed)
    expected[reserve] = True
    np.testing.assert_array_equal(changed, expected)


@pytest.mark.parametrize("decision_slot", [0, 1])
def test_my_active_being_replaced_moves_every_switch_cell(decision_slot: int) -> None:
    """The switch block pairs the candidate with my active in the deciding
    slot. In singles the other slot is empty, so bumping its row moves
    nothing -- the control that this is not "any public row"; in doubles
    the partner is present and its row moves every switch cell too."""
    head, params, rows = _init()
    params = _open(params)
    doubles = rows._replace(public_active=DOUBLES_ACTIVE)

    def switch_cells(rows: ReadoutRows, public: jax.Array) -> np.ndarray:
        logits = head.apply(
            params, rows._replace(public=public), decision_slot=decision_slot
        )
        return np.asarray(logits[:NUM_SWITCH_CELLS])

    leaving, partner = decision_slot, 1 - decision_slot
    for game in (rows, doubles):
        base = switch_cells(game, game.public)
        moved = switch_cells(game, game.public.at[leaving].add(1.0))
        assert (~np.isclose(base, moved, atol=1e-6)).all()
    base = switch_cells(doubles, doubles.public)
    moved = switch_cells(doubles, doubles.public.at[partner].add(1.0))
    assert (~np.isclose(base, moved, atol=1e-6)).all()
    # Singles decides slot 0 only; slot 1 holds no partner, so its row is
    # inert.
    if decision_slot == 0:
        base = switch_cells(rows, rows.public)
        np.testing.assert_array_equal(
            base, switch_cells(rows, rows.public.at[partner].add(1.0))
        )


def test_a_move_row_moves_only_its_own_move_cells() -> None:
    head, params, rows = _init()
    params = _open(params)
    bumped = rows._replace(move=rows.move.at[2].add(1.0))
    base = np.asarray(head.apply(params, rows))
    moved = np.asarray(head.apply(params, bumped))

    changed = ~np.isclose(base, moved, atol=1e-6)
    expected = np.zeros_like(changed)
    row_start = MOVE_CELL_OFFSET + 2 * NUM_TARGET_SLOTS
    expected[row_start : row_start + NUM_TARGET_SLOTS] = True
    np.testing.assert_array_equal(changed, expected)
    # Control: it did change something, so the invariance above is not the
    # readout simply ignoring its move rows.
    assert changed.any()


def _opponent_term(params: dict, src: jax.Array, rows: ReadoutRows, block: str):
    """The opponent-team term written out longhand from the leaves."""
    leaves = params["params"]["opponent_team"]
    scale = 1.0 / np.sqrt(WIDTH)
    key = rows.opponent @ leaves["opponent_key"]["kernel"]
    belief_key = rows.opponent @ leaves["belief_key"]["kernel"]
    score = (rows.opponent @ leaves["opponent_score"]["kernel"])[:, 0]
    pair = (src @ leaves[f"{block}_opponent_query"]["kernel"]) @ key.T * scale + score
    belief_logits = (src @ leaves[f"{block}_belief_query"]["kernel"]) @ belief_key.T
    belief_logits = jnp.where(rows.opponent_alive, belief_logits * scale, -1e9)
    belief = jax.nn.softmax(belief_logits, axis=-1) * rows.opponent_alive
    return np.asarray((belief * pair).sum(-1))


def test_opponent_term_is_the_belief_weighted_pair_over_alive_rows() -> None:
    """Switch and move logits carry the same opponent-team term, written out
    from the leaves: uniform belief over the alive rows at init (the belief
    query is zero), the readout's learned belief once opened."""
    head, params, rows = _init()
    opened = _open(params)
    for p in (params, opened):
        logits = np.asarray(head.apply(p, rows))
        without = np.asarray(head.apply(p, _without_their_team(rows)))
        switch_term = logits[:NUM_SWITCH_CELLS] - without[:NUM_SWITCH_CELLS]
        np.testing.assert_allclose(
            switch_term, _opponent_term(p, rows.private, rows, "switch"), atol=1e-4
        )
        move_block = (logits - without)[MOVE_CELL_OFFSET:OTHER_CELL_OFFSET].reshape(
            len(MOVE_INDICES), NUM_TARGET_SLOTS
        )
        # One value per move, broadcast over its target cells.
        np.testing.assert_allclose(
            move_block, np.broadcast_to(move_block[:, :1], move_block.shape), atol=1e-4
        )
        np.testing.assert_allclose(
            move_block[:, 0], _opponent_term(p, rows.move, rows, "move"), atol=1e-4
        )
    # Positive control: opened, the term is not zero.
    opened_logits = np.asarray(head.apply(opened, rows))
    assert np.abs(_opponent_term(opened, rows.private, rows, "switch")).max() > 1e-3
    assert not np.allclose(opened_logits, np.asarray(head.apply(params, rows)))


def test_a_fainted_or_absent_opponent_row_never_reaches_a_logit() -> None:
    head, params, rows = _init()
    # Open the pair queries only: with the belief left uniform over the
    # alive rows, every alive row reaches every cell (a saturated belief
    # would hide a row behind a near-zero weight and blunt the control).
    team = params["params"]["opponent_team"]
    for name in ("switch_opponent_query", "move_opponent_query"):
        team[name]["kernel"] = jax.random.normal(
            jax.random.key(7), team[name]["kernel"].shape
        )
    base = np.asarray(head.apply(params, rows))
    for dead in (6 + 4, 6 + 5):
        bumped = rows._replace(public=rows.public.at[dead].add(3.0))
        np.testing.assert_array_equal(base, np.asarray(head.apply(params, bumped)))
    # Control: an alive row moves every switch cell and every move cell.
    bumped = rows._replace(public=rows.public.at[6 + 1].add(3.0))
    moved = np.asarray(head.apply(params, bumped))
    changed = ~np.isclose(base, moved, atol=1e-6)
    assert changed[:OTHER_CELL_OFFSET].all()
    assert not changed[OTHER_CELL_OFFSET:].any()


def test_the_opponent_key_is_shared_and_trained_by_the_move_block() -> None:
    """The switch block's sparse gradient is not what has to train the
    opponent factors: a loss over the MOVE cells alone reaches
    `opponent_key`, `belief_key` and `opponent_score`, the same leaves the
    switch term reads."""
    head, params, rows = _init()
    nudged = jax.tree.map(lambda x: x, params)
    team = nudged["params"]["opponent_team"]
    for name in ("move_opponent_query", "move_belief_query"):
        team[name]["kernel"] = team[name]["kernel"] + 1e-2

    def move_loss(p: dict) -> jax.Array:
        return jnp.sum(head.apply(p, rows)[MOVE_CELL_OFFSET:OTHER_CELL_OFFSET])

    grads = jax.grad(move_loss)(nudged)["params"]["opponent_team"]
    for name in ("opponent_key", "belief_key", "opponent_score"):
        assert np.abs(np.asarray(grads[name]["kernel"])).max() > 0, name
    # And the switch block's own queries move from step 1 (zero factor over
    # live keys), while a move-only loss leaves them untouched.
    switch_grads = jax.grad(lambda p: jnp.sum(head.apply(p, rows)[:NUM_SWITCH_CELLS]))(
        params
    )["params"]["opponent_team"]
    assert np.abs(np.asarray(switch_grads["switch_opponent_query"]["kernel"])).max() > 0
    np.testing.assert_array_equal(grads["switch_opponent_query"]["kernel"], 0.0)


def test_team_preview_switch_logits_read_the_opponent_team() -> None:
    """No active exists at preview, so the leaving and partner pairs are
    silent and only the opponent-team term separates candidates beyond
    their own score."""
    head, params, rows = _init()
    params = _open(params)
    preview = rows._replace(public_active=jnp.zeros(NUM_PUBLIC_SLOTS, bool))
    with_team = np.asarray(head.apply(params, preview))[:NUM_SWITCH_CELLS]
    no_team = np.asarray(head.apply(params, _without_their_team(preview)))[
        :NUM_SWITCH_CELLS
    ]
    assert not np.allclose(with_team, no_team, atol=1e-6)
    np.testing.assert_allclose(
        with_team - no_team,
        _opponent_term(params, rows.private, preview, "switch"),
        atol=1e-4,
    )


def _trunk_cfg(num_blocks: int = 2) -> ConfigDict:
    return ConfigDict(
        dict(
            num_blocks=num_blocks,
            num_heads=2,
            qk_size=WIDTH // 2,
            v_size=WIDTH // 2,
            model_size=WIDTH,
            qk_layer_norm=True,
            use_bias=True,
            hidden_size=2 * WIDTH,
        )
    )


def test_every_block_has_its_own_weights() -> None:
    trunk = Trunk(_trunk_cfg(num_blocks=3))
    sequence = jnp.zeros((NUM_SEQUENCE_ROWS, WIDTH))
    valid = jnp.ones(NUM_SEQUENCE_ROWS, bool)
    params = trunk.init(jax.random.key(0), sequence, valid, READ_MASK)
    leaves = jax.tree.leaves(params)
    assert leaves, "trunk has no params"
    for leaf in leaves:
        assert leaf.shape[0] == 3, leaf.shape


def test_the_cls_row_survives_a_fully_masked_step() -> None:
    """A terminal step masks every action row off. Masked attention uses a
    -1e9 floor rather than -inf, so an empty key set is finite either way --
    what the unconditionally-valid CLS row actually buys is that the value
    head still reads a real vector there instead of the hard zero every
    masked row is set to.
    """
    trunk = Trunk(_trunk_cfg())
    sequence = jax.random.normal(jax.random.key(0), (NUM_SEQUENCE_ROWS, WIDTH))
    valid = jnp.zeros(NUM_SEQUENCE_ROWS, bool).at[CLS_ROW].set(True)
    params = trunk.init(jax.random.key(1), sequence, valid, READ_MASK)

    out = np.asarray(trunk.apply(params, sequence, valid, READ_MASK))
    assert np.isfinite(out).all()
    assert np.abs(out[CLS_ROW]).max() > 0

    # Control: mask the CLS row too and the value head's input is exactly
    # zero -- a constant, carrying nothing about the state.
    empty = jnp.zeros(NUM_SEQUENCE_ROWS, bool)
    blanked = np.asarray(trunk.apply(params, sequence, empty, READ_MASK))
    assert np.isfinite(blanked).all()
    np.testing.assert_array_equal(blanked[CLS_ROW], 0.0)


def test_an_invalid_row_is_inert() -> None:
    trunk = Trunk(_trunk_cfg())
    sequence = jax.random.normal(jax.random.key(0), (NUM_SEQUENCE_ROWS, WIDTH))
    valid = jnp.ones(NUM_SEQUENCE_ROWS, bool).at[5].set(False)
    params = trunk.init(jax.random.key(1), sequence, valid, READ_MASK)

    perturbed = sequence.at[5].add(10.0)
    base = np.asarray(trunk.apply(params, sequence, valid, READ_MASK))
    moved = np.asarray(trunk.apply(params, perturbed, valid, READ_MASK))
    np.testing.assert_allclose(base, moved, atol=0)

    # Control: mark it valid and the same perturbation reaches the others.
    live = jnp.ones(NUM_SEQUENCE_ROWS, bool)
    assert not np.allclose(
        np.asarray(trunk.apply(params, sequence, live, READ_MASK)),
        np.asarray(trunk.apply(params, perturbed, live, READ_MASK)),
    )
