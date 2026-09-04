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
    NUM_TARGET_SLOTS,
    OTHER_CELL_OFFSET,
    RESERVE_ENTITY_INDICES,
    TARGET_SLOT_INDICES,
)
from rl.model.constants import (
    CLS_ROW,
    MOVE_ROWS,
    NUM_SEQUENCE_ROWS,
    PRIVATE_ROWS,
    SEQUENCE_GROUP_IDS,
    SEQUENCE_READ_MASK,
    SEQUENCE_SLICES,
    TARGET_ROWS,
    SequenceGroup,
)
from rl.model.heads import FlatActionReadout
from rl.model.trunk import Trunk

READ_MASK = jnp.asarray(SEQUENCE_READ_MASK)

WIDTH = 32


def _readout():
    return FlatActionReadout(ConfigDict(dict(qk_size=WIDTH)))


def _rows(key):
    keys = jax.random.split(key, 3)
    return (
        jax.random.normal(keys[0], (len(RESERVE_ENTITY_INDICES), WIDTH)),
        jax.random.normal(keys[1], (len(MOVE_INDICES), WIDTH)),
        jax.random.normal(keys[2], (len(TARGET_SLOT_INDICES), WIDTH)),
    )


def _init():
    head = _readout()
    rows = _rows(jax.random.key(0))
    params = head.init(jax.random.key(1), *rows)
    return head, params, rows


# --- layout ---------------------------------------------------------------


def test_sequence_layout_is_derived_and_contiguous():
    assert NUM_SEQUENCE_ROWS == 80
    assert len(SEQUENCE_GROUP_IDS) == NUM_SEQUENCE_ROWS
    covered = []
    for group, sl in SEQUENCE_SLICES.items():
        covered.extend(range(sl.start, sl.stop))
        assert (SEQUENCE_GROUP_IDS[sl] == int(group)).all()
    assert covered == list(range(NUM_SEQUENCE_ROWS))
    assert SEQUENCE_SLICES[SequenceGroup.CLS].start == CLS_ROW
    # The three slices the readout owns are disjoint -- an off-by-one would
    # hand a head someone else's rows and nothing else would notice.
    assert PRIVATE_ROWS.stop == MOVE_ROWS.start
    assert MOVE_ROWS.stop == TARGET_ROWS.start


# --- the readout's init contract ------------------------------------------


def test_every_logit_is_exactly_zero_at_init():
    """The policy starts UNIFORM over legal cells, so
    compute_policy_metrics(prior=None) is the consistent anchor and the PG
    has no lecun noise posing as an action preference to unlearn."""
    head, params, rows = _init()
    logits = head.apply(params, *rows)
    assert logits.shape[-1] == NUM_ACTION_CELLS
    np.testing.assert_array_equal(np.asarray(logits), 0.0)


def test_zero_init_query_gets_live_gradient_and_the_key_unfreezes():
    """The two-factor stall guard (CLAUDE.md 13: a learned grid behind a
    zero-init scale sat at lecun init for 60k steps).

    `query` is the zero factor, but its gradient is a rank-1 outer product
    of LIVE rows, so it moves at step 1. `key`'s gradient is proportional to
    query and is therefore exactly zero for one step -- and the control
    below is that nudging query off zero unfreezes it, i.e. this is a
    one-step unfreeze and not a stalled product.
    """
    head, params, rows = _init()

    def total(p):
        return jnp.sum(head.apply(p, *rows))

    grads = jax.grad(total)(params)["params"]
    assert np.abs(np.asarray(grads["query"]["kernel"])).max() > 0
    assert np.abs(np.asarray(grads["local_src"]["kernel"])).max() > 0
    assert np.abs(np.asarray(grads["local_tgt"]["kernel"])).max() > 0
    assert np.abs(np.asarray(grads["switch"]["kernel"])).max() > 0
    np.testing.assert_array_equal(np.asarray(grads["key"]["kernel"]), 0.0)

    nudged = jax.tree.map(lambda x: x, params)
    nudged["params"]["query"]["kernel"] = nudged["params"]["query"]["kernel"] + 1e-2
    key_grad = jax.grad(total)(nudged)["params"]["key"]["kernel"]
    assert np.abs(np.asarray(key_grad)).max() > 0


def test_the_pointer_is_not_symmetric():
    """A move-src/target-tgt cell is not a target-src/move-tgt cell, so
    query and key must not share one projection."""
    head, params, rows = _init()
    p = jax.tree.map(lambda x: x, params)
    p["params"]["query"]["kernel"] = jax.random.normal(
        jax.random.key(3), p["params"]["query"]["kernel"].shape
    )
    _, move_rows, target_rows = rows
    logits = np.asarray(head.apply(p, rows[0], move_rows, target_rows))
    block = logits[MOVE_CELL_OFFSET:OTHER_CELL_OFFSET].reshape(
        len(MOVE_INDICES), NUM_TARGET_SLOTS
    )
    square = min(block.shape)
    assert not np.allclose(
        block[:square, :square], block[:square, :square].T, atol=1e-6
    )


# --- the readout writes only the cells its modality owns -------------------


def _open(params, seed=5):
    """Non-zero every zero-init leaf, so the grid actually varies."""
    keys = iter(jax.random.split(jax.random.key(seed), 8))
    return jax.tree.map(lambda x: x + jax.random.normal(next(keys), x.shape), params)


@pytest.mark.parametrize("reserve", [0, 3])
def test_a_sheet_row_moves_only_its_own_switch_cell(reserve):
    head, params, rows = _init()
    params = _open(params)
    private_rows, move_rows, target_rows = rows

    bumped = private_rows.at[reserve].add(1.0)
    base = np.asarray(head.apply(params, *rows))
    moved = np.asarray(head.apply(params, bumped, move_rows, target_rows))

    changed = ~np.isclose(base, moved, atol=1e-6)
    expected = np.zeros_like(changed)
    expected[reserve] = True
    np.testing.assert_array_equal(changed, expected)


def test_a_move_row_moves_only_its_own_move_cells():
    head, params, rows = _init()
    params = _open(params)
    private_rows, move_rows, target_rows = rows

    bumped = move_rows.at[2].add(1.0)
    base = np.asarray(head.apply(params, *rows))
    moved = np.asarray(head.apply(params, private_rows, bumped, target_rows))

    changed = ~np.isclose(base, moved, atol=1e-6)
    expected = np.zeros_like(changed)
    row_start = MOVE_CELL_OFFSET + 2 * NUM_TARGET_SLOTS
    expected[row_start : row_start + NUM_TARGET_SLOTS] = True
    np.testing.assert_array_equal(changed, expected)
    # Control: it did change something, so the invariance above is not the
    # readout simply ignoring its move rows.
    assert changed.any()


# --- the trunk ------------------------------------------------------------


def _trunk_cfg(num_blocks=2):
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


def test_every_block_has_its_own_weights():
    trunk = Trunk(_trunk_cfg(num_blocks=3))
    sequence = jnp.zeros((NUM_SEQUENCE_ROWS, WIDTH))
    valid = jnp.ones(NUM_SEQUENCE_ROWS, bool)
    params = trunk.init(jax.random.key(0), sequence, valid, READ_MASK)
    leaves = jax.tree.leaves(params)
    assert leaves, "trunk has no params"
    for leaf in leaves:
        assert leaf.shape[0] == 3, leaf.shape


def test_the_cls_row_survives_a_fully_masked_step():
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


def test_an_invalid_row_is_inert():
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
