"""The wire's structured action mask -> the block cells the model scores.

The mirror of `cellsFromStructuredMask` in `service/src/tests/harness.ts`,
which asserts the same contract from the other side. Bit positions index the
ActionEnum slot lists, never raw enum values, and the block offsets derive
from the slot-list lengths, so these tests are also what pins the layout as
the cross-language contract.
"""

import numpy as np
import pytest

from rl.environment.data import (
    ALLY_SWITCH_INDICES,
    MOVE_CELL_OFFSET,
    MOVE_INDICES,
    NUM_ACTION_CELLS,
    NUM_ACTION_FEATURES,
    NUM_SWITCH_CELLS,
    NUM_TARGET_SLOTS,
    OTHER_CELL_OFFSET,
    RESERVE_ENTITY_INDICES,
    TARGET_SLOT_INDICES,
)
from rl.environment.protos.service_pb2 import ActionMask, ActionRequestKind
from rl.environment.utils import (
    _cells_from_packed_grid,
    _cells_from_structured_mask,
    get_action_mask,
)


def make_mask(kind, **kwargs):
    fields = dict(move_targets=[0] * len(MOVE_INDICES), **kwargs)
    return ActionMask(kind=kind, **fields)


def cells(mask_vector):
    return {int(cell) for cell in np.flatnonzero(mask_vector)}


def move_cell(move_slot, target_bit):
    return MOVE_CELL_OFFSET + move_slot * NUM_TARGET_SLOTS + target_bit


def test_move_slot_lights_only_its_own_cells():
    mask = make_mask(ActionRequestKind.ACTION_REQUEST_KIND__MOVE)
    mask.move_targets[3] = 1 << 2
    assert cells(_cells_from_structured_mask(mask)) == {move_cell(3, 2)}

    # Positive control: the same bit on a different move slot moves the cell,
    # so the test could distinguish a wrong offset from a right one.
    other = make_mask(ActionRequestKind.ACTION_REQUEST_KIND__MOVE)
    other.move_targets[7] = 1 << 2
    assert cells(_cells_from_structured_mask(other)) == {move_cell(7, 2)}


def test_clearing_one_bit_clears_exactly_one_cell():
    mask = make_mask(ActionRequestKind.ACTION_REQUEST_KIND__MOVE)
    mask.move_targets[0] = (1 << 2) | (1 << 0)
    both = cells(_cells_from_structured_mask(mask))
    mask.move_targets[0] = 1 << 2
    one = cells(_cells_from_structured_mask(mask))
    assert len(both) == 2 and len(one) == 1
    assert one < both


@pytest.mark.parametrize(
    "kind",
    [
        ActionRequestKind.ACTION_REQUEST_KIND__MOVE,
        ActionRequestKind.ACTION_REQUEST_KIND__FORCE_SWITCH,
        ActionRequestKind.ACTION_REQUEST_KIND__TEAM_PREVIEW,
    ],
)
def test_switch_bits_are_kind_invariant(kind):
    """One question -- "may this mon come in" -- one cell per reserve. The
    kind (and the ally half) matter only to the service's DECODER, which
    picks between a lead and a `switch` choice string; the cells are
    identical, which is exactly what lets the readout serve preview and
    battle switches with one head. `active_slot` likewise no longer moves
    the mask -- harness.ts asserts the decode side of that contract."""
    mask = make_mask(kind, switch_slots=0b000101, active_slot=0)
    assert cells(_cells_from_structured_mask(mask)) == {0, 2}
    other_half = make_mask(kind, switch_slots=0b000101, active_slot=1)
    assert cells(_cells_from_structured_mask(other_half)) == {0, 2}


def test_standalone_actions_light_the_other_block():
    mask = make_mask(ActionRequestKind.ACTION_REQUEST_KIND__FORCE_SWITCH)
    mask.other_srcs = 1 << 5
    assert cells(_cells_from_structured_mask(mask)) == {OTHER_CELL_OFFSET + 5}


@pytest.mark.parametrize(
    "kind",
    [
        ActionRequestKind.ACTION_REQUEST_KIND___UNSPECIFIED,
        ActionRequestKind.ACTION_REQUEST_KIND__WAIT,
    ],
)
def test_a_request_with_no_choice_is_all_legal(kind):
    """Nothing is being asked. Every cell stays legal so masked averages
    downstream never meet an empty row, and the service answers "default" for
    whichever cell comes back."""
    mask_vector = _cells_from_structured_mask(make_mask(kind))
    assert mask_vector.shape == (NUM_ACTION_CELLS,)
    assert mask_vector.all()


def test_legacy_grid_folds_onto_block_cells():
    """The replay-shard shim: every reachable class of grid cell must land on
    its block cell -- battle switch (ALLY_i_SWITCH src, RESERVE_j tgt) and
    team-preview lead (RESERVE_j src) both onto switch cell j, a move onto
    its (slot, target) cell, a diagonal standalone onto the other block."""
    grid = np.zeros((NUM_ACTION_FEATURES, NUM_ACTION_FEATURES), dtype=bool)
    grid[ALLY_SWITCH_INDICES[0], RESERVE_ENTITY_INDICES[2]] = True
    grid[RESERVE_ENTITY_INDICES[4], TARGET_SLOT_INDICES[0]] = True
    grid[MOVE_INDICES[3], TARGET_SLOT_INDICES[2]] = True
    grid[TARGET_SLOT_INDICES[5], TARGET_SLOT_INDICES[5]] = True
    assert cells(_cells_from_packed_grid(grid)) == {
        2,
        4,
        move_cell(3, 2),
        OTHER_CELL_OFFSET + 5,
    }

    # The WAIT sentinel -- an all-lit grid -- folds onto all-lit cells.
    assert _cells_from_packed_grid(np.ones_like(grid)).all()


def test_bundled_states_decode_to_real_decisions():
    """End to end on ex.bin: the fixture must carry the structured mask, and
    every state must offer at least one choice over the 295 cells."""
    from rl.environment.data import EX_BATCH

    states = EX_BATCH.trajectories[0].states
    assert states, "ex.bin has no states"
    for state in states:
        assert state.HasField("structured_action_mask")
        assert not state.packed_action_mask, "the packed grid is retired"
        mask_vector = get_action_mask(state)
        assert mask_vector.shape == (NUM_ACTION_CELLS,)
        assert mask_vector.sum() >= 1


def test_switch_block_size_matches_reserves():
    assert NUM_SWITCH_CELLS == len(RESERVE_ENTITY_INDICES)
    assert NUM_ACTION_CELLS == (
        NUM_SWITCH_CELLS + len(MOVE_INDICES) * NUM_TARGET_SLOTS + NUM_TARGET_SLOTS
    )


def test_uniform_kl_gradient_is_pi_minus_one_over_k():
    """The zero-avoiding term's whole justification, checked numerically.

    d/d y_b KL(u || pi) = pi_b - 1/k: bounded by 1 for ANY pi, zero-sum over
    legal cells, and with NO pi prefactor -- which is why it still acts on a
    cell the policy has abandoned. The control is the entropy bonus at the
    same starved cell, whose gradient is pi-prefactored and therefore ~0
    there; that contrast is the reason this term exists.
    """
    import jax
    import jax.numpy as jnp

    from rl.model.utils import legal_log_policy
    from rl.online.training.loss import uniform_kl_rows

    legal = jnp.asarray([True, True, True, True, False])
    k = int(legal.sum())
    # One cell starved to ~1e-6 of the mass.
    logits = jnp.asarray([0.0, 0.1, -0.2, -13.0, 0.0])

    def loss(y):
        return uniform_kl_rows(legal_log_policy(y, legal), legal)

    grad = np.asarray(jax.grad(loss)(logits))
    pi = np.asarray(jnp.exp(legal_log_policy(logits, legal)))
    expected = np.where(np.asarray(legal), pi - 1.0 / k, 0.0)
    np.testing.assert_allclose(grad, expected, atol=1e-5)

    assert abs(grad.sum()) < 1e-5, "must be zero-sum over legal cells"
    assert np.abs(grad).max() <= 1.0 + 1e-6, "must be bounded by 1"
    # The starved cell still feels a full -1/k pull.
    starved = 3
    assert grad[starved] == pytest.approx(-1.0 / k, abs=1e-5)

    def entropy(y):
        log_pi = legal_log_policy(y, legal)
        return (jnp.exp(log_pi) * jnp.where(legal, log_pi, 0.0)).sum()

    entropy_grad = np.asarray(jax.grad(entropy)(logits))
    assert abs(entropy_grad[starved]) < 1e-3, (
        "control: the entropy bonus is pi-prefactored and is numerically "
        "dead at this cell, which is the deficit the uniform KL fills"
    )
