"""The wire's structured action mask -> the (src, tgt) grid the model scores.

The mirror of `cellsFromStructuredMask` in `service/src/tests/harness.ts`,
which asserts the same contract from the other side. Bit positions index the
ActionEnum slot lists, never raw enum values, so these tests are also what
pins the slot lists themselves as the cross-language contract.
"""

import numpy as np
import pytest

from rl.environment.data import (
    ALLY_SWITCH_INDICES,
    MOVE_INDICES,
    RESERVE_ENTITY_INDICES,
    TARGET_SLOT_INDICES,
    TEAM_PREVIEW_TGT,
)
from rl.environment.protos.service_pb2 import ActionMask, ActionRequestKind
from rl.environment.utils import _grid_from_structured_mask, get_action_mask

TARGET_AUTO_BIT = list(TARGET_SLOT_INDICES).index(TEAM_PREVIEW_TGT)


def make_mask(kind, **kwargs):
    fields = dict(move_targets=[0] * len(MOVE_INDICES), **kwargs)
    return ActionMask(kind=kind, **fields)


def cells(grid):
    return {(int(src), int(tgt)) for src, tgt in np.argwhere(grid)}


def test_move_slot_lights_only_its_own_row():
    mask = make_mask(ActionRequestKind.ACTION_REQUEST_KIND__MOVE)
    mask.move_targets[3] = 1 << TARGET_AUTO_BIT
    assert cells(_grid_from_structured_mask(mask)) == {
        (MOVE_INDICES[3], TEAM_PREVIEW_TGT)
    }

    # Positive control: the same bit on a different move slot moves the cell,
    # so the test could distinguish a wrong slot list from a right one.
    other = make_mask(ActionRequestKind.ACTION_REQUEST_KIND__MOVE)
    other.move_targets[7] = 1 << TARGET_AUTO_BIT
    assert cells(_grid_from_structured_mask(other)) == {
        (MOVE_INDICES[7], TEAM_PREVIEW_TGT)
    }


def test_clearing_one_bit_clears_exactly_one_cell():
    mask = make_mask(ActionRequestKind.ACTION_REQUEST_KIND__MOVE)
    mask.move_targets[0] = (1 << TARGET_AUTO_BIT) | (1 << 0)
    both = cells(_grid_from_structured_mask(mask))
    mask.move_targets[0] = 1 << TARGET_AUTO_BIT
    one = cells(_grid_from_structured_mask(mask))
    assert len(both) == 2 and len(one) == 1
    assert one < both


@pytest.mark.parametrize("active_slot", [0, 1])
def test_active_slot_picks_the_ally_half(active_slot):
    """`switch_slots` says which mon comes in; `active_slot` which active it
    replaces. Without the second a singles battle legalises ALLY_2_SWITCH and
    the model can pick a cell the service cannot decode."""
    mask = make_mask(
        ActionRequestKind.ACTION_REQUEST_KIND__MOVE,
        switch_slots=0b000101,
        active_slot=active_slot,
    )
    expected_src = ALLY_SWITCH_INDICES[active_slot]
    assert cells(_grid_from_structured_mask(mask)) == {
        (expected_src, RESERVE_ENTITY_INDICES[0]),
        (expected_src, RESERVE_ENTITY_INDICES[2]),
    }


def test_team_preview_names_the_mon_in_the_src_half():
    """One cell per candidate. Until 2026-08-29 preview lit one cell per
    REMAINING POSITION -- up to 7 -- while the decoder ignored the target, so
    the policy spread its mass over that many exact duplicates."""
    preview = make_mask(
        ActionRequestKind.ACTION_REQUEST_KIND__TEAM_PREVIEW, switch_slots=0b111111
    )
    assert cells(_grid_from_structured_mask(preview)) == {
        (int(reserve), TEAM_PREVIEW_TGT) for reserve in RESERVE_ENTITY_INDICES
    }

    # Positive control: the identical switch_slots under a MOVE request puts
    # the same mons in the TGT half instead, so `kind` is genuinely read.
    battle = make_mask(
        ActionRequestKind.ACTION_REQUEST_KIND__MOVE, switch_slots=0b111111
    )
    assert cells(_grid_from_structured_mask(battle)) == {
        (int(ALLY_SWITCH_INDICES[0]), int(reserve))
        for reserve in RESERVE_ENTITY_INDICES
    }


def test_standalone_actions_sit_on_the_diagonal():
    mask = make_mask(ActionRequestKind.ACTION_REQUEST_KIND__FORCE_SWITCH)
    mask.other_srcs = 1 << 5
    slot = int(TARGET_SLOT_INDICES[5])
    assert cells(_grid_from_structured_mask(mask)) == {(slot, slot)}


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
    grid = _grid_from_structured_mask(make_mask(kind))
    assert grid.all()


def test_bundled_states_decode_to_real_decisions():
    """End to end on ex.bin: the fixture must carry the structured mask, and
    every state must offer at least one and far fewer than 1681 choices."""
    from rl.environment.data import EX_BATCH

    states = EX_BATCH.trajectories[0].states
    assert states, "ex.bin has no states"
    reachable = set()
    for move_slot in MOVE_INDICES:
        for target in TARGET_SLOT_INDICES:
            reachable.add((int(move_slot), int(target)))
    for switch_src in ALLY_SWITCH_INDICES:
        for reserve in RESERVE_ENTITY_INDICES:
            reachable.add((int(switch_src), int(reserve)))
    for reserve in RESERVE_ENTITY_INDICES:
        reachable.add((int(reserve), TEAM_PREVIEW_TGT))
    for slot in TARGET_SLOT_INDICES:
        reachable.add((int(slot), int(slot)))

    for state in states:
        assert state.HasField("structured_action_mask")
        assert not state.packed_action_mask, "the packed grid is retired"
        grid = get_action_mask(state)
        assert grid.sum() >= 1
        assert cells(grid) <= reachable


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
