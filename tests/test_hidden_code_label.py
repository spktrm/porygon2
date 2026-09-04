"""The belief head's hidden-token label (2026-09-05, encoder.OppCodeLabels).

The label is the code network's reading of the tokens the matched public
row does NOT show. Three contracts, each with the half that proves the
test could fail: the label is blind to a never-hidden token (hp lives on
the state token, which the revealed row reads) while the critic's code is
not; it does read a hidden move; and revealing every token of a mon on
its public row empties the label (`hidden_any` False), where unmatching
the mon fills it again (nothing an unmatched mon carries is public).
"""

import dataclasses

import jax
import numpy as np
import pytest

from rl.environment.protos.features_pb2 import (
    EntityPrivateNodeFeature,
    EntityRevealedNodeFeature,
)
from rl.model.state_features import PUBLIC_MOVE_INDICES

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

PRIVATE_MOVE_COLUMNS = [
    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__MOVEID0,
    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__MOVEID1,
    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__MOVEID2,
    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__MOVEID3,
]
SHARED_ID_COLUMNS = [
    EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES,
    EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ABILITY,
    EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ITEM,
    *PUBLIC_MOVE_INDICES.tolist(),
]


def _with_cell(actor_input, leaf, step, row, column, value):
    env = actor_input.env
    table = np.asarray(getattr(env, leaf)).copy()
    table[step, row, column] = value
    return dataclasses.replace(
        actor_input, env=dataclasses.replace(env, **{leaf: jax.numpy.asarray(table)})
    )


def _hidden_move_slot(actor_input, step, mon, public_row):
    """A private move slot whose id is on no public slot, or None."""
    private = np.asarray(actor_input.env.opp_private_team[step, mon])
    public_moves = np.asarray(actor_input.env.revealed_team[step, public_row])[
        PUBLIC_MOVE_INDICES
    ]
    for slot, column in enumerate(PRIVATE_MOVE_COLUMNS):
        if private[column] not in public_moves and private[column] > 3:
            return slot, column
    return None


def _first_matched_with_hidden_move(actor_input, base):
    from rl.model.player_model import belief_alignment

    for step, mon in np.argwhere(np.asarray(base.belief_matched)):
        _, public_row_index = belief_alignment(
            actor_input.env.opp_private_team[step], actor_input.env.info[step]
        )
        public_row = int(public_row_index[mon])
        found = _hidden_move_slot(actor_input, step, mon, public_row)
        if found is not None:
            return int(step), int(mon), public_row, found
    pytest.skip("no matched opponent mon with a hidden move in the fixture")


def test_hidden_code_reads_hidden_tokens_and_not_the_state_token(
    real_model_and_trajectory, real_model_apply
):
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    base = real_model_apply(params, actor_input, actor_output, HeadParams())
    step, mon, public_row, (slot, move_column) = _first_matched_with_hidden_move(
        actor_input, base
    )
    assert bool(base.belief_hidden_any[step, mon])

    # hp is on the state token, which is never hidden: the critic's code
    # may move, the label must not.
    hp_column = EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__HP_RATIO
    hp_now = int(actor_input.env.opp_private_team[step, mon, hp_column])
    hp_flipped = _with_cell(
        actor_input, "opp_private_team", step, mon, hp_column, max(hp_now // 2, 1)
    )
    hp_moved = real_model_apply(params, hp_flipped, actor_output, HeadParams())
    np.testing.assert_array_equal(
        np.asarray(base.hidden_code[step, mon], np.float32),
        np.asarray(hp_moved.hidden_code[step, mon], np.float32),
    )

    # Positive control: a hidden move IS read. Walk ids until the argmax
    # moves in some group -- a single id may land in the same class.
    move_now = int(actor_input.env.opp_private_team[step, mon, move_column])
    label_moved = False
    for offset in range(1, 40):
        candidate = _with_cell(
            actor_input, "opp_private_team", step, mon, move_column, move_now + offset
        )
        moved = real_model_apply(params, candidate, actor_output, HeadParams())
        if not np.array_equal(
            np.asarray(base.hidden_code[step, mon], np.float32),
            np.asarray(moved.hidden_code[step, mon], np.float32),
        ):
            label_moved = True
            break
    assert label_moved, "no hidden-move id moved the label"


def test_hidden_any_empties_on_full_reveal_and_refills_when_unmatched(
    real_model_and_trajectory, real_model_apply
):
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    base = real_model_apply(params, actor_input, actor_output, HeadParams())
    step, mon, public_row, _ = _first_matched_with_hidden_move(actor_input, base)

    # Copy every id token of the sheet row onto its public row.
    revealed = actor_input
    private = np.asarray(actor_input.env.opp_private_team[step, mon])
    for column in SHARED_ID_COLUMNS:
        revealed = _with_cell(
            revealed, "revealed_team", step, public_row, column, int(private[column])
        )
    shown = real_model_apply(params, revealed, actor_output, HeadParams())
    assert bool(shown.belief_matched[step, mon])
    assert not bool(shown.belief_hidden_any[step, mon])
    # Other mons on the step are untouched.
    others = np.arange(6) != mon
    np.testing.assert_array_equal(
        np.asarray(base.belief_hidden_any[step])[others],
        np.asarray(shown.belief_hidden_any[step])[others],
    )

    # Unmatch the same mon: everything it carries counts as hidden again.
    idx_column = EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ENTITY_IDX
    unmatched = _with_cell(revealed, "opp_private_team", step, mon, idx_column, 0)
    lost = real_model_apply(params, unmatched, actor_output, HeadParams())
    assert not bool(lost.belief_matched[step, mon])
    assert bool(lost.belief_hidden_any[step, mon])
