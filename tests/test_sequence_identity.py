"""Semantic row identities without initialising the full network."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.environment.data import TARGET_SLOT_INDICES
from rl.environment.interfaces import PlayerEnvOutput
from rl.environment.protos.features_pb2 import (
    EntityPrivateNodeFeature,
    EntityPublicNodeFeature,
    InfoFeature,
)
from rl.environment.protos.service_pb2 import TargetSlot
from rl.model.constants import (
    ACTIVE_STATE_ROWS,
    HISTORY_ACTIVE_ROWS,
    HISTORY_ENTITY_ROWS,
    NUM_ACTIVE_SLOTS,
    NUM_HISTORY_REGISTERS,
    NUM_PRIVATE_SLOTS,
    NUM_PUBLIC_SLOTS,
    OPP_PRIVATE_ROWS,
    POLICY_READABLE_ROWS,
    PRIVATE_ROWS,
    PUBLIC_ROWS,
    SEQUENCE_SLICES,
    TARGET_ROWS,
    SequenceGroup,
)
from rl.model.identity import private_positions, sequence_identities


@pytest.fixture
def env_step():
    public = np.zeros((NUM_PUBLIC_SLOTS, len(EntityPublicNodeFeature.keys())), np.int32)
    public[
        :NUM_PRIVATE_SLOTS, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE
    ] = 1
    public[:, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE] = [
        2,
        1,
        0,
        0,
        0,
        0,
    ] * 2
    private = np.zeros(
        (NUM_PRIVATE_SLOTS, len(EntityPrivateNodeFeature.keys())), np.int32
    )
    opponent = private.copy()
    key_column = EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ENTITY_IDX
    private[:, key_column] = [0, 3, 1, 2, 5, 6]
    opponent[:, key_column] = [9, 0, 8, 7, 11, 12]
    info = np.zeros(len(InfoFeature.keys()), np.int32)
    info[
        InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0 : InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11
        + 1
    ] = np.arange(NUM_PUBLIC_SLOTS)
    return PlayerEnvOutput(
        public_team=jnp.asarray(public),
        private_team=jnp.asarray(private),
        opp_private_team=jnp.asarray(opponent),
        info=jnp.asarray(info),
    )


def identity_rows(env_step, side=None, position=None, *, include_opponent=True):
    if side is None:
        side = jnp.asarray([[10.0, 0.0], [0.0, 20.0]])
    if position is None:
        position = jnp.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    return sequence_identities(
        env_step,
        side,
        position,
        jnp.zeros((len(TARGET_SLOT_INDICES), 2)),
        include_opponent=include_opponent,
    )


def test_private_positions_match_keys_and_side_not_sheet_order(env_step):
    np.testing.assert_array_equal(
        private_positions(env_step, env_step.private_team, 1), [0, 0, 2, 1, 0, 0]
    )
    np.testing.assert_array_equal(
        private_positions(env_step, env_step.opp_private_team, 0), [0, 0, 1, 2, 0, 0]
    )
    crossed = dataclasses.replace(env_step, private_team=env_step.opp_private_team)
    np.testing.assert_array_equal(
        private_positions(crossed, crossed.private_team, 1), 0
    )


def test_same_side_and_position_are_identical_across_entity_representations(env_step):
    rows = identity_rows(env_step)
    np.testing.assert_array_equal(rows[PRIVATE_ROWS][2], rows[PUBLIC_ROWS][0])
    np.testing.assert_array_equal(rows[OPP_PRIVATE_ROWS][3], rows[PUBLIC_ROWS][6])
    assert not np.array_equal(rows[PRIVATE_ROWS][2], rows[OPP_PRIVATE_ROWS][3])
    assert not np.array_equal(rows[PRIVATE_ROWS][2], rows[PRIVATE_ROWS][0])


def test_active_slot_rows_carry_the_identity_of_the_slot_they_describe(env_step):
    rows = identity_rows(env_step)
    # The fixture's actives: mine at public rows 0 (first) and 1 (second),
    # theirs at rows 6 and 7 -- the active-slot order.
    occupants = rows[PUBLIC_ROWS][jnp.asarray([0, 1, 6, 7])]
    np.testing.assert_array_equal(rows[ACTIVE_STATE_ROWS], occupants)
    np.testing.assert_array_equal(rows[HISTORY_ACTIVE_ROWS], occupants)
    assert len({tuple(np.asarray(row)) for row in rows[ACTIVE_STATE_ROWS]}) == 4


def test_history_shares_current_public_position_and_fields_get_side_only(env_step):
    rows = identity_rows(env_step)
    changed = identity_rows(env_step, position=jnp.full((3, 2), 500.0))
    np.testing.assert_array_equal(rows[HISTORY_ENTITY_ROWS], rows[PUBLIC_ROWS])
    np.testing.assert_array_equal(changed[HISTORY_ENTITY_ROWS], changed[PUBLIC_ROWS])
    assert not np.array_equal(rows[HISTORY_ENTITY_ROWS], changed[HISTORY_ENTITY_ROWS])
    for group in (SequenceGroup.FIELD, SequenceGroup.HISTORY_FIELD):
        field = np.asarray(rows[SEQUENCE_SLICES[group]])
        np.testing.assert_array_equal(field, [[0, 0], [0, 20], [10, 0]])
        np.testing.assert_array_equal(field, changed[SEQUENCE_SLICES[group]])


def test_history_position_tracks_current_active_role_changes(env_step):
    position_column = EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE
    public = env_step.public_team.at[0, position_column].set(0)
    public = public.at[2, position_column].set(2)
    changed_env = dataclasses.replace(env_step, public_team=public)
    base = np.asarray(identity_rows(env_step)[HISTORY_ENTITY_ROWS])
    changed = np.asarray(identity_rows(changed_env)[HISTORY_ENTITY_ROWS])
    np.testing.assert_array_equal(changed[0], base[2])
    np.testing.assert_array_equal(changed[2], base[0])
    assert not np.array_equal(changed[0], base[0])
    untouched = [1, *range(3, NUM_PUBLIC_SLOTS)]
    np.testing.assert_array_equal(changed[untouched], base[untouched])


@pytest.mark.parametrize(
    "target,expected",
    [
        (TargetSlot.TARGET_SLOT___UNSPECIFIED, [0, 0]),
        (TargetSlot.TARGET_SLOT__DEFAULT, [0, 0]),
        (TargetSlot.TARGET_SLOT__ALLY_1, [3, 23]),
        (TargetSlot.TARGET_SLOT__ALLY_1_PASS, [3, 23]),
        (TargetSlot.TARGET_SLOT__ALLY_2, [2, 22]),
        (TargetSlot.TARGET_SLOT__ALLY_2_PASS, [2, 22]),
        (TargetSlot.TARGET_SLOT__ENEMY_1, [13, 3]),
        (TargetSlot.TARGET_SLOT__ENEMY_2, [12, 2]),
        (TargetSlot.TARGET_SLOT__AUTO, [0, 0]),
        (TargetSlot.TARGET_SLOT__ALL, [10, 20]),
        (TargetSlot.TARGET_SLOT__ALLY_SIDE, [0, 20]),
        (TargetSlot.TARGET_SLOT__FOE_SIDE, [10, 0]),
        (TargetSlot.TARGET_SLOT__ALLY_TEAM, [0, 20]),
        (TargetSlot.TARGET_SLOT__RANDOM_NORMAL, [10, 0]),
        (TargetSlot.TARGET_SLOT__ALL_ADJACENT, [10, 20]),
        (TargetSlot.TARGET_SLOT__ALL_ADJACENT_FOES, [10, 0]),
        (TargetSlot.TARGET_SLOT__ALLIES, [0, 20]),
    ],
)
def test_every_target_has_semantic_identity_without_pokemon_content(
    env_step, target, expected
):
    target_index = list(TARGET_SLOT_INDICES).index(target)
    np.testing.assert_array_equal(
        identity_rows(env_step)[TARGET_ROWS][target_index], expected
    )


def test_opponent_truth_changes_only_privileged_identities(env_step):
    base = identity_rows(env_step)
    key_column = EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ENTITY_IDX
    changed_env = dataclasses.replace(
        env_step, opp_private_team=env_step.opp_private_team.at[:, key_column].set(0)
    )
    changed = identity_rows(changed_env)
    np.testing.assert_array_equal(
        base[POLICY_READABLE_ROWS], changed[POLICY_READABLE_ROWS]
    )
    assert not np.array_equal(base[OPP_PRIVATE_ROWS], changed[OPP_PRIVATE_ROWS])
    actor_env = dataclasses.replace(env_step, opp_private_team=())
    actor_rows = identity_rows(actor_env, include_opponent=False)
    np.testing.assert_array_equal(
        base[POLICY_READABLE_ROWS], actor_rows[POLICY_READABLE_ROWS]
    )


def test_side_gradients_are_shared_and_both_side_targets_sum(env_step):
    def target_sum(side):
        rows = identity_rows(env_step, side=side)
        target_index = list(TARGET_SLOT_INDICES).index(TargetSlot.TARGET_SLOT__ALL)
        return rows[TARGET_ROWS][target_index].sum()

    gradient = jax.jit(jax.grad(target_sum))(jnp.ones((2, 2)))
    np.testing.assert_array_equal(gradient, 1)


def test_rl_history_inputs_ignore_snapshots_but_preserve_aligned_memory(env_step):
    from types import SimpleNamespace

    from rl.model.encoder import Encoder

    row_width = 4
    slot_states = jnp.arange(NUM_PUBLIC_SLOTS * row_width, dtype=jnp.float32).reshape(
        1, NUM_PUBLIC_SLOTS, row_width
    )
    field_state = jnp.ones((1, 3, row_width))
    snapshots = jnp.full_like(slot_states, 1000)
    registers = jnp.ones((1, NUM_HISTORY_REGISTERS, row_width))
    actives = jnp.full((1, NUM_ACTIVE_SLOTS, row_width), 2.0)
    history_output = object()
    order = jnp.asarray([2, 0, 1, 3, 4, 5, 8, 7, 6, 9, 10, -1])
    info = env_step.info.at[
        InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0 : InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11
        + 1
    ].set(order)
    batched_env = dataclasses.replace(env_step, info=info[None])

    def read_inputs(memory, nodes):
        encoder = SimpleNamespace(
            encode_history=lambda *args: (
                memory,
                field_state,
                nodes,
                registers,
                actives,
                history_output,
            )
        )
        return Encoder._history_inputs(encoder, batched_env, None, None)

    rows, valid, field, register_rows, active_rows, output = read_inputs(
        slot_states, snapshots
    )
    np.testing.assert_array_equal(register_rows, registers)
    np.testing.assert_array_equal(active_rows, actives)
    assert output is history_output
    np.testing.assert_array_equal(rows[0, :-1], slot_states[0, order[:-1]])
    np.testing.assert_array_equal(valid[0], order >= 0)
    np.testing.assert_array_equal(field, field_state)
    changed_snapshots = read_inputs(slot_states, snapshots * -100)
    for actual, expected in zip(
        changed_snapshots[:-1], (rows, valid, field, registers, actives), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    changed_memory = read_inputs(slot_states + 10, snapshots)[0]
    np.testing.assert_array_equal(changed_memory - rows, 10)
