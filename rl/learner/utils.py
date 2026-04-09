from typing import Any, Dict, TypeVar

import chex
import jax
import jax.numpy as jnp

from rl.environment.data import CAT_VF_SUPPORT, NUM_PACKED_SET_FEATURES
from rl.environment.interfaces import Trajectory
from rl.environment.protos.features_pb2 import FieldFeature, PackedSetFeature
from rl.environment.protos.service_pb2 import ActionEnum
from rl.learner.config import Porygon2LearnerConfig
from rl.learner.targets import PlayerTargets
from rl.utils import average

T = TypeVar("T")


def promote_map(tree: T, dtype) -> T:
    return jax.tree.map(lambda x: x.astype(dtype), tree)


def renormalize(loss: jax.Array, mask: jax.Array) -> jax.Array:
    """The `normalization` is the number of steps over which loss is computed."""
    chex.assert_equal_shape((loss, mask))
    loss = jnp.sum(loss * mask)
    normalization = jnp.sum(mask)
    return loss / (normalization + (normalization == 0.0))


def collect_batch_telemetry_data(
    batch: Trajectory, config: Porygon2LearnerConfig, player_targets: PlayerTargets
) -> Dict[str, Any]:
    player_valid = jnp.bitwise_not(batch.player_transitions.env_output.done)
    player_lengths = player_valid.sum(0)

    history_lengths = batch.player_history.field[
        ..., FieldFeature.FIELD_FEATURE__VALID
    ].sum(0)

    can_move = batch.player_transitions.env_output.action_mask[
        ...,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_1 : ActionEnum.ACTION_ENUM__ALLY_2_MOVE_4_WILDCARD
        + 1,
        :,
    ].any((-2, -1))
    can_wildcard = (
        batch.player_transitions.env_output.action_mask[
            ...,
            ActionEnum.ACTION_ENUM__ALLY_1_MOVE_1_WILDCARD : ActionEnum.ACTION_ENUM__ALLY_1_MOVE_4_WILDCARD
            + 1,
            :,
        ].any((-2, -1))
    ) | (
        batch.player_transitions.env_output.action_mask[
            ...,
            ActionEnum.ACTION_ENUM__ALLY_2_MOVE_1_WILDCARD : ActionEnum.ACTION_ENUM__ALLY_2_MOVE_4_WILDCARD
            + 1,
            :,
        ].any((-2, -1))
    )
    can_switch = batch.player_transitions.env_output.action_mask[
        ...,
        ActionEnum.ACTION_ENUM__RESERVE_1 : ActionEnum.ACTION_ENUM__RESERVE_6 + 1,
        :,
    ].any((-2, -1))
    can_act = can_move & can_switch & player_valid

    src_action_index = (
        batch.player_transitions.agent_output.actor_output.action_head.src_index
    )
    tgt_action_index = (
        batch.player_transitions.agent_output.actor_output.action_head.tgt_index
    )
    did_move = (
        (src_action_index >= ActionEnum.ACTION_ENUM__ALLY_1_MOVE_1)
        & (src_action_index <= ActionEnum.ACTION_ENUM__ALLY_2_MOVE_4_WILDCARD)
        & can_move
    )
    did_wildcard = (
        (
            (src_action_index >= ActionEnum.ACTION_ENUM__ALLY_1_MOVE_1_WILDCARD)
            & (src_action_index <= ActionEnum.ACTION_ENUM__ALLY_1_MOVE_4_WILDCARD)
        )
        | (
            (src_action_index >= ActionEnum.ACTION_ENUM__ALLY_2_MOVE_1_WILDCARD)
            & (src_action_index <= ActionEnum.ACTION_ENUM__ALLY_2_MOVE_4_WILDCARD)
        )
    ) & can_move
    did_switch = (
        (src_action_index >= ActionEnum.ACTION_ENUM__RESERVE_1)
        & (src_action_index <= ActionEnum.ACTION_ENUM__RESERVE_6)
        & (tgt_action_index >= ActionEnum.ACTION_ENUM__ALLY_1)
        & (tgt_action_index <= ActionEnum.ACTION_ENUM__ALLY_2)
        & can_switch
    )
    move_ratio = renormalize(did_move, can_act)
    switch_ratio = renormalize(did_switch, can_act)

    wildcard_turn = jnp.where(
        did_move & did_wildcard,
        jnp.arange(player_valid.shape[0], dtype=jnp.int32)[:, None],
        player_valid.shape[0],
    ).min(axis=0)

    final_reward = batch.player_transitions.env_output.win_reward[-1]

    telemetry = dict(
        player_trajectory_length_mean=player_lengths.mean(),
        player_trajectory_length_min=player_lengths.min(),
        player_trajectory_length_max=player_lengths.max(),
        history_lengths_mean=history_lengths.mean(),
        player_proactive_switch_win_advantange=average(
            player_targets.win_advantages, player_valid & did_switch & can_move
        ),
        player_passive_switch_win_advantange=average(
            player_targets.win_advantages, player_valid & did_switch & ~can_move
        ),
        player_wildcard_hold_win_advantange=average(
            player_targets.win_advantages,
            player_valid & did_move & ~did_wildcard & can_wildcard,
        ),
        player_proactive_switch_pot_advantange=average(
            player_targets.potential_advantages, player_valid & did_switch & can_move
        ),
        player_passive_switch_pot_advantange=average(
            player_targets.potential_advantages, player_valid & did_switch & ~can_move
        ),
        player_wildcard_hold_pot_advantange=average(
            player_targets.potential_advantages,
            player_valid & did_move & ~did_wildcard & can_wildcard,
        ),
        move_ratio=move_ratio,
        switch_ratio=switch_ratio,
        wildcard_turn=wildcard_turn.mean(),
        reward_mean=(final_reward @ CAT_VF_SUPPORT).mean(),
        early_finish_rate=(jnp.abs(final_reward @ CAT_VF_SUPPORT) < 1)
        .astype(jnp.float32)
        .mean(),
    )

    if config.smogon_format != "randombattle":
        builder_valid = jnp.bitwise_not(batch.builder_transitions.env_output.done)
        builder_lengths = builder_valid.sum(0)

        team_tokens = batch.builder_history.packed_team_member_tokens.reshape(
            -1,
            NUM_PACKED_SET_FEATURES,
            batch.builder_history.packed_team_member_tokens.shape[1],
        )
        team_evs = team_tokens[
            :,
            PackedSetFeature.PACKED_SET_FEATURE__HP_EV : PackedSetFeature.PACKED_SET_FEATURE__SPE_EV
            + 1,
        ]
        ev_prob = team_evs / 128
        ev_entropy = -jnp.sum(ev_prob * jnp.log(ev_prob + 1e-8), axis=-1).mean()

        ev_reward = batch.builder_transitions.env_output.ev_reward[-1].mean()

        telemetry.update(
            dict(
                builder_trajectory_length_mean=builder_lengths.mean(),
                builder_trajectory_length_min=builder_lengths.min(),
                builder_trajectory_length_max=builder_lengths.max(),
                builder_ev_entropy=ev_entropy,
                builder_ev_reward=ev_reward,
            )
        )

    return telemetry


def calculate_r2(
    value_prediction: jax.Array,
    value_target: jax.Array,
    mask: jax.Array = None,
    eps: float = 1e-8,
) -> jax.Array:
    """Calculate the R-squared (coefficient of determination) value."""

    if mask is None:
        mask = jnp.ones_like(value_prediction)

    # Calculate residual sum of squares (SS_residual)
    ss_residual = jnp.sum((value_target - value_prediction) ** 2, where=mask)

    # Calculate total sum of squares (SS_total)
    mean_target = jnp.mean(value_target, where=mask)
    ss_total = jnp.sum((value_target - mean_target) ** 2, where=mask)

    return 1 - (ss_residual / (ss_total + eps))
