from typing import Any, Dict

import chex
import jax
import jax.numpy as jnp

from rl.environment.interfaces import Trajectory
from rl.environment.protos.features_pb2 import ActionMaskFeature, FieldFeature
from rl.model.utils import BIAS_VALUE


def renormalize(loss: jax.Array, mask: jax.Array) -> jax.Array:
    """The `normalization` is the number of steps over which loss is computed."""
    chex.assert_equal_shape((loss, mask))
    loss = jnp.sum(loss * mask)
    normalization = jnp.sum(mask)
    return loss / (normalization + (normalization == 0.0))


def collect_batch_telemetry_data(batch: Trajectory) -> Dict[str, Any]:
    valid = jnp.bitwise_not(batch.player_transitions.env_output.done)
    lengths = valid.sum(0)

    history_lengths = batch.player_history.field[
        ..., FieldFeature.FIELD_FEATURE__VALID
    ].sum(0)

    can_move = batch.player_transitions.env_output.action_type_mask[..., 0]
    can_switch = batch.player_transitions.env_output.action_type_mask[..., 1]
    can_act = can_move & can_switch & valid

    move_ratio = renormalize(
        batch.player_transitions.agent_output.action_type == 0, can_act
    )
    switch_ratio = renormalize(
        batch.player_transitions.agent_output.action_type == 1, can_act
    )

    wildcard_turn = jnp.where(
        batch.player_transitions.agent_output.wildcard_slot
        != ActionMaskFeature.ACTION_MASK_FEATURE__CAN_NORMAL,
        jnp.arange(valid.shape[0], dtype=jnp.int32)[:, None],
        -BIAS_VALUE,
    ).min(axis=0)

    final_reward = batch.player_transitions.env_output.win_reward[-1]

    return dict(
        trajectory_length_mean=lengths.mean(),
        trajectory_length_min=lengths.min(),
        trajectory_length_max=lengths.max(),
        history_lengths_mean=history_lengths.mean(),
        move_ratio=move_ratio,
        switch_ratio=switch_ratio,
        wildcard_turn=wildcard_turn,
        reward_mean=final_reward.mean(),
        early_finish_rate=(jnp.abs(final_reward) < 1).astype(jnp.float32).mean(),
    )


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
