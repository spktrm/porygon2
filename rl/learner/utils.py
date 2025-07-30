from typing import Any, Dict

import chex
import jax
import jax.numpy as jnp

from rl.environment.interfaces import Transition
from rl.environment.protos.features_pb2 import FieldFeature


def renormalize(loss: jax.Array, mask: jax.Array) -> jax.Array:
    """The `normalization` is the number of steps over which loss is computed."""
    chex.assert_equal_shape((loss, mask))
    loss = jnp.sum(loss * mask)
    normalization = jnp.sum(mask)
    return loss / (normalization + (normalization == 0.0))


def collect_batch_telemetry_data(batch: Transition) -> Dict[str, Any]:
    valid = jnp.bitwise_not(batch.timestep.env.done)
    lengths = valid.sum(0)

    history_lengths = batch.timestep.history.field[
        ..., FieldFeature.FIELD_FEATURE__VALID
    ].sum(0)

    can_move = batch.timestep.env.legal[..., :4].any(axis=-1)
    can_switch = batch.timestep.env.legal[..., 4:].any(axis=-1)
    can_act = can_move & can_switch & valid

    move_ratio = renormalize(batch.actor_step.action < 4, can_act)
    switch_ratio = renormalize(batch.actor_step.action >= 4, can_act)

    return dict(
        trajectory_length_mean=lengths.mean(),
        trajectory_length_min=lengths.min(),
        trajectory_length_max=lengths.max(),
        history_lengths_mean=history_lengths.mean(),
        move_ratio=move_ratio,
        switch_ratio=switch_ratio,
        reward_mean=batch.timestep.env.win_reward[-1].mean(),
        early_finish_rate=(jnp.abs(batch.timestep.env.win_reward[-1]) < 1)
        .astype(jnp.float32)
        .mean(),
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
