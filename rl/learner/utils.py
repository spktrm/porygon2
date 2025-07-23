from typing import Any, Dict

import chex
import jax.numpy as jnp
import numpy as np

from rl.environment.data import ACTION_STRINGS
from rl.environment.interfaces import TimeStep, Transition
from rl.environment.protos.features_pb2 import FieldFeature, MovesetFeature


def renormalize(loss: chex.Array, mask: chex.Array) -> chex.Array:
    """The `normalization` is the number of steps over which loss is computed."""
    chex.assert_equal_shape((loss, mask))
    loss = jnp.sum(loss * mask)
    normalization = jnp.sum(mask)
    return loss / (normalization + (normalization == 0.0))


def collect_action_prob_telemetry_data(batch: TimeStep) -> Dict[str, Any]:
    valid_mask = batch.env.valid.reshape(-1)

    actions_available = batch.env.moveset[
        ..., 0, :, MovesetFeature.MOVESET_FEATURE__ACTION_ID
    ]
    actions_index = np.eye(actions_available.shape[-1])[batch.actor.action]

    actions = (actions_available * actions_index).sum(axis=-1).reshape(-1)
    probabilities = (batch.actor.policy * actions_index).sum(axis=-1).reshape(-1)

    # Find unique actions and their indices
    unique_actions, inverse_indices = np.unique(actions, return_inverse=True)

    # One-hot encode the actions
    one_hot = np.eye(len(unique_actions))[inverse_indices]

    # Aggregate probabilities for each action
    sum_probs = np.sum(
        one_hot * probabilities[..., None], axis=0, where=valid_mask[..., None]
    )
    count_probs = np.sum(one_hot, axis=0, where=valid_mask[..., None])

    # Compute the mean probabilities
    mean_probs = sum_probs / np.where(count_probs == 0, 1, count_probs)

    unique_actions = unique_actions.astype(int)
    mean_probs = mean_probs.astype(float)

    return {ACTION_STRINGS[k]: v for k, v in zip(unique_actions, mean_probs)}


def collect_batch_telemetry_data(batch: Transition) -> Dict[str, Any]:
    valid = jnp.bitwise_not(batch.timestep.env.done)
    lengths = valid.sum(0)

    history_lengths = batch.timestep.history.field[
        ..., FieldFeature.FIELD_FEATURE__VALID
    ].sum(0)

    can_move = batch.timestep.env.legal[..., :4].any(axis=-1)
    can_switch = batch.timestep.env.legal[..., 4:].any(axis=-1)
    can_act = can_move & can_switch & valid

    move_ratio = renormalize(batch.actorstep.action < 4, can_act)
    switch_ratio = renormalize(batch.actorstep.action >= 4, can_act)

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
    value_prediction: chex.Array,
    value_target: chex.Array,
    mask: chex.Array = None,
    eps: float = 1e-8,
) -> chex.Array:
    """Calculate the R-squared (coefficient of determination) value."""

    chex.assert_rank(value_prediction, 2)
    chex.assert_rank(value_target, 2)
    chex.assert_rank(mask, 2)

    if mask is None:
        mask = jnp.ones_like(value_prediction)

    # Calculate residual sum of squares (SS_residual)
    ss_residual = jnp.sum((value_target - value_prediction) ** 2, where=mask)

    # Calculate total sum of squares (SS_total)
    mean_target = jnp.mean(value_target, where=mask)
    ss_total = jnp.sum((value_target - mean_target) ** 2, where=mask)

    return 1 - (ss_residual / (ss_total + eps))
