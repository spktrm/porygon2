from typing import Any, Dict

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from ml.func import get_average_logit_value, get_loss_entropy, renormalize
from rlenv.data import ACTION_STRINGS
from rlenv.interfaces import TimeStep
from rlenv.protos.features_pb2 import FeatureMoveset


def conditional_breakpoint(pred):
    def true_fn():
        pass

    def false_fn():
        jax.debug.breakpoint()

    jax.lax.cond(pred, true_fn, false_fn)


def collect_action_prob_telemetry_data(batch: TimeStep) -> Dict[str, Any]:
    valid_mask = batch.env.valid.reshape(-1)

    actions_available = batch.env.moveset[..., 0, :, FeatureMoveset.MOVESET_ACTION_ID]
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


@jax.jit
def collect_batch_telemetry_data(batch: TimeStep) -> Dict[str, Any]:
    valid = batch.env.valid
    lengths = valid.sum(0)

    can_move = batch.env.legal[..., :4].any(axis=-1)
    can_switch = batch.env.legal[..., 4:].any(axis=-1)

    move_ratio = renormalize(batch.actor.action < 4, can_move & can_switch & valid)
    switch_ratio = renormalize(batch.actor.action >= 4, can_move & can_switch & valid)

    return dict(
        actor_steps=valid.sum(),
        trajectory_length_mean=lengths.mean(),
        trajectory_length_min=lengths.min(),
        trajectory_length_max=lengths.max(),
        move_ratio=move_ratio,
        switch_ratio=switch_ratio,
        draw_ratio=batch.env.draw.any(axis=0).astype(float).mean(),
        early_finish_ratio=(
            jnp.abs(batch.actor.rewards.win_rewards[..., 0] * batch.env.valid).sum(0)
            != 1
        ).mean(),
        reward_sum=jnp.abs(batch.actor.rewards.win_rewards * batch.env.valid[..., None])
        .sum(0)
        .mean(),
    )


@jax.jit
def collect_gradient_telemetry_data(grads: chex.ArrayTree) -> Dict[str, Any]:
    logs = dict(
        gradient_norm=optax.global_norm(grads),
    )
    # for module_name in ["encoder", "policy_head", "value_head"]:
    #     for key, value in grads["params"][module_name].items():
    #         logs[f"{module_name}_{key}_abs_grad_max"] = jax.tree.reduce(
    #             lambda a, b: jnp.maximum(a, b),
    #             jax.tree.map(lambda x: jnp.abs(x).max(), value),
    #         )
    return logs


@jax.jit
def collect_nn_telemetry_data(state: train_state.TrainState) -> Dict[str, Any]:
    return dict(
        param_norm=optax.global_norm(state.params),
    )


def collect_loss_value_telemetry_data(
    value_loss: chex.Array,
    policy_loss: chex.Array,
    entropy_loss: chex.Array = None,
) -> dict[str, Any]:
    logs = {
        "value_loss": value_loss,
        "policy_loss": policy_loss,
    }
    if entropy_loss is not None:
        logs["entropy_loss"] = entropy_loss
    return logs


def collect_policy_stats_telemetry_data(
    logits: chex.Array,
    policy: chex.Array,
    log_policy: chex.Array,
    legal_mask: chex.Array,
    state_mask: chex.Array,
    prev_policy: chex.Array,
    ratio: chex.Array,
) -> dict[str, Any]:
    move_mask = legal_mask[..., :4]
    move_mask_sum = move_mask.sum(axis=-1)
    move_entropy = get_loss_entropy(policy[..., :4], state_mask & (move_mask_sum > 1))

    switch_mask = legal_mask[..., 4:]
    switch_mask_sum = switch_mask.sum(axis=-1)
    switch_entropy = get_loss_entropy(
        policy[..., 4:], state_mask & (switch_mask_sum > 1)
    )

    avg_logit_value = get_average_logit_value(logits, legal_mask, state_mask)
    kl_div = optax.kl_divergence(log_policy, prev_policy)

    return {
        "move_entropy": move_entropy,
        "switch_entropy": switch_entropy,
        "avg_logit_value": avg_logit_value,
        "kl_div": renormalize(kl_div, state_mask),
        "ratio": renormalize(ratio, state_mask),
    }


def calculate_explained_variance(
    value_prediction: chex.Array, value_target: chex.Array, mask: chex.Array = None
):
    value_prediction = jnp.squeeze(value_prediction)
    value_target = jnp.squeeze(value_target)
    if mask is None:
        mask = jnp.ones_like(value_prediction)
    mask = jnp.squeeze(mask)
    explained_variance = 1 - (
        jnp.square(jnp.std(value_target - value_prediction, where=mask))
        / jnp.square(jnp.std(value_target, where=mask))
    )
    return explained_variance


def calculate_r2(
    value_prediction: chex.Array, value_target: chex.Array, mask: chex.Array = None
) -> chex.Array:
    """
    Calculate the R-squared (coefficient of determination) value.

    Args:
        value_prediction: Predicted values (chex.Array).
        value_target: True target values (chex.Array).
        mask: Optional mask to include/exclude certain values (chex.Array, default is None).

    Returns:
        R-squared value as a chex.Array.
    """
    value_prediction = jnp.squeeze(value_prediction)
    value_target = jnp.squeeze(value_target)
    if mask is None:
        mask = jnp.ones_like(value_prediction)
    else:
        mask = jnp.squeeze(mask)

    # Calculate residual sum of squares (SS_residual)
    residuals = value_target - value_prediction
    ss_residual = jnp.sum(jnp.square(residuals), where=mask)

    # Calculate total sum of squares (SS_total)
    mean_target = jnp.mean(value_target, where=mask)
    ss_total = jnp.sum(jnp.square(value_target - mean_target), where=mask)

    # Add epsilon to avoid division by zero
    epsilon = 1e-8
    r2 = 1 - (ss_residual / (ss_total + epsilon))
    return r2


def collect_value_stats_telemetry_data(
    value_prediction: chex.Array, value_target: chex.Array, mask: chex.Array = None
) -> dict[str, Any]:
    explained_variance = calculate_explained_variance(
        value_prediction, value_target, mask
    )
    return {"value_function_explained_variance": jnp.maximum(-1, explained_variance)}
