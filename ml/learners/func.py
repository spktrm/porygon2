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
from rlenv.protos.features_pb2 import AbsoluteEdgeFeature, MovesetFeature


def conditional_breakpoint(pred):
    def true_fn():
        pass

    def false_fn():
        jax.debug.breakpoint()

    jax.lax.cond(pred, true_fn, false_fn)


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


@jax.jit
def collect_batch_telemetry_data(batch: TimeStep) -> Dict[str, Any]:
    valid = batch.env.valid
    lengths = valid.sum(0)

    history_lengths = batch.history.major_history.absolute_edges[
        ..., AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__VALID
    ].sum(0)

    can_move = batch.env.legal[..., :4].any(axis=-1)
    can_switch = batch.env.legal[..., 4:].any(axis=-1)

    move_ratio = renormalize(batch.actor.action < 4, can_move & can_switch & valid)
    switch_ratio = renormalize(batch.actor.action >= 4, can_move & can_switch & valid)

    return dict(
        actor_steps=valid.sum(),
        trajectory_length_mean=lengths.mean(),
        trajectory_length_min=lengths.min(),
        trajectory_length_max=lengths.max(),
        history_lengths_mean=history_lengths.mean(),
        move_ratio=move_ratio,
        switch_ratio=switch_ratio,
        draw_ratio=batch.env.draw.any(axis=0).astype(float).mean(),
        early_finish_ratio=(
            jnp.abs(batch.env.rewards.win_rewards[..., 0] * batch.env.valid).sum(0) != 1
        ).mean(),
        reward_sum=jnp.abs(batch.env.rewards.win_rewards * batch.env.valid[..., None])
        .sum(0)
        .mean(),
    )


@jax.jit
def collect_parameter_and_gradient_telemetry_data(
    params: chex.ArrayTree, grads: chex.ArrayTree
) -> Dict[str, Any]:
    logs = dict(
        param_norm=optax.global_norm(params),
        gradient_norm=optax.global_norm(grads),
    )
    # logs.update(per_module_gradient_stats(grads))
    return logs


@jax.jit
def efficient_per_module_gradient_stats(grads: chex.ArrayTree) -> Dict[str, Any]:
    stats = {}

    # List of modules to analyze separately
    modules = ["encoder", "policy_head", "value_head"]

    # For each module, calculate relevant statistics
    for module_name in modules:
        if module_name not in grads["params"]:
            continue

        module_grads = grads["params"][module_name]

        # Calculate statistics directly on tree leaves without concatenation
        def calc_stats_fn(grads):
            # Compute statistics in a vectorized way
            abs_grads = jnp.abs(grads)

            # Basic statistics that can be calculated without concatenation
            norm = jnp.sqrt(jnp.sum(grads**2))
            mean = jnp.mean(abs_grads)

            # Approximate percentiles using mean and std
            std = jnp.std(abs_grads)
            approx_25th = jnp.maximum(mean - 0.674 * std, 0)  # Approximation
            approx_75th = mean + 0.674 * std

            # Signal-to-noise
            snr = jnp.mean(grads) / (std + 1e-8)

            # Zero gradient percentage
            zero_pct = jnp.mean(abs_grads < 1e-8)

            # Min and max
            grad_min = jnp.min(abs_grads)
            grad_max = jnp.max(abs_grads)

            return {
                "norm": norm,
                "mean": mean,
                "approx_25th": approx_25th,
                "approx_75th": approx_75th,
                "snr": snr,
                "zero_pct": zero_pct,
                "min": grad_min,
                "max": grad_max,
            }

        # Process each parameter group separately and aggregate
        module_stats = {
            "norm": 0.0,
            "mean": 0.0,
            "approx_25th": 0.0,
            "approx_75th": 0.0,
            "snr": 0.0,
            "zero_pct": 0.0,
            "min": float("inf"),
            "max": 0.0,
        }

        param_count = 0

        for key, value in module_grads.items():
            # Process each leaf in the tree
            for leaf in jax.tree_util.tree_leaves(value):
                if not hasattr(leaf, "size"):
                    continue

                leaf_stats = calc_stats_fn(leaf)
                leaf_size = leaf.size

                # Weighted average for appropriate metrics
                module_stats["mean"] = (
                    module_stats["mean"] * param_count + leaf_stats["mean"] * leaf_size
                ) / (param_count + leaf_size)
                module_stats["approx_25th"] = (
                    module_stats["approx_25th"] * param_count
                    + leaf_stats["approx_25th"] * leaf_size
                ) / (param_count + leaf_size)
                module_stats["approx_75th"] = (
                    module_stats["approx_75th"] * param_count
                    + leaf_stats["approx_75th"] * leaf_size
                ) / (param_count + leaf_size)
                module_stats["snr"] = (
                    module_stats["snr"] * param_count + leaf_stats["snr"] * leaf_size
                ) / (param_count + leaf_size)
                module_stats["zero_pct"] = (
                    module_stats["zero_pct"] * param_count
                    + leaf_stats["zero_pct"] * leaf_size
                ) / (param_count + leaf_size)

                # For norm, we accumulate squares and take sqrt at the end
                module_stats["norm"] += leaf_stats["norm"] ** 2

                # For min/max, we take the global min/max
                module_stats["min"] = jnp.minimum(
                    module_stats["min"], leaf_stats["min"]
                )
                module_stats["max"] = jnp.maximum(
                    module_stats["max"], leaf_stats["max"]
                )

                param_count += leaf_size

        # Finalize calculations
        module_stats["norm"] = jnp.sqrt(module_stats["norm"])

        # Prefix for this module's statistics
        prefix = f"{module_name}_"

        # Add to the overall stats dictionary
        stats[f"{prefix}grad_norm"] = module_stats["norm"]
        stats[f"{prefix}grad_mean"] = module_stats["mean"]
        stats[f"{prefix}grad_approx_25th"] = module_stats["approx_25th"]
        stats[f"{prefix}grad_approx_75th"] = module_stats["approx_75th"]
        stats[f"{prefix}grad_signal_to_noise"] = module_stats["snr"]
        stats[f"{prefix}grad_zero_percentage"] = module_stats["zero_pct"]
        stats[f"{prefix}grad_min"] = module_stats["min"]
        stats[f"{prefix}grad_max"] = module_stats["max"]

    return stats


def _calculate_stats_for_tree(
    grads_tree: chex.ArrayTree, prefix: str = ""
) -> Dict[str, jnp.ndarray] | None:
    """
    Helper function to calculate gradient statistics for a given PyTree of gradients.

    It flattens all gradients within the tree into a single vector for analysis.
    """
    # Get all gradient arrays from the PyTree
    leaves = jax.tree_util.tree_leaves(grads_tree)

    # If there are no parameters in this submodule, skip it
    if not leaves:
        return None

    # Flatten all gradient arrays into a single 1D vector
    flat_grads = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])

    # Calculate statistics on the entire flattened array
    abs_grads = jnp.abs(flat_grads)

    norm = jnp.linalg.norm(flat_grads)
    mean = jnp.mean(abs_grads)
    std = jnp.std(abs_grads)
    grad_min = jnp.min(abs_grads)
    grad_max = jnp.max(abs_grads)
    p25, p50, p75 = jnp.percentile(abs_grads, jnp.array([25, 50, 75]))
    snr = jnp.mean(flat_grads) / (jnp.std(flat_grads) + 1e-8)
    zero_pct = jnp.mean(abs_grads < 1e-8)

    return {
        f"{prefix}grad_norm": norm,
        f"{prefix}grad_mean": mean,
        f"{prefix}grad_std": std,
        f"{prefix}grad_min": grad_min,
        f"{prefix}grad_max": grad_max,
        f"{prefix}grad_p25": p25,
        f"{prefix}grad_p50": p50,
        f"{prefix}grad_p75": p75,
        f"{prefix}grad_signal_to_noise": snr,
        f"{prefix}grad_zero_percentage": zero_pct,
    }


@jax.jit
def per_module_gradient_stats(grads: chex.ArrayTree) -> Dict[str, Any]:
    """
    Calculates gradient statistics for high-level modules and their direct sub-modules.

    This function iterates through 'encoder', 'policy_head', and 'value_head',
    calculating stats for the entire module and then for each sub-module
    one level deep.

    Args:
        grads: A PyTree of gradients, expected to have a 'params' key
               containing the module parameters.

    Returns:
        A dictionary containing gradient statistics, with keys prefixed
        by module and sub-module names.
    """
    stats = {}
    top_level_modules = ["encoder", "policy_head", "value_head"]

    for module_name in top_level_modules:
        if module_name not in grads.get("params", {}):
            continue

        module_grads_tree = grads["params"][module_name]

        # 1. Calculate and store stats for the entire top-level module
        prefix = f"{module_name}_"
        top_level_stats = _calculate_stats_for_tree(module_grads_tree, prefix)
        if top_level_stats is not None:
            stats.update(top_level_stats)

        # 2. Iterate through sub-modules (1 level deep)
        for sub_module_name, sub_module_grads_tree in module_grads_tree.items():
            # Ensure the item is a sub-module (a dict-like PyTree) and not a direct parameter array
            if isinstance(sub_module_grads_tree, dict):
                prefix = f"{module_name}_{sub_module_name}_"
                sub_module_stats = _calculate_stats_for_tree(
                    sub_module_grads_tree, prefix
                )
                if sub_module_stats is not None:
                    stats.update(sub_module_stats)

    return stats


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


def collect_regularisation_telemetry_data(
    policy: chex.Array,
    regularisation_policy: chex.Array,
    legal_mask: chex.Array,
    state_mask: chex.Array,
) -> dict[str, Any]:
    raw_reg_rewards = regularisation_policy.mean(where=legal_mask, axis=-1).mean(
        where=state_mask
    )
    norm_reg_rewards = jnp.squeeze((policy * regularisation_policy).sum(axis=-1)).mean(
        where=state_mask
    )
    return {
        "raw_reg_rewards": raw_reg_rewards,
        "norm_reg_rewards": norm_reg_rewards,
    }


def collect_policy_stats_telemetry_data(
    logits: chex.Array,
    policy: chex.Array,
    log_policy: chex.Array,
    legal_mask: chex.Array,
    state_mask: chex.Array,
    prev_policy: chex.Array,
    ratio: chex.Array,
    adv_pi: chex.Array,
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

    mean_adv_pi = adv_pi.mean(where=state_mask)
    std_adv_pi = adv_pi.std(where=state_mask)
    max_adv_pi = jnp.where(state_mask, adv_pi, -1e9).max()
    min_adv_pi = jnp.where(state_mask, adv_pi, 1e9).min()

    return {
        "move_entropy": move_entropy,
        "switch_entropy": switch_entropy,
        "avg_logit_value": avg_logit_value,
        "kl_div": renormalize(kl_div, state_mask),
        "ratio": renormalize(ratio, state_mask),
        "mean_adv_pi": mean_adv_pi,
        "std_adv_pi": std_adv_pi,
        "max_adv_pi": max_adv_pi,
        "min_adv_pi": min_adv_pi,
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
    r2 = calculate_r2(value_prediction, value_target, mask)
    return {
        "value_function_explained_variance": jnp.maximum(-1, explained_variance),
        "value_function_r2": jnp.maximum(r2, 0),
        "value_target_mean": jnp.mean(value_target, where=mask),
        "value_target_std": jnp.std(value_target, where=mask),
    }
