import math
import os
from typing import Callable

import chex
import jax
import jax.numpy as jnp

Params = chex.ArrayTree
Optimizer = Callable[[Params, Params], Params]  # (params, grads) -> params

BIAS_VALUE = -1e30


def legal_policy(
    logits: jax.Array, legal_actions: jax.Array | None = None
) -> jax.Array:
    """A soft-max policy that respects legal_actions."""
    if legal_actions is None:
        legal_actions = logits > BIAS_VALUE
    chex.assert_equal_shape((logits, legal_actions), dims=(0, 1, -1))
    # Fiddle a bit to make sure we don't generate NaNs or Inf in the middle.
    l_min = logits.min(axis=-1, keepdims=True)
    logits = jnp.where(legal_actions, logits, l_min)
    logits -= logits.max(axis=-1, keepdims=True)
    logits *= legal_actions
    exp_logits = jnp.where(
        legal_actions, jnp.exp(logits), 0
    )  # Illegal actions become 0.
    exp_logits_sum = jnp.sum(exp_logits, axis=-1, keepdims=True)
    return exp_logits / exp_logits_sum


def legal_log_policy(
    logits: jax.Array, legal_actions: jax.Array | None = None
) -> jax.Array:
    """Return the log of the policy on legal action, 0 on illegal action."""
    if legal_actions is None:
        legal_actions = logits > BIAS_VALUE
    chex.assert_equal_shape((logits, legal_actions), dims=(0, 1, -1))
    # logits_masked has illegal actions set to -inf.
    logits_masked = logits + jnp.log(legal_actions)
    max_legal_logit = logits_masked.max(axis=-1, keepdims=True)
    logits_masked = logits_masked - max_legal_logit
    # exp_logits_masked is 0 for illegal actions.
    exp_logits_masked = jnp.exp(logits_masked)

    baseline = jnp.log(jnp.sum(exp_logits_masked, axis=-1, keepdims=True))
    # Subtract baseline from logits. We do not simply return
    #     logits_masked - baseline
    # because that has -inf for illegal actions, or
    #     legal_actions * (logits_masked - baseline)
    # because that leads to 0 * -inf == nan for illegal actions.
    log_policy = legal_actions * (logits - max_legal_logit - baseline)
    return log_policy


def get_most_recent_file(dir_path: str, pattern: str = None):
    # List all files in the directory
    if not os.path.exists(dir_path):
        return None
    files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f)) and "ckpt" in f
    ]

    if pattern is not None:
        files = list(filter(lambda x: pattern in x, files))

    if not files:
        return None

    # Sort files by creation time
    most_recent_file = max(files, key=os.path.getctime)

    return most_recent_file


def get_num_params(vars: Params, n: int = 3) -> dict[str, dict[str, float]]:
    def calculate_params(key: str, vars: Params) -> int:
        total = 0
        for key, value in vars.items():
            if isinstance(value, jax.Array):
                total += math.prod(value.shape)
            else:
                total += calculate_params(key, value)
        return total

    def build_param_dict(
        vars: Params, total_params: int, current_depth: int
    ) -> dict[str, dict[str, float]]:
        param_dict = {}
        for key, value in vars.items():
            if isinstance(value, jax.Array):
                num_params = math.prod(value.shape)
                param_dict[key] = {
                    "num_params": num_params,
                    "ratio": num_params / total_params,
                }
            else:
                nested_params = calculate_params(key, value)
                param_entry = {
                    "num_params": nested_params,
                    "ratio": nested_params / total_params,
                }
                if current_depth < n - 1:
                    param_entry["details"] = build_param_dict(
                        value, total_params, current_depth + 1
                    )
                param_dict[key] = param_entry
        return param_dict

    total_params = calculate_params("base", vars)
    return build_param_dict(vars, total_params, 0)


def assert_no_nan_or_inf(gradients, path=""):
    if isinstance(gradients, dict):
        for key, value in gradients.items():
            new_path = f"{path}/{key}" if path else key
            assert_no_nan_or_inf(value, new_path)
    else:
        if jnp.isnan(gradients).any() or jnp.isinf(gradients).any():
            raise ValueError(f"Gradient at {path} contains NaN or Inf values.")
