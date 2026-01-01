from __future__ import annotations

import math
import os
from typing import Callable, NamedTuple, TypeVar, cast

import chex
import jax
import jax.numpy as jnp


class ParamsContainer(NamedTuple):
    step_count: int

    player_frame_count: int
    builder_frame_count: int

    player_params: chex.ArrayTree
    builder_params: chex.ArrayTree

    def __repr__(self) -> str:
        return f"ParamsContainer(step_count={self.step_count})"


Params = chex.ArrayTree
Optimizer = Callable[[Params, Params], Params]  # (params, grads) -> params
PredT = TypeVar("PredT")  # whatever structure 'pred' has, we return the same


def legal_policy(logits: jax.Array, legal_actions: jax.Array) -> jax.Array:
    """A soft-max policy that respects legal_actions."""
    chex.assert_equal_shape((logits, legal_actions), dims=-1)
    # Fiddle a bit to make sure we don't generate NaNs or Inf in the middle.
    masked_logits = jnp.where(legal_actions, logits, jnp.finfo(logits.dtype).min)
    policy = jax.nn.softmax(masked_logits, axis=-1)
    return jnp.where(legal_actions, policy, 0.0)


def legal_log_policy(logits: jax.Array, legal_actions: jax.Array) -> jax.Array:
    """Return the log of the policy on legal action, 0 on illegal action."""
    chex.assert_equal_shape((logits, legal_actions), dims=-1)
    masked_logits = jnp.where(legal_actions, logits, jnp.finfo(logits.dtype).min)
    log_policy = jax.nn.log_softmax(masked_logits, axis=-1)
    return jnp.where(legal_actions, log_policy, 0.0)


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
                    "ratio": f"{num_params / total_params:.3f}",
                }
            else:
                nested_params = calculate_params(key, value)
                param_entry = {
                    "num_params": nested_params,
                    "ratio": f"{nested_params / total_params:.3f}",
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


def promote_map(pred: PredT) -> PredT:
    def maybe_promote(x):
        if isinstance(x, jnp.ndarray) and x.dtype == jnp.bfloat16:
            return x.astype(jnp.float32)
        return x

    out = jax.tree_util.tree_map(maybe_promote, pred)
    return cast(PredT, out)
