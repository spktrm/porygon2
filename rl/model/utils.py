from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import NamedTuple, TypeVar

import chex
import jax
import jax.numpy as jnp


class ParamsContainer(NamedTuple):
    step_count: int

    player_frame_count: int
    builder_frame_count: int

    player_params: chex.ArrayTree
    builder_params: chex.ArrayTree

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParamsContainer):
            return NotImplemented
        return self.step_count == other.step_count

    def __repr__(self) -> str:
        return f"ParamsContainer(step_count={self.step_count})"


Params = chex.ArrayTree
Optimizer = Callable[[Params, Params], Params]  # (params, grads) -> params
PredT = TypeVar("PredT")  # whatever structure 'pred' has, we return the same


def legal_policy(logits: jax.Array, legal_actions: jax.Array) -> jax.Array:
    """A soft-max policy that respects legal_actions."""
    chex.assert_equal_shape((logits, legal_actions), dims=-1)
    # Fiddle a bit to make sure we don't generate NaNs or Inf in the middle.
    masked_logits = jnp.where(legal_actions, logits, -1e9)
    policy = jax.nn.softmax(masked_logits, axis=-1)
    return jnp.where(legal_actions, policy, 0.0)


def legal_log_policy(logits: jax.Array, legal_actions: jax.Array) -> jax.Array:
    """Return the log of the policy on legal action, 0 on illegal action."""
    chex.assert_equal_shape((logits, legal_actions), dims=-1)
    masked_logits = jnp.where(legal_actions, logits, -1e9)
    log_policy = jax.nn.log_softmax(masked_logits, axis=-1)
    return jnp.where(legal_actions, log_policy, 0.0)


def get_num_params(vars: Params, n: int = 3) -> dict[str, dict[str, float]]:
    def calculate_params(key: str, vars: Params) -> int:
        total = 0
        for key, value in vars.items():
            # Recurse on mappings, count everything else as an array leaf:
            # checkpoint-restored trees carry numpy arrays, which fail an
            # isinstance(value, jax.Array) check and would be recursed into.
            if isinstance(value, Mapping):
                total += calculate_params(key, value)
            else:
                total += math.prod(value.shape)
        return total

    def build_param_dict(
        vars: Params, total_params: int, current_depth: int
    ) -> dict[str, dict[str, float]]:
        param_dict = {}
        for key, value in vars.items():
            if not isinstance(value, Mapping):
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
        return dict(
            sorted(
                param_dict.items(),
                key=lambda item: getattr(item[1], "num_params", 0),
                reverse=True,
            )
        )

    total_params = calculate_params("base", vars)
    return build_param_dict(vars, total_params, 0)

