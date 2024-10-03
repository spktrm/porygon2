import os
from typing import Callable

import chex
import jax
import jax.numpy as jnp

Params = chex.ArrayTree
Optimizer = Callable[[Params, Params], Params]  # (params, grads) -> params


def get_most_recent_file(dir_path):
    # List all files in the directory
    files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
    ]

    if not files:
        return None

    # Sort files by creation time
    most_recent_file = max(files, key=os.path.getctime)

    return most_recent_file


def breakpoint_if_nonfinite(x):
    is_finite = jnp.isfinite(x).all()

    def true_fn(x):
        pass

    def false_fn(x):
        jax.debug.breakpoint()

    jax.lax.cond(is_finite, true_fn, false_fn, x)
