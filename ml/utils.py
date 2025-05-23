import os
from typing import Callable

import chex
import jax
import jax.numpy as jnp

Params = chex.ArrayTree
Optimizer = Callable[[Params, Params], Params]  # (params, grads) -> params


def get_most_recent_file(dir_path: str, pattern: str = None):
    # List all files in the directory
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


def breakpoint_w_func(func: callable, x):
    func_val = func(x)

    def true_fn(x):
        jax.debug.breakpoint()

    def false_fn(x):
        pass

    jax.lax.cond(func_val, true_fn, false_fn, x)


def breakpoint_if_nonfinite(x):
    breakpoint_w_func(x, lambda z: jnp.isfinite(z).all())
