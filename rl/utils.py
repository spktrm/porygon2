import os

import jax
import jax.experimental
import jax.experimental.compilation_cache
import jax.experimental.compilation_cache.compilation_cache
import jax.numpy as jnp

JAX_JIT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "../.jax_jit_cache")


def init_jax_jit_cache(jax_jit_cache_path: str = JAX_JIT_CACHE_PATH):
    if not os.path.exists(jax_jit_cache_path):
        os.mkdir(jax_jit_cache_path)
    jax.experimental.compilation_cache.compilation_cache.set_cache_dir(
        jax_jit_cache_path
    )


def average(values: jax.Array, valid: jax.Array, axis: int | None = None):
    """Calculate the average of values, ignoring invalid entries."""
    return jnp.where(valid, values, 0).sum(axis=axis) / (
        jnp.sum(valid, axis=axis).clip(min=1)
    )
