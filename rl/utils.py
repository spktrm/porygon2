import jax
import jax.numpy as jnp


def average(values: jax.Array, valid: jax.Array, axis: int | None = None):
    """Calculate the average of values, ignoring invalid entries."""
    return jnp.where(valid, values, 0).sum(axis=axis) / (
        jnp.sum(valid, axis=axis).clip(min=1)
    )
