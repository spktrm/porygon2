import chex
import jax
import jax.numpy as jnp


def traj_aware_average(values: jax.Array, valid: jax.Array):
    """Calculate the average of values, ignoring invalid entries."""
    chex.assert_rank(valid, 2)  # (T, B)

    traj_valid_sum = valid.sum(axis=0, keepdims=True).clip(min=1)
    traj_average = jnp.sum(values * valid) / traj_valid_sum
    return traj_average.mean()


def average(values: jax.Array, valid: jax.Array, axis: int | None = None):
    """Calculate the average of values, ignoring invalid entries."""
    return jnp.where(valid, values, 0).sum(axis=axis) / (
        jnp.sum(valid, axis=axis).clip(min=1)
    )
