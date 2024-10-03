import jax
import numpy as np

from typing import Sequence, TypeVar

from rlenv.data import NUM_HISTORY


T = TypeVar("T")


# @jax.jit
def stack_steps(steps: Sequence[T], axis: int = 0) -> T:
    return jax.tree.map(lambda *xs: np.stack(xs, axis=axis), *steps)


# @jax.jit
def padnstack(arr: np.ndarray) -> np.ndarray:
    stacked = np.resize(arr, (NUM_HISTORY, *arr.shape[1:]))
    mask = np.arange(NUM_HISTORY) < arr.shape[0]
    return np.where(mask[..., *((None,) * (len(arr.shape) - 1))], stacked, 0)
