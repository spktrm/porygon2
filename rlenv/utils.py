import jax
import numpy as np

from typing import Callable, Sequence, TypeVar

from rlenv.data import NUM_HISTORY


T = TypeVar("T")


# @jax.jit
def stack_steps(steps: Sequence[T]) -> T:
    return jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *steps)


def padnstack(
    arr: np.ndarray,
    pad_fn: Callable[[np.ndarray], np.ndarray] = lambda arr: arr.copy(),
) -> np.ndarray:
    num_repeats = NUM_HISTORY - arr.shape[0]
    padding: np.ndarray = pad_fn(arr[0])[None]
    return np.concatenate((arr, padding.repeat(num_repeats, 0)))
