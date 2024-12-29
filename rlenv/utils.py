from typing import Sequence, TypeVar

import jax
import numpy as np

from rlenv.data import NUM_HISTORY

T = TypeVar("T")


def stack_steps(steps: Sequence[T], axis: int = 0) -> T:
    return jax.tree.map(lambda *xs: np.stack(xs, axis=axis), *steps)


def concatenate_steps(steps: Sequence[T], axis: int = 0) -> T:
    return jax.tree.map(lambda *xs: np.concatenate(xs, axis=axis), *steps)


def padnstack(arr: np.ndarray, padding: int = NUM_HISTORY) -> np.ndarray:
    output_shape = (padding, *arr.shape[1:])
    result = np.zeros(output_shape, dtype=arr.dtype)
    length_to_copy = min(padding, arr.shape[0])
    result[:length_to_copy] = arr[:length_to_copy]
    return result
