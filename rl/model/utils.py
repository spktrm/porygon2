import os
from typing import Callable

import chex

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
