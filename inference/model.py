from typing import NamedTuple, Tuple
import chex
import jax
import os
import pickle
import functools

import numpy as np

from inference.interfaces import PredictionResponse
from ml.arch.config import get_model_cfg
from ml.arch.model import get_model

from rlenv.interfaces import EnvStep


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


class InferenceModel:
    def __init__(self, seed: int = 42):
        self.np_rng = np.random.RandomState(seed)

        model_config = get_model_cfg()
        self.network = get_model(model_config)

        latest_ckpt = get_most_recent_file("ml/ckpts")
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        self.params = step["params"]

    @functools.partial(jax.jit, static_argnums=(0,))
    def _network_jit_apply(
        self, env_step: EnvStep
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        return self.network.apply(self.params, env_step)

    def predict(self, env_step: EnvStep):
        pi, v, log_pi, logit = self._network_jit_apply(env_step)
        action = np.apply_along_axis(
            lambda x: self.np_rng.choice(range(pi.shape[-1]), p=x), axis=-1, arr=pi
        )
        return PredictionResponse(
            pi=pi.flatten().tolist(),
            v=v.item(),
            log_pi=log_pi.flatten().tolist(),
            logit=logit.flatten().tolist(),
            action=action.item(),
        )
