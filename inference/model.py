import pickle
from typing import Tuple

import chex
import numpy as np

from inference.interfaces import PredictionResponse
from ml.arch.config import get_model_cfg
from ml.arch.model import get_model
from ml.config import FineTuning
from ml.utils import get_most_recent_file
from rlenv.interfaces import EnvStep


class InferenceModel:
    def __init__(self, fpath: str = None, seed: int = 42):
        self.np_rng = np.random.RandomState(seed)

        model_config = get_model_cfg()
        self.network = get_model(model_config)
        self.finetuning = FineTuning()

        if not fpath:
            fpath = get_most_recent_file("./ckpts")
        print(f"loading checkpoint from {fpath}")
        with open(fpath, "rb") as f:
            step = pickle.load(f)
        self.params = step["params"]

    # @functools.partial(jax.jit, static_argnums=(0,))
    def _network_jit_apply(
        self, env_step: EnvStep
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        output = self.network.apply(self.params, env_step)
        pi = self.finetuning.post_process_policy(output.pi, env_step.legal)
        return pi, output.v, output.log_pi, output.logit

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
