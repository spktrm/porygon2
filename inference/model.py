import functools
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from inference.interfaces import PredictionResponse
from ml.arch.config import get_model_cfg
from ml.arch.model import get_model
from ml.config import FineTuning
from ml.utils import get_most_recent_file
from rlenv.env import get_ex_step
from rlenv.interfaces import EnvStep, HistoryStep, ModelOutput

np.set_printoptions(precision=2, suppress=True)
jnp.set_printoptions(precision=2, suppress=True)


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
        print("initializing...")
        ex, hx = get_ex_step()
        self.predict(ex, hx)
        print("model initialized!")

    @functools.partial(jax.jit, static_argnums=(0,))
    def _network_jit_apply(self, env_step: EnvStep, history_step: HistoryStep):
        return self.network.apply(self.params, env_step, history_step)

    def predict(self, env_step: EnvStep, history_step: HistoryStep):
        output: ModelOutput = self._network_jit_apply(env_step, history_step)
        finetuned_pi = self.finetuning._threshold(output.pi, env_step.legal)
        action = np.apply_along_axis(
            lambda x: self.np_rng.choice(range(finetuned_pi.shape[-1]), p=x),
            axis=-1,
            arr=finetuned_pi,
        )
        return PredictionResponse(
            pi=output.pi.flatten().tolist(),
            log_pi=output.log_pi.flatten().tolist(),
            logit=output.logit.flatten().tolist(),
            v=output.v.item(),
            action=action.item(),
        )
