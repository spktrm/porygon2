import functools
import pickle
import threading

import jax
import jax.numpy as jnp
import numpy as np

from inference.interfaces import PredictionResponse
from rl.actor.agent import Agent
from rl.environment.interfaces import TimeStep
from rl.environment.utils import get_ex_step
from rl.model.player_model import get_player_model
from rl.model.utils import BIAS_VALUE, get_most_recent_file

np.set_printoptions(precision=2, suppress=True)
jnp.set_printoptions(precision=2, suppress=True)


def threshold(arr: np.ndarray, thresh: float = 0.1):
    pi = np.where(arr < thresh, 0, arr)
    return pi / pi.sum(axis=-1, keepdims=True)


def restrict_values(arr: np.ndarray):
    if arr.dtype.name == "bfloat16":
        return np.clip(arr, a_min=BIAS_VALUE, a_max=-BIAS_VALUE)
    else:
        return np.nan_to_num(arr)


class InferenceModel:
    def __init__(self, fpath: str = None, seed: int = 42):
        self.np_rng = np.random.RandomState(seed)

        self.network = get_player_model()

        apply = functools.partial(self.network.apply, temp=0.2)
        self.agent = Agent(jax.vmap(apply, in_axes=(None, 1)), threading.Lock())
        self.rng_key = jax.random.PRNGKey(seed)

        if not fpath:
            fpath = get_most_recent_file("./ckpts")
        print(f"loading checkpoint from {fpath}")
        with open(fpath, "rb") as f:
            step = pickle.load(f)

        self.params = step["params"]
        print("initializing...")
        self.predict(get_ex_step(expand=False))  # warm up the model
        print("model initialized!")

    def split_rng(self) -> jax.Array:
        self.rng_key, subkey = jax.random.split(self.rng_key)
        return subkey

    def predict(self, timestep: TimeStep):
        rng_key = self.split_rng()
        actor_step = self.agent.step(rng_key, self.params, timestep)
        model_output = actor_step.model_output
        action = actor_step.action
        return PredictionResponse(
            pi=restrict_values(model_output.pi).flatten().tolist(),
            log_pi=restrict_values(model_output.log_pi).flatten().tolist(),
            logit=restrict_values(model_output.logit).flatten().tolist(),
            v=model_output.v.item(),
            action=action.item(),
        )
