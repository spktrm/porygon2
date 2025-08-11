import pickle
import threading

import jax
import jax.numpy as jnp
import numpy as np

from inference.interfaces import HeadOutput, ResetResponse, StepResponse
from rl.actor.actor import ACTION_TYPE_MAPPING
from rl.actor.agent import Agent
from rl.environment.env import TeamBuilderEnvironment
from rl.environment.interfaces import PlayerActorInput, PolicyHeadOutput
from rl.environment.utils import get_ex_player_step
from rl.model.builder_model import get_builder_model
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
    def __init__(
        self,
        fpath: str = None,
        seed: int = 42,
        precision: int = 2,
        do_threshold: bool = False,
    ):
        self.np_rng = np.random.RandomState(seed)

        self.player_network = get_player_model()
        self.builder_network = get_builder_model()

        self._agent = Agent(
            player_apply_fn=jax.vmap(self.player_network.apply, in_axes=(None, 1)),
            builder_apply_fn=jax.vmap(self.builder_network.apply, in_axes=(None, 1)),
            gpu_lock=threading.Lock(),
            do_threshold=do_threshold,
        )
        self.rng_key = jax.random.key(seed)
        self.precision = precision

        if not fpath:
            fpath = get_most_recent_file("./ckpts")
        print(f"loading checkpoint from {fpath}")
        with open(fpath, "rb") as f:
            step = pickle.load(f)

        self.player_params = step["player_state"]["params"]
        self.builder_params = step["builder_state"]["params"]

        print("initializing...")
        self.reset()  # warm up the model
        self.step(get_ex_player_step(expand=False))  # warm up the model
        print("model initialized!")

    def split_rng(self) -> jax.Array:
        self.rng_key, subkey = jax.random.split(self.rng_key)
        return subkey

    def reset(self):
        rng_key = self.split_rng()
        builder_subkeys = jax.random.split(rng_key, 7)

        builder_env = TeamBuilderEnvironment()

        builder_env_output = builder_env.reset()
        for subkey in builder_subkeys:
            builder_agent_output = self._agent.step_builder(
                subkey, self.builder_params, builder_env_output
            )
            if builder_env_output.done.item():
                break
            builder_env_output = builder_env.step(builder_agent_output.action.item())

        # Send set tokens to the player environment.
        tokens_buffer = np.asarray(builder_env_output.tokens, dtype=np.int16)
        return ResetResponse(
            tokens=tokens_buffer,
            v=builder_agent_output.actor_output.v.item(),
        )

    def _jax_head_to_pydantic(self, head_output: PolicyHeadOutput) -> HeadOutput:
        return HeadOutput(
            logits=restrict_values(head_output.logits),
            policy=restrict_values(head_output.policy),
            log_policy=restrict_values(head_output.log_policy),
        )

    def step(self, timestep: PlayerActorInput):
        rng_key = self.split_rng()
        actor_step = self._agent.step_player(rng_key, self.player_params, timestep)
        model_output = actor_step.actor_output
        return StepResponse(
            action_type_head=self._jax_head_to_pydantic(model_output.action_type_head),
            move_head=self._jax_head_to_pydantic(model_output.move_head),
            switch_head=self._jax_head_to_pydantic(model_output.switch_head),
            v=model_output.v.item(),
            action_type=ACTION_TYPE_MAPPING[actor_step.action_type_head.item()],
            move_slot=actor_step.move_head.item(),
            switch_slot=actor_step.switch_head.item(),
        )
