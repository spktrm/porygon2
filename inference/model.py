import pickle

import jax
import jax.numpy as jnp
import numpy as np

from inference.interfaces import ResetResponse, StepResponse
from rl.actor.actor import ACTION_TYPE_MAPPING
from rl.actor.agent import Agent
from rl.environment.env import TeamBuilderEnvironment
from rl.environment.interfaces import PlayerActorInput, SamplingConfig
from rl.environment.utils import get_ex_player_step
from rl.learner.config import get_learner_config
from rl.model.builder_model import get_builder_model
from rl.model.config import get_builder_model_config, get_player_model_config
from rl.model.player_model import get_player_model
from rl.model.utils import BIAS_VALUE, get_most_recent_file

np.set_printoptions(precision=2, suppress=True)
jnp.set_printoptions(precision=2, suppress=True)


def restrict_values(arr: np.ndarray):
    if arr.dtype.name == "bfloat16":
        return np.clip(arr, a_min=BIAS_VALUE, a_max=-BIAS_VALUE)
    else:
        return np.nan_to_num(arr)


class InferenceModel:
    def __init__(self, fpath: str = None, seed: int = 42, precision: int = 2):
        self.learner_config = get_learner_config()
        self.player_model_config = get_player_model_config(
            self.learner_config.generation
        )
        self.builder_model_config = get_builder_model_config(
            self.learner_config.generation
        )

        self.player_network = get_player_model(self.player_model_config)
        self.builder_network = get_builder_model(self.builder_model_config)

        self._agent = Agent(
            player_apply_fn=jax.vmap(self.player_network.apply, in_axes=(None, 1)),
            builder_apply_fn=jax.vmap(self.builder_network.apply, in_axes=(None, 1)),
            player_sampling_config=SamplingConfig(temp=1.0, min_p=0.05),
            builder_sampling_config=SamplingConfig(temp=1.0, min_p=0.05),
        )
        self.rng_key = jax.random.key(seed)
        self.precision = precision

        if not fpath:
            fpath = get_most_recent_file("./ckpts/gen9")
        print(f"loading checkpoint from {fpath}")
        with open(fpath, "rb") as f:
            step = pickle.load(f)

        self.player_params = step["player_state"]["params"]
        self.builder_params = step["builder_state"]["params"]

        print("initializing...")
        self.reset()  # warm up the model

        ts: PlayerActorInput = jax.tree.map(lambda x: x[:, 0], get_ex_player_step())
        self.step(
            PlayerActorInput(
                env=jax.tree.map(lambda x: x[0], ts.env), history=ts.history
            )
        )  # warm up the model
        print("model initialized!")

    def split_rng(self) -> jax.Array:
        self.rng_key, subkey = jax.random.split(self.rng_key)
        return subkey

    def reset(self):
        rng_key = self.split_rng()
        builder_subkeys = jax.random.split(rng_key, 7)

        builder_env = TeamBuilderEnvironment(self.learner_config.generation)

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

    def step(self, timestep: PlayerActorInput):
        rng_key = self.split_rng()
        actor_step = self._agent.step_player(rng_key, self.player_params, timestep)
        model_output = actor_step.actor_output
        return StepResponse(
            action_type_logits=restrict_values(model_output.action_type_logits),
            move_logits=restrict_values(model_output.move_logits),
            wildcard_logits=restrict_values(model_output.wildcard_logits),
            switch_logits=restrict_values(model_output.switch_logits),
            v=model_output.v.item(),
            action_type=ACTION_TYPE_MAPPING[actor_step.action_type.item()],
            move_slot=actor_step.move_slot.item(),
            switch_slot=actor_step.switch_slot.item(),
            wildcard_slot=actor_step.wildcard_slot.item(),
        )
