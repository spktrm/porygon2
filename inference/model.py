import cloudpickle as pickle
import jax
import jax.numpy as jnp
import numpy as np

from inference.interfaces import ResetResponse, StepResponse
from rl.actor.agent import Agent
from rl.environment.env import TeamBuilderEnvironment
from rl.environment.interfaces import BuilderTransition, PlayerActorInput
from rl.environment.utils import get_ex_player_step
from rl.learner.config import get_learner_config
from rl.model.builder_model import get_builder_model
from rl.model.config import get_builder_model_config, get_player_model_config
from rl.model.heads import HeadParams
from rl.model.player_model import get_player_model
from rl.model.utils import get_most_recent_file

np.set_printoptions(precision=2, suppress=True)
jnp.set_printoptions(precision=2, suppress=True)


def restrict_values(arr: np.ndarray):
    if arr.dtype.name == "bfloat16":
        finfo = jnp.finfo(arr.dtype)
        return np.clip(arr, a_min=finfo.min, a_max=finfo.max)
    else:
        return np.nan_to_num(arr)


class InferenceModel:
    def __init__(
        self,
        generation: int,
        fpath: str = None,
        seed: int = 42,
        player_head_params: HeadParams = HeadParams(),
        builder_head_params: HeadParams = HeadParams(),
    ):
        self._learner_config = get_learner_config()
        self._player_model_config = get_player_model_config(
            self._learner_config.generation, train=False
        )
        self._builder_model_config = get_builder_model_config(
            self._learner_config.generation, train=False
        )

        self._player_network = get_player_model(self._player_model_config)
        self._builder_network = get_builder_model(self._builder_model_config)

        self._agent = Agent(
            player_apply_fn=self._player_network.apply,
            builder_apply_fn=self._builder_network.apply,
            player_head_params=player_head_params,
            builder_head_params=builder_head_params,
        )
        self._rng_key = jax.random.key(seed)

        if not fpath:
            fpath = get_most_recent_file(f"./ckpts/gen{generation}")
        print(f"loading checkpoint from {fpath}")
        with open(fpath, "rb") as f:
            step = pickle.load(f)

        self._player_params = step["player_state"]["params"]
        self._builder_params = step["builder_state"]["params"]

        print("initializing...")
        self._builder_env = TeamBuilderEnvironment(
            generation=self._learner_config.generation, smogon_format="ou_all_formats"
        )
        self.reset()  # warm up the model

        ex_actor_input, _ = jax.tree.map(lambda x: x[:, 0], get_ex_player_step())
        self.step(
            PlayerActorInput(
                env=jax.tree.map(lambda x: x[0], ex_actor_input.env),
                packed_history=ex_actor_input.packed_history,
                history=ex_actor_input.history,
            )
        )  # warm up the model
        print("model initialized!")

    def split_rng(self, num_splits: int = 1) -> tuple[jax.Array]:
        self._rng_key, *subkeys = jax.random.split(self._rng_key, num_splits + 1)
        if num_splits == 1:
            return subkeys[0]
        return tuple(subkeys)

    def reset(self):

        rng_key = self.split_rng()

        builder_subkeys = jax.random.split(
            rng_key, self._builder_env._max_trajectory_length + 1
        )
        build_traj = []

        builder_actor_input = self._builder_env.reset()
        for builder_step_index in range(builder_subkeys.shape[0]):
            builder_agent_output = self._agent.step_builder(
                builder_subkeys[builder_step_index],
                self._builder_params,
                builder_actor_input,
            )
            builder_transition = BuilderTransition(
                env_output=builder_actor_input.env,
                agent_output=builder_agent_output,
            )
            build_traj.append(builder_transition)
            if builder_actor_input.env.done.item():
                break
            builder_actor_input = self._builder_env.step(builder_agent_output)

        # Send set tokens to the player environment.
        return ResetResponse(
            species_indices=builder_actor_input.history.species_tokens.reshape(
                -1
            ).tolist(),
            packed_set_indices=builder_actor_input.history.packed_set_tokens.reshape(
                -1
            ).tolist(),
            v=builder_agent_output.actor_output.value_head.logits.item(),
        )

    def step(self, timestep: PlayerActorInput):
        rng_key = self.split_rng()

        agent_output = self._agent.step_player(rng_key, self._player_params, timestep)
        actor_output = agent_output.actor_output
        return StepResponse(
            v=actor_output.value_head.logits.item(),
            log_prob=actor_output.action_head.log_prob.item(),
            entropy=actor_output.action_head.entropy.item(),
            src=agent_output.actor_output.action_head.src_index.item(),
            tgt=agent_output.actor_output.action_head.tgt_index.item(),
        )
