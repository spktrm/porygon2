import jax
import jax.numpy as jnp
import numpy as np

from inference.interfaces import ResetResponse, StepResponse
from rl import checkpoint
from rl.environment.env import TeamBuilderEnvironment
from rl.environment.interfaces import PlayerActorInput
from rl.environment.utils import (
    ACTOR_HISTORY_MIN_LENGTH,
    clip_history,
    clip_packed_history,
    get_ex_player_step,
)
from rl.model.builder_model import get_builder_model
from rl.model.config import get_builder_model_config, get_player_model_config
from rl.model.heads import HeadParams
from rl.model.player_model import get_player_model
from rl.model.utils import ParamsContainer
from rl.online.agent import Agent, resolve_actor_device
from rl.online.config import get_learner_config

np.set_printoptions(precision=2, suppress=True)
jnp.set_printoptions(precision=2, suppress=True)


class InferenceModel:
    """One checkpoint behind the HTTP eval server: the EMA `target_params`
    (what the actors and the league play) through the same actor networks
    and Agent main.py builds, on config.player_actor_device."""

    def __init__(
        self,
        generation: int,
        fpath: str = None,
        seed: int = 42,
        player_head_params: HeadParams = HeadParams(),
        builder_head_params: HeadParams = HeadParams(),
    ):
        self._learner_config = get_learner_config()
        actor_device, actor_dtype = resolve_actor_device(
            self._learner_config.player_actor_device
        )
        self._player_model_config = get_player_model_config(
            self._learner_config.generation, train=False, dtype=actor_dtype
        )
        self._builder_model_config = get_builder_model_config(
            self._learner_config.generation, train=False, dtype=actor_dtype
        )

        self._player_network = get_player_model(self._player_model_config)
        self._builder_network = get_builder_model(self._builder_model_config)

        self._agent = Agent(
            player_apply_fn=self._player_network.apply,
            builder_apply_fn=self._builder_network.apply,
            player_head_params=player_head_params,
            builder_head_params=builder_head_params,
            device=actor_device,
        )
        self._rng_key = jax.random.key(seed)

        if not fpath:
            fpath = checkpoint.most_recent_ckpt_dir(f"./ckpts/gen{generation}")
        print(f"loading checkpoint from {fpath}")
        # The Agent keys its device cache by container identity, so ONE
        # container for the process: both heads' params are committed once.
        self._params = ParamsContainer(
            step_count=0,
            player_frame_count=0,
            builder_frame_count=0,
            player_params=checkpoint.load_component(fpath, "player", "target_params"),
            builder_params=checkpoint.load_component(fpath, "builder", "target_params"),
        )

        print("initializing...")
        self._builder_env = TeamBuilderEnvironment(
            generation=self._learner_config.generation, smogon_format="ou"
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
        """Builds a team the way BuilderActor.unroll does, one agent step
        per packed-set feature until the environment reports done."""
        rng_key = self.split_rng()

        builder_subkeys = jax.random.split(rng_key, self._builder_env.length + 1)
        builder_actor_input = self._builder_env.reset(builder_subkeys[0])
        for builder_step_index in range(1, builder_subkeys.shape[0]):
            builder_agent_output = self._agent.step_builder(
                builder_subkeys[builder_step_index],
                self._params,
                builder_actor_input,
            )
            if builder_actor_input.env.done.item():
                break
            builder_actor_input = self._builder_env.step(builder_agent_output)

        team_tokens = builder_actor_input.history.packed_team_member_tokens
        return ResetResponse(packed_team=team_tokens.reshape(-1).tolist())

    def step(self, timestep: PlayerActorInput):
        """One request. The history is tail-windowed exactly as
        PlayerActor.clip_actor_history does before the Agent pads it to
        its joint bucket level."""
        rng_key = self.split_rng()
        timestep = PlayerActorInput(
            env=timestep.env,
            packed_history=clip_packed_history(
                timestep.packed_history, min_length=ACTOR_HISTORY_MIN_LENGTH
            ),
            history=clip_history(timestep.history, min_length=ACTOR_HISTORY_MIN_LENGTH),
        )

        agent_output = self._agent.step_player(rng_key, self._params, timestep)
        actor_output = agent_output.actor_output

        floats = dict(
            v_win=actor_output.value_head.expectation.item(),
            log_prob=actor_output.action_head.log_prob.item(),
            entropy=actor_output.action_head.entropy.item(),
        )
        floats = {name: round(value, 3) for name, value in floats.items()}

        return StepResponse(
            **floats,
            cell=actor_output.action_head.action_index.item(),
        )
