import jax
import numpy as np

from rl.actor.agent import Agent
from rl.environment.env import TeamBuilderEnvironment
from rl.environment.interfaces import BuilderTransition
from rl.environment.utils import split_rng
from rl.learner.learner import Learner
from rl.model.utils import Params, ParamsContainer


class BuilderActor:
    """Manages the state of a single agent/environment interaction loop."""

    def __init__(self, agent: Agent, learner: Learner, rng_seed: int = 42):
        self._agent = agent
        self._env = TeamBuilderEnvironment(
            generation=learner.config.generation,
            smogon_format=learner.config.smogon_format,
        )
        self._learner = learner
        self._rng_key = jax.random.key(rng_seed)

    def split_rng(self) -> jax.Array:
        self._rng_key, subkey = split_rng(self._rng_key)
        return subkey

    def pull_main_player(self) -> ParamsContainer:
        league = self._learner.league
        return league.get_main_player()

    def unroll(self, rng_key: jax.Array, builder_params: Params) -> None:
        """Run unroll_length agent/environment steps, returning the trajectory."""

        builder_unroll_length = self._env.length
        builder_subkeys = split_rng(rng_key, builder_unroll_length + 1)
        build_traj = []

        # Reset the builder environment.
        builder_actor_input = self._env.reset(builder_subkeys[0])

        # Rollout the builder environment.
        for builder_step_index in range(1, builder_subkeys.shape[0]):
            builder_agent_output = self._agent.step_builder(
                builder_subkeys[builder_step_index],
                builder_params,
                builder_actor_input,
            )
            builder_transition = BuilderTransition(
                env_output=builder_actor_input.env,
                agent_output=builder_agent_output,
            )
            build_traj.append(builder_transition)
            if builder_actor_input.env.done.item():
                break
            builder_actor_input = self._env.step(builder_agent_output)

        if len(build_traj) < builder_unroll_length:
            build_traj += [builder_transition] * (
                builder_unroll_length - len(build_traj)
            )

        # Pack the trajectory and reset parent state.
        builder_trajectory = jax.device_get(build_traj)
        builder_trajectory: BuilderTransition = jax.tree.map(
            lambda *xs: np.stack(xs), *builder_trajectory
        )

        add_cond = self._learner.builder_replay._add_cv
        with add_cond:
            add_cond.wait_for(
                lambda: self._learner.done
                or self._learner.builder_replay.ready_to_add()
            )
            if self._learner.done:
                return
            self._learner.builder_replay.add_trajectory(
                builder_trajectory, builder_actor_input.history
            )

        sample_cond = self._learner.builder_replay._sample_cv
        with sample_cond:
            sample_cond.notify_all()
