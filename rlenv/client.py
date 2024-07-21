import jax
import chex
import functools
import numpy as np
import flax.linen as nn

from typing import Sequence

from ml.utils import Params

from rlenv.data import SocketPath
from rlenv.env import ParallelEnvironment, EnvStep
from rlenv.interfaces import ActorStep, TimeStep
from rlenv.utils import stack_steps


class BatchCollector:
    def __init__(self, network: nn.Module, path: SocketPath, batch_size: int):
        self.game = ParallelEnvironment(batch_size, path)
        self.network = network

    def _batch_of_states_apply_action(self, actions: chex.Array) -> Sequence[EnvStep]:
        """Apply a batch of `actions` to a parallel list of `states`."""
        return self.game.step(list(actions))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _network_jit_apply(self, params: Params, env_steps: EnvStep) -> chex.Array:
        rollout = jax.vmap(self.network.apply, (None, 0), 0)
        return rollout(params, env_steps)

    def actor_step(self, params: Params, env_step: EnvStep):
        pi, _, _, _ = self._network_jit_apply(params, env_step)
        action = np.apply_along_axis(
            lambda x: np.random.choice(range(pi.shape[-1]), p=x), axis=-1, arr=pi
        )
        actor_step = ActorStep(policy=pi, rewards=(), action=action)
        return action, actor_step

    def collect_batch_trajectory(
        self, params: Params, resolution: int = 32
    ) -> TimeStep:
        states = self.game.reset()
        timesteps = []
        env_step: EnvStep = stack_steps(states)

        state_index = 0
        while True:
            prev_env_step = env_step
            a, actor_step = self.actor_step(params, env_step)

            states = self._batch_of_states_apply_action(a)
            env_step = stack_steps(states)
            timestep = TimeStep(
                env=prev_env_step,
                actor=ActorStep(
                    action=actor_step.action,
                    policy=actor_step.policy,
                    rewards=env_step.rewards,
                ),
            )
            timesteps.append(timestep)

            if (~env_step.valid).all() and state_index % resolution == 0:
                break

            state_index += 1

        # Concatenate all the timesteps together to form a single rollout [T, B, ..]
        return stack_steps(timesteps)
