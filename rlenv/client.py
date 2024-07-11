import jax
import chex
import functools
import numpy as np
import flax.linen as nn

from typing import Any, Dict, Sequence, TypeVar

from rlenv.env import ParallelEnvironment, EnvStep, ActorStep, TimeStep

from ml.config import RNaDConfig


T = TypeVar("T")


# @jax.jit
def stack_steps(steps: Sequence[T]) -> T:
    return jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *steps)


class BatchCollector:
    def __init__(self, network: nn.Module, config: RNaDConfig):
        self.config = config

        # The random facilities for jax and numpy.
        self.np_rng = np.random.RandomState(self.config.seed)
        self.game = ParallelEnvironment(self.config.batch_size)
        self.network = network

    def _batch_of_states_apply_action(self, actions: chex.Array) -> Sequence[EnvStep]:
        """Apply a batch of `actions` to a parallel list of `states`."""
        return self.game.step(list(actions))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _network_jit_apply(
        self, params: Dict[str, Any], env_steps: EnvStep
    ) -> chex.Array:
        def apply_network(env_step: EnvStep):
            pi, _, _, _ = self.network.apply(params, env_step)
            # return self.config.finetune.post_process_policy(pi, env_step.legal)
            return pi

        # Use jax.vmap to vectorize `apply_network` across the first dimension of env_steps
        vmapped_apply_network = jax.vmap(apply_network)

        # Apply the vectorized function to the batch of env_steps
        return vmapped_apply_network(env_steps)

    def actor_step(self, params: Dict[str, Any], env_step: EnvStep):
        pi = self._network_jit_apply(params, env_step)
        action = np.apply_along_axis(
            lambda x: self.np_rng.choice(range(pi.shape[-1]), p=x), axis=-1, arr=pi
        )
        action_oh = jax.nn.one_hot(action, pi.shape[-1])
        actor_step = ActorStep(policy=pi, action_oh=action_oh, rewards=())
        return action, actor_step

    def collect_batch_trajectory(
        self, params: Dict[str, Any], resolution: int = 32
    ) -> TimeStep:
        states = self.game.reset()
        timesteps = []
        env_step: EnvStep = stack_steps(states)

        n = 0
        while True:
            prev_env_step = env_step
            a, actor_step = self.actor_step(params, env_step)

            states = self._batch_of_states_apply_action(a)
            env_step = stack_steps(states)
            timestep = TimeStep(
                env=prev_env_step,
                actor=ActorStep(
                    action_oh=actor_step.action_oh,
                    policy=actor_step.policy,
                    rewards=env_step.rewards,
                ),
            )
            timesteps.append(timestep)

            if (~env_step.valid).all() and n % resolution == 0:
                break

            n += 1

        # Concatenate all the timesteps together to form a single rollout [T, B, ..]
        return stack_steps(timesteps)
