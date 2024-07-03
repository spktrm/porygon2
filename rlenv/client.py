import jax
import chex
import uvloop
import functools
import numpy as np

from typing import Sequence

from rlenv.env import ParallelEnvironment, EnvStep, ActorStep, TimeStep

from ml.config import RNaDConfig
from ml.model import get_model
from ml.utils import Params

uvloop.install()


class BatchCollector:
    def __init__(self, config: RNaDConfig):
        self.config = config

        # The random facilities for jax and numpy.
        self._np_rng = np.random.RandomState(self.config.seed)

        self.game = ParallelEnvironment(self.config.batch_size)
        self.network = get_model()
        self.params = None

    def set_params(self, params):
        self.params = jax.block_until_ready(params)

    def _batch_of_states_as_env_step(self, envs: Sequence[EnvStep]) -> EnvStep:
        return jax.tree_util.tree_map(lambda *e: np.stack(e, axis=0), *envs)

    def _batch_of_states_apply_action(self, actions: chex.Array) -> Sequence[EnvStep]:
        """Apply a batch of `actions` to a parallel list of `states`."""
        return self.game.step(list(actions))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _network_jit_apply(self, params: Params, env_steps: EnvStep) -> chex.Array:
        def apply_network(env_step: EnvStep):
            pi, _, _, _ = self.network.apply(params, env_step)
            # return self.config.finetune.post_process_policy(pi, env_step.legal)
            return pi

        # Use jax.vmap to vectorize `apply_network` across the first dimension of env_steps
        vmapped_apply_network = jax.vmap(apply_network)

        # Apply the vectorized function to the batch of env_steps
        return vmapped_apply_network(env_steps)

    def actor_step(self, env_step: EnvStep):
        # pi = self._network_jit_apply(self.params, env_step)
        # pi = np.asarray(pi).astype("float64")

        pi = env_step.legal
        # TODO(author18): is this policy normalization really needed?
        pi = pi / np.sum(pi, axis=-1, keepdims=True)

        action = np.apply_along_axis(
            lambda x: self._np_rng.choice(range(pi.shape[-1]), p=x), axis=-1, arr=pi
        )
        # TODO(author16): reapply the legal actions mask to bullet-proof sampling.
        action_oh = np.zeros(pi.shape, dtype="float64")
        action_oh[range(pi.shape[0]), action] = 1.0

        actor_step = ActorStep(
            policy=pi, action_oh=action_oh, rewards=()
        )  # pytype: disable=wrong-arg-types  # numpy-scalars

        return action, actor_step

    def collect_batch_trajectory(self, resolution: int = 32) -> TimeStep:
        states = self.game.reset()
        timesteps = []
        env_step = self._batch_of_states_as_env_step(states)

        for i in range(self.config.trajectory_max):
            prev_env_step = env_step
            a, actor_step = self.actor_step(env_step)

            states = self._batch_of_states_apply_action(a)
            env_step = self._batch_of_states_as_env_step(states)
            timestep = TimeStep(
                env=prev_env_step,
                actor=ActorStep(
                    action_oh=actor_step.action_oh,
                    policy=actor_step.policy,
                    rewards=env_step.rewards,
                ),
            )
            timesteps.append(timestep)

            if (env_step.valid == 0).all() and i % resolution == 0:
                break

        # Concatenate all the timesteps together to form a single rollout [T, B, ..]
        return jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *timesteps)
