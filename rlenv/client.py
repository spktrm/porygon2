import functools
import pickle
from typing import Callable, Sequence

import chex
import flax.linen as nn
import jax
import numpy as np

from ml.config import FineTuning
from ml.utils import Params
from rlenv.data import EVALUATION_SOCKET_PATH, SocketPath
from rlenv.env import EnvStep, ParallelEnvironment
from rlenv.interfaces import ActorStep, ModelOutput, TimeStep
from rlenv.utils import stack_steps


class BatchCollector:
    def __init__(self, network: nn.Module, path: SocketPath, batch_size: int):
        self.game = ParallelEnvironment(batch_size, path)
        self.network = network
        self.is_eval = True if path == EVALUATION_SOCKET_PATH else False
        self.finetuning = FineTuning()

    def _batch_of_states_apply_action(self, actions: chex.Array) -> Sequence[EnvStep]:
        """Apply a batch of `actions` to a parallel list of `states`."""
        return self.game.step(list(actions))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _network_jit_apply(self, params: Params, env_steps: EnvStep) -> chex.Array:
        rollout: Callable[[Params, EnvStep], ModelOutput] = jax.vmap(
            self.network.apply, (None, 0), 0
        )
        output = rollout(params, env_steps)
        pi = output.pi
        return jax.lax.cond(
            self.is_eval,
            functools.partial(self.finetuning.post_process_policy, pi, env_steps.legal),
            lambda: pi,
        )

    def actor_step(self, params: Params, env_step: EnvStep):
        pi = self._network_jit_apply(params, env_step)
        try:
            action = np.apply_along_axis(
                lambda x: np.random.choice(range(pi.shape[-1]), p=x), axis=-1, arr=pi
            )
        except Exception as e:
            with open("bad_state.pkl", "wb") as f:
                pickle.dump(env_step, f)

            raise e

        actor_step = ActorStep(
            policy=pi, win_rewards=(), hp_rewards=(), fainted_rewards=(), action=action
        )
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
                    win_rewards=env_step.win_rewards,
                    hp_rewards=env_step.hp_rewards,
                    fainted_rewards=env_step.fainted_rewards,
                    switch_rewards=env_step.switch_rewards,
                ),
            )
            timesteps.append(timestep)

            if (~env_step.valid).all() and state_index % resolution == 0:
                break

            state_index += 1

        # Concatenate all the timesteps together to form a single rollout [T, B, ..]
        return stack_steps(timesteps)
