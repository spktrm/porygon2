import queue
from typing import Callable

import jax
import numpy as np

from rl.actor.agent import Agent
from rl.environment.env import SinglePlayerSyncEnvironment
from rl.environment.interfaces import TimeStep, Transition
from rl.environment.protos.features_pb2 import InfoFeature
from rl.environment.utils import clip_history
from rl.model.utils import Params


class Actor:
    """Manages the state of a single agent/environment interaction loop."""

    def __init__(
        self,
        agent: Agent,
        env: SinglePlayerSyncEnvironment,
        unroll_length: int,
        params_for_actor: Callable[[], tuple[int, int, Params]],
        queue: queue.Queue | None = None,
        rng_seed: int = 42,
    ):
        self._agent = agent
        self._env = env
        self._unroll_length = unroll_length
        self._queue = queue
        self._params_for_actor = params_for_actor
        self._rng_key = jax.random.PRNGKey(rng_seed)

    def _preprocess_timestep(self, timestep: TimeStep):
        return TimeStep(
            env=timestep.env,
            history=clip_history(timestep.history, resolution=128),
        )

    def unroll(self, rng_key: jax.Array, frame_count: int, params: Params):
        """Run unroll_length agent/environment steps, returning the trajectory."""
        unprocessed_timestep = self._env.reset()
        traj = []
        # Unroll one longer if trajectory is empty.
        subkeys = jax.random.split(rng_key, self._unroll_length)
        for i in range(self._unroll_length):
            timestep = self._preprocess_timestep(unprocessed_timestep)
            actorstep = self._agent.step(subkeys[i], params, timestep)
            transition = Transition(
                timestep=TimeStep(env=timestep.env), actorstep=actorstep
            )
            traj.append(transition)
            if timestep.env.done.item():
                break
            unprocessed_timestep = self._env.step(actorstep.action)

        if len(traj) < self._unroll_length:
            traj += [transition] * (self._unroll_length - len(traj))

        # Pack the trajectory and reset parent state.
        trajectory = jax.device_get(traj)
        trajectory: Transition = jax.tree.map(lambda *xs: np.stack(xs), *trajectory)
        trajectory = Transition(
            timestep=TimeStep(
                env=trajectory.timestep.env,
                history=unprocessed_timestep.history,
            ),
            actorstep=trajectory.actorstep,
        )

        request_count = trajectory.timestep.env.info[
            ..., InfoFeature.INFO_FEATURE__REQUEST_COUNT
        ]
        if np.any(request_count[1:] < request_count[:-1]):
            raise ValueError("Request count should be non-decreasing.")

        return trajectory

    def split_rng(self) -> jax.Array:
        self._rng_key, subkey = jax.random.split(self._rng_key)
        return subkey

    def unroll_and_push(self, frame_count: int, params: Params):
        """Run one unroll and send trajectory to learner."""
        params = jax.device_put(params)
        subkey = self.split_rng()
        act_out = self.unroll(rng_key=subkey, frame_count=frame_count, params=params)
        if self._queue is not None:
            self._queue.put(act_out)

    def pull_params(self):
        return self._params_for_actor()
