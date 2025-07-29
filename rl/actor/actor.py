import queue

import jax
import numpy as np

from rl.actor.agent import Agent
from rl.environment.env import SinglePlayerSyncEnvironment
from rl.environment.interfaces import TimeStep, Transition
from rl.environment.protos.features_pb2 import InfoFeature
from rl.environment.utils import clip_history
from rl.learner.learner import Learner
from rl.model.utils import Params


class Actor:
    """Manages the state of a single agent/environment interaction loop."""

    def __init__(
        self,
        agent: Agent,
        env: SinglePlayerSyncEnvironment,
        unroll_length: int,
        learner: Learner,
        queue: queue.Queue | None = None,
        rng_seed: int = 42,
    ):
        self._agent = agent
        self._env = env
        self._unroll_length = unroll_length
        self._queue = queue
        self._learner = learner
        self._rng_key = jax.random.PRNGKey(rng_seed)

    def _preprocess_timestep(self, timestep: TimeStep):
        return TimeStep(
            env=timestep.env,
            history=clip_history(timestep.history, resolution=128),
        )

    def unroll(
        self,
        rng_key: jax.Array,
        frame_count: int,
        player_params: Params,
        builder_params: Params,
    ):
        """Run unroll_length agent/environment steps, returning the trajectory."""
        subkeys = jax.random.split(rng_key, self._unroll_length + 1)

        actor_reset = self._agent.reset(subkeys[0], builder_params)
        tokens_buffer = np.asarray(actor_reset.tokens, dtype=np.int16).view(np.uint8)
        unprocessed_timestep = self._env.reset(tokens_buffer.tobytes())

        traj = []
        # Unroll one longer if trajectory is empty.
        for i in range(1, self._unroll_length):
            timestep = self._preprocess_timestep(unprocessed_timestep)
            actor_step = self._agent.step(subkeys[i], player_params, timestep)
            transition = Transition(
                timestep=TimeStep(env=timestep.env), actor_step=actor_step
            )
            traj.append(transition)
            if timestep.env.done.item():
                break
            unprocessed_timestep = self._env.step(actor_step.action)

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
            actor_step=trajectory.actor_step,
            # Only the first step uses this
            actor_reset=jax.tree.map(lambda x: x[None], actor_reset),
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

    def unroll_and_push(
        self, frame_count: int, player_params: Params, builder_params: Params
    ):
        """Run one unroll and send trajectory to learner."""
        player_params = jax.device_put(player_params)
        builder_params = jax.device_put(builder_params)
        subkey = self.split_rng()
        act_out = self.unroll(
            rng_key=subkey,
            frame_count=frame_count,
            player_params=player_params,
            builder_params=builder_params,
        )
        if self._queue is not None:
            self._queue.put(act_out)

    def pull_params(self):
        return self._learner.params_for_actor
