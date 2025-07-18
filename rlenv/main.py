import functools
import logging
import queue
import threading
from typing import Callable, overload

import jax
import jax.numpy as jnp
import numpy as np
import uvloop
from flax.training import train_state
from websockets.sync.client import connect

from ml.utils import Params
from rlenv.env import clip_history, process_state
from rlenv.interfaces import ActorStep, ModelOutput, TimeStep, Transition
from rlenv.protos.features_pb2 import InfoFeature
from rlenv.protos.service_pb2 import (
    ClientRequest,
    EnvironmentResponse,
    ResetRequest,
    StepRequest,
)

# Define the server URI
SERVER_URI = "ws://localhost:8080"

uvloop.install()

logging.basicConfig()
logger = logging.getLogger(__name__)


class NullLogger:
    """Logger that does nothing."""

    def write(self, _):
        pass

    def close(self):
        pass


class AbslLogger:
    """Writes to logging.info."""

    def write(self, d):
        logging.info(d)

    def close(self):
        pass


class SinglePlayerSyncEnvironment:
    def __init__(self, username: str):

        self.username = username
        self.rqid = None
        self.last_state = None

        self.websocket = connect(SERVER_URI, additional_headers={"username": username})

    def _recv(self):
        server_message_data = self.websocket.recv()
        server_message = EnvironmentResponse.FromString(server_message_data)
        self.rqid = server_message.state.rqid
        ex, hx = process_state(server_message.state)
        self.last_state = TimeStep(env=ex, history=hx)
        return self.last_state

    def reset(self):
        self.rqid = None
        reset_message = ClientRequest(reset=ResetRequest(username=self.username))
        self.websocket.send(reset_message.SerializeToString())
        return self._recv()

    def _is_done(self):
        if self.last_state is None:
            return False
        return self.last_state.env.done.item()

    def step(self, action: int | np.ndarray | jax.Array):
        if isinstance(action, jax.Array):
            action = jax.block_until_ready(action).item()
        elif isinstance(action, np.ndarray):
            action = action.item()
        if self._is_done():
            return self.last_state
        step_message = ClientRequest(
            step=StepRequest(action=action, username=self.username, rqid=self.rqid),
        )
        self.websocket.send(step_message.SerializeToString())
        return self._recv()


class Agent:
    """A stateless agent interface."""

    def __init__(
        self, apply_fn: Callable[[TimeStep], ModelOutput], gpu_lock: threading.Lock
    ):
        """Constructs an Agent object."""

        self._apply_fn = apply_fn
        self._lock = gpu_lock

    def step(self, rng_key, params: Params, timestep: TimeStep) -> ActorStep:
        with self._lock:
            return self._step(rng_key, params, timestep)

    @overload
    def _step(self, rng_key, params: Params, timestep: TimeStep) -> ActorStep: ...
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(self, rng_key, params: Params, timestep: TimeStep) -> ActorStep:
        """For a given single-step, unbatched timestep, output the chosen action."""
        # Pad timestep, state to be [T, B, ...] and [B, ...] respectively.

        timestep = TimeStep(
            env=jax.tree.map(lambda t: t[None, None, ...], timestep.env),
            history=jax.tree.map(lambda t: t[:, None, ...], timestep.history),
        )

        model_output = self._apply_fn(params, timestep)
        # Remove the padding from above.
        model_output = jax.tree.map(lambda t: jnp.squeeze(t, axis=(0, 1)), model_output)
        # Sample an action and return.
        action = jax.random.categorical(rng_key, model_output.logit)
        return ActorStep(action=action, model_output=model_output)

    def unroll(self, params: Params, trajectory: TimeStep) -> ActorStep:
        """Unroll the agent along trajectory."""
        model_output = self._apply_fn(params, trajectory)
        return ActorStep(model_output=model_output)


class NetworkContainer:
    def __init__(self, state: train_state.TrainState):
        self._params_for_actor = (int(state.step), jax.device_get(state.params))

    def update(self, state: train_state.TrainState):
        """Update the internal state with a new TrainState."""
        self._params_for_actor = (int(state.step), jax.device_get(state.params))


class Actor:
    """Manages the state of a single agent/environment interaction loop."""

    def __init__(
        self,
        agent: Agent,
        env: SinglePlayerSyncEnvironment,
        unroll_length: int,
        learner_state: NetworkContainer,
        queue: queue.Queue | None = None,
        rng_seed: int = 42,
        logger=None,
    ):
        self._agent = agent
        self._env = env
        self._unroll_length = unroll_length
        self._queue = queue
        self._learner_state = learner_state
        self._rng_key = jax.random.PRNGKey(rng_seed)

        if logger is None:
            logger = NullLogger()
        self._logger = logger

    def _preprocess_timestep(self, timestep: TimeStep):
        return TimeStep(
            env=timestep.env,
            history=clip_history(timestep.history, resolution=64),
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
        return self._learner_state._params_for_actor
