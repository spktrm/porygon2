import functools
import logging
import queue
import threading
from typing import Callable, overload

import jax
import jax.numpy as jnp
import numpy as np
import uvloop
from websockets.sync.client import connect

from ml.arch.model import get_dummy_model
from ml.utils import Params
from rlenv.env import clip_history, get_ex_step, process_state
from rlenv.interfaces import ActorStep, ModelOutput, TimeStep, Transition
from rlenv.protos.service_pb2 import (
    Action,
    ClientMessage,
    ConnectMessage,
    ResetMessage,
    ServerMessage,
    StepMessage,
)
from rlenv.protos.state_pb2 import State
from rlenv.utils import FairLock

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
    def __init__(self, player_id: int, game_id: int):
        self.player_id = player_id
        self.game_id = game_id

        self.rqid = None
        self.last_state = None

        self.websocket = connect(SERVER_URI)
        self._connect()

    def _connect(self):
        connect_message = ClientMessage(
            player_id=self.player_id, game_id=self.game_id, connect=ConnectMessage()
        )
        self.websocket.send(connect_message.SerializeToString())

    def _recv(self):
        server_message_data = self.websocket.recv()
        server_message = ServerMessage.FromString(server_message_data)
        self.rqid = server_message.game_state.rqid
        state = State.FromString(server_message.game_state.state)
        ex, hx = process_state(state)
        self.last_state = TimeStep(env=ex, history=hx)
        return self.last_state

    def reset(self):
        reset_message = ClientMessage(
            player_id=self.player_id, game_id=self.game_id, reset=ResetMessage()
        )
        self.websocket.send(reset_message.SerializeToString())
        return self._recv()

    def _is_done(self):
        if self.last_state is None:
            return False
        return not (self.last_state.env.valid.item())

    def step(self, action: int):
        if self._is_done():
            return self.last_state
        step_message = ClientMessage(
            player_id=self.player_id,
            game_id=self.game_id,
            step=StepMessage(action=Action(rqid=self.rqid, value=action)),
        )
        self.websocket.send(step_message.SerializeToString())
        return self._recv()


class Agent:
    """A stateless agent interface."""

    def __init__(self, apply_fn: Callable[[TimeStep], ModelOutput], gpu_lock: FairLock):
        """Constructs an Agent object.

        Args:
          num_actions: Number of possible actions for the agent. Assumes a flat,
            discrete, 0-indexed action space.
          obs_spec: The observation spec of the environment.
          net_factory: A function from num_actions to a Haiku module representing
            the agent. This module should have an initial_state() function and an
            unroll function.
        """
        self._apply_fn = apply_fn
        self._gpu_lock = gpu_lock

    def step(self, rng_key, params: Params, timestep: TimeStep) -> ActorStep:
        with self._gpu_lock:
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


class Actor:
    """Manages the state of a single agent/environment interaction loop."""

    def __init__(
        self,
        agent: Agent,
        env: SinglePlayerSyncEnvironment,
        unroll_length: int,
        params_for_actor: Callable[[], tuple[int, Params]],
        queue: queue.Queue | None = None,
        rng_seed: int = 42,
        logger=None,
    ):
        self._agent = agent
        self._env = env
        self._unroll_length = unroll_length
        self._queue = queue
        self._params_for_actor = params_for_actor
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
            if not timestep.env.valid.item():
                break
            unprocessed_timestep = self._env.step(actorstep.action.item())

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


def run_actor(actor: Actor, stop_signal: list[bool]):
    """Runs an actor to produce num_trajectories trajectories."""
    while not stop_signal[0]:
        frame_count, params = actor.pull_params()
        print(frame_count)
        actor.unroll_and_push(frame_count, params)


def host_to_device_worker(
    batch_size: int,
    trajectory_queue: queue.Queue[Transition],
    batch_queue: queue.Queue,
    stop_signal: list[bool],
):
    """Elementary data pipeline."""
    batch = []
    while not stop_signal[0]:
        # Try to get a batch. Skip the iteration if we couldn't.
        try:
            for _ in range(len(batch), batch_size):
                # As long as possible while keeping learner_test time reasonable.
                batch.append(trajectory_queue.get(timeout=10))
        except queue.Empty:
            continue

        assert len(batch) == batch_size
        # Prepare for consumption, then put batch onto device.
        stacked_batch: Transition = jax.tree.map(
            lambda *xs: np.stack(xs, axis=1), *batch
        )

        resolution = 64
        num_valid = stacked_batch.timestep.env.valid.sum(0).max().item() + 1
        num_valid = int(np.ceil(num_valid / resolution) * resolution)

        stacked_batch = Transition(
            timestep=TimeStep(
                env=jax.tree.map(lambda x: x[:num_valid], stacked_batch.timestep.env),
                history=clip_history(
                    stacked_batch.timestep.history, resolution=resolution
                ),
            ),
            actorstep=jax.tree.map(lambda x: x[:num_valid], stacked_batch.actorstep),
        )

        batch_queue.put(stacked_batch)

        # Clean out the built-up batch.
        batch = []


def evaluate(trajectory_queue: queue.Queue[Transition], stop_signal: list[bool]):
    while not stop_signal[0]:
        try:
            eval_trajectory = trajectory_queue.get(timeout=10)
        except queue.Empty:
            continue
        else:
            print(eval_trajectory.timestep.env.valid.sum())


def main():
    # A thunk that builds a new environment.
    # Substitute your environment here!

    network = get_dummy_model()
    ts = get_ex_step()
    params = network.init(jax.random.PRNGKey(42), ts)

    apply_fn = jax.vmap(network.apply, in_axes=(None, 1), out_axes=1)

    lock = threading.Lock()
    agent = Agent(apply_fn, lock)

    # Construct the actors on different threads.
    # stop_signal in a list so the reference is shared.
    actor_threads: list[threading.Thread] = []
    stop_signal = [False]
    NUM_GAMES = 32
    UNROLL_LENGTH = 192
    batch_size = 4
    num_eval_actors = 4

    eval_queue: queue.Queue[Transition] = queue.Queue(maxsize=num_eval_actors)
    trajectory_queue: queue.Queue[Transition] = queue.Queue(maxsize=batch_size)
    batch_queue: queue.Queue[Transition] = queue.Queue(maxsize=1)

    steps = 0

    def get_params():
        return steps, params

    player_id = 0
    for game_id in range(NUM_GAMES):
        for _ in range(2):
            actor = Actor(
                agent,
                SinglePlayerSyncEnvironment(player_id, game_id),
                UNROLL_LENGTH,
                trajectory_queue,
                get_params,
                rng_seed=int(str(game_id) + str(player_id)),
                logger=logger,
            )
            args = (actor, stop_signal)
            actor_threads.append(threading.Thread(target=run_actor, args=args))
            player_id += 1

    for eval_id in range(num_eval_actors):
        game_id = player_id = 10_000 + eval_id
        actor = Actor(
            agent,
            SinglePlayerSyncEnvironment(game_id, player_id),
            UNROLL_LENGTH,
            eval_queue,
            get_params,
            rng_seed=int(str(game_id) + str(player_id)),
            logger=logger,
        )
        args = (actor, stop_signal)
        actor_threads.append(threading.Thread(target=run_actor, args=args))

    # Start the actors and learner.
    for t in actor_threads:
        t.start()

    transfer_thread = threading.Thread(
        target=host_to_device_worker,
        args=(batch_size, trajectory_queue, batch_queue, stop_signal),
    )
    transfer_thread.start()

    eval_thread = threading.Thread(
        target=evaluate,
        args=(eval_queue, stop_signal),
    )
    eval_thread.start()

    for step_index in range(10000):
        batch = batch_queue.get()
        valid_count = batch.timestep.env.valid.sum()
        steps += valid_count

    # Stop.
    stop_signal[0] = True
    for t in actor_threads:
        t.join()


if __name__ == "__main__":
    main()
