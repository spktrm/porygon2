import asyncio
import functools
import pickle
from abc import ABC, abstractmethod
from typing import Callable, List, Sequence, Tuple

import chex
import flax.linen as nn
import jax
import numpy as np
import uvloop
import websockets
from tqdm import tqdm

from ml.arch.model import get_model
from ml.config import FineTuning
from ml.learners.func import collect_batch_telemetry_data
from ml.utils import Params
from rlenv.env import get_ex_step, process_state
from rlenv.interfaces import ActorStep, EnvStep, HistoryStep, ModelOutput, TimeStep
from rlenv.protos.servicev2_pb2 import (
    Action,
    ClientMessage,
    ConnectMessage,
    ResetMessage,
    ServerMessage,
    StepMessage,
)
from rlenv.protos.state_pb2 import State
from rlenv.utils import add_batch, concatenate_steps, stack_steps, stack_trajectories

# Define the server URI
SERVER_URI = "ws://localhost:8080"

uvloop.install()


class SinglePlayerEnvironment:
    websocket: websockets.WebSocketClientProtocol

    def __init__(self, player_id: int, game_id: int):
        self.player_id = player_id
        self.game_id = game_id
        self.state = None
        self.history = None
        self.mess = None
        self.queue = asyncio.Queue()

    def get_history(self):
        return self.history

    def is_done(self):
        return self.state.info.done

    @classmethod
    async def create(cls, player_id: int, game_id: int):
        """Async factory method for initialization."""
        self = cls(player_id, game_id)
        await self._init()
        return self

    async def _init(self):
        self.websocket = await websockets.connect(SERVER_URI)
        connect_message = ClientMessage(
            player_id=self.player_id, game_id=self.game_id, connect=ConnectMessage()
        )
        await self.websocket.send(connect_message.SerializeToString())
        asyncio.create_task(self.read_continuously())

    async def read_continuously(self):
        while True:
            try:
                incoming_message = await self.websocket.recv()
                await self.queue.put(incoming_message)
            except websockets.ConnectionClosed:
                print(f"Connection closed for player {self.player_id}")
                break

    def _process_state(self):
        ex, hx = process_state(self.state)
        self.history = hx
        return ex, hx

    async def reset(self):
        await self._reset()
        return self._process_state()

    async def _reset(self):
        reset_message = ClientMessage(
            player_id=self.player_id, game_id=self.game_id, reset=ResetMessage()
        )
        await self.websocket.send(reset_message.SerializeToString())
        return await self._recv()

    async def step(self, action: int):
        await self._step(action)
        return self._process_state()

    async def _step(self, action: int):
        if self.state and self.state.info.done:
            return self.state
        step_message = ClientMessage(
            player_id=self.player_id,
            game_id=self.game_id,
            step=StepMessage(
                action=Action(rqid=self.mess.game_state.rqid, value=action)
            ),
        )
        await self.websocket.send(step_message.SerializeToString())
        return await self._recv()

    async def _recv(self):
        server_message_data = await self.queue.get()
        server_message = ServerMessage.FromString(server_message_data)
        self.mess = server_message
        self.state = State.FromString(server_message.game_state.state)
        return self.state


class TwoPlayerEnvironment:
    def __init__(self, game_id: int, player_ids: List[int]):
        self.game_id = game_id
        self.player_ids = player_ids
        self.players = []
        self.state_queue = asyncio.Queue()
        self.current_state: State = None
        self.current_step: Tuple[EnvStep, HistoryStep] = None
        self.current_player = None
        self.dones = np.zeros(2)

        self.hxs = {}

    def get_history(self):
        history = concatenate_steps([self.hxs[0], self.hxs[1]])
        return history

    async def initialize_players(self):
        """Initialize both players asynchronously."""
        self.players = await asyncio.gather(
            *[
                SinglePlayerEnvironment.create(pid, self.game_id)
                for pid in self.player_ids
            ]
        )
        self.current_player = self.players[0]  # Start with the first player

    def is_done(self):
        return self.dones.sum() == 2

    async def _reset(self):
        """Reset both players, enqueue their initial states, and return the first state to act on."""
        # Reset both players and add their states to the queue
        while not self.state_queue.empty():
            await self.state_queue.get()
        self.dones = np.zeros(2)

        states = await asyncio.gather(*[player._reset() for player in self.players])

        for player, state in zip(self.players, states):
            await self.state_queue.put((player, state))

        # Retrieve the first state to act on and set the corresponding player
        self.current_player, self.current_state = await self.state_queue.get()
        self.current_step = self._process_state()
        return self.current_step

    async def _step(self, action: int):
        """Process the action for the current player asynchronously."""
        # Start the action processing in the background without waiting for it to complete
        asyncio.create_task(self._perform_action(self.current_player, action))

        # Retrieve the next player and state from the queue, waiting if necessary
        self.current_player, self.current_state = await self.state_queue.get()
        self.current_step = self._process_state()
        return self.current_step

    def _process_state(self):
        ex, hx = process_state(self.current_state)
        self.hxs[int(self.current_state.info.playerIndex)] = hx
        return ex, hx

    async def _perform_action(self, player: SinglePlayerEnvironment, action: int):
        """Helper method to send the action to the player and enqueue the resulting state."""
        # Perform the step and add the resulting state along with the player back into the queue
        state = await player._step(action)
        if not self.is_done():
            self.dones[int(state.info.playerIndex)] = state.info.done
        if not state.info.done:
            await self.state_queue.put((player, state))
        if self.is_done():
            await self.state_queue.put((player, state))


class BatchEnvironment(ABC):
    def __init__(self, num_envs: int):
        self.num_envs = num_envs

    @abstractmethod
    def is_done(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: List[int]) -> Tuple[List[EnvStep], List[HistoryStep]]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Tuple[List[EnvStep], List[HistoryStep]]:
        raise NotImplementedError


class BatchTwoPlayerEnvironment(BatchEnvironment):
    def __init__(self, num_envs: int):
        super().__init__(num_envs)
        try:
            self.loop = asyncio.get_event_loop()
        except:
            self.loop = asyncio.new_event_loop()

        self.envs: List[TwoPlayerEnvironment] = []
        self.player_id_count = 0
        for game_id in range(num_envs):
            env = TwoPlayerEnvironment(
                game_id, [self.player_id_count, self.player_id_count + 1]
            )
            self.loop.run_until_complete(env.initialize_players())
            self.envs.append(env)
            self.player_id_count += 2

    def get_history(self):
        history_steps = [env.get_history() for env in self.envs]
        stacked_history = stack_steps(history_steps, axis=1)
        return stacked_history

    def is_done(self):
        return np.array([env.is_done() for env in self.envs]).all()

    def step(self, actions: List[int]):
        out = self.loop.run_until_complete(
            asyncio.gather(
                *[env._step(action) for env, action in zip(self.envs, actions)]
            ),
        )
        return list(zip(*out))

    def reset(self):
        out = self.loop.run_until_complete(
            asyncio.gather(*[env._reset() for env in self.envs])
        )
        return list(zip(*out))


class BatchSinglePlayerEnvironment(BatchEnvironment):
    def __init__(self, num_envs: int, is_eval: bool = True):
        super().__init__(num_envs)
        try:
            self.loop = asyncio.get_event_loop()
        except:
            self.loop = asyncio.new_event_loop()

        self.envs: List[SinglePlayerEnvironment] = []
        self.player_id_count = 0
        if is_eval:
            for game_id in range(num_envs):
                env = self.loop.run_until_complete(
                    SinglePlayerEnvironment.create(10_000 + game_id, 10_000 + game_id)
                )
                self.envs.append(env)
                self.player_id_count += 1
        else:
            for game_id in range(num_envs):
                for _ in range(2):
                    env = self.loop.run_until_complete(
                        SinglePlayerEnvironment.create(self.player_id_count, game_id)
                    )
                    self.envs.append(env)
                    self.player_id_count = self.player_id_count + 1

    def get_history(self):
        history_steps = [env.get_history() for env in self.envs]
        stacked_history = stack_steps(history_steps, axis=1)
        return stacked_history

    def is_done(self):
        return np.array([env.is_done() for env in self.envs]).all()

    def step(self, actions: List[int]):
        states = self.loop.run_until_complete(
            asyncio.gather(
                *[env._step(action) for env, action in zip(self.envs, actions)]
            ),
        )
        return zip(*[process_state(state) for state in states])

    def reset(self):
        states = self.loop.run_until_complete(
            asyncio.gather(*[env._reset() for env in self.envs])
        )
        return zip(*[process_state(state) for state in states])


class BatchCollectorV2:
    def __init__(
        self,
        network: nn.Module,
        batch_size: int,
        env_constructor: BatchEnvironment = BatchTwoPlayerEnvironment,
    ):
        self.game: BatchEnvironment = env_constructor(batch_size)
        self.network = network
        self.finetuning = FineTuning()
        self.batch_size = batch_size

    def _batch_of_states_apply_action(self, actions: chex.Array) -> Sequence[EnvStep]:
        """Apply a batch of `actions` to a parallel list of `states`."""
        return self.game.step(list(actions))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _network_jit_apply(
        self, params: Params, env_step: EnvStep, history_step: HistoryStep
    ) -> chex.Array:
        rollout: Callable[[Params, EnvStep], ModelOutput] = jax.vmap(
            self.network.apply, (None, 1, 1), 1
        )
        output = rollout(params, env_step, history_step)
        return output.pi

    def actor_step(self, params: Params, env_step: EnvStep, history_step: HistoryStep):
        pi = self._network_jit_apply(params, env_step, history_step)
        action = np.apply_along_axis(
            lambda x: np.random.choice(range(pi.shape[-1]), p=x), axis=-1, arr=pi
        )
        return action.squeeze(), ActorStep(action=action, policy=pi)

    def collect_batch_trajectory(
        self, params: Params, resolution: int = 32
    ) -> TimeStep:
        ex, hx = self.game.reset()
        timesteps = []

        env_step: EnvStep = stack_steps(ex, axis=1)
        history_step: HistoryStep = stack_steps(hx, axis=1)

        state_index = 0
        while True:
            prev_env_step = env_step

            # history_step = trim_history(history_step)
            a, actor_step = self.actor_step(params, env_step, history_step)

            ex, hx = self._batch_of_states_apply_action(a)

            env_step = stack_steps(ex, axis=1)
            history_step = stack_steps(hx, axis=1)

            timestep = TimeStep(
                env=prev_env_step,
                actor=ActorStep(
                    action=actor_step.action,
                    policy=actor_step.policy,
                    rewards=env_step.rewards,
                ),
            )
            timesteps.append(timestep)

            if self.game.is_done() and state_index % resolution == 0:
                break

            state_index += 1

        # Concatenate all the timesteps together to form a single rollout [T, B, ..]
        batch: TimeStep = concatenate_steps(timesteps, axis=0)
        batch = TimeStep(
            env=batch.env, actor=batch.actor, history=self.game.get_history()
        )

        if (batch.env.turn[1:] < batch.env.turn[:-1]).any():
            raise ValueError

        if (batch.env.seed_hash != batch.env.seed_hash[0]).any():
            raise ValueError

        if not np.all(np.array([env.is_done() for env in self.game.envs])):
            raise ValueError

        return batch


class SingleTrajectoryTrainingBatchCollector(BatchCollectorV2):
    def __init__(self, network: nn.Module, batch_size: int):
        super().__init__(network, batch_size, BatchTwoPlayerEnvironment)


class DoubleTrajectoryTrainingBatchCollector(BatchCollectorV2):
    game: BatchSinglePlayerEnvironment

    def __init__(self, network: nn.Module, batch_size: int):
        super().__init__(
            network,
            batch_size,
            functools.partial(BatchSinglePlayerEnvironment, is_eval=False),
        )

    async def collect_trajectory(
        self, params: Params, environment_index: int
    ) -> TimeStep:
        environment = self.game.envs[environment_index]

        env_step = await environment.reset()
        env_step = add_batch(env_step, axis=0)
        timesteps = []

        state_index = 0
        while True:
            prev_env_step = env_step
            a, actor_step = self.actor_step(params, env_step)

            env_step = await environment.step(a.item())
            env_step = add_batch(env_step, axis=0)
            timestep = TimeStep(
                env=prev_env_step,
                actor=ActorStep(
                    action=actor_step.action,
                    policy=actor_step.policy,
                    rewards=env_step.rewards,
                ),
            )
            timesteps.append(timestep)

            if environment.is_done():
                break

            state_index += 1

        # Concatenate all the timesteps together to form a single rollout [T, B, ..]
        trajectory: TimeStep = stack_steps(timesteps)

        return trajectory

    def collect_batch_trajectory(
        self, params: Params, resolution: int = 32
    ) -> TimeStep:
        tasks = [
            self.collect_trajectory(params, env_index)
            for env_index in range(2 * self.batch_size)
        ]
        trajectories = self.game.loop.run_until_complete(
            asyncio.gather(*tasks),
        )
        batch = stack_trajectories(trajectories)
        return batch


class EvalBatchCollector(BatchCollectorV2):
    def __init__(self, network: nn.Module, batch_size: int):
        super().__init__(network, batch_size, BatchSinglePlayerEnvironment)


def main():
    # batch_progress = tqdm(desc="Batch: ")
    # game_progress = tqdm(desc="Games: ")
    # state_progress = tqdm(desc="States: ")

    training_progress = tqdm(desc="training: ")
    evaluation_progress = tqdm(desc="evaluation: ")

    num_envs = 8
    network = get_model()
    training_env = SingleTrajectoryTrainingBatchCollector(network, num_envs)
    evaluation_env = EvalBatchCollector(network, 4)

    ex, hx = get_ex_step()
    params = network.init(jax.random.PRNGKey(42), ex, hx)

    np.zeros(num_envs)

    while True:
        batch = training_env.collect_batch_trajectory(params)
        with open("rlenv/ex_batch", "wb") as f:
            pickle.dump(batch, f)

        return

        collect_batch_telemetry_data(batch)

        training_progress.update(batch.env.valid.sum())
        batch = evaluation_env.collect_batch_trajectory(params)
        evaluation_progress.update(batch.env.valid.sum())

        if (
            np.sign(
                (
                    (batch.actor.rewards.fainted_rewards * batch.env.valid[..., None])
                ).sum(0)
            )
            != (batch.actor.rewards.win_rewards * batch.env.valid[..., None]).sum(0)
        ).all():
            raise ValueError

        # win_rewards = np.sign(
        #     (batch.actor.win_rewards[..., 0] * batch.env.valid).sum(0)
        # )
        # avg_reward = avg_reward * (1 - tau) + win_rewards.astype(float) * tau
        # state_progress.update(batch.env.valid.sum())
        # game_progress.update(num_envs)
        # batch_progress.update(1)
        # batch_progress.set_description(np.array2string(avg_reward))


if __name__ == "__main__":
    main()
