import asyncio
import functools
from abc import ABC, abstractmethod
from typing import Callable, List, Sequence

import chex
import flax.linen as nn
import jax
import numpy as np
import uvloop
import websockets
from tqdm import tqdm

from ml.arch.model import get_dummy_model
from ml.config import FineTuning
from ml.utils import Params
from rlenv.protos.servicev2_pb2 import ConnectMessage
from rlenvv2.data import EX_STATE, NUM_EDGE_FIELDS, NUM_MOVE_FIELDS
from rlenvv2.interfaces import ActorStep, EnvStep, ModelOutput, TimeStep
from rlenvv2.protos.features_pb2 import FeatureEdge, FeatureEntity
from rlenvv2.protos.servicev2_pb2 import (
    Action,
    ClientMessage,
    ResetMessage,
    ServerMessage,
    StepMessage,
)
from rlenvv2.protos.state_pb2 import State
from rlenvv2.utils import padnstack, stack_steps

# Define the server URI
SERVER_URI = "ws://localhost:8080"

uvloop.install()


def get_history(state: State, player_index: int):
    history = state.history
    history_length = history.length
    moveset = np.frombuffer(state.moveset, dtype=np.int16).reshape(
        (2, -1, NUM_MOVE_FIELDS)
    )
    team = np.frombuffer(bytearray(state.team), dtype=np.int16).reshape((2, 6, -1))
    team[..., FeatureEntity.ENTITY_SIDE] ^= player_index
    team.flags.writeable = False
    history_edges = np.frombuffer(bytearray(history.edges), dtype=np.int16).reshape(
        (history_length, -1, NUM_EDGE_FIELDS)
    )
    edge_affecting_side = history_edges[..., FeatureEdge.EDGE_AFFECTING_SIDE]
    history_edges[..., FeatureEdge.EDGE_AFFECTING_SIDE] = np.where(
        edge_affecting_side < 2,
        edge_affecting_side ^ player_index,
        edge_affecting_side,
    )
    history_edges.flags.writeable = False
    history_nodes = np.frombuffer(bytearray(history.nodes), dtype=np.int16).reshape(
        (history_length, 12, -1)
    )
    history_nodes[..., FeatureEntity.ENTITY_SIDE] ^= player_index
    history_nodes.flags.writeable = False
    history_side_conditions = np.frombuffer(
        bytearray(history.sideConditions), dtype=np.uint8
    ).reshape((history_length, 2, -1))
    history_field = np.frombuffer(bytearray(history.field), dtype=np.uint8).reshape(
        (history_length, -1)
    )
    return (
        moveset,
        team,
        padnstack(history_edges),
        padnstack(history_nodes),
        padnstack(history_side_conditions),
        padnstack(history_field),
    )


def get_legal_mask(state: State):
    buffer = np.frombuffer(state.legalActions, dtype=np.uint8)
    mask = np.unpackbits(buffer, axis=-1)
    return mask[:10].astype(bool)


def process_state(state: State, is_eval: bool = False, done: bool = False) -> EnvStep:
    player_index = state.info.playerIndex
    (
        moveset,
        team,
        history_edges,
        history_nodes,
        history_side_conditions,
        history_field,
    ) = get_history(state, player_index)
    return EnvStep(
        ts=state.info.ts,
        draw_ratio=(
            1 - (1 - state.info.turn / 100) ** 2 if is_eval else state.info.drawRatio
        ),
        valid=~np.array(done, dtype=bool),
        player_id=np.array(player_index, dtype=np.int32),
        game_id=np.array(state.info.gameId, dtype=np.int32),
        turn=np.array(state.info.turn, dtype=np.int32),
        heuristic_action=np.array(state.info.heuristicAction, dtype=np.int32),
        heuristic_dist=np.frombuffer(state.info.heuristicDist, dtype=np.float32),
        prev_action=np.array(state.info.lastAction, dtype=np.int32),
        prev_move=np.array(state.info.lastMove, dtype=np.int32),
        win_rewards=(
            np.array([state.info.winReward, -state.info.winReward], dtype=np.float32)
            if done
            else np.zeros(2)
        ),
        hp_rewards=np.array(
            [state.info.hpReward, -state.info.hpReward], dtype=np.float32
        ),
        fainted_rewards=np.array(
            [state.info.faintedReward, -state.info.faintedReward], dtype=np.float32
        ),
        switch_rewards=np.array(
            [state.info.switchReward, -state.info.switchReward], dtype=np.float32
        ),
        longevity_rewards=np.array(
            [state.info.longevityReward, -state.info.longevityReward], dtype=np.float32
        ),
        legal=get_legal_mask(state),
        team=team,
        moveset=moveset,
        history_edges=history_edges,
        history_nodes=history_nodes,
        history_side_conditions=history_side_conditions,
        history_field=history_field,
    )


def get_ex_step() -> EnvStep:
    return process_state(EX_STATE)


class SinglePlayerEnvironment:
    websocket: websockets.WebSocketClientProtocol

    def __init__(self, player_id: int, game_id: int):
        self.player_id = player_id
        self.game_id = game_id
        self.state = None
        self.mess = None
        self.queue = asyncio.Queue()

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

    async def _reset(self):
        reset_message = ClientMessage(
            player_id=self.player_id, game_id=self.game_id, reset=ResetMessage()
        )
        await self.websocket.send(reset_message.SerializeToString())
        return await self._recv()

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
        self.current_step: EnvStep = None
        self.current_player = None
        self.done_count = 0

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
        return self.done_count >= 2

    async def _reset(self):
        """Reset both players, enqueue their initial states, and return the first state to act on."""
        # Reset both players and add their states to the queue
        while not self.state_queue.empty():
            await self.state_queue.get()
        self.done_count = 0

        states = await asyncio.gather(*[player._reset() for player in self.players])

        for player, state in zip(self.players, states):
            await self.state_queue.put((player, state))

        # Retrieve the first state to act on and set the corresponding player
        self.current_player, self.current_state = await self.state_queue.get()
        self.current_step = process_state(self.current_state, done=self.is_done())
        return self.current_step

    async def _step(self, action: int):
        """Process the action for the current player asynchronously."""
        # Start the action processing in the background without waiting for it to complete
        asyncio.create_task(self._perform_action(self.current_player, action))

        # Retrieve the next player and state from the queue, waiting if necessary
        self.current_player, self.current_state = await self.state_queue.get()
        self.current_step = process_state(self.current_state, done=self.is_done())
        return self.current_step

    async def _perform_action(self, player: SinglePlayerEnvironment, action: int):
        """Helper method to send the action to the player and enqueue the resulting state."""
        # Perform the step and add the resulting state along with the player back into the queue
        state = await player._step(action)
        self.done_count += state.info.done
        if not state.info.done:
            await self.state_queue.put((player, state))
        if self.done_count >= 2:
            await self.state_queue.put((player, state))


class BatchEnvironment(ABC):
    def __init__(self, num_envs: int):
        self.num_envs = num_envs

    @abstractmethod
    def is_done(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: List[int]) -> List[EnvStep]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> List[EnvStep]:
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

    def is_done(self):
        return np.array([env.is_done() for env in self.envs]).all()

    def step(self, actions: List[int]):
        return self.loop.run_until_complete(
            asyncio.gather(
                *[env._step(action) for env, action in zip(self.envs, actions)]
            ),
        )

    def reset(self):
        return self.loop.run_until_complete(
            asyncio.gather(*[env._reset() for env in self.envs])
        )


class BatchSinglePlayerEnvironment(BatchEnvironment):
    def __init__(self, num_envs: int):
        super().__init__(num_envs)
        try:
            self.loop = asyncio.get_event_loop()
        except:
            self.loop = asyncio.new_event_loop()

        self.envs: List[SinglePlayerEnvironment] = []
        self.player_id_count = 0
        for game_id in range(num_envs):
            env = self.loop.run_until_complete(
                SinglePlayerEnvironment.create(10_000 + game_id, 10_000 + game_id)
            )
            self.envs.append(env)
            self.player_id_count += 1

    def is_done(self):
        return np.array([env.is_done() for env in self.envs]).all()

    def step(self, actions: List[int]):
        states = self.loop.run_until_complete(
            asyncio.gather(
                *[env._step(action) for env, action in zip(self.envs, actions)]
            ),
        )
        return [
            process_state(state, done=env.is_done())
            for state, env in zip(states, self.envs)
        ]

    def reset(self):
        states = self.loop.run_until_complete(
            asyncio.gather(*[env._reset() for env in self.envs])
        )
        return [
            process_state(state, done=env.is_done())
            for state, env in zip(states, self.envs)
        ]


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

    def _batch_of_states_apply_action(self, actions: chex.Array) -> Sequence[EnvStep]:
        """Apply a batch of `actions` to a parallel list of `states`."""
        return self.game.step(list(actions))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _network_jit_apply(self, params: Params, env_steps: EnvStep) -> chex.Array:
        rollout: Callable[[Params, EnvStep], ModelOutput] = jax.vmap(
            self.network.apply, (None, 0), 0
        )
        output = rollout(params, env_steps)
        return output.pi

    def actor_step(self, params: Params, env_step: EnvStep):
        pi = self._network_jit_apply(params, env_step)
        action = np.apply_along_axis(
            lambda x: np.random.choice(range(pi.shape[-1]), p=x), axis=-1, arr=pi
        )
        return action, ActorStep(
            action=action, policy=pi, win_rewards=(), hp_rewards=(), fainted_rewards=()
        )

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
                    longevity_rewards=env_step.longevity_rewards,
                ),
            )
            timesteps.append(timestep)

            if self.game.is_done() and state_index % resolution == 0:
                break

            state_index += 1

        # Concatenate all the timesteps together to form a single rollout [T, B, ..]
        batch: TimeStep = stack_steps(timesteps)
        # if not ((batch.env.ts[1:] >= batch.env.ts[:-1])).all():
        #     raise ValueError

        if not ((batch.env.turn[1:] >= batch.env.turn[:-1])).all():
            raise ValueError

        return batch


def main():
    batch_progress = tqdm(desc="Batch: ")
    game_progress = tqdm(desc="Games: ")
    state_progress = tqdm(desc="States: ")

    num_envs = 3
    network = get_dummy_model()
    env = BatchCollectorV2(network, num_envs, BatchSinglePlayerEnvironment)

    ex = get_ex_step()
    params = network.init(jax.random.PRNGKey(42), ex)

    avg_reward = np.zeros(num_envs)
    tau = 1e-2

    while True:
        batch = env.collect_batch_trajectory(params)
        win_rewards = np.sign(
            (batch.actor.win_rewards[..., 0] * batch.env.valid).sum(0)
        )
        avg_reward = avg_reward * (1 - tau) + win_rewards.astype(float) * tau
        state_progress.update(batch.env.valid.sum())
        game_progress.update(num_envs)
        batch_progress.update(1)
        batch_progress.set_description(np.array2string(avg_reward))


if __name__ == "__main__":
    main()
