import struct
import asyncio
import uvloop
import numpy as np

from typing import Sequence

from rlenv.data import EX_STATE, SOCKET_PATH
from rlenv.interfaces import EnvStep
from rlenv.protos.state_pb2 import State
from rlenv.protos.action_pb2 import Action
from rlenv.utils import padnstack

uvloop.install()


def get_history(state: State):
    history = state.history
    history_length = history.length
    moveset = np.frombuffer(state.moveset, dtype=np.int32).reshape((4, -1))
    team = np.frombuffer(state.team, dtype=np.int32).reshape((7, -1))
    history_active_entities = np.frombuffer(history.active, dtype=np.int32).reshape(
        (history_length, 2, -1)
    )
    history_side_conditions = np.frombuffer(
        history.sideConditions, dtype=np.uint8
    ).reshape((history_length, 2, -1))
    history_volatile_status = np.frombuffer(
        history.volatileStatus, dtype=np.uint8
    ).reshape((history_length, 2, -1))
    history_boosts = np.frombuffer(history.boosts, dtype=np.uint8).reshape(
        (history_length, 2, -1)
    )
    history_hyphen_args = np.frombuffer(history.hyphenArgs, dtype=np.uint8).reshape(
        (history_length, 2, -1)
    )
    history_weather = np.frombuffer(history.weather, dtype=np.uint8).reshape(
        (history_length, -1)
    )
    history_pseudoweather = np.frombuffer(
        history.pseudoweather, dtype=np.uint8
    ).reshape((history_length, -1))
    history_turn_context = np.frombuffer(history.turnContext, dtype=np.int32).reshape(
        (history_length, -1)
    )
    return (
        moveset,
        team,
        padnstack(history_active_entities),
        padnstack(history_side_conditions),
        padnstack(history_volatile_status),
        padnstack(history_boosts),
        padnstack(history_weather),
        padnstack(history_pseudoweather),
        padnstack(history_hyphen_args),
        padnstack(history_turn_context),
    )


def process_state(state: State) -> EnvStep:
    (
        moveset,
        team,
        active_entities,
        side_conditions,
        volatile_status,
        boosts,
        weather,
        pseudoweather,
        hyphen_args,
        turn_context,
    ) = get_history(state)
    # hyphen_args = np.unpackbits(hyphen_args, axis=-1).view(bool).astype(float)
    return EnvStep(
        valid=not state.info.done,
        player_id=state.info.playerIndex,
        game_id=state.info.gameId,
        turn=state.info.turn,
        rewards=np.array(
            [state.info.playerOneReward, state.info.playerTwoReward], dtype=np.float32
        ),
        legal=np.array(
            [
                state.legalActions.move1,
                state.legalActions.move2,
                state.legalActions.move3,
                state.legalActions.move4,
                state.legalActions.switch1,
                state.legalActions.switch2,
                state.legalActions.switch3,
                state.legalActions.switch4,
                state.legalActions.switch5,
                state.legalActions.switch6,
            ],
            dtype=bool,
        ),
        active_entities=active_entities,
        side_conditions=side_conditions,
        volatile_status=volatile_status,
        boosts=boosts,
        weather=weather,
        pseudoweather=pseudoweather,
        hyphen_args=hyphen_args,
        turn_context=turn_context,
        team=team,
        moveset=moveset,
    )


def get_ex_step() -> EnvStep:
    return process_state(EX_STATE)


class Environment:
    env_idx: int
    state: State
    env_step: EnvStep
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

    @classmethod
    async def init(cls, env_idx: int):
        reader, writer = await asyncio.open_unix_connection(SOCKET_PATH)
        self = cls()
        self.env_idx = env_idx
        self.reader = reader
        self.writer = writer
        return self

    async def _read(self) -> EnvStep:
        msg_size_bytes = await self.reader.readexactly(4)
        msg_size = np.frombuffer(msg_size_bytes, dtype=np.int32).item()

        # Read the message
        buffer = await self.reader.readexactly(msg_size)

        state = State.FromString(buffer)
        self.state = state
        self.env_step = process_state(state)

        return self.env_step

    async def step(self, action_index: int):
        if self.state.info.done:
            return self.env_step

        action = Action()
        action.gameId = self.state.info.gameId
        action.playerIndex = self.state.info.playerIndex
        action.index = action_index

        # Serialize the action
        serialized_action = action.SerializeToString()

        # Get the size of the serialized action
        action_size = len(serialized_action)

        # Create a byte buffer for the size (4 bytes for a 32-bit integer)
        size_buffer = struct.pack(">I", action_size)

        # Write the size buffer followed by the serialized action
        self.writer.write(size_buffer)
        self.writer.write(serialized_action)

        # Ensure the data is actually sent
        await self.writer.drain()

        # Read and return the state
        return await self._read()

    async def reset(self):
        return await self._read()

    async def close(self):
        self.writer.close()
        await self.writer.wait_closed()
        if not self.reader.at_eof():
            self.reader.feed_eof()


class ParallelEnvironment:
    def __init__(self, num_envs: int):
        try:
            self._loop = asyncio.get_event_loop()
        except:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        tasks = [Environment.init(env_idx) for env_idx in range(num_envs)]
        self._games = self._loop.run_until_complete(asyncio.gather(*tasks))

    def step(self, actions: Sequence[Action]):
        tasks = [game.step(action) for action, game in zip(actions, self._games)]
        return self._loop.run_until_complete(asyncio.gather(*tasks))

    def reset(self):
        tasks = [game.reset() for game in self._games]
        return self._loop.run_until_complete(asyncio.gather(*tasks))

    def close(self):
        tasks = [game.close() for game in self._games]
        return self._loop.run_until_complete(asyncio.gather(*tasks))
