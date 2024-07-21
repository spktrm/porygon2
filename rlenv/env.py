import struct
import asyncio
import uvloop
import numpy as np

from typing import Sequence

from rlenv.data import EX_STATE, SocketPath
from rlenv.interfaces import EnvStep
from rlenv.protos.state_pb2 import State
from rlenv.protos.action_pb2 import Action
from rlenv.utils import padnstack

uvloop.install()


def get_history(state: State):
    history = state.history
    history_length = history.length
    moveset = np.frombuffer(state.moveset, dtype=np.int16).reshape((4, -1))

    team = np.frombuffer(state.team, dtype=np.int16).reshape((6, -1))
    my_public = np.frombuffer(state.myPublic, dtype=np.int16).reshape((6, -1))
    opp_public = np.frombuffer(state.oppPublic, dtype=np.int16).reshape((6, -1))
    side_entities = np.stack((team, my_public, opp_public))

    active_entities = np.frombuffer(history.active, dtype=np.int16).reshape(
        (history_length, 2, -1)
    )
    side_conditions = np.frombuffer(history.sideConditions, dtype=np.uint8).reshape(
        (history_length, 2, -1)
    )
    volatile_status = np.frombuffer(history.volatileStatus, dtype=np.uint8).reshape(
        (history_length, 2, -1)
    )
    boosts = np.frombuffer(history.boosts, dtype=np.int8).reshape(
        (history_length, 2, -1)
    )
    hyphen_args = np.frombuffer(history.hyphenArgs, dtype=np.uint8).reshape(
        (history_length, 2, -1)
    )
    weather = np.frombuffer(history.weather, dtype=np.uint8).reshape(
        (history_length, -1)
    )
    pseudoweather = np.frombuffer(history.pseudoweather, dtype=np.uint8).reshape(
        (history_length, -1)
    )
    turn_context = np.frombuffer(history.turnContext, dtype=np.int16).reshape(
        (history_length, -1)
    )
    return (
        moveset,
        side_entities,
        padnstack(active_entities),
        padnstack(side_conditions),
        padnstack(volatile_status),
        padnstack(boosts),
        padnstack(weather),
        padnstack(pseudoweather),
        padnstack(hyphen_args),
        padnstack(turn_context, np.zeros_like),
    )


def process_state(state: State) -> EnvStep:
    (
        moveset,
        side_entities,
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
        valid=~np.array(state.info.done, dtype=bool),
        player_id=np.array(state.info.playerIndex, dtype=np.int32),
        game_id=np.array(state.info.gameId, dtype=np.int32),
        turn=np.array(state.info.turn, dtype=np.int32),
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
        side_entities=side_entities,
        moveset=moveset,
    )


def get_ex_step() -> EnvStep:
    return process_state(EX_STATE)


class Environment:
    env_idx: int
    state: State
    stage: int
    env_step: EnvStep
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

    @classmethod
    async def init(cls, path: SocketPath, env_idx: int):
        reader, writer = await asyncio.open_unix_connection(path)
        self = cls()
        self.stage = 0
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
        self.env_step = self.get_env_step(self.state, self.stage)
        return self.env_step

    @staticmethod
    def get_env_step(state: State, stage: int):
        env_step = process_state(state)
        if stage == 0:
            env_step.legal[4] = env_step.legal[:4].any()
            env_step.legal[:4] = 0
        else:
            env_step.legal[4:] = 0
        return env_step

    async def _check_for_single_action(self) -> EnvStep:
        legal = self.env_step.legal
        if legal.sum() == 1:
            return await self.step(legal.argmax())
        else:
            return self.env_step

    def _get_acton_bytes(self, action_index: int) -> bytes:
        action = Action()
        action.key = self.state.key
        action.index = action_index

        # Serialize the action
        return action.SerializeToString()

    async def step(self, action: int) -> EnvStep:
        if self.state.info.done:
            return self.env_step

        choosing_switch = action > 4
        choosing_move = action < 4

        if choosing_switch or choosing_move:
            self.stage = 0
            serialized_action = self._get_acton_bytes(action)
            # Get the size of the serialized action
            action_size = len(serialized_action)

            # Create a byte buffer for the size (4 bytes for a 32-bit integer)
            size_buffer = struct.pack(">I", action_size)

            # Write the size buffer followed by the serialized action
            self.writer.write(size_buffer)
            self.writer.write(serialized_action)
            await self.writer.drain()
            await self._read()
            return await self._check_for_single_action()

        else:
            self.stage = 1
            self.env_step = self.get_env_step(self.state, self.stage)
            return await self._check_for_single_action()

    async def reset(self):
        return await self._read()

    async def close(self):
        self.writer.close()
        await self.writer.wait_closed()
        if not self.reader.at_eof():
            self.reader.feed_eof()


class ParallelEnvironment:
    def __init__(self, num_envs: int, path: SocketPath):
        try:
            self._loop = asyncio.get_event_loop()
        except:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        tasks = [Environment.init(path, env_idx) for env_idx in range(num_envs)]
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
