import asyncio
import struct
import chex

import numpy as np

from google.protobuf.descriptor import FieldDescriptor

from typing import Any, Sequence, Tuple


from rlenv.data import (
    NUM_BOOSTS_FIELDS,
    NUM_POKEMON_FIELDS,
    NUM_PSEUDOWEATHER_FIELDS,
    NUM_SIDE_CONDITION_FIELDS,
    NUM_VOLATILE_STATUS_FIELDS,
)


from rlenv.protos.history_pb2 import Boost, Sidecondition, Volatilestatus
from rlenv.protos.state_pb2 import State
from rlenv.protos.action_pb2 import Action


@chex.dataclass(frozen=True)
class EnvStep:
    valid: chex.Array = ()
    turn: chex.Array = ()
    turn_context: chex.Array = ()
    active_entities: chex.Array = ()
    side_entities: chex.Array = ()
    boosts: chex.Array = ()
    side_conditions: chex.Array = ()
    volatile_status: chex.Array = ()
    additional_features: chex.Array = ()
    terrain: chex.Array = ()
    pseudoweather: chex.Array = ()
    weather: chex.Array = ()
    actions: chex.Array = ()
    legal: chex.Array = ()
    game_id: chex.Array = ()
    player_id: chex.Array = ()
    rewards: chex.Array = ()


@chex.dataclass(frozen=True)
class ActorStep:
    action_oh: chex.Array = ()
    policy: chex.Array = ()
    rewards: chex.Array = ()


@chex.dataclass(frozen=True)
class TimeStep:
    env: EnvStep = EnvStep()
    actor: ActorStep = ActorStep()


def set_tensor_from_fields(
    arr: np.ndarray,
    proto_list: Sequence[Tuple[FieldDescriptor, Any]],
    pre_idx: Sequence[int],
) -> None:
    for field, value in proto_list:
        arr[*pre_idx, field.index] = value


def set_tensor_from_repeated(
    arr: np.ndarray,
    proto_list: Sequence[Boost] | Sequence[Volatilestatus] | Sequence[Sidecondition],
    pre_idx: Sequence[int],
) -> None:
    for item in proto_list:
        arr[*pre_idx, item.index] = item.value


def get_history(state: State, num_history: int = 8):
    history = state.history
    history_active_entities = np.zeros(
        (num_history, 2, NUM_POKEMON_FIELDS), dtype=np.int32
    )
    history_side_conditions = np.zeros(
        (num_history, 2, NUM_SIDE_CONDITION_FIELDS), dtype=np.int32
    )
    history_volatile_status = np.zeros(
        (num_history, 2, NUM_VOLATILE_STATUS_FIELDS), dtype=np.int32
    )
    history_boosts = np.zeros((num_history, 2, NUM_BOOSTS_FIELDS), dtype=np.int32)

    history_weather = np.zeros((num_history,), dtype=np.int32)
    history_pseudoweather = np.zeros(
        (num_history, NUM_PSEUDOWEATHER_FIELDS), dtype=np.int32
    )

    for step_idx, step in enumerate(history):
        history_weather[step_idx] = step.weather

        set_tensor_from_fields(
            history_pseudoweather, step.pseudoweather.ListFields(), (step_idx,)
        )

        for side_idx, side in enumerate([step.p1, step.p2]):
            set_tensor_from_fields(
                history_active_entities, side.active.ListFields(), (step_idx, side_idx)
            )
            set_tensor_from_repeated(
                history_volatile_status, side.volatileStatus, (step_idx, side_idx)
            )
            set_tensor_from_repeated(
                history_side_conditions, side.sideConditions, (step_idx, side_idx)
            )
            set_tensor_from_repeated(history_boosts, side.boosts, (step_idx, side_idx))

    return (
        history_active_entities,
        history_side_conditions,
        history_volatile_status,
        history_boosts,
        history_weather,
        history_pseudoweather,
    )


def process_state(state: State) -> EnvStep:
    (
        active_entities,
        side_conditions,
        volatile_status,
        boosts,
        weather,
        pseudoweather,
    ) = get_history(state)
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
    )


SOCKET_PATH = "/tmp/pokemon.sock"


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
        remaining_msg_bytes = np.frombuffer(msg_size_bytes, dtype=np.int32).item()

        buffer = b""
        while remaining_msg_bytes > 0:
            buffer += await self.reader.readexactly(remaining_msg_bytes)
            remaining_msg_bytes -= len(buffer)

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
