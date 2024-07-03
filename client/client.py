import asyncio
import random
import struct
from typing import Sequence
import websockets
import uvloop

import numpy as np
import jax.numpy as jnp

from tqdm import tqdm

from protos.state_pb2 import State
from protos.action_pb2 import Action

uvloop.install()
pbar = tqdm()

SOCKET_PATH = "/tmp/pokemon.sock"


class Environment:
    env_idx: int
    state: State
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

    async def _read(self) -> State:
        msg_size_bytes = await self.reader.readexactly(4)
        remaining_msg_bytes = np.frombuffer(msg_size_bytes, dtype=np.int32).item()

        buffer = b""
        while remaining_msg_bytes > 0:
            buffer += await self.reader.readexactly(remaining_msg_bytes)
            remaining_msg_bytes -= len(buffer)

        self.state = State.FromString(buffer)
        pbar.update(1)

        return self.state

    async def step(self, action: Action):
        if self.state.info.done:
            return self.state

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


def main():
    num_envs = 12
    env = ParallelEnvironment(num_envs)

    while True:
        states = env.reset()

        dones = np.zeros(num_envs, dtype=bool)
        done = np.all(dones)

        trajectory = []

        while not done:
            actions = []
            for env_idx, state in enumerate(states):
                actions.append(
                    Action(
                        gameId=env_idx,
                        playerIndex=state.info.playerIndex,
                        index=random.choice(
                            [
                                index
                                for index, value in enumerate(
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
                                    ]
                                )
                                if value
                            ]
                        ),
                    )
                )
                dones[state.info.gameId] = state.info.done

            done = np.all(dones)
            if done:
                break

            states = env.step(actions)
            trajectory.append(states)

    env.close()


if __name__ == "__main__":
    main()
