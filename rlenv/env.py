import asyncio
import struct
from typing import Sequence

import numpy as np
import uvloop

from rlenv.data import EX_STATE, NUM_EDGE_FIELDS, NUM_MOVE_FIELDS, SocketPath
from rlenv.interfaces import EnvStep
from rlenv.protos.action_pb2 import Action
from rlenv.protos.features_pb2 import FeatureEdge, FeatureEntity
from rlenv.protos.state_pb2 import State
from rlenv.utils import padnstack

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


def process_state(state: State, is_eval: bool = False) -> EnvStep:
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
        valid=~np.array(state.info.done, dtype=bool),
        player_id=np.array(player_index, dtype=np.int32),
        game_id=np.array(state.info.gameId, dtype=np.int32),
        turn=np.array(state.info.turn, dtype=np.int32),
        heuristic_action=np.array(state.info.heuristicAction, dtype=np.int32),
        heuristic_dist=np.frombuffer(state.info.heuristicDist, dtype=np.float32),
        prev_action=np.array(state.info.lastAction, dtype=np.int32),
        prev_move=np.array(state.info.lastMove, dtype=np.int32),
        win_rewards=np.array(
            [state.info.winReward, -state.info.winReward], dtype=np.float32
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


class Environment:
    env_idx: int
    state: State
    env_step: EnvStep
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

    @classmethod
    async def init(cls, path: SocketPath, env_idx: int):
        reader, writer = await asyncio.open_unix_connection(path)
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
        self.env_step = process_state(self.state)
        return self.env_step

    async def _check_for_single_action(self) -> EnvStep:
        legal = self.env_step.legal
        if legal.sum() == 1:
            return await self.step(legal.argmax())
        else:
            return self.env_step

    def _get_action_bytes(self, action_index: int) -> bytes:
        action = Action()
        action.key = self.state.key
        action.index = action_index
        return action.SerializeToString()

    async def step(self, action_index: int) -> EnvStep:
        if self.state.info.done:
            return self.env_step

        serialized_action = self._get_action_bytes(action_index)
        await self._write_action(serialized_action)
        await self._read()
        return await self._check_for_single_action()

    async def _write_action(self, serialized_action: bytes):
        # Get the size of the serialized action
        action_size = len(serialized_action)
        # Create a byte buffer for the size (4 bytes for a 32-bit integer)
        size_buffer = struct.pack(">I", action_size)
        # Write the size buffer followed by the serialized action
        self.writer.write(size_buffer)
        self.writer.write(serialized_action)
        await self.writer.drain()

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
