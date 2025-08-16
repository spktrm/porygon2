import functools

import jax
import jax.numpy as jnp
from websockets.sync.client import connect

from rl.environment.data import PACKED_SETS
from rl.environment.interfaces import BuilderEnvOutput
from rl.environment.protos.service_pb2 import (
    Action,
    ClientRequest,
    EnvironmentResponse,
    ResetRequest,
    StepRequest,
)
from rl.environment.utils import process_state

SERVER_URI = "ws://localhost:8080"


class SinglePlayerSyncEnvironment:
    def __init__(self, username: str, generation: int = 3):

        self.username = username
        self.rqid = None
        self.last_state = None

        self.websocket = connect(
            SERVER_URI,
            additional_headers={"username": username},
        )
        self.generation = generation

    def _recv(self):
        server_message_data = self.websocket.recv()
        server_message = EnvironmentResponse.FromString(server_message_data)
        self.rqid = server_message.state.rqid
        self.last_state = process_state(server_message.state)
        return self.last_state

    def reset(self, team_indices: list[int]):
        self.rqid = None
        reset_message = ClientRequest(
            reset=ResetRequest(
                username=self.username,
                team_indices=team_indices,
                smogon_format=f"gen{self.generation}ou",
            )
        )
        self.websocket.send(reset_message.SerializeToString())
        return self._recv()

    def _is_done(self):
        if self.last_state is None:
            return False
        return self.last_state.env.done.item()

    def step(self, action: Action):
        if self._is_done():
            return self.last_state
        step_message = ClientRequest(
            step=StepRequest(action=action, username=self.username, rqid=self.rqid),
        )
        self.websocket.send(step_message.SerializeToString())
        return self._recv()


class TeamBuilderEnvironment:
    def __init__(self, generation: int, smogon_tier: str = "ou"):
        data = PACKED_SETS[f"gen{generation}"]

        self.generation = generation
        self.smogon_tier = smogon_tier

        self.start_mask = jnp.asarray(data[f"gen{generation}{smogon_tier}"])
        self.state = BuilderEnvOutput()
        self.masks = jnp.asarray(data["mask"])

    def reset(self) -> BuilderEnvOutput:
        self.pos = 0
        self.state = self._reset()
        return self.state

    def step(self, action: int) -> BuilderEnvOutput:
        if self.state.done.item():
            return self.state
        self.state = self._step(action, self.pos, self.state)
        self.pos += 1
        return self.state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset(self):
        tokens = jnp.ones(6, dtype=jnp.int32) * -1
        return BuilderEnvOutput(
            mask=self.start_mask, tokens=tokens, done=jnp.array(False)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(self, action: int, pos: int, state: BuilderEnvOutput):
        new_mask = jnp.take(self.masks, action, axis=0)
        token_mask = jax.nn.one_hot(pos, 6, dtype=jnp.bool)
        tokens = jnp.where(token_mask, action, state.tokens)
        mask = state.mask & ~new_mask
        return BuilderEnvOutput(mask=mask, tokens=tokens, done=jnp.array(pos >= 5))
