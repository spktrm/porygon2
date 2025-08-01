import jax
import numpy as np
from websockets.sync.client import connect

from rl.environment.protos.service_pb2 import (
    ClientRequest,
    EnvironmentResponse,
    ResetRequest,
    StepRequest,
)
from rl.environment.utils import process_state

SERVER_URI = "ws://localhost:8080"


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
        self.last_state = process_state(server_message.state)
        return self.last_state

    def reset(self, team_indices: list[int]):
        self.rqid = None
        reset_message = ClientRequest(
            reset=ResetRequest(username=self.username, team_indices=team_indices)
        )
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
