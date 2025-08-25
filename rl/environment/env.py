import functools

import jax
import jax.numpy as jnp
from websockets.sync.client import connect

from rl.environment.data import DEFAULT_SMOGON_FORMAT, MASKS, SET_TOKENS
from rl.environment.interfaces import (
    BuilderActorInput,
    BuilderAgentOutput,
    BuilderEnvOutput,
    BuilderHistoryOutput,
)
from rl.environment.protos.enums_pb2 import SpeciesEnum
from rl.environment.protos.features_pb2 import PackedSetFeature
from rl.environment.protos.service_pb2 import (
    Action,
    ClientRequest,
    EnvironmentResponse,
    ResetRequest,
    StepRequest,
)
from rl.environment.utils import get_ex_builder_step, process_state

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

    def reset(self, species_indices: list[int], packed_set_indices: list[int]):
        self.rqid = None
        reset_message = ClientRequest(
            reset=ResetRequest(
                username=self.username,
                species_indices=species_indices,
                packed_set_indices=packed_set_indices,
                smogon_format=f"gen{self.generation}{DEFAULT_SMOGON_FORMAT}",
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
    def __init__(
        self,
        generation: int,
        smogon_format: str = DEFAULT_SMOGON_FORMAT,
        num_team_members: int = 6,
        max_ts: int = 12,
    ):

        self.smogon_format = smogon_format
        self.generation = generation
        self.num_team_members = num_team_members
        self.max_ts = max_ts

        self.duplicate_masks = ~MASKS[generation]["duplicate"]
        self.packed_set_masks = (
            SET_TOKENS[generation][smogon_format][
                ..., PackedSetFeature.PACKED_SET_FEATURE__SPECIES
            ]
            != SpeciesEnum.SPECIES_ENUM___NULL
        )
        self.start_mask = self.packed_set_masks.any(axis=-1)

        self.ex = get_ex_builder_step(
            generation=generation, smogon_format=smogon_format
        )

        self.state: BuilderActorInput
        self.reset()

    def reset(self) -> BuilderActorInput:
        self.state = self._reset()
        return self.state

    def step(self, agent_output: BuilderAgentOutput) -> BuilderActorInput:
        if self.state.env.done.item():
            return self.state
        self.state = self._step(
            agent_output.species, agent_output.packed_set, self.state
        )
        return self.state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset(self):
        return BuilderActorInput(
            env=BuilderEnvOutput(
                species_mask=self.start_mask,
                packed_set_mask=jnp.ones(
                    self.packed_set_masks.shape[1], dtype=jnp.bool
                ),
                pos=jnp.array(0, dtype=jnp.int32),
                done=jnp.array(0, dtype=jnp.bool),
            ),
            history=jax.tree.map(lambda x: jnp.squeeze(x, axis=1), self.ex.history),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(
        self,
        species_token: jax.Array,
        packed_set_token: jax.Array,
        state: BuilderActorInput,
    ):
        pos = state.env.pos

        pos_one_hot = jax.nn.one_hot(
            pos % self.num_team_members, self.num_team_members, dtype=jnp.bool
        )
        pos_one_hot = jnp.where(state.env.done, 0, pos_one_hot)

        env = state.env
        history = state.history

        next_species_tokens = jnp.where(
            (pos < 6) & pos_one_hot, species_token, history.species_tokens
        )
        next_packed_sets = jnp.where(
            (pos >= 6) & pos_one_hot, packed_set_token, history.packed_set_tokens
        )

        next_pos = pos + 1
        species_mask = env.species_mask & self.duplicate_masks[species_token]

        packed_set_mask = self.packed_set_masks[
            next_species_tokens[next_pos % self.num_team_members]
        ]
        packed_set_mask = jnp.where(
            (~packed_set_mask).all(keepdims=True), True, packed_set_mask
        )

        return BuilderActorInput(
            env=BuilderEnvOutput(
                species_mask=species_mask,
                packed_set_mask=packed_set_mask,
                pos=next_pos,
                done=next_pos >= self.max_ts,
            ),
            history=BuilderHistoryOutput(
                species_tokens=next_species_tokens,
                packed_set_tokens=next_packed_sets,
            ),
        )
