import functools

import jax
import jax.numpy as jnp
from websockets.sync.client import connect

from rl.environment.data import MASKS, SET_MASK
from rl.environment.interfaces import (
    BuilderActorInput,
    BuilderAgentOutput,
    BuilderEnvOutput,
)
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
                smogon_format=f"gen{self.generation}_ou_all_formats",
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
        smogon_format: str,
        num_team_members: int = 6,
        num_metagame_slots: int = 32,
        max_trajectory_length: int = 24,
        min_trajectory_length: int = 1,
        *,
        initial_metagame_token: int | None = None,
    ):
        self.smogon_format = smogon_format
        self.generation = generation
        self.num_team_members = num_team_members
        self.num_metagame_slots = num_metagame_slots
        self.max_trajectory_length = max_trajectory_length
        self.min_trajectory_length = min_trajectory_length
        self.rng_key = jax.random.key(42)

        if max_trajectory_length < min_trajectory_length:
            raise ValueError(
                "max_trajectory_length must be greater than or equal to min_trajectory_length"
            )

        self.initial_continue_mask = (
            jnp.ones(2, dtype=jnp.bool)
            if min_trajectory_length <= 1
            else jnp.array([1, 0], dtype=jnp.bool)
        )
        self.metagame_mask = (
            jnp.ones((self.num_metagame_slots,), dtype=jnp.bool)
            if initial_metagame_token is None
            else jax.nn.one_hot(
                initial_metagame_token, self.num_metagame_slots, dtype=jnp.bool
            )
        )

        self.duplicate_masks = ~MASKS[generation]["duplicate"]

        self.start_mask = SET_MASK[generation][smogon_format].any(axis=-1)

        self.ex = get_ex_builder_step()

        self.state: BuilderActorInput
        self.reset()

    def split_rng(self):
        subkey, self.rng_key = jax.random.split(self.rng_key)
        return subkey

    def reset(self) -> BuilderActorInput:
        key = self.split_rng()
        self.state = self._reset(key)
        return self.state

    def step(self, agent_output: BuilderAgentOutput) -> BuilderActorInput:
        if self.state.env.done.item():
            return self.state
        self.state = self._step(
            metagame_token=agent_output.actor_output.metagame_head.action_index,
            continue_token=agent_output.actor_output.continue_head.action_index,
            selection_token=agent_output.actor_output.selection_head.action_index,
            species_token=agent_output.actor_output.species_head.action_index,
            packed_set_token=agent_output.actor_output.packed_set_head.action_index,
            state=self.state,
        )
        return self.state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset(self, rng_key: jax.Array):
        species_mask = self.start_mask

        species_subkeys = jax.random.split(rng_key, self.num_team_members)
        packed_set_subkeys = jax.random.split(rng_key, self.num_team_members)

        species_tokens = []
        packed_set_tokens = []

        for i in range(self.num_team_members):
            species_policy = species_mask / species_mask.sum()

            species_token_i = jax.random.choice(
                species_subkeys[i], species_mask.shape[-1], (1,), p=species_policy
            ).squeeze()
            species_mask = species_mask & self.duplicate_masks[species_token_i]
            species_tokens.append(species_token_i)

            packed_set_mask = SET_MASK[self.generation][self.smogon_format][
                species_token_i
            ]
            packed_set_policy = packed_set_mask / packed_set_mask.sum()
            packed_set_token_i = jax.random.choice(
                packed_set_subkeys[i],
                packed_set_mask.shape[-1],
                (1,),
                p=packed_set_policy,
            ).squeeze()
            packed_set_tokens.append(packed_set_token_i)

        return BuilderActorInput(
            env=BuilderEnvOutput(
                continue_mask=self.initial_continue_mask,
                species_mask=species_mask,
                species_tokens=jnp.array(species_tokens),
                packed_set_tokens=jnp.array(packed_set_tokens),
                done=jnp.array(0, dtype=jnp.bool),
                ts=jnp.array(0, dtype=jnp.int32),
                metagame_token=jnp.array(0, dtype=jnp.int32),
                metagame_mask=self.metagame_mask,
            )
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(
        self,
        metagame_token: jax.Array,
        continue_token: jax.Array,
        selection_token: jax.Array,
        species_token: jax.Array,
        packed_set_token: jax.Array,
        state: BuilderActorInput,
    ):
        ts = state.env.ts
        next_ts = ts + 1

        cont_edits = continue_token == 0
        stop_edits = continue_token == 1
        traj_over = next_ts >= self.max_trajectory_length

        done = state.env.done | traj_over | stop_edits

        selection_oh = (
            cont_edits
            & ~state.env.done
            & jax.nn.one_hot(selection_token, self.num_team_members, dtype=jnp.bool)
        )

        old_species_mask = jnp.take(
            self.duplicate_masks, state.env.species_tokens, axis=0
        )
        new_species_mask = self.duplicate_masks[species_token]

        species_tokens = jnp.where(
            selection_oh, species_token, state.env.species_tokens
        )
        species_mask = self.start_mask & jnp.prod(
            jnp.where(
                selection_oh[..., None], new_species_mask[None], old_species_mask
            ),
            axis=0,
        )
        packed_set_tokens = jnp.where(
            selection_oh, packed_set_token, state.env.packed_set_tokens
        )
        metagame_mask = jax.nn.one_hot(
            metagame_token, self.num_metagame_slots, dtype=jnp.bool
        )

        continue_mask = jnp.array(
            [1, next_ts > self.min_trajectory_length], dtype=jnp.bool
        )

        return BuilderActorInput(
            env=BuilderEnvOutput(
                continue_mask=continue_mask,
                species_mask=species_mask,
                species_tokens=species_tokens,
                packed_set_tokens=packed_set_tokens,
                done=done,
                ts=next_ts,
                metagame_token=metagame_token,
                metagame_mask=metagame_mask,
            ),
        )
