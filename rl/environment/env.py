import functools

import jax
import jax.numpy as jnp
from websockets.sync.client import connect

from rl.environment.data import (
    NUM_ABILITIES,
    NUM_GENDERS,
    NUM_ITEMS,
    NUM_MOVES,
    NUM_NATURES,
    NUM_PACKED_TEAM_MEMBER_FEATURES,
    NUM_TYPECHART,
    PackedSetFeature,
)
from rl.environment.interfaces import (
    BuilderActorInput,
    BuilderAgentOutput,
    BuilderEnvOutput,
    BuilderHistoryOutput,
)
from rl.environment.protos.service_pb2 import (
    Action,
    ClientRequest,
    ResetRequest,
    StepRequest,
    WorkerResponse,
)
from rl.environment.utils import process_state

SERVER_URI = "ws://localhost:8080"


class SinglePlayerSyncEnvironment:
    def __init__(self, username: str, generation: int = 3, smogon_format: str = "ou"):

        self.username = username
        self.rqid = None
        self.last_state = None
        self.game_id = None

        self.websocket = connect(
            SERVER_URI,
            additional_headers={"username": username},
        )
        self.generation = generation
        self.smogon_format = smogon_format
        self.metgame_token = None

    def _set_game_id(self, game_id: str):
        self.game_id = game_id

    def _reset_game_id(self):
        self.game_id = None

    def _recv(self):
        recv_data = self.websocket.recv()
        worker_response = WorkerResponse.FromString(recv_data)
        environment_response = worker_response.environment_response
        self.rqid = environment_response.state.rqid

        self.last_state = process_state(environment_response.state)
        return self.last_state

    def reset(self, packed_team: list[int]):
        self.rqid = None
        reset_message = ClientRequest(
            reset=ResetRequest(
                username=self.username,
                smogon_format=f"gen{self.generation}{self.smogon_format}",
                game_id=self.game_id,
                packed_team=packed_team,
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


@functools.partial(jax.jit, static_argnames=["r", "N"])
def generate_order(key, r, N):
    total_size = r * N
    selection_order = jax.random.permutation(key, jnp.arange(total_size))

    # 1. Entry point is now i % N == 1
    entry_priorities = selection_order.reshape(r, N)[:, 1]
    block_gate_priority = jnp.repeat(entry_priorities, N)

    # 2. Effective priority
    effective_priority = jnp.maximum(selection_order, block_gate_priority)

    # 3. Get the full sorted order
    sorted_indices = jnp.argsort(effective_priority)

    # 4. FIXED: Instead of boolean masking, we use a static filter
    # We find which positions in the 'sorted_indices' do NOT contain an i % N == 0
    # But wait—it's easier to just calculate the valid indices first!

    # Alternative JIT-safe approach:
    # Use jnp.where with a fixed-size size argument or simple slicing if possible.
    # Since we must return r * (N-1), we use jnp.take with static indices.

    is_valid = (sorted_indices % N) != 0
    # Sort the boolean mask to push all 'True' values to the front
    # and then slice the first r*(N-1) elements.
    valid_positions = jnp.argsort(~is_valid)

    return sorted_indices[valid_positions[: r * (N - 1)]]


class TeamBuilderEnvironment:
    state: BuilderActorInput

    def __init__(
        self,
        generation: int,
        smogon_format: str,
        num_team_members: int = 6,
    ):
        self._smogon_format = smogon_format
        self._generation = generation
        self._num_team_members = num_team_members

        teammate_mask = jnp.load(
            f"data/data/gen{generation}/mask/teammates_mask_{smogon_format}.npy"
        )
        self.initial_species_mask = teammate_mask.any(axis=-1)
        self.item_masks = jnp.load(
            f"data/data/gen{generation}/mask/items_mask_{smogon_format}.npy"
        )
        self.ability_masks = jnp.load(
            f"data/data/gen{generation}/mask/abilities_mask_{smogon_format}.npy"
        )
        self.move_masks = jnp.load(
            f"data/data/gen{generation}/mask/moves_mask_{smogon_format}.npy"
        )
        self.ev_masks = jnp.load(
            f"data/data/gen{generation}/mask/ev_mask_{smogon_format}.npy"
        )
        self.nature_masks = jnp.load(
            f"data/data/gen{generation}/mask/nature_mask_{smogon_format}.npy"
        )
        self.gender_masks = jnp.load(
            f"data/data/gen{generation}/mask/gender_mask_{smogon_format}.npy"
        )
        self.teratype_masks = jnp.load(
            f"data/data/gen{generation}/mask/teratypes_mask_{smogon_format}.npy"
        )

        self.length = num_team_members * (NUM_PACKED_TEAM_MEMBER_FEATURES - 1)

    def reset(self, key: jax.Array) -> BuilderActorInput:
        self.state = self._reset(key)
        return self.state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset(self, key: jax.Array) -> BuilderActorInput:
        order = generate_order(
            key, self._num_team_members, NUM_PACKED_TEAM_MEMBER_FEATURES
        )
        row = order // NUM_PACKED_TEAM_MEMBER_FEATURES
        col = order % NUM_PACKED_TEAM_MEMBER_FEATURES

        return BuilderActorInput(
            env=BuilderEnvOutput(
                species_mask=self.initial_species_mask,
                item_mask=jnp.ones((NUM_ITEMS,), dtype=jnp.bool),
                ability_mask=jnp.ones((NUM_ABILITIES,), dtype=jnp.bool),
                move_mask=jnp.ones((NUM_MOVES,), dtype=jnp.bool),
                ev_mask=jnp.ones((64,), dtype=jnp.bool),
                nature_mask=jnp.ones((NUM_NATURES,), dtype=jnp.bool),
                gender_mask=jnp.ones((NUM_GENDERS,), dtype=jnp.bool),
                teratype_mask=jnp.ones((NUM_TYPECHART,), dtype=jnp.bool),
                ts=jnp.array(0, dtype=jnp.int32),
                done=jnp.array(0, dtype=jnp.bool),
            ),
            history=BuilderHistoryOutput(
                packed_team_member_tokens=jnp.zeros(
                    (self._num_team_members * NUM_PACKED_TEAM_MEMBER_FEATURES,),
                    dtype=jnp.int32,
                ),
                order=order,
                member_position=row,
                member_attribute=col,
            ),
        )

    def step(self, agent_output: BuilderAgentOutput) -> BuilderActorInput:
        if self.state.env.done.item():
            return self.state

        self.state = self._step(
            species_index=agent_output.actor_output.species_head.action_index,
            item_index=agent_output.actor_output.item_head.action_index,
            ability_index=agent_output.actor_output.ability_head.action_index,
            move_index=agent_output.actor_output.move_head.action_index,
            ev_index=agent_output.actor_output.ev_head.action_index,
            nature_index=agent_output.actor_output.nature_head.action_index,
            teratype_index=agent_output.actor_output.teratype_head.action_index,
            gender_index=agent_output.actor_output.gender_head.action_index,
            state=self.state,
        )
        return self.state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(
        self,
        species_index: int,
        item_index: int,
        ability_index: int,
        move_index: int,
        ev_index: int,
        nature_index: int,
        teratype_index: int,
        gender_index: int,
        state: BuilderActorInput,
    ) -> BuilderActorInput:

        ts = state.env.ts
        next_ts = ts + 1

        concat_tokens = jnp.stack(
            (
                species_index,
                species_index,
                item_index,
                ability_index,
                move_index,
                move_index,
                move_index,
                move_index,
                nature_index,
                gender_index,
                ev_index,
                ev_index,
                ev_index,
                ev_index,
                ev_index,
                ev_index,
                teratype_index,
                teratype_index,
            )
        )
        token_index = concat_tokens @ jax.nn.one_hot(
            state.history.member_attribute[ts],
            NUM_PACKED_TEAM_MEMBER_FEATURES,
            dtype=jnp.int32,
        )

        curr_order = state.history.order[ts]
        new_history = (
            state.history.packed_team_member_tokens.at[curr_order]
            .set(token_index)
            .reshape(-1, NUM_PACKED_TEAM_MEMBER_FEATURES)
        )

        next_row = state.history.member_position[next_ts]
        next_col = state.history.member_attribute[next_ts]

        current_species = new_history[
            next_row, PackedSetFeature.PACKED_SET_FEATURE__SPECIES
        ]
        current_moves = new_history[
            next_row,
            PackedSetFeature.PACKED_SET_FEATURE__MOVE1 : PackedSetFeature.PACKED_SET_FEATURE__MOVE4
            + 1,
        ]
        current_evs = new_history[
            next_row,
            PackedSetFeature.PACKED_SET_FEATURE__HP_EV : PackedSetFeature.PACKED_SET_FEATURE__SPE_EV
            + 1,
        ]

        move_mask = (
            jnp.take(self.move_masks, current_species, axis=0)
            .at[current_moves]
            .set(False)
        )

        ev_col = (next_col - PackedSetFeature.PACKED_SET_FEATURE__HP_EV).clip(
            min=0, max=6
        )
        evs_sum = current_evs.sum()
        evs_remaining = 127 - evs_sum
        evs_mask = jnp.take(self.ev_masks, current_species, axis=0)[ev_col] & (
            jnp.arange(64) <= evs_remaining
        )
        # If no EVs have been allocated, all EV options should be available
        evs_mask = evs_mask + (evs_mask.sum(keepdims=True) == 0)

        next_species_mask = state.env.species_mask.at[current_species].set(False)

        return BuilderActorInput(
            env=BuilderEnvOutput(
                species_mask=next_species_mask,
                item_mask=jnp.take(self.item_masks, current_species, axis=0),
                ability_mask=jnp.take(self.ability_masks, current_species, axis=0),
                move_mask=move_mask,
                ev_mask=evs_mask,
                teratype_mask=jnp.take(self.teratype_masks, current_species, axis=0),
                nature_mask=jnp.take(self.nature_masks, current_species, axis=0),
                gender_mask=jnp.take(self.gender_masks, current_species, axis=0),
                ts=next_ts,
                done=state.env.done,
            ),
            history=BuilderHistoryOutput(
                packed_team_member_tokens=new_history.reshape(-1),
                order=state.history.order,
                member_position=state.history.member_position,
                member_attribute=state.history.member_attribute,
            ),
        )
