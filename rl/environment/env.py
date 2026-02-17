import functools

import jax
import jax.numpy as jnp
import numpy as np
from websockets.sync.client import connect

from rl.environment.data import (
    NUM_ABILITIES,
    NUM_GENDERS,
    NUM_ITEMS,
    NUM_MOVES,
    NUM_NATURES,
    NUM_PACKED_SET_FEATURES,
    NUM_SPECIES,
    NUM_TYPECHART,
    STOI,
    ITOS,
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
from rl.environment.utils import generate_order, process_state

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
                packed_teams=packed_team,
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

        self.initial_species_mask = teammate_mask.any(axis=-1)
        self.initial_item_mask = self.item_masks.any(axis=0)
        self.initial_ability_mask = self.ability_masks.any(axis=0)
        self.initial_move_mask = self.move_masks.any(axis=0)
        self.initial_ev_mask = self.ev_masks.any(axis=0).any(axis=0)
        self.initial_nature_mask = self.nature_masks.any(axis=0)
        self.initial_gender_mask = self.gender_masks.any(axis=0)
        self.initial_teratype_mask = self.teratype_masks.any(axis=0)

        self.length = num_team_members * (NUM_PACKED_SET_FEATURES - 1)

    def reset(self, key: jax.Array) -> BuilderActorInput:
        self.state = self._reset(key)
        return self.state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset(self, key: jax.Array) -> BuilderActorInput:
        order = generate_order(key, self._num_team_members, NUM_PACKED_SET_FEATURES)
        member_position = order // NUM_PACKED_SET_FEATURES
        member_attribute = order % NUM_PACKED_SET_FEATURES

        return BuilderActorInput(
            env=BuilderEnvOutput(
                species_mask=self.initial_species_mask,
                item_mask=self.initial_item_mask,
                ability_mask=self.initial_ability_mask,
                move_mask=self.initial_move_mask,
                ev_mask=self.initial_ev_mask,
                teratype_mask=self.initial_teratype_mask,
                nature_mask=self.initial_nature_mask,
                gender_mask=self.initial_gender_mask,
                ts=jnp.array(0, dtype=jnp.int32),
                curr_order=order[0],
                curr_attribute=member_attribute[0],
                curr_position=member_position[0],
                done=jnp.array(0, dtype=jnp.bool),
            ),
            history=BuilderHistoryOutput(
                packed_team_member_tokens=jnp.zeros(
                    (self._num_team_members * NUM_PACKED_SET_FEATURES,),
                    dtype=jnp.int32,
                ),
                order=order,
                member_position=member_position,
                member_attribute=member_attribute,
            ),
        )

    def step(self, agent_output: BuilderAgentOutput) -> BuilderActorInput:
        if self.state.env.done.item():
            return self.state

        self.state = self._step(
            action_index=agent_output.actor_output.action_head.action_index,
            state=self.state,
        )
        return self.state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(
        self, action_index: jax.Array, state: BuilderActorInput
    ) -> BuilderActorInput:

        ts = state.env.ts
        next_ts = ts + 1

        action_index = action_index.squeeze()

        new_packed_team_member_tokens = state.history.packed_team_member_tokens.at[
            state.env.curr_order
        ].set(action_index)

        curr_order = state.history.order[ts]
        curr_position = state.history.member_position[ts]
        curr_attribute = state.history.member_attribute[ts]

        next_order = state.history.order[next_ts]
        next_position = state.history.member_position[next_ts]
        next_attribute = state.history.member_attribute[next_ts]

        packed_set_tokens = new_packed_team_member_tokens.reshape(
            -1, NUM_PACKED_SET_FEATURES
        )

        curr_species = packed_set_tokens[
            curr_position, PackedSetFeature.PACKED_SET_FEATURE__SPECIES
        ]
        curr_item = packed_set_tokens[
            curr_position, PackedSetFeature.PACKED_SET_FEATURE__ITEM
        ]
        curr_ability = packed_set_tokens[
            curr_position, PackedSetFeature.PACKED_SET_FEATURE__ABILITY
        ]
        curr_moves = packed_set_tokens[
            curr_position,
            PackedSetFeature.PACKED_SET_FEATURE__MOVE1 : PackedSetFeature.PACKED_SET_FEATURE__MOVE4
            + 1,
        ]
        curr_nature = packed_set_tokens[
            curr_position, PackedSetFeature.PACKED_SET_FEATURE__NATURE
        ]
        curr_gender = packed_set_tokens[
            curr_position, PackedSetFeature.PACKED_SET_FEATURE__GENDER
        ]
        curr_evs = packed_set_tokens[
            curr_position,
            PackedSetFeature.PACKED_SET_FEATURE__HP_EV : PackedSetFeature.PACKED_SET_FEATURE__SPE_EV
            + 1,
        ]
        curr_teratype = packed_set_tokens[
            curr_position, PackedSetFeature.PACKED_SET_FEATURE__TERATYPE
        ]

        curr_species_mask = jnp.where(
            (curr_species == 0)[None],
            self.initial_species_mask.at[
                packed_set_tokens[:, PackedSetFeature.PACKED_SET_FEATURE__SPECIES]
            ].set(False)
            & (self.ability_masks[:, curr_ability] + (curr_ability == 0))
            & (self.item_masks[:, curr_item] + (curr_item == 0))
            & (self.move_masks[:, curr_moves] + (curr_moves == 0)[None]).all(-1)
            & (self.nature_masks[:, curr_nature] + (curr_nature == 0))
            & (self.gender_masks[:, curr_gender] + (curr_gender == 0))
            & jnp.take(self.ev_masks, curr_evs, axis=1).any(-1).any(-1)
            + (curr_evs == 0).all(-1)
            & (self.teratype_masks[:, curr_teratype] + (curr_teratype == 0)),
            jax.nn.one_hot(curr_species, NUM_SPECIES, dtype=bool),
        )

        curr_item_mask = curr_species_mask @ self.item_masks
        curr_ability_mask = curr_species_mask @ self.ability_masks
        curr_move_mask = (curr_species_mask @ self.move_masks).at[curr_moves].set(False)
        curr_teratype_mask = curr_species_mask @ self.teratype_masks
        curr_nature_mask = curr_species_mask @ self.nature_masks
        curr_gender_mask = curr_species_mask @ self.gender_masks

        next_species = packed_set_tokens[
            next_position, PackedSetFeature.PACKED_SET_FEATURE__SPECIES
        ]
        next_item = packed_set_tokens[
            next_position, PackedSetFeature.PACKED_SET_FEATURE__ITEM
        ]
        next_ability = packed_set_tokens[
            next_position, PackedSetFeature.PACKED_SET_FEATURE__ABILITY
        ]
        next_moves = packed_set_tokens[
            next_position,
            PackedSetFeature.PACKED_SET_FEATURE__MOVE1 : PackedSetFeature.PACKED_SET_FEATURE__MOVE4
            + 1,
        ]
        next_nature = packed_set_tokens[
            next_position, PackedSetFeature.PACKED_SET_FEATURE__NATURE
        ]
        next_gender = packed_set_tokens[
            next_position, PackedSetFeature.PACKED_SET_FEATURE__GENDER
        ]
        next_evs = packed_set_tokens[
            next_position,
            PackedSetFeature.PACKED_SET_FEATURE__HP_EV : PackedSetFeature.PACKED_SET_FEATURE__SPE_EV
            + 1,
        ]
        next_teratype = packed_set_tokens[
            next_position, PackedSetFeature.PACKED_SET_FEATURE__TERATYPE
        ]

        next_species_mask = jnp.where(
            (next_species == 0)[None],
            self.initial_species_mask.at[
                packed_set_tokens[:, PackedSetFeature.PACKED_SET_FEATURE__SPECIES]
            ].set(False)
            & (self.ability_masks[:, next_ability] + (next_ability == 0))
            & (self.item_masks[:, next_item] + (next_item == 0))
            & (self.move_masks[:, next_moves] + (next_moves == 0)[None]).all(-1)
            & (self.nature_masks[:, next_nature] + (next_nature == 0))
            & (self.gender_masks[:, next_gender] + (next_gender == 0))
            & jnp.take(self.ev_masks, next_evs, axis=1).any(-1).any(-1)
            + (next_evs == 0).all(-1)
            & (self.teratype_masks[:, next_teratype] + (next_teratype == 0)),
            jax.nn.one_hot(next_species, NUM_SPECIES, dtype=bool),
        )
        next_item_mask = next_species_mask @ self.item_masks
        next_ability_mask = next_species_mask @ self.ability_masks
        next_move_mask = (next_species_mask @ self.move_masks).at[next_moves].set(False)
        next_teratype_mask = next_species_mask @ self.teratype_masks
        next_nature_mask = next_species_mask @ self.nature_masks
        next_gender_mask = next_species_mask @ self.gender_masks

        ev_col = (next_attribute - PackedSetFeature.PACKED_SET_FEATURE__HP_EV).clip(
            min=0, max=6
        )
        evs_sum = next_evs.sum()
        evs_remaining = 127 - evs_sum
        evs_mask = jnp.take(self.ev_masks, next_species_mask, axis=0).any(axis=0)[
            ev_col
        ] & (jnp.arange(64) <= evs_remaining)
        # If no EVs have been allocated, all EV options should be available
        evs_mask = evs_mask + (evs_mask.sum(keepdims=True) == 0)

        return BuilderActorInput(
            env=BuilderEnvOutput(
                species_mask=next_species_mask,
                item_mask=next_item_mask,
                ability_mask=next_ability_mask,
                move_mask=next_move_mask,
                ev_mask=evs_mask,
                teratype_mask=next_teratype_mask,
                nature_mask=next_nature_mask,
                gender_mask=next_gender_mask,
                ts=next_ts,
                curr_order=next_order,
                curr_attribute=next_attribute,
                curr_position=next_position,
                done=state.env.done,
            ),
            history=BuilderHistoryOutput(
                packed_team_member_tokens=new_packed_team_member_tokens,
                order=state.history.order,
                member_position=state.history.member_position,
                member_attribute=state.history.member_attribute,
            ),
        )
