import functools
import numpy as np

import jax
import jax.numpy as jnp
from websockets.sync.client import connect

from rl.environment.data import (
    NUM_MOVES,
    NUM_PACKED_SET_FEATURES,
    NUM_SPECIES,
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
from rl.environment.protos.enums_pb2 import TypechartEnum
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

    @property
    def num_team_members(self) -> int:
        return self._num_team_members

        # Load Masks
        teammate_mask = jnp.load(
            f"data/data/gen{generation}/mask/teammates_mask_{smogon_format}.npy"
        )
        self.species_usage = jnp.load(
            f"data/data/gen{generation}/usage/species_usage_{smogon_format}.npy"
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

        # Initial Masks
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
                ev_reward=jnp.array(0, dtype=jnp.float32),
                species_reward=jnp.array(0, dtype=jnp.float32),
                curr_order=order[0],
                curr_attribute=member_attribute[0],
                curr_position=member_position[0],
                done=jnp.array(False, dtype=jnp.bool),
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

        # 1. Apply Agent Action
        state = self._transition(state, action_index)

        # 2. Initialize Autofill Loop Logic
        #    Check if the state resulting from the agent's action forces a move
        def _get_initial_autofill():
            return self._get_autofill_state(state)

        def _skip_initial():
            return jnp.array(False), jnp.array(0, dtype=jnp.int32)

        is_forced, forced_action = jax.lax.cond(
            state.env.done, _skip_initial, _get_initial_autofill
        )

        init_val = (state, is_forced, forced_action)

        def cond_fun(val):
            state, is_forced, _ = val
            return is_forced & (~state.env.done)

        def body_fun(val):
            curr_state, _, action_to_apply = val

            # Transition with forced action
            next_state = self._transition(curr_state, action_to_apply)

            # Check if *next* state forces another move
            def _calc_next_autofill():
                return self._get_autofill_state(next_state)

            def _return_no_op():
                return jnp.array(False), jnp.array(0, dtype=jnp.int32)

            next_is_forced, next_action = jax.lax.cond(
                next_state.env.done, _return_no_op, _calc_next_autofill
            )

            return (next_state, next_is_forced, next_action)

        final_val = jax.lax.while_loop(cond_fun, body_fun, init_val)

        return final_val[0]

    def _transition(
        self, state: BuilderActorInput, action_index: jax.Array
    ) -> BuilderActorInput:
        ts = state.env.ts
        curr_order = state.env.curr_order
        curr_attribute = state.env.curr_attribute
        action_index = action_index.squeeze()

        new_packed_team_member_tokens = state.history.packed_team_member_tokens.at[
            curr_order
        ].set(action_index)

        next_ts = ts + 1
        total_steps = state.history.order.shape[0]
        is_done = next_ts >= total_steps

        def _done_branch(_):
            team_evs = new_packed_team_member_tokens.reshape(
                -1, NUM_PACKED_SET_FEATURES
            )[
                ...,
                PackedSetFeature.PACKED_SET_FEATURE__HP_EV : PackedSetFeature.PACKED_SET_FEATURE__SPE_EV
                + 1,
            ]
            ev_fulfillment_per_member = team_evs.sum(axis=-1) / 127.5
            return state.replace(
                env=state.env.replace(
                    ts=next_ts,
                    ev_reward=ev_fulfillment_per_member.mean(),  # Average EV fulfillment across the team
                    done=jnp.array(True, dtype=jnp.bool),
                ),
                history=state.history.replace(
                    packed_team_member_tokens=new_packed_team_member_tokens
                ),
            )

        def _continue_branch(_):
            next_order = state.history.order[next_ts]
            next_position = state.history.member_position[next_ts]
            next_attribute = state.history.member_attribute[next_ts]

            packed_set_tokens = new_packed_team_member_tokens.reshape(
                -1, NUM_PACKED_SET_FEATURES
            )

            # --- Member Context Selection ---
            # [cite_start]Extract all current attributes for the member being edited [cite: 54, 55, 56]
            current_member_tokens = packed_set_tokens[next_position]
            m_species = current_member_tokens[
                PackedSetFeature.PACKED_SET_FEATURE__SPECIES
            ]
            m_item = current_member_tokens[PackedSetFeature.PACKED_SET_FEATURE__ITEM]
            m_ability = current_member_tokens[
                PackedSetFeature.PACKED_SET_FEATURE__ABILITY
            ]
            m_moves = current_member_tokens[
                PackedSetFeature.PACKED_SET_FEATURE__MOVE1 : PackedSetFeature.PACKED_SET_FEATURE__MOVE4
                + 1
            ]
            m_evs = current_member_tokens[
                PackedSetFeature.PACKED_SET_FEATURE__HP_EV : PackedSetFeature.PACKED_SET_FEATURE__SPE_EV
                + 1
            ]
            m_nature = current_member_tokens[
                PackedSetFeature.PACKED_SET_FEATURE__NATURE
            ]
            m_gender = current_member_tokens[
                PackedSetFeature.PACKED_SET_FEATURE__GENDER
            ]
            m_tera = current_member_tokens[
                PackedSetFeature.PACKED_SET_FEATURE__TERATYPE
            ]

            # --- Proactive Species Pool (Fix 2) ---
            # [cite_start]A species is valid only if it satisfies EVERY attribute selected so far. [cite: 53]
            # Attributes set to 0 (Unspecified) are ignored in the constraint.
            species_pool = (
                self.initial_species_mask
                & (self.item_masks[:, m_item] | (m_item == 0))
                & (self.ability_masks[:, m_ability] | (m_ability == 0))
                & (self.move_masks[:, m_moves[0]] | (m_moves[0] == 0))
                & (self.move_masks[:, m_moves[1]] | (m_moves[1] == 0))
                & (self.move_masks[:, m_moves[2]] | (m_moves[2] == 0))
                & (self.move_masks[:, m_moves[3]] | (m_moves[3] == 0))
                & (self.nature_masks[:, m_nature] | (m_nature == 0))
                & (self.gender_masks[:, m_gender] | (m_gender == 0))
                & (self.teratype_masks[:, m_tera] | (m_tera == 0))
            )

            # Prevent picking duplicate species across the team (Species Clause)
            # Find all species currently on the team excluding the current slot
            other_species = packed_set_tokens[
                :, PackedSetFeature.PACKED_SET_FEATURE__SPECIES
            ]
            other_species_mask = jax.nn.one_hot(
                other_species, NUM_SPECIES, dtype=jnp.bool_
            ).any(axis=0)
            other_species_mask = other_species_mask.at[m_species].set(
                False
            )  # Don't mask yourself

            species_pool = species_pool & ~other_species_mask

            # Finalize species mask for this step
            next_species_mask = jnp.where(
                (m_species == 0)[None],
                species_pool,
                jax.nn.one_hot(m_species, NUM_SPECIES, dtype=jnp.bool_),
            )

            # --- Derived Attribute Masks ---
            next_item_mask = next_species_mask @ self.item_masks
            next_ability_mask = next_species_mask @ self.ability_masks
            next_teratype_mask = next_species_mask @ self.teratype_masks
            next_nature_mask = next_species_mask @ self.nature_masks
            next_gender_mask = next_species_mask @ self.gender_masks

            # Vectorized Move Exclusion (Prevents selecting the same move twice)
            already_selected_moves_mask = jax.nn.one_hot(
                m_moves, NUM_MOVES, dtype=jnp.bool_
            ).any(axis=0)
            next_move_mask = (
                next_species_mask @ self.move_masks
            ) & ~already_selected_moves_mask

            # --- Accurate EV Budget Logic (512 Total / 252 Stat) ---
            ev_col = (next_attribute - PackedSetFeature.PACKED_SET_FEATURE__HP_EV).clip(
                0, 5
            )
            is_ev_attr = (
                next_attribute >= PackedSetFeature.PACKED_SET_FEATURE__HP_EV
            ) & (next_attribute <= PackedSetFeature.PACKED_SET_FEATURE__SPE_EV)

            # 2. Budget Calculation
            current_slot_val = m_evs[ev_col]
            evs_sum_others = m_evs.sum() - current_slot_val

            # Max total tokens = 127 (510 EVs / 4 = 127.5 -> 127 integer tokens)
            evs_remaining_total = 127 - evs_sum_others

            # 3. Mask Generation
            valid_evs_flat = next_species_mask @ self.ev_masks.reshape(
                self.ev_masks.shape[0], -1
            )
            species_allowed_evs = valid_evs_flat.reshape(6, 64)[ev_col]

            # Max single stat tokens = 63 (252 EVs / 4 = 63)
            # Upper bound is min(63, Remaining Total Budget)
            stat_upper_bound = jnp.minimum(63, evs_remaining_total)
            budget_mask = jnp.arange(64) <= stat_upper_bound

            final_ev_mask = jnp.where(
                is_ev_attr, species_allowed_evs & budget_mask, self.initial_ev_mask
            )
            final_ev_mask = final_ev_mask.at[0].set(True)  # Ensure '0' is always legal

            return state.replace(
                env=state.env.replace(
                    species_mask=next_species_mask,
                    item_mask=next_item_mask,
                    ability_mask=next_ability_mask,
                    move_mask=next_move_mask,
                    ev_mask=final_ev_mask,
                    teratype_mask=next_teratype_mask,
                    nature_mask=next_nature_mask,
                    gender_mask=next_gender_mask,
                    ts=next_ts,
                    curr_order=next_order,
                    curr_attribute=next_attribute,
                    curr_position=next_position,
                    species_reward=jnp.where(
                        curr_attribute == PackedSetFeature.PACKED_SET_FEATURE__SPECIES,
                        jnp.take(self.species_usage, action_index, axis=0, mode="clip"),
                        jnp.array(0.0, dtype=jnp.float32),
                    ),
                    done=jnp.array(False, dtype=jnp.bool),
                ),
                history=state.history.replace(
                    packed_team_member_tokens=new_packed_team_member_tokens,
                ),
            )

        return jax.lax.cond(is_done, _done_branch, _continue_branch, operand=None)

    def make_edit_step(
        self,
        history: BuilderHistoryOutput,
        member_idx: int,
    ) -> BuilderActorInput:
        """Return a BuilderActorInput for editing *member_idx*'s teratype.

        The step uses ``ts = num_team_members * NUM_PACKED_SET_FEATURES + j``
        where ``j`` is the position of this teratype slot in the original build
        order.  The model uses ``ts >= num_team_members * NUM_PACKED_SET_FEATURES``
        to detect edit steps and switch to bidirectional attention.
        """
        team_tokens = np.array(history.packed_team_member_tokens)
        order = np.array(history.order)

        flat_idx = (
            member_idx * NUM_PACKED_SET_FEATURES
            + PackedSetFeature.PACKED_SET_FEATURE__TERATYPE
        )

        # Position of this slot in the original autoregressive build order.
        j = int(np.argmax(order == flat_idx))

        # Edit ts is offset beyond the full token array so it is always >=
        # num_team_members * NUM_PACKED_SET_FEATURES.
        edit_ts = self._num_team_members * NUM_PACKED_SET_FEATURES + j

        # Use the species-specific teratype mask when the species is known.
        species_idx = int(
            team_tokens[
                member_idx * NUM_PACKED_SET_FEATURES
                + PackedSetFeature.PACKED_SET_FEATURE__SPECIES
            ]
        )
        if species_idx > 0:
            teratype_mask = np.array(self.teratype_masks[species_idx])
        else:
            teratype_mask = np.array(self.initial_teratype_mask)
        teratype_mask = teratype_mask.copy()
        # Disable UNSPECIFIED (0) – a valid edit must choose a real type.
        teratype_mask[TypechartEnum.TYPECHART_ENUM___UNSPECIFIED] = False

        return BuilderActorInput(
            env=BuilderEnvOutput(
                species_mask=self.initial_species_mask,
                item_mask=self.initial_item_mask,
                ability_mask=self.initial_ability_mask,
                move_mask=self.initial_move_mask,
                ev_mask=self.initial_ev_mask,
                teratype_mask=teratype_mask,
                nature_mask=self.initial_nature_mask,
                gender_mask=self.initial_gender_mask,
                ts=jnp.array(edit_ts, dtype=jnp.int32),
                curr_order=jnp.array(flat_idx, dtype=jnp.int32),
                curr_attribute=jnp.array(
                    PackedSetFeature.PACKED_SET_FEATURE__TERATYPE, dtype=jnp.int32
                ),
                curr_position=jnp.array(member_idx, dtype=jnp.int32),
                done=jnp.array(False, dtype=jnp.bool_),
                ev_reward=jnp.array(0.0, dtype=jnp.float32),
                species_reward=jnp.array(0.0, dtype=jnp.float32),
            ),
            history=history,
        )

    def _get_autofill_state(
        self, state: BuilderActorInput
    ) -> tuple[jax.Array, jax.Array]:
        env = state.env
        curr = env.curr_attribute

        def check_mask(mask):
            return (jnp.sum(mask) == 1), jnp.argmax(mask).astype(jnp.int32)

        # Helper for ignored attributes: Always return (False, 0) so the agent (or 0) takes over
        def ignore_attribute():
            return jnp.array(False, dtype=jnp.bool), jnp.array(0, dtype=jnp.int32)

        branches = [
            lambda: check_mask(env.species_mask),  # 0
            lambda: check_mask(env.item_mask),  # 1
            lambda: check_mask(env.ability_mask),  # 2
            lambda: check_mask(env.move_mask),  # 3
            lambda: check_mask(env.nature_mask),  # 4
            lambda: check_mask(env.gender_mask),  # 5
            lambda: check_mask(env.ev_mask),  # 6
            lambda: check_mask(env.teratype_mask),  # 7
            ignore_attribute,  # 8: Fallback/Ignored
        ]

        predicates = [
            curr == PackedSetFeature.PACKED_SET_FEATURE__SPECIES,
            curr == PackedSetFeature.PACKED_SET_FEATURE__ITEM,
            curr == PackedSetFeature.PACKED_SET_FEATURE__ABILITY,
            (curr >= PackedSetFeature.PACKED_SET_FEATURE__MOVE1)
            & (curr <= PackedSetFeature.PACKED_SET_FEATURE__MOVE4),
            curr == PackedSetFeature.PACKED_SET_FEATURE__NATURE,
            curr == PackedSetFeature.PACKED_SET_FEATURE__GENDER,
            (curr >= PackedSetFeature.PACKED_SET_FEATURE__HP_EV)
            & (curr <= PackedSetFeature.PACKED_SET_FEATURE__SPE_EV),
            curr == PackedSetFeature.PACKED_SET_FEATURE__TERATYPE,
        ]

        # If no predicate matches, use index 8 (ignore_attribute)
        branch_idx = jnp.select(predicates, jnp.arange(8), default=8)

        return jax.lax.switch(branch_idx, branches)
