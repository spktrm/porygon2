import functools

import jax
import jax.numpy as jnp
from websockets.sync.client import connect

from rl.environment.data import (
    DEFAULT_SMOGON_FORMAT,
    MASKS,
    NUM_NATURES,
    NUM_PACKED_SET_FEATURES,
    NUM_TYPECHART,
    get_format_species_mask,
)
from rl.environment.interfaces import (
    BuilderActorInput,
    BuilderAgentOutput,
    BuilderEnvOutput,
)
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

    def reset(self, packed_set_indices: list[int]):
        self.rqid = None
        reset_message = ClientRequest(
            reset=ResetRequest(
                username=self.username,
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


def construct_packed_set(
    species_token: jax.Array,
    ability_token: jax.Array,
    item_token: jax.Array,
    moveset_tokens: jax.Array,
    nature_token: jax.Array,
    teratype_token: jax.Array,
    ev_spread: jax.Array,
) -> jax.Array:
    new_packed_set = jnp.zeros((NUM_PACKED_SET_FEATURES), dtype=jnp.int32)
    new_packed_set = new_packed_set.at[
        PackedSetFeature.PACKED_SET_FEATURE__SPECIES
    ].set(species_token)
    new_packed_set = new_packed_set.at[
        PackedSetFeature.PACKED_SET_FEATURE__ABILITY
    ].set(ability_token)
    new_packed_set = new_packed_set.at[PackedSetFeature.PACKED_SET_FEATURE__ITEM].set(
        item_token
    )
    new_packed_set = new_packed_set.at[
        PackedSetFeature.PACKED_SET_FEATURE__TERATYPE
    ].set(teratype_token)
    new_packed_set = new_packed_set.at[PackedSetFeature.PACKED_SET_FEATURE__NATURE].set(
        nature_token
    )
    new_packed_set = new_packed_set.at[
        PackedSetFeature.PACKED_SET_FEATURE__MOVE1 : PackedSetFeature.PACKED_SET_FEATURE__MOVE4
        + 1
    ].set(moveset_tokens)
    new_packed_set = new_packed_set.at[
        PackedSetFeature.PACKED_SET_FEATURE__HP_EV : PackedSetFeature.PACKED_SET_FEATURE__SPE_EV
        + 1
    ].set(jnp.round(512 * ev_spread, 0).astype(jnp.int32))
    new_packed_set = new_packed_set.at[
        PackedSetFeature.PACKED_SET_FEATURE__HP_IV : PackedSetFeature.PACKED_SET_FEATURE__SPE_IV
        + 1
    ].set(31)
    return new_packed_set


class TeamBuilderEnvironment:
    def __init__(
        self,
        generation: int,
        smogon_format: str = DEFAULT_SMOGON_FORMAT,
        num_team_members: int = 6,
        max_ts: int = 32,
    ):

        self.smogon_format = smogon_format
        self.generation = generation
        self.num_team_members = num_team_members
        self.max_ts = max_ts
        self.rng_key = jax.random.key(42)

        self.duplicate_masks = ~MASKS[generation]["duplicate"]

        self.start_mask = get_format_species_mask(generation, smogon_format)

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
            agent_output.actor_output.continue_head.action_index,
            agent_output.actor_output.selection_head.action_index,
            agent_output.actor_output.species_head.action_index,
            agent_output.actor_output.ability_head.action_index,
            agent_output.actor_output.item_head.action_index,
            agent_output.actor_output.moveset_head.action_index,
            agent_output.actor_output.nature_head.action_index,
            agent_output.actor_output.teratype_head.action_index,
            agent_output.actor_output.ev_head.action_index,
            self.state,
        )
        return self.state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset(self, rng_key: jax.Array):
        species_mask = self.start_mask

        species_subkeys = jax.random.split(rng_key, self.num_team_members)
        item_subkeys = jax.random.split(rng_key, self.num_team_members)
        move_subkeys = jax.random.split(rng_key, self.num_team_members)
        ability_subkeys = jax.random.split(rng_key, self.num_team_members)
        nature_subkeys = jax.random.split(rng_key, self.num_team_members)
        teratype_subkeys = jax.random.split(rng_key, self.num_team_members)
        ev_spread_subkeys = jax.random.split(rng_key, self.num_team_members)

        packed_set_tokens = []

        for i in range(self.num_team_members):
            species_policy = species_mask / species_mask.sum()

            species_token = jax.random.choice(
                species_subkeys[i], species_mask.shape[-1], (1,), p=species_policy
            ).squeeze()
            species_mask = species_mask & self.duplicate_masks[species_token]

            learnset_mask = jnp.take(
                MASKS[self.generation]["learnset"], species_token, axis=0
            )
            move_policy = learnset_mask / learnset_mask.sum()
            moveset_tokens = jax.random.choice(
                move_subkeys[i],
                learnset_mask.shape[-1],
                (4,),
                p=move_policy,
                replace=False,
            ).squeeze()

            item_mask = jnp.take(MASKS[self.generation]["items"], species_token, axis=0)
            item_policy = item_mask / item_mask.sum()
            item_token = jax.random.choice(
                item_subkeys[i], item_mask.shape[-1], (1,), p=item_policy
            ).squeeze()

            ability_mask = jnp.take(
                MASKS[self.generation]["abilities"], species_token, axis=0
            )
            ability_policy = ability_mask / ability_mask.sum()
            ability_token = jax.random.choice(
                ability_subkeys[i], ability_mask.shape[-1], (1,), p=ability_policy
            ).squeeze()

            nature_token = jax.random.randint(
                nature_subkeys[i], (1,), 0, NUM_NATURES
            ).squeeze()
            teratype_token = jax.random.randint(
                teratype_subkeys[i], (1,), 0, NUM_TYPECHART
            ).squeeze()
            ev_spread = jax.random.uniform(ev_spread_subkeys[i], (6,)).squeeze()

            packed_set_tokens.append(
                construct_packed_set(
                    species_token,
                    ability_token,
                    item_token,
                    moveset_tokens,
                    nature_token,
                    teratype_token,
                    ev_spread,
                )
            )

        return BuilderActorInput(
            env=BuilderEnvOutput(
                species_mask=species_mask,
                packed_set_tokens=jnp.array(packed_set_tokens),
                done=jnp.array(0, dtype=jnp.bool),
                ts=jnp.array(0, dtype=jnp.int32),
            )
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(
        self,
        continue_token: jax.Array,
        selection_token: jax.Array,
        species_token: jax.Array,
        ability_token: jax.Array,
        item_token: jax.Array,
        moveset_tokens: jax.Array,
        nature_token: jax.Array,
        teratype_token: jax.Array,
        ev_spread: jax.Array,
        state: BuilderActorInput,
    ):
        ts = state.env.ts
        next_ts = ts + 1

        cont_edits = continue_token == 0
        stop_edits = continue_token == 1
        traj_over = next_ts >= self.max_ts

        done = state.env.done | traj_over | stop_edits

        selection_oh = (
            cont_edits
            & ~state.env.done
            & jax.nn.one_hot(selection_token, self.num_team_members, dtype=jnp.bool)
        )

        species_tokens = state.env.packed_set_tokens[
            ..., PackedSetFeature.PACKED_SET_FEATURE__SPECIES
        ]
        old_species_token = species_tokens[selection_token]
        old_species_mask = self.duplicate_masks[old_species_token]
        new_species_mask = self.duplicate_masks[species_token]

        species_mask = jnp.where(old_species_mask, state.env.species_mask, True)
        species_mask = jnp.where(new_species_mask, species_mask, False)

        new_packed_set = construct_packed_set(
            species_token,
            ability_token,
            item_token,
            moveset_tokens,
            nature_token,
            teratype_token,
            ev_spread,
        )

        packed_set_tokens = jnp.where(
            selection_oh[..., None], new_packed_set, state.env.packed_set_tokens
        )

        return BuilderActorInput(
            env=BuilderEnvOutput(
                species_mask=species_mask,
                packed_set_tokens=packed_set_tokens,
                done=done,
                ts=next_ts,
            ),
        )
