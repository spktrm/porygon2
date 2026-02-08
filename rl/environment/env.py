import functools
import json

import jax
import jax.numpy as jnp
from websockets.sync.client import connect

from rl.environment.data import (
    HUMAN_SPECIES_COUNTS,
    HUMAN_TEAMMATE_COUNTS,
    MASKS,
    NUM_ABILITIES,
    NUM_ITEMS,
    NUM_MOVES,
    NUM_PACKED_TEAM_MEMBER_FEATURES,
    NUM_SPECIES,
    NUM_TYPECHART,
    SET_MASK,
    PackedTeamMemberFeature,
)
from rl.environment.interfaces import (
    BuilderActorInput,
    BuilderAgentOutput,
    BuilderEnvOutput,
    BuilderFullActorInput,
    BuilderFullEnvOutput,
    BuilderFullHistoryOutput,
    BuilderHistoryOutput,
)
from rl.environment.protos.enums_pb2 import SpeciesEnum
from rl.environment.protos.service_pb2 import (
    Action,
    ClientRequest,
    ResetRequest,
    StepRequest,
    WorkerResponse,
)
from rl.environment.utils import get_ex_builder_step, process_state

SERVER_URI = "ws://localhost:8080"


class SinglePlayerSyncEnvironment:
    def __init__(self, username: str, generation: int = 3):

        self.username = username
        self.rqid = None
        self.last_state = None
        self.game_id = None

        self.websocket = connect(
            SERVER_URI,
            additional_headers={"username": username},
        )
        self.generation = generation
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

    def reset(
        self,
        species_indices: list[int],
        packed_set_indices: list[int],
    ):
        self.rqid = None
        reset_message = ClientRequest(
            reset=ResetRequest(
                username=self.username,
                species_indices=species_indices,
                packed_set_indices=packed_set_indices,
                smogon_format=f"gen{self.generation}_ou_all_formats",
                game_id=self.game_id,
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
        max_trajectory_length: int = 6,
        min_trajectory_length: int = 1,
    ):
        if num_team_members < 1 or num_team_members > 6:
            raise ValueError("num_team_members must be between 1 and 6")
        if max_trajectory_length < min_trajectory_length:
            raise ValueError(
                "max_trajectory_length must be greater than or equal to min_trajectory_length"
            )

        self._smogon_format = smogon_format
        self._generation = generation
        self._num_team_members = num_team_members
        self._max_trajectory_length = max_trajectory_length
        self._min_trajectory_length = min_trajectory_length

        self.species_probs = jnp.asarray(
            HUMAN_SPECIES_COUNTS[generation][smogon_format.replace("_all_formats", "")]
        )
        self.teammate_probs = jnp.asarray(
            HUMAN_TEAMMATE_COUNTS[generation][smogon_format.replace("_all_formats", "")]
        )

        with open(f"data/data/gen{generation}/{smogon_format}.json", "r") as f:
            packed_sets = json.load(f)
        self.packed_set_mask = jnp.array(list(len(v) for v in packed_sets.values())) > 0

        self.duplicate_masks = ~MASKS[generation]["duplicate"]

        self.start_mask = (
            SET_MASK[generation][smogon_format].any(axis=-1) & self.packed_set_mask
        )

        self.ex = get_ex_builder_step()

    def reset(self) -> BuilderActorInput:
        self.state = self._reset()
        return self.state

    def step(self, agent_output: BuilderAgentOutput) -> BuilderActorInput:
        if self.state.env.done.item():
            return self.state
        self.state = self._step(
            species_token=agent_output.actor_output.species_head.action_index,
            packed_set_token=agent_output.actor_output.packed_set_head.action_index,
            state=self.state,
        )
        return self.state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset(self) -> BuilderActorInput:
        species_mask = self.start_mask

        species_tokens = (
            jnp.ones((self._num_team_members,), dtype=jnp.int32)
            * SpeciesEnum.SPECIES_ENUM___UNK
        )
        packed_set_tokens = jnp.zeros_like(species_tokens)

        start_target = self.species_probs * species_mask
        start_target = start_target / (start_target.sum() + 1e-8)

        return BuilderActorInput(
            env=BuilderEnvOutput(
                species_mask=species_mask,
                done=jnp.array(0, dtype=jnp.bool),
                ts=jnp.array(0, dtype=jnp.int32),
                cum_teammate_reward=jnp.array(0, dtype=jnp.float32),
                cum_species_reward=jnp.array(0, dtype=jnp.float32),
                target_species_probs=start_target,
            ),
            history=BuilderHistoryOutput(
                species_tokens=species_tokens,
                packed_set_tokens=packed_set_tokens,
            ),
        )

    def _calculate_species_rewards(self, species_tokens: jax.Array):
        return jnp.take(self.species_probs, species_tokens, axis=0).mean(
            where=species_tokens != SpeciesEnum.SPECIES_ENUM___UNK
        )

    def _calculate_teammate_rewards(self, team_tokens: jax.Array):
        pairwise_rewards = jnp.take(
            jnp.take(self.teammate_probs, team_tokens, axis=0),
            team_tokens,
            axis=1,
        )

        not_unk_mask = team_tokens != SpeciesEnum.SPECIES_ENUM___UNK
        mask = jnp.outer(not_unk_mask, not_unk_mask)
        mask = jnp.clip(jnp.tril(mask), 0, 1) - jnp.eye(self._num_team_members)

        mask_sum = mask.sum().clip(min=1)
        teammate_reward = jnp.where(mask, pairwise_rewards, 0).sum() / mask_sum

        return teammate_reward

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(
        self,
        species_token: jax.Array,
        packed_set_token: jax.Array,
        state: BuilderActorInput,
    ):
        ts = state.env.ts
        next_ts = ts + 1

        traj_over = next_ts >= self._max_trajectory_length
        done = state.env.done | traj_over

        selection_oh = jax.nn.one_hot(
            ts, num_classes=self._num_team_members, dtype=jnp.bool_
        )
        species_tokens = jnp.where(
            selection_oh, species_token, state.history.species_tokens
        )
        species_mask = (
            jnp.take(self.duplicate_masks, species_tokens, axis=0).all(axis=0)
            & self.packed_set_mask
        )
        packed_set_tokens = jnp.where(
            selection_oh, packed_set_token, state.history.packed_set_tokens
        )

        completed_count = (species_tokens != SpeciesEnum.SPECIES_ENUM___UNK).sum()
        conditional = jnp.take(self.teammate_probs, species_tokens, axis=0)
        teammate_conditionals = conditional.sum(axis=0) / completed_count.clip(min=1)

        # So it favours teammates more as time goes on
        alpha = (next_ts / (state.history.species_tokens.shape[0] - 1)).clip(0, 1) ** (
            1 / 2
        )
        target_species_probs = (
            1 - alpha
        ) * self.species_probs + alpha * teammate_conditionals

        target_species_probs = jnp.where(species_mask, target_species_probs, 0)
        target_species_probs = target_species_probs / (
            target_species_probs.sum() + 1e-8
        )

        return BuilderActorInput(
            env=BuilderEnvOutput(
                species_mask=species_mask,
                done=done,
                ts=next_ts,
                # Prevent reward hacking by choosing common species
                cum_teammate_reward=self._calculate_teammate_rewards(
                    state.history.species_tokens
                ),
                cum_species_reward=self._calculate_species_rewards(species_tokens),
                target_species_probs=target_species_probs,
            ),
            history=BuilderHistoryOutput(
                species_tokens=species_tokens,
                packed_set_tokens=packed_set_tokens,
            ),
        )


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


class TeamBuilderEnvironmentFull:
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
        self.teratype_masks = jnp.load(
            f"data/data/gen{generation}/mask/teratypes_mask_{smogon_format}.npy"
        )

    def reset(self, key: jax.Array) -> BuilderFullActorInput:
        self.state = self._reset(key)
        return self.state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset(self, key: jax.Array) -> BuilderFullActorInput:
        order = generate_order(
            key, self._num_team_members, NUM_PACKED_TEAM_MEMBER_FEATURES
        )
        row = order // NUM_PACKED_TEAM_MEMBER_FEATURES
        col = order % NUM_PACKED_TEAM_MEMBER_FEATURES

        return BuilderFullActorInput(
            env=BuilderFullEnvOutput(
                species_mask=self.initial_species_mask,
                item_mask=jnp.ones((NUM_ITEMS,), dtype=jnp.bool),
                ability_mask=jnp.ones((NUM_ABILITIES,), dtype=jnp.bool),
                move_mask=jnp.ones((NUM_MOVES,), dtype=jnp.bool),
                ev_mask=jnp.ones((64,), dtype=jnp.bool),
                teratype_mask=jnp.ones((NUM_TYPECHART,), dtype=jnp.bool),
                ts=jnp.array(0, dtype=jnp.int32),
                done=jnp.array(0, dtype=jnp.bool),
            ),
            history=BuilderFullHistoryOutput(
                packed_team_member_tokens=jnp.zeros(
                    (self._num_team_members, NUM_PACKED_TEAM_MEMBER_FEATURES),
                    dtype=jnp.int32,
                ),
                order=order,
                row=row,
                col=col,
            ),
        )

    def step(self, token_index: int) -> BuilderFullActorInput:
        if self.state.env.done.item():
            return self.state
        self.state = self._step(
            token_index=token_index,
            state=self.state,
        )
        return self.state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(
        self, token_index: int, state: BuilderFullActorInput
    ) -> BuilderFullActorInput:

        ts = state.env.ts
        next_ts = ts + 1

        row = state.history.row[ts]
        col = state.history.col[ts]
        new_history = state.history.packed_team_member_tokens.at[row, col].set(
            token_index
        )

        next_row = state.history.row[next_ts]
        next_col = state.history.col[next_ts]

        current_species = new_history[
            next_row, PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__SPECIES
        ]
        current_moves = new_history[
            next_row,
            PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__MOVEID0 : PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__MOVEID3
            + 1,
        ]
        current_evs = new_history[
            next_row,
            PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__EV_HP : PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__EV_SPE
            + 1,
        ]

        move_mask = (
            jnp.take(self.move_masks, current_species, axis=0)
            .at[current_moves]
            .set(False)
        )

        ev_col = (
            next_col - PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__EV_HP
        ).clip(min=0, max=6)
        evs_sum = current_evs.sum()
        evs_remaining = 127 - evs_sum
        evs_mask = jnp.take(self.ev_masks, current_species, axis=0)[ev_col] & (
            jnp.arange(64) <= evs_remaining
        )
        # If no EVs have been allocated, all EV options should be available
        evs_mask = evs_mask + (evs_mask.sum(keepdims=True) == 0)

        next_species_mask = state.env.species_mask.at[current_species].set(False)

        return BuilderFullActorInput(
            env=BuilderFullEnvOutput(
                species_mask=next_species_mask,
                item_mask=jnp.take(self.item_masks, current_species, axis=0),
                ability_mask=jnp.take(self.ability_masks, current_species, axis=0),
                move_mask=move_mask,
                ev_mask=evs_mask,
                teratype_mask=jnp.take(self.teratype_masks, current_species, axis=0),
                ts=next_ts,
                done=state.env.done,
            ),
            history=BuilderFullHistoryOutput(
                packed_team_member_tokens=new_history,
                order=state.history.order,
                row=state.history.row,
                col=state.history.col,
            ),
        )


def mask_to_prob(mask: jax.Array) -> jax.Array:
    masked_probs = jnp.where(mask, 1.0, 0.0)
    return masked_probs / (masked_probs.sum() + 1e-8)


jnp.set_printoptions(linewidth=120)


def main():
    env = TeamBuilderEnvironmentFull(generation=9, smogon_format="ou")

    for s in range(64):
        key = jax.random.PRNGKey(s)

        state = env.reset(key)
        print(state.history.packed_team_member_tokens)

        for i in range(state.history.order.shape[-1]):
            subkey, key = jax.random.split(key)

            current_col = state.history.col[i].item()
            token_index = 0

            if (
                current_col
                == PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__SPECIES
            ):
                token_index = jax.random.choice(
                    subkey, NUM_SPECIES, p=mask_to_prob(state.env.species_mask)
                )
            elif (
                current_col == PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__ITEM
            ):
                token_index = jax.random.choice(
                    subkey, NUM_ITEMS, p=mask_to_prob(state.env.item_mask)
                )
            elif (
                current_col
                == PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__ABILITY
            ):
                token_index = jax.random.choice(
                    subkey, NUM_ABILITIES, p=mask_to_prob(state.env.ability_mask)
                )
            elif current_col in [
                PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__MOVEID0,
                PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__MOVEID1,
                PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__MOVEID2,
                PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__MOVEID3,
            ]:
                token_index = jax.random.choice(
                    subkey, NUM_MOVES, p=mask_to_prob(state.env.move_mask)
                )
                print(token_index)
            elif current_col in [
                PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__EV_HP,
                PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__EV_ATK,
                PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__EV_DEF,
                PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__EV_SPA,
                PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__EV_SPD,
                PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__EV_SPE,
            ]:
                token_index = jax.random.choice(
                    subkey, 64, p=mask_to_prob(state.env.ev_mask)
                )
            elif (
                current_col
                == PackedTeamMemberFeature.PACKED_TEAM_MEMBER_FEATURE__TERA_TYPE
            ):
                token_index = jax.random.choice(
                    subkey, NUM_TYPECHART, p=mask_to_prob(state.env.teratype_mask)
                )

            state = env.step(token_index=token_index)
            print(state.history.packed_team_member_tokens)
        print(state.history.packed_team_member_tokens)


if __name__ == "__main__":
    # with jax.disable_jit():
    main()
