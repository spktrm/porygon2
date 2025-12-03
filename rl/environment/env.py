import functools
import json

import jax
import jax.numpy as jnp
from websockets.sync.client import connect

from rl.environment.data import (
    HUMAN_SPECIES_COUNTS,
    HUMAN_TEAMMATE_COUNTS,
    MASKS,
    SET_MASK,
)
from rl.environment.interfaces import (
    BuilderActorInput,
    BuilderAgentOutput,
    BuilderEnvOutput,
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
        self.current_ckpt = None
        self.opponent_ckpt = None

        self.websocket = connect(
            SERVER_URI,
            additional_headers={"username": username},
        )
        self.generation = generation
        self.metgame_token = None

    def _set_current_ckpt(self, ckpt: str):
        self.current_ckpt = ckpt

    def _set_opponent_ckpt(self, ckpt: str):
        self.opponent_ckpt = ckpt

    def _reset_ckpts(self):
        self.current_ckpt = None
        self.opponent_ckpt = None

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
                current_ckpt=self.current_ckpt,
                opponent_ckpt=self.opponent_ckpt,
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
            step=StepRequest(actions=[action], username=self.username, rqid=self.rqid),
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

        return BuilderActorInput(
            env=BuilderEnvOutput(
                species_mask=species_mask,
                done=jnp.array(0, dtype=jnp.bool),
                ts=jnp.array(0, dtype=jnp.int32),
                cum_teammate_reward=jnp.array(0, dtype=jnp.float32),
                cum_species_reward=jnp.array(0, dtype=jnp.float32),
                target_species_log_probs=jnp.log(
                    (self.species_probs / self.species_probs.sum()) + 1e-8
                ),
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
        teammate_conditionals = jnp.where(species_mask, teammate_conditionals, 0)

        # So it favours teammates more as time goes on
        alpha = (next_ts / (state.history.species_tokens.shape[0] - 1)).clip(0, 1) ** (
            1 / 2
        )
        target_species_probs = (
            1 - alpha
        ) * self.species_probs + alpha * teammate_conditionals
        target_species_log_probs = jnp.log(
            (target_species_probs / target_species_probs.sum().clip(min=1e-6)) + 1e-8
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
                target_species_log_probs=target_species_log_probs,
            ),
            history=BuilderHistoryOutput(
                species_tokens=species_tokens,
                packed_set_tokens=packed_set_tokens,
            ),
        )
