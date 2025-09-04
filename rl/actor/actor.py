import queue

import jax
import numpy as np

from rl.actor.agent import Agent
from rl.environment.env import SinglePlayerSyncEnvironment, TeamBuilderEnvironment
from rl.environment.interfaces import (
    BuilderTransition,
    PlayerActorInput,
    PlayerAgentOutput,
    PlayerTransition,
    Trajectory,
)
from rl.environment.protos.features_pb2 import ActionType
from rl.environment.protos.service_pb2 import Action
from rl.environment.utils import clip_history
from rl.learner.learner import Learner
from rl.model.utils import Params

ACTION_TYPE_MAPPING = {
    0: ActionType.ACTION_TYPE__MOVE,
    1: ActionType.ACTION_TYPE__SWITCH,
    2: ActionType.ACTION_TYPE__TEAMPREVIEW,
}


class Actor:
    """Manages the state of a single agent/environment interaction loop."""

    def __init__(
        self,
        agent: Agent,
        env: SinglePlayerSyncEnvironment,
        unroll_length: int,
        learner: Learner,
        queue: queue.Queue | None = None,
        rng_seed: int = 42,
    ):
        self._agent = agent
        self._player_env = env
        self._builder_env = TeamBuilderEnvironment(env.generation)
        self._unroll_length = unroll_length
        self._queue = queue
        self._learner = learner
        self._rng_key = jax.random.key(rng_seed)

    def clip_actor_history(self, timestep: PlayerActorInput):
        return PlayerActorInput(
            env=timestep.env,
            history=clip_history(timestep.history, resolution=128),
        )

    def player_agent_output_to_action(self, agent_output: PlayerAgentOutput):
        """Post-processes the actor step to ensure it has the correct shape."""
        return Action(
            action_type=ACTION_TYPE_MAPPING[
                agent_output.actor_output.action_type_head.action_index.item()
            ],
            move_slot=agent_output.actor_output.move_head.action_index.item(),
            switch_slot=agent_output.actor_output.switch_head.action_index.item(),
            wildcard_slot=agent_output.actor_output.wildcard_head.action_index.item(),
        )

    def unroll(
        self,
        rng_key: jax.Array,
        frame_count: int,
        player_params: Params,
        builder_params: Params,
    ):
        """Run unroll_length agent/environment steps, returning the trajectory."""
        builder_key, player_key = jax.random.split(rng_key)
        builder_unroll_length = self._builder_env.max_ts + 1

        builder_subkeys = jax.random.split(builder_key, builder_unroll_length)
        player_subkeys = jax.random.split(player_key, self._unroll_length)

        build_traj = []

        # Reset the builder environment.
        builder_actor_input = self._builder_env.reset()
        # Rollout the builder environment.
        for builder_step_index in range(builder_subkeys.shape[0]):
            builder_agent_output = self._agent.step_builder(
                builder_subkeys[builder_step_index], builder_params, builder_actor_input
            )
            builder_transition = BuilderTransition(
                env_output=builder_actor_input.env,
                agent_output=builder_agent_output,
            )
            build_traj.append(builder_transition)
            if builder_actor_input.env.done.item():
                break
            builder_actor_input = self._builder_env.step(builder_agent_output)

        if len(build_traj) < builder_unroll_length:
            build_traj += [builder_transition] * (
                builder_unroll_length - len(build_traj)
            )

        # Pack the trajectory and reset parent state.
        builder_trajectory = jax.device_get(build_traj)
        builder_trajectory: BuilderTransition = jax.tree.map(
            lambda *xs: np.stack(xs), *builder_trajectory
        )

        player_traj = []

        # Reset the player environment.
        set_tokens = builder_actor_input.env.packed_set_tokens.reshape(-1).tolist()
        player_actor_input = self._player_env.reset(set_tokens)

        # Rollout the player environment.
        for player_step_index in range(player_subkeys.shape[0]):
            player_actor_input_clipped = self.clip_actor_history(player_actor_input)
            player_agent_output = self._agent.step_player(
                player_subkeys[player_step_index],
                player_params,
                player_actor_input_clipped,
            )
            player_transition = PlayerTransition(
                env_output=player_actor_input_clipped.env,
                agent_output=player_agent_output,
            )
            player_traj.append(player_transition)
            if player_actor_input_clipped.env.done.item():
                break

            action = self.player_agent_output_to_action(player_agent_output)
            player_actor_input = self._player_env.step(action)

        if len(player_traj) < self._unroll_length:
            player_traj += [player_transition] * (
                self._unroll_length - len(player_traj)
            )

        # Pack the trajectory and reset parent state.
        player_trajectory = jax.device_get(player_traj)
        player_trajectory: PlayerTransition = jax.tree.map(
            lambda *xs: np.stack(xs), *player_trajectory
        )

        trajectory = Trajectory(
            builder_transitions=builder_trajectory,
            player_transitions=player_trajectory,
            # builder_history=builder_actor_input.history,
            player_history=player_actor_input.history,
        )

        return trajectory

    def split_rng(self) -> jax.Array:
        self._rng_key, subkey = jax.random.split(self._rng_key)
        return subkey

    def unroll_and_push(
        self, frame_count: int, player_params: Params, builder_params: Params
    ):
        """Run one unroll and send trajectory to learner."""
        player_params = jax.device_put(player_params)
        builder_params = jax.device_put(builder_params)
        subkey = self.split_rng()
        act_out = self.unroll(
            rng_key=subkey,
            frame_count=frame_count,
            player_params=player_params,
            builder_params=builder_params,
        )
        if self._queue is not None:
            self._queue.put(act_out)

    def pull_params(self):
        with self._learner.param_lock:
            step_count, player_params, builder_params = self._learner.params_for_actor
            return int(step_count), player_params, builder_params
