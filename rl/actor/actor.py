import jax
import numpy as np

from rl.actor.agent import BuilderAgent, PlayerAgent
from rl.environment.env import SinglePlayerSyncEnvironment, TeamBuilderEnvironment
from rl.environment.interfaces import (
    BuilderTrajectory,
    BuilderTransition,
    PlayerActorInput,
    PlayerAgentOutput,
    PlayerTrajectory,
    PlayerTransition,
)
from rl.environment.protos.service_pb2 import Action, ActionEnum
from rl.environment.utils import clip_history, split_rng
from rl.learner.buffer import BuilderMetadata, BuilderReplayBuffer
from rl.model.utils import Params, ParamsContainer, promote_map

ACTION_MAPPING = {
    0: ActionEnum.ACTION_ENUM__MOVE_1_TARGET_NA,
    1: ActionEnum.ACTION_ENUM__MOVE_2_TARGET_NA,
    2: ActionEnum.ACTION_ENUM__MOVE_3_TARGET_NA,
    3: ActionEnum.ACTION_ENUM__MOVE_4_TARGET_NA,
    4: ActionEnum.ACTION_ENUM__SWITCH_1,
    5: ActionEnum.ACTION_ENUM__SWITCH_2,
    6: ActionEnum.ACTION_ENUM__SWITCH_3,
    7: ActionEnum.ACTION_ENUM__SWITCH_4,
    8: ActionEnum.ACTION_ENUM__SWITCH_5,
    9: ActionEnum.ACTION_ENUM__SWITCH_6,
}


class BuilderActor:
    """Manages the state of a single agent/environment interaction loop."""

    def __init__(
        self,
        agent: BuilderAgent,
        env: TeamBuilderEnvironment,
        unroll_length: int,
        rng_seed: int = 42,
    ):
        self._agent = agent
        self._env = env
        self._unroll_length = unroll_length
        self._rng_key = jax.random.key(rng_seed)

    def _unroll(
        self, rng_key: jax.Array, frame_count: int, params: Params
    ) -> BuilderTrajectory:
        """Run unroll_length agent/environment steps, returning the trajectory."""
        builder_unroll_length = self._env._max_trajectory_length + 1
        builder_subkeys = split_rng(rng_key, builder_unroll_length)

        traj = []

        # Reset the builder environment.
        actor_input = self._env.reset()

        # Rollout the builder environment.
        for builder_step_index in range(builder_subkeys.shape[0]):
            agent_output = self._agent.step_builder(
                builder_subkeys[builder_step_index], params, actor_input
            )
            transition = BuilderTransition(
                env_output=actor_input.env, agent_output=agent_output
            )
            traj.append(transition)
            if actor_input.env.done.item():
                break
            actor_input = self._env.step(agent_output)

        if len(traj) < builder_unroll_length:
            traj += [transition] * (builder_unroll_length - len(traj))

        # Pack the trajectory and reset parent state.
        trajectory = jax.device_get(traj)
        trajectory: BuilderTransition = jax.tree.map(
            lambda *xs: np.stack(xs), *trajectory
        )

        trajectory = BuilderTrajectory(
            transitions=trajectory,
            history=actor_input.history,
        )

        return promote_map(trajectory)

    def split_rng(self) -> jax.Array:
        self._rng_key, subkey = split_rng(self._rng_key)
        return subkey

    def unroll(self, params_container: ParamsContainer):
        """Run one unroll and send trajectory to learner."""
        params = jax.device_put(params_container.builder_params)
        subkey = self.split_rng()
        act_out = self._unroll(
            rng_key=subkey,
            frame_count=params_container.frame_count,
            params=params,
        )
        return act_out


class PlayerActor:
    """Manages the state of a single agent/environment interaction loop."""

    def __init__(
        self,
        agent: PlayerAgent,
        env: SinglePlayerSyncEnvironment,
        builder_replay_buffer: BuilderReplayBuffer,
        unroll_length: int,
        rng_seed: int = 42,
    ):
        self._agent = agent
        self._env = env
        self._builder_replay_buffer = builder_replay_buffer
        self._unroll_length = unroll_length
        self._rng_key = jax.random.key(rng_seed)

    def clip_actor_history(self, timestep: PlayerActorInput):
        return PlayerActorInput(
            env=timestep.env,
            history=clip_history(timestep.history, resolution=128),
        )

    def player_agent_output_to_action(self, agent_output: PlayerAgentOutput):
        """Post-processes the actor step to ensure it has the correct shape."""
        return Action(
            action=ACTION_MAPPING[
                agent_output.actor_output.action_head.action_index.item()
            ],
            wildcard=agent_output.actor_output.wildcard_head.action_index.item(),
        )

    def _unroll(self, rng_key: jax.Array, frame_count: int, params: Params):
        """Run unroll_length agent/environment steps, returning the trajectory."""
        player_subkeys = split_rng(rng_key, self._unroll_length)

        player_traj = []

        # Sample a builder trajectory to condition on.
        builder_key, builder_trajectory, builder_metadata = (
            self._builder_replay_buffer.sample_for_player()
        )
        player_actor_input = self._env.reset(
            builder_trajectory.history.species_tokens.reshape(-1).tolist(),
            builder_trajectory.history.packed_set_tokens.reshape(-1).tolist(),
        )

        # Rollout the player environment.
        for player_step_index in range(player_subkeys.shape[0]):
            player_actor_input_clipped = self.clip_actor_history(player_actor_input)
            player_agent_output = self._agent.step_player(
                player_subkeys[player_step_index],
                params,
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
            player_actor_input = self._env.step(action)

        # Update winrate of team
        update_ema = 1e-2
        new_winrate = (
            (1 - update_ema) * builder_metadata.winrate
            + update_ema * player_transition.env_output.win_reward.item()
        )
        self._builder_replay_buffer.update(
            builder_key, BuilderMetadata(winrate=new_winrate)
        )

        if len(player_traj) < self._unroll_length:
            player_traj += [player_transition] * (
                self._unroll_length - len(player_traj)
            )

        # Pack the trajectory and reset parent state.
        player_traj = jax.device_get(player_traj)
        stacked: PlayerTransition = jax.tree.map(lambda *xs: np.stack(xs), *player_traj)

        trajectory = PlayerTrajectory(
            transitions=stacked, history=player_actor_input.history
        )

        return promote_map(trajectory)

    def split_rng(self) -> jax.Array:
        self._rng_key, subkey = split_rng(self._rng_key)
        return subkey

    def set_current_ckpt(self, ckpt: int):
        self._env._set_current_ckpt(ckpt)

    def set_opponent_ckpt(self, ckpt: int):
        self._env._set_opponent_ckpt(ckpt)

    def reset_ckpts(self):
        self._env._reset_ckpts()

    def unroll(self, params_container: ParamsContainer):
        params = jax.device_put(params_container.player_params)
        subkey = self.split_rng()
        act_out = self._unroll(
            rng_key=subkey,
            frame_count=params_container.frame_count,
            params=params,
        )
        self.reset_ckpts()
        return act_out
