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
from rl.environment.protos.service_pb2 import Action, ActionEnum
from rl.environment.utils import clip_history, split_rng
from rl.learner.league import MAIN_KEY, pfsp
from rl.learner.learner import Learner
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


class Actor:
    """Manages the state of a single agent/environment interaction loop."""

    def __init__(
        self,
        agent: Agent,
        env: SinglePlayerSyncEnvironment,
        unroll_length: int,
        learner: Learner,
        rng_seed: int = 42,
    ):
        self._agent = agent
        self._player_env = env
        self._builder_env = TeamBuilderEnvironment(
            generation=env.generation, smogon_format="ou_all_formats"
        )
        self._unroll_length = unroll_length
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
            action=ACTION_MAPPING[
                agent_output.actor_output.action_head.action_index.item()
            ],
            wildcard=agent_output.actor_output.wildcard_head.action_index.item(),
        )

    def unroll(
        self,
        rng_key: jax.Array,
        player_frame_count: int,
        builder_frame_count: int,
        player_params: Params,
        builder_params: Params,
    ) -> Trajectory:
        """Run unroll_length agent/environment steps, returning the trajectory."""
        builder_key, player_key = split_rng(rng_key, 2)
        builder_unroll_length = self._builder_env._max_trajectory_length + 1

        builder_subkeys = split_rng(builder_key, builder_unroll_length)
        player_subkeys = split_rng(player_key, self._unroll_length)

        build_traj = []

        # Reset the builder environment.
        builder_actor_input = self._builder_env.reset()

        # Rollout the builder environment.
        for builder_step_index in range(builder_subkeys.shape[0]):
            builder_agent_output = self._agent.step_builder(
                builder_subkeys[builder_step_index],
                builder_params,
                builder_actor_input,
            )
            builder_transition = BuilderTransition(
                env_output=builder_actor_input.env,
                agent_output=builder_agent_output,
            )
            build_traj.append(builder_transition)
            builder_actor_input = self._builder_env.step(builder_agent_output)
            # Swap when we break since we want a continuous trajectory
            if builder_actor_input.env.done.item():
                break

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
        player_actor_input = self._player_env.reset(
            builder_actor_input.history.species_tokens.reshape(-1).tolist(),
            builder_actor_input.history.packed_set_tokens.reshape(-1).tolist(),
        )

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
            builder_history=builder_actor_input.history,
            player_transitions=player_trajectory,
            player_history=player_actor_input.history,
        )

        return promote_map(trajectory)

    def split_rng(self) -> jax.Array:
        self._rng_key, subkey = split_rng(self._rng_key)
        return subkey

    def set_current_ckpt(self, ckpt: int):
        self._player_env._set_current_ckpt(ckpt)

    def set_opponent_ckpt(self, ckpt: int):
        self._player_env._set_opponent_ckpt(ckpt)

    def reset_ckpts(self):
        self._player_env._reset_ckpts()

    def unroll_and_push(self, params_container: ParamsContainer, do_push: bool = True):
        """Run one unroll and send trajectory to learner."""
        player_params = jax.device_put(params_container.player_params)
        builder_params = jax.device_put(params_container.builder_params)
        subkey = self.split_rng()
        act_out = self.unroll(
            rng_key=subkey,
            player_frame_count=params_container.player_frame_count,
            builder_frame_count=params_container.builder_frame_count,
            player_params=player_params,
            builder_params=builder_params,
        )
        self.reset_ckpts()

        if self._player_env.username.startswith("train") and do_push:
            self._learner.enqueue_traj(act_out)
        return act_out

    def pull_main_player(self) -> ParamsContainer:
        league = self._learner.league
        return league.get_main_player()

    def _pfsp_branch(self) -> ParamsContainer | None:
        historical = [
            player
            for player in self._learner.league.players.values()
            if player.step_count != MAIN_KEY
        ]
        if not historical:  # No historical players to play against
            return None

        main_player = self.pull_main_player()
        win_rates = self._learner.league.get_winrate((main_player, historical))
        pick_idx = np.random.choice(
            len(historical), p=pfsp(win_rates, weighting="squared")
        )
        return historical[pick_idx]

    def get_match(self) -> tuple[ParamsContainer, bool]:
        coin_toss = np.random.random()

        # Make sure you can beat the League (PFSP)
        # We only store trajectories from the the perspective of the main player,
        # so we need to oversample playing against it such that the proportion of
        # games played against it is 50%.
        if coin_toss < 0.5:
            opponent = self._pfsp_branch()
            if opponent is not None:  # Found a historical opponent
                return opponent, False

        return self.pull_main_player(), True

    def update_player_league_stats(
        self, sender: ParamsContainer, receiver: ParamsContainer, trajectory: Trajectory
    ):
        """Update league stats based on trajectory outcome."""
        payoff = trajectory.player_transitions.env_output.win_reward[-1]
        self._learner.league.update_payoff(sender, receiver, payoff)
