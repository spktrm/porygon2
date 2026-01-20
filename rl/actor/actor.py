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
from rl.environment.protos.service_pb2 import Action
from rl.environment.utils import clip_history, clip_packed_history, split_rng
from rl.learner.league import pfsp
from rl.learner.learner import Learner
from rl.model.utils import Params, ParamsContainer, promote_map


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
            packed_history=clip_packed_history(timestep.packed_history, resolution=128),
            history=clip_history(timestep.history, resolution=128),
        )

    def player_agent_output_to_action(self, agent_output: PlayerAgentOutput):
        """Post-processes the actor step to ensure it has the correct shape."""
        return Action(
            src=agent_output.actor_output.action_head.src_index.item(),
            tgt=agent_output.actor_output.action_head.tgt_index.item(),
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
            player_packed_history=player_actor_input.packed_history,
            player_history=player_actor_input.history,
        )

        return promote_map(trajectory)

    def split_rng(self) -> jax.Array:
        self._rng_key, subkey = split_rng(self._rng_key)
        return subkey

    def set_current_ckpt(self, ckpt: str):
        self._player_env._set_current_ckpt(ckpt)

    def set_opponent_ckpt(self, ckpt: str):
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

    def get_current_player(self) -> ParamsContainer:
        return self._learner.get_current_player()

    def _pfsp_branch(self):
        historical = [
            player
            for player in self._learner.league.players.values()
            if player.player_type == "historical"
        ]
        if not historical:  # No historical players to play against
            return None

        current_player = self.get_current_player()
        win_rates = self._learner.league.get_winrate((current_player, historical))
        pick_idx: int = np.random.choice(
            len(historical), p=pfsp(win_rates, weighting="squared")
        )
        return historical[pick_idx], True

    def _selfplay_branch(self, opponent: ParamsContainer):
        # Play self-play match
        current_player = self.get_current_player()
        if self._learner.league.get_winrate((current_player, opponent)) > 0.3:
            return opponent, False

        # If opponent is too strong, look for a checkpoint
        # as curriculum
        historical = [
            player
            for player in self._learner.league.players.values()
            if player.player_type == "historical" and player.parent == opponent
        ]
        win_rates = self._learner.league.get_winrate((current_player, historical))
        pick_idx: int = np.random.choice(
            len(historical), p=pfsp(win_rates, weighting="squared")
        )
        return historical[pick_idx], True

    def _verification_branch(self, opponent: ParamsContainer):
        # Check exploitation
        exploiters = set(
            [
                player
                for player in self._learner.league.players.values()
                if player.player_type == "main_exploiter"
            ]
        )
        exp_historical = [
            player
            for player in self._learner.league.players.values()
            if player.player_type == "historical" and player.parent in exploiters
        ]
        current_player = self.get_current_player()
        win_rates = self._learner.league.get_winrate((current_player, exp_historical))
        if len(win_rates) and win_rates.min() < 0.3:
            pick_idx: int = np.random.choice(
                len(exp_historical), p=pfsp(win_rates, weighting="squared")
            )
            return exp_historical[pick_idx], True

        # Check forgetting
        historical = [
            player
            for player in self._learner.league.players.values()
            if player.player_type == "historical"
            and player.parent == opponent.get_key()
        ]
        win_rates = self._learner.league.get_winrate((current_player, historical))
        if len(win_rates) and win_rates.min() < 0.7:
            pick_idx: int = np.random.choice(
                len(historical), p=pfsp(win_rates, weighting="squared")
            )
            return historical[pick_idx], True

        return None

    def get_match(self):
        current_player = self.get_current_player()
        if current_player.player_type == "main_player":
            return self.get_main_player_match()
        elif current_player.player_type == "main_exploiter":
            return self.get_main_exploiter_match()
        elif current_player.player_type == "league_exploiter":
            return self.get_league_exploiter_match()
        else:
            raise ValueError(f"Unknown player type: {current_player.player_type}")

    def get_main_player_match(self):
        coin_toss = np.random.random()

        main_agents = [
            player
            for player in self._learner.league.players.values()
            if player.player_type == "main_player"
        ]
        opponent = main_agents[np.random.choice(len(main_agents))]

        # Make sure you can beat the League
        if coin_toss < 0.5:
            result = self._pfsp_branch()
            if result is None:
                return opponent, True
            return result

        # Verify if there are some rare players we omitted
        if coin_toss < 0.5 + 0.15:
            request = self._verification_branch(opponent)
            if request is not None:
                return request

        return self._selfplay_branch(opponent)

    def get_main_exploiter_match(self):
        main_agents = [
            player
            for player in self._learner.league.players.values()
            if player.player_type == "main_player"
        ]
        opponent = main_agents[np.random.choice(len(main_agents))]

        current_player = self.get_current_player()

        if self._learner.league.get_winrate((current_player, opponent)) > 0.1:
            return opponent, True

        historical = [
            player
            for player in self._learner.league.players.values()
            if player.player_type == "historical"
            and player.parent == opponent.get_key()
        ]
        win_rates = self._learner.league.get_winrate((current_player, historical))

        pick_idx: int = np.random.choice(
            len(historical), p=pfsp(win_rates, weighting="variance")
        )
        return historical[pick_idx], True

    def get_league_exploiter_match(self):
        historical = [player for player in self._learner.league.players.values()]
        current_player = self.get_current_player()
        win_rates = self._learner.league.get_winrate((current_player, historical))
        pick_idx: int = np.random.choice(
            len(historical), p=pfsp(win_rates, weighting="linear_capped")
        )
        return historical[pick_idx], True

    def update_player_league_stats(
        self, sender: ParamsContainer, receiver: ParamsContainer, trajectory: Trajectory
    ):
        """Update league stats based on trajectory outcome."""
        payoff = trajectory.player_transitions.env_output.win_reward[-1]
        self._learner.league.update_payoff(sender, receiver, payoff)
