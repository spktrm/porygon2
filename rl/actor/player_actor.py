import jax
import numpy as np

from rl.actor.agent import Agent
from rl.environment.data import CAT_VF_SUPPORT, NUM_PACKED_SET_FEATURES
from rl.environment.env import SinglePlayerSyncEnvironment
from rl.environment.interfaces import (
    PlayerActorInput,
    PlayerAgentOutput,
    PlayerTransition,
    Trajectory,
)
from rl.environment.protos.features_pb2 import PackedSetFeature
from rl.environment.protos.service_pb2 import Action
from rl.environment.utils import (
    NUM_PACKED_SET_FEATURES,
    clip_history,
    clip_packed_history,
    split_rng,
)
from rl.learner.league import MAIN_KEY, pfsp
from rl.learner.learner import Learner
from rl.model.builder_model import get_packed_team_string
from rl.model.utils import Params, ParamsContainer, promote_map


class PlayerActor:
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
        self._env = env
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

    def unroll(self, rng_key: jax.Array, player_params: Params) -> Trajectory:
        """Run unroll_length agent/environment steps, returning the trajectory."""

        player_subkeys = split_rng(rng_key, self._unroll_length)

        sample_cond = self._learner.team_store._sample_cv
        with sample_cond:
            sample_cond.wait_for(self._learner.team_store.ready_to_sample)
            builder_trajectory, builder_history = (
                self._learner.team_store.sample_trajectory()
            )

        add_cond = self._learner.team_store._add_cv
        with add_cond:
            add_cond.notify_all()

        player_traj = []

        # Reset the player environment.
        team_tokens = builder_history.packed_team_member_tokens
        if np.any(team_tokens[..., PackedSetFeature.PACKED_SET_FEATURE__TERATYPE] == 0):
            raise ValueError(
                get_packed_team_string(team_tokens.reshape(-1, NUM_PACKED_SET_FEATURES))
            )

        player_actor_input = self._env.reset(team_tokens.reshape(-1).tolist())

        # Sample a DIAYN skill for this episode
        skill_id = np.int32(np.random.randint(0, self._learner.config.num_skills))
        player_actor_input = player_actor_input.replace(
            env=player_actor_input.env.replace(skill_id=skill_id)
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
            player_actor_input = self._env.step(action)
            # Preserve skill_id across steps
            player_actor_input = player_actor_input.replace(
                env=player_actor_input.env.replace(skill_id=skill_id)
            )

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
            builder_history=builder_history,
            player_transitions=player_trajectory,
            player_packed_history=player_actor_input.packed_history,
            player_history=player_actor_input.history,
        )

        return promote_map(trajectory)

    def split_rng(self) -> jax.Array:
        self._rng_key, subkey = split_rng(self._rng_key)
        return subkey

    def set_game_id(self, game_id: int):
        self._env._set_game_id(game_id)

    def reset_game_id(self):
        self._env._reset_game_id()

    def unroll_and_push(self, params_container: ParamsContainer, do_push: bool = True):
        """Run one unroll and send trajectory to learner."""
        player_params = jax.device_put(params_container.player_params)
        subkey = self.split_rng()

        act_out = self.unroll(rng_key=subkey, player_params=player_params)
        self.reset_game_id()

        if self._env.username.startswith("train") and do_push:
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
        payoff = (
            trajectory.player_transitions.env_output.win_reward[-1] @ CAT_VF_SUPPORT
        )
        self._learner.league.update_payoff(sender, receiver, payoff)
