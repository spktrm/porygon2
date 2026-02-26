import jax
import numpy as np

from rl.actor.agent import Agent
from rl.environment.data import (
    CAT_VF_SUPPORT,
    NUM_ABILITIES,
    NUM_GENDERS,
    NUM_ITEMS,
    NUM_MOVES,
    NUM_NATURES,
    NUM_PACKED_SET_FEATURES,
    NUM_SPECIES,
    NUM_TYPECHART,
)
from rl.environment.env import SinglePlayerSyncEnvironment
from rl.environment.interfaces import (
    BuilderActorInput,
    BuilderEnvOutput,
    BuilderHistoryOutput,
    BuilderTransition,
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

    def _run_edit_steps(
        self,
        rng_key: jax.Array,
        builder_params: Params,
        builder_history: BuilderHistoryOutput,
    ) -> tuple[list[BuilderTransition], np.ndarray]:
        """Run builder edit steps for any team members with invalid teratypes.

        Uses bidirectional attention (is_edit=True) so each edited token can
        attend to the full already-placed team context when selecting a
        replacement value.

        Returns a list of BuilderTransition items (one per invalid member) and
        the updated packed_team_member_tokens array.
        """
        team_tokens = np.array(builder_history.packed_team_member_tokens)
        order = np.array(builder_history.order)

        edit_traj: list[BuilderTransition] = []
        # 6 team members; pre-split rng keys to index by member position.
        edit_rng_keys = split_rng(rng_key, 6)

        for m in range(6):
            flat_idx = (
                m * NUM_PACKED_SET_FEATURES
                + PackedSetFeature.PACKED_SET_FEATURE__TERATYPE
            )
            if team_tokens[flat_idx] != 0:
                continue  # Teratype already valid; no edit needed.

            # Find where in the builder ordering this token was originally placed.
            # generate_order always includes every non-UNSPECIFIED slot, so
            # flat_idx (teratype, index 17 in each member block) must be present.
            matches = np.where(order == flat_idx)[0]
            if len(matches) == 0:
                continue  # Defensive: position not found; skip this member.
            k = int(matches[0])

            # Build a broad teratype mask: all types are valid except UNSPECIFIED (0).
            teratype_mask = np.ones(NUM_TYPECHART, dtype=np.bool_)
            teratype_mask[0] = False

            # Construct the edit step input.  The history carries the full
            # (possibly partially-invalid) team so the bidirectional encoder can
            # use all surrounding context when making the edit decision.
            edit_actor_input = BuilderActorInput(
                env=BuilderEnvOutput(
                    species_mask=np.ones(NUM_SPECIES, dtype=np.bool_),
                    item_mask=np.ones(NUM_ITEMS, dtype=np.bool_),
                    ability_mask=np.ones(NUM_ABILITIES, dtype=np.bool_),
                    move_mask=np.ones(NUM_MOVES, dtype=np.bool_),
                    ev_mask=np.ones(64, dtype=np.bool_),
                    teratype_mask=teratype_mask,
                    nature_mask=np.ones(NUM_NATURES, dtype=np.bool_),
                    gender_mask=np.ones(NUM_GENDERS, dtype=np.bool_),
                    done=np.array(False, dtype=np.bool_),
                    ts=np.array(k, dtype=np.int32),
                    ev_reward=np.array(0.0, dtype=np.float32),
                    species_reward=np.array(0.0, dtype=np.float32),
                    curr_order=np.array(flat_idx, dtype=np.int32),
                    curr_attribute=np.array(
                        PackedSetFeature.PACKED_SET_FEATURE__TERATYPE,
                        dtype=np.int32,
                    ),
                    curr_position=np.array(m, dtype=np.int32),
                    is_edit=np.array(True, dtype=np.bool_),
                ),
                history=BuilderHistoryOutput(
                    packed_team_member_tokens=team_tokens,
                    order=order,
                    member_position=builder_history.member_position,
                    member_attribute=builder_history.member_attribute,
                ),
            )

            edit_agent_output = self._agent.step_builder(
                edit_rng_keys[m],
                builder_params,
                edit_actor_input,
            )

            # Apply the selected teratype and continue.
            new_teratype = int(
                edit_agent_output.actor_output.action_head.action_index.item()
            )
            team_tokens[flat_idx] = new_teratype

            edit_traj.append(
                BuilderTransition(
                    env_output=edit_actor_input.env,
                    agent_output=edit_agent_output,
                )
            )

        return edit_traj, team_tokens

    def unroll(
        self, rng_key: jax.Array, player_params: Params, builder_params: Params
    ) -> Trajectory:
        """Run unroll_length agent/environment steps, returning the trajectory."""

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

        # Split RNG: one key for edit steps, remaining for player steps.
        rng_key, edit_rng = split_rng(rng_key)
        player_subkeys = split_rng(rng_key, self._unroll_length)

        # Run edit steps for any team members with invalid teratypes before
        # starting the player game.  Edit transitions are appended to the
        # builder trajectory so the builder model is trained on them too.
        edit_transitions, team_tokens = self._run_edit_steps(
            edit_rng, builder_params, builder_history
        )

        if edit_transitions:
            edit_stacked: BuilderTransition = jax.tree.map(
                lambda *xs: np.stack(xs), *edit_transitions
            )
            builder_trajectory = jax.tree.map(
                lambda a, b: np.concatenate([a, b], axis=0),
                builder_trajectory,
                edit_stacked,
            )
            builder_history = builder_history.replace(
                packed_team_member_tokens=team_tokens
            )

        player_actor_input = self._env.reset(team_tokens.reshape(-1).tolist())

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
        builder_params = jax.device_put(params_container.builder_params)
        subkey = self.split_rng()

        act_out = self.unroll(
            rng_key=subkey,
            player_params=player_params,
            builder_params=builder_params,
        )
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
