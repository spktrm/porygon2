import jax
import numpy as np

from rl.environment.data import CAT_VF_SUPPORT
from rl.environment.env import ActorStopped, SinglePlayerSyncEnvironment
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
    clip_history_windows_tail,
    clip_packed_history,
    split_rng,
)
from rl.model.builder_model import get_packed_team_string
from rl.model.utils import Params, ParamsContainer
from rl.online.agent import Agent
from rl.online.guards import should_push_trajectory
from rl.online.inference import InferenceServer
from rl.online.league import LIVE_KEYS, MAIN_KEY, PlayerRef, pfsp
from rl.online.training import Learner


def chunk_spans(
    num_steps: int, chunk_length: int, game_done: bool
) -> list[tuple[int, int]]:
    """Inclusive (start, end) row spans splitting a ``num_steps``-step game
    into fixed-length chunks with a one-row overlap (stride
    chunk_length - 1): each span's end row is the next span's start row —
    the bootstrap-only row there, the trained row here. The final span may
    be shorter than chunk_length (the caller pads it); it exists only when
    the game actually ended (``game_done``) — a capped no-done game's
    trailing partial segment carries no outcome and is dropped."""
    stride = chunk_length - 1
    spans: list[tuple[int, int]] = []
    for chunk_index in range(1 + max(0, (num_steps - 2) // stride)):
        start = chunk_index * stride
        end = start + chunk_length - 1
        if end <= num_steps - 1:
            spans.append((start, end))
            if end == num_steps - 1:
                break
        else:
            if game_done:
                spans.append((start, num_steps - 1))
            break
    return spans


class PlayerActor:
    """Manages the state of a single agent/environment interaction loop."""

    def __init__(
        self,
        agent: Agent,
        env: SinglePlayerSyncEnvironment,
        unroll_length: int,
        learner: Learner,
        rng_seed: int = 42,
        is_eval: bool = False,
        inference_client: InferenceServer | None = None,
        pinned_opponent: ParamsContainer | None = None,
    ):
        self._agent = agent
        self._env = env
        # The env polls this while blocked on the game server so a game
        # whose other side has already unwound can't pin this thread.
        env.stop_check = self._stop_requested
        self._unroll_length = unroll_length
        self._learner = learner
        self._rng_key = jax.random.key(rng_seed)
        # When set, per-step inference goes through the shared batched
        # InferenceServer (rl/online/inference.py) instead of this actor's
        # own batch-1 Agent.step_player dispatch. Training actors get one;
        # eval actors deliberately don't (different sampling temperature
        # via eval_agent's HeadParams, and 3 low-volume threads aren't
        # worth a second server).
        self._inference_client = inference_client
        # Eval actors must never contribute to training data, nor consume the
        # builder replay buffer's reuse budget. This flag gates both.
        self._is_eval = is_eval
        # Best-response runs: when set, EVERY game is played against this
        # frozen container — get_match short-circuits before the
        # PFSP/verification/mirror split, so the pin is 100%, and the
        # opponent side is never trainable.
        self._pinned_opponent = pinned_opponent

    def clip_actor_history(self, timestep: PlayerActorInput, min_length: int = 64):
        return PlayerActorInput(
            env=timestep.env,
            packed_history=clip_packed_history(
                timestep.packed_history, min_length=min_length
            ),
            history=clip_history(timestep.history, min_length=min_length),
        )

    def player_agent_output_to_action(self, agent_output: PlayerAgentOutput):
        """Post-processes the actor step to ensure it has the correct shape."""
        return Action(cell=agent_output.actor_output.action_head.action_index.item())

    def _snapshot_window(self, actor_input: PlayerActorInput):
        """Fixed-length trailing history window as of ``actor_input``'s
        step — the burn-in-plus-chunk context stored with a chunk whose
        LAST row this step is. The recurrent history scan runs from h0
        over exactly such a trailing window at act time too (the service's
        getHistory windows to NUM_HISTORY the same way), so training on
        these windows matches acting with no stored-carry staleness."""
        history_length = self._learner.config.player_history_length
        history_window, packed_window = clip_history_windows_tail(
            actor_input.history, actor_input.packed_history, history_length
        )
        return packed_window, history_window

    def unroll(
        self,
        rng_key: jax.Array,
        player_params: Params | ParamsContainer,
    ) -> list[Trajectory]:
        """Run one full game (up to unroll_length steps) and return it as a
        list of fixed-length chunks of player_chunk_length transitions each.

        Chunks are start-aligned with a ONE-ROW overlap (stride
        player_chunk_length - 1): each chunk's final row is the next
        chunk's first row. The learner trains every row except a chunk's
        final one there (compute_player_targets masks it), and the final
        row supplies the value bootstrap at the cut — so every step of the
        game gets its policy-gradient signal exactly once. The game's
        terminal chunk is the only one carrying the outcome reward; a
        game truncated by the unroll_length cap without a done drops its
        trailing partial segment (rare, and those steps have no outcome
        signal to give).

        player_params is device-resident Params on the direct-Agent path,
        or the host ParamsContainer when routing through the batched
        InferenceServer (which owns device transfer) — see unroll_and_push."""

        player_subkeys = split_rng(rng_key, self._unroll_length)
        player_traj = []

        team_tokens = None
        builder_trajectory = ()
        builder_history = ()
        if self._learner.config.smogon_format != "randombattle":
            run_state = self._learner.run_state
            builder_replay = run_state.builder_replay
            sample_cond = builder_replay._sample_cv
            with sample_cond:
                # done-aware, like builder_actor.py's add-side wait and
                # Learner.enqueue_traj already are: once training is
                # shutting down no builder trajectory is ever coming, so a
                # bare ready_to_sample predicate blocked this thread forever
                # (_stop_workers notifies this CV precisely so
                # waiters re-check and see done).
                sample_cond.wait_for(
                    lambda: run_state.done or builder_replay.ready_to_sample()
                )
                if run_state.done:
                    raise ActorStopped(
                        "training stopped during " "builder-replay sample wait"
                    )
                # Eval samples teams read-only: it doesn't increment reuse
                # counts, so it can't evict builder trajectories that training
                # would otherwise consume.
                builder_trajectory, builder_history = builder_replay.sample_trajectory(
                    increment=not self._is_eval
                )

            add_cond = builder_replay._add_cv
            with add_cond:
                add_cond.notify_all()

            # Reset the player environment.
            team_tokens = builder_history.packed_team_member_tokens
            if np.any(
                team_tokens[..., PackedSetFeature.PACKED_SET_FEATURE__TERATYPE] == 0
            ):
                raise ValueError(
                    get_packed_team_string(
                        team_tokens.reshape(-1, NUM_PACKED_SET_FEATURES)
                    )
                )
            team_tokens = team_tokens.reshape(-1).tolist()

        player_actor_input = self._env.reset(team_tokens)

        chunk_length = self._learner.config.player_chunk_length
        stride = chunk_length - 1
        # Trailing history windows keyed by the step index they were taken
        # at — one per chunk boundary, plus the final step's.
        window_snapshots: dict[int, tuple] = {}

        # Rollout the player environment.
        for player_step_index in range(player_subkeys.shape[0]):
            if self._stop_requested():
                # Bail on the STEP boundary rather than playing the game
                # out: at shutdown this trajectory is going nowhere, and a
                # game can still have hundreds of steps to run.
                raise ActorStopped("training stopped mid-unroll")
            player_actor_input_clipped = self.clip_actor_history(player_actor_input)
            if self._inference_client is not None:
                # player_params is the host ParamsContainer on this path
                # (see unroll_and_push) — the server owns device transfer.
                player_agent_output = self._inference_client.step_player(
                    player_subkeys[player_step_index],
                    player_params,
                    player_actor_input_clipped,
                )
            else:
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
            if player_step_index > 0 and player_step_index % stride == 0:
                window_snapshots[player_step_index] = self._snapshot_window(
                    player_actor_input
                )
            if player_actor_input_clipped.env.done.item():
                break

            action = self.player_agent_output_to_action(player_agent_output)
            player_actor_input = self._env.step(action)

        player_traj = jax.device_get(player_traj)
        num_steps = len(player_traj)
        game_done = bool(np.asarray(player_traj[-1].env_output.done).item())
        # Completed-game side data for every chunk (Trajectory docstring):
        # the outcome is only knowable here, after the whole game ran.
        game_outcome = (
            float(np.asarray(player_traj[-1].env_output.win_reward) @ CAT_VF_SUPPORT)
            if game_done
            else float("nan")
        )
        final_window = self._snapshot_window(player_actor_input)

        def make_chunk(rows: list[PlayerTransition], window, start: int) -> Trajectory:
            if len(rows) < chunk_length:
                # Same padding convention as the pre-chunk whole-game path:
                # copies of the terminal step with done zeroed — cumsum-done
                # masking in the learner excludes them, and win_reward /
                # public_team survive at [-1] for the outcome consumers.
                padding_step = PlayerTransition(
                    env_output=rows[-1].env_output.replace(
                        done=np.zeros_like(rows[-1].env_output.done)
                    ),
                    agent_output=rows[-1].agent_output,
                )
                rows = rows + [padding_step] * (chunk_length - len(rows))
            packed_window, history_window = window
            return Trajectory(
                builder_transitions=builder_trajectory,
                builder_history=builder_history,
                player_transitions=jax.tree.map(lambda *xs: np.stack(xs), *rows),
                player_packed_history=packed_window,
                player_history=history_window,
                game_outcome=np.array([game_outcome], dtype=np.float32),
                game_length=np.array([num_steps], dtype=np.int32),
                game_step_offset=np.array([start], dtype=np.int32),
            )

        return [
            make_chunk(
                player_traj[start : end + 1],
                window_snapshots.get(end, final_window),
                start,
            )
            for start, end in chunk_spans(num_steps, chunk_length, game_done)
        ]

    def _stop_requested(self) -> bool:
        """True once training is shutting down. Reads run_state fresh
        every call — the bundle object is replaced on a rebuild."""
        return bool(self._learner.run_state.done)

    def split_rng(self) -> jax.Array:
        self._rng_key, subkey = split_rng(self._rng_key)
        return subkey

    def set_game_id(self, game_id: int):
        self._env._set_game_id(game_id)

    def reset_game_id(self):
        self._env._reset_game_id()

    def unroll_and_push(self, params_container: ParamsContainer, do_push: bool = True):
        """Run one unroll and send trajectory to learner."""
        if self._inference_client is not None:
            # The server owns device transfer behind a versioned cache, so
            # every actor playing the same params version shares ONE device
            # copy — the per-actor device_put below made 12 separate copies
            # of the identical live params, one per actor per game.
            player_params = params_container
        else:
            player_params = jax.device_put(params_container.player_params)
        subkey = self.split_rng()

        chunks = self.unroll(rng_key=subkey, player_params=player_params)
        self.reset_game_id()

        if should_push_trajectory(self._is_eval, do_push, self._env.username):
            for chunk in chunks:
                self._learner.enqueue_traj(chunk)
        # Callers consume whole-game facts off the return value
        # (win_reward[-1] payoff, public_team[-1] mon differential): the
        # last chunk's padding rows copy the terminal
        # step, so it serves all of them.
        return chunks[-1]

    def pull_own_player(self) -> ParamsContainer:
        """The live player's current params."""
        return self._learner.league.get_live(MAIN_KEY)

    def pull_main_player(self) -> ParamsContainer:
        """Live main's params. Resolves to the same container as
        pull_own_player (get_main_player is a MAIN_KEY-only alias for
        get_live), but its caller means it specifically: it weights against
        the player whose blind spots it is checking."""
        return self._learner.league.get_main_player()

    def _pfsp_branch(self) -> ParamsContainer | None:
        historical = [
            player
            for player in self._learner.league.players.values()
            if player.step_count not in LIVE_KEYS
        ]
        if not historical:  # No historical players to play against
            return None

        own_player = self.pull_own_player()
        win_rates = self._learner.league.get_winrate((own_player, historical))
        pick_idx = np.random.choice(
            len(historical), p=pfsp(win_rates, weighting="squared")
        )
        # Selection above is metadata-only; load params for just the chosen ref.
        return self._learner.league.materialize(historical[pick_idx])

    def _concerning_opponents(
        self, candidates: list[PlayerRef]
    ) -> tuple[list[PlayerRef], np.ndarray] | None:
        """Among ``candidates``, which are reliable, real weak spots right
        now — win-rate below exploit_ctrl_target with enough games to
        trust the reading (exploit_ctrl_min_games_per_opponent). Returns
        (players, win_rates) restricted to just those, or None if none
        qualify."""
        if not candidates:
            return None
        config = self._learner.config
        league = self._learner.league
        main_player = self.pull_main_player()
        win_rates = np.atleast_1d(league.get_winrate((main_player, candidates)))
        games = np.array(
            [
                league.games.get((MAIN_KEY, p.step_count), 0.0)
                + league.games.get((p.step_count, MAIN_KEY), 0.0)
                for p in candidates
            ]
        )
        concerning = (games >= config.exploit_ctrl_min_games_per_opponent) & (
            win_rates < config.exploit_ctrl_target
        )
        if not concerning.any():
            return None
        return (
            [p for p, c in zip(candidates, concerning) if c],
            win_rates[concerning],
        )

    def _verification_branch(self) -> ParamsContainer | None:
        """Forces extra games against any historical opponent that's
        currently a real, reliable weak spot — adapted from AlphaStar's
        MainPlayer._verification_branch (the public league-management
        pseudocode: github.com/chengyu2/learning_alpha_star/multiagent.py).

        AlphaStar scopes this check to snapshots descended from a
        MainExploiter; with no such subset in this league it checks all
        historical opponents. AlphaStar's
        separate "forgetting" check (a monotonic-suffix trick on win-rate
        history) is not ported — the threshold check in
        _concerning_opponents is the direct substitute.

        Uses exploit_ctrl_target/exploit_ctrl_min_games_per_opponent —
        since the ExploitabilityController's removal (2026-08-14) this
        branch is those fields' sole consumer: the weak-spot question is
        now acted on purely through matchmaking, never through a loss
        caution scale.
        """
        league = self._learner.league
        historical = [
            player
            for player in league.players.values()
            if player.step_count not in LIVE_KEYS
        ]
        if not historical:
            return None

        found = self._concerning_opponents(historical)
        if found is None:
            return None
        concerning_players, concerning_win_rates = found

        pick_idx = np.random.choice(
            len(concerning_players), p=pfsp(concerning_win_rates, weighting="squared")
        )
        return league.materialize(concerning_players[pick_idx])

    def get_match(self) -> tuple[ParamsContainer, bool]:
        if self._pinned_opponent is not None:
            return self._pinned_opponent, False

        # 50% PFSP / 15% verification / 35% mirror self-play, matching
        # AlphaStar's MainPlayer split.
        coin_toss = np.random.random()

        # Make sure you can beat the League (PFSP)
        # We only store trajectories from the the perspective of the main player,
        # so we need to oversample playing against it such that the proportion of
        # games played against it is 50%.
        if coin_toss < 0.5:
            opponent = self._pfsp_branch()
            if opponent is not None:  # Found a historical opponent
                return opponent, False
        elif coin_toss < 0.65:
            # AlphaStar-style verification slice (their split is 50% PFSP /
            # 15% verification / 35% self-play — matched here): force
            # attention on a real, reliable weak spot beyond what the
            # broader PFSP draw above already biases toward. Falls through
            # to mirror self-play, same as the PFSP branch above, if
            # nothing currently qualifies as concerning.
            opponent = self._verification_branch()
            if opponent is not None:
                return opponent, False

        return self.pull_own_player(), True

    def update_player_league_stats(
        self, sender: ParamsContainer, receiver: ParamsContainer, trajectory: Trajectory
    ):
        """Update league stats based on trajectory outcome."""
        payoff = (
            trajectory.player_transitions.env_output.win_reward[-1] @ CAT_VF_SUPPORT
        )
        self._learner.league.update_payoff(sender, receiver, payoff)
