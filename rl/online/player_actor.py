from typing import Literal

import jax
import numpy as np

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
from rl.model.builder_model import get_packed_team_string
from rl.model.heads import HeadParams
from rl.model.utils import Params, ParamsContainer
from rl.online.agent import Agent
from rl.online.guards import should_push_trajectory
from rl.online.inference import InferenceServer
from rl.online.league import (
    LEAGUE_EXPLOITER_KEY,
    LIVE_KEYS,
    MAIN_EXPLOITER_KEY,
    MAIN_KEY,
    PlayerRef,
    pfsp,
)
from rl.online.learner import Learner

Population = Literal["main", "main_exploiter", "league_exploiter"]
_LIVE_KEY_BY_POPULATION: dict[Population, int] = {
    "main": MAIN_KEY,
    "main_exploiter": MAIN_EXPLOITER_KEY,
    "league_exploiter": LEAGUE_EXPLOITER_KEY,
}


class ActorStopped(Exception):
    """Raised inside an unroll when its population began shutting down —
    unwinds the actor thread out of a blocking wait it would otherwise
    never leave (e.g. the builder-replay sample wait, where no data is
    coming once that population's producers have stopped). Handled as a
    clean loop exit by main.py's actor runners, never as an error —
    without it, Ctrl-C left actor threads stuck in these waits, tripping
    the shutdown straggler check."""


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
        population: Population = "main",
        inference_client: InferenceServer | None = None,
        explore: bool = False,
        explore_temp_range: tuple[float, float] | None = None,
    ):
        self._agent = agent
        self._env = env
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
        # Three-population redesign (docs/exploiter-phase-plan.md): which
        # live population this actor generates trajectories for.
        # main_exploiter/league_exploiter don't pin to a fixed target SET
        # drawn once — AlphaStar's own exploiters re-sample PFSP fresh every
        # match from their candidate pool (lineage-restricted to
        # origin=="main" for main_exploiter, unrestricted for
        # league_exploiter — see get_match()), never freezing a small
        # subset for the population's whole lifetime the way the old
        # single-generic-exploiter design's pin_opponent_steps did.
        self.population = population
        self._live_key = _LIVE_KEY_BY_POPULATION[population]
        # Exploration-ladder actor (config.num_explore_actors /
        # explore_temp_range): samples a FRESH temperature per game —
        # log-uniform over the range, the continuous analogue of R2D2's
        # geometrically-spaced per-actor epsilon ladder — and every
        # trajectory it produces is tagged so train_step routes it to the
        # observer Q critic only (see Trajectory.explore). Per-game, not
        # per-actor: an unroll is one padded game, so one draw per unroll
        # gives a coherent behaviour policy per game and a continuous
        # spectrum across games. The recorded log_policy always reflects
        # the tempered logits, so Retrace's ISRs are correct for free.
        # Explore actors never route via the InferenceServer (it serves
        # everyone at the base temperature).
        self._explore = explore
        self._explore_temp_range = explore_temp_range
        if explore:
            assert inference_client is None, (
                "explore actors sample per-game temperatures via their own "
                "Agent; the batched InferenceServer has no per-request "
                "head_params"
            )
            assert explore_temp_range is not None
        self._temp_rng = np.random.default_rng(rng_seed)

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
        return Action(
            src=agent_output.actor_output.action_head.src_index.item(),
            tgt=agent_output.actor_output.action_head.tgt_index.item(),
        )

    def unroll(
        self, rng_key: jax.Array, player_params: Params | ParamsContainer
    ) -> Trajectory:
        """Run unroll_length agent/environment steps, returning the trajectory.

        player_params is device-resident Params on the direct-Agent path,
        or the host ParamsContainer when routing through the batched
        InferenceServer (which owns device transfer) — see unroll_and_push."""

        player_subkeys = split_rng(rng_key, self._unroll_length)
        player_traj = []

        team_tokens = None
        builder_trajectory = ()
        builder_history = ()
        if self._learner.config.smogon_format != "randombattle":
            pop = self._learner.populations[self.population]
            builder_replay = pop.builder_replay
            sample_cond = builder_replay._sample_cv
            with sample_cond:
                # done-aware, like builder_actor.py's add-side wait and
                # Learner.enqueue_traj already are: once this population is
                # shutting down no builder trajectory is ever coming, so a
                # bare ready_to_sample predicate blocked this thread forever
                # (_stop_population_workers notifies this CV precisely so
                # waiters re-check and see done).
                sample_cond.wait_for(
                    lambda: pop.done or builder_replay.ready_to_sample()
                )
                if pop.done:
                    raise ActorStopped(
                        f"population {self.population} stopped during "
                        "builder-replay sample wait"
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

        # One temperature per game (unrolls are one padded game): drawn
        # log-uniform over explore_temp_range for ladder actors, base
        # HeadParams otherwise (None -> the agent's own default).
        head_params = None
        if self._explore:
            lo, hi = self._explore_temp_range
            head_params = HeadParams(
                temp=float(np.exp(self._temp_rng.uniform(np.log(lo), np.log(hi))))
            )

        # Rollout the player environment.
        for player_step_index in range(player_subkeys.shape[0]):
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
                    head_params=head_params,
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
            padding_step = PlayerTransition(
                env_output=player_actor_input_clipped.env.replace(
                    done=np.zeros_like(player_actor_input_clipped.env.done)
                ),
                agent_output=player_agent_output,
            )
            player_traj += [padding_step] * (self._unroll_length - len(player_traj))

        # Pack the trajectory and reset parent state.
        player_trajectory = jax.device_get(player_traj)
        player_trajectory: PlayerTransition = jax.tree.map(
            lambda *xs: np.stack(xs), *player_trajectory
        )

        return Trajectory(
            builder_transitions=builder_trajectory,
            builder_history=builder_history,
            player_transitions=player_trajectory,
            player_packed_history=player_actor_input.packed_history,
            player_history=player_actor_input.history,
            explore=np.array([self._explore]),
        )

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

        act_out = self.unroll(rng_key=subkey, player_params=player_params)
        self.reset_game_id()

        if should_push_trajectory(self._is_eval, do_push, self._env.username):
            act_out = jax.device_get(act_out)
            self._learner.enqueue_traj(self.population, act_out)
        return act_out

    def pull_own_player(self) -> ParamsContainer:
        """This actor's own live population's current params — MAIN_KEY for
        a main actor, MAIN_EXPLOITER_KEY/LEAGUE_EXPLOITER_KEY otherwise."""
        return self._learner.league.get_live(self._live_key)

    def pull_main_player(self) -> ParamsContainer:
        """Thin alias, kept for callers that specifically want main (not
        'this actor's own population') regardless of self.population —
        e.g. _verification_branch/_concerning_opponents intentionally
        always weight against live main, since that's whose blind spots
        they're checking, not whichever population happens to be asking."""
        return self._learner.league.get_main_player()

    def _pfsp_branch(
        self, allowed_steps: frozenset[int] | None = None
    ) -> ParamsContainer | None:
        historical = [
            player
            for player in self._learner.league.players.values()
            if player.step_count not in LIVE_KEYS
            and (allowed_steps is None or player.step_count in allowed_steps)
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

        Checks exploiter-origin opponents FIRST, matching AlphaStar's own
        scoping (their exploitation check is specifically restricted to
        historical snapshots descended from a MainExploiter, not checked
        against plain history uniformly) — a promoted exploiter represents
        a deliberately found, proven weakness, worth monitoring more
        sensitively than an opponent that's merely whatever main happened
        to be when an overdue window expired. Falls back to checking ALL
        historical opponents if no exploiter-origin one currently
        qualifies, as a general safety net AlphaStar's version doesn't
        need (it has a separate "forgetting" check — a monotonic-suffix
        trick on win-rate history — for that; not ported here, see
        _concerning_opponents' docstring... the threshold check above is
        the direct substitute).

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

        # Either exploiter type's promotions represent a deliberately found,
        # proven weakness — check both before falling back to plain history.
        exploiter_origin = [p for p in historical if p.origin != "main"]
        found = self._concerning_opponents(
            exploiter_origin
        ) or self._concerning_opponents(historical)
        if found is None:
            return None
        concerning_players, concerning_win_rates = found

        pick_idx = np.random.choice(
            len(concerning_players), p=pfsp(concerning_win_rates, weighting="squared")
        )
        return league.materialize(concerning_players[pick_idx])

    def _main_lineage_steps(self) -> frozenset[int]:
        """origin=="main" historical step_counts, recomputed fresh on every
        call — MainExploiter's candidate pool is a live filter over the
        current league, not a fixed set decided once (AlphaStar's own
        MainExploiter PFSP-samples fresh from the opponent's descendants
        every match; freezing a subset at creation time and reusing it for
        the population's whole lifetime would just be re-inventing the old
        single-generic-exploiter design's pin_opponent_steps under a new
        name)."""
        return frozenset(
            ref.step_count
            for ref in self._learner.league.players.values()
            if ref.origin == "main"
        )

    def _league_exploiter_match(self) -> tuple[ParamsContainer, bool]:
        """LeagueExploiter: PFSP over the whole historical population (any
        origin), unrestricted, re-sampled every match — identical to
        _pfsp_branch()'s unrestricted draw, since "the whole population"
        needs no filtering at all. Never mirror self-play, never
        verification."""
        opponent = self._pfsp_branch()
        if opponent is not None:
            return opponent, False
        raise RuntimeError(
            "league_exploiter actor found no historical opponents to play "
            "— the league is empty."
        )

    def _main_exploiter_match(self) -> tuple[ParamsContainer, bool]:
        """AlphaStar's actual MainExploiter rule: if our own win-rate
        against LIVE main is reliable and above
        main_exploiter_live_target_winrate_floor, play live main directly
        (zero disk cost — already device-resident) with is_trainable=False
        (this is a genuinely different population, not mirror self-play;
        main does not automatically learn from this match). Otherwise fall
        back to a fresh PFSP draw restricted to main's own lineage
        (_main_lineage_steps). Every game either branch plays records into
        the shared payoff table via update_player_league_stats, so the
        live-win-rate signal keeps updating with no new statistics code."""
        config = self._learner.config
        league = self._learner.league
        games = league.games.get(
            (MAIN_EXPLOITER_KEY, MAIN_KEY), 0.0
        ) + league.games.get((MAIN_KEY, MAIN_EXPLOITER_KEY), 0.0)
        if games >= config.main_exploiter_live_target_min_games:
            live_winrate = league._win_rate_by_steps(MAIN_EXPLOITER_KEY, MAIN_KEY)
            if live_winrate > config.main_exploiter_live_target_winrate_floor:
                return league.get_live(MAIN_KEY), False
        lineage_steps = self._main_lineage_steps()
        opponent = (
            self._pfsp_branch(allowed_steps=lineage_steps) if lineage_steps else None
        )
        if opponent is not None:
            return opponent, False
        raise RuntimeError(
            'main_exploiter actor found no origin=="main" historical '
            "opponents to play — main hasn't added any league snapshots yet."
        )

    def get_match(self) -> tuple[ParamsContainer, bool]:
        if self.population == "league_exploiter":
            return self._league_exploiter_match()
        if self.population == "main_exploiter":
            return self._main_exploiter_match()

        # main: unchanged 50% PFSP / 15% verification / 35% mirror
        # self-play split.
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
