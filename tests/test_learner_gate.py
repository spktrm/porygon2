"""_should_add_new_player: the AlphaStar checkpoint-pacing gate.

Runs the real unbound method against a stub learner (a full Learner needs
devices, actors, and a service), so the gate logic is tested with the real
League underneath.
"""

from types import SimpleNamespace

import numpy as np

from rl.model.utils import ParamsContainer
from rl.online.league import MAIN_KEY, League, PlayerRef
from rl.online.learner import _STEP_OFFSET, Learner

MIN_FRAMES = 200_000
MAX_FRAMES = 9_000_000
MIN_INITIAL_STEPS = 50_000


def make_ref(step: int, frames: int, origin: str = "main") -> PlayerRef:
    return PlayerRef(
        step_count=step,
        snapshot_dir=f"/nonexistent/p_{step:08}",
        player_frame_count=frames,
        builder_frame_count=0,
        origin=origin,
    )


def make_learner(main_frames: int, players: list[PlayerRef], main_steps: int = 10_000):
    league = League(
        main_player=ParamsContainer(
            step_count=MAIN_KEY,
            player_frame_count=main_frames,
            builder_frame_count=0,
            player_params={},
            builder_params={},
        ),
        players=players,
    )
    stub = SimpleNamespace(
        league=league,
        config=SimpleNamespace(
            add_player_min_frames=MIN_FRAMES,
            add_player_max_frames=MAX_FRAMES,
            minimum_historical_player_steps=MIN_INITIAL_STEPS,
        ),
    )
    pop = SimpleNamespace(
        live_key=MAIN_KEY,
        player_state=SimpleNamespace(step_count=np.array(main_steps)),
    )
    return stub, pop


def gate(stub, pop):
    return Learner._should_add_new_player(stub, pop)


def test_empty_league_waits_for_minimum_steps():
    stub, pop = make_learner(main_frames=MIN_FRAMES + 1, players=[], main_steps=100)
    assert gate(stub, pop) is None


def test_empty_league_initial_add_after_minimum_steps():
    stub, pop = make_learner(
        main_frames=MIN_FRAMES + 1, players=[], main_steps=MIN_INITIAL_STEPS + 1
    )
    assert gate(stub, pop) == "initial"


def test_no_add_soon_after_own_checkpoint():
    stub, pop = make_learner(
        main_frames=1_000_000,
        players=[make_ref(100, frames=1_000_000 - MIN_FRAMES // 2)],
    )
    assert gate(stub, pop) is None


def test_overdue_after_max_frames():
    stub, pop = make_learner(
        main_frames=MAX_FRAMES + 1_000,
        players=[make_ref(100, frames=0)],
    )
    assert gate(stub, pop) == "overdue"


def test_dominant_before_max_frames():
    ref = make_ref(100, frames=0)
    stub, pop = make_learner(main_frames=MIN_FRAMES + 1, players=[ref])
    main = stub.league.get_main_player()
    for _ in range(20):
        stub.league.update_payoff(main, ref, payoff=1.0)
    assert gate(stub, pop) == "dominant"


def test_not_dominant_at_prior_win_rate():
    # Prior is 0.5 with no games recorded: min win rate 0.5 < 0.7.
    stub, pop = make_learner(
        main_frames=MIN_FRAMES + 1, players=[make_ref(100, frames=0)]
    )
    assert gate(stub, pop) is None


def test_exploiter_publication_does_not_reset_pacing():
    """Regression (2026-08-14): an exploiter timeout-publication carries a
    +100M/+200M step key and a tiny own frame count. Pacing must stay
    anchored to main's OWN last snapshot — before the fix this scenario
    fired "overdue" on every league-management tick, snapshotting main
    every ~10 steps and handing the GPU straight back to the exploiters."""
    main_frames = 24_000_000
    stub, pop = make_learner(
        main_frames=main_frames,
        players=[
            # Main checkpointed itself recently.
            make_ref(25_510, frames=main_frames - MIN_FRAMES // 2),
            # Exploiters published at own-step 20k with small own counters;
            # their league key is own_step + the origin's disjoint offset.
            make_ref(
                20_000 + _STEP_OFFSET["main_exploiter"],
                frames=2_000_000,
                origin="main_exploiter",
            ),
            make_ref(
                20_000 + _STEP_OFFSET["league_exploiter"],
                frames=2_500_000,
                origin="league_exploiter",
            ),
        ],
    )
    assert gate(stub, pop) is None


def test_dominance_is_judged_against_all_historicals():
    """AlphaStar's ready_to_checkpoint takes win_rates.min() over every
    Historical, exploiter-origin included — beating only your own lineage
    is not dominance."""
    main_ref = make_ref(100, frames=0)
    exp_ref = make_ref(
        20_000 + _STEP_OFFSET["main_exploiter"], frames=0, origin="main_exploiter"
    )
    stub, pop = make_learner(main_frames=MIN_FRAMES + 1, players=[main_ref, exp_ref])
    main = stub.league.get_main_player()
    for _ in range(20):
        stub.league.update_payoff(main, main_ref, payoff=1.0)
        stub.league.update_payoff(main, exp_ref, payoff=-1.0)
    assert gate(stub, pop) is None


def test_build_population_seeds_host_step_from_restored_state():
    """Regression (2026-08-14 overdue add storm): league keys are
    host_step + _STEP_OFFSET and League.get_latest_player picks newest as
    max(key), so a session-local host_step restarting at 0 left the
    pre-restart snapshot "latest" forever (frames_passed never reset ->
    "overdue" every management tick, plus a p_{step:08} overwrite hazard).
    _build_population must seed host_step from the state's own restored
    step_count — and a freshly forked exploiter (step_count 0) must still
    start at 0."""
    from rl.online.config import Porygon2LearnerConfig

    league = League(
        main_player=ParamsContainer(
            step_count=MAIN_KEY,
            player_frame_count=0,
            builder_frame_count=0,
            player_params={},
            builder_params={},
        ),
        players=[],
    )
    stub = SimpleNamespace(
        config=Porygon2LearnerConfig(),
        league=league,
        _active="main",
        _plasticity_probe_jit=None,
        _restore_controller_state=lambda pop, blob: None,
        _create_params_container=lambda pop: None,
    )
    stub.league.update_live = lambda key, container: None

    def build(name, steps):
        return Learner._build_population(
            stub,
            name,
            player_state=SimpleNamespace(
                step_count=np.array(steps), frame_count=np.array(0)
            ),
            builder_state=SimpleNamespace(frame_count=np.array(0)),
            wandb_run=None,
        )

    assert build("main", 71_139).host_step == 71_139
    assert build("main_exploiter", 0).host_step == 0
