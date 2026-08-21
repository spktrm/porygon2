"""League roster, payoff bookkeeping, PFSP, eviction, and serialization."""

import numpy as np
import pytest

from rl.online.league import (
    MAIN_KEY,
    League,
    PlayerRef,
    pfsp,
)
from rl.model.utils import ParamsContainer

# A ref written by an older revision: a foreign origin tag in a disjoint
# (far higher) step_count range. The roster must still be able to answer
# "when did MAIN last checkpoint?" with one of these present.
FOREIGN_ORIGIN = "main_exploiter"
FOREIGN_STEP = 100_020_000


def make_container(step_count: int, frames: int = 0) -> ParamsContainer:
    return ParamsContainer(
        step_count=step_count,
        player_frame_count=frames,
        builder_frame_count=0,
        player_params={"w": np.zeros(1)},
        builder_params={"w": np.zeros(1)},
    )


def make_ref(step: int, frames: int = 0, origin: str = "main") -> PlayerRef:
    return PlayerRef(
        step_count=step,
        snapshot_dir=f"/nonexistent/p_{step:08}",
        player_frame_count=frames,
        builder_frame_count=0,
        origin=origin,
    )


@pytest.fixture
def league():
    return League(main_player=make_container(MAIN_KEY), players=[])


class TestPfsp:
    def test_normalises_to_one(self):
        for weighting in ("variance", "linear", "linear_capped", "squared", "inverse_squared"):
            probs = pfsp(np.array([0.1, 0.5, 0.9]), weighting=weighting)
            assert probs.shape == (3,)
            np.testing.assert_allclose(probs.sum(), 1.0, rtol=1e-6)
            assert (probs >= 0).all()

    def test_squared_prefers_hard_opponents(self):
        probs = pfsp(np.array([0.1, 0.9]), weighting="squared")
        assert probs[0] > probs[1]

    def test_degenerate_weights_fall_back_to_uniform(self):
        # All win rates 1.0 under "squared" -> zero mass everywhere.
        probs = pfsp(np.array([1.0, 1.0, 1.0, 1.0]), weighting="squared")
        np.testing.assert_allclose(probs, 0.25)


class TestRoster:
    def test_latest_player_ignores_live_and_respects_origin(self, league):
        assert league.get_latest_player() is None
        league.add_player(make_ref(100, frames=1_000, origin="main"))
        league.add_player(make_ref(FOREIGN_STEP, frames=50, origin=FOREIGN_ORIGIN))

        # Unfiltered: the foreign key range wins max().
        assert league.get_latest_player().step_count == FOREIGN_STEP
        # Regression (2026-08-14): main's checkpoint pacing must see its
        # OWN newest snapshot, or a foreign-origin publication pins
        # "latest" forever and the overdue gate fires every tick.
        assert league.get_latest_player(origin="main").step_count == 100
        assert league.get_latest_player(origin="main").player_frame_count == 1_000

    def test_latest_player_none_for_absent_origin(self, league):
        league.add_player(make_ref(FOREIGN_STEP, origin=FOREIGN_ORIGIN))
        assert league.get_latest_player(origin="main") is None

    def test_live_populations_are_not_roster_entries(self, league):
        league.update_live(MAIN_KEY, make_container(MAIN_KEY))
        assert len(league.players) == 0
        assert league.has_live(MAIN_KEY)


class TestPayoff:
    def test_prior_win_rate_is_half(self, league):
        league.add_player(make_ref(100))
        main = league.get_main_player()
        wr = league.get_winrate((main, league.players[100]))
        np.testing.assert_allclose(wr, 0.5)

    def test_wins_raise_and_losses_lower_win_rate(self, league):
        league.add_player(make_ref(100))
        main = league.get_main_player()
        opp = league.players[100]
        for _ in range(20):
            league.update_payoff(main, opp, payoff=1.0)
        wr = float(league.get_winrate((main, opp)).item())
        assert wr > 0.9
        # Symmetric ledger: the opponent's view is the complement.
        wr_opp = float(league.get_winrate((opp, main)).item())
        np.testing.assert_allclose(wr + wr_opp, 1.0, atol=1e-6)

    def test_draws_count_half(self, league):
        league.add_player(make_ref(100))
        main = league.get_main_player()
        opp = league.players[100]
        for _ in range(50):
            league.update_payoff(main, opp, payoff=0.0)
        wr = float(league.get_winrate((main, opp)).item())
        np.testing.assert_allclose(wr, 0.5, atol=1e-6)

    def test_update_for_evicted_player_is_ignored(self, league):
        league.add_player(make_ref(100))
        ghost = make_container(999)
        league.update_payoff(league.get_main_player(), ghost, payoff=1.0)
        assert league.games == {} or all(999 not in k for k in league.games)

    def test_get_winrate_vectorises_over_opponents(self, league):
        for step in (100, 200, 300):
            league.add_player(make_ref(step))
        main = league.get_main_player()
        opponents = [v for v in league.players.values()]
        wr = league.get_winrate((main, opponents))
        assert wr.shape == (3,)


class TestEviction:
    def test_roster_capped_at_league_size(self):
        league = League(main_player=make_container(MAIN_KEY), players=[], league_size=4)
        for step in range(100, 100 + 10 * 10, 10):
            league.add_player(make_ref(step))
        assert len(league.players) == 4

    def test_beaten_and_sampled_opponent_evicted_first(self):
        league = League(main_player=make_container(MAIN_KEY), players=[], league_size=2)
        league.add_player(make_ref(100))
        league.add_player(make_ref(200))
        main = league.get_main_player()
        # Main farms player 100 reliably; player 200 stays challenging.
        for _ in range(30):
            league.update_payoff(main, league.players[100], payoff=1.0)
            league.update_payoff(main, league.players[200], payoff=-1.0)
        league.add_player(make_ref(300))
        assert 100 not in league.players
        assert 200 in league.players
        # The evicted player's payoff rows are garbage-collected too.
        assert all(100 not in k for k in league.games)


class TestSerialization:
    def test_roundtrip_preserves_roster_and_stats(self, league):
        league.add_player(make_ref(100, frames=5_000, origin="main"))
        league.add_player(
            make_ref(FOREIGN_STEP, frames=42, origin=FOREIGN_ORIGIN)
        )
        main = league.get_main_player()
        for _ in range(5):
            league.update_payoff(main, league.players[100], payoff=1.0)

        restored = League.deserialize(league.serialize())
        restored.update_main_player(make_container(MAIN_KEY))

        assert set(restored.players) == {100, FOREIGN_STEP}
        assert restored.players[FOREIGN_STEP].origin == FOREIGN_ORIGIN
        assert restored.players[100].player_frame_count == 5_000
        np.testing.assert_allclose(
            restored.get_winrate((restored.get_main_player(), restored.players[100])),
            league.get_winrate((main, league.players[100])),
        )
        # Regression guard for the origin-filtered pacing gate surviving a resume.
        assert restored.get_latest_player(origin="main").step_count == 100
