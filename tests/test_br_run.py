"""Best-response child runs (2026-08-27): root scoping, explicit init
paths, the matchmaking pin, target registration, publish/import, and the
league-add suppression gate. Plain stubs throughout; every contract test
carries the positive control that proves it could fail."""

import argparse
import os
import types

import numpy as np
import pytest

from rl import checkpoint
from rl.model.utils import ParamsContainer
from rl.online.artifact import ckpt_root, load_train_state
from rl.online.config import Porygon2LearnerConfig
from rl.online.league import MAIN_KEY, League
from rl.online.player_actor import PlayerActor
from rl.online.training.league_ops import (
    BR_STEP_OFFSET,
    import_br_snapshots,
    publish_br_snapshot,
    ref_label,
    register_br_target,
    target_step_count,
)


def make_container(step_count: int, frames: int = 0) -> ParamsContainer:
    return ParamsContainer(
        step_count=step_count,
        player_frame_count=frames,
        builder_frame_count=0,
        player_params={"w": np.zeros(1)},
        builder_params={"w": np.zeros(1)},
    )


def make_config(**overrides) -> Porygon2LearnerConfig:
    return Porygon2LearnerConfig().replace(**overrides)


def write_target_ckpt(root, step: int = 123) -> str:
    """A minimal target checkpoint dir: params components plus a dirname
    the step-count fallback can parse."""
    target_dir = os.path.join(str(root), f"ckpt_{step:08}")
    checkpoint.save_param_snapshot(
        target_dir,
        player_components=dict(params={"w": np.full(1, 3.0)}),
        builder_components=dict(params={"w": np.full(1, 4.0)}),
    )
    return target_dir


class TestCkptRoot:
    def test_subdir_scopes_the_root(self):
        base = make_config()
        child = make_config(ckpt_subdir=os.path.join("br", "x"))
        assert ckpt_root(child) == os.path.join(ckpt_root(base), "br", "x")
        # Control: without the subdir the root is unchanged.
        assert "br" not in ckpt_root(base)

    def test_parent_scan_ignores_child_subtree(self, tmp_path):
        parent = tmp_path / "gen9"
        (parent / "ckpt_00000100").mkdir(parents=True)
        (parent / "br" / "x" / "ckpt_00099999").mkdir(parents=True)
        (parent / "players").mkdir()
        found = checkpoint.most_recent_ckpt_dir(str(parent))
        # Control half: the direct child IS found; the br subtree's higher
        # step never wins because the scan is non-recursive.
        assert found is not None and found.endswith("ckpt_00000100")


class TestExplicitInitPath:
    def test_missing_explicit_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_train_state(
                make_config(), None, None, mode="params", ckpt_path="/nonexistent/ckpt"
            )

    def test_scratch_with_explicit_path_raises(self):
        with pytest.raises(ValueError):
            load_train_state(
                make_config(), None, None, mode="scratch", ckpt_path="/anything"
            )

    def test_explicit_path_routes_to_params_load(self, tmp_path, monkeypatch):
        # Control: a path that exists reaches load_from_params with that
        # exact dir (never the most-recent lookup, never scratch).
        import rl.online.artifact as artifact

        seen = {}

        def fake_load(ckpt_path, *args):
            seen["path"] = ckpt_path
            return "sentinel"

        monkeypatch.setattr(artifact, "load_from_params", fake_load)
        result = load_train_state(
            make_config(), None, None, mode="params", ckpt_path=str(tmp_path)
        )
        assert result == "sentinel"
        assert seen["path"] == str(tmp_path)


class TestPinnedMatchmaking:
    def _actor(self, pinned, league=None) -> PlayerActor:
        actor = PlayerActor.__new__(PlayerActor)
        actor._pinned_opponent = pinned
        actor._learner = types.SimpleNamespace(
            league=league,
            config=make_config(),
        )
        return actor

    def test_pin_short_circuits_every_draw(self):
        pinned = make_container(123)
        actor = self._actor(pinned)
        for _ in range(100):
            opponent, trainable = actor.get_match()
            assert opponent is pinned
            assert trainable is False

    def test_unpinned_reproduces_existing_branches(self):
        # Control: with no pin and an empty roster, every branch falls
        # through to mirror self-play — (own container, trainable=True).
        league = League(main_player=make_container(MAIN_KEY), players=[])
        actor = self._actor(None, league=league)
        own = league.get_live(MAIN_KEY)
        for _ in range(50):
            opponent, trainable = actor.get_match()
            assert opponent is own
            assert trainable is True


class TestTargetRegistration:
    def test_registers_and_scores_payoff(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target_dir = write_target_ckpt(tmp_path, step=123)
        config = make_config(
            br_target_ckpt=target_dir, ckpt_subdir=os.path.join("br", "x")
        )
        league = League(main_player=make_container(MAIN_KEY), players=[])
        pinned = register_br_target(league, config)

        assert pinned.step_count == 123
        np.testing.assert_array_equal(pinned.player_params["w"], np.full(1, 3.0))
        assert league.players[123].origin == "target"
        # The target snapshot lands under the BR's OWN subtree.
        assert league.players[123].snapshot_dir.startswith(
            os.path.abspath(os.path.join("ckpts", "gen9", "br", "x"))
        )

        # Registered, games score: this is the exploitability curve's
        # data path (update_payoff silently drops unregistered players).
        own = league.get_live(MAIN_KEY)
        league.update_payoff(own, pinned, payoff=1.0)
        assert league.games[(MAIN_KEY, 123)] == 1.0
        assert league.wins[(MAIN_KEY, 123)] == 1.0
        # Control: an unregistered ghost still no-ops silently.
        league.update_payoff(own, make_container(999), payoff=1.0)
        assert league.games.get((MAIN_KEY, 999), 0.0) == 0.0

    def test_reregistration_is_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target_dir = write_target_ckpt(tmp_path, step=123)
        config = make_config(
            br_target_ckpt=target_dir, ckpt_subdir=os.path.join("br", "x")
        )
        league = League(main_player=make_container(MAIN_KEY), players=[])
        register_br_target(league, config)
        ref_before = league.players[123]
        register_br_target(league, config)
        assert league.players[123] is ref_before


def make_br_run_state(value: float, step: int = 500):
    tree = {"w": np.full(1, value)}
    return types.SimpleNamespace(
        player_state=types.SimpleNamespace(
            params=dict(tree),
            target_params=dict(tree),
            step_count=np.int32(step),
            frame_count=np.int32(7),
        ),
        builder_state=types.SimpleNamespace(
            params=dict(tree),
            target_params=dict(tree),
            step_count=np.int32(step),
            frame_count=np.int32(7),
        ),
    )


class TestPublishAndImport:
    def _config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target_dir = write_target_ckpt(tmp_path, step=123)
        return make_config(
            br_target_ckpt=target_dir, ckpt_subdir=os.path.join("br", "x")
        )

    def test_publish_lands_in_parent_players(self, tmp_path, monkeypatch):
        config = self._config(tmp_path, monkeypatch)
        snapshot_dir = publish_br_snapshot(make_br_run_state(1.0), config)

        expected_key = BR_STEP_OFFSET + 123
        assert snapshot_dir.endswith(f"p_{expected_key:08}")
        # Parent root, never the BR subtree.
        assert os.sep + "br" + os.sep not in snapshot_dir
        loaded = checkpoint.load_component(snapshot_dir, "player", "params")
        np.testing.assert_array_equal(loaded["w"], np.full(1, 1.0))

    def test_second_publish_overwrites_in_place(self, tmp_path, monkeypatch):
        config = self._config(tmp_path, monkeypatch)
        first = publish_br_snapshot(make_br_run_state(1.0), config)
        second = publish_br_snapshot(make_br_run_state(2.0, step=900), config)
        assert first == second
        loaded = checkpoint.load_component(second, "player", "params")
        # Control: the content genuinely changed — latest params win.
        np.testing.assert_array_equal(loaded["w"], np.full(1, 2.0))
        players_root = os.path.dirname(second)
        assert sorted(os.listdir(players_root)) == [f"p_{BR_STEP_OFFSET + 123:08}"]

    def test_import_registers_once_and_ignores_orphans(self, tmp_path, monkeypatch):
        config = self._config(tmp_path, monkeypatch)
        publish_br_snapshot(make_br_run_state(1.0), config)
        # An ordinary (provenance-less) snapshot dir must stay invisible.
        orphan = os.path.join("ckpts", "gen9", "players", "p_00000777")
        checkpoint.save_param_snapshot(
            orphan,
            player_components=dict(params={"w": np.zeros(1)}),
            builder_components=dict(params={"w": np.zeros(1)}),
        )

        parent_config = config.replace(ckpt_subdir=None, br_target_ckpt=None)
        league = League(main_player=make_container(MAIN_KEY), players=[])
        imported = import_br_snapshots(league, parent_config)
        key = BR_STEP_OFFSET + 123

        assert len(imported) == 1
        ref = league.players[key]
        assert ref.origin == "br"
        assert ref.player_frame_count == 7
        assert 777 not in league.players
        # br refs stay out of main's snapshot-pacing clock.
        assert league.get_latest_player(origin="main") is None
        assert ref_label(ref) == "br-123"
        # Idempotent: a second scan imports nothing.
        assert import_br_snapshots(league, parent_config) == []


class TestLeagueAddSuppression:
    def _learner(self, config, frames: int):
        from rl.online.training.learner import Learner

        learner = Learner.__new__(Learner)
        learner.config = config
        learner.league = League(
            main_player=make_container(MAIN_KEY, frames=frames), players=[]
        )
        return learner

    def test_br_mode_never_touches_run_state(self):
        # run_state=None: any attempt to snapshot would raise — the gate
        # must return before the machinery runs.
        learner = self._learner(make_config(br_target_ckpt="/some/ckpt"), frames=10**9)
        assert learner._manage_league(None, step=60_000) is None

    def test_main_mode_reaches_the_machinery(self):
        # Control: same call in main mode DOES reach should_add_new_player
        # (past the min-frames gate, empty roster -> it reads
        # run_state.player_state and explodes on the None stub).
        learner = self._learner(make_config(), frames=10**9)
        with pytest.raises(AttributeError):
            learner._manage_league(None, step=60_000)


class TestBrWinrateStop:
    def _learner(self, threshold: float):
        from rl.online.league import PlayerRef
        from rl.online.training.learner import Learner

        learner = Learner.__new__(Learner)
        learner.config = make_config(
            br_target_ckpt="/some/ckpt", br_stop_winrate=threshold
        )
        learner.league = League(main_player=make_container(MAIN_KEY), players=[])
        learner.league.add_player(
            PlayerRef(
                step_count=123,
                snapshot_dir="/nonexistent/p_00000123",
                player_frame_count=0,
                builder_frame_count=0,
                origin="target",
            )
        )
        learner.done = False
        return learner

    def _play(self, learner, wins: int, losses: int):
        own = learner.league.get_live(MAIN_KEY)
        target = make_container(123)
        for _ in range(wins):
            learner.league.update_payoff(own, target, payoff=1.0)
        for _ in range(losses):
            learner.league.update_payoff(own, target, payoff=-1.0)

    def test_stops_on_reliable_winrate(self):
        learner = self._learner(0.7)
        self._play(learner, wins=25, losses=0)
        learner._manage_league(None, step=100)
        assert learner.done is True

    def test_holds_below_games_floor(self):
        # Control: a perfect record on too few games must not stop —
        # the Laplace-prior reliability floor is the point.
        learner = self._learner(0.7)
        self._play(learner, wins=5, losses=0)
        learner._manage_league(None, step=100)
        assert learner.done is False

    def test_holds_below_threshold(self):
        learner = self._learner(0.7)
        self._play(learner, wins=13, losses=12)
        learner._manage_league(None, step=100)
        assert learner.done is False

    def test_zero_threshold_is_off(self):
        learner = self._learner(0.0)
        self._play(learner, wins=50, losses=0)
        learner._manage_league(None, step=100)
        assert learner.done is False


class TestTrainStepBudget:
    def test_idle_ticks_do_not_consume_num_steps(self, monkeypatch):
        # Regression (first BR run, 2026-08-27): `for _ in range(num_steps)`
        # burned loop iterations on idle warm-up ticks and ended the run at
        # 1891 of 5000 train steps. num_steps must bound host_step.
        import queue as queue_mod
        import threading

        import rl.online.training.learner as learner_mod
        from rl.online.training.learner import Learner

        monkeypatch.setattr(learner_mod, "start_workers", lambda *a, **k: None)
        monkeypatch.setattr(learner_mod, "stop_workers", lambda *a, **k: None)

        device_q = queue_mod.Queue()
        for _ in range(3):
            device_q.put({"x": np.zeros(1)})
        run_state = types.SimpleNamespace(
            host_step=0,
            lifetime_step=0,
            frames_trained_total=0,
            created_at_frame=0,
            device_q=device_q,
            player_state=types.SimpleNamespace(frame_count=np.int32(0)),
        )

        learner = Learner.__new__(Learner)
        learner.config = make_config(num_steps=3, br_target_ckpt=None)
        learner.run_state = run_state
        learner.done = False
        learner.gpu_lock = threading.Lock()
        learner._train_step = lambda rs, batch: {}
        learner._handle_periodic_tasks = lambda rs, step, logs: None
        checkpoints = []
        learner._write_checkpoint = lambda rs, synchronous=False: checkpoints.append(
            (run_state.host_step, synchronous)
        )

        # Idle ticks (None) interleaved with ready ones: under the old
        # loop, 6 idle ticks + 3 batches would exhaust a budget of 3
        # before the third train step (the positive-control half).
        readiness = iter(
            [None, None, run_state, None, run_state, None, None, run_state]
        )
        learner._ready_run_state = lambda: next(readiness)

        learner.train()

        assert run_state.host_step == 3
        # Completion writes exactly one synchronous checkpoint.
        assert checkpoints == [(3, True)]


class TestResolveRunSetup:
    def _args(self, **overrides):
        defaults = dict(
            debug=False,
            load_mode=None,
            init_ckpt=None,
            num_steps=None,
            br_target=None,
            run_tag=None,
            br_winrate=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_plain_run_unchanged(self, monkeypatch):
        from rl.online.main import resolve_run_setup

        monkeypatch.delenv("LOAD_STATE_MODE", raising=False)
        config, mode, init_ckpt, job_name = resolve_run_setup(
            make_config(), self._args()
        )
        assert (mode, init_ckpt, job_name) == ("checkpoint", None, "main")
        assert config.br_target_ckpt is None and config.ckpt_subdir is None

    def test_fresh_br_derives_params_mode(self, tmp_path, monkeypatch):
        from rl.online.main import resolve_run_setup

        monkeypatch.chdir(tmp_path)
        target_dir = write_target_ckpt(tmp_path, step=123)
        config, mode, init_ckpt, job_name = resolve_run_setup(
            make_config(), self._args(br_target=target_dir, num_steps=50)
        )
        assert mode == "params"
        assert init_ckpt == os.path.abspath(target_dir)
        assert job_name == f"br-ckpt_{123:08}"
        assert config.ckpt_subdir == os.path.join("br", f"ckpt_{123:08}")
        assert config.num_steps == 50

    def test_existing_br_tree_resumes(self, tmp_path, monkeypatch):
        from rl.online.main import resolve_run_setup

        monkeypatch.chdir(tmp_path)
        target_dir = write_target_ckpt(tmp_path, step=123)
        os.makedirs(os.path.join("ckpts", "gen9", "br", "tagged", "ckpt_00000010"))
        config, mode, init_ckpt, job_name = resolve_run_setup(
            make_config(), self._args(br_target=target_dir, run_tag="tagged")
        )
        assert (mode, init_ckpt, job_name) == ("checkpoint", None, "br-tagged")

    def test_br_without_num_steps_defaults_winrate_stop(self, tmp_path, monkeypatch):
        from rl.online.main import resolve_run_setup

        monkeypatch.chdir(tmp_path)
        target_dir = write_target_ckpt(tmp_path, step=123)
        config, _, _, _ = resolve_run_setup(
            make_config(), self._args(br_target=target_dir)
        )
        assert config.br_stop_winrate == 0.7
        # num_steps keeps the effectively-unbounded config default.
        assert config.num_steps == Porygon2LearnerConfig().num_steps

    def test_br_with_num_steps_keeps_winrate_stop_off(self, tmp_path, monkeypatch):
        from rl.online.main import resolve_run_setup

        monkeypatch.chdir(tmp_path)
        target_dir = write_target_ckpt(tmp_path, step=123)
        config, _, _, _ = resolve_run_setup(
            make_config(), self._args(br_target=target_dir, num_steps=50)
        )
        assert config.br_stop_winrate == 0.0

    def test_explicit_br_winrate_wins(self, tmp_path, monkeypatch):
        from rl.online.main import resolve_run_setup

        monkeypatch.chdir(tmp_path)
        target_dir = write_target_ckpt(tmp_path, step=123)
        config, _, _, _ = resolve_run_setup(
            make_config(),
            self._args(br_target=target_dir, num_steps=50, br_winrate=0.8),
        )
        assert config.br_stop_winrate == 0.8

    def test_br_rejects_conflicting_flags(self, tmp_path):
        from rl.online.main import resolve_run_setup

        target_dir = write_target_ckpt(tmp_path, step=123)
        with pytest.raises(SystemExit):
            resolve_run_setup(
                make_config(),
                self._args(br_target=target_dir, load_mode="scratch"),
            )


class TestTargetStepCount:
    def test_scalars_win_over_dirname(self, tmp_path):
        target_dir = os.path.join(str(tmp_path), "ckpt_00000123")
        checkpoint.save_param_snapshot(
            target_dir,
            player_components=dict(
                params={"w": np.zeros(1)}, scalars=dict(step_count=456)
            ),
            builder_components=dict(params={"w": np.zeros(1)}),
        )
        assert target_step_count(target_dir) == 456

    def test_dirname_fallback(self, tmp_path):
        assert target_step_count(str(write_target_ckpt(tmp_path, step=123))) == 123

    def test_unparseable_raises(self, tmp_path):
        bare = tmp_path / "no-digits-here"
        bare.mkdir()
        with pytest.raises(ValueError):
            target_step_count(str(bare))
