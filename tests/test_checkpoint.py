"""Checkpoint save/load roundtrips and directory discovery."""

import os

import numpy as np
import pytest

from rl import checkpoint


def tree_equal(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            tree_equal(a[k], b[k])
    else:
        np.testing.assert_array_equal(a, b)


@pytest.fixture
def params():
    rng = np.random.default_rng(0)
    return {"params": {"encoder": rng.normal(size=(4, 4)), "v_head": rng.normal(size=(4,))}}


def test_param_snapshot_roundtrip(tmp_path, params):
    snap = str(tmp_path / "p_00000100")
    checkpoint.save_param_snapshot(
        snap,
        player_components=dict(params=params, target_params=params),
        builder_components=dict(params={"w": np.ones(2)}),
    )
    tree_equal(checkpoint.load_component(snap, "player", "params"), params)
    tree_equal(checkpoint.load_component(snap, "player", "target_params"), params)
    tree_equal(checkpoint.load_component(snap, "builder", "params"), {"w": np.ones(2)})
    assert checkpoint.has_component(snap, "player", "params")
    # Params-only snapshot: no optimiser state was ever written.
    assert not checkpoint.has_component(snap, "player", "opt_state")


def test_train_state_roundtrip(tmp_path, params):
    ckpt_dir = str(tmp_path / "ckpt_00000100")
    league_bytes = b"league-state"
    controller_bytes = b"controller-state"
    checkpoint.save_train_state(
        ckpt_dir,
        learner_config={"generation": 9},
        player_state_components=dict(params=params, scalars={"step_count": 100}),
        builder_state_components=dict(params={"w": np.zeros(2)}),
        league_bytes=league_bytes,
        controller_bytes=controller_bytes,
    )
    full = checkpoint.load_full(ckpt_dir)
    tree_equal(full["player_state"]["params"], params)
    assert full["player_state"]["scalars"]["step_count"] == 100
    assert full["league"] == league_bytes
    assert full["controllers"] == controller_bytes
    assert full["meta"]["learner_config"] == {"generation": 9}


def test_missing_league_and_controller_bytes_are_none(tmp_path, params):
    ckpt_dir = str(tmp_path / "ckpt_00000001")
    checkpoint.save_train_state(
        ckpt_dir,
        learner_config={},
        player_state_components=dict(params=params),
        builder_state_components=dict(params={}),
        league_bytes=b"x",
    )
    assert checkpoint.load_controller_bytes(ckpt_dir) is None
    assert checkpoint.load_league_bytes(str(tmp_path / "nope")) is None


def test_ckpt_dir_discovery(tmp_path):
    root = str(tmp_path)
    for step in (100, 2000, 50):
        os.makedirs(os.path.join(root, f"ckpt_{step:08}"))
    (tmp_path / "not_a_ckpt").mkdir()

    dirs = checkpoint.list_ckpt_dirs(root)
    assert [s for s, _ in dirs] == [50, 100, 2000]
    assert checkpoint.most_recent_ckpt_dir(root).endswith("ckpt_00002000")


def test_most_recent_ckpt_dir_empty_root(tmp_path):
    assert checkpoint.most_recent_ckpt_dir(str(tmp_path)) is None


def test_loaders_skip_writer_scratch_files(tmp_path, params):
    """A process killed mid-_dump leaves '<name>.tmp.<pid>.<tid>' beside the
    completed component; the loaders must skip it (2026-08-15: a truncated
    tmp file fed to pickle aborted an otherwise-healthy resume)."""
    ckpt_dir = str(tmp_path / "ckpt_00000100")
    checkpoint.save_train_state(
        ckpt_dir,
        learner_config={},
        player_state_components=dict(params=params),
        builder_state_components=dict(params={}),
        league_bytes=b"x",
    )
    (tmp_path / "ckpt_00000100" / "player" / "params.tmp.1.2").write_bytes(b"junk")
    full = checkpoint.load_full(ckpt_dir)
    assert set(full["player_state"]) == {"params"}
