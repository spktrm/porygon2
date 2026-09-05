"""BR-init transform (rl/online/artifact.py::apply_br_init).

Plain-dict trees shaped like the real player params (top-level "params"
collection, "action_head" readout subtree) — no model init, so this
stays in the fast suite. The init_fn stub records the key it is called
with, which is what lets the seed-independence contract be pinned: the
first sp75 probe drew its "fresh" component from the lineage seed the
TARGET also grew from, so the perturb was a rewind along the target's
own training path (measured cos 0.95 to the target at the BR's first
checkpoint, 2026-08-30).
"""

import jax
import numpy as np
import pytest

from rl.online.artifact import apply_br_init
from rl.online.config import Porygon2LearnerConfig

LINEAGE_KEY = jax.random.key(42)


def _fresh_tree():
    return {
        "params": {
            "encoder": {"w": np.full((2, 3), 0.1, dtype=np.float32)},
            "action_head": {
                "query": np.zeros((3,), dtype=np.float32),
                "key": np.full((3,), 0.0625, dtype=np.float32),
            },
        }
    }


def _merged_tree():
    return {
        "params": {
            "encoder": {"w": np.full((2, 3), 0.9, dtype=np.float32)},
            "action_head": {
                "query": np.full((3,), 0.7, dtype=np.float32),
                "key": np.full((3,), -0.3, dtype=np.float32),
            },
        }
    }


def _recording_init_fn(calls):
    def init_fn(key):
        calls.append(key)
        return _fresh_tree()

    return init_fn


def _refusing_init_fn(key):
    raise AssertionError("target mode must not pay an init call")


def _config(**overrides):
    return Porygon2LearnerConfig().replace(br_target_ckpt="/x", **overrides)


def test_target_is_identity_without_init_call():
    merged = _merged_tree()
    out = apply_br_init(merged, _refusing_init_fn, _config(br_init="target"))
    assert out is merged


def test_fresh_draw_is_not_the_lineage_seed_and_is_per_run():
    def draw_key(**overrides):
        calls = []
        apply_br_init(
            _merged_tree(),
            _recording_init_fn(calls),
            _config(br_init="head-reset", **overrides),
        )
        assert len(calls) == 1
        return jax.random.key_data(calls[0])

    key_a = draw_key(ckpt_subdir="br/a")
    key_b = draw_key(ckpt_subdir="br/b")
    key_a_again = draw_key(ckpt_subdir="br/a")
    # Never the lineage seed (the target's own ancestor — the sp75 bug),
    # unique per run identity (a constant fold is the same bug one BR
    # generation down), reproducible for the same identity.
    for key in (key_a, key_b):
        assert not np.array_equal(key, jax.random.key_data(LINEAGE_KEY))
    assert not np.array_equal(key_a, key_b)
    np.testing.assert_array_equal(key_a, key_a_again)


def test_head_reset_grafts_fresh_readout_only():
    fresh = _fresh_tree()
    merged = _merged_tree()
    # Positive control: the graft must actually change something.
    assert not np.array_equal(
        merged["params"]["action_head"]["query"],
        fresh["params"]["action_head"]["query"],
    )
    out = apply_br_init(merged, _recording_init_fn([]), _config(br_init="head-reset"))
    np.testing.assert_array_equal(
        out["params"]["action_head"]["query"],
        fresh["params"]["action_head"]["query"],
    )
    np.testing.assert_array_equal(
        out["params"]["action_head"]["key"],
        fresh["params"]["action_head"]["key"],
    )
    np.testing.assert_array_equal(
        out["params"]["encoder"]["w"], merged["params"]["encoder"]["w"]
    )


def test_head_reset_renamed_readout_fails_loudly():
    fresh = _fresh_tree()
    merged = _merged_tree()
    for tree in (fresh, merged):
        tree["params"]["readout"] = tree["params"].pop("action_head")

    def init_fn(key):
        return fresh

    with pytest.raises(KeyError):
        apply_br_init(merged, init_fn, _config(br_init="head-reset"))


def test_shrink_perturb_interpolates_every_leaf():
    out = apply_br_init(
        _merged_tree(),
        _recording_init_fn([]),
        _config(br_init="shrink-perturb", br_perturb_frac=0.75),
    )
    np.testing.assert_allclose(
        np.asarray(out["params"]["encoder"]["w"]),
        0.25 * 0.9 + 0.75 * 0.1,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(out["params"]["action_head"]["query"]),
        0.25 * 0.7,
        rtol=1e-6,
    )
    assert np.asarray(out["params"]["encoder"]["w"]).dtype == np.float32


def test_shrink_perturb_endpoints():
    inherit = apply_br_init(
        _merged_tree(),
        _recording_init_fn([]),
        _config(br_init="shrink-perturb", br_perturb_frac=0.0),
    )
    np.testing.assert_array_equal(
        np.asarray(inherit["params"]["encoder"]["w"]),
        _merged_tree()["params"]["encoder"]["w"],
    )
    reset = apply_br_init(
        _merged_tree(),
        _recording_init_fn([]),
        _config(br_init="shrink-perturb", br_perturb_frac=1.0),
    )
    np.testing.assert_array_equal(
        np.asarray(reset["params"]["encoder"]["w"]),
        _fresh_tree()["params"]["encoder"]["w"],
    )


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        apply_br_init(
            _merged_tree(),
            _recording_init_fn([]),
            _config(br_init="shrink-perturb", br_perturb_frac=1.5),
        )
    with pytest.raises(ValueError):
        apply_br_init(_merged_tree(), _refusing_init_fn, _config(br_init="banana"))
