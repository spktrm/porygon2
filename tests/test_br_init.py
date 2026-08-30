"""BR-init transform (rl/online/artifact.py::apply_br_init).

Plain-dict trees shaped like the real player params (top-level "params"
collection, "action_head" readout subtree) — no model init, so this
stays in the fast suite.
"""

import numpy as np
import pytest

from rl.online.artifact import apply_br_init
from rl.online.config import Porygon2LearnerConfig


def _trees():
    fresh = {
        "params": {
            "encoder": {"w": np.full((2, 3), 0.1, dtype=np.float32)},
            "action_head": {
                "query": np.zeros((3,), dtype=np.float32),
                "key": np.full((3,), 0.0625, dtype=np.float32),
            },
        }
    }
    merged = {
        "params": {
            "encoder": {"w": np.full((2, 3), 0.9, dtype=np.float32)},
            "action_head": {
                "query": np.full((3,), 0.7, dtype=np.float32),
                "key": np.full((3,), -0.3, dtype=np.float32),
            },
        }
    }
    return fresh, merged


def _config(**overrides):
    return Porygon2LearnerConfig().replace(br_target_ckpt="/x", **overrides)


def test_target_is_identity():
    fresh, merged = _trees()
    out = apply_br_init(merged, fresh, _config(br_init="target"))
    assert out is merged


def test_head_reset_grafts_fresh_readout_only():
    fresh, merged = _trees()
    # Positive control: the graft must actually change something.
    assert not np.array_equal(
        merged["params"]["action_head"]["query"],
        fresh["params"]["action_head"]["query"],
    )
    out = apply_br_init(merged, fresh, _config(br_init="head-reset"))
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
    fresh, merged = _trees()
    for tree in (fresh, merged):
        tree["params"]["readout"] = tree["params"].pop("action_head")
    with pytest.raises(KeyError):
        apply_br_init(merged, fresh, _config(br_init="head-reset"))


def test_shrink_perturb_interpolates_every_leaf():
    fresh, merged = _trees()
    out = apply_br_init(
        merged, fresh, _config(br_init="shrink-perturb", br_perturb_frac=0.75)
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
    fresh, merged = _trees()
    inherit = apply_br_init(
        merged, fresh, _config(br_init="shrink-perturb", br_perturb_frac=0.0)
    )
    np.testing.assert_array_equal(
        np.asarray(inherit["params"]["encoder"]["w"]),
        merged["params"]["encoder"]["w"],
    )
    reset = apply_br_init(
        merged, fresh, _config(br_init="shrink-perturb", br_perturb_frac=1.0)
    )
    np.testing.assert_array_equal(
        np.asarray(reset["params"]["encoder"]["w"]),
        fresh["params"]["encoder"]["w"],
    )


def test_bad_inputs_raise():
    fresh, merged = _trees()
    with pytest.raises(ValueError):
        apply_br_init(
            merged, fresh, _config(br_init="shrink-perturb", br_perturb_frac=1.5)
        )
    with pytest.raises(ValueError):
        apply_br_init(merged, fresh, _config(br_init="banana"))
