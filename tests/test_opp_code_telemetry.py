"""The opponent-code / join-key drift panels read the leaves they name.

`_OPP_CODE_LEAVES` (rl/online/training/telemetry.py) resolves each key by
path into the params tree, so a rename silently breaks the panel unless
its known init is pinned: all four leaves are lecun-uniform at fan-in 256
(rms 0.0625). The doubling control proves the reading follows the leaf.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.online.training.telemetry import (
    _GRAD_SUBTREES,
    _OPP_CODE_LEAVES,
    head_param_telemetry,
)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

_INIT_RMS = 0.0625


def _leaf_path(key):
    (path,) = _OPP_CODE_LEAVES[key]
    return path


def test_opp_code_leaves_read_their_init(real_model_and_trajectory):
    _, params, _, _ = real_model_and_trajectory
    grads = jax.tree.map(jnp.zeros_like, params)
    logs = head_param_telemetry(params, grads)
    for key in _OPP_CODE_LEAVES:
        value = float(logs[key])
        assert 0.055 <= value <= 0.070, (key, value)
    for key in (
        "player_opp_code_logits_grad_norm",
        "player_opp_code_embedding_grad_norm",
        "player_entity_index_tag_grad_norm",
    ):
        assert key in _GRAD_SUBTREES
        assert float(logs[key]) == 0.0, key


def test_opp_code_rms_follows_its_leaf(real_model_and_trajectory):
    _, params, _, _ = real_model_and_trajectory
    grads = jax.tree.map(jnp.zeros_like, params)
    before = head_param_telemetry(params, grads)
    for key in _OPP_CODE_LEAVES:
        path = ("params",) + _leaf_path(key)

        def double_this_leaf(key_path, leaf, path=path):
            if tuple(entry.key for entry in key_path) == path:
                return leaf * 2.0
            return leaf

        scaled = jax.tree_util.tree_map_with_path(double_this_leaf, params)
        after = head_param_telemetry(scaled, grads)
        assert float(after[key]) == pytest.approx(2.0 * float(before[key]), rel=1e-3)
        untouched = [other for other in _OPP_CODE_LEAVES if other != key]
        for other in untouched:
            assert float(after[other]) == float(before[other]), (key, other)
    assert np.isfinite([float(before[key]) for key in _OPP_CODE_LEAVES]).all()
