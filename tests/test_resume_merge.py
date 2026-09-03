"""Checkpoint-mode resume merges every tree BY PATH (2026-09-02).

Before this, checkpoint mode took params / target / reg / opt_state
verbatim, so a single added or removed param leaf forced params mode --
which resets Adam and the step counts and starts a fresh league. Now an
added leaf keeps its fresh init and fresh zero moments, a removed leaf is
dropped everywhere, and the step count is the checkpoint's.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rl.online.artifact import merge_opt_state, merge_params


def _tree(extra: dict | None = None, scale: float = 1.0):
    params = {
        "encoder": {"kernel": jnp.full((2, 3), scale), "bias": jnp.full((3,), scale)},
        "head": {"kernel": jnp.full((3, 1), scale)},
    }
    if extra:
        params["encoder"].update(extra)
    return {"params": params}


def test_merge_params_reports_added_and_dropped():
    fresh = _tree({"new_leaf": jnp.zeros((4,))}, scale=0.0)
    loaded = _tree({"old_leaf": jnp.ones((5,))}, scale=1.0)
    merged, kept_fresh, dropped = merge_params(fresh, loaded)
    assert kept_fresh == ["/params/encoder/new_leaf"]
    assert dropped == ["/params/encoder/old_leaf"]
    assert "old_leaf" not in merged["params"]["encoder"]
    assert np.all(merged["params"]["encoder"]["new_leaf"] == 0)
    # Shared leaves take the checkpoint's value.
    assert np.all(merged["params"]["encoder"]["kernel"] == 1)
    assert np.all(merged["params"]["head"]["kernel"] == 1)


def test_merge_keeps_an_added_subtree_fresh_everywhere():
    """A whole new top-level module (the 2026-09-03 dynamics_head) resumed
    in checkpoint mode: its params keep their fresh init and its Adam
    moments start at zero, while the shared leaves take the checkpoint's."""
    fresh = _tree({}, scale=0.0)
    fresh["params"]["dynamics_head"] = {"Dense_0": {"kernel": jnp.full((3, 2), 0.5)}}
    loaded = _tree({}, scale=1.0)
    merged, kept_fresh, dropped = merge_params(fresh, loaded)
    assert kept_fresh == ["/params/dynamics_head"]
    assert dropped == []
    assert np.all(merged["params"]["dynamics_head"]["Dense_0"]["kernel"] == 0.5)
    assert np.all(merged["params"]["encoder"]["kernel"] == 1)

    optimiser = optax.adam(1e-3)
    fresh_state = optimiser.init(fresh)
    loaded_state = optimiser.init(loaded)
    grads = jax.tree.map(jnp.ones_like, loaded)
    _, loaded_state = optimiser.update(grads, loaded_state, loaded)
    merged_state = merge_opt_state(fresh_state, loaded_state)
    adam = merged_state[0]
    assert int(adam.count) == 1
    assert np.all(adam.mu["params"]["dynamics_head"]["Dense_0"]["kernel"] == 0)
    assert np.all(adam.mu["params"]["encoder"]["kernel"] != 0)


def test_merge_opt_state_walks_optax_containers():
    optimiser = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(1e-3))
    fresh = optimiser.init(_tree({"new_leaf": jnp.zeros((4,))}))
    loaded_params = _tree({"old_leaf": jnp.ones((5,))})
    loaded = optimiser.init(loaded_params)
    # Step the loaded state so its moments and count are nonzero.
    grads = jax.tree.map(jnp.ones_like, loaded_params)
    _, loaded = optimiser.update(grads, loaded, loaded_params)

    merged = merge_opt_state(fresh, loaded)
    assert type(merged) is type(fresh)
    adam = merged[1][0]
    assert type(adam).__name__ == "ScaleByAdamState"
    assert int(adam.count) == 1
    encoder_mu = adam.mu["params"]["encoder"]
    assert "old_leaf" not in encoder_mu
    assert np.all(encoder_mu["new_leaf"] == 0)
    assert np.all(encoder_mu["kernel"] != 0)
    # The merged state is usable: an update on the FRESH tree's structure.
    fresh_params = _tree({"new_leaf": jnp.zeros((4,))})
    grads = jax.tree.map(jnp.ones_like, fresh_params)
    optimiser.update(grads, merged, fresh_params)


def test_merge_opt_state_identity_when_trees_agree():
    optimiser = optax.adam(1e-3)
    params = _tree()
    loaded = optimiser.init(params)
    merged = merge_opt_state(optimiser.init(params), loaded)
    assert jax.tree_util.tree_structure(merged) == jax.tree_util.tree_structure(loaded)
    for merged_leaf, loaded_leaf in zip(
        jax.tree_util.tree_leaves(merged), jax.tree_util.tree_leaves(loaded)
    ):
        assert np.array_equal(merged_leaf, loaded_leaf)
