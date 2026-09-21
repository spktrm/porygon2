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


def _tree(extra: dict[str, jax.Array] | None = None, scale: float = 1.0) -> dict:
    params = {
        "encoder": {"kernel": jnp.full((2, 3), scale), "bias": jnp.full((3,), scale)},
        "head": {"kernel": jnp.full((3, 1), scale)},
    }
    if extra:
        params["encoder"].update(extra)
    return {"params": params}


def test_merge_params_reports_added_and_dropped() -> None:
    fresh = _tree({"new_leaf": jnp.zeros((4,))}, scale=0.0)
    loaded = _tree({"old_leaf": jnp.ones((5,))}, scale=1.0)
    merged, kept_fresh, dropped, _ = merge_params(fresh, loaded)
    assert kept_fresh == ["/params/encoder/new_leaf"]
    assert dropped == ["/params/encoder/old_leaf"]
    assert "old_leaf" not in merged["params"]["encoder"]
    assert np.all(merged["params"]["encoder"]["new_leaf"] == 0)
    # Shared leaves take the checkpoint's value.
    assert np.all(merged["params"]["encoder"]["kernel"] == 1)
    assert np.all(merged["params"]["head"]["kernel"] == 1)


def test_merge_keeps_an_added_subtree_fresh_everywhere() -> None:
    """A whole new top-level module (a head added or RENAMED so it re-inits,
    and a second added control head) resumed in checkpoint mode: its
    params keep their fresh init and its Adam moments
    start at zero, while the shared leaves take the checkpoint's; a subtree
    only the checkpoint carries is dropped."""
    fresh = _tree({}, scale=0.0)
    fresh["params"]["renamed_head"] = {"Dense_0": {"kernel": jnp.full((3, 2), 0.5)}}
    fresh["params"]["added_control"] = {"Dense_0": {"kernel": jnp.full((2, 2), 0.25)}}
    loaded = _tree({}, scale=1.0)
    loaded["params"]["old_head"] = {"Dense_0": {"kernel": jnp.full((3, 2), 7.0)}}
    merged, kept_fresh, dropped, _ = merge_params(fresh, loaded)
    assert kept_fresh == ["/params/renamed_head", "/params/added_control"]
    assert dropped == ["/params/old_head"]
    assert "old_head" not in merged["params"]
    assert np.all(merged["params"]["renamed_head"]["Dense_0"]["kernel"] == 0.5)
    assert np.all(merged["params"]["added_control"]["Dense_0"]["kernel"] == 0.25)
    assert np.all(merged["params"]["encoder"]["kernel"] == 1)

    optimiser = optax.adam(1e-3)
    fresh_state = optimiser.init(fresh)
    loaded_state = optimiser.init(loaded)
    grads = jax.tree.map(jnp.ones_like, loaded)
    _, loaded_state = optimiser.update(grads, loaded_state, loaded)
    merged_state = merge_opt_state(fresh_state, loaded_state)
    adam = merged_state[0]
    assert int(adam.count) == 1
    assert np.all(adam.mu["params"]["renamed_head"]["Dense_0"]["kernel"] == 0)
    assert np.all(adam.mu["params"]["encoder"]["kernel"] != 0)


def test_merge_opt_state_walks_optax_containers() -> None:
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


def test_merge_opt_state_identity_when_trees_agree() -> None:
    optimiser = optax.adam(1e-3)
    params = _tree()
    loaded = optimiser.init(params)
    merged = merge_opt_state(optimiser.init(params), loaded)
    assert jax.tree_util.tree_structure(merged) == jax.tree_util.tree_structure(loaded)
    for merged_leaf, loaded_leaf in zip(
        jax.tree_util.tree_leaves(merged), jax.tree_util.tree_leaves(loaded)
    ):
        assert np.array_equal(merged_leaf, loaded_leaf)


def _grouped(groups: int, value: float) -> dict:
    return {
        "params": {
            "encoder": {
                "sequence_group_bias": jnp.full((groups, 3), value),
                "input_normalisation": {"group_scale": jnp.full((groups, 3), value)},
                "other_table": jnp.full((groups, 3), value),
            }
        }
    }


def test_a_new_sequence_group_extends_the_per_group_leaves() -> None:
    """One more SequenceGroup grows the leading axis of the per-group leaves;
    the trained rows of the old groups survive and only the new row is fresh,
    in the params and in the Adam moments alike."""
    fresh = _grouped(5, 0.0)
    loaded = _grouped(4, 1.0)
    merged, kept_fresh, _, extended = merge_params(fresh, loaded)
    encoder = merged["params"]["encoder"]
    for leaf in (
        encoder["sequence_group_bias"],
        encoder["input_normalisation"]["group_scale"],
    ):
        assert leaf.shape == (5, 3)
        assert np.all(leaf[:4] == 1) and np.all(leaf[4:] == 0)
    assert extended == [
        "/params/encoder/sequence_group_bias (rows 4 -> 5)",
        "/params/encoder/input_normalisation/group_scale (rows 4 -> 5)",
    ]
    # The control: an unlisted leaf with the same growth is NOT half-loaded.
    assert np.all(encoder["other_table"] == 0)
    assert kept_fresh == ["/params/encoder/other_table (shape (4, 3) -> (5, 3))"]
    # A listed leaf that SHRANK, or whose row width changed, is not extended.
    _, shrunk, _, none_extended = merge_params(_grouped(3, 0.0), loaded)
    assert none_extended == [] and len(shrunk) == 3

    optimiser = optax.adam(1e-3)
    opt_state = merge_opt_state(optimiser.init(fresh), optimiser.init(loaded))
    moments = opt_state[0].mu["params"]["encoder"]["sequence_group_bias"]
    assert moments.shape == (5, 3)
