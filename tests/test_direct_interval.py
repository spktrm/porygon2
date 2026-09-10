"""Direct-probe copy initialisation, live action control and selection boundary."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rl.model.config import get_player_model_config
from rl.model.constants import POLICY_READABLE_ROWS
from rl.model.interval_transition import DirectIntervalValue
from rl.offline.direct_interval import select_checkpoint


def test_direct_copy_initialisation_and_live_action_control() -> None:
    cfg = get_player_model_config(9, train=True).transition
    cfg.block.model_size = 16
    cfg.row_read_width = 2
    cfg.prior.mlp.layer_sizes = (16, 8, 32)
    rows = jax.random.normal(jax.random.key(2), (len(POLICY_READABLE_ROWS), 16))
    root = jax.nn.log_softmax(jnp.array([1.0, 2.0, 3.0]))
    first_action = jax.nn.one_hot(1, cfg.action_classes)
    second_action = jax.nn.one_hot(4, cfg.action_classes)
    conditioned = DirectIntervalValue(cfg, 3, jnp.float32, True)
    state_only = DirectIntervalValue(cfg, 3, jnp.float32, False)
    params = jax.jit(conditioned.init)(jax.random.key(3), rows, first_action, root)
    state_params = jax.jit(state_only.init)(jax.random.key(3), rows, first_action, root)
    for actual, expected in zip(jax.tree.leaves(params), jax.tree.leaves(state_params)):
        np.testing.assert_array_equal(actual, expected)
    apply = jax.jit(conditioned.apply)
    np.testing.assert_array_equal(apply(params, rows, first_action, root), root)
    gradient = jax.jit(
        jax.grad(
            lambda variables: optax.softmax_cross_entropy(
                conditioned.apply(variables, rows, first_action, root),
                jnp.array([1.0, 0.0, 0.0]),
            )
        )
    )(params)
    assert float(jnp.linalg.norm(gradient["params"]["delta"]["Dense_2"]["kernel"])) > 0
    np.testing.assert_array_equal(gradient["params"]["delta"]["Dense_0"]["kernel"], 0)
    opened = copy.deepcopy(params)
    kernel = opened["params"]["delta"]["Dense_2"]["kernel"]
    opened["params"]["delta"]["Dense_2"]["kernel"] = jax.random.normal(
        jax.random.key(8), kernel.shape
    )
    original = apply(opened, rows, first_action, root)
    changed = apply(opened, rows, second_action, root)
    assert not np.allclose(original, changed)
    state_apply = jax.jit(state_only.apply)
    np.testing.assert_array_equal(
        state_apply(opened, rows, first_action, root),
        state_apply(opened, rows, second_action, root),
    )
    np.testing.assert_allclose((original - root).mean(), 0, atol=1e-6)


def test_checkpoint_selection_ignores_train_and_final_and_uses_earliest_tie() -> None:
    records = [
        {"step": 200, "split": "validation", "prior_delta_gain": 0.1},
        {"step": 100, "split": "validation", "prior_delta_gain": 0.1},
        {"step": 0, "split": "validation", "prior_delta_gain": 0.0},
        {"step": 500, "split": "train", "prior_delta_gain": 0.99},
        {"step": 1000, "split": "final_test", "prior_delta_gain": 0.95},
    ]
    assert select_checkpoint(records)["step"] == 100
    with pytest.raises(ValueError, match="validation"):
        select_checkpoint([])


def test_root_row_bypass_preserves_shared_init_and_has_live_action_gradient() -> None:
    cfg = get_player_model_config(9, train=True).transition
    cfg.block.model_size = 16
    cfg.row_read_width = 2
    cfg.prior.mlp.layer_sizes = (16, 8, 32)
    rows = jax.random.normal(jax.random.key(12), (len(POLICY_READABLE_ROWS), 16))
    root = jax.nn.log_softmax(jnp.array([1.0, 2.0, 3.0]))
    bypass = DirectIntervalValue(cfg, 3, jnp.float32, True, "rows")
    latent = DirectIntervalValue(cfg, 3, jnp.float32, True)
    params = jax.jit(bypass.init)(jax.random.key(13), rows, jnp.array(0), root)
    control = jax.jit(latent.init)(
        jax.random.key(13), rows, jax.nn.one_hot(0, cfg.action_classes), root
    )
    for name in ["row_read", "delta"]:
        for actual, expected in zip(
            jax.tree.leaves(params["params"][name]),
            jax.tree.leaves(control["params"][name]),
        ):
            np.testing.assert_array_equal(actual, expected)
    apply = jax.jit(bypass.apply)
    np.testing.assert_array_equal(apply(params, rows, jnp.array(0), root), root)

    def loss(variables: dict) -> jax.Array:
        return optax.softmax_cross_entropy(
            bypass.apply(variables, rows, jnp.array(0), root),
            jnp.array([1.0, 0.0, 0.0]),
        )

    gradient = jax.jit(jax.grad(loss))(params)
    assert float(jnp.linalg.norm(gradient["params"]["delta"]["Dense_2"]["kernel"])) > 0
    opened = copy.deepcopy(params)
    kernel = opened["params"]["delta"]["Dense_2"]["kernel"]
    opened["params"]["delta"]["Dense_2"]["kernel"] = jax.random.normal(
        jax.random.key(14), kernel.shape
    )
    assert not np.allclose(
        apply(opened, rows, jnp.array(0), root), apply(opened, rows, jnp.array(1), root)
    )
    opened_gradient = jax.jit(jax.grad(loss))(opened)
    assert (
        float(jnp.linalg.norm(opened_gradient["params"]["action_projection"]["kernel"]))
        > 0
    )
