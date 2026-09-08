"""Posterior visibility, ancestral deployment and matched-control contracts."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.model.config import get_player_model_config
from rl.model.constants import POLICY_READABLE_ROWS, SEQUENCE_GROUP_IDS, SequenceGroup
from rl.model.interval_transition import (
    IntervalTransition,
    posterior_movement,
    warm_start_decoder,
)
from rl.offline.train_interval import balanced_kl, consistency_terms, game_bootstrap


@pytest.fixture(scope="module")
def interval_models():
    cfg = get_player_model_config(9, train=True).transition
    cfg.block.model_size = 32
    cfg.block.hidden_size = 64
    cfg.block.num_heads = 2
    cfg.block.qk_size = 16
    cfg.block.v_size = 16
    cfg.block.num_blocks = 1
    cfg.row_read_width = 2
    cfg.prior.mlp.layer_sizes = (32, 32)
    cfg.posterior.mlp.layer_sizes = (32, 32)
    rows = jax.random.normal(jax.random.key(1), (len(POLICY_READABLE_ROWS), 32))
    successor = rows + jax.random.normal(jax.random.key(2), rows.shape)
    action = jax.nn.one_hot(3, cfg.action_classes)
    models = {}
    for evidence in ("combined", "history"):
        model = IntervalTransition(cfg, jnp.float32, evidence)
        variables = jax.jit(model.init)(
            jax.random.key(3), rows, action, successor, jax.random.key(4)
        )
        models[evidence] = (model, variables, jax.jit(model.apply))
    return models, rows, successor, action


def test_matched_arms_have_identical_initial_parameters(interval_models):
    models, *_ = interval_models
    combined = models["combined"][1]
    history = models["history"][1]
    assert jax.tree.structure(combined) == jax.tree.structure(history)
    for combined_leaf, history_leaf in zip(
        jax.tree.leaves(combined), jax.tree.leaves(history)
    ):
        np.testing.assert_array_equal(combined_leaf, history_leaf)


def test_history_posterior_visibility_has_positive_controls(interval_models):
    models, rows, successor, action = interval_models
    groups = np.asarray(SEQUENCE_GROUP_IDS[POLICY_READABLE_ROWS])
    nonhistory_row = int(np.flatnonzero(groups == SequenceGroup.PRIVATE_ENTITY)[0])
    history_row = int(np.flatnonzero(groups == SequenceGroup.HISTORY_ENTITY)[0])
    changed = successor.at[nonhistory_row, 0].add(50)
    changed_history = successor.at[history_row, 0].add(50)
    model, variables, apply = models["history"]
    original = apply(variables, rows, action, successor, jax.random.key(5))
    restricted = apply(variables, rows, action, changed, jax.random.key(5))
    visible = apply(variables, rows, action, changed_history, jax.random.key(5))
    np.testing.assert_array_equal(
        original.posterior_logits[0], restricted.posterior_logits[0]
    )
    assert not np.allclose(original.posterior_logits[1], restricted.posterior_logits[1])
    assert not np.allclose(original.posterior_logits[0], visible.posterior_logits[0])
    _, combined_variables, combined_apply = models["combined"]
    reference = combined_apply(
        combined_variables, rows, action, successor, jax.random.key(5)
    )
    unrestricted = combined_apply(
        combined_variables, rows, action, changed, jax.random.key(5)
    )
    assert not np.allclose(
        reference.posterior_logits[0], unrestricted.posterior_logits[0]
    )


def test_prior_is_independent_of_posterior_parameters_with_live_decoder(
    interval_models,
):
    models, rows, _, action = interval_models
    model, variables, _ = models["history"]
    variables = copy.deepcopy(variables)
    kernel = variables["params"]["dynamics"]["dynamics_out_proj"]["kernel"]
    variables["params"]["dynamics"]["dynamics_out_proj"]["kernel"] = (
        jnp.eye(kernel.shape[0]) * 0.1
    )
    apply = jax.jit(
        lambda params: model.apply(
            params, rows, action, jax.random.key(19), method=model.sample_prior
        )
    )
    original = apply(variables)
    changed = copy.deepcopy(variables)
    for name in ("behaviour_posterior", "residual_posterior"):
        changed["params"][name] = jax.tree.map(
            lambda value: value + 10, changed["params"][name]
        )
    np.testing.assert_array_equal(original, apply(changed))
    assert not np.allclose(original, rows)
    changed["params"]["dynamics"]["code_table"] += 5
    assert not np.allclose(original, apply(changed))


def test_dynamics_kl_does_not_backpropagate_through_posterior_choice(interval_models):
    models, rows, successor, action = interval_models
    model, variables, _ = models["history"]

    def loss(next_rows):
        prediction = model.apply(variables, rows, action, next_rows, jax.random.key(7))
        return balanced_kl(
            prediction.prior_logits, prediction.posterior_logits, 0.5, 0.0, 0.0
        )[0]

    gradient = jax.jit(jax.grad(loss))(successor)
    np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))


def test_warm_start_is_strict_and_copies_only_decoder(interval_models):
    models, *_ = interval_models
    params = models["history"][1]["params"]
    source = jax.tree.map(lambda value: value + 1, params["dynamics"])
    copied, names = warm_start_decoder(params, source)
    assert names
    for source_leaf, copied_leaf in zip(
        jax.tree.leaves(source), jax.tree.leaves(copied["dynamics"])
    ):
        np.testing.assert_array_equal(source_leaf, copied_leaf)
    np.testing.assert_array_equal(
        copied["behaviour_prior"]["Dense_0"]["kernel"],
        params["behaviour_prior"]["Dense_0"]["kernel"],
    )
    with pytest.raises(ValueError, match="Missing or incompatible"):
        warm_start_decoder(params, {})


def test_consistency_counts_disappearance_but_not_both_absent():
    rows = jnp.ones((2, len(POLICY_READABLE_ROWS), 4))
    successor = jnp.zeros_like(rows)
    valid = jnp.ones(rows.shape[:-1], bool)
    assert float(consistency_terms(rows, successor, rows, valid)) == pytest.approx(1)
    assert float(consistency_terms(rows, successor, successor, valid)) == pytest.approx(
        0
    )
    assert float(
        consistency_terms(rows, successor, rows * 1000, ~valid)
    ) == pytest.approx(0)


def test_bootstrap_pools_both_sides_by_game():
    games = np.asarray(["game1", "game1", "game2", "game2"])
    energy = np.asarray([1.0, 9.0, 2.0, 8.0])
    assert game_bootstrap(energy, energy, games) == [0.0, 0.0]
    assert game_bootstrap(np.zeros(4), energy, games) == [1.0, 1.0]


def test_unknown_evidence_rejected():
    with pytest.raises(ValueError, match="evidence"):
        posterior_movement(jnp.ones((1, 1)), jnp.ones((1, 1)), "unknown")
