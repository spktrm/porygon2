"""What makes `player_consequence_trunk_grad = 0` an observer of the trunk:
no gradient reaches it, AND the heads' gradients cannot move the multiplier
the global-norm clip applies to it. The second half is pinned on optimiser
UPDATES -- a gradient-reach test cannot see a clip coupling."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rl.environment.interfaces import ConsequenceInputs
from rl.online.artifact import optimiser_partitions, player_optimiser
from rl.online.config import get_learner_config
from rl.online.training.consequence import live_trunk_gradient, scale_trunk_gradient


def _variables() -> dict:
    return {
        "params": {
            "encoder": {"trunk": {"kernel": jnp.ones((4, 4))}},
            "action_head": {"kernel": jnp.ones((4, 2))},
            "consequence": {"sampler": {"out_proj": {"kernel": jnp.ones((4, 4))}}},
        }
    }


def _gradients(head_scale: float) -> dict:
    # Far above the clip (10), so the multiplier is what the test reads.
    gradients = jax.tree.map(lambda leaf: jnp.full_like(leaf, 100.0), _variables())
    gradients["params"]["consequence"] = jax.tree.map(
        lambda leaf: leaf * head_scale, gradients["params"]["consequence"]
    )
    return gradients


def test_partitions_name_the_consequence_subtree_alone() -> None:
    labels = optimiser_partitions(_variables())["params"]
    assert set(jax.tree.leaves(labels["consequence"])) == {"consequence"}
    assert set(jax.tree.leaves(labels["encoder"])) == {"model"}
    assert set(jax.tree.leaves(labels["action_head"])) == {"model"}


def test_head_gradients_do_not_move_the_models_update() -> None:
    config = get_learner_config()
    variables = _variables()
    optimiser = player_optimiser(config)
    state = optimiser.init(variables)
    quiet, _ = optimiser.update(_gradients(0.0), state, variables)
    loud, _ = optimiser.update(_gradients(1000.0), state, variables)
    for name in ("encoder", "action_head"):
        jax.tree.map(
            np.testing.assert_array_equal, quiet["params"][name], loud["params"][name]
        )

    # The control: ONE chain over the whole tree, and the same head gradients
    # shrink the model's update through the shared clip multiplier.
    shared = optax.chain(
        optax.clip_by_global_norm(config.player_clip_gradient),
        optax.sgd(1.0),
    )
    shared_state = shared.init(variables)
    shared_quiet, _ = shared.update(_gradients(0.0), shared_state, variables)
    shared_loud, _ = shared.update(_gradients(1000.0), shared_state, variables)
    assert not np.allclose(
        shared_quiet["params"]["encoder"]["trunk"]["kernel"],
        shared_loud["params"]["encoder"]["trunk"]["kernel"],
    )


def test_trunk_gradient_scale_changes_the_gradient_and_never_the_value() -> None:
    rows = jax.random.normal(jax.random.key(0), (16, 8))
    inputs = ConsequenceInputs(
        state_rows=rows,
        state_valid=jnp.ones(16, bool),
        source_row=rows[0],
        target_row=rows[1],
        cls_row=rows[2],
    )

    def reach(scale: float) -> jax.Array:
        def loss(state_rows):
            scaled = scale_trunk_gradient(
                dataclasses.replace(inputs, state_rows=state_rows), scale
            )
            return jnp.square(scaled.state_rows).sum()

        return jax.grad(loss)(rows)

    np.testing.assert_array_equal(np.asarray(reach(0.0)), 0.0)
    np.testing.assert_allclose(reach(0.25), 0.25 * reach(1.0), rtol=1e-6)
    assert np.abs(np.asarray(reach(1.0))).max() > 0
    np.testing.assert_array_equal(
        np.asarray(scale_trunk_gradient(inputs, 0.0).state_rows), np.asarray(rows)
    )


def test_the_ramp_is_linear_between_its_ends_and_zero_when_the_knob_is() -> None:
    config = dataclasses.replace(
        get_learner_config(),
        player_consequence_trunk_grad=0.8,
        player_consequence_ramp_start_step=1000,
        player_consequence_ramp_steps=200,
    )
    steps = jnp.asarray([0, 1000, 1100, 1200, 50000], jnp.int32)
    ramp = jax.vmap(lambda step: live_trunk_gradient(step, config))(steps)
    np.testing.assert_allclose(ramp, [0.0, 0.0, 0.4, 0.8, 0.8], atol=1e-6)
    off = dataclasses.replace(config, player_consequence_trunk_grad=0.0)
    np.testing.assert_array_equal(
        np.asarray(jax.vmap(lambda step: live_trunk_gradient(step, off))(steps)), 0.0
    )
