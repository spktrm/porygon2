"""The position potential rides the info vector for the learner's PBRS
channel (2026-09-11). The policy must never read it: the potential is a
learner-side training signal, not a model input."""

import jax
import numpy as np
import pytest

from rl.environment.protos.features_pb2 import InfoFeature
from rl.model.heads import HeadParams

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _with_info(actor_input, slot, value):
    info = np.asarray(actor_input.env.info).copy()
    info[..., slot] = value
    return actor_input.replace(env=actor_input.env.replace(info=info))


def _leaves(output):
    return [np.asarray(leaf, dtype=np.float32) for leaf in jax.tree.leaves(output)]


def test_the_model_never_reads_the_potential_slot(
    real_model_and_trajectory, real_model_apply
):
    _, params, actor_input, actor_output = real_model_and_trajectory
    base = _leaves(real_model_apply(params, actor_input, actor_output, HeadParams()))

    moved_input = _with_info(
        actor_input, InfoFeature.INFO_FEATURE__STATE_POTENTIAL, 12345
    )
    moved = _leaves(real_model_apply(params, moved_input, actor_output, HeadParams()))
    for base_leaf, moved_leaf in zip(base, moved, strict=True):
        np.testing.assert_array_equal(base_leaf, moved_leaf)

    # Positive control: a slot the encoder DOES read moves the output, so the
    # comparison above can fail.
    has_previous = np.asarray(actor_input.env.info)[
        ..., InfoFeature.INFO_FEATURE__HAS_PREV_ACTION
    ]
    control_input = _with_info(
        actor_input, InfoFeature.INFO_FEATURE__HAS_PREV_ACTION, 1 - has_previous
    )
    control = _leaves(
        real_model_apply(params, control_input, actor_output, HeadParams())
    )
    assert any(
        not np.array_equal(base_leaf, control_leaf)
        for base_leaf, control_leaf in zip(base, control, strict=True)
    )


def test_the_potential_loss_reaches_only_its_head(real_model_and_trajectory):
    """The head reads CLS under stop_gradient, so a loss on its output (the
    human-fitted potential's channel returns) reaches its own params and
    nothing it reads."""
    import jax.numpy as jnp

    from tests.conftest import open_zero_init_paths

    network, params, actor_input, actor_output = real_model_and_trajectory
    target = jnp.asarray(
        np.random.default_rng(0).uniform(-1, 1, actor_input.env.done.shape),
        jnp.float32,
    )

    def potential_loss(params: dict) -> jax.Array:
        output = network.apply(params, actor_input, actor_output, HeadParams())
        return jnp.mean(jnp.square(output.potential_head.logits - target))

    def cls_value_loss(params: dict) -> jax.Array:
        output = network.apply(params, actor_input, actor_output, HeadParams())
        return jnp.mean(output.value_head.expectation)

    def reached_by(grads: dict) -> set[str]:
        reached = set()
        for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
            if float(jnp.abs(leaf).max()) > 0.0:
                reached.add(path[1].key)
        return reached

    # The head's last kernel is zero-init, which alone would block every
    # gradient behind it -- opened, so "the trunk is not reached" cannot
    # pass vacuously.
    opened = open_zero_init_paths(params, ["potential_head"])
    assert reached_by(jax.jit(jax.grad(potential_loss))(opened)) == {"potential_head"}
    # Control: the same CLS row read WITHOUT stop_gradient reaches the trunk,
    # so the instrument can see what the stop_gradient removes.
    assert "encoder" in reached_by(jax.jit(jax.grad(cls_value_loss))(opened))
