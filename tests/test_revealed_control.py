"""The revealed-row matched control beside the belief head (2026-09-04).

Two contracts, each with the positive control that proves it could fail:
its CE trains its own MLP and NOTHING else (the belief head's CE, same
labels, does reach the encoder), and it reads the matched mon's OWN
pre-trunk public row alone -- another public row's species does not move
it (it does move the belief head, which attends across rows).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rl.environment.protos.features_pb2 import EntityRevealedNodeFeature

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _with_column(actor_input, leaf, row, column, value):
    import dataclasses

    env = actor_input.env
    table = jnp.asarray(getattr(env, leaf)).at[:, row, column].set(value)
    return dataclasses.replace(
        actor_input, env=dataclasses.replace(env, **{leaf: table})
    )


def _control_and_belief_ce(network, actor_input, actor_output):
    from rl.model.heads import HeadParams

    def control_ce(params):
        out = network.apply(params, actor_input, actor_output, HeadParams())
        labels = jax.lax.stop_gradient(out.hidden_code.astype(jnp.float32))
        ce = optax.softmax_cross_entropy(
            logits=out.revealed_belief_logits.astype(jnp.float32), labels=labels
        )
        return jnp.mean(ce)

    def belief_ce(params):
        out = network.apply(params, actor_input, actor_output, HeadParams())
        labels = jax.lax.stop_gradient(out.hidden_code.astype(jnp.float32))
        ce = optax.softmax_cross_entropy(
            logits=out.belief_logits.astype(jnp.float32), labels=labels
        )
        return jnp.mean(ce)

    return control_ce, belief_ce


def test_revealed_control_gradient_stays_in_its_mlp(real_model_and_trajectory):
    network, params, actor_input, actor_output = real_model_and_trajectory
    control_ce, belief_ce = _control_and_belief_ce(network, actor_input, actor_output)

    grads = jax.jit(jax.grad(control_ce))(params)
    for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
        keys = tuple(entry.key for entry in path)
        if "revealed_belief" in keys:
            assert float(jnp.abs(leaf).max()) > 0.0, keys
        else:
            assert float(jnp.abs(leaf).max()) == 0.0, keys

    # Positive control: the belief head's CE on the SAME labels reaches
    # the encoder, so the stop_gradient above is what keeps the control's
    # out, not the labels being constant.
    belief_grads = jax.jit(jax.grad(belief_ce))(params)
    encoder_norm = sum(
        float(jnp.abs(leaf).max())
        for path, leaf in jax.tree_util.tree_leaves_with_path(belief_grads)
        if "encoder" in tuple(entry.key for entry in path)
    )
    assert encoder_norm > 0.0


def test_revealed_control_reads_the_matched_row_only(
    real_model_and_trajectory, real_model_apply
):
    from rl.model.heads import HeadParams
    from rl.model.player_model import belief_alignment

    network, params, actor_input, actor_output = real_model_and_trajectory
    base = real_model_apply(params, actor_input, actor_output, HeadParams())
    matched = np.asarray(base.belief_matched)
    step, mon = np.argwhere(matched)[0]
    _, public_row_index = belief_alignment(
        actor_input.env.opp_private_team[step], actor_input.env.info[step]
    )
    own_row = int(public_row_index[mon])
    other_row = (own_row + 1) % 12
    assert other_row != own_row

    species_column = EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
    current = int(actor_input.env.revealed_team[step, other_row, species_column])
    other_flipped = _with_column(
        actor_input, "revealed_team", other_row, species_column, current + 1
    )
    moved = real_model_apply(params, other_flipped, actor_output, HeadParams())
    np.testing.assert_array_equal(
        np.asarray(base.revealed_belief_logits[step, mon], dtype=np.float32),
        np.asarray(moved.revealed_belief_logits[step, mon], dtype=np.float32),
    )
    # Positive control: the belief head attends across the public rows,
    # so the same flip moves ITS logits for the same mon.
    assert not np.array_equal(
        np.asarray(base.belief_logits[step, mon], dtype=np.float32),
        np.asarray(moved.belief_logits[step, mon], dtype=np.float32),
    )

    own_current = int(actor_input.env.revealed_team[step, own_row, species_column])
    own_flipped = _with_column(
        actor_input, "revealed_team", own_row, species_column, own_current + 1
    )
    own_moved = real_model_apply(params, own_flipped, actor_output, HeadParams())
    assert not np.array_equal(
        np.asarray(base.revealed_belief_logits[step, mon], dtype=np.float32),
        np.asarray(own_moved.revealed_belief_logits[step, mon], dtype=np.float32),
    )
