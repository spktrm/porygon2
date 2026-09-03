"""The dynamics head (2026-09-03): one-step latent self-prediction of the
public / my-private / field rows' next pre-trunk content.

Fast half: the alignment across a step and the loss bracket on synthetic
rows, each with the control that proves the test can fail. Slow half: on
the real model, the loss gradient reaches the head AND the encoder/trunk
(the term exists to shape them) and never the value heads.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.environment.data import NUM_ENTITY_PRIVATE_FEATURES
from rl.environment.interfaces import PlayerEnvOutput
from rl.environment.protos.features_pb2 import EntityPrivateNodeFeature, InfoFeature
from rl.model.constants import (
    DYNAMICS_GROUP_SLICES,
    NUM_DYNAMICS_ROWS,
    NUM_PRIVATE_SLOTS,
    NUM_PUBLIC_SLOTS,
)
from rl.model.player_model import dynamics_alignment
from rl.online.training.train_step import dynamics_losses

_ORDER = slice(
    InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0,
    InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11 + 1,
)
_IDX = EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ENTITY_IDX


def _env(order, entity_idx):
    info = np.zeros(len(InfoFeature.keys()), dtype=np.int32)
    info[_ORDER] = order
    private = np.zeros((NUM_PRIVATE_SLOTS, NUM_ENTITY_PRIVATE_FEATURES), np.int32)
    private[:, _IDX] = entity_idx
    return PlayerEnvOutput(info=jnp.asarray(info), private_team=jnp.asarray(private))


def test_alignment_follows_the_public_resort_and_the_private_key():
    order_now = np.arange(12, dtype=np.int32)
    order_now[11] = -1  # an unrevealed opponent slot
    # A switch on my side: rows 0 and 3 swap; the opponent half is fixed.
    order_next = order_now.copy()
    order_next[[0, 3]] = order_next[[3, 0]]
    idx_now = np.array([1, 2, 3, 0, 5, 6], dtype=np.int32)  # mon 3 never fielded
    idx_next = np.array([3, 2, 1, 0, 5, 6], dtype=np.int32)  # request re-sorted

    matched, next_index = jax.tree.map(
        np.asarray,
        dynamics_alignment(_env(order_now, idx_now), _env(order_next, idx_next)),
    )
    public = DYNAMICS_GROUP_SLICES["public"]
    private = DYNAMICS_GROUP_SLICES["private"]
    field = DYNAMICS_GROUP_SLICES["field"]
    expected_public = np.arange(12)
    expected_public[[0, 3]] = [3, 0]
    assert matched[public].tolist() == [True] * 11 + [False]
    assert next_index[public][:11].tolist() == expected_public[:11].tolist()
    assert matched[private].tolist() == [True, True, True, False, True, True]
    assert (next_index[private] - NUM_PUBLIC_SLOTS).tolist()[:3] == [2, 1, 0]
    assert matched[field].all()
    assert (next_index[field] - NUM_PUBLIC_SLOTS - NUM_PRIVATE_SLOTS).tolist() == [
        0,
        1,
        2,
    ]
    # Control: a different next-step order gives a different map.
    other_next = order_now.copy()
    other_next[[1, 2]] = other_next[[2, 1]]
    _, other_index = jax.tree.map(
        np.asarray,
        dynamics_alignment(_env(order_now, idx_now), _env(other_next, idx_next)),
    )
    assert other_index[public][:11].tolist() != next_index[public][:11].tolist()


def _batch_env(orders, entity_idx):
    """(T, B=1) env with the given per-step public orders."""
    steps = [_env(order, entity_idx) for order in orders]
    return jax.tree.map(lambda *leaves: jnp.stack(leaves)[:, None], *steps)


def test_losses_align_the_target_across_the_resort():
    num_steps, width = 4, 8
    rng = np.random.default_rng(0)
    # Each stable entity has ONE fixed content vector; rows carry them in
    # whatever order the step sorts them, so a perfect copy predictor
    # scores 0 only if the alignment undoes the resort.
    content = rng.normal(size=(NUM_DYNAMICS_ROWS, width)).astype(np.float32)
    orders = [np.arange(12, dtype=np.int32) for _ in range(num_steps)]
    orders[2][[0, 4]] = orders[2][[4, 0]]
    orders[3][[0, 4]] = orders[3][[4, 0]]
    entity_idx = np.arange(1, 7, dtype=np.int32)
    target = np.stack(
        [
            np.concatenate([content[order], content[12:18], content[18:]], axis=0)
            for order in orders
        ]
    )[:, None]
    env = _batch_env(orders, entity_idx)
    acted = jnp.ones((num_steps, 1), bool)
    valid = jnp.ones((num_steps, 1), bool)
    pred = jnp.asarray(target)  # any (T, B, R, D); only [:-1] is read

    _, logs = dynamics_losses(pred, jnp.asarray(target), env, acted, valid)
    assert float(logs["player_dynamics_copy_loss"]) == pytest.approx(0.0, abs=1e-6)
    assert float(logs["player_dynamics_rows_frac"]) == 1.0
    # Control: the same rows scored WITHOUT the alignment (identity map)
    # see the resort as change on the two swapped rows.
    swapped = jnp.asarray(target[1:]) - jnp.asarray(target[:-1])
    assert float(jnp.abs(swapped).max()) > 0.0

    # A perfect predictor of the ALIGNED next row scores 0 too, and a
    # negated one scores the maximum 2 -- the loss reads the prediction.
    _, next_index = jax.vmap(jax.vmap(dynamics_alignment))(
        jax.tree.map(lambda leaf: leaf[:-1], env),
        jax.tree.map(lambda leaf: leaf[1:], env),
    )
    aligned_next = jnp.take_along_axis(
        jnp.asarray(target[1:]), next_index[..., None], axis=2
    )
    perfect = jnp.concatenate([aligned_next, aligned_next[-1:]], axis=0)
    loss, _ = dynamics_losses(perfect, jnp.asarray(target), env, acted, valid)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    loss, _ = dynamics_losses(-perfect, jnp.asarray(target), env, acted, valid)
    assert float(loss) == pytest.approx(2.0, abs=1e-6)

    # Masks: the done row at t contributes nothing; a never-fielded mon
    # (ENTITY_IDX 0) is unmatched and lowers the row supply.
    acted_done = acted.at[1, 0].set(False)
    _, logs = dynamics_losses(-perfect, jnp.asarray(target), env, acted_done, valid)
    assert float(logs["player_dynamics_rows_frac"]) == 1.0
    unfielded_idx = entity_idx.copy()
    unfielded_idx[5] = 0
    env_unfielded = _batch_env(orders, unfielded_idx)
    _, logs = dynamics_losses(
        -perfect, jnp.asarray(target), env_unfielded, acted, valid
    )
    assert float(logs["player_dynamics_rows_frac"]) == pytest.approx(
        (NUM_DYNAMICS_ROWS - 1) / NUM_DYNAMICS_ROWS
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_dynamics_gradient_reaches_the_trunk_and_not_the_critics(
    real_model_and_trajectory,
):
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    num_steps = int(actor_input.env.done.shape[0])
    env = jax.tree.map(lambda leaf: leaf[:, None], actor_input.env)
    acted = jnp.ones((num_steps, 1), bool)

    def dynamics_only(params):
        out = network.apply(params, actor_input, actor_output, HeadParams())
        pred = out.dynamics_pred[:, None]
        target = jax.lax.stop_gradient(out.dynamics_target)[:, None]
        loss, _ = dynamics_losses(pred, target, env, acted, acted)
        return loss

    grads = jax.jit(jax.grad(dynamics_only))(params)
    reached = set()
    for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
        keys = tuple(entry.key for entry in path)
        if float(jnp.abs(leaf).max()) > 0.0:
            reached.add(keys[1])
    assert "dynamics_head" in reached
    assert "encoder" in reached
    for critic in ("value_head", "priv_value_head", "belief_head", "action_head"):
        assert critic not in reached, critic
