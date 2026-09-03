"""The delta dynamics head (2026-09-03; delta form 2026-09-04): one-step
latent self-prediction of the public / my-private / field rows' CHANGE
in pre-trunk content.

Fast half: the alignment across a step and the loss bracket on synthetic
rows, each with the control that proves the test can fail -- the zero
predictor scores exactly 1, a perfect one 0, and a constant added to both
rows (a persisted token) leaves the loss unchanged. Slow half: on the
real model, the loss gradient reaches the head AND the encoder/trunk (the
term exists to shape them) and never the value heads, and the zero-init
output kernel receives a non-zero gradient at init.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.environment.data import (
    NUM_ENTITY_PRIVATE_FEATURES,
    NUM_ENTITY_PUBLIC_FEATURES,
)
from rl.environment.interfaces import PlayerEnvOutput
from rl.environment.protos.features_pb2 import (
    EntityPrivateNodeFeature,
    EntityPublicNodeFeature,
    InfoFeature,
)
from rl.model.constants import (
    DYNAMICS_GROUP_SLICES,
    NUM_DYNAMICS_ROWS,
    NUM_PRIVATE_SLOTS,
    NUM_PUBLIC_SLOTS,
)
from rl.model.player_model import dynamics_alignment
from rl.model.utils import open_zero_init_paths
from rl.online.training.train_step import DYNAMICS_SCALE_FLOOR, dynamics_losses

_ORDER = slice(
    InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0,
    InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11 + 1,
)
_IDX = EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ENTITY_IDX
_HP = EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO


def _env(order, entity_idx, hp=None):
    info = np.zeros(len(InfoFeature.keys()), dtype=np.int32)
    info[_ORDER] = order
    private = np.zeros((NUM_PRIVATE_SLOTS, NUM_ENTITY_PRIVATE_FEATURES), np.int32)
    private[:, _IDX] = entity_idx
    public = np.zeros((NUM_PUBLIC_SLOTS, NUM_ENTITY_PUBLIC_FEATURES), np.int32)
    if hp is not None:
        public[:, _HP] = hp
    return PlayerEnvOutput(
        info=jnp.asarray(info),
        private_team=jnp.asarray(private),
        public_team=jnp.asarray(public),
    )


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


def _batch_env(orders, entity_idx, hps=None):
    """(T, B=1) env with the given per-step public orders."""
    if hps is None:
        hps = [None] * len(orders)
    steps = [_env(order, entity_idx, hp) for order, hp in zip(orders, hps)]
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

    zero = jnp.zeros_like(pred)
    loss, logs = dynamics_losses(zero, jnp.asarray(target), env, acted, valid)
    # Content is per entity, so every ALIGNED delta is exactly 0: the zero
    # predictor is perfect and each group's scale is 0 -- only if the
    # alignment undoes the resort. Control: the unaligned rows do see it.
    assert float(logs["player_dynamics_rows_frac"]) == 1.0
    assert float(loss) == 0.0
    for group in DYNAMICS_GROUP_SLICES:
        assert float(logs[f"player_dynamics_scale_{group}"]) == 0.0
    swapped = jnp.asarray(target[1:]) - jnp.asarray(target[:-1])
    assert float(jnp.abs(swapped).max()) > 0.0

    # Real change: perturb the next-step content per entity so the delta
    # is non-zero; a perfect predictor of the ALIGNED delta scores 0, the
    # zero predictor exactly 1, and the negated delta exactly 4.
    drift = rng.normal(size=target.shape).astype(np.float32) * 0.1
    target_moving = target + drift
    _, next_index = jax.vmap(jax.vmap(dynamics_alignment))(
        jax.tree.map(lambda leaf: leaf[:-1], env),
        jax.tree.map(lambda leaf: leaf[1:], env),
    )
    aligned_next = jnp.take_along_axis(
        jnp.asarray(target_moving[1:]), next_index[..., None], axis=2
    )
    delta = aligned_next - jnp.asarray(target_moving[:-1])
    perfect = jnp.concatenate([delta, delta[-1:]], axis=0)
    label = jnp.asarray(target_moving)
    loss, logs = dynamics_losses(perfect, label, env, acted, valid)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)
    for group in DYNAMICS_GROUP_SLICES:
        assert float(logs[f"player_dynamics_scale_{group}"]) > 0.0
    loss, _ = dynamics_losses(jnp.zeros_like(perfect), label, env, acted, valid)
    assert float(loss) == pytest.approx(1.0, abs=1e-5)
    loss, _ = dynamics_losses(-perfect, label, env, acted, valid)
    assert float(loss) == pytest.approx(4.0, abs=1e-4)

    # Copy-invariance: a constant added to BOTH rows of every entity (a
    # persisted token under the linear pool) leaves the loss unchanged --
    # the property the cosine form lacked. Control: adding it to the
    # next row only IS a change, and moves the loss.
    constant = rng.normal(size=(1, 1, 1, width)).astype(np.float32) * 3.0
    loss_shifted, _ = dynamics_losses(-perfect, label + constant, env, acted, valid)
    assert float(loss_shifted) == pytest.approx(4.0, abs=1e-4)
    label_next_only = label.at[1:].add(constant)
    loss_next_only, _ = dynamics_losses(-perfect, label_next_only, env, acted, valid)
    assert abs(float(loss_next_only) - 4.0) > 0.1

    # Masks: the done row at t contributes nothing; a never-fielded mon
    # (ENTITY_IDX 0) is unmatched and lowers the row supply.
    acted_done = acted.at[1, 0].set(False)
    _, logs = dynamics_losses(-perfect, label, env, acted_done, valid)
    assert float(logs["player_dynamics_rows_frac"]) == 1.0
    unfielded_idx = entity_idx.copy()
    unfielded_idx[5] = 0
    env_unfielded = _batch_env(orders, unfielded_idx)
    _, logs = dynamics_losses(-perfect, label, env_unfielded, acted, valid)
    assert float(logs["player_dynamics_rows_frac"]) == pytest.approx(
        (NUM_DYNAMICS_ROWS - 1) / NUM_DYNAMICS_ROWS
    )
    # An all-masked group is 0 and finite, not NaN: no valid step at all.
    loss, logs = dynamics_losses(-perfect, label, env, jnp.zeros_like(acted), valid)
    assert float(loss) == 0.0
    assert np.isfinite(float(logs["player_dynamics_gain_hp_moved"]))


def test_hp_moved_gain_reads_the_hp_rows_only():
    """`gain_hp_moved` is the public gain on rows whose wire HP_RATIO
    changed across the step (aligned), scaled on that subset; `hp_share`
    is the public delta's energy in the given subspace."""
    num_steps, width = 3, 8
    rng = np.random.default_rng(1)
    orders = [np.arange(12, dtype=np.int32) for _ in range(num_steps)]
    entity_idx = np.arange(1, 7, dtype=np.int32)
    hps = [np.full(12, 100, np.int32) for _ in range(num_steps)]
    hps[1][2] = 60  # row 2 loses hp across step 0 -> 1
    hps[2][2] = 60
    env = _batch_env(orders, entity_idx, hps)
    acted = jnp.ones((num_steps, 1), bool)
    target = jnp.asarray(rng.normal(size=(num_steps, 1, NUM_DYNAMICS_ROWS, width)))
    delta = target[1:] - target[:-1]
    perfect = jnp.concatenate([delta, delta[-1:]], axis=0)
    # Perfect on the moved row only: the moved-subset gain reads 1 while
    # the public gain is well below it.
    only_moved = jnp.zeros_like(perfect).at[:, :, 2].set(perfect[:, :, 2])
    _, logs = dynamics_losses(only_moved, target, env, acted, acted)
    assert float(logs["player_dynamics_gain_hp_moved"]) == pytest.approx(1.0, abs=1e-5)
    assert float(logs["player_dynamics_gain_public"]) < 0.5
    assert float(logs["player_dynamics_hp_moved_frac"]) == pytest.approx(
        1 / 24, abs=1e-6
    )
    # Control: the same prediction with hp unchanged has no moved rows.
    env_still = _batch_env(orders, entity_idx)
    _, logs = dynamics_losses(only_moved, target, env_still, acted, acted)
    assert float(logs["player_dynamics_hp_moved_frac"]) == 0.0
    assert float(logs["player_dynamics_gain_hp_moved"]) == pytest.approx(1.0)
    # hp_share: a basis spanning the whole space reads 1; the empty
    # (zero-column) basis reads 0.
    _, logs = dynamics_losses(
        only_moved, target, env, acted, acted, hp_basis=jnp.eye(width)
    )
    assert float(logs["player_dynamics_hp_share"]) == pytest.approx(1.0, rel=1e-4)
    _, logs = dynamics_losses(
        only_moved, target, env, acted, acted, hp_basis=jnp.zeros((width, 2))
    )
    assert float(logs["player_dynamics_hp_share"]) == 0.0


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

    grad_fn = jax.jit(jax.grad(dynamics_only))

    def reached_by(grads):
        reached = set()
        for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
            keys = tuple(entry.key for entry in path)
            if float(jnp.abs(leaf).max()) > 0.0:
                reached.add(keys[1])
        return reached

    # Fresh params: the output kernel is zero at init (the copy baseline)
    # and still receives a non-zero gradient -- one zero factor over live
    # inputs -- while NOTHING behind it does yet: the gradient into the
    # head's input is W_out^T times the residual, identically zero at step
    # 0. So the reach into the encoder is asserted on OPENED params below,
    # the two-halves rule for a zero-init path.
    fresh_grads = grad_fn(params)
    head = params["params"]["dynamics_delta_head"]
    assert float(jnp.abs(head["Dense_1"]["kernel"]).max()) == 0.0
    out_grad = fresh_grads["params"]["dynamics_delta_head"]["Dense_1"]["kernel"]
    assert float(jnp.abs(out_grad).max()) > 0.0
    assert reached_by(fresh_grads) == {"dynamics_delta_head"}

    opened = open_zero_init_paths(params, ["dynamics_delta_head"])
    reached = reached_by(grad_fn(opened))
    assert "dynamics_delta_head" in reached
    assert "encoder" in reached
    for critic in ("value_head", "priv_value_head", "belief_head", "action_head"):
        assert critic not in reached, critic


def test_zero_delta_group_is_floored_not_divided_by_zero():
    """A batch on which a group's rows did not move (the field rows, most
    batches) must not turn a small non-zero prediction into a loss of
    hundreds: the normaliser is floored at DYNAMICS_SCALE_FLOOR. Control:
    the same prediction against an ordinary-scale delta scores the plain
    ratio, well under the floored value."""
    num_steps, width = 4, 8
    rng = np.random.default_rng(1)
    content = rng.normal(size=(NUM_DYNAMICS_ROWS, width)).astype(np.float32)
    orders = [np.arange(12, dtype=np.int32) for _ in range(num_steps)]
    target = np.stack([content for _ in orders])[:, None]
    env = _batch_env(orders, np.arange(1, 7, dtype=np.int32))
    acted = jnp.ones((num_steps, 1), bool)
    valid = jnp.ones((num_steps, 1), bool)
    small = jnp.full(target.shape, 0.02, jnp.float32)
    loss, logs = dynamics_losses(small, jnp.asarray(target), env, acted, valid)
    for group in DYNAMICS_GROUP_SLICES:
        assert float(logs[f"player_dynamics_scale_{group}"]) == 0.0
    expected = width * 0.02**2 / DYNAMICS_SCALE_FLOOR
    assert float(loss) == pytest.approx(expected, rel=1e-4)
    assert float(loss) < 1.0
    assert bool(jnp.isfinite(loss))
    # Control: an ordinary delta scale is above the floor, so the floor is
    # inert and the loss is the raw ratio.
    moving = target + rng.normal(size=target.shape).astype(np.float32) * 0.5
    loss_moving, logs = dynamics_losses(small, jnp.asarray(moving), env, acted, valid)
    for group in DYNAMICS_GROUP_SLICES:
        assert float(logs[f"player_dynamics_scale_{group}"]) > DYNAMICS_SCALE_FLOOR
    assert float(loss_moving) == pytest.approx(1.0, abs=0.1)
