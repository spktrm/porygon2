"""The latent transition model (2026-09-05, rl/model/transition.py).

Fast half, on synthetic rows: the grounding loss aligns the label across
the public resort with the copy predictor at exactly 1 and a perfect one
at 0 (the label is each row's CHANGE, placed in the NEXT step's layout); the hp-moved and
transition-split instruments read their subsets; the zero-delta floor;
the standalone module's init contract (g is the copy predictor, every
mask logit 0, the posterior reads t+1 where the prior does not, out_proj
is the ONE zero factor and receives a live gradient at init while the
code / action paths behind it do not until it opens); the loss bracket
finite on an all-masked batch with the KL halves' gradients landing on
the right side. Slow half, on the real model: the whole transition term's
gradient reaches the transition subtree and NOTHING it reads -- not the
encoder (both ends of the transition are under stop_gradient) and not
the shared readout / deployable critic (frozen params on imagined rows)
-- with the real losses as the control that those paths are live.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from ml_collections import ConfigDict

from rl.environment.data import (
    NUM_ENTITY_PRIVATE_FEATURES,
    NUM_ENTITY_PUBLIC_FEATURES,
    NUM_ENTITY_REVEALED_FEATURES,
    NUM_FIELD_FEATURES,
)
from rl.environment.interfaces import PlayerActorOutput, PlayerEnvOutput
from rl.environment.protos.features_pb2 import (
    EntityPrivateNodeFeature,
    EntityPublicNodeFeature,
    EntityRevealedNodeFeature,
    FieldFeature,
    InfoFeature,
)
from rl.model.constants import (
    DYNAMICS_GROUP_SLICES,
    NUM_DYNAMICS_ROWS,
    NUM_PRIVATE_SLOTS,
    NUM_PUBLIC_SLOTS,
    POLICY_READABLE_ROWS,
)
from rl.model.player_model import dynamics_alignment
from rl.model.utils import open_zero_init_paths
from rl.online.training.telemetry import action_axis_masks
from rl.online.training.train_step import (
    DYNAMICS_SCALE_FLOOR,
    dynamics_losses,
    masked_percentile,
    transition_edges,
    transition_losses,
    transition_reveals,
)

_ORDER = slice(
    InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0,
    InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11 + 1,
)
_IDX = EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ENTITY_IDX
_HP = EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO
_SIDE = EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE
NUM_CELLS = 295


def _env(order, entity_idx, hp=None, request_count=0, revealed=None, kind=0):
    info = np.zeros(len(InfoFeature.keys()), dtype=np.int32)
    info[_ORDER] = order
    info[InfoFeature.INFO_FEATURE__REQUEST_COUNT] = request_count
    info[InfoFeature.INFO_FEATURE__REQUEST_TYPE] = kind
    public = np.zeros((NUM_PUBLIC_SLOTS, NUM_ENTITY_PUBLIC_FEATURES), np.int32)
    public[:6, _SIDE] = 1  # rows 0-5 are mine (SIDE 1), 6-11 theirs
    if hp is not None:
        public[:, _HP] = hp
    if revealed is None:
        revealed = np.zeros((NUM_PUBLIC_SLOTS, NUM_ENTITY_REVEALED_FEATURES), np.int32)
    private = np.zeros((NUM_PRIVATE_SLOTS, NUM_ENTITY_PRIVATE_FEATURES), np.int32)
    private[:, _IDX] = entity_idx
    return PlayerEnvOutput(
        info=jnp.asarray(info),
        done=jnp.zeros((), bool),
        public_team=jnp.asarray(public),
        revealed_team=jnp.asarray(revealed),
        private_team=jnp.asarray(private),
        action_mask=jnp.zeros((NUM_CELLS,), bool).at[:3].set(True),
    )


def _batch_env(
    orders, entity_idx, hps=None, request_counts=None, revealed=None, kinds=None
):
    """(T, B=1) env with the given per-step public orders."""
    if hps is None:
        hps = [None] * len(orders)
    if request_counts is None:
        request_counts = list(range(len(orders)))
    if revealed is None:
        revealed = [None] * len(orders)
    if kinds is None:
        kinds = [0] * len(orders)
    steps = [
        _env(order, entity_idx, hp, count, shown, kind)
        for order, hp, count, shown, kind in zip(
            orders, hps, request_counts, revealed, kinds
        )
    ]
    return jax.tree.map(lambda *leaves: jnp.stack(leaves)[:, None], *steps)


def _history_field(request_counts):
    """(H, B=1) window whose valid steps carry the given request counts."""
    field = np.zeros((len(request_counts), 1, NUM_FIELD_FEATURES), np.int32)
    field[:, 0, FieldFeature.FIELD_FEATURE__VALID] = 1
    field[:, 0, FieldFeature.FIELD_FEATURE__REQUEST_COUNT] = request_counts
    return jnp.asarray(field)


def _next_index(env):
    _, next_index = jax.vmap(jax.vmap(dynamics_alignment))(
        jax.tree.map(lambda leaf: leaf[:-1], env),
        jax.tree.map(lambda leaf: leaf[1:], env),
    )
    return np.asarray(next_index)


def _in_next_layout(values, next_index):
    """(T-1, B, R, D) values indexed by the CURRENT step's rows, placed
    where the next step's layout has them, padded to T with a copy of the
    last step (the self-paired final row the loss masks)."""
    placed = np.zeros_like(values)
    np.put_along_axis(placed, next_index[..., None], values, axis=2)
    return jnp.asarray(np.concatenate([placed, placed[-1:]], axis=0))


def _perfect(label, next_index):
    """The grounding head's exact answer: each current row's t -> t+1
    change, placed where the next step's layout has that row. The copy
    predictor is the all-zero prediction."""
    label = np.asarray(label)
    aligned_next = np.take_along_axis(label[1:], next_index[..., None], axis=2)
    return _in_next_layout(aligned_next - label[:-1], next_index)


def _ground(pred, target, env, acted, valid, **kwargs):
    loss, logs, _ = dynamics_losses(
        pred, jax.lax.stop_gradient(pred), target, env, acted, valid, **kwargs
    )
    return loss, logs


def test_alignment_follows_the_public_resort_and_the_private_key():
    order_now = np.arange(12, dtype=np.int32)
    order_next = order_now.copy()
    order_next[[0, 1]] = order_next[[1, 0]]  # my actives swap
    order_next[[6, 8]] = order_next[[8, 6]]  # an opponent switch
    entity_now = np.array([1, 2, 3, 4, 5, 6], np.int32)
    entity_next = np.array([2, 1, 3, 4, 5, 6], np.int32)  # request re-sorted
    matched, next_index = dynamics_alignment(
        _env(order_now, entity_now), _env(order_next, entity_next)
    )
    matched, next_index = np.asarray(matched), np.asarray(next_index)
    assert matched.all()
    np.testing.assert_array_equal(
        next_index[:12], [1, 0, 2, 3, 4, 5, 8, 7, 6, 9, 10, 11]
    )
    np.testing.assert_array_equal(next_index[12:18], [13, 12, 14, 15, 16, 17])
    np.testing.assert_array_equal(next_index[18:], [18, 19, 20])
    # A never-fielded mon (idx 0) and one whose key left the request are
    # unmatched.
    entity_gone = np.array([1, 0, 3, 4, 5, 9], np.int32)
    matched, _ = dynamics_alignment(
        _env(order_now, entity_now), _env(order_next, entity_gone)
    )
    matched = np.asarray(matched)
    assert matched[:12].all()
    np.testing.assert_array_equal(
        matched[12:18], [True, False, True, True, True, False]
    )


def test_grounding_aligns_the_label_across_the_resort():
    num_steps, width = 4, 8
    rng = np.random.default_rng(0)
    # Each stable entity has ONE fixed content vector; rows carry them in
    # whatever order the step sorts them.
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
    next_index = _next_index(env)

    # Static content: every aligned delta is 0, so the scale is 0 per
    # group and the zero prediction (each entity's content carried to its
    # NEXT row) scores 0. Control: the unaligned rows do see the resort.
    loss, logs = _ground(
        jnp.zeros_like(jnp.asarray(target)), jnp.asarray(target), env, acted, valid
    )
    assert float(logs["player_transition_ground_rows_frac"]) == 1.0
    assert float(loss) == 0.0
    for group in DYNAMICS_GROUP_SLICES:
        assert float(logs[f"player_transition_ground_scale_{group}"]) == 0.0
    assert float(jnp.abs(jnp.asarray(target[1:] - target[:-1])).max()) > 0.0

    # Real change: the perfect predictor IS each row's aligned change in
    # the next step's layout, the zero prediction (copy) scores exactly
    # 1, and the reflected one (-delta) exactly 4.
    drift = rng.normal(size=target.shape).astype(np.float32) * 0.1
    label = jnp.asarray(target + drift)
    perfect = _perfect(label, next_index)
    loss, logs = _ground(perfect, label, env, acted, valid)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)
    for group in DYNAMICS_GROUP_SLICES:
        assert float(logs[f"player_transition_ground_scale_{group}"]) > 0.0
    loss, logs = _ground(jnp.zeros_like(label), label, env, acted, valid)
    assert float(loss) == pytest.approx(1.0, abs=1e-5)
    for group in DYNAMICS_GROUP_SLICES:
        assert float(logs[f"player_transition_gain_{group}"]) == pytest.approx(
            0.0, abs=1e-5
        )
    reflected = -perfect
    loss, _ = _ground(reflected, label, env, acted, valid)
    assert float(loss) == pytest.approx(4.0, abs=1e-4)

    # Copy-invariance: a constant added to the label at every step (a
    # persisted token) leaves the label's change, and the loss, unchanged.
    # Control: adding it to the next rows of the label only IS a change.
    constant = jnp.asarray(rng.normal(size=(1, 1, 1, width)).astype(np.float32) * 3.0)
    loss_shifted, _ = _ground(reflected, label + constant, env, acted, valid)
    assert float(loss_shifted) == pytest.approx(4.0, abs=1e-4)
    loss_next_only, _ = _ground(
        reflected, label.at[1:].add(constant), env, acted, valid
    )
    assert abs(float(loss_next_only) - 4.0) > 0.1

    # Masks: the done row at t contributes nothing; a never-fielded mon
    # (ENTITY_IDX 0) is unmatched and lowers the row supply; an all-masked
    # batch is 0 and finite.
    acted_done = acted.at[1, 0].set(False)
    _, logs = _ground(reflected, label, env, acted_done, valid)
    assert float(logs["player_transition_ground_rows_frac"]) == 1.0
    unfielded_idx = entity_idx.copy()
    unfielded_idx[5] = 0
    env_unfielded = _batch_env(orders, unfielded_idx)
    _, logs = _ground(reflected, label, env_unfielded, acted, valid)
    assert float(logs["player_transition_ground_rows_frac"]) == pytest.approx(
        (NUM_DYNAMICS_ROWS - 1) / NUM_DYNAMICS_ROWS
    )
    loss, logs = _ground(reflected, label, env, jnp.zeros_like(acted), valid)
    assert float(loss) == 0.0
    assert np.isfinite(float(logs["player_transition_gain_hp_moved"]))


def test_prior_panels_read_the_prior_decode():
    """`gain_public_prior` scores the second prediction: the posterior
    decode perfect and the prior decode a copy reads 1 / 0."""
    num_steps, width = 3, 8
    rng = np.random.default_rng(3)
    orders = [np.arange(12, dtype=np.int32) for _ in range(num_steps)]
    env = _batch_env(orders, np.arange(1, 7, dtype=np.int32))
    acted = jnp.ones((num_steps, 1), bool)
    label = jnp.asarray(rng.normal(size=(num_steps, 1, NUM_DYNAMICS_ROWS, width)))
    perfect = _perfect(label, _next_index(env))
    copy = jnp.zeros_like(label)
    _, logs, _ = dynamics_losses(perfect, copy, label, env, acted, acted)
    assert float(logs["player_transition_gain_public"]) == pytest.approx(1.0, abs=1e-5)
    assert float(logs["player_transition_gain_public_prior"]) == pytest.approx(
        0.0, abs=1e-5
    )


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
    label = jnp.asarray(rng.normal(size=(num_steps, 1, NUM_DYNAMICS_ROWS, width)))
    perfect = _perfect(label, _next_index(env))
    copy = jnp.zeros_like(label)
    # Perfect on the moved row only: the moved-subset gain reads 1 while
    # the public gain is well below it.
    only_moved = copy.at[:, :, 2].set(perfect[:, :, 2])
    _, logs = _ground(only_moved, label, env, acted, acted)
    assert float(logs["player_transition_gain_hp_moved"]) == pytest.approx(
        1.0, abs=1e-5
    )
    assert float(logs["player_transition_gain_public"]) < 0.5
    assert float(logs["player_transition_hp_moved_frac"]) == pytest.approx(
        1 / 24, abs=1e-6
    )
    # Control: the same prediction with hp unchanged has no moved rows.
    env_still = _batch_env(orders, entity_idx)
    _, logs = _ground(only_moved, label, env_still, acted, acted)
    assert float(logs["player_transition_hp_moved_frac"]) == 0.0
    assert float(logs["player_transition_gain_hp_moved"]) == pytest.approx(1.0)
    # hp_share: a basis spanning the whole space reads 1; the empty
    # (zero-column) basis reads 0.
    _, logs = _ground(only_moved, label, env, acted, acted, hp_basis=jnp.eye(width))
    assert float(logs["player_transition_hp_share"]) == pytest.approx(1.0, rel=1e-4)
    _, logs = _ground(
        only_moved, label, env, acted, acted, hp_basis=jnp.zeros((width, 2))
    )
    assert float(logs["player_transition_hp_share"]) == 0.0


def test_zero_delta_group_is_floored_not_divided_by_zero():
    """A batch on which a group's rows did not move must not turn a small
    non-zero error into a loss of hundreds: the normaliser is floored at
    DYNAMICS_SCALE_FLOOR. Control: an ordinary-scale delta scores the
    plain ratio, well under the floored value."""
    num_steps, width = 4, 8
    rng = np.random.default_rng(1)
    content = rng.normal(size=(NUM_DYNAMICS_ROWS, width)).astype(np.float32)
    orders = [np.arange(12, dtype=np.int32) for _ in range(num_steps)]
    label = jnp.asarray(np.stack([content for _ in orders])[:, None])
    env = _batch_env(orders, np.arange(1, 7, dtype=np.int32))
    acted = jnp.ones((num_steps, 1), bool)
    valid = jnp.ones((num_steps, 1), bool)
    perfect = _perfect(label, _next_index(env))
    loss, logs = _ground(perfect + 0.02, label, env, acted, valid)
    for group in DYNAMICS_GROUP_SLICES:
        assert float(logs[f"player_transition_ground_scale_{group}"]) == 0.0
    expected = width * 0.02**2 / DYNAMICS_SCALE_FLOOR
    assert float(loss) == pytest.approx(expected, rel=1e-4)
    assert float(loss) < 1.0
    assert bool(jnp.isfinite(loss))
    moving = label + jnp.asarray(rng.normal(size=label.shape).astype(np.float32) * 0.5)
    perfect_moving = _perfect(moving, _next_index(env))
    loss_moving, logs = _ground(perfect_moving + 0.02, moving, env, acted, valid)
    for group in DYNAMICS_GROUP_SLICES:
        assert (
            float(logs[f"player_transition_ground_scale_{group}"])
            > DYNAMICS_SCALE_FLOOR
        )
    assert float(loss_moving) == pytest.approx(0.0, abs=0.05)


def test_transition_splits_read_spanned_edges_and_opponent_reveals():
    """Edges stamped with request t+1 are the steps transition t -> t+1
    spans; an opponent row that changes an id token across a matched step
    is a reveal, my own row's is not. The split gains read their subsets:
    a predictor exact on the short transitions and a copy on the long one
    scores 1 on `_short` and 0 on `_long`, and the reveal split reads the
    same predictor the other way round."""
    num_steps, width = 4, 8
    rng = np.random.default_rng(2)
    orders = [np.arange(12, dtype=np.int32) for _ in range(num_steps)]
    # Transition 0->1 spans 1 edge, 1->2 spans 5, 2->3 spans 2.
    window = [1, 2, 2, 2, 2, 2, 3, 3]
    species = EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
    revealed = [
        np.zeros((NUM_PUBLIC_SLOTS, NUM_ENTITY_REVEALED_FEATURES), np.int32)
        for _ in range(num_steps)
    ]
    revealed[2][7, species] = 5  # an opponent reveal on transition 1 -> 2
    revealed[3][7, species] = 5
    revealed[3][1, species] = 9  # my own row: not an opponent reveal
    env = _batch_env(orders, np.arange(1, 7, dtype=np.int32), revealed=revealed)
    history_field = _history_field(window)

    edges = transition_edges(env, history_field)
    np.testing.assert_array_equal(np.asarray(edges)[:, 0], [1, 5, 2])
    matched, next_index = jax.vmap(jax.vmap(dynamics_alignment))(
        jax.tree.map(lambda leaf: leaf[:-1], env),
        jax.tree.map(lambda leaf: leaf[1:], env),
    )
    reveal = transition_reveals(env, matched, next_index)
    np.testing.assert_array_equal(np.asarray(reveal)[:, 0], [False, True, False])

    label = jnp.asarray(
        rng.normal(size=(num_steps, 1, NUM_DYNAMICS_ROWS, width)).astype(np.float32)
    )
    acted = jnp.ones((num_steps, 1), bool)
    valid = jnp.ones((num_steps, 1), bool)
    perfect = _perfect(label, np.asarray(next_index))
    copy = jnp.zeros_like(label)
    pred = copy.at[jnp.asarray([0, 2])].set(perfect[jnp.asarray([0, 2])])
    _, logs, splits = dynamics_losses(
        pred, pred, label, env, acted, valid, history_field=history_field
    )
    assert float(logs["player_transition_gain_public_short"]) == pytest.approx(1.0)
    assert float(logs["player_transition_gain_public_long"]) == pytest.approx(
        0.0, abs=1e-5
    )
    assert float(logs["player_transition_gain_public_no_reveal"]) == pytest.approx(1.0)
    assert float(logs["player_transition_gain_public_reveal"]) == pytest.approx(
        0.0, abs=1e-5
    )
    assert float(logs["player_transition_edges_mean"]) == pytest.approx(8 / 3)
    assert float(logs["player_transition_edges_p90"]) == 5.0
    assert float(logs["player_transition_reveal_frac"]) == pytest.approx(1 / 3)
    assert set(splits) == {"short", "long", "reveal", "no_reveal"}


def test_masked_percentile_ignores_masked_out_values():
    values = jnp.asarray([[9, 1, 2], [3, 4, 100]])
    mask = jnp.asarray([[False, True, True], [True, True, False]])
    assert float(masked_percentile(values, mask, 0.0)) == 1.0
    assert float(masked_percentile(values, mask, 1.0)) == 4.0
    assert float(masked_percentile(values, mask, 1 / 3)) == 2.0
    assert float(masked_percentile(values, jnp.zeros_like(mask), 0.9)) == -1.0


# ---- the standalone module -------------------------------------------


def _small_transition_cfg(code_groups=2):
    from rl.model.config import get_player_model_config

    cfg = ConfigDict(get_player_model_config(generation=9, train=True).transition)
    cfg.code_groups = code_groups
    cfg.prior.mlp.layer_sizes = (64, code_groups * cfg.code_classes)
    cfg.posterior.mlp.layer_sizes = cfg.prior.mlp.layer_sizes
    return cfg


def _module(code_groups=2):
    from rl.model.transition import TransitionModel

    cfg = _small_transition_cfg(code_groups)
    return TransitionModel(cfg, dtype=jnp.float32), cfg


def _rows(rng, num_steps, width=256):
    rows = jnp.asarray(
        rng.normal(size=(num_steps, len(POLICY_READABLE_ROWS), width)).astype(
            np.float32
        )
    )
    valid = jnp.ones(rows.shape[:2], bool).at[:, 5].set(False)
    return jnp.where(valid[..., None], rows, 0.0), valid


@pytest.fixture(scope="module")
def module_and_params():
    module, cfg = _module()
    rng = np.random.default_rng(0)
    rows, valid = _rows(rng, 3)
    cells = jnp.asarray([0, 7, 290])
    next_rows, next_valid = _rows(np.random.default_rng(1), 3)
    params = module.init(
        jax.random.PRNGKey(0), rows, valid, cells, next_rows, next_valid
    )
    apply = jax.jit(module.apply)
    return module, cfg, params, apply, (rows, valid, cells, next_rows, next_valid)


def test_module_init_is_the_copy_predictor_with_silent_readouts(module_and_params):
    module, cfg, params, apply, inputs = module_and_params
    rows, valid, cells, next_rows, next_valid = inputs
    out = apply(params, *inputs)
    # out_proj is zero: the imagined rows ARE the current rows, and the
    # prior decode too. Every mask logit is 0 (the readout's init
    # contract).
    np.testing.assert_array_equal(np.asarray(out.pred), np.asarray(rows))
    np.testing.assert_array_equal(np.asarray(out.mask_logits), 0.0)
    # The grounding head starts AT the copy predictor: zero change.
    np.testing.assert_array_equal(np.asarray(out.ground), 0.0)
    assert out.prior_logits.shape == (3, cfg.code_groups, cfg.code_classes)
    assert out.prior_logits.dtype == jnp.float32
    # A one-hot code per group, straight-through.
    np.testing.assert_allclose(np.asarray(out.post_one_hot).sum(-1), 1.0, atol=1e-6)
    # Invalid rows stay zero.
    assert float(jnp.abs(out.pred[:, 5]).max()) == 0.0


def test_posterior_reads_the_next_rows_and_the_prior_does_not(module_and_params):
    module, cfg, params, apply, inputs = module_and_params
    rows, valid, cells, next_rows, next_valid = inputs
    base = apply(params, *inputs)
    other_next, _ = _rows(np.random.default_rng(7), 3)
    moved = apply(params, rows, valid, cells, other_next, next_valid)
    np.testing.assert_array_equal(
        np.asarray(base.prior_logits), np.asarray(moved.prior_logits)
    )
    assert not np.array_equal(
        np.asarray(base.post_logits), np.asarray(moved.post_logits)
    )
    # Positive control for the prior: it reads the taken cell.
    other_cells = jnp.asarray([1, 8, 291])
    moved = apply(params, rows, valid, other_cells, next_rows, next_valid)
    assert not np.array_equal(
        np.asarray(base.prior_logits), np.asarray(moved.prior_logits)
    )
    # WHICH row changed is legible: the same delta placed on row 2 and on
    # row 9 gives different posterior logits. A mean pool over the rows
    # is invariant to this by construction, which is why the read is
    # per row and flattened in row order.
    delta = jnp.asarray(np.random.default_rng(11).normal(size=(3, 256)), jnp.float32)
    on_row_2 = apply(params, rows, valid, cells, rows.at[:, 2].add(delta), valid)
    on_row_9 = apply(params, rows, valid, cells, rows.at[:, 9].add(delta), valid)
    assert not np.array_equal(
        np.asarray(on_row_2.post_logits), np.asarray(on_row_9.post_logits)
    )


def test_row_read_masks_invalid_rows_and_reads_valid_ones():
    from rl.model.transition import RowRead

    read = RowRead(width=4, dtype=jnp.float32)
    rows = jnp.asarray(np.random.default_rng(0).normal(size=(6, 8)), jnp.float32)
    valid = jnp.asarray([True, True, False, True, False, True])
    params = read.init(jax.random.PRNGKey(0), rows, valid)
    base = read.apply(params, rows, valid).reshape(6, 4)
    # An invalid row contributes exactly zero, whatever its content.
    np.testing.assert_array_equal(np.asarray(base[2]), 0.0)
    garbage = rows.at[2].set(1e3)
    np.testing.assert_array_equal(
        np.asarray(read.apply(params, garbage, valid)), np.asarray(base.reshape(-1))
    )
    # Live-row control: the same edit on a valid row moves ITS slot only.
    moved = read.apply(params, rows.at[3].set(1e3), valid).reshape(6, 4)
    assert not np.array_equal(np.asarray(moved[3]), np.asarray(base[3]))
    np.testing.assert_array_equal(
        np.asarray(jnp.delete(moved, 3, axis=0)),
        np.asarray(jnp.delete(base, 3, axis=0)),
    )


def test_out_proj_is_the_single_zero_factor(module_and_params):
    module, cfg, params, apply, inputs = module_and_params
    probe = jax.random.normal(
        jax.random.PRNGKey(3), (3, len(POLICY_READABLE_ROWS), 256)
    )

    def imagined_energy(params):
        return jnp.sum(module.apply(params, *inputs).pred * probe)

    grads = jax.jit(jax.grad(imagined_energy))(params)["params"]

    def rms(tree):
        return float(
            jnp.sqrt(
                jnp.mean(
                    jnp.square(
                        jnp.concatenate(
                            [jnp.ravel(leaf) for leaf in jax.tree.leaves(tree)]
                        )
                    )
                )
            )
        )

    # Fresh: out_proj's gradient is live (one zero factor over live
    # inputs); everything behind it -- the blocks, the code and action
    # projections, the table -- is W_out^T times the residual, i.e. 0.
    assert rms(grads["out_proj"]) > 0.0
    for behind in ("blocks", "code_proj", "action_proj", "code_table"):
        assert rms(grads[behind]) == 0.0, behind
    opened = open_zero_init_paths(params, ["out_proj"])
    grads = jax.jit(jax.grad(imagined_energy))(opened)["params"]
    for behind in ("blocks", "code_proj", "action_proj", "code_table"):
        assert rms(grads[behind]) > 0.0, behind


def test_straight_through_sample_draws_from_the_posterior_with_a_key():
    """With a key the forward one-hot is a categorical DRAW from `probs`
    (every class the unimix floor reaches is decoded at its own rate);
    without one it is the argmax -- the old mode decode. Either way the
    backward is the probabilities' own gradient."""
    from rl.model.transition import straight_through_sample

    probs = jnp.asarray([[0.6, 0.3, 0.1, 0.0]] * 4000, jnp.float32).reshape(4000, 1, 4)
    mode = straight_through_sample(probs, None)
    np.testing.assert_array_equal(np.asarray(mode), np.asarray(probs > 0.5))
    key = jax.random.PRNGKey(3)
    drawn = straight_through_sample(probs, key)
    np.testing.assert_allclose(np.asarray(drawn).sum(-1), 1.0, atol=1e-6)
    # The mode decode is the CONTROL: its frequencies are one-hot on the
    # leading class; the draw's match the probabilities.
    np.testing.assert_array_equal(np.asarray(mode).mean(0)[0], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        np.asarray(drawn).mean(0)[0], [0.6, 0.3, 0.1, 0.0], atol=0.03
    )
    # A class at exactly zero mass is never drawn.
    assert float(np.asarray(drawn)[..., 3].sum()) == 0.0
    # Different keys give different draws; the same key the same draw.
    again = straight_through_sample(probs, jax.random.PRNGKey(4))
    assert not np.array_equal(np.asarray(drawn), np.asarray(again))
    np.testing.assert_array_equal(
        np.asarray(drawn), np.asarray(straight_through_sample(probs, key))
    )
    # Straight-through: d(out)/d(probs) is the identity in both forms.
    cotangent = jnp.asarray(
        np.random.default_rng(0).normal(size=probs.shape), jnp.float32
    )
    for rng in (None, key):
        _, vjp = jax.vjp(lambda p: straight_through_sample(p, rng), probs)
        np.testing.assert_allclose(np.asarray(vjp(cotangent)[0]), np.asarray(cotangent))


def test_module_samples_the_posterior_only_under_a_sampling_rng(module_and_params):
    """`apply` without a "sampling" rng decodes the posterior's mode
    (init, probes, the offline harness); with one it draws, so the
    decoded code is not always the mode and the key is split per step."""
    module, cfg, params, apply, inputs = module_and_params
    mode = apply(params, *inputs)
    np.testing.assert_array_equal(
        np.asarray(mode.post_one_hot.argmax(-1)),
        np.asarray(mode.post_logits.argmax(-1)),
    )
    # Flat posterior logits (fresh params) make the draw a near-uniform
    # categorical over 16 classes: across many keys the drawn class is
    # the mode a small fraction of the time, never always.
    # The rng'd trace is a different executable from the fixture's, so the
    # logits agree to kernel-selection precision, never bitwise.
    is_mode = []
    for seed in range(64):
        drawn = apply(params, *inputs, rngs={"sampling": jax.random.PRNGKey(seed)})
        np.testing.assert_allclose(
            np.asarray(drawn.post_logits),
            np.asarray(mode.post_logits),
            rtol=1e-4,
            atol=1e-5,
        )
        assert drawn.pred.shape == mode.pred.shape
        is_mode.append(
            np.asarray(drawn.post_one_hot.argmax(-1) == mode.post_logits.argmax(-1))
        )
    is_mode = np.stack(is_mode)
    assert 0.0 < is_mode.mean() < 0.5
    # Per-step keys: the three steps do not all draw the same class.
    drawn = apply(params, *inputs, rngs={"sampling": jax.random.PRNGKey(0)})
    classes = np.asarray(drawn.post_one_hot.argmax(-1))
    assert len({tuple(row) for row in classes}) > 1


def test_code_groups_zero_drops_the_code_path_only():
    module, cfg = _module(code_groups=0)
    rng = np.random.default_rng(0)
    rows, valid = _rows(rng, 2)
    cells = jnp.asarray([0, 7])
    next_rows, next_valid = _rows(np.random.default_rng(1), 2)
    params = module.init(
        jax.random.PRNGKey(0), rows, valid, cells, next_rows, next_valid
    )
    with_code, _ = _module(code_groups=2)
    params_with = with_code.init(
        jax.random.PRNGKey(0), rows, valid, cells, next_rows, next_valid
    )
    assert set(params_with["params"]) - set(params["params"]) == {
        "code_table",
        "code_proj",
        "row_read",
        "prior_read_net",
        "posterior_read_net",
    }
    out = jax.jit(module.apply)(params, rows, valid, cells, next_rows, next_valid)
    assert out.prior_logits.shape == (2, 0, cfg.code_classes)
    np.testing.assert_array_equal(np.asarray(out.pred), np.asarray(rows))
    # No code, no draw: a "sampling" rng is accepted and unused.
    with_rng = module.apply(
        params,
        rows,
        valid,
        cells,
        next_rows,
        next_valid,
        rngs={"sampling": jax.random.PRNGKey(0)},
    )
    np.testing.assert_array_equal(np.asarray(with_rng.pred), np.asarray(out.pred))


# ---- the loss bracket ------------------------------------------------


def _synthetic_pred(rng, num_steps, code_groups=2, code_classes=16, n_bins=None):
    from rl.environment.data import CAT_VF_SUPPORT
    from rl.environment.interfaces import CategoricalValueHeadOutput

    if n_bins is None:
        n_bins = len(CAT_VF_SUPPORT)
    support = jnp.asarray(CAT_VF_SUPPORT, jnp.float32)

    def value_head():
        logits = jnp.asarray(rng.normal(size=(num_steps, 1, n_bins)).astype(np.float32))
        log_probs = jax.nn.log_softmax(logits)
        return CategoricalValueHeadOutput(
            logits=logits, log_probs=log_probs, expectation=jnp.exp(log_probs) @ support
        )

    rows = len(POLICY_READABLE_ROWS)
    post_logits = jnp.asarray(
        rng.normal(size=(num_steps, 1, code_groups, code_classes)), jnp.float32
    )
    return PlayerActorOutput(
        transition_cons_err=jnp.asarray(rng.random((num_steps, 1, rows)), jnp.float32),
        transition_cons_scale=jnp.asarray(
            rng.random((num_steps, 1, rows)), jnp.float32
        ),
        transition_prior_logits=jnp.asarray(
            rng.normal(size=(num_steps, 1, code_groups, code_classes)), jnp.float32
        ),
        transition_post_logits=post_logits,
        transition_post_one_hot=jax.nn.one_hot(post_logits.argmax(-1), code_classes),
        value_head=value_head(),
        transition_value_head=value_head(),
        transition_value_head_prior=value_head(),
        transition_pred_rms=jnp.ones((1,), jnp.float32),
        transition_log_policy=jax.nn.log_softmax(
            jnp.asarray(rng.normal(size=(num_steps, 1, NUM_CELLS)), jnp.float32)
        ),
        transition_mask_logits=jnp.asarray(
            rng.normal(size=(num_steps, 1, NUM_CELLS)), jnp.float32
        ),
        transition_kind_logits=jnp.asarray(
            rng.normal(size=(num_steps, 1, 4)), jnp.float32
        ),
        transition_done_logit=jnp.asarray(rng.normal(size=(num_steps, 1)), jnp.float32),
    )


def _bracket_inputs(num_steps=4):
    from rl.environment.data import CAT_VF_SUPPORT
    from rl.online.config import Porygon2LearnerConfig

    rng = np.random.default_rng(5)
    pred = _synthetic_pred(rng, num_steps)
    orders = [np.arange(12, dtype=np.int32) for _ in range(num_steps)]
    kinds = ([0, 1, 0, 2] * num_steps)[:num_steps]
    env = _batch_env(orders, np.arange(1, 7, dtype=np.int32), kinds=kinds)
    # `_env` legalises the three switch cells only; one move cell (6, the
    # first move x target cell) makes `has_move` live so a taken switch
    # counts as VOLUNTARY, and the taken cell alternates switch / move.
    env = dataclasses.replace(env, action_mask=env.action_mask.at[..., 6].set(True))
    action_index = jnp.asarray(
        [[0, 6][step % 2] for step in range(num_steps)], jnp.int32
    )[:, None]
    acted = jnp.ones((num_steps, 1), bool)
    n_bins = len(CAT_VF_SUPPORT)
    win_returns = jax.nn.one_hot(
        jnp.asarray(rng.integers(0, n_bins, size=(num_steps, 1))), n_bins
    )
    v_target = jnp.zeros((num_steps, 1), jnp.float32)
    target_log_policy = jax.nn.log_softmax(
        jnp.asarray(rng.normal(size=(num_steps, 1, NUM_CELLS)), jnp.float32)
    )
    return dict(
        pred=pred,
        env_output=env,
        acted_mask=acted,
        value_mask=acted,
        policy_mask=acted,
        flat_action_mask=env.action_mask,
        win_returns=win_returns,
        v_target=v_target,
        target_log_policy=target_log_policy,
        cat_vf_support=jnp.asarray(CAT_VF_SUPPORT, jnp.float32),
        splits={},
        axis=action_axis_masks(env.action_mask, action_index),
        config=Porygon2LearnerConfig(),
    )


def test_bracket_is_finite_on_an_all_masked_batch_and_reads_its_labels():
    inputs = _bracket_inputs()
    loss, logs = transition_losses(**inputs)
    assert bool(jnp.isfinite(loss))
    for key, value in logs.items():
        assert np.isfinite(np.asarray(value, np.float32)).all(), key
    assert float(logs["player_transition_rows_frac"]) == 1.0  # of the T-1 transitions
    # kind labels are the NEXT step's request type: steps 1..3 = [1, 0, 2].
    kind_logits = np.asarray(inputs["pred"].transition_kind_logits)[:-1, 0]
    expected = np.mean(kind_logits.argmax(-1) == np.array([1, 0, 2]))
    assert float(logs["player_transition_kind_acc"]) == pytest.approx(expected)
    # A mask readout that says "legal" exactly on the legal cells is
    # perfect on both counts; one that says legal everywhere has full
    # recall and ~1% accuracy.
    legal = inputs["flat_action_mask"].astype(jnp.float32)
    exact = dataclasses.replace(
        inputs["pred"], transition_mask_logits=jnp.where(legal > 0, 5.0, -5.0)
    )
    _, exact_logs = transition_losses(**{**inputs, "pred": exact})
    assert float(exact_logs["player_transition_mask_acc"]) == 1.0
    assert float(exact_logs["player_transition_mask_recall"]) == 1.0
    assert float(exact_logs["player_transition_mask_exact_frac"]) == 1.0
    everything = dataclasses.replace(
        inputs["pred"], transition_mask_logits=jnp.full_like(legal, 5.0)
    )
    _, all_logs = transition_losses(**{**inputs, "pred": everything})
    assert float(all_logs["player_transition_mask_recall"]) == 1.0
    assert float(all_logs["player_transition_mask_acc"]) == pytest.approx(4 / NUM_CELLS)
    # All masked: every term 0 and finite, no NaN from an empty average.
    empty = jnp.zeros_like(inputs["acted_mask"])
    loss, logs = transition_losses(**{**inputs, "acted_mask": empty})
    assert float(loss) == 0.0
    for key, value in logs.items():
        assert np.isfinite(np.asarray(value, np.float32)).all(), key


def test_kl_halves_land_on_their_side_and_the_free_nats_clip():
    """dyn: the prior's gradient carries `dyn_coef`, the posterior's
    `rep_coef` (DreamerV3's balancing); under the free-nats clip a
    transition below F has no gradient at all. The prior/posterior are
    made to AGREE per transition where the clip should fire."""
    inputs = _bracket_inputs()
    # The balancing read is pinned at DreamerV3's rep 0.1 explicitly --
    # the default is 0.0 (Step 3b D) and would pass the posterior half
    # vacuously.
    config = dataclasses.replace(inputs["config"], player_transition_rep_coef=0.1)
    inputs = {**inputs, "config": config}

    def kl_part(prior_logits, post_logits, free_nats):
        pred = dataclasses.replace(
            inputs["pred"],
            transition_prior_logits=prior_logits,
            transition_post_logits=post_logits,
        )
        cfg = dataclasses.replace(config, player_transition_free_nats=free_nats)
        _, logs = transition_losses(**{**inputs, "pred": pred, "config": cfg})
        return logs["player_loss_transition_kl"]

    prior = inputs["pred"].transition_prior_logits
    post = inputs["pred"].transition_post_logits
    grad_prior, grad_post = jax.grad(kl_part, argnums=(0, 1))(prior, post, 0.0)
    assert float(jnp.abs(grad_prior).max()) > 0.0
    assert float(jnp.abs(grad_post).max()) > 0.0
    # Scale the two halves: doubling dyn_coef doubles the prior gradient
    # and leaves the posterior's untouched.
    doubled = dataclasses.replace(
        config, player_transition_dyn_coef=2 * config.player_transition_dyn_coef
    )
    inputs_doubled = {**inputs, "config": doubled}

    def kl_part_doubled(prior_logits, post_logits):
        pred = dataclasses.replace(
            inputs["pred"],
            transition_prior_logits=prior_logits,
            transition_post_logits=post_logits,
        )
        cfg = dataclasses.replace(doubled, player_transition_free_nats=0.0)
        _, logs = transition_losses(**{**inputs_doubled, "pred": pred, "config": cfg})
        return logs["player_loss_transition_kl"]

    grad_prior_2, grad_post_2 = jax.grad(kl_part_doubled, argnums=(0, 1))(prior, post)
    np.testing.assert_allclose(
        np.asarray(grad_prior_2), 2 * np.asarray(grad_prior), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(grad_post_2), np.asarray(grad_post), rtol=1e-5
    )
    # Free nats: identical prior and posterior have KL 0 < F, so the clip
    # holds the loss at F per transition with zero gradient; the
    # unclipped panel reads 0.
    grad_prior_free, grad_post_free = jax.grad(kl_part, argnums=(0, 1))(post, post, 1.0)
    assert float(jnp.abs(grad_prior_free).max()) == 0.0
    assert float(jnp.abs(grad_post_free).max()) == 0.0
    pred_same = dataclasses.replace(inputs["pred"], transition_prior_logits=post)
    _, logs = transition_losses(**{**inputs, "pred": pred_same})
    assert float(logs["player_transition_kl"]) == pytest.approx(0.0, abs=1e-6)
    assert float(logs["player_transition_kl_free_frac"]) == 1.0
    assert float(logs["player_loss_transition_kl"]) == pytest.approx(
        (config.player_transition_dyn_coef + config.player_transition_rep_coef)
        * config.player_transition_free_nats
    )
    assert float(logs["player_transition_prior_post_agree"]) == 1.0
    # rep_coef 0 (Step 3b D, Stochastic MuZero's form): the posterior's
    # KL-side gradient is exactly zero and the prior's is untouched -- the
    # rep 0.1 read above is the control that the half was live.
    rep_off = dataclasses.replace(config, player_transition_rep_coef=0.0)

    def kl_part_rep_off(prior_logits, post_logits):
        pred = dataclasses.replace(
            inputs["pred"],
            transition_prior_logits=prior_logits,
            transition_post_logits=post_logits,
        )
        cfg = dataclasses.replace(rep_off, player_transition_free_nats=0.0)
        _, logs = transition_losses(**{**inputs, "pred": pred, "config": cfg})
        return logs["player_loss_transition_kl"]

    grad_prior_off, grad_post_off = jax.grad(kl_part_rep_off, argnums=(0, 1))(
        prior, post
    )
    assert float(jnp.abs(grad_post_off).max()) == 0.0
    np.testing.assert_allclose(
        np.asarray(grad_prior_off), np.asarray(grad_prior), rtol=1e-5
    )


# ---- the real model --------------------------------------------------


def _network_with_frozen_transition_value_head():
    from rl.model.config import get_player_model_config
    from rl.model.player_model import get_player_model

    cfg = get_player_model_config(generation=9, train=True)
    cfg.transition.value_trains_v_head = False
    return get_player_model(cfg)


def _value_head_from_logits(logits):
    from rl.environment.data import CAT_VF_SUPPORT
    from rl.environment.interfaces import CategoricalValueHeadOutput

    log_probs = jax.nn.log_softmax(logits)
    support = jnp.asarray(CAT_VF_SUPPORT, jnp.float32)
    return CategoricalValueHeadOutput(
        logits=logits, log_probs=log_probs, expectation=jnp.exp(log_probs) @ support
    )


def test_value_delta_r2_and_value_gain_bracket_copy_and_real():
    """The calibration panels (Step 3b A) read the imagined value against
    the COPY predictor: an imagined head equal to the real head at t scores
    exactly 0 on the delta-R2 and on the gain, one equal to the real head
    at t+1 scores exactly 1 on both -- the posterior and the prior-mode
    variants each pinned at both ends, the switch / move splits with them,
    and a half-way head strictly between."""
    num_steps = 6
    inputs = _bracket_inputs(num_steps=num_steps)
    n_bins = int(inputs["win_returns"].shape[-1])
    # Real logits that FIT labels which differ between consecutive steps,
    # so the real next state beats the copy on the CE by more than the
    # gain's 1e-3 denominator floor.
    labels = jnp.asarray([[0], [1], [2], [0], [2], [1]], jnp.int32)
    win_returns = jax.nn.one_hot(labels, n_bins)
    real_logits = 4.0 * win_returns + jnp.asarray(
        np.random.default_rng(3).normal(scale=0.3, size=(num_steps, 1, n_bins)),
        jnp.float32,
    )
    real = _value_head_from_logits(real_logits)
    shifted = jax.tree.map(lambda leaf: jnp.concatenate([leaf[1:], leaf[-1:]]), real)
    inputs = {**inputs, "win_returns": win_returns}

    def read(imagined, prior):
        pred = dataclasses.replace(
            inputs["pred"],
            value_head=real,
            transition_value_head=imagined,
            transition_value_head_prior=prior,
        )
        _, logs = transition_losses(**{**inputs, "pred": pred})
        return {key: float(value) for key, value in logs.items()}

    copy_logs = read(imagined=real, prior=shifted)
    for name in ("", "_switch", "_move"):
        assert copy_logs[f"player_transition_value_delta_r2{name}"] == 0.0, name
    assert copy_logs["player_transition_value_gain"] == 0.0
    assert copy_logs["player_transition_value_delta_r2_prior"] == pytest.approx(
        1.0, abs=1e-5
    )
    assert copy_logs["player_transition_value_gain_prior"] == pytest.approx(
        1.0, abs=1e-5
    )
    assert copy_logs["player_transition_value_ce_copy"] > (
        copy_logs["player_transition_value_ce_real"] + 1e-3
    )

    real_logs = read(imagined=shifted, prior=real)
    for name in ("", "_switch", "_move"):
        assert real_logs[f"player_transition_value_delta_r2{name}"] == pytest.approx(
            1.0, abs=1e-5
        ), name
    assert real_logs["player_transition_value_gain"] == pytest.approx(1.0, abs=1e-5)
    assert real_logs["player_transition_value_delta_r2_prior"] == 0.0
    assert real_logs["player_transition_value_gain_prior"] == 0.0

    halfway = _value_head_from_logits(0.5 * (real.logits + shifted.logits))
    half_logs = read(imagined=halfway, prior=halfway)
    assert 0.0 < half_logs["player_transition_value_delta_r2"] < 1.0
    assert 0.0 < half_logs["player_transition_value_gain"] < 1.0

    # The split rows are the ones the axis names: the fixture legalises a
    # move on every step, so its taken switches are all VOLUNTARY.
    axis = inputs["axis"]
    assert bool(axis.has_move.all())
    assert [int(value) for value in axis.taken_switch[:, 0]] == [1, 0, 1, 0, 1, 0]


def test_cons_coef_zero_keeps_the_panels_and_drops_the_gradient():
    """`player_transition_cons_coef` multiplies the consistency loss in the
    SUM only: the logged loss and the per-group gains are identical at 0
    and 1, and the loss's gradient into the consistency error is exactly 0
    at coef 0 -- with coef 1 as the control that the path is live."""
    inputs = _bracket_inputs()
    logs_by_coef = {}
    grad_by_coef = {}
    for coef in (0.0, 1.0):
        config = dataclasses.replace(inputs["config"], player_transition_cons_coef=coef)

        def loss_of(cons_err, config=config):
            pred = dataclasses.replace(inputs["pred"], transition_cons_err=cons_err)
            loss, _ = transition_losses(**{**inputs, "pred": pred, "config": config})
            return loss

        _, logs = transition_losses(**{**inputs, "config": config})
        logs_by_coef[coef] = {key: float(value) for key, value in logs.items()}
        grad_by_coef[coef] = jax.grad(loss_of)(inputs["pred"].transition_cons_err)

    cons_keys = [
        key
        for key in logs_by_coef[1.0]
        if key.startswith("player_transition_cons_gain_")
        or key == "player_loss_transition_cons"
    ]
    assert "player_loss_transition_cons" in cons_keys
    for key in cons_keys:
        assert logs_by_coef[0.0][key] == logs_by_coef[1.0][key], key
    assert logs_by_coef[1.0]["player_loss_transition_cons"] > 0.0
    assert float(jnp.abs(grad_by_coef[0.0]).max()) == 0.0
    assert float(jnp.abs(grad_by_coef[1.0]).max()) > 0.0


@pytest.mark.gpu
@pytest.mark.slow
def test_trains_v_head_off_is_bit_identical_to_the_frozen_clone(
    real_model_and_trajectory,
):
    """The knob's off is the frozen clone of 2026-09-05: the same params
    through `value_trains_v_head` True and False give the same forward,
    every leaf of the imagined value head included -- WITH the control
    that the imagined head does read `v_head`'s params (perturbing them
    moves it), so the equality is not two heads ignoring the same
    parameters."""
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    frozen_network = _network_with_frozen_transition_value_head()
    opened = open_zero_init_paths(params, ["action_head", "out_proj"])

    live = jax.jit(network.apply)(opened, actor_input, actor_output, HeadParams())
    frozen = jax.jit(frozen_network.apply)(
        opened, actor_input, actor_output, HeadParams()
    )
    for (path, live_leaf), frozen_leaf in zip(
        jax.tree_util.tree_leaves_with_path(live), jax.tree.leaves(frozen)
    ):
        np.testing.assert_array_equal(
            np.asarray(live_leaf),
            np.asarray(frozen_leaf),
            err_msg=jax.tree_util.keystr(path),
        )

    perturbed = dict(opened)
    perturbed["params"] = dict(opened["params"])
    perturbed["params"]["v_head"] = jax.tree.map(
        lambda leaf: leaf + 0.1, opened["params"]["v_head"]
    )
    moved = jax.jit(frozen_network.apply)(
        perturbed, actor_input, actor_output, HeadParams()
    )
    assert not np.array_equal(
        np.asarray(live.transition_value_head.logits),
        np.asarray(moved.transition_value_head.logits),
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_transition_gradient_reaches_the_model_and_nothing_it_reads(
    real_model_and_trajectory,
):
    from rl.environment.data import CAT_VF_SUPPORT
    from rl.model.heads import HeadParams
    from rl.online.config import Porygon2LearnerConfig

    network, params, actor_input, actor_output = real_model_and_trajectory
    num_steps = int(actor_input.env.done.shape[0])
    env = jax.tree.map(lambda leaf: leaf[:, None], actor_input.env)
    acted = jnp.ones((num_steps, 1), bool)
    config = Porygon2LearnerConfig()
    n_bins = len(CAT_VF_SUPPORT)
    win_returns = jax.nn.one_hot(
        jnp.asarray(np.random.default_rng(0).integers(0, n_bins, size=(num_steps, 1))),
        n_bins,
    )

    def transition_only(params, network=network):
        out = network.apply(params, actor_input, actor_output, HeadParams())
        batched = jax.tree.map(lambda leaf: leaf[:, None], out)
        target = jax.lax.stop_gradient(batched.dynamics_target)
        loss_ground, _, splits = dynamics_losses(
            batched.transition_ground,
            batched.transition_ground_prior,
            target,
            env,
            acted,
            acted,
        )
        loss_rest, _ = transition_losses(
            batched,
            env,
            acted,
            acted,
            acted,
            env.action_mask,
            win_returns,
            jnp.zeros((num_steps, 1), jnp.float32),
            jax.lax.stop_gradient(batched.action_head.log_policy),
            jnp.asarray(CAT_VF_SUPPORT, jnp.float32),
            splits,
            action_axis_masks(env.action_mask, batched.action_head.action_index),
            config,
        )
        return loss_ground + loss_rest

    grad_fn = jax.jit(jax.grad(transition_only))

    def reached_by(grads):
        reached = set()
        for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
            keys = tuple(entry.key for entry in path)
            if float(jnp.abs(leaf).max()) > 0.0:
                reached.add(keys[1])
        return reached

    # The readout's `query` is zero-init, so pi is UNIFORM on every row and
    # the next-policy KL between pi(h-hat) and pi(h_{t+1}) is identically
    # 0 with a zero gradient -- opened (out_proj stays closed: g is still
    # the copy predictor) so a "the readout is not reached" read cannot
    # pass vacuously, per the conftest rule. With g the copy predictor the
    # next-policy KL through a LIVE readout is KL(pi_{t+1} || pi_t) on the
    # real trunk -- the anti-switch smoothing force of 2026-09-05 -- so
    # the transition term must reach its OWN subtree and nothing it reads:
    # not the encoder (g's input and the consistency target are both under
    # stop_gradient) and not the action readout (frozen params on imagined
    # rows). The one amendment (2026-09-06, Step 3b): the shared `v_head`
    # trains on the imagined CLS row under `value_trains_v_head`, so the
    # pin is {transition, v_head} with the knob on and exactly {transition}
    # with it off -- the on/off pair is the control for both halves.
    opened = open_zero_init_paths(params, ["action_head"])
    reached = reached_by(grad_fn(opened))
    assert (
        float(jnp.abs(opened["params"]["transition"]["out_proj"]["kernel"]).max())
        == 0.0
    )
    assert reached == {"transition", "v_head"}, reached

    frozen_network = _network_with_frozen_transition_value_head()
    reached_frozen = reached_by(
        jax.jit(jax.grad(lambda params: transition_only(params, frozen_network)))(
            opened
        )
    )
    assert reached_frozen == {"transition"}, reached_frozen

    # Positive control for the negative half: the same params, the same
    # `reached_by`, the REAL value CE and the real log-policy -- the shared
    # heads and the encoder are reachable, so their absence above is the
    # stop_gradients and not a dead path.
    def real_losses(params):
        out = network.apply(params, actor_input, actor_output, HeadParams())
        value_ce = optax.softmax_cross_entropy(
            logits=out.value_head.logits.astype(jnp.float32),
            labels=win_returns[:, 0],
        ).mean()
        return value_ce + out.action_head.log_policy.astype(jnp.float32).mean()

    control = reached_by(jax.jit(jax.grad(real_losses))(opened))
    for expected in ("encoder", "action_head", "v_head"):
        assert expected in control, expected
    assert "transition" not in control
