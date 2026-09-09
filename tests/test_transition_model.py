"""The latent transition model (2026-09-05; latent actions and the K-step
unroll 2026-09-07, rl/model/transition.py).

Fast half, on synthetic rows: the grounding loss aligns the label across
the public resort with the copy predictor at exactly 1 and a perfect one
at 0 (the label is each row's CHANGE, placed in the NEXT step's layout);
the hp-moved and transition-split instruments read their subsets; the
zero-delta floor; the free functions (the legal enumeration with its
overflow, the EXACT decode objective as conditional entropy with a
finite-difference gradient check and the collapsed stationary point,
Gumbel-top-k against sequential draws without replacement, the support
set, the prefix targets and the candidate loss at its exact conditional);
the standalone module's contracts (g is the copy predictor with token
conditioning, `dynamics_out_proj` the ONE zero factor, no validity or
legality input to `imagine` -- the real next rows move the posterior and
nothing rollout-side -- an appearing row is legible to the posterior and
imaginable from the slot embedding, the encoder and posterior are
recomputed at the imagined state, draws happen only under a sampling
rng, `generate` draws distinct codes inside its support); the loss
bracket finite on an all-masked batch with the unroll masks, the KL
halves, the copy/real calibration ends, the coefficient offs, the joint
termination NLL and the newly-valid split each pinned. Slow half, on the
real model: the whole transition term's gradient reaches the transition
subtree (and the shared critic under `value_trains_v_head`) and NOTHING
it reads -- not the encoder and not the action readout whose log_policy
the latent target is built from -- with the real losses as the control
that those paths are live.
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
        win_reward=jnp.zeros((3,), jnp.float32),
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


def test_legal_enumeration_indexes_the_taken_cell_and_counts_overflow():
    from rl.model.transition import legal_enumeration

    legal = jnp.zeros(NUM_CELLS, bool).at[jnp.asarray([4, 9, 30])].set(True)
    cells, cell_valid, taken_index, overflow = legal_enumeration(legal, 9, 16)
    assert cells[:3].tolist() == [4, 9, 30] and cell_valid.tolist()[:4] == [
        True,
        True,
        True,
        False,
    ]
    assert int(taken_index) == 1 and not bool(overflow)
    # The taken cell is not legal (a padded row): overflow, index 0.
    _, _, taken_index, overflow = legal_enumeration(legal, 5, 16)
    assert bool(overflow) and int(taken_index) == 0
    # More legal cells than the width: overflow even when the taken cell
    # sits inside the enumerated prefix -- never silently renormalised.
    wide = jnp.zeros(NUM_CELLS, bool).at[:20].set(True)
    _, _, _, overflow = legal_enumeration(wide, 0, 16)
    assert bool(overflow)
    _, _, _, overflow = legal_enumeration(jnp.zeros(NUM_CELLS, bool), 0, 16)
    assert bool(overflow)


def test_exact_decode_loss_is_conditional_entropy_and_matches_finite_differences():
    """The objective of plan §4 summed exactly over every code: a
    symmetric encoder (every legal cell the same distribution) sits at
    loss log(n), mutual information 0, with an exactly zero gradient (the
    collapsed stationary point); a distinct one-hot per cell reads near
    0 / log(n) with accuracy ~1 (the unimix floor is the gap); the
    analytic gradient of a random encoder matches central differences."""
    from rl.model.transition_objectives import exact_decode_loss

    num_valid, num_codes = 4, 8
    cell_valid = jnp.arange(6) < num_valid
    same = jnp.tile(
        jnp.asarray(np.random.default_rng(0).normal(size=num_codes)), (6, 1)
    )
    read = exact_decode_loss(same.astype(jnp.float32), cell_valid)
    np.testing.assert_allclose(float(read.loss), np.log(num_valid), rtol=1e-6)
    np.testing.assert_allclose(float(read.mutual_information), 0.0, atol=1e-6)
    grad = jax.grad(lambda logits: exact_decode_loss(logits, cell_valid).loss)(same)
    # Analytically zero; f32 leaves ~1e-8 of rounding.
    assert float(jnp.abs(grad).max()) < 1e-6
    distinct = 30.0 * jax.nn.one_hot(jnp.arange(6), num_codes)
    read = exact_decode_loss(distinct, cell_valid)
    assert float(read.loss) < 0.05 and float(read.accuracy) > 0.98
    assert float(read.mutual_information) > np.log(num_valid) - 0.05
    # f32 on purpose: the suite runs with x64 off, so a float64 request
    # would silently become f32 anyway.
    random = jnp.asarray(
        np.random.default_rng(1).normal(size=(6, num_codes)), jnp.float32
    )

    def loss_of(logits):
        return exact_decode_loss(logits, cell_valid).loss

    analytic = np.asarray(jax.grad(loss_of)(random))
    numeric = np.zeros_like(analytic)
    eps = 1e-3
    for cell in range(6):
        for code in range(num_codes):
            up = random.at[cell, code].add(eps)
            down = random.at[cell, code].add(-eps)
            numeric[cell, code] = (float(loss_of(up)) - float(loss_of(down))) / (
                2 * eps
            )
    # f32 central differences (x64 is off in the suite): ~1e-4 of noise
    # on gradients of magnitude ~1e-2.
    np.testing.assert_allclose(analytic, numeric, atol=5e-4)
    # Invalid cells carry no gradient; the bound 0 <= loss <= log(n).
    assert np.all(analytic[num_valid:] == 0.0)
    assert 0.0 <= float(loss_of(random)) <= np.log(num_valid) + 1e-6
    # An empty legal set reads 0 everywhere, finite.
    empty = exact_decode_loss(random.astype(jnp.float32), jnp.zeros(6, bool))
    assert float(empty.loss) == 0.0 and float(empty.accuracy) == 0.0


def test_gumbel_top_k_matches_sequential_draws_without_replacement():
    from rl.model.categoricals import gumbel_top_k

    probs = np.array([0.5, 0.3, 0.15, 0.05])
    log_probs = jnp.log(jnp.asarray(probs))
    # No key: the deterministic top-k.
    assert gumbel_top_k(log_probs, 3, None).tolist() == [0, 1, 2]
    keys = jax.random.split(jax.random.key(0), 20000)
    draws = np.asarray(jax.vmap(lambda key: gumbel_top_k(log_probs, 2, key))(keys))
    # Every draw is a pair of DISTINCT indices.
    assert np.all(draws[:, 0] != draws[:, 1])
    # First draw ~ p; second draw ~ p renormalised without the first
    # (the Plackett-Luce law), checked on the 0-then-x conditionals.
    first = np.bincount(draws[:, 0], minlength=4) / len(draws)
    np.testing.assert_allclose(first, probs, atol=0.02)
    after_zero = draws[draws[:, 0] == 0, 1]
    second = np.bincount(after_zero, minlength=4)[1:] / len(after_zero)
    np.testing.assert_allclose(second, probs[1:] / probs[1:].sum(), atol=0.03)
    # -inf entries are never drawn ahead of finite ones.
    masked = log_probs.at[1].set(-jnp.inf)
    draws = np.asarray(jax.vmap(lambda key: gumbel_top_k(masked, 3, key))(keys[:500]))
    assert not np.any(draws[:, :3] == 1) or np.all(draws[:, 2] != 1) is False


def test_support_set_is_the_smallest_prefix_holding_the_mass():
    from rl.model.categoricals import support_set

    probs = jnp.asarray([0.05, 0.5, 0.3, 0.1, 0.05])
    assert support_set(probs, 0.79).tolist() == [False, True, True, False, False]
    assert support_set(probs, 0.81).tolist() == [False, True, True, True, False]
    assert support_set(probs, 1.0).tolist() == [True] * 5
    # Ties resolve by index: two equal leaders, the lower index first.
    tied = jnp.asarray([0.4, 0.4, 0.2])
    assert support_set(tied, 0.3).tolist() == [True, False, False]


def test_candidate_targets_exclude_the_prefix_and_the_loss_reads_the_exact_conditional():
    from rl.model.transition_objectives import candidate_loss, candidate_targets

    p_target = jnp.asarray([0.4, 0.3, 0.2, 0.1, 0.0, 0.0])
    support = jnp.asarray([True, True, True, True, False, False])
    teacher = jnp.asarray([1, 0, 3, 5])  # slot 3's code is outside the support
    targets = candidate_targets(p_target, teacher, support)
    # Slot 0: the full target; slot 1: no code 1; slot 2: no codes 1, 0.
    np.testing.assert_allclose(np.asarray(targets.targets[0]), np.asarray(p_target))
    assert targets.allowed[0].all()
    expected_1 = np.array([0.4, 0.0, 0.2, 0.1, 0.0, 0.0]) / 0.7
    np.testing.assert_allclose(np.asarray(targets.targets[1]), expected_1, rtol=1e-6)
    assert not bool(targets.allowed[1, 1]) and not bool(targets.allowed[1, 4])
    expected_2 = np.array([0.0, 0.0, 0.2, 0.1, 0.0, 0.0]) / 0.3
    np.testing.assert_allclose(np.asarray(targets.targets[2]), expected_2, rtol=1e-6)
    assert targets.occupied.tolist() == [True, True, True, False]
    # The exact conditional scores CE - H = 0 on every occupied slot; a
    # perturbation scores > 0 (the positive control); drawn codes get
    # exactly zero probability.
    exact_logits = jnp.log(jnp.maximum(targets.targets, 1e-12))
    read = candidate_loss(exact_logits, targets)
    kl = np.asarray(read.cross_entropy - read.target_entropy)
    np.testing.assert_allclose(kl[:3], 0.0, atol=1e-5)
    # The loss is the CE (the target entropies remain): the first slot's
    # once and the mean of the two later occupied slots' once, averaged.
    entropy = np.asarray(read.target_entropy)
    np.testing.assert_allclose(
        float(read.loss), 0.5 * (entropy[0] + entropy[1:3].mean()), rtol=1e-5
    )
    perturbed = candidate_loss(exact_logits.at[1, 2].add(1.0), targets)
    assert float(perturbed.loss) > float(read.loss) + 1e-3
    from rl.model.categoricals import masked_log_softmax

    assert (
        float(jnp.exp(masked_log_softmax(exact_logits[1], targets.allowed[1]))[1])
        == 0.0
    )
    # One occupied slot: the loss is the first CE alone.
    single = candidate_targets(p_target, jnp.asarray([2, 5, 5, 5]), support)
    assert single.occupied.tolist() == [True, False, False, False]
    read = candidate_loss(exact_logits, single)
    np.testing.assert_allclose(
        float(read.loss), float(read.cross_entropy[0]), rtol=1e-6
    )


def _small_transition_cfg(code_groups=2, unroll_steps=2):
    from rl.model.config import get_player_model_config

    cfg = ConfigDict(get_player_model_config(generation=9, train=True).transition)
    cfg.code_groups = code_groups
    cfg.unroll_steps = unroll_steps
    cfg.prior.mlp.layer_sizes = (64, code_groups * cfg.code_classes)
    cfg.posterior.mlp.layer_sizes = cfg.prior.mlp.layer_sizes
    cfg.generator.num_blocks = 1
    cfg.action_encoder.mlp.layer_sizes = (64, cfg.action_classes)
    return cfg


def _module(code_groups=2, unroll_steps=2):
    from rl.model.transition import TransitionModel

    cfg = _small_transition_cfg(code_groups, unroll_steps)
    return TransitionModel(cfg, dtype=jnp.float32), cfg


def _rows(rng, num_steps, width=256):
    rows = jnp.asarray(
        rng.normal(size=(num_steps, len(POLICY_READABLE_ROWS), width)).astype(
            np.float32
        )
    )
    valid = jnp.ones(rows.shape[:2], bool).at[:, 5].set(False)
    return jnp.where(valid[..., None], rows, 0.0)


LEGAL_CELLS = [0, 1, 7, 8, 290]


def _inputs(rng, num_steps=3):
    rows = _rows(rng, num_steps)
    cells = jnp.asarray([0, 7, 290][:num_steps])
    legal = jnp.zeros((num_steps, NUM_CELLS), bool).at[:, LEGAL_CELLS].set(True)
    log_policy = jnp.where(legal, jnp.log(1.0 / len(LEGAL_CELLS)), 0.0)
    return rows, cells, legal, log_policy


@pytest.fixture(scope="module")
def module_and_params():
    module, cfg = _module()
    inputs = _inputs(np.random.default_rng(0))
    params = jax.jit(module.init)(jax.random.PRNGKey(0), *inputs)
    apply = jax.jit(module.apply)
    return module, cfg, params, apply, inputs


def test_module_init_is_the_copy_predictor_with_token_conditioning(module_and_params):
    module, cfg, params, apply, inputs = module_and_params
    rows = inputs[0]
    out = apply(params, *inputs)
    # dynamics_out_proj is zero: every unrolled state IS the root rows
    # (the second step imagines from the first, which is the root), and
    # the prior decode too. The zero row stays zero because the copy
    # holds, not because anything masked it.
    for offset in range(cfg.unroll_steps):
        np.testing.assert_array_equal(
            np.asarray(out.steps.pred[offset]), np.asarray(rows)
        )
    np.testing.assert_array_equal(np.asarray(out.first.pred_prior), np.asarray(rows))
    # The grounding head starts AT the copy predictor: zero change.
    np.testing.assert_array_equal(np.asarray(out.first.ground), 0.0)
    num_steps = rows.shape[0]
    assert out.steps.prior_logits.shape == (
        cfg.unroll_steps,
        num_steps,
        cfg.code_groups,
        cfg.code_classes,
    )
    assert out.steps.prior_logits.dtype == jnp.float32
    assert out.nodes.generator_logits.shape == (
        cfg.unroll_steps + 1,
        num_steps,
        cfg.num_candidates,
        cfg.action_classes,
    )
    assert out.root.action_logits.shape == (
        num_steps,
        cfg.max_cells,
        cfg.action_classes,
    )
    # One-hot codes, straight-through; the enumeration found every taken
    # cell without overflow.
    np.testing.assert_allclose(
        np.asarray(out.steps.post_one_hot).sum(-1), 1.0, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(out.steps.action_one_hot).sum(-1), 1.0, atol=1e-6
    )
    assert not bool(out.root.action_overflow.any()) and not bool(
        out.nodes.node_overflow.any()
    )
    assert out.root.action_taken_index.tolist() == [0, 2, 4]
    # The latent target is a distribution; the teacher order is distinct.
    np.testing.assert_allclose(
        np.asarray(out.nodes.generator_target).sum(-1), 1.0, atol=1e-5
    )
    for node in np.asarray(out.nodes.teacher_codes).reshape(-1, cfg.num_candidates):
        assert len(set(node.tolist())) == cfg.num_candidates


def test_imagine_has_no_validity_or_legality_input(module_and_params):
    """The structural no-oracle contract: g takes rows, a latent action
    and a chance code and nothing else; the unroll takes no validity, no
    next mask. Then the behavioural half: the real next rows move the
    posterior (the control) and nothing rollout-side at the root."""
    import inspect

    from rl.model.transition import TransitionModel

    assert list(inspect.signature(TransitionModel.imagine).parameters) == [
        "self",
        "rows",
        "action_one_hot",
        "code_one_hot",
    ]
    assert list(inspect.signature(TransitionModel.__call__).parameters) == [
        "self",
        "rows",
        "action_cell",
        "legal",
        "log_policy",
    ]
    module, cfg, params, apply, inputs = module_and_params
    rows, cells, legal, log_policy = inputs
    base = apply(params, *inputs)
    other = _rows(np.random.default_rng(7), 3)
    # Change the rows at steps 1 and 2 only: the root state is untouched.
    moved_rows = rows.at[1:].set(other[1:])
    moved = apply(params, moved_rows, cells, legal, log_policy)
    assert not np.array_equal(
        np.asarray(base.steps.post_logits[0, 0]),
        np.asarray(moved.steps.post_logits[0, 0]),
    )
    for group, leaf in (("root", "action_logits"), ("first", "pred_prior")):
        np.testing.assert_array_equal(
            np.asarray(getattr(getattr(base, group), leaf)[0]),
            np.asarray(getattr(getattr(moved, group), leaf)[0]),
        )
    np.testing.assert_array_equal(
        np.asarray(base.steps.prior_logits[0, 0]),
        np.asarray(moved.steps.prior_logits[0, 0]),
    )
    np.testing.assert_array_equal(
        np.asarray(base.nodes.generator_logits[0, 0]),
        np.asarray(moved.nodes.generator_logits[0, 0]),
    )
    # Positive control for the encoder: it reads the taken cell.
    other_cells = jnp.asarray([1, 8, 0])
    moved = apply(params, rows, other_cells, legal, log_policy)
    assert not np.array_equal(
        np.asarray(base.nodes.align_logits[0]), np.asarray(moved.nodes.align_logits[0])
    )


def test_posterior_reads_a_row_that_appears_at_t_plus_one(module_and_params):
    """Row 5 is zero at every step of the fixture. Give it content at step
    1 only: the first transition's posterior (whose label is step 1)
    moves. The 2026-09-05 read masked the delta by `valid & next_valid`
    and was blind to exactly this. WHICH row appears is legible too:
    the same content on row 5 and on row 9 read differently."""
    module, cfg, params, apply, inputs = module_and_params
    rows, cells, legal, log_policy = inputs
    base = apply(params, *inputs)
    delta = jnp.asarray(np.random.default_rng(11).normal(size=256), jnp.float32)
    on_row_5 = apply(params, rows.at[1, 5].set(delta), cells, legal, log_policy)
    on_row_9 = apply(params, rows.at[1, 9].add(delta), cells, legal, log_policy)
    assert not np.array_equal(
        np.asarray(base.steps.post_logits[0, 0]),
        np.asarray(on_row_5.steps.post_logits[0, 0]),
    )
    assert not np.array_equal(
        np.asarray(on_row_5.steps.post_logits[0, 0]),
        np.asarray(on_row_9.steps.post_logits[0, 0]),
    )


def test_row_read_is_bias_free_and_reads_a_zero_row_as_zero():
    from rl.model.transition import RowRead

    read = RowRead(width=4, dtype=jnp.float32)
    rows = jnp.asarray(np.random.default_rng(0).normal(size=(6, 8)), jnp.float32)
    rows = rows.at[2].set(0.0)
    params = read.init(jax.random.PRNGKey(0), rows)
    base = read.apply(params, rows).reshape(6, 4)
    np.testing.assert_array_equal(np.asarray(base[2]), 0.0)
    # A live-row edit moves ITS slot only: position is the identity.
    moved = read.apply(params, rows.at[3].set(1e3)).reshape(6, 4)
    assert not np.array_equal(np.asarray(moved[3]), np.asarray(base[3]))
    np.testing.assert_array_equal(
        np.asarray(jnp.delete(moved, 3, axis=0)),
        np.asarray(jnp.delete(base, 3, axis=0)),
    )


def _rms(tree):
    return float(
        jnp.sqrt(
            jnp.mean(
                jnp.square(
                    jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree.leaves(tree)])
                )
            )
        )
    )


BEHIND_THE_ZERO_FACTOR = (
    "dynamics_blocks",
    "chance_token_proj",
    "code_table",
    "slot_embedding",
    "action_table",
    "condition_type_embedding",
    "action_encoder",
)


def test_dynamics_out_proj_is_the_single_zero_factor(module_and_params):
    module, cfg, params, apply, inputs = module_and_params
    probe = jax.random.normal(jax.random.PRNGKey(3), inputs[0].shape)

    def imagined_energy(params):
        return jnp.sum(module.apply(params, *inputs).steps.pred[0] * probe)

    grads = jax.jit(jax.grad(imagined_energy))(params)["params"]
    # Fresh: out_proj's gradient is live (one zero factor over live
    # inputs); everything behind it -- the blocks, the tokens' tables and
    # projections, the encoder the action token is drawn from -- is
    # W_out^T times the residual, i.e. 0.
    assert _rms(grads["dynamics_out_proj"]) > 0.0
    for behind in BEHIND_THE_ZERO_FACTOR:
        assert _rms(grads[behind]) == 0.0, behind
    opened = open_zero_init_paths(params, ["dynamics_out_proj"])
    grads = jax.jit(jax.grad(imagined_energy))(opened)["params"]
    for behind in BEHIND_THE_ZERO_FACTOR:
        assert _rms(grads[behind]) > 0.0, behind


def test_zero_row_at_t_is_imagined_from_the_slot_embedding(module_and_params):
    """Row 5 is zero at t. With the zero factor opened its imagined row
    is nonzero, depends on the rest of the state, and has a live gradient
    into the blocks and the slot embedding. The analytic control: rows 5
    and 6 both zeroed carry NO identity but the slot embedding, so with
    it zeroed their imagined rows are identical (attention is permutation
    equivariant) and with it live they differ."""
    module, cfg, params, apply, inputs = module_and_params
    rows, cells, legal, log_policy = inputs
    opened = open_zero_init_paths(params, ["dynamics_out_proj"])
    both_zero = rows.at[:, 6].set(0.0)
    out = apply(opened, both_zero, cells, legal, log_policy)
    imagined_5 = np.asarray(out.steps.pred[0, :, 5])
    assert np.abs(imagined_5).max() > 0.0
    other = both_zero.at[:, 2].add(1.0)
    moved = apply(opened, other, cells, legal, log_policy)
    assert not np.array_equal(imagined_5, np.asarray(moved.steps.pred[0, :, 5]))
    assert not np.array_equal(imagined_5, np.asarray(out.steps.pred[0, :, 6]))

    def row_energy(params):
        return jnp.sum(
            module.apply(params, both_zero, cells, legal, log_policy).steps.pred[
                0, :, 5
            ]
            ** 2
        )

    grads = jax.jit(jax.grad(row_energy))(opened)["params"]
    assert _rms(grads["slot_embedding"]) > 0.0
    assert _rms(grads["dynamics_blocks"]) > 0.0
    no_identity = dict(opened)
    no_identity["params"] = dict(opened["params"])
    no_identity["params"]["slot_embedding"] = jnp.zeros_like(
        opened["params"]["slot_embedding"]
    )
    same = apply(no_identity, both_zero, cells, legal, log_policy)
    np.testing.assert_array_equal(
        np.asarray(same.steps.pred[0, :, 5]), np.asarray(same.steps.pred[0, :, 6])
    )


def test_unroll_recomputes_the_encoder_and_the_posterior_at_the_imagined_state(
    module_and_params,
):
    """Node 1's alignment logits are the encoder on the IMAGINED state
    hhat_1 with step 1's recorded cell -- exactly `action_logits(pred[0],
    cells[1])` -- and the second transition's posterior reads the delta
    from hhat_1, not from the real step 1."""
    module, cfg, params, apply, inputs = module_and_params
    rows, cells, legal, log_policy = inputs
    opened = open_zero_init_paths(params, ["dynamics_out_proj"])
    out = apply(opened, *inputs)
    imagined = out.steps.pred[0, 0]
    assert not np.array_equal(np.asarray(imagined), np.asarray(rows[1]))
    # The unrolled trace and the standalone method are different
    # executables (TF32 matmuls on the GPU): agreement to ~1e-2, never
    # bitwise; the real-state control differs by O(1).
    recomputed = module.apply(opened, imagined, cells[1], method="action_logits")
    np.testing.assert_allclose(
        np.asarray(out.nodes.align_logits[1, 0]),
        np.asarray(recomputed),
        rtol=2e-2,
        atol=1e-2,
    )
    real = module.apply(opened, rows[1], cells[1], method="action_logits")
    assert not np.allclose(
        np.asarray(out.nodes.align_logits[1, 0]), np.asarray(real), rtol=2e-2, atol=1e-2
    )
    # The alignment TARGET is the real-state distribution (sg), which the
    # root node of start step 1 also carries.
    np.testing.assert_allclose(
        np.asarray(out.nodes.align_target[1, 0]),
        np.asarray(out.nodes.align_target[0, 1]),
        rtol=1e-6,
    )
    posterior = module.apply(
        opened, imagined, out.steps.action_one_hot[1, 0], rows[2], method="posterior"
    )
    np.testing.assert_allclose(
        np.asarray(out.steps.post_logits[1, 0]),
        np.asarray(posterior),
        rtol=2e-2,
        atol=1e-2,
    )


def test_unroll_steps_one_is_the_single_step_model(module_and_params):
    module, cfg, params, apply, inputs = module_and_params
    single, _ = _module(unroll_steps=1)
    out = apply(params, *inputs)
    once = jax.jit(single.apply)(params, *inputs)
    assert once.steps.pred.shape[0] == 1 and once.nodes.generator_logits.shape[0] == 2
    np.testing.assert_array_equal(
        np.asarray(once.steps.pred[0]), np.asarray(out.steps.pred[0])
    )
    np.testing.assert_array_equal(
        np.asarray(once.root.action_logits), np.asarray(out.root.action_logits)
    )


def test_module_samples_only_under_a_sampling_rng(module_and_params):
    """`apply` without a "sampling" rng decodes every draw's mode (init,
    probes, the offline harness): the posterior's argmax, the encoder's
    argmax, the deterministic top-J teacher order; with one it draws, so
    the decoded codes are not always the mode and the key is split per
    step."""
    from rl.model.categoricals import gumbel_top_k, unimix_probs

    module, cfg, params, apply, inputs = module_and_params
    mode = apply(params, *inputs)
    np.testing.assert_array_equal(
        np.asarray(mode.steps.post_one_hot.argmax(-1)),
        np.asarray(mode.steps.post_logits.argmax(-1)),
    )
    np.testing.assert_array_equal(
        np.asarray(mode.steps.action_one_hot[0].argmax(-1)),
        np.asarray(unimix_probs(mode.nodes.align_logits[0]).argmax(-1)),
    )
    for step in range(3):
        np.testing.assert_array_equal(
            np.asarray(mode.nodes.teacher_codes[0, step]),
            np.asarray(
                gumbel_top_k(
                    jnp.where(
                        mode.nodes.support_mask[0, step],
                        jnp.log(mode.nodes.generator_target[0, step]),
                        -jnp.inf,
                    ),
                    cfg.num_candidates,
                    None,
                )
            ),
        )
    # The rng'd trace is a different executable from the fixture's, so
    # the root encoder's logits (the one read no draw feeds) agree to
    # kernel-selection precision, never bitwise. The prior's logits DO
    # move: they read the drawn action code.
    is_mode = []
    for seed in range(48):
        drawn = apply(params, *inputs, rngs={"sampling": jax.random.PRNGKey(seed)})
        np.testing.assert_allclose(
            np.asarray(drawn.root.action_logits),
            np.asarray(mode.root.action_logits),
            rtol=1e-4,
            atol=1e-5,
        )
        is_mode.append(
            np.asarray(
                drawn.steps.post_one_hot[0].argmax(-1)
                == mode.steps.post_logits[0].argmax(-1)
            )
        )
    is_mode = np.stack(is_mode)
    assert 0.0 < is_mode.mean() < 0.5
    drawn = apply(params, *inputs, rngs={"sampling": jax.random.PRNGKey(0)})
    classes = np.asarray(drawn.steps.post_one_hot[0].argmax(-1))
    assert len({tuple(row) for row in classes}) > 1
    assert not np.array_equal(
        np.asarray(drawn.nodes.teacher_codes[0]),
        np.asarray(mode.nodes.teacher_codes[0]),
    )


def test_generate_draws_distinct_codes_inside_the_support(module_and_params):
    module, cfg, params, apply, inputs = module_and_params
    rows = inputs[0][0]
    generate = jax.jit(
        lambda params, rows, key, threshold: module.apply(
            params, rows, key, threshold, method="generate"
        )
    )
    for seed in range(4):
        drawn = generate(params, rows, jax.random.PRNGKey(seed), 0.99)
        codes = np.asarray(drawn.codes)
        assert len(set(codes.tolist())) == cfg.num_candidates
        assert np.all(np.asarray(drawn.log_draw_conditionals) <= 0.0)
        assert bool(np.asarray(drawn.support_mask)[codes][drawn.occupied].all())
    # A tight threshold on the fresh (near-flat) generator keeps a small
    # support: the occupied count is the support size, the rest padded
    # with distinct codes, and the retained mass reads the occupied ones.
    tight = generate(params, rows, jax.random.PRNGKey(0), 0.05)
    support = int(np.asarray(tight.support_mask).sum())
    assert support < cfg.num_candidates
    assert int(np.asarray(tight.occupied).sum()) == support
    assert len(set(np.asarray(tight.codes).tolist())) == cfg.num_candidates
    np.testing.assert_allclose(
        float(tight.retained_mass),
        float(np.asarray(tight.rho_at_codes)[np.asarray(tight.occupied)].sum()),
        rtol=1e-6,
    )
    # No key: the deterministic argmax order, the same on every call.
    first = generate(params, rows, None, 0.99)
    again = generate(params, rows, None, 0.99)
    np.testing.assert_array_equal(np.asarray(first.codes), np.asarray(again.codes))


def test_code_groups_zero_drops_the_chance_path_only():
    module, cfg = _module(code_groups=0)
    inputs = _inputs(np.random.default_rng(0), num_steps=2)
    params = jax.jit(module.init)(jax.random.PRNGKey(0), *inputs)
    with_code, _ = _module(code_groups=2)
    params_with = jax.jit(with_code.init)(jax.random.PRNGKey(0), *inputs)
    assert set(params_with["params"]) - set(params["params"]) == {
        "code_table",
        "chance_token_proj",
        "row_read",
        "prior_latent_net",
        "posterior_latent_net",
    }
    out = jax.jit(module.apply)(params, *inputs)
    assert out.steps.prior_logits.shape == (2, 2, 0, cfg.code_classes)
    np.testing.assert_array_equal(np.asarray(out.steps.pred[0]), np.asarray(inputs[0]))
    # Without a chance code the sequence through g is rows + one token.
    with_rng = jax.jit(module.apply)(
        params, *inputs, rngs={"sampling": jax.random.PRNGKey(0)}
    )
    np.testing.assert_array_equal(
        np.asarray(with_rng.steps.pred), np.asarray(out.steps.pred)
    )


NUM_OFFSETS = 2
NUM_CODES = 64
NUM_CANDIDATES = 8
MAX_CELLS = 16


def _value_head_from_logits(logits):
    from rl.environment.data import CAT_VF_SUPPORT
    from rl.environment.interfaces import CategoricalValueHeadOutput

    log_probs = jax.nn.log_softmax(logits)
    support = jnp.asarray(CAT_VF_SUPPORT, jnp.float32)
    return CategoricalValueHeadOutput(
        logits=logits, log_probs=log_probs, expectation=jnp.exp(log_probs) @ support
    )


def _synthetic_pred(rng, num_steps, code_groups=2, code_classes=16, n_bins=None):
    from rl.environment.data import CAT_VF_SUPPORT
    from rl.model.categoricals import support_set

    if n_bins is None:
        n_bins = len(CAT_VF_SUPPORT)

    def normal(*shape):
        return jnp.asarray(rng.normal(size=shape).astype(np.float32))

    def value_head(*lead):
        return _value_head_from_logits(normal(*lead, 1, n_bins))

    rows = len(POLICY_READABLE_ROWS)
    nodes = NUM_OFFSETS + 1
    post_logits = normal(NUM_OFFSETS, num_steps, 1, code_groups, code_classes)
    target = jax.nn.softmax(normal(nodes, num_steps, 1, NUM_CODES), axis=-1)
    support = jax.vmap(jax.vmap(jax.vmap(lambda p: support_set(p, 0.99))))(target)
    teacher = jnp.argsort(-target, axis=-1)[..., :NUM_CANDIDATES]
    cells = jnp.broadcast_to(jnp.arange(MAX_CELLS), (num_steps, 1, MAX_CELLS))
    cell_valid = cells < 4
    return PlayerActorOutput(
        transition_cons_err=jnp.asarray(rng.random((num_steps, 1, rows)), jnp.float32),
        transition_cons_scale=jnp.asarray(
            rng.random((num_steps, 1, rows)), jnp.float32
        ),
        transition_cons_valid=jnp.ones((num_steps, 1, rows), bool),
        transition_prior_logits=normal(
            NUM_OFFSETS, num_steps, 1, code_groups, code_classes
        ),
        transition_post_logits=post_logits,
        transition_post_one_hot=jax.nn.one_hot(post_logits.argmax(-1), code_classes),
        value_head=value_head(num_steps),
        transition_value_head=value_head(NUM_OFFSETS, num_steps),
        transition_value_head_prior=value_head(num_steps),
        transition_pred_rms=jnp.ones((num_steps, 1), jnp.float32),
        transition_newly_valid=jnp.asarray(
            [[step % 2 == 1] for step in range(num_steps)]
        ),
        transition_kind_logits=normal(NUM_OFFSETS, num_steps, 1, 4),
        transition_done_logit=normal(NUM_OFFSETS, num_steps, 1),
        transition_terminal_logits=normal(NUM_OFFSETS, num_steps, 1, n_bins),
        transition_action_logits=normal(num_steps, 1, MAX_CELLS, NUM_CODES),
        transition_action_cells=cells,
        transition_action_cell_valid=cell_valid,
        transition_action_taken_index=jnp.zeros((num_steps, 1), jnp.int32),
        transition_action_overflow=jnp.zeros((num_steps, 1), bool),
        transition_action_one_hot=jax.nn.one_hot(
            jnp.asarray(rng.integers(0, NUM_CODES, size=(NUM_OFFSETS, num_steps, 1))),
            NUM_CODES,
        ),
        transition_generator_logits=normal(
            nodes, num_steps, 1, NUM_CANDIDATES, NUM_CODES
        ),
        transition_teacher_codes=teacher,
        transition_generator_target=target,
        transition_support_mask=support,
        transition_node_overflow=jnp.zeros((nodes, num_steps, 1), bool),
        transition_align_logits=normal(nodes, num_steps, 1, NUM_CODES),
        transition_align_target=jax.nn.softmax(
            normal(nodes, num_steps, 1, NUM_CODES), axis=-1
        ),
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
    return dict(
        pred=pred,
        env_output=env,
        acted_mask=acted,
        value_mask=acted,
        win_returns=win_returns,
        v_target=v_target,
        cat_vf_support=jnp.asarray(CAT_VF_SUPPORT, jnp.float32),
        splits={},
        axis=action_axis_masks(env.action_mask, action_index),
        config=Porygon2LearnerConfig(),
    )


def test_unroll_masks_stop_at_a_done_row_and_the_chunk_boundary():
    from rl.online.training.train_step import unroll_masks

    # Rows 0-2 acted, row 3 done (value-eligible, not acted), row 4 a
    # new game's first acted row, row 5 the bootstrap-only tail (value
    # eligible, acted -- never a start with a successor).
    acted = jnp.asarray([[1], [1], [1], [0], [1], [1]], bool)
    value = jnp.asarray([[1], [1], [1], [1], [1], [1]], bool)
    nodes, transitions = unroll_masks(acted, value, 2)
    assert [int(row[0]) for row in nodes[0]] == [1, 1, 1, 0, 1, 1]
    # Node 1: acted at t and t+1 -- row 2 (t+1 = 3, done) is out.
    assert [int(row[0]) for row in nodes[1]] == [1, 1, 0, 0, 1, 0]
    assert [int(row[0]) for row in nodes[2]] == [1, 0, 0, 0, 0, 0]
    # Transition 0 predicts t+1: row 2 -> the done row 3 is a valid
    # TARGET; row 5 has no successor.
    assert [int(row[0]) for row in transitions[0]] == [1, 1, 1, 0, 1, 0]
    # Transition 1 predicts t+2 through an acted t+1: rows 0, 1 only.
    assert [int(row[0]) for row in transitions[1]] == [1, 1, 0, 0, 0, 0]


def test_bracket_is_finite_on_an_all_masked_batch_and_reads_its_labels():
    inputs = _bracket_inputs()
    loss, logs = transition_losses(**inputs)
    assert bool(jnp.isfinite(loss))
    for key, value in logs.items():
        assert np.isfinite(np.asarray(value, np.float32)).all(), key
    assert float(logs["player_transition_rows_frac"]) == 0.75  # 3 of 4 starts
    assert float(logs["player_transition_unroll_rows_frac"]) == 0.5
    # kind labels are the NEXT step's request type: steps 1..3 = [1, 0, 2].
    kind_logits = np.asarray(inputs["pred"].transition_kind_logits)[0, :-1, 0]
    expected = np.mean(kind_logits.argmax(-1) == np.array([1, 0, 2]))
    assert float(logs["player_transition_kind_acc"]) == pytest.approx(expected)
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

    def kl_part(prior_logits, post_logits, free_nats, config=config):
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
    # Both offsets carry gradient (the family is averaged over them).
    assert float(jnp.abs(grad_prior[1]).max()) > 0.0
    # Doubling dyn_coef doubles the prior gradient and leaves the
    # posterior's untouched.
    doubled = dataclasses.replace(
        config, player_transition_dyn_coef=2 * config.player_transition_dyn_coef
    )
    grad_prior_2, grad_post_2 = jax.grad(
        lambda p, q: kl_part(p, q, 0.0, doubled), argnums=(0, 1)
    )(prior, post)
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
    assert float(logs["player_transition_kl_k1"]) == pytest.approx(0.0, abs=1e-6)
    assert float(logs["player_transition_kl_free_frac"]) == 1.0
    assert float(logs["player_loss_transition_kl"]) == pytest.approx(
        (config.player_transition_dyn_coef + config.player_transition_rep_coef)
        * config.player_transition_free_nats
    )
    assert float(logs["player_transition_prior_post_agree"]) == 1.0
    # rep_coef 0 (Step 3b D, Stochastic MuZero's form): the posterior's
    # KL-side gradient is exactly zero and the prior's is untouched.
    rep_off = dataclasses.replace(config, player_transition_rep_coef=0.0)
    grad_prior_off, grad_post_off = jax.grad(
        lambda p, q: kl_part(p, q, 0.0, rep_off), argnums=(0, 1)
    )(prior, post)
    assert float(jnp.abs(grad_post_off).max()) == 0.0
    np.testing.assert_allclose(
        np.asarray(grad_prior_off), np.asarray(grad_prior), rtol=1e-5
    )


def test_value_delta_r2_and_value_gain_bracket_copy_and_real():
    """The calibration panels read the imagined value against the COPY
    predictor: an imagined head equal to the real head at t scores
    exactly 0 on the delta-R2 and on the gain, one equal to the real head
    at t+k+1 scores exactly 1 -- the posterior and the prior-mode
    variants each pinned at both ends, the switch / move / newly-valid
    splits and the second offset with them, and a half-way head strictly
    between."""
    num_steps = 6
    inputs = _bracket_inputs(num_steps=num_steps)
    n_bins = int(inputs["win_returns"].shape[-1])
    labels = jnp.asarray([[0], [1], [2], [0], [2], [1]], jnp.int32)
    win_returns = jax.nn.one_hot(labels, n_bins)
    real_logits = 4.0 * win_returns + jnp.asarray(
        np.random.default_rng(3).normal(scale=0.3, size=(num_steps, 1, n_bins)),
        jnp.float32,
    )
    real = _value_head_from_logits(real_logits)

    def shifted_by(offset):
        return jax.tree.map(
            lambda leaf: jnp.concatenate([leaf[offset:]] + [leaf[-1:]] * offset), real
        )

    def stacked(*heads):
        return jax.tree.map(lambda *leaves: jnp.stack(leaves), *heads)

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

    copy_logs = read(imagined=stacked(real, real), prior=shifted_by(1))
    for name in ("", "_switch", "_move", "_newly_valid", "_no_newly_valid", "_k1"):
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

    real_logs = read(imagined=stacked(shifted_by(1), shifted_by(2)), prior=real)
    for name in ("", "_switch", "_move", "_newly_valid", "_no_newly_valid", "_k1"):
        assert real_logs[f"player_transition_value_delta_r2{name}"] == pytest.approx(
            1.0, abs=1e-5
        ), name
    assert real_logs["player_transition_value_gain"] == pytest.approx(1.0, abs=1e-5)
    assert real_logs["player_transition_value_delta_r2_prior"] == 0.0
    assert real_logs["player_transition_value_gain_prior"] == 0.0

    halfway = _value_head_from_logits(0.5 * (real.logits + shifted_by(1).logits))
    half_logs = read(imagined=stacked(halfway, halfway), prior=halfway)
    assert 0.0 < half_logs["player_transition_value_delta_r2"] < 1.0
    assert 0.0 < half_logs["player_transition_value_gain"] < 1.0

    # The split rows are the ones the axis names: the fixture legalises a
    # move on every step, so its taken switches are all VOLUNTARY.
    axis = inputs["axis"]
    assert bool(axis.has_move.all())
    assert [int(value) for value in axis.taken_switch[:, 0]] == [1, 0, 1, 0, 1, 0]


def _coef_zero_keeps_the_panel_and_drops_the_gradient(coef_name, leaf, panel):
    inputs = _bracket_inputs()
    logs_by_coef = {}
    grad_by_coef = {}
    for coef in (0.0, 1.0):
        config = dataclasses.replace(inputs["config"], **{coef_name: coef})

        def loss_of(value, config=config):
            pred = dataclasses.replace(inputs["pred"], **{leaf: value})
            loss, _ = transition_losses(**{**inputs, "pred": pred, "config": config})
            return loss

        _, logs = transition_losses(**{**inputs, "config": config})
        logs_by_coef[coef] = {key: float(value) for key, value in logs.items()}
        grad_by_coef[coef] = jax.grad(loss_of)(getattr(inputs["pred"], leaf))
    assert logs_by_coef[0.0][panel] == logs_by_coef[1.0][panel]
    assert logs_by_coef[1.0][panel] > 0.0
    assert float(jnp.abs(grad_by_coef[0.0]).max()) == 0.0
    assert float(jnp.abs(grad_by_coef[1.0]).max()) > 0.0


def test_cons_coef_zero_keeps_the_panels_and_drops_the_gradient():
    _coef_zero_keeps_the_panel_and_drops_the_gradient(
        "player_transition_cons_coef",
        "transition_cons_err",
        "player_loss_transition_cons",
    )


def test_consistency_masks_absent_rows_but_scores_previous_choices_when_present():
    """Singles' absent slots cannot dominate; doubles' present slots stay live.

    The positive control enables the same previous-action rows, with the
    same large error and zero movement: validity, not group identity or
    movement magnitude, determines whether they receive a gradient.
    """
    from rl.model.constants import SEQUENCE_GROUP_IDS, SequenceGroup

    inputs = _bracket_inputs()
    pred = inputs["pred"]
    group_ids = SEQUENCE_GROUP_IDS[POLICY_READABLE_ROWS]
    previous = jnp.asarray(group_ids == SequenceGroup.PREV_ACTION)
    cls_rows = jnp.asarray(group_ids == SequenceGroup.CLS)
    errors = jnp.full_like(pred.transition_cons_err, 2.0)
    scales = jnp.ones_like(errors)
    scales = jnp.where(previous, 0.0, scales)
    config = inputs["config"].replace(player_transition_cons_coef=1.0)

    def read(error, valid):
        changed = pred.replace(
            transition_cons_err=error,
            transition_cons_scale=scales,
            transition_cons_valid=valid,
        )
        return transition_losses(**{**inputs, "pred": changed, "config": config})

    absent = jnp.broadcast_to(cls_rows, errors.shape)
    corrupted = jnp.where(previous, 4000.0, errors)
    base_loss, base_logs = read(errors, absent)
    loss, logs = read(corrupted, absent)
    assert float(loss) == float(base_loss)
    # Only CLS is eligible: empty groups must not dilute its loss.
    assert float(logs["player_loss_transition_cons"]) == pytest.approx(2.0)
    assert float(logs["player_transition_cons_gain_prev_action"]) == 0.0
    assert float(logs["player_transition_cons_gain_newly_valid"]) == float(
        base_logs["player_transition_cons_gain_newly_valid"]
    )
    absent_grad = jax.grad(lambda error: read(error, absent)[0])(corrupted)
    np.testing.assert_array_equal(np.asarray(absent_grad[..., previous]), 0.0)
    assert float(jnp.abs(absent_grad[..., cls_rows]).max()) > 0.0

    present = jnp.broadcast_to(cls_rows | previous, errors.shape)
    _, present_logs = read(corrupted, present)
    assert float(present_logs["player_loss_transition_cons"]) > 1000.0
    present_grad = jax.grad(lambda error: read(error, present)[0])(corrupted)
    assert float(jnp.abs(present_grad[:-1, ..., previous]).max()) > 0.0
    # The last stored row remains bootstrap-only even when its slots are valid.
    np.testing.assert_array_equal(np.asarray(present_grad[-1]), 0.0)

    empty = jnp.zeros_like(absent)
    _, empty_logs = read(corrupted, empty)
    assert float(empty_logs["player_loss_transition_cons"]) == 0.0
    empty_grad = jax.grad(lambda error: read(error, empty)[0])(corrupted)
    np.testing.assert_array_equal(np.asarray(empty_grad), 0.0)


@pytest.mark.gpu
@pytest.mark.slow
def test_previous_action_consistency_mask_retains_appearances_and_disappearances(
    real_model_and_trajectory, real_model_apply
):
    """Exercise the actual encoder/label wiring for within-request choices.

    This is the previous-choice feature seam used by doubles, not a claim
    that the separately broken doubles battle alignment is supported.
    """
    from rl.model.constants import SEQUENCE_GROUP_IDS, SequenceGroup
    from rl.model.heads import HeadParams

    _, params, actor_input, actor_output = real_model_and_trajectory
    info = jnp.asarray(actor_input.env.info)
    num_steps = info.shape[0]
    assert num_steps >= 4
    previous = SEQUENCE_GROUP_IDS[POLICY_READABLE_ROWS] == SequenceGroup.PREV_ACTION

    def forward(flags):
        changed_info = info.at[:, InfoFeature.INFO_FEATURE__HAS_PREV_ACTION].set(flags)
        changed_input = actor_input.replace(
            env=actor_input.env.replace(info=changed_info)
        )
        return real_model_apply(params, changed_input, actor_output, HeadParams())

    absent = forward(jnp.zeros(num_steps, jnp.int32))
    np.testing.assert_array_equal(
        np.asarray(absent.transition_cons_valid[:, previous]), False
    )
    positions = jnp.arange(num_steps) % 4
    flags = (positions == 1) | (positions == 2)
    present = forward(flags.astype(jnp.int32))
    next_flags = np.concatenate((np.asarray(flags)[1:], np.asarray(flags)[-1:]))
    expected = np.asarray(flags) | next_flags
    np.testing.assert_array_equal(expected[:4], [True, True, True, False])
    np.testing.assert_array_equal(
        np.asarray(present.transition_cons_valid[:, previous]),
        np.broadcast_to(expected[:, None], (num_steps, int(previous.sum()))),
    )


def test_decode_coef_zero_keeps_the_panel_and_drops_the_gradient():
    _coef_zero_keeps_the_panel_and_drops_the_gradient(
        "player_transition_decode_coef",
        "transition_action_logits",
        "player_loss_transition_decode",
    )


def test_align_coef_zero_keeps_the_panel_and_drops_the_gradient():
    _coef_zero_keeps_the_panel_and_drops_the_gradient(
        "player_transition_align_coef",
        "transition_align_logits",
        "player_loss_transition_align",
    )


def test_generator_loss_reaches_every_node_and_reads_the_exact_conditional():
    """The generator term's gradient lands on every node's logits, and a
    generator whose logits ARE the log conditional targets scores a KL
    of exactly 0 on every occupied slot (the loss is then the targets'
    entropy)."""
    from rl.model.transition_objectives import candidate_targets

    inputs = _bracket_inputs()
    pred = inputs["pred"]

    def loss_of(logits):
        moved = dataclasses.replace(pred, transition_generator_logits=logits)
        loss, _ = transition_losses(**{**inputs, "pred": moved})
        return loss

    grads = jax.grad(loss_of)(pred.transition_generator_logits)
    for node in range(NUM_OFFSETS + 1):
        assert float(jnp.abs(grads[node]).max()) > 0.0, node
    targets = jax.vmap(jax.vmap(jax.vmap(candidate_targets)))(
        pred.transition_generator_target,
        pred.transition_teacher_codes,
        pred.transition_support_mask,
    )
    exact = jnp.log(jnp.maximum(targets.targets, 1e-12))
    _, logs = transition_losses(
        **{
            **inputs,
            "pred": dataclasses.replace(pred, transition_generator_logits=exact),
        }
    )
    for key in (
        "player_transition_generator_kl_first",
        "player_transition_generator_kl_later",
        "player_transition_generator_kl_first_k1",
        "player_transition_generator_kl_later_k2",
    ):
        assert float(logs[key]) == pytest.approx(0.0, abs=1e-4), key
    assert 0.0 < float(logs["player_transition_generator_coverage"]) <= 1.0


def test_termination_loss_is_the_joint_nll_and_uses_actual_terminal_rows_only():
    """BCE(done) + done * CE(outcome | done) is the negative log-
    likelihood of {continue, loss, draw, win} factorised through the done
    probability -- checked against the four-outcome NLL on a hand-built
    row -- and the conditional CE is present only on rows whose
    successor is the game's actual terminal row (an empty terminal set
    reads 0 and finite)."""
    from rl.environment.data import CAT_VF_SUPPORT

    num_steps = 4
    inputs = _bracket_inputs(num_steps=num_steps)
    env = inputs["env_output"]
    n_bins = len(CAT_VF_SUPPORT)
    # Step 3 is the game's terminal row with a recorded WIN; the others
    # carry the all-zero reward vector (scalar zero, not a label).
    done = jnp.zeros((num_steps, 1), bool).at[3, 0].set(True)
    win_reward = jnp.zeros((num_steps, 1, n_bins), jnp.float32).at[3, 0, 2].set(1.0)
    env = dataclasses.replace(env, done=done, win_reward=win_reward)
    acted = jnp.asarray([[1], [1], [1], [0]], bool)
    inputs = {**inputs, "env_output": env, "acted_mask": acted}
    _, logs = transition_losses(**inputs)
    # Exactly one terminal transition at offset 0 (start 2 -> row 3).
    assert float(logs["player_transition_terminal_rows"]) == 1.0
    pred = inputs["pred"]
    done_logit = float(pred.transition_done_logit[0, 2, 0])
    terminal_logits = np.asarray(pred.transition_terminal_logits[0, 2, 0])
    joint = np.log(1 / (1 + np.exp(-done_logit))) + (
        terminal_logits[2] - np.log(np.exp(terminal_logits).sum())
    )
    expected_ce = -(terminal_logits[2] - np.log(np.exp(terminal_logits).sum()))
    assert float(logs["player_transition_terminal_ce"]) == pytest.approx(
        expected_ce, rel=1e-5
    )
    # The joint term on that row is exactly -log P(done, win).
    done_labels = np.array([0.0, 0.0, 1.0])
    logits = np.asarray(pred.transition_done_logit[0, :3, 0])
    bce = np.log1p(np.exp(logits)) - done_labels * logits
    per_row = bce.copy()
    per_row[2] += expected_ce
    assert per_row[2] == pytest.approx(-joint, rel=1e-5)
    # Offset 0's termination average over its three eligible starts.
    _, only_first = transition_losses(
        **{
            **inputs,
            "pred": dataclasses.replace(
                pred,
                transition_done_logit=pred.transition_done_logit.at[1].set(0.0),
                transition_terminal_logits=pred.transition_terminal_logits.at[1].set(
                    0.0
                ),
            ),
        }
    )
    # Offset 1 has two eligible starts (0 -> row 2, 1 -> row 3): BCE at
    # logit 0 on both, and the second's successor is the terminal row, so
    # it adds the uniform three-way CE.
    offset_1 = np.log(2.0) + 0.5 * np.log(3.0)
    assert float(only_first["player_loss_transition_termination"]) == pytest.approx(
        0.5 * (per_row.mean() + offset_1), rel=1e-5
    )
    # No terminal row anywhere: the conditional CE contributes nothing
    # and the panels stay finite.
    none = dataclasses.replace(env, done=jnp.zeros_like(done))
    _, logs = transition_losses(**{**inputs, "env_output": none})
    assert float(logs["player_transition_terminal_rows"]) == 0.0
    assert np.isfinite(float(logs["player_transition_terminal_ce"]))
    assert float(logs["player_transition_terminal_ce"]) == 0.0


def test_newly_valid_split_reads_only_transitions_with_an_appearing_row():
    inputs = _bracket_inputs()
    pred = inputs["pred"]
    real = _value_head_from_logits(pred.value_head.logits)
    shifted = jax.tree.map(lambda leaf: jnp.concatenate([leaf[1:], leaf[-1:]]), real)
    # Imagined = real at t+1 on the odd (newly-valid) starts, = copy on
    # the even ones: the newly-valid split reads 1, the other 0.
    odd = jnp.asarray([[step % 2 == 1] for step in range(4)])
    mixed = jax.tree.map(
        lambda next_leaf, copy_leaf: jnp.where(
            odd.reshape((4, 1) + (1,) * (next_leaf.ndim - 2)), next_leaf, copy_leaf
        ),
        shifted,
        real,
    )
    imagined = jax.tree.map(lambda leaf: jnp.stack([leaf, leaf]), mixed)
    moved = dataclasses.replace(
        pred,
        value_head=real,
        transition_value_head=imagined,
        transition_newly_valid=odd,
    )
    _, logs = transition_losses(**{**inputs, "pred": moved})
    assert float(logs["player_transition_value_delta_r2_newly_valid"]) == pytest.approx(
        1.0, abs=1e-5
    )
    assert float(logs["player_transition_value_delta_r2_no_newly_valid"]) == 0.0
    assert float(logs["player_transition_newly_valid_frac"]) == pytest.approx(1 / 3)


def _network_with_frozen_transition_value_head():
    from rl.model.config import get_player_model_config
    from rl.model.player_model import get_player_model

    cfg = get_player_model_config(generation=9, train=True)
    cfg.transition.value_trains_v_head = False
    return get_player_model(cfg)


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
    opened = open_zero_init_paths(params, ["action_head", "dynamics_out_proj"])

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
            win_returns,
            jnp.zeros((num_steps, 1), jnp.float32),
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

    # The readout's `query` is zero-init, so pi is UNIFORM on every row
    # -- opened (dynamics_out_proj stays closed: g is still the copy
    # predictor) so a "the readout is not reached" read cannot pass
    # vacuously. The base policy the latent target is built from is the
    # live readout's log_policy: without its EXPLICIT stop_gradient the
    # generator and decode terms would reach the readout and the trunk
    # (launch check 2's collapse shape). The transition term must reach
    # its OWN subtree and, under `value_trains_v_head`, the shared
    # v_head -- exactly {transition, v_head} on and {transition} off.
    opened = open_zero_init_paths(params, ["action_head"])
    reached = reached_by(grad_fn(opened))
    assert (
        float(
            jnp.abs(opened["params"]["transition"]["dynamics_out_proj"]["kernel"]).max()
        )
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


def test_exported_leaves_match_the_actor_output_dataclass():
    """The `transition_*` identity is written ONCE, in
    TransitionOutput.exported. This pins it against PlayerActorOutput: a
    field added to one side and not the other fails here rather than at the
    next learner launch."""
    from rl.model.transition import (
        INTERNAL_LEAVES,
        FirstStepOutputs,
        NodeOutputs,
        RootOutputs,
        StepOutputs,
        TransitionOutput,
    )

    # The leaves player_model._forward_transition DERIVES rather than
    # passes through -- everything else must come from exported().
    derived = {
        "transition_cons_err",
        "transition_cons_scale",
        "transition_cons_valid",
        "transition_value_head",
        "transition_value_head_prior",
        "transition_pred_rms",
        "transition_newly_valid",
    }
    marker = jnp.zeros(())
    output = jax.tree.map(
        lambda _: marker,
        TransitionOutput(
            root=RootOutputs(*[None] * len(RootOutputs._fields)),
            nodes=NodeOutputs(*[None] * len(NodeOutputs._fields)),
            steps=StepOutputs(*[None] * len(StepOutputs._fields)),
            first=FirstStepOutputs(*[None] * len(FirstStepOutputs._fields)),
        ),
        is_leaf=lambda leaf: leaf is None,
    )
    exported = set(output.exported())
    declared = {f.name for f in dataclasses.fields(PlayerActorOutput)}
    assert exported <= declared, exported - declared
    transition_fields = {name for name in declared if name.startswith("transition_")}
    assert transition_fields == exported | derived

    # The internal states are the ONLY leaves held back.
    held_back = set()
    for group in output:
        held_back |= {
            name for name in type(group)._fields if f"transition_{name}" not in exported
        }
    assert held_back == set(INTERNAL_LEAVES)

    # The learner's batch vmap places B after the offset axis on exactly
    # the node and step leaves, plus the value head read off the imagined
    # states -- derived from the same grouping.
    offset_leading = {
        f"transition_{name}"
        for group in (NodeOutputs, StepOutputs)
        for name in group._fields
        if name not in INTERNAL_LEAVES
    } | {"transition_value_head"}
    assert set(PlayerActorOutput.OFFSET_LEADING_LEAVES) == offset_leading

    # Positive control: the checks above are not vacuous.
    assert not (exported | {"transition_not_a_field"}) <= declared
    assert transition_fields != (exported | derived) - {"transition_ground"}
