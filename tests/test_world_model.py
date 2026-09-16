"""Contracts of the event world model (rl/model/world_model.py): copy at
init for every sample and step count, the single zero factor, the sparse
update, the loss normalisation points, the grammar likelihood on a
hand-built event, the declared-token mask, and the flow recovering two
modes the mean control averages."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from ml_collections import ConfigDict

from rl.environment.data import NUM_MOVES
from rl.model import world_model as wm
from rl.model.constants import (
    HISTORY_ENTITY_ROWS,
    NUM_PUBLIC_SEQUENCE_ROWS,
    NUM_PUBLIC_SLOTS,
    PUBLIC_ROWS,
)
from rl.offline.event_labels import NO_SLOT, DeclaredKind, EventKind

WIDTH = 32


def _cfg(width: int = WIDTH, blocks: int = 1) -> ConfigDict:
    cfg = ConfigDict()
    cfg.enabled = True
    cfg.model_size = width
    cfg.flow_steps = 4
    cfg.scale_floor = 1e-2
    block = ConfigDict()
    block.num_blocks = blocks
    block.num_heads = 2
    block.qk_size = width // 2
    block.v_size = width // 2
    block.model_size = width
    block.hidden_size = 2 * width
    block.use_bias = True
    block.qk_layer_norm = True
    cfg.decoder = block
    cfg.flow = ConfigDict()
    cfg.flow.block = block
    return cfg


def _tokens(kind=EventKind.MOVE, actor=0, move=50, target=6):
    return wm.EventTokens(
        kind=jnp.asarray(kind, jnp.int32),
        actor=jnp.asarray(actor, jnp.int32),
        move=jnp.asarray(move, jnp.int32),
        target=jnp.asarray(target, jnp.int32),
    )


def _model_and_params(seed: int = 0, cfg: ConfigDict | None = None):
    if cfg is None:
        cfg = _cfg()
    model = wm.EventWorldModel(cfg)
    rows = jax.random.normal(
        jax.random.key(seed), (NUM_PUBLIC_SEQUENCE_ROWS, cfg.model_size)
    )
    row_valid = jnp.ones(NUM_PUBLIC_SEQUENCE_ROWS, jnp.bool_)
    row_mask = jnp.ones(NUM_PUBLIC_SEQUENCE_ROWS, jnp.bool_)
    scale = jnp.ones(wm.NUM_PUBLIC_GROUPS, jnp.float32)
    # step_terms touches every submodule, so every parameter exists.
    params = model.init(
        jax.random.key(seed + 1),
        rows,
        row_valid,
        rows,
        jnp.asarray(0),
        jnp.asarray(0),
        _tokens(),
        jnp.asarray(True),
        row_mask,
        scale,
        jnp.zeros_like(rows),
        jnp.asarray(0.5),
        method=wm.EventWorldModel.step_terms,
    )
    return model, params, rows, row_valid, row_mask, scale


def _perturb(params, path_predicate, seed=3):
    flat = jax.tree_util.tree_leaves_with_path(params)
    keys = jax.random.split(jax.random.key(seed), len(flat))
    leaves = []
    for key, (path, leaf) in zip(keys, flat):
        name = jax.tree_util.keystr(path)
        if path_predicate(name):
            leaf = leaf + 0.05 * jax.random.normal(key, leaf.shape, leaf.dtype)
        leaves.append(leaf)
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params), leaves)


def test_every_sample_is_the_copy_predictor_at_init() -> None:
    model, params, rows, _, row_mask, scale = _model_and_params()
    for steps in (1, 2, 4):
        cfg = _cfg()
        cfg.flow_steps = steps
        model_k = wm.EventWorldModel(cfg)
        for seed in range(3):
            imagined = model_k.apply(
                params,
                rows,
                _tokens(),
                row_mask,
                scale,
                jax.random.key(seed),
                method=wm.EventWorldModel.imagine,
            )
            np.testing.assert_array_equal(np.asarray(imagined), np.asarray(rows))


def test_out_proj_is_the_single_zero_factor_and_the_update_is_sparse() -> None:
    model, params, rows, _, _, scale = _model_and_params()
    row_mask = jnp.zeros(NUM_PUBLIC_SEQUENCE_ROWS, jnp.bool_).at[:8].set(True)
    # Everything but out_proj perturbed: still the copy predictor.
    others = _perturb(params, lambda name: "flow" in name and "out_proj" not in name)
    still_copy = model.apply(
        others,
        rows,
        _tokens(),
        row_mask,
        scale,
        jax.random.key(0),
        method=wm.EventWorldModel.imagine,
    )
    np.testing.assert_array_equal(np.asarray(still_copy), np.asarray(rows))
    # out_proj alone opened: the masked rows move, the others are bit-identical.
    opened = _perturb(params, lambda name: "flow" in name and "out_proj" in name)
    moved = np.asarray(
        model.apply(
            opened,
            rows,
            _tokens(),
            row_mask,
            scale,
            jax.random.key(0),
            method=wm.EventWorldModel.imagine,
        )
    )
    base = np.asarray(rows)
    assert not np.allclose(moved[:8], base[:8])
    np.testing.assert_array_equal(moved[8:], base[8:])


def test_loss_normalisation_points() -> None:
    energy = jax.random.uniform(jax.random.key(0), (4, NUM_PUBLIC_SEQUENCE_ROWS)) + 0.5
    mask = jnp.ones((4, NUM_PUBLIC_SEQUENCE_ROWS), jnp.bool_)
    x1 = jnp.sqrt(energy)
    for prediction, expected in ((jnp.zeros_like(x1), 1.0), (x1, 0.0), (-x1, 4.0)):
        loss, per_group = wm.pooled_group_loss(
            jnp.square(prediction - x1), energy, mask, 1e-2
        )
        np.testing.assert_allclose(float(loss), expected, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(per_group), expected, rtol=1e-5)
    # An all-static group is floored, not divided by zero.
    loss, _ = wm.pooled_group_loss(jnp.zeros_like(x1), jnp.zeros_like(x1), mask, 1e-2)
    assert np.isfinite(float(loss)) and float(loss) == 0.0


def test_update_rows_follow_the_touched_slots_and_their_target_rows() -> None:
    touched = jnp.zeros(NUM_PUBLIC_SLOTS, jnp.bool_).at[3].set(True)
    rows = np.asarray(
        wm.update_rows(
            touched,
            jnp.asarray(False),
            jnp.asarray([3, -1]),
            jnp.asarray([7, -1]),
        )
    )
    assert rows[PUBLIC_ROWS.start + 3] and rows[HISTORY_ENTITY_ROWS.start + 3]
    assert not rows[PUBLIC_ROWS.start + 4]
    assert (
        rows[wm.ALLY_TARGET_SEQUENCE_ROWS[0]]
        and not rows[wm.ALLY_TARGET_SEQUENCE_ROWS[1]]
    )
    assert not rows[wm.ENEMY_TARGET_SEQUENCE_ROWS].any()
    assert rows[wm.SUMMARY_ROWS].all()
    assert not rows[wm.FIELD_UPDATE_ROWS].any()
    with_field = np.asarray(
        wm.update_rows(
            touched, jnp.asarray(True), jnp.asarray([-1, -1]), jnp.asarray([-1, -1])
        )
    )
    assert with_field[wm.FIELD_UPDATE_ROWS].all()


def test_grammar_nll_on_a_hand_built_event() -> None:
    num_revealed = jnp.asarray(4)
    tokens = _tokens(kind=EventKind.MOVE, actor=1, move=50, target=2)
    logits = wm.DecoderLogits(
        kind=jnp.zeros(wm.NUM_EVENT_KINDS),
        new_turn=jnp.zeros(()),
        actor=jnp.zeros(wm.NUM_SLOT_CLASSES),
        move=jnp.zeros(NUM_MOVES),
        target=jnp.zeros(wm.NUM_SLOT_CLASSES),
        touched=jnp.zeros(wm.NUM_TOUCHED_BITS),
    )
    touched = jnp.zeros(wm.NUM_TOUCHED_BITS, jnp.bool_)
    nll = np.asarray(
        wm.grammar_nll(logits, tokens, touched, jnp.asarray(False), num_revealed)
    )
    real_moves = NUM_MOVES - wm.REAL_MOVE_FLOOR
    expected = [
        np.log(wm.NUM_EVENT_KINDS) + np.log(2.0),
        np.log(4),  # four revealed slots, no NO_SLOT for a move
        np.log(real_moves),
        np.log(5),  # four revealed targets + NO_SLOT
        wm.NUM_TOUCHED_BITS * np.log(2.0),
    ]
    np.testing.assert_allclose(nll, expected, rtol=1e-5)
    # A switch may name the next unrevealed slot; a residual has no actor.
    switch = np.asarray(wm.actor_mask(jnp.asarray(EventKind.SWITCH), num_revealed))
    assert switch[:5].all() and not switch[5:].any()
    residual = np.asarray(wm.actor_mask(jnp.asarray(EventKind.RESIDUAL), num_revealed))
    assert residual[NO_SLOT] and not residual[:NUM_PUBLIC_SLOTS].any()


def test_opponent_move_query_cannot_read_the_declared_token() -> None:
    model, params, rows, row_valid, _, _ = _model_and_params()
    params = _perturb(params, lambda name: "decoder" in name or "table" in name)

    def logits(declared_kind, declared_arg, actor_is_mine):
        return model.apply(
            params,
            rows,
            row_valid,
            jnp.asarray(declared_kind, jnp.int32),
            jnp.asarray(declared_arg, jnp.int32),
            _tokens(),
            jnp.asarray(actor_is_mine),
            method=wm.EventWorldModel.decode,
        )

    theirs_a = logits(DeclaredKind.MOVE, 50, False)
    theirs_b = logits(DeclaredKind.SWITCH, 3, False)
    np.testing.assert_array_equal(np.asarray(theirs_a.move), np.asarray(theirs_b.move))
    np.testing.assert_array_equal(
        np.asarray(theirs_a.target), np.asarray(theirs_b.target)
    )
    np.testing.assert_array_equal(
        np.asarray(theirs_a.touched), np.asarray(theirs_b.touched)
    )
    # KIND and ACTOR read the declaration (ordering); the own-side MOVE
    # query reads it too -- the controls that the mask, not a dead path,
    # is what holds above.
    assert not np.allclose(np.asarray(theirs_a.kind), np.asarray(theirs_b.kind))
    mine_a = logits(DeclaredKind.MOVE, 50, True)
    mine_b = logits(DeclaredKind.SWITCH, 3, True)
    assert not np.allclose(np.asarray(mine_a.move), np.asarray(mine_b.move))


def test_flow_recovers_two_modes_the_mean_control_averages() -> None:
    cfg = _cfg(width=16, blocks=1)
    model = wm.EventWorldModel(cfg)
    rows = jax.random.normal(jax.random.key(0), (NUM_PUBLIC_SEQUENCE_ROWS, 16))
    row_valid = jnp.ones(NUM_PUBLIC_SEQUENCE_ROWS, jnp.bool_)
    row_mask = jnp.ones(NUM_PUBLIC_SEQUENCE_ROWS, jnp.bool_)
    scale = jnp.ones(wm.NUM_PUBLIC_GROUPS, jnp.float32)
    mode = jnp.ones((NUM_PUBLIC_SEQUENCE_ROWS, 16))
    tokens = _tokens()
    params = model.init(
        jax.random.key(1),
        rows,
        row_valid,
        rows,
        jnp.asarray(0),
        jnp.asarray(0),
        tokens,
        jnp.asarray(True),
        row_mask,
        scale,
        jnp.zeros_like(rows),
        jnp.asarray(0.5),
        method=wm.EventWorldModel.step_terms,
    )
    optimiser = optax.adam(3e-3)
    opt_state = optimiser.init(params)

    def loss_fn(params, rng):
        sign_key, noise_key, time_key = jax.random.split(rng, 3)
        sign = jnp.where(jax.random.bernoulli(sign_key), 1.0, -1.0)
        next_rows = rows + sign * mode
        noise = jax.random.normal(noise_key, rows.shape)
        time = jax.random.uniform(time_key)
        terms = model.apply(
            params,
            rows,
            row_valid,
            next_rows,
            jnp.asarray(0),
            jnp.asarray(0),
            tokens,
            jnp.asarray(True),
            row_mask,
            scale,
            noise,
            time,
            method=wm.EventWorldModel.step_terms,
        )
        return terms.flow_error.mean() + terms.mean_error.mean()

    @jax.jit
    def update(params, opt_state, rng):
        loss, grads = jax.value_and_grad(loss_fn)(params, rng)
        updates, opt_state = optimiser.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    rng = jax.random.key(4)
    for _ in range(300):
        rng, step_key = jax.random.split(rng)
        params, opt_state, loss = update(params, opt_state, step_key)

    imagine = jax.jit(
        lambda p, k: model.apply(
            p, rows, tokens, row_mask, scale, k, method=wm.EventWorldModel.imagine
        )
    )
    keys = jax.random.split(jax.random.key(5), 32)
    samples = np.stack([np.asarray(imagine(params, key)) for key in keys])
    signs = np.sign((samples - np.asarray(rows)).mean(axis=(1, 2)))
    # Both modes appear among the samples, and each sample sits near ±mode.
    assert (signs > 0).any() and (signs < 0).any()
    distances = np.abs(np.abs(samples - np.asarray(rows)).mean(axis=(1, 2)) - 1.0)
    assert distances.mean() < 0.35
    # The mean control lands between the modes.
    mean_delta = model.apply(
        params,
        rows,
        model.apply(params, tokens, method=wm.EventWorldModel.event_token_embeddings),
        method=lambda m, r, e: m.mean_step(r, e),
    )
    assert np.abs(np.asarray(mean_delta)).mean() < 0.35
