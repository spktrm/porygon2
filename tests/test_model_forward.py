"""Real player-model init + forward on the bundled example trajectory.

Marked gpu: runs wherever JAX puts it (the training box GPU, with
preallocation disabled by conftest so it coexists with a live learner).
Marked slow (~1 min): deselect with `-m "not slow"` for the quick suite.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.model.constants import NUM_FIELD_ROWS, NUM_PUBLIC_SLOTS, PRIVATE_ROWS

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_init_produces_finite_params(real_model_and_trajectory, real_model_apply):
    _, params, _, _ = real_model_and_trajectory
    leaves = jax.tree.leaves(params)
    assert leaves
    for leaf in leaves:
        assert np.isfinite(np.asarray(leaf, dtype=np.float32)).all()


def test_forward_outputs_finite_and_shaped(real_model_and_trajectory, real_model_apply):
    network, params, actor_input, actor_output = real_model_and_trajectory
    from rl.model.heads import HeadParams

    out = real_model_apply(params, actor_input, actor_output, HeadParams())
    T = actor_input.env.done.shape[0]

    log_probs = np.asarray(out.value_head.log_probs, dtype=np.float32)
    assert log_probs.shape[0] == T
    assert np.isfinite(log_probs).all()
    # log_probs is a categorical distribution over the value support.
    np.testing.assert_allclose(np.exp(log_probs).sum(-1), 1.0, atol=1e-3)

    pi_lp = np.asarray(out.action_head.log_prob, dtype=np.float32)
    assert np.isfinite(pi_lp).all()


def test_forward_is_deterministic(real_model_and_trajectory, real_model_apply):
    network, params, actor_input, actor_output = real_model_and_trajectory
    from rl.model.heads import HeadParams

    a = real_model_apply(params, actor_input, actor_output, HeadParams())
    b = real_model_apply(params, actor_input, actor_output, HeadParams())
    np.testing.assert_array_equal(
        np.asarray(a.value_head.log_probs, dtype=np.float32),
        np.asarray(b.value_head.log_probs, dtype=np.float32),
    )


def _assembled_rows(network, params, actor_input):
    """The trunk's input sequence for one timestep, BEFORE any attention.

    Every identity a row carries -- side, group, position -- is additive and
    applied here, so this is where an identity bug is visible. After the
    trunk every row has mixed with every other and the reading would be
    behavioural rather than structural.
    """

    def call(module, env_step):
        encoder = module.encoder
        width = encoder.cfg.entity_size
        zero_slots = jnp.zeros((NUM_PUBLIC_SLOTS, width), env_step.field.dtype)
        sequence, _, _ = encoder._assemble_sequence(
            env_step,
            zero_slots.astype(encoder.cfg.dtype),
            jnp.zeros(NUM_PUBLIC_SLOTS, jnp.bool_),
            jnp.zeros((NUM_FIELD_ROWS, width), encoder.cfg.dtype),
            zero_slots.astype(encoder.cfg.dtype),
        )
        return sequence

    env_step = jax.tree.map(lambda x: x[0], actor_input.env)
    return np.asarray(
        jax.jit(lambda p, e: network.apply(p, e, method=call))(params, env_step),
        dtype=np.float32,
    )


def test_private_sheet_is_not_tagged_with_the_opponents_side(
    real_model_and_trajectory,
):
    """My private sheet must not carry the tag that marks OPPONENT rows.

    The service writes ENTITY_PUBLIC_NODE_FEATURE__SIDE = isMySide(...), so
    side_bias row 1 is mine and row 0 is theirs. Until 2026-08-28 the sheet
    was tagged side_bias(0) -- the opponent's row -- which put my six sheet
    rows under the wrong side. The sheet owns private_side_bias instead, and
    side_bias must not reach it at all.
    """
    network, params, actor_input, _ = real_model_and_trajectory
    base = _assembled_rows(network, params, actor_input)[PRIVATE_ROWS]

    moved_side = _assembled_rows(
        network, _perturbed(params, ("side_bias", "embedding")), actor_input
    )[PRIVATE_ROWS]
    np.testing.assert_allclose(base, moved_side, atol=0)


def test_private_side_bias_is_the_live_route(real_model_and_trajectory):
    """The positive control for the test above: perturbing the sheet's OWN
    tag does move its rows, so that test is not passing merely because
    nothing reaches them."""
    network, params, actor_input, _ = real_model_and_trajectory
    base = _assembled_rows(network, params, actor_input)[PRIVATE_ROWS]
    moved = _assembled_rows(
        network, _perturbed(params, ("private_side_bias",)), actor_input
    )[PRIVATE_ROWS]
    assert not np.allclose(base, moved)


def _field_tokens(network, params, actor_input):
    """The (global, my-side, opp-side) field token triple for one timestep."""

    def call(module, field):
        return module.encoder._embed_field(field)[0]

    field = jax.tree.map(lambda x: x[0], actor_input.env.field)
    return jax.jit(lambda p, f: network.apply(p, f, method=call))(params, field)


def _perturbed(params, path, delta=1.0):
    tree = jax.tree.map(lambda x: x, params)
    node = tree["params"]["encoder"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = node[path[-1]] + delta
    return tree


def test_field_side_tokens_do_not_read_the_active_status_table(
    real_model_and_trajectory,
):
    """The my/opp side-condition tokens must not borrow pos_bias.

    pos_bias is indexed by ENTITY_PUBLIC_NODE_FEATURE__ACTIVE (= scoreOrder,
    {0, 2} in singles), so before 2026-08-28 its row 0 meant both "benched
    pokemon" and "opponent side conditions" — and that bias was the only
    thing separating my hazards from theirs, since both sides share
    side_condition_linear.
    """
    network, params, actor_input, _ = real_model_and_trajectory
    base = _field_tokens(network, params, actor_input)

    moved_pos = _field_tokens(
        network, _perturbed(params, ("pos_bias", "embedding")), actor_input
    )
    np.testing.assert_allclose(np.asarray(base), np.asarray(moved_pos), atol=0)

    # Positive control: the replacement IS on the path, so the test above is
    # not passing merely because nothing reaches these tokens.
    moved_side = _field_tokens(
        network, _perturbed(params, ("field_side_bias",)), actor_input
    )
    assert not np.allclose(np.asarray(base[1]), np.asarray(moved_side[1]))
    assert not np.allclose(np.asarray(base[2]), np.asarray(moved_side[2]))
    # The global field token carries no side, so it must be untouched.
    np.testing.assert_allclose(np.asarray(base[0]), np.asarray(moved_side[0]), atol=0)


def _with_private_column(actor_input, row, column, value):
    """actor_input with private_team[:, row, column] set to `value`."""
    import dataclasses

    env = actor_input.env
    team = jnp.asarray(env.private_team).at[:, row, column].set(value)
    return dataclasses.replace(
        actor_input, env=dataclasses.replace(env, private_team=team)
    )


def test_private_condition_reaches_only_its_own_sheet_row(
    real_model_and_trajectory,
):
    """The truth channel is wired: a candidate's CURRENT hp on the wire moves
    its own assembled sheet row and no other -- the input-level half of what
    probe C measures behaviourally after training. Probe C's baseline read
    was r ~ 0.00 precisely because this input did not exist."""
    from rl.environment.protos.features_pb2 import EntityPrivateNodeFeature

    network, params, actor_input, _ = real_model_and_trajectory
    base = _assembled_rows(network, params, actor_input)[PRIVATE_ROWS]

    halved = _with_private_column(
        actor_input,
        2,
        EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__HP_RATIO,
        4096,
    )
    moved = _assembled_rows(network, params, halved)[PRIVATE_ROWS]

    changed_rows = ~np.all(np.isclose(base, moved, atol=1e-6), axis=-1)
    assert changed_rows[2], "the perturbed candidate's own row must move"
    assert not changed_rows[
        [0, 1, 3, 4, 5]
    ].any(), "condition must be entity-local at assembly time"


def test_entity_index_tag_links_private_to_public(real_model_and_trajectory):
    """The alignment key: perturbing the shared entity_index_tag table moves
    BOTH public and private rows (one table, applied twice), and changing a
    private row's ENTITY_IDX on the wire moves only that row. idx 0 (never
    fielded) keys the absent row, same as a filler public row."""
    from rl.environment.protos.features_pb2 import EntityPrivateNodeFeature
    from rl.model.constants import PUBLIC_ROWS

    network, params, actor_input, _ = real_model_and_trajectory
    base = _assembled_rows(network, params, actor_input)

    moved = _assembled_rows(
        network, _perturbed(params, ("entity_index_tag",)), actor_input
    )
    assert not np.allclose(base[PRIVATE_ROWS], moved[PRIVATE_ROWS], atol=1e-6)
    assert not np.allclose(base[PUBLIC_ROWS], moved[PUBLIC_ROWS], atol=1e-6)

    rekeyed_input = _with_private_column(
        actor_input,
        1,
        EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ENTITY_IDX,
        12,
    )
    rekeyed = _assembled_rows(network, params, rekeyed_input)[PRIVATE_ROWS]
    changed_rows = ~np.all(np.isclose(base[PRIVATE_ROWS], rekeyed, atol=1e-6), axis=-1)
    assert changed_rows[1]
    assert not changed_rows[[0, 2, 3, 4, 5]].any()
