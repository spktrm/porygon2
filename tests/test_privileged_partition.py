"""The leak partition (2026-09-01): the opponent-truth rows and VALUE_CLS
are learner-only BY MASK, and leak-freedom must be transitive across trunk
depth. Both halves of every invariance test carry the positive control the
CLAUDE.md test-trap rule demands: a perturbation the mask should pass, and
proof the blocked perturbation genuinely moves the row allowed to see it.

Fast half: trunk-level, tiny width, multi-block (depth is what makes a mask
hole compound). Slow half: the real model end to end -- perturbing
`opp_private_team` on the wire must be invisible to the action head and the
deployable value head while the privileged head moves.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import ConfigDict

from rl.model.constants import (
    NUM_SEQUENCE_ROWS,
    OPP_PRIVATE_ROWS,
    SEQUENCE_READ_MASK,
    VALUE_CLS_ROW,
)
from rl.model.trunk import Trunk

READ_MASK = jnp.asarray(SEQUENCE_READ_MASK)

WIDTH = 32


def _trunk_cfg(num_blocks=3):
    cfg = ConfigDict()
    cfg.num_blocks = num_blocks
    cfg.num_heads = 2
    cfg.qk_size = 16
    cfg.v_size = 16
    cfg.model_size = WIDTH
    cfg.hidden_size = 2 * WIDTH
    cfg.qk_layer_norm = True
    cfg.use_bias = False
    return cfg


_POLICY_READABLE = np.array(
    [
        row not in range(OPP_PRIVATE_ROWS.start, OPP_PRIVATE_ROWS.stop)
        and row != VALUE_CLS_ROW
        for row in range(NUM_SEQUENCE_ROWS)
    ]
)


def test_read_mask_partition_is_leak_free_by_construction():
    # The static matrix itself: no policy-readable in-edge from the
    # learner-only partition, VALUE_CLS out-degree 0, and (control) the
    # policy-readable block is complete.
    blocked = SEQUENCE_READ_MASK[np.ix_(_POLICY_READABLE, ~_POLICY_READABLE)]
    assert not blocked.any()
    assert not SEQUENCE_READ_MASK[_POLICY_READABLE][:, VALUE_CLS_ROW].any()
    assert SEQUENCE_READ_MASK[np.ix_(_POLICY_READABLE, _POLICY_READABLE)].all()
    assert SEQUENCE_READ_MASK[VALUE_CLS_ROW].all()


def test_secret_rows_are_invisible_to_policy_readable_rows_at_depth():
    trunk = Trunk(_trunk_cfg())
    sequence = jax.random.normal(jax.random.key(0), (NUM_SEQUENCE_ROWS, WIDTH))
    valid = jnp.ones(NUM_SEQUENCE_ROWS, bool)
    params = trunk.init(jax.random.key(1), sequence, valid, READ_MASK)

    perturbed = sequence.at[OPP_PRIVATE_ROWS].add(10.0)
    base = np.asarray(trunk.apply(params, sequence, valid, READ_MASK), dtype=np.float32)
    moved = np.asarray(
        trunk.apply(params, perturbed, valid, READ_MASK), dtype=np.float32
    )

    # Every policy-readable row is BIT-identical across three blocks of
    # mixing -- transitivity, not just first-block masking.
    np.testing.assert_array_equal(base[_POLICY_READABLE], moved[_POLICY_READABLE])
    # Control #1: VALUE_CLS, the one row allowed to read them, moves.
    assert not np.allclose(base[VALUE_CLS_ROW], moved[VALUE_CLS_ROW])
    # Control #2: the same-size perturbation on a policy-readable row does
    # reach its peers -- the invariance above is the mask, not a dead trunk.
    control = sequence.at[3].add(10.0)
    control_out = np.asarray(
        trunk.apply(params, control, valid, READ_MASK), dtype=np.float32
    )
    assert not np.allclose(base[_POLICY_READABLE], control_out[_POLICY_READABLE])


def test_value_cls_is_read_by_nothing():
    trunk = Trunk(_trunk_cfg())
    sequence = jax.random.normal(jax.random.key(2), (NUM_SEQUENCE_ROWS, WIDTH))
    valid = jnp.ones(NUM_SEQUENCE_ROWS, bool)
    params = trunk.init(jax.random.key(3), sequence, valid, READ_MASK)

    perturbed = sequence.at[VALUE_CLS_ROW].add(10.0)
    base = np.asarray(trunk.apply(params, sequence, valid, READ_MASK), dtype=np.float32)
    moved = np.asarray(
        trunk.apply(params, perturbed, valid, READ_MASK), dtype=np.float32
    )
    others = np.arange(NUM_SEQUENCE_ROWS) != VALUE_CLS_ROW
    np.testing.assert_array_equal(base[others], moved[others])
    # Control: its own output moves (it reads itself).
    assert not np.allclose(base[VALUE_CLS_ROW], moved[VALUE_CLS_ROW])


def test_belief_alignment_matches_only_the_opponent_half():
    import numpy as np

    from rl.environment.protos.features_pb2 import (
        EntityPrivateNodeFeature,
        InfoFeature,
    )
    from rl.model.player_model import belief_alignment

    num_features = 23
    opp = np.zeros((6, num_features), dtype=np.int32)
    idx_col = EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ENTITY_IDX
    # Mon 0: fielded, stable index 4 (sits in the opp half below).
    opp[0, idx_col] = 5
    # Mon 1: never fielded.
    opp[1, idx_col] = 0
    # Mon 2: index 1 -- present ONLY in the MY half below; must NOT match
    # (the cross-side alias control).
    opp[2, idx_col] = 2

    info = np.zeros(InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11 + 1, dtype=np.int32)
    order = np.full(12, -1, dtype=np.int32)
    order[0] = 1  # my half carries stable index 1
    order[7] = 4  # opp half row 7 carries stable index 4
    info[
        InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0 : InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11
        + 1
    ] = order

    matched, rows = jax.tree.map(
        np.asarray, belief_alignment(jnp.asarray(opp), jnp.asarray(info))
    )
    assert matched.tolist() == [True, False, False, False, False, False]
    assert rows[0] == 7


@pytest.mark.gpu
@pytest.mark.slow
def test_opp_private_team_cannot_reach_the_policy(
    real_model_and_trajectory, real_model_apply
):
    """End to end on the real model: the wire leaf the opponent truth rides
    must be invisible to everything an actor ships, while the privileged
    head (the one consumer) moves -- the positive control that the leaf is
    genuinely live."""
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    base = real_model_apply(params, actor_input, actor_output, HeadParams())

    opp = np.asarray(actor_input.env.opp_private_team).copy()
    assert opp.any(), "fixture must carry real opponent truth (regenerate ex.bin)"
    # Scramble the truth: reverse the mon rows on every step. Same schema,
    # different content -- a pure information perturbation.
    perturbed_env = actor_input.env.replace(
        opp_private_team=jnp.asarray(opp[:, ::-1, :])
    )
    perturbed_input = actor_input.replace(env=perturbed_env)
    moved = real_model_apply(params, perturbed_input, actor_output, HeadParams())

    np.testing.assert_array_equal(
        np.asarray(base.action_head.log_policy, dtype=np.float32),
        np.asarray(moved.action_head.log_policy, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(base.action_head.log_prob, dtype=np.float32),
        np.asarray(moved.action_head.log_prob, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(base.value_head.log_probs, dtype=np.float32),
        np.asarray(moved.value_head.log_probs, dtype=np.float32),
    )
    # The transition model (2026-09-05) runs from the policy's information
    # set: the prior, g and every head on the imagined rows read the
    # policy-readable rows at t, and the posterior reads the policy-
    # readable rows at t+1 -- leak-free by the trunk's read mask -- so
    # every transition leaf is pinned, the posterior included. The
    # grounding label is pre-trunk content of the same rows.
    for leaf in (
        "transition_prior_logits",
        "transition_post_logits",
        "transition_ground",
        "transition_ground_prior",
        "transition_mask_logits",
        "transition_log_policy",
        "transition_kind_logits",
        "transition_done_logit",
        "transition_cons_err",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(base, leaf), dtype=np.float32),
            np.asarray(getattr(moved, leaf), dtype=np.float32),
            err_msg=leaf,
        )
    np.testing.assert_array_equal(
        np.asarray(base.dynamics_target, dtype=np.float32),
        np.asarray(moved.dynamics_target, dtype=np.float32),
    )
    # Controls: the privileged head and the code labels DO move.
    assert not np.allclose(
        np.asarray(base.priv_value_head.expectation, dtype=np.float32),
        np.asarray(moved.priv_value_head.expectation, dtype=np.float32),
    )
    assert not np.array_equal(
        np.asarray(base.opp_code, dtype=np.float32),
        np.asarray(moved.opp_code, dtype=np.float32),
    )
