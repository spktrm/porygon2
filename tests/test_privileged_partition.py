"""The leak partition (2026-09-01; public tier 2026-09-15): the opponent-truth
rows and VALUE_CLS are learner-only BY MASK, the public tier reads nothing
above itself, and leak-freedom must be transitive across trunk depth. Both halves of every invariance test carry the positive control the
CLAUDE.md test-trap rule demands: a perturbation the mask should pass, and
proof the blocked perturbation genuinely moves the row allowed to see it.

Fast half: trunk-level, tiny width, multi-block (depth is what makes a mask
hole compound). Slow half: the real model end to end -- perturbing
`opp_private_team` on the wire must be invisible to the action head and the
deployable value head while the privileged head moves.
"""

from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import ConfigDict

from rl.environment.interfaces import PlayerActorInput, PlayerActorOutput
from rl.model.constants import (
    NUM_SEQUENCE_ROWS,
    OPP_PRIVATE_ROWS,
    PRIVATE_REGISTER_ROWS,
    PRIVATE_TIER_ROWS,
    PRIVILEGED_REGISTER_ROWS,
    PUBLIC_CLS_ROW,
    PUBLIC_REGISTER_ROWS,
    PUBLIC_TIER_ROWS,
    SEQUENCE_READ_MASK,
    VALUE_CLS_ROW,
)
from rl.model.trunk import Trunk

READ_MASK = jnp.asarray(SEQUENCE_READ_MASK)

WIDTH = 32


def _trunk_cfg(num_blocks: int = 3) -> ConfigDict:
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
        and row
        not in range(PRIVILEGED_REGISTER_ROWS.start, PRIVILEGED_REGISTER_ROWS.stop)
        and row not in (VALUE_CLS_ROW, PUBLIC_CLS_ROW)
        for row in range(NUM_SEQUENCE_ROWS)
    ]
)


def test_read_mask_partition_is_leak_free_by_construction() -> None:
    # The static matrix itself: no policy-readable in-edge from the
    # learner-only partition, no public in-edge from the private tier,
    # VALUE_CLS out-degree 0, and (control) the private rows read the whole
    # policy-readable set while every row reads the public tier.
    blocked = SEQUENCE_READ_MASK[np.ix_(_POLICY_READABLE, ~_POLICY_READABLE)]
    assert not blocked.any()
    assert not SEQUENCE_READ_MASK[_POLICY_READABLE][:, VALUE_CLS_ROW].any()
    assert not SEQUENCE_READ_MASK[np.ix_(PUBLIC_TIER_ROWS, PRIVATE_TIER_ROWS)].any()
    assert SEQUENCE_READ_MASK[np.ix_(PRIVATE_TIER_ROWS, _POLICY_READABLE)].all()
    assert SEQUENCE_READ_MASK[:, PUBLIC_TIER_ROWS].all()
    assert SEQUENCE_READ_MASK[VALUE_CLS_ROW, _POLICY_READABLE].all()
    assert len(PUBLIC_TIER_ROWS) + len(PRIVATE_TIER_ROWS) == _POLICY_READABLE.sum()
    # PUBLIC_CLS: in-edges from the public tier and itself only, out-degree 0.
    public_cls_reads = np.flatnonzero(SEQUENCE_READ_MASK[PUBLIC_CLS_ROW])
    np.testing.assert_array_equal(
        public_cls_reads, np.sort(np.r_[PUBLIC_TIER_ROWS, PUBLIC_CLS_ROW])
    )
    others = np.arange(NUM_SEQUENCE_ROWS) != PUBLIC_CLS_ROW
    assert not SEQUENCE_READ_MASK[others, PUBLIC_CLS_ROW].any()
    # The registers are ordinary rows of their tier: public ones read the
    # public tier, private ones the policy-readable set, privileged ones
    # the secret partition too -- and the privileged pair is read by
    # VALUE_CLS and its secret siblings only.
    public_reg = np.arange(*PUBLIC_REGISTER_ROWS.indices(NUM_SEQUENCE_ROWS))
    private_reg = np.arange(*PRIVATE_REGISTER_ROWS.indices(NUM_SEQUENCE_ROWS))
    privileged_reg = np.arange(*PRIVILEGED_REGISTER_ROWS.indices(NUM_SEQUENCE_ROWS))
    assert np.isin(public_reg, PUBLIC_TIER_ROWS).all()
    assert np.isin(private_reg, PRIVATE_TIER_ROWS).all()
    assert not _POLICY_READABLE[privileged_reg].any()
    assert SEQUENCE_READ_MASK[np.ix_(privileged_reg, _POLICY_READABLE)].all()
    assert SEQUENCE_READ_MASK[
        np.ix_(privileged_reg, np.arange(*OPP_PRIVATE_ROWS.indices(NUM_SEQUENCE_ROWS)))
    ].all()
    assert not SEQUENCE_READ_MASK[np.ix_(_POLICY_READABLE, privileged_reg)].any()
    assert SEQUENCE_READ_MASK[VALUE_CLS_ROW, privileged_reg].all()


def test_private_rows_are_invisible_to_public_rows_at_depth() -> None:
    trunk = Trunk(_trunk_cfg())
    sequence = jax.random.normal(jax.random.key(4), (NUM_SEQUENCE_ROWS, WIDTH))
    valid = jnp.ones(NUM_SEQUENCE_ROWS, bool)
    params = trunk.init(jax.random.key(5), sequence, valid, READ_MASK)

    perturbed = sequence.at[PRIVATE_TIER_ROWS].add(10.0)
    base = np.asarray(trunk.apply(params, sequence, valid, READ_MASK), dtype=np.float32)
    moved = np.asarray(
        trunk.apply(params, perturbed, valid, READ_MASK), dtype=np.float32
    )

    # The public tier and the public critic's row are BIT-identical across
    # three blocks of mixing: functions of public state alone at any depth.
    np.testing.assert_array_equal(base[PUBLIC_TIER_ROWS], moved[PUBLIC_TIER_ROWS])
    np.testing.assert_array_equal(base[PUBLIC_CLS_ROW], moved[PUBLIC_CLS_ROW])
    # Control #1: the private rows, which read themselves, move.
    assert not np.allclose(base[PRIVATE_TIER_ROWS], moved[PRIVATE_TIER_ROWS])
    # Control #2: the same-size perturbation on a public row reaches its
    # public peers -- the invariance above is the mask, not a dead trunk.
    control = sequence.at[PUBLIC_TIER_ROWS[0]].add(10.0)
    control_out = np.asarray(
        trunk.apply(params, control, valid, READ_MASK), dtype=np.float32
    )
    assert not np.allclose(base[PUBLIC_TIER_ROWS], control_out[PUBLIC_TIER_ROWS])


def test_secret_rows_are_invisible_to_policy_readable_rows_at_depth() -> None:
    trunk = Trunk(_trunk_cfg())
    sequence = jax.random.normal(jax.random.key(0), (NUM_SEQUENCE_ROWS, WIDTH))
    valid = jnp.ones(NUM_SEQUENCE_ROWS, bool)
    params = trunk.init(jax.random.key(1), sequence, valid, READ_MASK)

    perturbed = sequence.at[OPP_PRIVATE_ROWS].add(10.0)
    perturbed = perturbed.at[PRIVILEGED_REGISTER_ROWS].add(10.0)
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


def test_value_cls_is_read_by_nothing() -> None:
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


@pytest.mark.gpu
@pytest.mark.slow
def test_opp_private_team_cannot_reach_the_policy(
    real_model_and_trajectory: tuple[
        nn.Module, dict, PlayerActorInput, PlayerActorOutput
    ],
    real_model_apply: Callable,
) -> None:
    """End to end on the real model: the wire leaf the opponent truth rides
    must be invisible to everything an actor ships, while the privileged
    head (the one consumer) moves -- the positive control that the leaf is
    genuinely live."""
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    base = real_model_apply(params, actor_input, actor_output, HeadParams())

    opp = np.asarray(actor_input.env.opp_private_team).copy()
    assert opp.any(), "fixture must carry real opponent truth (regenerate ex.bin)"
    # Scramble the truth: every opponent row becomes a copy of the first
    # mon, on every step. Same schema, different content -- a pure
    # information perturbation. NOT a permutation of the rows: since the
    # per-row bias went (d5bb6a9, 2026-09-10) the sequence is a set, so
    # reversing the six rows moved nothing any attention read and the
    # control passed vacuously.
    perturbed_env = actor_input.env.replace(
        opp_private_team=jnp.asarray(np.broadcast_to(opp[:, :1, :], opp.shape))
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
    # Control: the privileged head DOES move.
    assert not np.allclose(
        np.asarray(base.priv_value_head.expectation, dtype=np.float32),
        np.asarray(moved.priv_value_head.expectation, dtype=np.float32),
    )
    # The public critic sits below the secret partition in the nesting too.
    np.testing.assert_array_equal(
        np.asarray(base.public_value_head.log_probs, dtype=np.float32),
        np.asarray(moved.public_value_head.log_probs, dtype=np.float32),
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_own_private_team_cannot_reach_the_public_critic(
    real_model_and_trajectory: tuple[
        nn.Module, dict, PlayerActorInput, PlayerActorOutput
    ],
    real_model_apply: Callable,
) -> None:
    """End to end: my own sheet must be invisible to the public critic
    while the deployable critic, which reads CLS in the private tier,
    moves -- the positive control that the leaf is live."""
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    base = real_model_apply(params, actor_input, actor_output, HeadParams())

    mine = np.asarray(actor_input.env.private_team).copy()
    assert mine.any(), "fixture must carry a real private sheet"
    perturbed_env = actor_input.env.replace(
        private_team=jnp.asarray(np.broadcast_to(mine[:, :1, :], mine.shape))
    )
    perturbed_input = actor_input.replace(env=perturbed_env)
    moved = real_model_apply(params, perturbed_input, actor_output, HeadParams())

    np.testing.assert_array_equal(
        np.asarray(base.public_value_head.log_probs, dtype=np.float32),
        np.asarray(moved.public_value_head.log_probs, dtype=np.float32),
    )
    assert not np.allclose(
        np.asarray(base.value_head.expectation, dtype=np.float32),
        np.asarray(moved.value_head.expectation, dtype=np.float32),
    )
