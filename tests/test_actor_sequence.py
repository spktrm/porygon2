"""The actor's sequence (2026-09-04): under cfg.train=False the encoder
assembles the policy-readable rows only -- no opponent code rows, no
VALUE_CLS -- and the trunk runs on them with the read mask's own sub-block.
The policy-readable rows have no in-edge from the dropped partition at any
block (SEQUENCE_READ_MASK), so on the same params the actor's policy and
value read the same numbers as the learner's, up to GEMM shape numerics.

Fast: the layout contract. Slow: the numerical equivalence on real params,
with the control that the readout is live (the zero-init readout would
make a uniform-vs-uniform comparison pass vacuously)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.model.constants import (
    CLS_ROW,
    DYNAMICS_TARGET_ROWS,
    HISTORY_ENTITY_ROWS,
    MOVE_ROWS,
    NUM_POLICY_READABLE_ROWS,
    NUM_SEQUENCE_ROWS,
    OPP_PRIVATE_ROWS,
    POLICY_READABLE_ROWS,
    PRIVATE_ROWS,
    PUBLIC_ROWS,
    SEQUENCE_GROUP_IDS,
    SEQUENCE_READ_MASK,
    TARGET_ROWS,
    VALUE_CLS_ROW,
    SequenceGroup,
)
from rl.model.heads import HeadParams


def test_actor_rows_are_the_policy_readable_prefix_plus_history():
    dropped = np.setdiff1d(np.arange(NUM_SEQUENCE_ROWS), POLICY_READABLE_ROWS)
    np.testing.assert_array_equal(
        dropped,
        np.concatenate(
            (np.arange(*OPP_PRIVATE_ROWS.indices(NUM_SEQUENCE_ROWS)), [VALUE_CLS_ROW])
        ),
    )
    assert NUM_POLICY_READABLE_ROWS == NUM_SEQUENCE_ROWS - len(dropped)
    # Every head-read row keeps its absolute index: they all sit below the
    # first dropped row, and the kept rows are the identity up to there.
    first_dropped = int(dropped.min())
    np.testing.assert_array_equal(
        POLICY_READABLE_ROWS[:first_dropped], np.arange(first_dropped)
    )
    for row in (
        CLS_ROW,
        PUBLIC_ROWS.stop - 1,
        PRIVATE_ROWS.stop - 1,
        MOVE_ROWS.stop - 1,
        TARGET_ROWS.stop - 1,
        int(np.max(DYNAMICS_TARGET_ROWS)),
    ):
        assert row < first_dropped
    # The history rows are the ones that move, and nothing indexes them
    # absolutely on the actor path: they shift down by the dropped count.
    kept_history = np.flatnonzero(
        SEQUENCE_GROUP_IDS[POLICY_READABLE_ROWS] == SequenceGroup.HISTORY_ENTITY
    )
    np.testing.assert_array_equal(
        kept_history,
        np.arange(*HISTORY_ENTITY_ROWS.indices(NUM_SEQUENCE_ROWS)) - len(dropped),
    )
    # The sub-mask the actor's trunk runs under is all-True: the kept rows
    # read each other freely, so no mask is being dropped with the rows.
    assert SEQUENCE_READ_MASK[np.ix_(POLICY_READABLE_ROWS, POLICY_READABLE_ROWS)].all()


@pytest.mark.gpu
@pytest.mark.slow
def test_actor_forward_matches_the_learner_forward_on_the_kept_rows(
    real_model_and_trajectory,
):
    from rl.model.config import get_player_model_config
    from rl.model.player_model import get_player_model
    from rl.model.utils import open_zero_init_paths

    _, params, actor_input, actor_output = real_model_and_trajectory
    params = open_zero_init_paths(params, ["action_head"])
    # f32 on both sides, and full-precision GEMMs (the GPU default for f32
    # is TF32, which reads ~1e-3 on the value logits), so the only
    # difference left is the sequence length -- a GEMM leading-dim change,
    # ~1e-6 relative here against the 0.05 the bf16 forward is allowed.
    learner = get_player_model(
        get_player_model_config(generation=9, train=True, dtype=jnp.float32)
    )
    actor = get_player_model(
        get_player_model_config(generation=9, train=False, dtype=jnp.float32)
    )
    with jax.default_matmul_precision("highest"):
        learner_out = jax.jit(learner.apply)(
            params, actor_input, actor_output, HeadParams()
        )
        actor_out = jax.jit(actor.apply)(
            params,
            actor_input,
            actor_output,
            HeadParams(),
            rngs={"sampling": jax.random.key(0)},
        )

    def diff(left, right) -> float:
        return float(
            np.abs(np.asarray(left, np.float32) - np.asarray(right, np.float32)).max()
        )

    # The actor emits no log_policy (learner-only), so compare the row
    # statistics it does emit -- each a function of the whole legal
    # log_policy, so a row-wise agreement is an agreement of the policy.
    for name in ("entropy", "normalized_entropy", "magnet_kl"):
        assert (
            diff(
                getattr(learner_out.action_head, name),
                getattr(actor_out.action_head, name),
            )
            < 1e-3
        ), name
    assert diff(learner_out.value_head.logits, actor_out.value_head.logits) < 1e-3
    # The actor carries none of the learner-only outputs.
    assert isinstance(actor_out.history_carry.valid, jax.Array)
    for name in (
        "belief_logits",
        "dynamics_target",
        "opp_code",
        "hidden_code",
        "log_policy",
    ):
        assert getattr(actor_out, name, ()) == ()
    assert actor_out.action_head.log_policy == ()
    # Control: the readout is live -- consecutive steps disagree, so the
    # equality above is not two uniform policies agreeing by construction.
    entropy = np.asarray(actor_out.action_head.entropy, np.float32)
    assert np.abs(entropy[10] - entropy[11]) > 1e-3
