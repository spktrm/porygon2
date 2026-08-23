"""Counterfactual value ladder (2026-08-16): the privileged
opp_private_team input must be readable ONLY by the main (everything)
value head — the policy and the private/public ladder heads must be bitwise
invariant to it, or the policy would train on information that does not
exist at deploy time."""

import jax
import numpy as np
from conftest import open_zero_init_paths
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _open_all_gates(params):
    """Residual gates are zero-init, which makes leak tests VACUOUS at
    init: a broken attention mask would have its leaked contribution
    multiplied by zero and pass anyway. Setting every *_gate param to 1
    forces all residual writes live, so the invariance assertions below
    genuinely exercise the masks."""

    def open_gate(path, leaf):
        names = [getattr(p, "key", "") for p in path]
        if any(str(name).endswith("gate") for name in names):
            return jax.numpy.ones_like(leaf)
        return leaf

    return jax.tree_util.tree_map_with_path(open_gate, params)


def test_policy_and_ladder_invariant_to_opp_private_team(
    real_model_and_trajectory,
):
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    params = _open_all_gates(params)
    # Gates are not the only zero-init path: e00a388's flat-at-init
    # contract zero-inits the Q head's OUTPUT paths too, so on init params
    # BOTH Q rungs emit all-zero logits. That makes the Q_private
    # sheet-invariance assertion vacuous and the Q_all difference
    # assertion impossible. Open them for the same reason the gates are
    # opened above.
    params = open_zero_init_paths(params, ("q_adapter", "q_macro_micro"))

    # ex.bin predates the field, so the baseline input carries all-zero
    # opp_private_team; the populated variant borrows the player's own
    # private team as a structurally valid sheet.
    populated = actor_input.replace(
        env=actor_input.env.replace(
            opp_private_team=actor_input.env.private_team
        )
    )

    base = network.apply(params, actor_input, actor_output, HeadParams())
    priv = network.apply(params, populated, actor_output, HeadParams())

    # The policy pathway never touches the sheet: bitwise identical.
    np.testing.assert_array_equal(
        np.asarray(base.action_head.log_policy),
        np.asarray(priv.action_head.log_policy),
    )
    # Same for the deployable and public ladder rungs.
    np.testing.assert_array_equal(
        np.asarray(base.private_value_logits), np.asarray(priv.private_value_logits)
    )
    np.testing.assert_array_equal(
        np.asarray(base.public_value_logits),
        np.asarray(priv.public_value_logits),
    )
    # Q_private is conditioned on the private value embedding — the
    # deployable information set — so it must be sheet-invariant too.
    np.testing.assert_array_equal(
        np.asarray(base.private_q_adv), np.asarray(priv.private_q_adv)
    )
    # With gates open the privileged heads MUST differ under a populated
    # sheet — if they don't, the value_read mask is over-masking and the
    # all rung never sees the sheet at all (the inverse failure mode).
    lp = np.asarray(priv.value_head.log_probs, dtype=np.float32)
    assert np.isfinite(lp).all()
    assert not np.array_equal(
        np.asarray(base.value_head.logits), np.asarray(priv.value_head.logits)
    )
    # Q_all reads the sheet through its value_all conditioning.
    assert not np.array_equal(
        np.asarray(base.q_adv), np.asarray(priv.q_adv)
    )


# NOTE (2026-08-16): the earlier empty-sheet all==private equality test was
# removed deliberately — the rungs now have separate query inits and
# residual gates (independent estimators per information route), so the
# degradation identity no longer holds and the all-vs-private gap includes
# an estimator component alongside the information value.


def test_ladder_heads_present_and_shaped(real_model_and_trajectory):
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    out = network.apply(params, actor_input, actor_output, HeadParams())
    T = actor_input.env.done.shape[0]
    n_bins = np.asarray(out.value_head.log_probs).shape[-1]
    assert np.asarray(out.private_value_logits).shape == (T, n_bins)
    assert np.asarray(out.public_value_logits).shape == (T, n_bins)
    assert np.isfinite(np.asarray(out.private_value_logits, dtype=np.float32)).all()
    assert np.isfinite(
        np.asarray(out.public_value_logits, dtype=np.float32)
    ).all()
    # Two-rung Q: same flat action grid and support for both rungs.
    A = int(np.prod(actor_input.env.action_mask.shape[-2:]))
    assert np.asarray(out.q_adv).shape == (T, A)
    assert np.asarray(out.private_q_adv).shape == (T, A)
    assert np.isfinite(np.asarray(out.private_q_adv, dtype=np.float32)).all()
