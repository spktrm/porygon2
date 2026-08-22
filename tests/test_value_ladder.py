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
        np.asarray(base.private_q_logits), np.asarray(priv.private_q_logits)
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
        np.asarray(base.q_logits), np.asarray(priv.q_logits)
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
    assert np.asarray(out.q_logits).shape == (T, A, n_bins)
    assert np.asarray(out.private_q_logits).shape == (T, A, n_bins)
    assert np.isfinite(np.asarray(out.private_q_logits, dtype=np.float32)).all()


def test_intrinsic_stack_on_private_rung(real_model_and_trajectory):
    """The ensemble and V_int read the PRIVATE rung: bitwise invariant to
    opp_private_team (the bonus must be the agent's own uncertainty), with
    the expected (T, K, n_bins) / (T,) shapes, and the ensemble must not be
    degenerate at init — the randomised prior keeps the K heads apart."""
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    params = _open_all_gates(params)
    populated = actor_input.replace(
        env=actor_input.env.replace(opp_private_team=actor_input.env.private_team)
    )
    base = network.apply(params, actor_input, actor_output, HeadParams())
    priv = network.apply(params, populated, actor_output, HeadParams())

    T = actor_input.env.done.shape[0]
    n_bins = np.asarray(base.value_head.log_probs).shape[-1]
    ens = np.asarray(base.ens_value_logits, dtype=np.float32)
    assert ens.ndim == 3 and ens.shape[0] == T and ens.shape[-1] == n_bins
    assert ens.shape[1] >= 2
    assert np.isfinite(ens).all()
    assert np.asarray(base.int_value).shape == (T,)
    assert np.isfinite(np.asarray(base.int_value, dtype=np.float32)).all()

    np.testing.assert_array_equal(
        np.asarray(base.ens_value_logits), np.asarray(priv.ens_value_logits)
    )
    np.testing.assert_array_equal(np.asarray(base.int_value), np.asarray(priv.int_value))

    # Heads differ at init (prior twin is live): the spread is not zero.
    assert np.abs(ens - ens.mean(axis=1, keepdims=True)).max() > 0.0


def test_q_ensemble_private_and_ucb_tilt_contract(real_model_and_trajectory):
    """Q ensemble reads the private rung (sheet-invariant, (T, A, K, n_bins),
    non-degenerate at init via the prior); HeadParams.ucb_c = 0 is bitwise
    pi; ucb_c > 0 changes the behaviour log_prob, keeps log_policy (pi)
    bitwise, and records KL(mu||pi) under the configured cap."""
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    params = _open_all_gates(params)
    populated = actor_input.replace(
        env=actor_input.env.replace(opp_private_team=actor_input.env.private_team)
    )
    base = network.apply(params, actor_input, actor_output, HeadParams())
    priv = network.apply(params, populated, actor_output, HeadParams())
    T = actor_input.env.done.shape[0]
    A = int(np.prod(actor_input.env.action_mask.shape[-2:]))
    n_bins = np.asarray(base.value_head.log_probs).shape[-1]
    q = np.asarray(base.q_ens_logits, dtype=np.float32)
    assert q.shape[0] == T and q.shape[1] == A and q.shape[-1] == n_bins
    assert q.shape[2] >= 2 and np.isfinite(q).all()
    np.testing.assert_array_equal(np.asarray(base.q_ens_logits), np.asarray(priv.q_ens_logits))
    assert np.abs(q - q.mean(axis=2, keepdims=True)).max() > 0.0
    # c = 0: bitwise pi everywhere, zero KL.
    assert np.all(np.asarray(base.action_head.ucb_kl) == 0.0)
    zero = network.apply(params, actor_input, actor_output, HeadParams(ucb_c=0.0))
    np.testing.assert_array_equal(
        np.asarray(base.action_head.log_prob), np.asarray(zero.action_head.log_prob)
    )
    # c > 0: the learner path (train=True, action given) still evaluates
    # log_prob under mu, pi untouched, KL within the cap.
    hot = network.apply(params, actor_input, actor_output, HeadParams(ucb_c=3.0))
    np.testing.assert_array_equal(
        np.asarray(base.action_head.log_policy), np.asarray(hot.action_head.log_policy)
    )
    kl = np.asarray(hot.action_head.ucb_kl, dtype=np.float32)
    assert np.isfinite(kl).all() and kl.max() > 0.0
    assert kl.max() <= network.cfg.q_ens.kl_max * 1.1
    assert not np.array_equal(
        np.asarray(base.action_head.log_prob), np.asarray(hot.action_head.log_prob)
    )
