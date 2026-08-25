"""The Q = V + A contract.

Replaces tests/test_value_ladder.py (deleted 2026-08-25 with the privileged
critic). That test pinned an INFORMATION routing contract; there is no
privileged input left to route, so what needs pinning now is the DECOMPOSITION:

    Q(s, a) = sg(V(s)) + A(s, a) - E_sg(pi)[A(s, .)]        over legal cells

with two identifiability tiers that are deliberately different:

  * micro is uniform-mean-zero within each modality's legal cells (identifies
    micro against macro). Uniform, NOT pi-weighted — a pi-weighted centring
    here would reintroduce the pi prefactor that docs/entropy-gradient-pressure.md
    shows can never restore a starved modality.
  * A is pi-mean-zero across all legal cells (identifies A against V). This one
    MUST be pi-weighted: it is what makes E_pi[Q] = V exactly, which is what
    makes V the correct NeuRD baseline.

Both stop-gradients are load-bearing. sg(V) closes the state route, so the
taken-cell Huber loss cannot satisfy itself with a state-only function (the
Step-6 verdict in docs/critic-weakness-analysis.md). sg(pi) stops the Q loss
from pushing the policy through the centring term.
"""

import numpy as np
import pytest

# NOTE: markers are per-test, not module-level — the compose_action_grid
# unit test below is pure numpy/jnp and belongs in the fast suite.
_SLOW = [pytest.mark.slow, pytest.mark.gpu]


def _legal(actor_input):
    mask = np.asarray(actor_input.env.action_mask)
    return mask.reshape(*mask.shape[:-2], -1).astype(bool)


@pytest.mark.slow
@pytest.mark.gpu
def test_flat_at_init_advantage_is_zero_and_q_is_v(
    real_model_and_trajectory, real_model_apply
):
    """Flat-at-init: every advantage cell is EXACTLY 0, so Q collapses to V.

    This is the anchor the whole head design rests on — the policy starts at
    calculate_hierarchical_prior and no lecun noise poses as an action
    preference for the loss to unlearn.
    """
    from rl.model.heads import HeadParams

    _, params, actor_input, actor_output = real_model_and_trajectory
    out = real_model_apply(params, actor_input, actor_output, HeadParams())

    adv = np.asarray(out.advantage, dtype=np.float32)
    assert not adv.any(), "advantage must be exactly zero at init"

    legal = _legal(actor_input)
    q = np.asarray(out.q, dtype=np.float32)
    v = np.asarray(out.value_head.expectation, dtype=np.float32)
    np.testing.assert_allclose(q[legal], np.broadcast_to(v[..., None], q.shape)[legal])


@pytest.mark.slow
@pytest.mark.gpu
def test_pi_centring_gives_expected_q_equals_v(
    real_model_and_trajectory, real_model_apply
):
    """E_pi[A] = 0 and therefore E_pi[Q] = V, on OPENED params.

    At init this would pass vacuously (A is identically zero), so the zero-init
    output paths are opened first — the same trap LESSONS.md section 7 records.
    """
    from conftest import open_zero_init_paths

    from rl.model.heads import HeadParams

    _, params, actor_input, actor_output = real_model_and_trajectory
    opened = open_zero_init_paths(params, ("advantage_head",))
    out = real_model_apply(opened, actor_input, actor_output, HeadParams())

    legal = _legal(actor_input)
    adv = np.asarray(out.advantage, dtype=np.float32)
    assert adv[legal].any(), "negative control: opened params must move A off zero"

    log_pi = np.asarray(out.action_head.log_policy, dtype=np.float32)
    pi = np.exp(log_pi) * legal
    pi = pi / np.maximum(pi.sum(-1, keepdims=True), 1e-8)

    e_adv = (pi * np.where(legal, adv, 0.0)).sum(-1)
    np.testing.assert_allclose(e_adv, 0.0, atol=2e-5)

    q = np.asarray(out.q, dtype=np.float32)
    v = np.asarray(out.value_head.expectation, dtype=np.float32)
    e_q = (pi * np.where(legal, q, 0.0)).sum(-1)
    np.testing.assert_allclose(e_q, v, atol=2e-5)


def test_compose_q_centring_and_closed_state_route():
    """E_pi[Q] == V exactly; a constant shift of A is invisible; illegal
    cells read 0; and the action axis survives the centring.

    Moved here from tests/test_targets.py::TestResidualQ when the
    composition moved out of the learner and into the model (2026-08-25).
    """
    import jax.numpy as jnp

    from rl.model.heads import compose_q

    legal = jnp.asarray([[True, True, False, True]])
    log_pi = jnp.log(jnp.asarray([[0.5, 0.25, 0.0, 0.25]]) + 1e-12)
    adv = jnp.asarray([[0.3, -0.1, 5.0, 0.2]])
    v = jnp.asarray([0.4])

    advantage, q = compose_q(v, adv, log_pi, legal)
    pi = jnp.exp(log_pi) * legal

    np.testing.assert_allclose(float((pi * q).sum(-1)[0]), 0.4, atol=1e-6)
    # E_pi[A] = 0 is the identifiability constraint that makes the above hold.
    np.testing.assert_allclose(float((pi * advantage).sum(-1)[0]), 0.0, atol=1e-6)
    assert float(q[0, 2]) == 0.0
    assert float(advantage[0, 2]) == 0.0

    # A constant shift of A is invisible: the state route is closed, so a
    # state-level offset can live ONLY in V.
    _, shifted = compose_q(v, adv + 7.0, log_pi, legal)
    np.testing.assert_allclose(np.asarray(shifted), np.asarray(q), atol=1e-5)

    # ...while the action axis survives: cell gaps equal the raw A gaps.
    np.testing.assert_allclose(float(q[0, 0] - q[0, 1]), 0.4, atol=1e-6)


def test_compose_q_stops_gradient_into_v_and_pi():
    """The two stop-gradients, at the function level.

    d/dV and d/dlog_pi of any function of compose_q's output are exactly
    zero; d/dA is not (the positive control).
    """
    import jax
    import jax.numpy as jnp

    from rl.model.heads import compose_q

    legal = jnp.asarray([[True, True, False, True]])
    log_pi = jnp.log(jnp.asarray([[0.5, 0.25, 0.0, 0.25]]) + 1e-12)
    adv = jnp.asarray([[0.3, -0.1, 5.0, 0.2]])
    v = jnp.asarray([0.4])

    def total(v_, adv_, log_pi_):
        return compose_q(v_, adv_, log_pi_, legal)[1].sum()

    dv, dadv, dlogpi = jax.grad(total, argnums=(0, 1, 2))(v, adv, log_pi)
    assert float(jnp.abs(dv).sum()) == 0.0, "sg(V) leaked"
    assert float(jnp.abs(dlogpi).sum()) == 0.0, "sg(pi) leaked"
    assert float(jnp.abs(dadv).sum()) > 0.0, "positive control: A must get gradient"


@pytest.mark.parametrize("seed", [0, 1])
def test_micro_tier_is_centred_uniformly_not_by_pi(seed):
    """The micro tier's within-modality centring is UNIFORM over legal cells.

    A pure unit test of compose_action_grid — no model, no GPU. Pins the tier
    that must NOT be pi-weighted: a pi-weighted within-modality mean would put
    a mass prefactor back on starved cells, which
    docs/entropy-gradient-pressure.md shows can never restore a dead modality.
    The negative control is the pi-weighted variant, which must NOT satisfy it.
    """
    import jax.numpy as jnp

    from rl.environment.data import (
        FLAT_MODALITY_MASK,
        NUM_MODALITY_FEATURES,
    )
    from rl.model.heads import compose_action_grid

    flat = np.asarray(FLAT_MODALITY_MASK)
    rng = np.random.default_rng(seed)
    macro = rng.normal(size=(NUM_MODALITY_FEATURES,)).astype(np.float32)
    micro = rng.normal(size=flat.shape).astype(np.float32)
    legal = rng.random(flat.shape) < 0.4
    # Guarantee every modality keeps at least one legal cell.
    for mod in np.unique(flat):
        cells = np.flatnonzero(flat == mod)
        legal[cells[0]] = True

    out = np.asarray(
        compose_action_grid(
            jnp.asarray(macro), jnp.asarray(micro), jnp.asarray(legal), reduce="mean"
        ),
        dtype=np.float32,
    )

    # A skewed pi: most mass on one cell, so uniform and pi-weighted means differ.
    pi = rng.random(flat.shape).astype(np.float32) ** 4 * legal
    pi = pi / pi.sum()

    for mod in np.unique(flat):
        sel = legal & (flat == mod)
        if not sel.any():
            continue
        resid = out[sel] - macro[mod]
        # Uniform mean of (out - macro) over the modality's legal cells is 0.
        assert abs(float(resid.mean())) < 2e-5, f"modality {mod} not uniform-centred"
        w = pi[sel]
        if w.sum() > 0 and sel.sum() > 1:
            pi_mean = float((resid * w).sum() / w.sum())
            # Negative control: the pi-weighted mean is NOT what was removed.
            assert abs(pi_mean) > 1e-6, (
                f"modality {mod}: pi-weighted mean vanished too — the centring "
                "statistic may have been switched to a pi-weighted one"
            )


@pytest.mark.slow
@pytest.mark.gpu
def test_q_loss_gradient_reaches_the_advantage_head_but_not_v_or_the_policy(
    real_model_and_trajectory,
):
    """Both stop-gradients, with a positive control.

    A taken-cell Q loss must move the advantage head (positive control) while
    leaving v_head and the policy head bitwise untouched — sg(V) closes the
    state route, sg(pi) stops the centring term from steering the policy.
    """
    import jax
    import jax.numpy as jnp

    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    legal = jnp.asarray(_legal(actor_input))

    def loss_fn(p):
        out = network.apply(p, actor_input, actor_output, HeadParams())
        q = out.q.astype(jnp.float32)
        # Taken-cell-shaped: one legal cell per row, exactly how loss_q reads it.
        taken = jnp.argmax(legal.astype(jnp.float32), axis=-1)
        return jnp.take_along_axis(q, taken[..., None], axis=-1).sum()

    grads = jax.grad(loss_fn)(params)["params"]

    def norm(subtree):
        return float(
            sum(np.square(np.asarray(x, np.float64)).sum() for x in jax.tree.leaves(subtree))
        )

    assert norm(grads["advantage_head"]) > 0.0, "positive control: A head must get gradient"
    assert norm(grads["v_head"]) == 0.0, "sg(V) leaked: the Q loss moved the critic"
    assert norm(grads["policy_head"]) == 0.0, "sg(pi) leaked: the Q loss moved the policy"
