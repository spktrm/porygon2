"""Real player-model init + forward on the bundled example trajectory.

Marked gpu: runs wherever JAX puts it (the training box GPU, with
preallocation disabled by conftest so it coexists with a live learner).
Marked slow (~1 min): deselect with `-m "not slow"` for the quick suite.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import open_zero_init_paths

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


def test_q_head_forward_shapes(real_model_and_trajectory, real_model_apply):
    """The structural two-rung hierarchical Q readout (docs/
    q-critic-plan.md): owned adapter + shared MacroMicroHead params in the
    tree, (T, A, n_bins) logits per rung, rung conditioning alive."""

    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    for subtree in ("policy_head", "advantage_head", "v_head"):
        assert subtree in params["params"]

    out = real_model_apply(params, actor_input, actor_output, HeadParams())
    T = actor_input.env.done.shape[0]
    A = int(np.prod(actor_input.env.action_mask.shape[-2:]))
    adv = np.asarray(out.advantage, dtype=np.float32)
    assert adv.shape == (T, A)
    assert np.isfinite(adv).all()
    # At init the head is identically ZERO (e00a388's flat-at-init
    # contract zero-inits every advantage output path), so any test of
    # live geometry has to open the zero paths first — otherwise it
    # passes vacuously.
    opened = open_zero_init_paths(params, ("advantage_head",))
    out_open = real_model_apply(opened, actor_input, actor_output, HeadParams())
    adv_open = np.asarray(out_open.advantage, dtype=np.float32)
    assert np.isfinite(adv_open).all()
    assert adv_open.any()
    # Full-support log_policy is present in train mode — the Retrace
    # target's expectation bootstrap depends on it.
    assert np.asarray(out.action_head.log_policy).shape[-1] == A


def test_q_head_is_flat_at_init_and_local_routes_get_gradient(
    real_model_and_trajectory, real_model_apply
):
    """The flat-at-init contract (every Q cell exactly 0) AND the
    regression test for the 2026-08-24 finding: the within-modality
    route must be a single zero-init factor, so a within-modality
    signal reaches its kernels at init WITHOUT the pointer gate having
    moved. Loss = sum over legal cells of advantage * (+-1 pattern by rank
    parity within each modality) — pure within-modality by construction."""
    from rl.environment.data import FLAT_MODALITY_MASK
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    out = real_model_apply(params, actor_input, actor_output, HeadParams())
    assert not np.asarray(out.advantage, dtype=np.float32).any()

    flat = np.asarray(FLAT_MODALITY_MASK)
    pattern = np.zeros(flat.shape, np.float32)
    for mod in np.unique(flat):
        cells = np.flatnonzero(flat == mod)
        pattern[cells] = np.where(np.arange(len(cells)) % 2 == 0, 1.0, -1.0)
    legal = np.asarray(actor_input.env.action_mask).reshape(-1, flat.shape[0])

    def loss(p):
        o = real_model_apply(p, actor_input, actor_output, HeadParams())
        return jnp.sum(
            o.advantage.astype(jnp.float32) * jnp.asarray(pattern) * jnp.asarray(legal)
        )

    grads = jax.grad(loss)(params)["params"]["advantage_head"]["macro_micro"]
    for name in ("micro_local_src", "micro_local_tgt"):
        g = np.asarray(grads[name]["kernel"], dtype=np.float32)
        assert np.isfinite(g).all() and np.abs(g).max() > 0.0, name
    # The gated pointer's q/k projections get NOTHING at init — the
    # product structure this test exists to document.
    for name in ("Dense_0", "Dense_1"):
        assert not np.asarray(
            grads["micro"][name]["kernel"], dtype=np.float32
        ).any(), name


def test_forward_is_deterministic(real_model_and_trajectory, real_model_apply):
    network, params, actor_input, actor_output = real_model_and_trajectory
    from rl.model.heads import HeadParams

    a = real_model_apply(params, actor_input, actor_output, HeadParams())
    b = real_model_apply(params, actor_input, actor_output, HeadParams())
    np.testing.assert_array_equal(
        np.asarray(a.value_head.log_probs, dtype=np.float32),
        np.asarray(b.value_head.log_probs, dtype=np.float32),
    )
