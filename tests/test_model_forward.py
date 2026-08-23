"""Real player-model init + forward on the bundled example trajectory.

Marked gpu: runs wherever JAX puts it (the training box GPU, with
preallocation disabled by conftest so it coexists with a live learner).
Marked slow (~1 min): deselect with `-m "not slow"` for the quick suite.
"""

import jax
import numpy as np
from conftest import open_zero_init_paths
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def model_and_inputs():
    from rl.environment.utils import get_ex_player_step
    from rl.model.config import get_player_model_config
    from rl.model.heads import HeadParams
    from rl.model.player_model import get_player_model

    network = get_player_model(get_player_model_config(generation=9, train=True))
    actor_input, actor_output = jax.device_put(
        jax.tree.map(lambda x: x[:, 0], get_ex_player_step())
    )
    params = network.init(jax.random.key(0), actor_input, actor_output, HeadParams())
    return network, params, actor_input, actor_output


def test_init_produces_finite_params(model_and_inputs):
    _, params, _, _ = model_and_inputs
    leaves = jax.tree.leaves(params)
    assert leaves
    for leaf in leaves:
        assert np.isfinite(np.asarray(leaf, dtype=np.float32)).all()


def test_forward_outputs_finite_and_shaped(model_and_inputs):
    network, params, actor_input, actor_output = model_and_inputs
    from rl.model.heads import HeadParams

    out = network.apply(params, actor_input, actor_output, HeadParams())
    T = actor_input.env.done.shape[0]

    log_probs = np.asarray(out.value_head.log_probs, dtype=np.float32)
    assert log_probs.shape[0] == T
    assert np.isfinite(log_probs).all()
    # log_probs is a categorical distribution over the value support.
    np.testing.assert_allclose(np.exp(log_probs).sum(-1), 1.0, atol=1e-3)

    pi_lp = np.asarray(out.action_head.log_prob, dtype=np.float32)
    assert np.isfinite(pi_lp).all()


def test_q_head_forward_shapes(model_and_inputs):
    """The structural two-rung hierarchical Q readout (docs/
    q-critic-plan.md): owned adapter + shared MacroMicroHead params in the
    tree, (T, A, n_bins) logits per rung, rung conditioning alive."""

    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = model_and_inputs
    for subtree in ("q_adapter", "q_macro_micro", "q_cond_proj", "q_cond_norm"):
        assert subtree in params["params"]
    assert "macro_micro_head" in params["params"]

    out = network.apply(params, actor_input, actor_output, HeadParams())
    T = actor_input.env.done.shape[0]
    A = int(np.prod(actor_input.env.action_mask.shape[-2:]))
    q_adv = np.asarray(out.q_adv, dtype=np.float32)
    assert q_adv.shape == (T, A)
    assert np.isfinite(q_adv).all()
    private_q_adv = np.asarray(out.private_q_adv, dtype=np.float32)
    assert private_q_adv.shape == (T, A)
    assert np.isfinite(private_q_adv).all()
    # The rungs share every head param but not their conditioning input.
    # At init both are identically ZERO (e00a388's flat-at-init contract
    # zero-inits every Q output path), so comparing them here would be
    # vacuous — open the zero paths first, then a difference proves the
    # conditioning is actually WIRED rather than merely untrained.
    opened = open_zero_init_paths(params, ("q_adapter", "q_macro_micro"))
    out_open = network.apply(opened, actor_input, actor_output, HeadParams())
    assert not np.array_equal(
        np.asarray(out_open.q_adv, dtype=np.float32),
        np.asarray(out_open.private_q_adv, dtype=np.float32),
    )
    # Full-support log_policy is present in train mode — the Retrace
    # target's expectation bootstrap depends on it.
    assert np.asarray(out.action_head.log_policy).shape[-1] == A


def test_forward_is_deterministic(model_and_inputs):
    network, params, actor_input, actor_output = model_and_inputs
    from rl.model.heads import HeadParams

    a = network.apply(params, actor_input, actor_output, HeadParams())
    b = network.apply(params, actor_input, actor_output, HeadParams())
    np.testing.assert_array_equal(
        np.asarray(a.value_head.log_probs, dtype=np.float32),
        np.asarray(b.value_head.log_probs, dtype=np.float32),
    )
