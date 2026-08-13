"""Real player-model init + forward on the bundled example trajectory.

Marked gpu: runs wherever JAX puts it (the training box GPU, with
preallocation disabled by conftest so it coexists with a live learner).
Deselect with `-m "not gpu"` for a fast CPU-only run.
"""

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def model_and_inputs():
    from rl.environment.utils import get_ex_player_step
    from rl.model.config import get_player_model_config
    from rl.model.player_model import get_player_model
    from rl.model.heads import HeadParams

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


def test_forward_is_deterministic(model_and_inputs):
    network, params, actor_input, actor_output = model_and_inputs
    from rl.model.heads import HeadParams

    a = network.apply(params, actor_input, actor_output, HeadParams())
    b = network.apply(params, actor_input, actor_output, HeadParams())
    np.testing.assert_array_equal(
        np.asarray(a.value_head.log_probs, dtype=np.float32),
        np.asarray(b.value_head.log_probs, dtype=np.float32),
    )
