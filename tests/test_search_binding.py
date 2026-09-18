"""The searching actor (cfg.search.enabled) on the real model: the bonus
is finite and bounded at init (the flow is the copy predictor, so every
leaf is the root's own value or terminal read), the value-blind arm reads
exactly 0, and opening the flow's projection changes what the search
adds (the control). The learner path carries no search leaf."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.environment.utils import get_ex_player_step
from rl.model.config import get_player_model_config
from rl.model.heads import HeadParams
from rl.model.player_model import actor_params_view, get_player_model
from rl.model.utils import open_zero_init_paths


def _search_config(value_blind: bool = False):
    cfg = get_player_model_config(generation=9, train=False, dtype=jnp.float32)
    cfg.encoder.with_public_cls = True
    cfg.world_model.enabled = True
    cfg.search.enabled = True
    cfg.search.num_samples = 2
    cfg.search.max_cells = 32
    cfg.search.max_events = 2
    cfg.search.value_blind = value_blind
    cfg.world_model.flow_steps = 1
    return cfg


@pytest.mark.slow
def test_search_is_a_no_op_at_init_and_moves_the_policy_once_opened() -> None:
    actor_input, actor_output = jax.tree.map(lambda x: x[:3, 0], get_ex_player_step())
    model = get_player_model(_search_config())
    with jax.default_matmul_precision("highest"):
        params = jax.jit(model.init)(
            jax.random.key(0), actor_input, actor_output, HeadParams()
        )
        view = actor_params_view(params)
        assert "world_model" in view["params"] and "public_value_head" in view["params"]
        # The action head at zero init is uniform; open it so the base policy
        # has structure the bonus could move.
        params = open_zero_init_paths(params, ["action_head"])
        apply = jax.jit(model.apply)
        out = apply(
            params,
            actor_input,
            actor_output,
            HeadParams(),
            rngs={"sampling": jax.random.key(1)},
        )
        kl = np.asarray(out.search_root_kl)
        # At init the flow is the copy predictor, so every leaf is the
        # root's own value or its terminal read; the decoder still reads
        # the declared token (END sampled at different rates per cell),
        # so the bonus is small but not exactly uniform.
        assert np.isfinite(kl).all() and kl.max() < 0.5
        assert not np.asarray(out.search_overflow).any()
        # Control: a live flow projection changes the leaves per cell.
        opened = open_zero_init_paths(params, ["world_model"], scale=0.5)
        moved = apply(
            opened,
            actor_input,
            actor_output,
            HeadParams(),
            rngs={"sampling": jax.random.key(1)},
        )
        moved_kl = np.asarray(moved.search_root_kl)
        assert np.isfinite(moved_kl).all() and not np.allclose(moved_kl, kl)
        # The value-blind arm: the same rollouts, no bonus, KL exactly 0.
        blind = get_player_model(_search_config(value_blind=True))
        blind_out = jax.jit(blind.apply)(
            opened,
            actor_input,
            actor_output,
            HeadParams(),
            rngs={"sampling": jax.random.key(1)},
        )
        np.testing.assert_allclose(np.asarray(blind_out.search_root_kl), 0.0, atol=1e-6)
    # The learner path never builds the search outputs.
    learner = get_player_model(get_player_model_config(generation=9, train=True))
    assert not learner.cfg.search.enabled
