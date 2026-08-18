"""Shared test setup.

Env vars must be set before jax/wandb are imported anywhere: tests run on
the training box, so JAX must not preallocate the GPU out from under a
live learner (see no-agent-testing memory) and wandb must never try to
sync.
"""

import logging
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TQDM_DISABLE", "1")

# The persistent-compile-cache MISS/not-writing warnings and the
# explain-cache-misses tracer notes are per-compile spam — thousands of
# lines per fresh-architecture test run, useless in test output.
logging.getLogger("jax._src.compiler").setLevel(logging.ERROR)
logging.getLogger("jax._src.dispatch").setLevel(logging.ERROR)
os.environ.setdefault("JAX_EXPLAIN_CACHE_MISSES", "false")

import pytest


@pytest.fixture(scope="session")
def real_model_and_trajectory():
    """Full-size player model initialised once per test session on the
    bundled real example trajectory — model init + first compile dominate
    the slow suite's runtime, so every slow test shares this one."""
    import jax

    from rl.environment.utils import get_ex_player_step
    from rl.model.config import get_player_model_config
    from rl.model.heads import HeadParams
    from rl.model.player_model import get_player_model

    # q_head_enabled matches the learner default (player_q_enabled=True)
    # so the shared fixture exercises the two-rung Q outputs too.
    network = get_player_model(
        get_player_model_config(generation=9, train=True, q_head_enabled=True)
    )
    actor_input, actor_output = jax.tree.map(lambda x: x[:, 0], get_ex_player_step())
    params = network.init(jax.random.key(0), actor_input, actor_output, HeadParams())
    return network, params, actor_input, actor_output
