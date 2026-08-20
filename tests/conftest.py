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

import jax.numpy as jnp
import pytest


def open_zero_init_paths(params, subtrees, seed=0, scale=0.02):
    """Perturb the all-zero leaves under `subtrees` with small noise.

    The flat-at-init contract (e00a388) zero-inits every micro/macro/adapter
    OUTPUT path, so on freshly-initialised params the Q rungs emit identical
    all-zero logits and any "the rungs must differ" assertion is vacuously
    false — not because the conditioning is dead, but because nothing has
    trained yet. Opening the zero paths lets the rung conditioning propagate,
    so the tests check the property they mean: that the pathway is WIRED, not
    that it currently carries signal.
    """
    import numpy as np
    from flax.traverse_util import flatten_dict, unflatten_dict

    rng = np.random.default_rng(seed)
    flat = dict(flatten_dict(params))
    opened = 0
    for k, v in flat.items():
        if not any(s in k for s in subtrees):
            continue
        arr = np.asarray(v, dtype=np.float32)
        if arr.size and not arr.any():
            flat[k] = jnp.asarray(rng.normal(0.0, scale, arr.shape), dtype=v.dtype)
            opened += 1
    assert opened, f"no zero-init leaves found under {subtrees} — contract changed?"
    return unflatten_dict(flat)


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

    network = get_player_model(get_player_model_config(generation=9, train=True))
    actor_input, actor_output = jax.tree.map(lambda x: x[:, 0], get_ex_player_step())
    params = network.init(jax.random.key(0), actor_input, actor_output, HeadParams())
    return network, params, actor_input, actor_output
