"""Shared test setup.

Env vars must be set before jax/wandb are imported anywhere: tests run on
the training box, so JAX must not preallocate the GPU out from under a
live learner (see no-agent-testing memory) and wandb must never try to
sync.
"""

import logging
import os
from collections.abc import Callable

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
# The learner's env sets JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=all (kernel +
# autotune caches alongside the executable cache). Under that flag every new
# executable rewrites the whole kernel cache file. Tests compile hundreds of
# small programs, so the executable cache alone is the right setting here; the
# learner keeps its own env. Explicit override (not setdefault) on purpose.
os.environ["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] = "none"

import flax.linen as nn
import pytest

from rl.environment.interfaces import PlayerActorInput, PlayerActorOutput

# Re-exported here because the model tests import it from conftest; the
# definition moved to rl.model.utils (2026-08-27) so the offline
# separation probe opens the SAME paths the tests do.
from rl.model.utils import open_zero_init_paths  # noqa: F401


def session_player_model_config(train: bool = True, dtype=None):
    """The session model's config, written once: the fixture builds from it,
    and so does any test that builds a second network to compare against
    the fixture's params (a differing head set misaligns the output trees),
    including the ACTOR-side network (train=False), which must carry the
    same trunk flags as the learner it shares params with.
    The PBRS channel's potential head (2026-09-11) is on, as it is whenever
    the channel runs, so the slot-invariance and gradient-reach tests read it;
    the normalised residual is on so the dtype and train-step tests cover it.
    """
    from rl.model.config import get_player_model_config

    if dtype is None:
        config = get_player_model_config(generation=9, train=train)
    else:
        config = get_player_model_config(generation=9, train=train, dtype=dtype)
    config.potential_head.enabled = True
    config.encoder.trunk.normalised_residual = True
    # The history recurrence under test; both forms exist during the
    # 2026-09-18 ablation. PORYGON_TEST_HISTORY_RECURRENCE=stacked runs the
    # session model on the other form.
    config.encoder.history_recurrence = os.environ.get(
        "PORYGON_TEST_HISTORY_RECURRENCE", "loop"
    )
    return config


@pytest.fixture(scope="session")
def real_model_and_trajectory() -> (
    tuple[nn.Module, dict, PlayerActorInput, PlayerActorOutput]
):
    """Full-size player model initialised once per test session on the
    bundled real example trajectory — model init + first compile dominate
    the slow suite's runtime, so every slow test shares this one."""
    import jax

    from rl.environment.utils import get_ex_player_step
    from rl.model.heads import HeadParams
    from rl.model.player_model import get_player_model

    network = get_player_model(session_player_model_config())
    actor_input, actor_output = jax.tree.map(lambda x: x[:, 0], get_ex_player_step())
    # Jitted init (2026-08-24): eager init dispatches the forward op by op
    # and compiles each nn.scan separately, and that cost is paid again
    # inside create_train_state (also jitted now).
    params = jax.jit(network.init)(
        jax.random.key(0), actor_input, actor_output, HeadParams()
    )
    return network, params, actor_input, actor_output


@pytest.fixture(scope="session")
def real_model_apply(
    real_model_and_trajectory: tuple[
        nn.Module, dict, PlayerActorInput, PlayerActorOutput
    ],
) -> Callable:
    """jax.jit(network.apply) for the session model: one compile, then
    milliseconds per call. Eager apply re-traces the whole module and
    dispatches op by op (the scans recompile per call) -- ~a minute each."""
    import jax

    network = real_model_and_trajectory[0]
    return jax.jit(network.apply)
