"""Step-2 NeuRD warm-up ramp (docs/critic-weakness-analysis.md): a pure
function of the traced step_count, off at warmup 0, never re-fires on a
resumed lineage."""

import jax.numpy as jnp

from rl.online.training.loss import warmup_scale


def test_warmup_scale_ramps_then_saturates():
    assert float(warmup_scale(jnp.int32(0), 100)) == 0.0
    assert abs(float(warmup_scale(jnp.int32(50), 100)) - 0.5) < 1e-6
    assert float(warmup_scale(jnp.int32(100), 100)) == 1.0
    assert float(warmup_scale(jnp.int32(10_000), 100)) == 1.0  # resumed lineage


def test_warmup_scale_disabled_is_one():
    assert float(warmup_scale(jnp.int32(0), 0)) == 1.0
    assert float(warmup_scale(jnp.int32(0), -1)) == 1.0
