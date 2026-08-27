"""The entropy-floor dual controller (2026-08-28): per-axis log-space
temperature stepped by loss.entropy_floor_step. Sign, bounds, and the
empty-batch freeze — each with the control proving the test could fail."""

import jax.numpy as jnp
import numpy as np

from rl.online.training.loss import entropy_floor_step

ALPHA_MIN = 0.005
ALPHA_MAX = 0.5


def step(log_alpha, entropy_value, rows=1.0, target=0.5, alpha_lr=1e-3):
    return float(
        entropy_floor_step(
            jnp.asarray(log_alpha, dtype=jnp.float32),
            target=target,
            entropy_value=jnp.asarray(entropy_value, dtype=jnp.float32),
            rows=jnp.asarray(rows, dtype=jnp.float32),
            alpha_lr=alpha_lr,
            alpha_min=ALPHA_MIN,
            alpha_max=ALPHA_MAX,
        )
    )


class TestDualAscentSign:
    def test_below_target_raises_alpha(self):
        base = float(np.log(0.05))
        got = step(base, entropy_value=0.2, target=0.5)
        assert got == np.float32(base + 1e-3 * 0.3)

    def test_above_target_lowers_alpha(self):
        base = float(np.log(0.05))
        got = step(base, entropy_value=0.8, target=0.5)
        assert got == np.float32(base - 1e-3 * 0.3)

    def test_at_target_is_a_fixed_point(self):
        base = float(np.log(0.05))
        assert step(base, entropy_value=0.5, target=0.5) == np.float32(base)


class TestBounds:
    def test_clamps_at_max(self):
        got = step(float(np.log(ALPHA_MAX)), entropy_value=0.0, target=0.5)
        assert got == np.float32(np.log(ALPHA_MAX))
        # Control: one notch below the ceiling, the same deficit moves it.
        near = float(np.log(ALPHA_MAX)) - 1e-2
        assert step(near, entropy_value=0.0, target=0.5) > near

    def test_clamps_at_min(self):
        got = step(float(np.log(ALPHA_MIN)), entropy_value=1.0, target=0.5)
        assert got == np.float32(np.log(ALPHA_MIN))
        near = float(np.log(ALPHA_MIN)) + 1e-2
        assert step(near, entropy_value=1.0, target=0.5) < near


class TestEmptyBatchFreeze:
    def test_zero_rows_freezes(self):
        # average() reads 0.0 on an empty mask — without the freeze that
        # registers as a full entropy deficit and inflates alpha.
        base = float(np.log(0.05))
        assert step(base, entropy_value=0.0, rows=0.0, target=0.5) == np.float32(base)
        # Control: the identical reading WITH rows moves it.
        assert step(base, entropy_value=0.0, rows=1.0, target=0.5) > base


def test_train_state_default_matches_ent_coef_init():
    """The state's default leaves equal log(player_ent_coef)'s default, so a
    state built off the config path scales the entropy term identically."""
    from rl.online.artifact import Porygon2PlayerTrainState
    from rl.online.config import get_learner_config

    coef = get_learner_config().player_ent_coef
    default = Porygon2PlayerTrainState.__dataclass_fields__[
        "log_ent_alpha_macro"
    ].default_factory()
    np.testing.assert_allclose(float(default), np.log(coef), rtol=1e-6)
