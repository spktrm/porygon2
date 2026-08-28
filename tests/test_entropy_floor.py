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


class TestDualTemperaturePersistence:
    """The temperatures are TrainState leaves, so nothing about them is
    reachable from `params` — a checkpoint that omits them resumes the
    controller at config init and throws away its whole dual-ascent
    trajectory. These pin the save/restore contract."""

    @staticmethod
    def _state(log_macro: float, log_micro: float):
        import optax

        from rl.online.artifact import Porygon2PlayerTrainState

        return Porygon2PlayerTrainState.create(
            apply_fn=lambda *a, **k: None,
            init_fn=lambda *a, **k: None,
            params={"w": jnp.zeros(2)},
            target_params={"w": jnp.zeros(2)},
            reg_params={"w": jnp.zeros(2)},
            tx=optax.sgd(0.1),
            step_count=jnp.array(1234, dtype=jnp.int32),
            frame_count=jnp.array(5678, dtype=jnp.int32),
            log_ent_alpha_macro=jnp.array(log_macro, dtype=jnp.float32),
            log_ent_alpha_micro=jnp.array(log_micro, dtype=jnp.float32),
        )

    def test_components_carry_the_temperatures(self):
        from rl.online.artifact import player_scalar_components

        state = self._state(np.log(0.075), np.log(0.005))
        scalars = player_scalar_components(state)
        np.testing.assert_allclose(
            np.exp(float(scalars["log_ent_alpha_macro"])), 0.075, rtol=1e-6
        )
        np.testing.assert_allclose(
            np.exp(float(scalars["log_ent_alpha_micro"])), 0.005, rtol=1e-6
        )

    def test_components_cover_every_non_parameter_leaf(self):
        """The control that makes this suite non-vacuous: a controller leaf
        added to the TrainState without a line in `player_scalar_components`
        fails HERE, instead of being silently dropped from every checkpoint
        the way the temperatures were."""
        from rl.online.artifact import (
            Porygon2PlayerTrainState,
            player_scalar_components,
        )

        # Written to their own component files, not the scalar block.
        stored_separately = {"params", "target_params", "reg_params", "opt_state"}
        # optax's internal update counter, redundant with step_count and
        # never read here.
        not_persisted = {"step"}
        leaves = {
            name
            for name, field in Porygon2PlayerTrainState.__dataclass_fields__.items()
            if field.metadata.get("pytree_node", True)
        }
        expected = leaves - stored_separately - not_persisted
        assert set(player_scalar_components(self._state(0.0, 0.0))) == expected

    def test_checkpoint_roundtrip_preserves_the_temperatures(self, tmp_path):
        from rl import checkpoint
        from rl.online.artifact import apply_player_scalars, player_scalar_components

        saved = self._state(np.log(0.075), np.log(0.005))
        ckpt_dir = str(tmp_path / "ckpt_00067000")
        checkpoint.save_train_state(
            ckpt_dir,
            learner_config={"generation": 9},
            player_state_components=dict(scalars=player_scalar_components(saved)),
            builder_state_components={},
            league_bytes=b"",
        )
        loaded = checkpoint.load_component(ckpt_dir, "player", "scalars")

        # Restore onto a state at the config-init default, so an unwired
        # restore reads 0.05 on both axes and the assert fails.
        fresh = self._state(np.log(0.05), np.log(0.05))
        resumed = apply_player_scalars(fresh, loaded)
        np.testing.assert_allclose(
            np.exp(float(resumed.log_ent_alpha_macro)), 0.075, rtol=1e-6
        )
        np.testing.assert_allclose(
            np.exp(float(resumed.log_ent_alpha_micro)), 0.005, rtol=1e-6
        )
        assert int(resumed.step_count) == 1234
        assert int(resumed.frame_count) == 5678

    def test_legacy_scalars_keep_the_live_temperatures(self):
        """A checkpoint predating this fix has no temperature keys; it must
        resume at config init rather than raise."""
        from rl.online.artifact import apply_player_scalars

        fresh = self._state(np.log(0.05), np.log(0.05))
        resumed = apply_player_scalars(fresh, {"step_count": 99, "frame_count": 100})
        np.testing.assert_allclose(
            np.exp(float(resumed.log_ent_alpha_macro)), 0.05, rtol=1e-6
        )
        np.testing.assert_allclose(
            np.exp(float(resumed.log_ent_alpha_micro)), 0.05, rtol=1e-6
        )
        assert int(resumed.step_count) == 99
