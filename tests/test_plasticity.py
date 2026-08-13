"""PlasticityController state machine and shrink-and-perturb param update."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rl.online.artifact import Porygon2PlayerTrainState
from rl.online.plasticity import (
    PlasticityController,
    _shrink_coefficient,
    shrink_and_perturb_player_state,
)


def make_controller(**overrides) -> PlasticityController:
    kwargs = dict(
        enabled=True,
        overdue_trigger=3,
        recovery_winrate=0.7,
        cooldown_frames=1_000,
        defer_to_exploiter=False,
    )
    kwargs.update(overrides)
    return PlasticityController(**kwargs)


class TestController:
    def test_dominant_resets_overdue_streak(self):
        c = make_controller()
        c.on_player_added("overdue")
        c.on_player_added("overdue")
        c.on_player_added("dominant")
        assert c.consecutive_overdue == 0

    def test_fires_only_after_consecutive_overdue_trigger(self):
        c = make_controller(overdue_trigger=3)
        for _ in range(2):
            c.on_player_added("overdue")
        assert not c.should_perturb(current_frames=10_000)
        c.on_player_added("overdue")
        assert c.should_perturb(current_frames=10_000)

    def test_disabled_never_fires(self):
        c = make_controller(enabled=False, overdue_trigger=1)
        c.on_player_added("overdue")
        assert not c.should_perturb(current_frames=10_000)

    def test_defer_to_exploiter_suppresses_and_recommends(self):
        c = make_controller(defer_to_exploiter=True, overdue_trigger=1)
        c.on_player_added("overdue")
        assert not c.should_perturb(current_frames=10_000)
        assert c.exploiter_phase_recommended

    def test_acknowledge_exploiter_episode_resets_clock(self):
        c = make_controller(defer_to_exploiter=True, overdue_trigger=1)
        c.on_player_added("overdue")
        c.acknowledge_exploiter_episode()
        assert not c.exploiter_phase_recommended

    def test_recovery_blocks_further_perturbations_and_overdue_counting(self):
        c = make_controller(overdue_trigger=1)
        c.on_player_added("overdue")
        assert c.should_perturb(current_frames=0)
        c.on_perturbation(recovery_ref_step=100, current_frames=0)
        assert c.recovering
        # While recovering: no counting, no firing.
        c.on_player_added("overdue")
        assert c.consecutive_overdue == 0
        assert not c.should_perturb(current_frames=10**9)

    def test_recovery_needs_cooldown_and_winrate(self):
        c = make_controller(overdue_trigger=1, recovery_winrate=0.7, cooldown_frames=1_000)
        c.on_player_added("overdue")
        c.on_perturbation(recovery_ref_step=100, current_frames=0)
        # Winrate reached but still inside the cooldown window.
        c.check_recovery(winrate_vs_ref=0.9, current_frames=500)
        assert c.recovering
        # Cooled down but winrate short.
        c.check_recovery(winrate_vs_ref=0.5, current_frames=2_000)
        assert c.recovering
        c.check_recovery(winrate_vs_ref=0.75, current_frames=2_000)
        assert not c.recovering
        assert c.recovery_ref_step is None

    def test_can_fire_again_after_full_recovery(self):
        # Recovery itself requires the cooldown to have elapsed
        # (check_recovery gates on it), so a recovered controller with a
        # fresh overdue streak may fire immediately.
        c = make_controller(overdue_trigger=1, cooldown_frames=1_000)
        c.on_player_added("overdue")
        c.on_perturbation(recovery_ref_step=100, current_frames=0)
        c.check_recovery(winrate_vs_ref=1.0, current_frames=10_000)  # recovered
        c.on_player_added("overdue")
        assert c.should_perturb(current_frames=20_000)
        assert c.perturbation_count == 1

    def test_state_dict_roundtrip(self):
        c = make_controller()
        c.on_player_added("overdue")
        c.on_perturbation(recovery_ref_step=123, current_frames=456)
        c.check_recovery(winrate_vs_ref=0.4, current_frames=500)

        fresh = make_controller()
        fresh.load_state_dict(c.state_dict())
        assert fresh.state_dict() == c.state_dict()
        assert fresh.recovering
        assert fresh.recovery_ref_step == 123


class TestShrinkAndPerturb:
    def test_shrink_coefficient_only_touches_params_collection(self):
        path = [jax.tree_util.DictKey("batch_stats"), jax.tree_util.DictKey("encoder")]
        assert _shrink_coefficient(path, {}, default_shrink=0.5) == 1.0
        path = [jax.tree_util.DictKey("params"), jax.tree_util.DictKey("encoder")]
        assert _shrink_coefficient(path, {"encoder": 0.9}, default_shrink=0.5) == 0.9
        assert _shrink_coefficient(path, {}, default_shrink=0.5) == 0.5

    def test_interpolates_toward_fresh_init_and_keeps_target(self):
        params = {"params": {"encoder": jnp.full((3,), 2.0), "v_head": jnp.full((3,), 4.0)}}
        fresh = {"params": {"encoder": jnp.zeros(3), "v_head": jnp.zeros(3)}}
        state = Porygon2PlayerTrainState.create(
            apply_fn=lambda *a, **k: None,
            init_fn=lambda rng: fresh,
            params=params,
            target_params=params,
            tx=optax.sgd(1e-3),
        )
        new_state = shrink_and_perturb_player_state(
            state,
            jax.random.PRNGKey(0),
            default_shrink=0.5,
            module_shrink=(("v_head", 0.75),),
        )
        np.testing.assert_allclose(new_state.params["params"]["encoder"], 1.0)  # 0.5*2
        np.testing.assert_allclose(new_state.params["params"]["v_head"], 3.0)  # 0.75*4
        # EMA target is the self-distillation anchor: untouched.
        np.testing.assert_allclose(new_state.target_params["params"]["encoder"], 2.0)

    def test_optimizer_state_is_reset(self):
        params = {"params": {"encoder": jnp.ones(3)}}
        state = Porygon2PlayerTrainState.create(
            apply_fn=lambda *a, **k: None,
            init_fn=lambda rng: params,
            params=params,
            target_params=params,
            tx=optax.adam(1e-3),
        )
        state = state.apply_gradients(grads={"params": {"encoder": jnp.ones(3)}})
        mu = state.opt_state[0].mu["params"]["encoder"]
        assert np.abs(np.asarray(mu)).max() > 0
        new_state = shrink_and_perturb_player_state(state, jax.random.PRNGKey(0), 0.5)
        new_mu = new_state.opt_state[0].mu["params"]["encoder"]
        np.testing.assert_allclose(np.asarray(new_mu), 0.0)

    def test_preserves_param_dtype(self):
        params = {"params": {"encoder": jnp.ones(3, dtype=jnp.bfloat16)}}
        state = Porygon2PlayerTrainState.create(
            apply_fn=lambda *a, **k: None,
            init_fn=lambda rng: {"params": {"encoder": jnp.zeros(3, dtype=jnp.float32)}},
            params=params,
            target_params=params,
            tx=optax.sgd(1e-3),
        )
        new_state = shrink_and_perturb_player_state(state, jax.random.PRNGKey(0), 0.5)
        assert new_state.params["params"]["encoder"].dtype == jnp.bfloat16
