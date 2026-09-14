"""The EMA and the old-policy snapshot follow the executable references on
accepted optimiser updates."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rl.online.artifact import Porygon2PlayerTrainState
from rl.online.training.train_step import apply_player_gradients


def make_state(step: int = 0) -> Porygon2PlayerTrainState:
    parameters = {"weight": jnp.array([1.0, -1.0])}
    return Porygon2PlayerTrainState.create(
        apply_fn=lambda *args: None,
        init_fn=lambda key: parameters,
        params=parameters,
        reg_params={"weight": jnp.array([4.0, 5.0])},
        old_policy_params={"weight": jnp.array([8.0, 9.0])},
        tx=optax.adam(0.01),
        step_count=jnp.array(step, dtype=jnp.int32),
        frame_count=jnp.array(7, dtype=jnp.int32),
    )


update = jax.jit(
    functools.partial(
        apply_player_gradients, reference_rate=0.25, old_policy_snap_steps=3
    ),
    donate_argnums=(0,),
)


def test_reference_tracks_pre_update_parameters_and_survives_donation() -> None:
    state = make_state()
    gradients = {"weight": jnp.array([0.4, -0.2])}
    expected_reference = np.asarray(state.reg_params["weight"]).copy()
    for accepted_step in range(1, 5):
        previous_parameters = np.asarray(state.params["weight"]).copy()
        previous_reference = expected_reference.copy()
        state, finite, rate = update(state, gradients, jnp.array(0.5), jnp.array(2))
        assert bool(finite)
        assert int(state.step_count) == accepted_step
        assert int(state.frame_count) == 7 + 2 * accepted_step
        assert float(rate) == 0.25
        expected_reference = 0.75 * previous_reference + 0.25 * previous_parameters
        np.testing.assert_allclose(
            state.reg_params["weight"], expected_reference, rtol=1e-6
        )
        post_update_reference = 0.75 * previous_reference + 0.25 * np.asarray(
            state.params["weight"]
        )
        assert np.max(np.abs(expected_reference - post_update_reference)) > 1e-3


@pytest.mark.parametrize("bad_gradient", [False, True])
def test_rejected_update_preserves_reference_optimiser_and_counters(
    bad_gradient: bool,
) -> None:
    state = make_state(step=2)
    before = jax.tree.map(lambda value: np.asarray(value).copy(), state)
    if bad_gradient:
        gradients = {"weight": jnp.array([jnp.nan, 0.2])}
        loss = jnp.array(0.5)
    else:
        gradients = {"weight": jnp.array([0.4, 0.2])}
        loss = jnp.array(jnp.inf)
    state, finite, rate = update(state, gradients, loss, jnp.array(2))
    assert not bool(finite)
    assert float(rate) == 0.0
    for actual, expected in zip(jax.tree.leaves(state), jax.tree.leaves(before)):
        np.testing.assert_array_equal(actual, expected)
    state, finite, rate = update(
        state, {"weight": jnp.array([0.4, 0.2])}, jnp.array(0.5), jnp.array(2)
    )
    assert bool(finite)
    assert float(rate) == 0.25
    assert int(state.step_count) == 3
    np.testing.assert_allclose(
        state.reg_params["weight"],
        0.75 * before.reg_params["weight"] + 0.25 * before.params["weight"],
        rtol=1e-6,
    )


def test_resumed_reference_continues_its_existing_average() -> None:
    state = make_state(step=10_000)
    reference = np.asarray(state.reg_params["weight"]).copy()
    parameters = np.asarray(state.params["weight"]).copy()
    state, _, _ = update(state, {"weight": jnp.ones(2)}, jnp.array(0.5), jnp.array(2))
    np.testing.assert_allclose(
        state.reg_params["weight"], 0.75 * reference + 0.25 * parameters, rtol=1e-6
    )
    assert not np.array_equal(state.reg_params["weight"], state.params["weight"])


@pytest.mark.parametrize("reference_rate", [0.0, 1.0])
def test_freeze_and_copy_endpoints_survive_repeated_donation(reference_rate) -> None:
    endpoint_update = jax.jit(
        functools.partial(
            apply_player_gradients,
            reference_rate=reference_rate,
            old_policy_snap_steps=3,
        ),
        donate_argnums=(0,),
    )
    state = make_state()
    original_reference = np.asarray(state.reg_params["weight"]).copy()
    for _ in range(3):
        previous_parameters = np.asarray(state.params["weight"]).copy()
        state, finite, rate = endpoint_update(
            state, {"weight": jnp.ones(2)}, jnp.array(0.5), jnp.array(2)
        )
        assert bool(finite)
        assert float(rate) == reference_rate
        if reference_rate == 0.0:
            np.testing.assert_array_equal(
                state.reg_params["weight"], original_reference
            )
        else:
            np.testing.assert_array_equal(
                state.reg_params["weight"], previous_parameters
            )
            assert not np.array_equal(
                state.reg_params["weight"], state.params["weight"]
            )


@pytest.mark.parametrize("reference_rate", [-0.01, 1.01, float("nan")])
def test_reference_rate_must_be_a_probability(reference_rate) -> None:
    with pytest.raises(ValueError, match="reference_rate must be between zero and one"):
        apply_player_gradients(
            make_state(),
            {"weight": jnp.ones(2)},
            jnp.array(0.5),
            jnp.array(2),
            reference_rate,
            3,
        )


def test_reference_retention_counts_every_accepted_update() -> None:
    reference_rate = 3.75e-5
    replay_update = jax.jit(
        functools.partial(
            apply_player_gradients,
            reference_rate=reference_rate,
            old_policy_snap_steps=3,
        ),
        donate_argnums=(0,),
    )
    state = make_state()
    original_reference = np.asarray(state.reg_params["weight"]).copy()
    parameters = np.asarray(state.params["weight"]).copy()
    for _ in range(8):
        state, finite, rate = replay_update(
            state, {"weight": jnp.zeros(2)}, jnp.array(0.0), jnp.array(2)
        )
        assert bool(finite)
        np.testing.assert_allclose(rate, reference_rate, rtol=1e-6)
    retention = (1.0 - reference_rate) ** 8
    expected = retention * original_reference + (1.0 - retention) * parameters
    np.testing.assert_allclose(state.reg_params["weight"], expected, atol=2e-6)
    assert np.max(np.abs(expected - original_reference)) > 1e-3


def test_old_policy_copies_post_update_parameters_every_snap_period() -> None:
    state = make_state()
    gradients = {"weight": jnp.array([0.4, -0.2])}
    old_policy = np.asarray(state.old_policy_params["weight"]).copy()
    for accepted_step in range(1, 8):
        previous_parameters = np.asarray(state.params["weight"]).copy()
        state, finite, _ = update(state, gradients, jnp.array(0.5), jnp.array(2))
        assert bool(finite)
        if accepted_step % 3 == 0:
            # POST-update parameters, exactly: the pre-update ones are a
            # positive control that the copy is not off by one step.
            old_policy = np.asarray(state.params["weight"]).copy()
            assert np.max(np.abs(old_policy - previous_parameters)) > 1e-3
        np.testing.assert_array_equal(state.old_policy_params["weight"], old_policy)
    assert int(state.step_count) == 7
    # Between snaps the live parameters keep moving away from the copy.
    assert np.max(np.abs(np.asarray(state.params["weight"]) - old_policy)) > 1e-3


def test_old_policy_snapshot_is_a_separate_buffer_under_donation() -> None:
    state = make_state(step=2)
    state, _, _ = update(
        state, {"weight": jnp.array([0.4, -0.2])}, jnp.array(0.5), jnp.array(2)
    )
    assert int(state.step_count) == 3
    np.testing.assert_array_equal(
        state.old_policy_params["weight"], state.params["weight"]
    )
    assert (
        state.old_policy_params["weight"].unsafe_buffer_pointer()
        != state.params["weight"].unsafe_buffer_pointer()
    )
    copied = np.asarray(state.old_policy_params["weight"]).copy()
    state, _, _ = update(
        state, {"weight": jnp.array([0.4, -0.2])}, jnp.array(0.5), jnp.array(2)
    )
    np.testing.assert_array_equal(state.old_policy_params["weight"], copied)
    assert not np.array_equal(state.old_policy_params["weight"], state.params["weight"])


def test_rejected_update_on_a_snap_step_keeps_the_old_snapshot() -> None:
    state = make_state(step=2)
    before = np.asarray(state.old_policy_params["weight"]).copy()
    state, finite, _ = update(
        state, {"weight": jnp.array([jnp.nan, 0.2])}, jnp.array(0.5), jnp.array(2)
    )
    assert not bool(finite)
    assert int(state.step_count) == 2
    np.testing.assert_array_equal(state.old_policy_params["weight"], before)


def test_snap_every_update_makes_the_old_policy_the_post_update_parameters() -> None:
    every_step = jax.jit(
        functools.partial(
            apply_player_gradients, reference_rate=0.25, old_policy_snap_steps=1
        ),
        donate_argnums=(0,),
    )
    state = make_state()
    for _ in range(3):
        state, _, _ = every_step(
            state, {"weight": jnp.array([0.4, -0.2])}, jnp.array(0.5), jnp.array(2)
        )
        np.testing.assert_array_equal(
            state.old_policy_params["weight"], state.params["weight"]
        )


def test_snap_period_must_be_positive() -> None:
    with pytest.raises(ValueError, match="old_policy_snap_steps"):
        apply_player_gradients(
            make_state(), {"weight": jnp.ones(2)}, jnp.array(0.5), jnp.array(2), 0.25, 0
        )
