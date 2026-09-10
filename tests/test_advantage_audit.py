"""Paired diagnostics tested with hand-computable, disagreeing critics."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.environment.interfaces import Batch, PlayerEnvOutput, PlayerTransition
from rl.online.training.action_telemetry import paired_advantage_audit
from rl.online.training.targets import two_hot


@pytest.mark.parametrize("discount", [1.0, 0.9])
def test_paired_advantages_masks_distance_and_outer_weight(
    discount: float,
) -> None:
    # Four unique chunks, one replay duplicate, and one unknown outcome.
    done = jnp.zeros((4, 6), dtype=bool).at[3, 0].set(True)
    legal = jnp.ones((4, 6, 2), dtype=bool).at[1, 0, 1].set(False)
    rewards = jnp.zeros((4, 6, 3)).at[..., 1].set(1.0)
    rewards = rewards.at[3, 0].set(jnp.array([0.0, 0.0, 1.0]))
    batch = Batch(
        player_transitions=PlayerTransition(
            env_output=PlayerEnvOutput(done=done, action_mask=legal, win_reward=rewards)
        ),
        game_length=jnp.array([[4, 20, 40, 80, 4, 4]]),
        game_step_offset=jnp.array([[0, 10, 20, 30, 0, 0]], dtype=jnp.int32),
        game_outcome=jnp.array([[1.0, 1.0, 1.0, 1.0, 1.0, jnp.nan]]),
        reuse_count=jnp.array([[0, 0, 0, 0, 1, 0]]),
    )
    public_values = jnp.broadcast_to(jnp.array([[-0.2], [0.2], [0.0], [0.0]]), (4, 6))
    privileged_values = jnp.broadcast_to(
        jnp.array([[0.4], [-0.4], [0.0], [0.0]]), (4, 6)
    )
    support = jnp.array([-1.0, 0.0, 1.0])
    public = jnp.log(two_hot(public_values, support))
    privileged = jnp.log(two_hot(privileged_values, support))
    axis = SimpleNamespace(
        has_both=legal.all(-1),
        taken_switch=jnp.zeros((4, 6), dtype=bool).at[0].set(True).at[1, 0].set(True),
    )
    config = SimpleNamespace(player_gamma=discount, player_lambda=0.0)
    isr = jnp.full((4, 6), 0.5)
    logs = jax.jit(
        lambda: paired_advantage_audit(
            batch, public, privileged, isr, isr, config, axis
        )
    )()
    assert logs["player_adv_audit_switch_all_count"] == 4
    assert logs["player_adv_audit_stay_all_count"] == 7
    for horizon in ["1_5", "6_15", "16_40", "41_plus"]:
        assert logs[f"player_adv_audit_switch_{horizon}_count"] == 1
    prefix = "player_adv_audit_switch_1_5"
    expected_return = discount**3
    expected_public = 0.5 * (discount * 0.2 + 0.2)
    expected_privileged = 0.5 * (discount * -0.4 - 0.4)
    np.testing.assert_allclose(
        logs[f"{prefix}_public_td_sum"], expected_public, atol=1e-6
    )
    np.testing.assert_allclose(
        logs[f"{prefix}_privileged_td_sum"], expected_privileged, atol=1e-6
    )
    np.testing.assert_allclose(
        logs[f"{prefix}_privileged_mc_sum"], expected_return - 0.4, atol=1e-6
    )
    np.testing.assert_allclose(
        logs[f"{prefix}_privileged_rho_mc_sum"],
        0.5 * (expected_return - 0.4),
        atol=1e-6,
    )
    assert logs[f"{prefix}_priv_negative_public_positive_sum"] == 1
    assert logs[f"{prefix}_privileged_td_negative_mc_positive_sum"] == 1
    assert all(np.isfinite(value) for value in logs.values())
    # The raw ratio builds c (the learner's v-trace since 2ac1e25): with a
    # live trace, zeroing it changes the audited advantage.
    traced = SimpleNamespace(player_gamma=discount, player_lambda=1.0)
    audit = jax.jit(
        lambda raw: paired_advantage_audit(
            batch, public, privileged, isr, raw, traced, axis
        )
    )
    assert (
        audit(isr)[f"{prefix}_public_td_sum"]
        != audit(jnp.zeros_like(isr))[f"{prefix}_public_td_sum"]
    )


def test_missing_metadata_produces_no_audit() -> None:
    assert paired_advantage_audit(Batch(), None, None, None, None, None, None) == {}
