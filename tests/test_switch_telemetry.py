"""Small loss-only checks against derivatives through actual shifted logits."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.environment.data import MOVE_CELL_OFFSET, NUM_ACTION_CELLS, NUM_SWITCH_CELLS
from rl.online.training.action_telemetry import switch_loss_telemetry
from rl.online.training.loss import policy_gradient_loss, support_hinge_loss
from rl.online.training.targets import reference_kl
from rl.utils import average


@pytest.mark.parametrize("objective", ["spo", "ppo"])
def test_switch_direction_matches_full_logit_derivative(objective: str) -> None:
    config = SimpleNamespace(
        player_pg_objective=objective,
        player_ppo_clip=0.2,
        player_pg_coef=0.8,
        player_ent_coef=0.01,
        player_mag_coef=0.2,
        player_support_tau=0.01,
        player_support_temperature=0.1,
        player_support_hinge_coef=0.0025,
    )
    legal = jnp.zeros((4, NUM_ACTION_CELLS), dtype=bool)
    legal = legal.at[:, [0, 1, MOVE_CELL_OFFSET, MOVE_CELL_OFFSET + 1]].set(True)
    legal = legal.at[2, MOVE_CELL_OFFSET:].set(False)  # Forced switch.
    switch_cells = jnp.arange(NUM_ACTION_CELLS) < NUM_SWITCH_CELLS
    taken = jnp.array([0, MOVE_CELL_OFFSET, 1, 0])
    taken_switch = taken < NUM_SWITCH_CELLS
    valid = jnp.array([True, True, True, False])
    choice = jnp.array([True, True, False, False])
    advantages = jnp.array([1.2, -0.4, 0.7, 1000.0])
    logits = jnp.broadcast_to(jnp.linspace(-1.5, 1.1, NUM_ACTION_CELLS), legal.shape)

    def distribution(shift: float | jax.Array) -> jax.Array:
        return jax.nn.log_softmax(
            jnp.where(legal, logits + shift * switch_cells, -1e9), axis=-1
        )

    reference = distribution(0.3)
    behaviour_taken = jnp.take_along_axis(distribution(-0.1), taken[:, None], -1)[:, 0]

    def terms(shift: float | jax.Array) -> jax.Array:
        log_policy = distribution(shift)
        log_ratio = jnp.take_along_axis(log_policy, taken[:, None], -1)[:, 0]
        log_ratio -= behaviour_taken
        ratio = jnp.exp(log_ratio)
        return jnp.stack(
            [
                config.player_pg_coef
                * policy_gradient_loss(
                    policy_ratios=ratio,
                    advantages=advantages,
                    valid=valid,
                    threshold=config.player_ppo_clip,
                    objective=objective,
                ),
                config.player_pg_coef
                * config.player_ent_coef
                * average(
                    jnp.where(legal, jnp.exp(log_policy) * log_policy, 0).sum(-1), valid
                ),
                config.player_pg_coef
                * config.player_mag_coef
                * average(reference_kl(log_policy, reference, legal), valid),
                config.player_pg_coef
                * config.player_support_hinge_coef
                * average(
                    support_hinge_loss(
                        log_policy,
                        legal,
                        config.player_support_tau,
                        temperature=config.player_support_temperature,
                    )[0],
                    valid,
                ),
            ]
        )

    log_policy = distribution(0.0)
    log_ratio = (
        jnp.take_along_axis(log_policy, taken[:, None], -1)[:, 0] - behaviour_taken
    )
    logs = jax.jit(
        lambda: switch_loss_telemetry(
            log_policy,
            reference,
            legal,
            switch_cells,
            taken_switch,
            valid,
            choice,
            jnp.exp(log_ratio),
            advantages,
            advantages * 2,
            config,
        )
    )()
    expected = jax.jit(jax.jacfwd(terms))(0.0)
    actual = jnp.stack(
        [
            logs[f"player_switch_logit_grad_{name}"]
            for name in ("pg", "entropy", "magnet", "support")
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    # Every legal cell here sits ~25x above tau: the hard hinge would be
    # exactly silent, the smooth one (T = .1) is silent to ~1e-14 of its
    # activation (the active cases are tests/test_support_hinge.py's).
    assert abs(float(logs["player_switch_logit_grad_support"])) < 1e-9
    assert logs["player_switch_logit_grad_pg_taken_switch"] < 0
    assert abs(float(logs["player_switch_logit_grad_actor_total"])) > 0.01
    np.testing.assert_allclose(
        logs["player_switch_logit_grad_actor_total"], expected.sum(), atol=1e-6
    )
    assert logs["player_choice_switch_count"] == 1
    assert logs["player_choice_stay_count"] == 1
    np.testing.assert_allclose(logs["player_choice_switch_adv_raw"], 2.4)
