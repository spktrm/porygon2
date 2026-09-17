"""Small loss-only checks against derivatives through actual shifted logits."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from rl.environment.data import MOVE_CELL_OFFSET, NUM_ACTION_CELLS, NUM_SWITCH_CELLS
from rl.online.training.action_telemetry import switch_loss_telemetry
from rl.online.training.loss import appo_policy_loss, uniform_kl_rows
from rl.online.training.targets import reference_kl
from rl.utils import average


def test_switch_direction_matches_full_logit_derivative() -> None:
    config = SimpleNamespace(
        player_pg_coef=0.8,
        player_ent_coef=0.01,
        player_mag_coef=0.2,
        player_uniform_kl_coef=0.05,
        player_ppo_clip=0.2,
        player_behaviour_ratio_clip=2.0,
    )
    legal = jnp.zeros((4, NUM_ACTION_CELLS), dtype=bool)
    legal = legal.at[:, [0, 1, MOVE_CELL_OFFSET, MOVE_CELL_OFFSET + 1]].set(True)
    legal = legal.at[2, MOVE_CELL_OFFSET:].set(False)  # Forced switch.
    switch_cells = jnp.arange(NUM_ACTION_CELLS) < NUM_SWITCH_CELLS
    move_cells = ~switch_cells
    taken = jnp.array([0, MOVE_CELL_OFFSET, 1, 0])
    taken_switch = taken < NUM_SWITCH_CELLS
    taken_move = ~taken_switch
    valid = jnp.array([True, True, True, False])
    choice = jnp.array([True, True, False, False])
    advantages = jnp.array([1.2, -0.4, 0.7, 1000.0])
    # Behaviour and old-policy log-probs of the taken action: row 0 sits
    # inside the PPO band, row 1 is pushed past it (its clip zeroes the
    # JVP there), row 2 has its mu/pi_old capped at 2.
    behaviour_log_prob = jnp.log(jnp.array([0.30, 0.10, 0.50, 0.25]))
    old_policy_log_prob = jnp.log(jnp.array([0.28, 0.40, 0.20, 0.25]))
    logits = jnp.broadcast_to(jnp.linspace(-1.5, 1.1, NUM_ACTION_CELLS), legal.shape)

    def distribution(shift: float | jax.Array, tangent=switch_cells) -> jax.Array:
        return jax.nn.log_softmax(
            jnp.where(legal, logits + shift * tangent, -1e9), axis=-1
        )

    reference = distribution(0.3)

    def terms(shift: float | jax.Array, tangent=switch_cells) -> jax.Array:
        log_policy = distribution(shift, tangent)
        taken_log_prob = jnp.take_along_axis(log_policy, taken[:, None], -1)[:, 0]
        return jnp.stack(
            [
                config.player_pg_coef
                * appo_policy_loss(
                    learner_log_prob=taken_log_prob,
                    behaviour_log_prob=behaviour_log_prob,
                    old_policy_log_prob=old_policy_log_prob,
                    advantages=advantages,
                    valid=valid,
                    clip_ppo=config.player_ppo_clip,
                    behaviour_ratio_clip=config.player_behaviour_ratio_clip,
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
                * config.player_uniform_kl_coef
                * average(uniform_kl_rows(log_policy, legal), valid),
            ]
        )

    log_policy = distribution(0.0)
    taken_log_prob = jnp.take_along_axis(log_policy, taken[:, None], -1)[:, 0]
    logs = jax.jit(
        lambda: switch_loss_telemetry(
            log_policy,
            reference,
            legal,
            switch_cells,
            taken_switch,
            move_cells,
            taken_move,
            valid,
            choice,
            taken_log_prob,
            behaviour_log_prob,
            old_policy_log_prob,
            advantages,
            config,
        )
    )()
    expected = jax.jit(jax.jacfwd(terms))(0.0)
    actual = jnp.stack(
        [
            logs[f"player_switch_logit_grad_{name}"]
            for name in ("pg", "entropy", "magnet", "uniform_kl")
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert logs["player_switch_logit_grad_pg_taken_switch"] < 0
    assert abs(float(logs["player_switch_logit_grad_actor_total"])) > 0.01
    np.testing.assert_allclose(
        logs["player_switch_logit_grad_actor_total"], expected.sum(), atol=1e-6
    )
    # The sharpen directions against the same full-logit derivative, the
    # logits scaled about their policy mean (all legal cells, then the legal
    # moves about their conditional mean).
    policy = jnp.where(legal, jnp.exp(log_policy), 0.0)
    legal_log_policy = jnp.where(legal, log_policy, 0.0)
    legal_moves = legal & move_cells
    move_mean = (policy * legal_moves * legal_log_policy).sum(-1) / jnp.maximum(
        (policy * legal_moves).sum(-1), 1e-8
    )
    tangents = {
        "sharpen": jnp.where(
            legal, log_policy - (policy * legal_log_policy).sum(-1)[:, None], 0.0
        ),
        "move_sharpen": jnp.where(legal_moves, log_policy - move_mean[:, None], 0.0),
    }
    for direction, tangent in tangents.items():
        expected = jax.jit(jax.jacfwd(lambda shift: terms(shift, tangent)))(0.0)
        actual = jnp.stack(
            [
                logs[f"player_{direction}_logit_grad_{name}"]
                for name in ("pg", "entropy", "magnet", "uniform_kl")
            ]
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
        # Positive control: the floor pulls towards uniform, so along a
        # sharpening direction its force is positive (descent flattens).
        assert logs[f"player_{direction}_logit_grad_uniform_kl"] > 0
    assert logs["player_choice_switch_count"] == 1
    assert logs["player_choice_stay_count"] == 1
    np.testing.assert_allclose(logs["player_choice_switch_adv_raw"], 1.2)
