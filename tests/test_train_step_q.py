"""Eager (unjitted) train_step smoke with the two-rung Q head enabled —
the wiring test for docs/q-critic-plan.md: model Q_all/Q_private
readouts -> Retrace targets -> CE losses -> gradients, end to end on the
bundled ex.bin trajectory. CPU-pinned so it can run next to a live
learner (randombattle config, so the builder branch self-skips).

Marked slow (~3 min eager): deselect with `-m "not slow"` for the quick
suite."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_train_step_player_q_smoke():
    from rl.environment.interfaces import (
        Batch,
        CategoricalValueHeadOutput,
        PlayerActorOutput,
        PlayerAgentOutput,
        PlayerPolicyHeadOutput,
        PlayerTransition,
    )
    from rl.environment.utils import get_ex_player_step
    from rl.model.builder_model import get_builder_model
    from rl.model.config import get_builder_model_config, get_player_model_config
    from rl.model.player_model import get_player_model
    from rl.online.artifact import create_train_state
    from rl.online.config import Porygon2LearnerConfig
    from rl.online.learner import train_step

    with jax.default_device(jax.devices("cpu")[0]):
        config = Porygon2LearnerConfig(player_q_enabled=True)
        player_net = get_player_model(
            get_player_model_config(config.generation, train=True, q_head_enabled=True)
        )
        builder_net = get_builder_model(
            get_builder_model_config(config.generation, train=True)
        )
        player_state, builder_state = create_train_state(
            player_net, builder_net, jax.random.key(0), config
        )

        actor_input, actor_output = get_ex_player_step()
        env = actor_input.env  # (T, B=1, ...)
        T, B = env.done.shape
        mask_width = env.action_mask.shape[-1]
        action_index = jnp.asarray(actor_output.action_head.action_index)

        batch = Batch(
            player_transitions=PlayerTransition(
                env_output=env,
                agent_output=PlayerAgentOutput(
                    actor_output=PlayerActorOutput(
                        action_head=PlayerPolicyHeadOutput(
                            action_index=action_index,
                            log_prob=jnp.full((T, B), -1.0, dtype=jnp.float32),
                            src_index=action_index // mask_width,
                            tgt_index=action_index % mask_width,
                        ),
                        value_head=CategoricalValueHeadOutput(
                            expectation=jnp.zeros((T, B), dtype=jnp.float32)
                        ),
                    )
                ),
            ),
            player_history=actor_input.history,
            player_packed_history=actor_input.packed_history,
        )

        new_player_state, _, logs = train_step(
            player_state, builder_state, batch, config
        )

    assert int(new_player_state.step_count) == 1
    for key in (
        "player_loss_q",
        "player_loss_q_private",
        "player_q_r2",
        "player_q_private_r2",
        "player_q_ev_gap",
        "player_q_switch_move_gap",
        # Gap discriminators: per-context calibration + data coverage +
        # head-independent conditional Retrace means.
        "player_q_r2_move",
        "player_q_r2_switch_forced",
        "player_q_r2_switch_voluntary",
        "player_q_switch_target_frac",
        "player_q_voluntary_switch_target_frac",
        "player_q_target_voluntary_switch",
        "player_q_target_move",
        # The per-module grad-norm loop picks the new head up by itself;
        # nonzero norm proves the CE loss actually reaches q_head params.
        "player_q_head_gradient_norm",
        # Stage 2 improvement term (observer since 2026-08-19: coef
        # zeroed, diagnostics stay): the forward-KL loss plus its
        # readouts — p_q vs pi switch mass on real-choice states.
        "player_loss_q_improve",
        "player_q_improve_pq_switch_mass",
        "player_q_improve_pi_switch_mass",
        "player_q_improve_pq_entropy",
        # Stage 3 Q-boosting (docs/q-boosting-plan.md): loss-free 3a
        # diagnostics — blend candidate's scale/agreement vs the v-trace
        # channel, plus Thm 3.1's Var_a~pi[Q] precondition readout.
        # (Calibration r2 fresh/replay needs batch.reuse_count, absent
        # in this minimal batch; player_q_boost_mix is host-side only.)
        "player_q_boost_adv_mean",
        "player_q_boost_adv_std",
        "player_q_boost_adv_corr",
        "player_q_boost_adv_sign_agree",
        "player_q_action_var",
    ):
        assert key in logs, key
        assert np.isfinite(np.asarray(logs[key], dtype=np.float32)).all(), key
    assert float(logs["player_q_head_gradient_norm"]) > 0.0

    # Stage-3 blend path at full mix: the boosted advantage swaps in
    # wholesale (runtime scalar, same compiled fn) and everything stays
    # finite — the exploiter-side mix=0 case is the default-args run above.
    with jax.default_device(jax.devices("cpu")[0]):
        _, _, logs_boost = train_step(
            player_state, builder_state, batch, config, q_boost_mix=np.float32(1.0)
        )
    assert np.isfinite(np.asarray(logs_boost["player_loss_pg"], dtype=np.float32))
    assert np.isfinite(
        np.asarray(logs_boost["player_state_adv_mean"], dtype=np.float32)
    )

    # Explore-row contract (2026-08-17), tested at its extreme: an
    # all-explore batch trains EVERY player loss — the tempered rows carry
    # exact ISRs, so policy/value masks stay live — while the explore-only
    # signals (league cadence, plasticity, builder) mask them elsewhere.
    batch_explore = batch.replace(explore=np.ones((1, B), dtype=bool))
    with jax.default_device(jax.devices("cpu")[0]):
        _, _, logs_explore = train_step(
            player_state, builder_state, batch_explore, config
        )
    assert float(logs_explore["player_policy_mask_sum"]) > 0.0
    assert float(logs_explore["player_value_mask_sum"]) > 0.0
    assert np.isfinite(np.asarray(logs_explore["player_loss_pg"], dtype=np.float32))
    assert np.isfinite(
        np.asarray(logs_explore["player_loss_v_win"], dtype=np.float32)
    )
    assert float(logs_explore["player_q_explore_frac"]) == 1.0
    assert "player_q_r2_explore" in logs_explore
    assert "player_learner_actor_forward_kl_own" in logs_explore
    assert np.isfinite(np.asarray(logs_explore["player_loss_q"], dtype=np.float32))
    assert float(logs_explore["player_q_head_gradient_norm"]) > 0.0

    # Observer contract: with player_q_enabled=False nothing q-flavoured
    # may appear (and the loss must not reference undefined q terms).
    # Explicit False — the default flipped to enabled in 817132a.
    config_off = Porygon2LearnerConfig(player_q_enabled=False)
    player_net_off = get_player_model(
        get_player_model_config(config_off.generation, train=True)
    )
    with jax.default_device(jax.devices("cpu")[0]):
        player_state_off, builder_state_off = create_train_state(
            player_net_off, builder_net, jax.random.key(0), config_off
        )
        _, _, logs_off = train_step(
            player_state_off, builder_state_off, batch, config_off
        )
    assert "player_loss_q" not in logs_off
    assert "player_loss_q_private" not in logs_off
    assert "player_q_switch_move_gap" not in logs_off
    assert "player_loss_q_improve" not in logs_off
    assert "player_q_boost_adv_mean" not in logs_off
