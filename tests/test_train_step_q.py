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


def live_scalars(config, neurd_coef=None):
    """The RuntimeScalars the live Learner would pass. Both fields are
    required, so a test that wants a term off has to say so explicitly —
    the old None defaults silently zeroed the policy gradient.

    Imports inside, like every other import here: this module is
    collected by the quick suite too, and model/jax imports are the
    expensive part."""
    from rl.online.config import RuntimeScalars

    return RuntimeScalars(
        magnet_coef=np.float32(config.player_magnet_kl_coef),
        neurd_coef=np.float32(
            config.player_neurd_coef if neurd_coef is None else neurd_coef
        ),
    )


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
        config = Porygon2LearnerConfig()
        player_net = get_player_model(
            get_player_model_config(config.generation, train=True)
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
            player_state, builder_state, batch, config, live_scalars(config)
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
        # Modality-resolved staleness: de-averaged actor KL and the
        # off-policy attenuation audit (isr = pi_target/mu_actor).
        "player_learner_actor_forward_kl_switch",
        "player_learner_actor_forward_kl_move",
        "player_isr_switch_voluntary",
        "player_isr_switch_forced",
        "player_isr_move",
        "player_isr_below1_switch_voluntary",
        "player_isr_below1_move",
        "player_q_target_voluntary_switch",
        "player_q_target_move",
        # Pivotal-state decision panel: tail statistics conditioned on the
        # critic-flagged switch-worthy states, not the taken action (the
        # explore-row return split needs own_rows, absent in this minimal
        # batch — same reason player_q_explore_frac isn't listed).
        "player_q_pivotal_frac",
        "player_q_pivotal_pi_switch_mass",
        "player_q_pivotal_taken_switch_frac",
        "player_q_pivotal_ret_switch",
        "player_q_pivotal_ret_stay",
        # The per-module grad-norm loop picks the new modules up by
        # itself; nonzero norm proves the CE loss actually reaches the
        # Q family's MacroMicroHead params.
        "player_q_macro_micro_gradient_norm",
        # All-action NeuRD, the policy gradient: loss value plus the
        # signed switch-modality push readout.
        "player_loss_neurd",
        "player_neurd_switch_push",
        "player_neurd_adv_std",
        # pi-prefactor decomposition: |d loss_neurd / d logit| on
        # switch vs non-switch legal cells of real-choice rows, split
        # into its pi and |adv| factors (grad ~ prob x absadv).
        "player_neurd_grad_switch",
        "player_neurd_grad_move",
        "player_neurd_grad_ratio",
        "player_neurd_prob_switch",
        "player_neurd_prob_move",
        "player_neurd_prob_ratio",
        "player_neurd_absadv_switch",
        "player_neurd_absadv_move",
        "player_neurd_absadv_ratio",
        # Loss-free critic-quality diagnostics. (Calibration r2
        # fresh/replay needs batch.reuse_count, absent in this minimal
        # batch.)
        "player_q_action_var",
        "player_q_action_var_p90",
    ):
        assert key in logs, key
        assert np.isfinite(np.asarray(logs[key], dtype=np.float32)).all(), key
    assert float(logs["player_q_macro_micro_gradient_norm"]) > 0.0

    # NeuRD at full coefficient (runtime scalar, same compiled fn): the
    # policy's only gradient path stays finite, and zeroing it is a
    # runtime scalar change through the SAME compiled fn.
    with jax.default_device(jax.devices("cpu")[0]):
        _, _, logs_neurd = train_step(
            player_state,
            builder_state,
            batch,
            config,
            scalars=live_scalars(config, neurd_coef=1.0),
        )
    assert np.isfinite(np.asarray(logs_neurd["player_loss_neurd"], dtype=np.float32))
    with jax.default_device(jax.devices("cpu")[0]):
        _, _, logs_off = train_step(
            player_state,
            builder_state,
            batch,
            config,
            scalars=live_scalars(config, neurd_coef=0.0),
        )
    assert np.isfinite(np.asarray(logs_off["player_loss_neurd"], dtype=np.float32))

    # Explore-row contract (2026-08-17), tested at its extreme: an
    # all-explore batch trains EVERY player loss — the tempered rows carry
    # exact ISRs, so policy/value masks stay live — while the explore-only
    # signals (league cadence, builder) mask them elsewhere.
    batch_explore = batch.replace(explore=np.ones((1, B), dtype=bool))
    with jax.default_device(jax.devices("cpu")[0]):
        _, _, logs_explore = train_step(
            player_state, builder_state, batch_explore, config, live_scalars(config)
        )
    assert float(logs_explore["player_policy_mask_sum"]) > 0.0
    assert float(logs_explore["player_value_mask_sum"]) > 0.0
    assert np.isfinite(
        np.asarray(logs_explore["player_loss_v_win"], dtype=np.float32)
    )
    assert float(logs_explore["player_q_explore_frac"]) == 1.0
    assert "player_q_r2_explore" in logs_explore
    assert "player_learner_actor_forward_kl_own" in logs_explore
    assert np.isfinite(np.asarray(logs_explore["player_loss_q"], dtype=np.float32))
    assert float(logs_explore["player_q_macro_micro_gradient_norm"]) > 0.0
