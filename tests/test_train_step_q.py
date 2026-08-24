"""Jitted train_step smoke with the two-rung Q head enabled -- the wiring
test for docs/q-critic-plan.md: model Q_all/Q_private readouts -> TD(0)
labels -> Huber losses -> gradients, end to end on the bundled ex.bin
trajectory (randombattle config, so the builder branch self-skips).

Runs on the GPU like the rest of the slow suite (it was CPU-pinned to sit
beside a live learner, but the slow suite already cannot: host-RAM
guard). ONE static config = one compile of the full forward + backward.

Marked slow: deselect with `-m "not slow"` for the quick suite."""

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
    from rl.online.training.train_step import TRAIN_STEP_JIT

    # Warm-up off: at step_count 0 the Step-2 ramp would zero NeuRD and
    # the policy-gradient assertions below would read 0 by design.
    config = Porygon2LearnerConfig(player_neurd_warmup_steps=0)
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
        # Completed-game side fields (Step-1 telemetry): present so the
        # critic_outcome_telemetry branch is traced inside the jit.
        game_outcome=jnp.ones((1, B), dtype=jnp.float32),
        game_length=jnp.full((1, B), T, dtype=jnp.int32),
        game_step_offset=jnp.zeros((1, B), dtype=jnp.int32),
    )

    # The learner's compiled train_step (donates the states; nothing below
    # reads the pre-step ones). The eager function was ~25 min here.
    new_player_state, _, logs = TRAIN_STEP_JIT(
        player_state, builder_state, batch, config
    )

    assert int(new_player_state.step_count) == 1
    for key in (
        "player_loss_q",
        "player_loss_q_private",
        "player_q_r2",
        "player_q_private_r2",
        "player_q_saturation_frac",
        "player_q_mse",
        "player_ref_kl",
        "player_ref_penalty_switch",
        "player_ref_penalty_move",
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
        # critic-flagged switch-worthy states, not the taken action.
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
        # Within/between-modality split + Q-head learning readouts
        # (2026-08-24): the within route is the pointer micro grid
        # behind a zero-init gate, so at init within == 0 exactly.
        "player_q_action_var_within_modality",
        "player_q_action_var_between_modality",
        "player_q_private_action_var_within_modality",
        "player_q_micro_scale_move",
        "player_q_micro_scale_switch",
        "player_q_micro_kernel_rms",
        "player_q_macro_out_rms",
        "player_q_adapter_out_rms",
        "player_q_grad_norm_micro",
        "player_q_grad_norm_macro",
    ):
        assert key in logs, key
        assert np.isfinite(np.asarray(logs[key], dtype=np.float32)).all(), key
    # Step-1 panels: present (NaN allowed where this one-game batch has no
    # rows in a slice), support counts finite.
    for key in (
        "player_q_label_var_outcome_move",
        "player_q_label_var_onestep_move",
        "player_mv_bin0_gap_realised",
        "player_mv_pooled_gap_critic",
        "player_v_outcome_r2_all",
        "player_v_onestep_r2",
        "player_q_loss_share_move",
    ):
        assert key in logs, key
    for key in (
        "player_q_support_vol_switch_rows",
        "player_q_support_chunk_vol_switch_frac",
        "player_q_target_edge_frac",
    ):
        assert np.isfinite(np.asarray(logs[key], dtype=np.float32)).all(), key
    assert float(logs["player_q_macro_micro_gradient_norm"]) > 0.0

    # NeuRD term finite at the base coefficient; coef 0 is a scalar
    # multiply (finiteness at that end is hierarchical_neurd's unit test),
    # and a second static config would be a second full compile.
    assert np.isfinite(np.asarray(logs["player_loss_neurd"], dtype=np.float32))

