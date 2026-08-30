"""Jitted train_step smoke: the whole learner update, end to end.

The forward through the flat readout, the v-trace and one-step targets, the
NashPG bracket (surrogate + entropy duals + magnet + the zero-avoiding KL),
the value loss and the gradients, on the bundled ex.bin trajectory
(randombattle config, so the builder branch self-skips). This is the ONLY
test that compiles the real train_step, so it is what catches a panel that
went stale or a shape that stopped matching.

Was tests/test_train_step_q.py until 2026-08-29, when the Q head it was
named for retired.

Runs on the GPU like the rest of the slow suite (it was CPU-pinned to sit
beside a live learner, but the slow suite already cannot: host-RAM
guard). ONE static config = one compile of the full forward + backward.

Marked slow: deselect with `-m "not slow"` for the quick suite."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_train_step_smoke():
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
        # NashPG policy update: the surrogate, its clip occupancy, the
        # differentiated entropy/magnet terms and the batch-advantage
        # statistics.
        "player_loss_pg",
        "player_ppo_clip_frac",
        "player_loss_entropy",
        "player_ref_kl",
        "player_loss_uniform_kl",
        "player_pg_adv_mean",
        "player_pg_adv_std",
        "player_reg_snapped",
        "player_loss_v_win",
        "player_loss_kl",
        # Per-level entropy observers.
        "player_entropy_macro",
        "player_entropy_micro_taken",
        # Modality-resolved staleness: de-averaged actor KL and the
        # off-policy attenuation audit (isr = pi_target/mu_actor).
        "player_learner_actor_forward_kl_switch",
        "player_learner_actor_forward_kl_move",
        "player_isr_switch_voluntary",
        "player_isr_switch_forced",
        "player_isr_move",
        "player_isr_below1_switch_voluntary",
        "player_isr_below1_move",
        # Realised behaviour frequency on the stay/switch axis.
        "player_taken_switch_frac",
        "player_taken_voluntary_switch_frac",
        # Policy mass by modality.
        "player_policy_prob_switch",
        "player_policy_prob_move",
        "player_policy_prob_ratio",
        # The flat readout's drift-from-init panels. These are the ONLY
        # forensics on the two-factor stall, so a rename that silently drops
        # them must fail here.
        "player_pointer_query_rms",
        "player_pointer_key_rms",
        "player_pointer_local_src_rms",
        "player_pointer_local_tgt_rms",
        "player_switch_head_rms",
        "player_other_head_rms",
        "player_trunk_attn_out_rms",
        "player_trunk_mlp_out_rms",
        "player_action_head_grad_norm",
        "player_trunk_grad_norm",
    ):
        assert key in logs, key
        assert np.isfinite(np.asarray(logs[key], dtype=np.float32)).all(), key

    # Step-1 panels: present (NaN allowed where this one-game batch has no
    # rows in a slice), support counts finite.
    for key in (
        "player_mv_bin0_gap_realised",
        "player_v_outcome_r2_all",
    ):
        assert key in logs, key
    for key in (
        "player_vol_switch_rows",
        "player_chunk_vol_switch_frac",
    ):
        assert np.isfinite(np.asarray(logs[key], dtype=np.float32)).all(), key

    # The gradient actually reaches both halves of the model.
    assert float(logs["player_action_head_grad_norm"]) > 0.0
    assert float(logs["player_trunk_grad_norm"]) > 0.0

    # At init every logit is exactly 0, so the policy is uniform over legal
    # cells and the reference it just snapped from is the same distribution.
    assert float(logs["player_ref_kl"]) == pytest.approx(0.0, abs=1e-5)
