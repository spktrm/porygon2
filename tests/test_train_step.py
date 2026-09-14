"""Jitted train_step smoke: the whole learner update, end to end.

The forward through the flat readout, the v-trace targets against the
old-policy snapshot, the APPO clipped surrogate with entropy and an EMAgnet
reference, the value loss and the gradients, on the bundled ex.bin trajectory
(randombattle config, so the builder branch self-skips). This is the ONLY
test that compiles the real train_step, so it is what catches a panel that
went stale or a shape that stopped matching.

Runs on the GPU like the rest of the slow suite (it was CPU-pinned to sit
beside a live learner, but the slow suite already cannot: host-RAM
guard). ONE static config = one compile of the full forward + backward.

Marked slow: deselect with `-m "not slow"` for the quick suite."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _ex_batch(actor_input, actor_output):
    """The bundled ex.bin trajectory as a (T, B=1) learner batch."""
    from rl.environment.interfaces import (
        Batch,
        CategoricalValueHeadOutput,
        PlayerActorOutput,
        PlayerAgentOutput,
        PlayerPolicyHeadOutput,
        PlayerTransition,
    )

    env = actor_input.env  # (T, B=1, ...)
    T, B = env.done.shape
    action_index = jnp.asarray(actor_output.action_head.action_index)

    batch = Batch(
        player_transitions=PlayerTransition(
            env_output=env,
            agent_output=PlayerAgentOutput(
                actor_output=PlayerActorOutput(
                    action_head=PlayerPolicyHeadOutput(
                        action_index=action_index,
                        log_prob=jnp.full((T, B), -1.0, dtype=jnp.float32),
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
        reuse_count=jnp.zeros((1, B), dtype=jnp.int32),
    )
    return batch


def test_train_step_smoke() -> None:
    from rl.environment.utils import get_ex_player_step
    from rl.model.builder_model import get_builder_model
    from rl.model.config import get_builder_model_config
    from rl.model.player_model import get_player_model
    from rl.online.artifact import create_train_state, player_model_config_for
    from rl.online.config import Porygon2LearnerConfig
    from rl.online.training.train_step import TRAIN_STEP_JIT

    config = Porygon2LearnerConfig()
    player_net = get_player_model(player_model_config_for(config))
    builder_net = get_builder_model(
        get_builder_model_config(config.generation, train=True)
    )
    player_state, builder_state = create_train_state(
        player_net, builder_net, jax.random.key(0), config
    )

    actor_input, actor_output = get_ex_player_step()
    batch = _ex_batch(actor_input, actor_output)

    # The learner's compiled train_step (donates the states; nothing below
    # reads the pre-step ones).
    new_player_state, _, logs = TRAIN_STEP_JIT(
        player_state, builder_state, batch, config
    )

    assert int(new_player_state.step_count) == 1
    assert "player_fresh_voluntary_switch_frac" in logs
    assert not any("pair_value" in key or "pair_population" in key for key in logs)
    for key in (
        "player_loss_pg",
        "player_loss_entropy",
        "player_ref_kl",
        "player_isr_ess",
        "player_trace_len_mean",
        "player_support_min_prob",
        "player_applied_delta_rms_pointer_key",
        "player_pg_adv_mean",
        "player_pg_adv_std",
        "player_reg_ema_rate",
        # The APPO surrogate's readouts and the target snapshot's clock.
        "player_ppo_clip_frac",
        "player_surrogate_ratio_mean",
        "player_behaviour_old_ratio_mean",
        "player_behaviour_old_ratio_clip_frac",
        "player_old_policy_age",
        "player_loss_v_win",
        "player_loss_kl",
        # Per-level entropy observers.
        "player_entropy_macro",
        "player_entropy_micro_taken",
        # Modality-resolved staleness: de-averaged actor KL and the
        # off-policy attenuation audit (isr = pi_learner/mu_actor).
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
        "player_fresh_voluntary_switch_count",
        "player_fresh_move_or_switch_count",
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
        # Trunk over-smoothing (plan step c-live).
        "player_trunk_row_cosine",
        "player_trunk_row_participation",
        "player_state_kernel_rms_hp",
        "player_state_kernel_rms_status",
        "player_state_kernel_rms_boosts",
        "player_state_kernel_rms_other",
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
    # cells and the initial reference has the same distribution.
    assert float(logs["player_ref_kl"]) == pytest.approx(0.0, abs=1e-5)
    # pi_old is that same init copy, so pi_live/pi_old is 1 on every row and
    # the surrogate ratio is exactly the capped mu/pi_old factor folded
    # back: min(1, clip * pi_old / mu), never above one. The snapshot is
    # one accepted step old after this first update.
    assert 0.0 < float(logs["player_surrogate_ratio_mean"]) <= 1.0 + 1e-5
    assert int(logs["player_old_policy_age"]) == 1


def test_train_step_runs_the_potential_channel() -> None:
    """One static config per train_step test: eta > 0 builds the potential
    head and runs its channel (2026-09-11). ex.bin predates the service's
    potential, so the slot is filled synthetically -- all zeros would hand
    the head a zero label and a zero gradient, and pass vacuously."""
    from rl.environment.protos.features_pb2 import InfoFeature
    from rl.environment.utils import get_ex_player_step
    from rl.model.builder_model import get_builder_model
    from rl.model.config import get_builder_model_config
    from rl.model.player_model import get_player_model
    from rl.online.artifact import create_train_state, player_model_config_for
    from rl.online.config import Porygon2LearnerConfig
    from rl.online.training.train_step import TRAIN_STEP_JIT

    config = Porygon2LearnerConfig().replace(player_potential_strength=0.05)
    player_net = get_player_model(player_model_config_for(config))
    builder_net = get_builder_model(
        get_builder_model_config(config.generation, train=True)
    )
    player_state, builder_state = create_train_state(
        player_net, builder_net, jax.random.key(0), config
    )
    head_before = [
        np.asarray(leaf)
        for leaf in jax.tree.leaves(player_state.params["params"]["potential_head"])
    ]

    actor_input, actor_output = get_ex_player_step()
    info = np.asarray(actor_input.env.info).copy()
    info[..., InfoFeature.INFO_FEATURE__STATE_POTENTIAL] = np.random.default_rng(
        0
    ).integers(-12000, 12000, size=info.shape[:-1])
    actor_input = actor_input.replace(
        env=actor_input.env.replace(info=jnp.asarray(info))
    )
    batch = _ex_batch(actor_input, actor_output)

    new_player_state, _, logs = TRAIN_STEP_JIT(
        player_state, builder_state, batch, config
    )

    for key in (
        "player_loss_potential",
        "player_potential_head_r2",
        "player_potential_head_fit_r2",
        "player_potential_adv_share",
        "player_potential_win_adv_corr",
        "player_potential_adv_switch",
        "player_potential_adv_move",
        "player_potential_head_grad_share",
        "player_potential_mean",
        "player_potential_std",
        "player_potential_switch_delta_mean",
    ):
        assert key in logs, key
        assert np.isfinite(np.asarray(logs[key])).all(), key
    # The zero-init head reads W = 0, so the channel carries the full
    # near-term-potential force on this first step.
    assert float(logs["player_potential_adv_share"]) > 0.0
    head_after = jax.tree.leaves(new_player_state.params["params"]["potential_head"])
    assert any(
        not np.array_equal(before, np.asarray(after))
        for before, after in zip(head_before, head_after, strict=True)
    )
