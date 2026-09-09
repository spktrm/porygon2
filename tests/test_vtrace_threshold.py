"""Learner-side thresholding of the policy entering v-trace
(rl/online/training/targets.py thresholded_target_ratio): scope, the
discard, and the trace it cuts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.model.utils import legal_log_policy
from rl.online.config import Porygon2LearnerConfig
from rl.online.training.targets import (
    compute_player_targets,
    thresholded_target_ratio,
    trace_run_length,
)

CELLS = 6


def _rows():
    # Three rows, four legal cells each; the taken action's target
    # probability is .40 / .004 / .40 and mu(a) = .25 everywhere.
    legal = jnp.asarray([[True] * 4 + [False] * 2] * 3)
    logits = jnp.log(
        jnp.asarray(
            [
                [0.40, 0.30, 0.20, 0.10, 1.0, 1.0],
                [0.60, 0.004, 0.30, 0.096, 1.0, 1.0],
                [0.40, 0.30, 0.20, 0.10, 1.0, 1.0],
            ]
        )
    )
    target_log_policy = legal_log_policy(logits, legal)
    action_index = jnp.asarray([0, 1, 0])
    behaviour_log_prob = jnp.full((3,), jnp.log(0.25))
    return target_log_policy, behaviour_log_prob, action_index, legal


def test_threshold_zero_is_the_raw_ratio_bit_identical():
    target_log_policy, behaviour, taken, legal = _rows()
    ratio, ratio_raw, kept, removed = thresholded_target_ratio(
        target_log_policy, behaviour, taken, legal, 0.0
    )
    np.testing.assert_array_equal(np.asarray(ratio), np.asarray(ratio_raw))
    expected = np.exp(np.asarray(target_log_policy)[np.arange(3), np.asarray(taken)])
    np.testing.assert_allclose(np.asarray(ratio_raw), expected / 0.25, rtol=1e-6)
    assert np.all(np.asarray(kept)) and not np.any(np.asarray(removed))


def test_below_the_line_is_discarded_and_above_untouched():
    target_log_policy, behaviour, taken, legal = _rows()
    ratio, ratio_raw, kept, removed = thresholded_target_ratio(
        target_log_policy, behaviour, taken, legal, 0.005
    )
    ratio, ratio_raw, kept = (np.asarray(x) for x in (ratio, ratio_raw, kept))
    # Row 1's taken action (.004) is discarded: ratio 0, raw ratio intact.
    assert ratio[1] == 0.0 and not kept[1]
    np.testing.assert_allclose(ratio_raw[1], 0.004 / 0.25, rtol=1e-6)
    # Rows 0 and 2 (positive control) are untouched -- nothing removed
    # there, so no renormalisation either.
    np.testing.assert_array_equal(ratio[[0, 2]], ratio_raw[[0, 2]])
    assert kept[0] and kept[2]
    # Row 1's kept actions were renormalised over the .996 that remains.
    np.testing.assert_allclose(np.asarray(removed), [0.0, 0.25, 0.0])


def test_a_row_entirely_below_the_line_keeps_its_ratio():
    # The reference's degenerate guard: every legal cell under the line.
    legal = jnp.asarray([[True] * 4 + [False] * 2])
    target_log_policy = legal_log_policy(jnp.zeros((1, CELLS)), legal)  # .25 each
    ratio, ratio_raw, kept, _ = thresholded_target_ratio(
        target_log_policy, jnp.log(jnp.asarray([0.25])), jnp.asarray([2]), legal, 0.3
    )
    np.testing.assert_array_equal(np.asarray(ratio), np.asarray(ratio_raw))
    assert bool(kept[0])


def _min_batch(done, win_reward, action_mask, action_index):
    from rl.environment.interfaces import (
        Batch,
        PlayerActorOutput,
        PlayerAgentOutput,
        PlayerEnvOutput,
        PlayerPolicyHeadOutput,
        PlayerTransition,
    )

    return Batch(
        player_transitions=PlayerTransition(
            env_output=PlayerEnvOutput(
                done=done, win_reward=win_reward, action_mask=action_mask
            ),
            agent_output=PlayerAgentOutput(
                actor_output=PlayerActorOutput(
                    action_head=PlayerPolicyHeadOutput(action_index=action_index)
                )
            ),
        )
    )


def test_discarded_row_produces_no_target_and_cuts_the_trace():
    # A 6-row chunk, terminal win on the last row, V = 0, lambda 1, gamma 1:
    # every row's return is the outcome unless the trace is cut.
    T = 6
    done = jnp.zeros((T, 1), bool).at[T - 1, 0].set(True)
    # Reward 0 (the centre bin) everywhere but the terminal win.
    win_reward = jnp.zeros((T, 1, 3), jnp.float32).at[:, :, 1].set(1.0)
    win_reward = win_reward.at[T - 1, 0].set(jnp.asarray([0.0, 0.0, 1.0]))
    batch = _min_batch(
        done, win_reward, jnp.ones((T, 1, 4), bool), jnp.zeros((T, 1), jnp.int32)
    )
    value_log_probs = jnp.full((T, 1, 3), jnp.log(1.0 / 3.0), jnp.float32)
    config = Porygon2LearnerConfig(player_gamma=1.0, player_lambda=1.0)
    raw = jnp.ones((T, 1), jnp.float32)
    cut = raw.at[3, 0].set(0.0)  # the taken action at row 3 was discarded

    reference, reference_logs = compute_player_targets(
        batch, value_log_probs, raw, config, isr_raw=raw
    )
    targets, logs = compute_player_targets(batch, value_log_probs, cut, config, raw)
    adv_reference = np.asarray(reference.pg_advantages)[:, 0]
    adv = np.asarray(targets.pg_advantages)[:, 0]
    # The discarded row itself: advantage 0 (rho = 0) and a bootstrap-only
    # value target; the rows after it are unchanged.
    assert adv[3] == 0.0 and adv_reference[3] != 0.0
    np.testing.assert_array_equal(adv[4:], adv_reference[4:])
    # (V = 0 everywhere, so the bootstrap-only target is two_hot(0) -- the
    # centre bin -- where the reference row carried the win.)
    np.testing.assert_allclose(np.asarray(targets.win_returns)[3, 0], [0, 1, 0])
    np.testing.assert_allclose(np.asarray(reference.win_returns)[3, 0], [0, 0, 1])
    # And every row BEFORE it lost the outcome: the trace was cut there.
    assert np.all(adv_reference[:3] != 0.0) and np.all(adv[:3] == 0.0)
    # The realised trace length says so, the raw twin does not.
    assert float(logs["player_trace_len_mean"]) < float(
        logs["player_trace_len_mean_raw"]
    )
    np.testing.assert_allclose(
        float(logs["player_trace_len_mean_raw"]),
        float(reference_logs["player_trace_len_mean"]),
    )
    # Twins: the raw ESS is 1 either way, the thresholded one is not.
    np.testing.assert_allclose(float(logs["player_isr_ess_raw"]), 1.0, atol=1e-6)
    assert float(logs["player_isr_ess"]) < 1.0


def test_trace_run_length_counts_to_the_first_cut():
    continues = jnp.asarray([[True], [True], [False], [True], [True]])
    runs = np.asarray(trace_run_length(continues))[:, 0]
    np.testing.assert_array_equal(runs, [2.0, 1.0, 0.0, 2.0, 1.0])


@pytest.mark.gpu
@pytest.mark.slow
def test_scope_is_the_v_trace_ratio_and_nothing_else():
    """The learner ratio, the surrogate, the magnet, the entropy term, the
    hinge and the forward KL are bit-identical under thresholding; the
    v-trace ESS is not. Two static configs, two compiles (the minimum for
    a comparison)."""
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
    from rl.model.utils import open_zero_init_paths
    from rl.online.artifact import create_train_state
    from rl.online.training.train_step import TRAIN_STEP_JIT

    actor_input, actor_output = get_ex_player_step()
    env = actor_input.env
    T, B = env.done.shape
    batch = Batch(
        player_transitions=PlayerTransition(
            env_output=env,
            agent_output=PlayerAgentOutput(
                actor_output=PlayerActorOutput(
                    action_head=PlayerPolicyHeadOutput(
                        action_index=jnp.asarray(actor_output.action_head.action_index),
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
        game_outcome=jnp.ones((1, B), dtype=jnp.float32),
        game_length=jnp.full((1, B), T, dtype=jnp.int32),
        game_step_offset=jnp.zeros((1, B), dtype=jnp.int32),
    )
    pinned = (
        "player_learner_actor_ratio",
        "player_loss_pg",
        "player_ref_kl",
        "player_loss_entropy",
        "player_loss_support",
        "player_learner_actor_forward_kl",
        "player_isr_ess_raw",
    )
    readings = {}
    # .3 rather than .005: the fixture's freshly opened readout is far from
    # abandoning anything, and a threshold that bites is the positive
    # control.
    for threshold in (0.0, 0.3):
        config = Porygon2LearnerConfig(player_prune_threshold=threshold)
        player_net = get_player_model(
            get_player_model_config(config.generation, train=True)
        )
        builder_net = get_builder_model(
            get_builder_model_config(config.generation, train=True)
        )
        player_state, builder_state = create_train_state(
            player_net, builder_net, jax.random.key(0), config
        )
        player_state = player_state.replace(
            params=open_zero_init_paths(player_state.params, ["action_head"]),
            target_params=open_zero_init_paths(
                player_state.target_params, ["action_head"]
            ),
        )
        _, _, logs = TRAIN_STEP_JIT(player_state, builder_state, batch, config)
        readings[threshold] = {key: np.asarray(logs[key]) for key in pinned}
        readings[threshold]["ess"] = np.asarray(logs["player_isr_ess"])
        readings[threshold]["discard"] = np.asarray(logs["player_discard_taken_frac"])
    for key in pinned:
        np.testing.assert_array_equal(readings[0.0][key], readings[0.3][key])
    assert readings[0.0]["discard"] == 0.0 and readings[0.3]["discard"] > 0.0
    assert readings[0.0]["ess"] != readings[0.3]["ess"]
