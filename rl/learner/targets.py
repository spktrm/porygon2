import numpy as np

from rl.environment.data import CAT_VF_SUPPORT
from rl.environment.interfaces import BuilderTargets, PlayerTargets, Trajectory


def segmented_cumsum(x: np.ndarray, discount: np.ndarray) -> np.ndarray:
    """Backward cumulative sum: result[t] = x[t] + discount[t] * result[t+1].

    Equivalent to jax_segmented_cumsum but runs in NumPy on CPU.
    The discount array encodes episode boundaries: setting discount[t]=0 at a
    terminal step prevents bootstrapping across episodes.
    """
    result = np.empty_like(x)
    result[-1] = x[-1]
    for t in range(x.shape[0] - 2, -1, -1):
        result[t] = x[t] + discount[t] * result[t + 1]
    return result


def compute_player_targets(
    traj: Trajectory,
    td_lambda: float,
    gae_lambda: float,
) -> PlayerTargets:
    """Compute TD(λ) returns and GAE advantages for the player trajectory.

    Called once when the trajectory is added to the replay buffer so that
    these targets do not need to be recomputed on every training step.
    """
    cat_vf_support = CAT_VF_SUPPORT

    player_valid = np.logical_not(traj.player_transitions.env_output.done)  # (T,)
    player_reward = traj.player_transitions.env_output.win_reward.astype(
        np.float32
    )  # (T, 3)
    player_value_probs = np.exp(
        traj.player_transitions.agent_output.actor_output.value_head.log_probs.astype(
            np.float32
        )
    )  # (T, 3)

    player_next_value_probs = np.concatenate(
        [player_value_probs[1:], player_value_probs[-1:]], axis=0
    )
    player_value_target = (
        player_reward + player_next_value_probs * player_valid[..., None]
    )
    player_value_delta = player_value_target - player_value_probs  # (T, 3)
    player_scalar_delta = player_value_delta @ cat_vf_support  # (T,)

    td_lambdas = (td_lambda * player_valid).astype(np.float32)  # (T,)
    gae_lambdas = (gae_lambda * player_valid).astype(np.float32)  # (T,)

    returns = (
        segmented_cumsum(player_value_delta, td_lambdas[..., None]) + player_value_probs
    )  # (T, 3)
    advantages = segmented_cumsum(player_scalar_delta, gae_lambdas)  # (T,)

    # --- ICM potential-based intrinsic advantage ---
    # Use the wm_head prediction error as a potential function Φ(s).
    # Potential-based shaping reward: F(t) = γ * Φ(s_{t+1}) - Φ(s_t)
    # At terminal steps the next-state potential is zero (player_valid == 0).
    potential = traj.player_transitions.agent_output.actor_output.wm_head.logits.astype(
        np.float32
    )  # (T,)
    # Per-trajectory normalisation so the intrinsic signal is scale-invariant.
    pot_std = potential.std()
    if pot_std > 1e-8:
        potential = potential / pot_std
    next_potential = np.concatenate(
        [potential[1:], np.zeros_like(potential[:1])], axis=0
    )
    icm_reward = next_potential * player_valid - potential  # (T,)
    intrinsic_advantage = segmented_cumsum(icm_reward, gae_lambdas)  # (T,)

    return PlayerTargets(
        returns=returns.astype(np.float32),
        advantages=advantages.astype(np.float32),
        intrinsic_advantage=intrinsic_advantage.astype(np.float32),
    )


def compute_builder_targets(
    traj: Trajectory,
    td_lambda: float,
    gae_lambda: float,
    entropy_normalising_constant: float,
) -> BuilderTargets:
    """Compute TD(λ) returns and GAE advantages for the builder trajectory.

    The builder reward is derived from the player's final win/loss/tie reward,
    so the full Trajectory (containing both builder and player data) is required.
    The entropy temperature, which changes over training, is intentionally *not*
    applied here; raw_ent_advantages must be scaled in train_step.
    """
    cat_vf_support = CAT_VF_SUPPORT
    builder_transitions = traj.builder_transitions

    builder_valid = np.logical_not(builder_transitions.env_output.done)  # (T_b,)
    T_b = builder_valid.shape[0]

    builder_value_probs = np.exp(
        builder_transitions.agent_output.actor_output.value_head.log_probs.astype(
            np.float32
        )
    )  # (T_b, 3)
    builder_log_prob = (
        builder_transitions.agent_output.actor_output.action_head.log_prob.astype(
            np.float32
        )
    )  # (T_b,)
    builder_ent_pred = builder_transitions.agent_output.actor_output.conditional_entropy_head.logits.astype(
        np.float32
    )  # (T_b,)

    # Place the final player reward at the first terminal position of the builder.
    final_reward = traj.player_transitions.env_output.win_reward[-1].astype(
        np.float32
    )  # (3,)

    num_valid_steps = int(builder_valid.sum())
    builder_reward = np.zeros((T_b, 3), dtype=np.float32)
    if num_valid_steps < T_b:
        builder_reward[num_valid_steps] = final_reward

    # Entropy delta: NLL + discounted future entropy - current entropy prediction.
    builder_ent_scaled = builder_ent_pred * entropy_normalising_constant  # (T_b,)
    next_builder_ent_scaled = (
        np.concatenate(
            [builder_ent_scaled[1:], np.zeros_like(builder_ent_scaled[:1])], axis=0
        )
        * builder_valid
    )
    builder_nll = -builder_log_prob  # (T_b,)
    builder_ent_delta = builder_nll + next_builder_ent_scaled - builder_ent_scaled

    # Value computation.
    builder_next_value_probs = np.concatenate(
        [builder_value_probs[1:], builder_value_probs[-1:]], axis=0
    )
    builder_value_target = (
        builder_reward + builder_next_value_probs * builder_valid[..., None]
    )
    builder_value_delta = builder_value_target - builder_value_probs  # (T_b, 3)

    td_lambdas = (td_lambda * builder_valid).astype(np.float32)  # (T_b,)
    gae_lambdas = (gae_lambda * builder_valid).astype(np.float32)  # (T_b,)

    returns = (
        segmented_cumsum(builder_value_delta, td_lambdas[..., None])
        + builder_value_probs
    )  # (T_b, 3)
    win_advantages = (
        segmented_cumsum(builder_value_delta, gae_lambdas[..., None]) @ cat_vf_support
    )  # (T_b,)
    ent_returns = (
        segmented_cumsum(builder_ent_delta, td_lambdas) + builder_ent_scaled
    ) / entropy_normalising_constant  # (T_b,)
    raw_ent_advantages = segmented_cumsum(builder_ent_delta, gae_lambdas)  # (T_b,)

    return BuilderTargets(
        returns=returns.astype(np.float32),
        win_advantages=win_advantages.astype(np.float32),
        raw_ent_advantages=raw_ent_advantages.astype(np.float32),
        ent_returns=ent_returns.astype(np.float32),
    )
