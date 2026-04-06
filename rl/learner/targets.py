import jax
import jax.numpy as jnp

from rl.environment.data import CAT_VF_SUPPORT
from rl.environment.interfaces import (
    BuilderActorOutput,
    BuilderTargets,
    PlayerActorOutput,
    PlayerTargets,
    Trajectory,
)


def segmented_cumsum(x: jax.Array, discount: jax.Array) -> jax.Array:
    """
    Backward cumulative sum using parallel associative scan.
    Best for very long sequence lengths on GPU/TPU.
    """

    def combine(acc, new_elem):
        # acc: values from the "future" (because we reverse the array)
        # new_elem: the current step's values
        v_acc, g_acc = acc
        v_new, g_new = new_elem

        # Combine the values and the cumulative discounts
        v_combined = v_new + g_new * v_acc
        g_combined = g_acc * g_new
        return v_combined, g_combined

    # 1. Reverse the arrays to compute backward
    rev_x = jnp.flip(x, axis=0)
    rev_discount = jnp.flip(discount, axis=0)

    # 2. Perform the parallel associative scan
    # associative_scan returns a tuple matching the inputs; we only need the values (idx 0)
    rev_result, _ = jax.lax.associative_scan(combine, (rev_x, rev_discount))

    # 3. Flip back to original temporal order
    return jnp.flip(rev_result, axis=0)


def compute_player_targets(
    traj: Trajectory,
    target_pred: PlayerActorOutput,
    importance_sampling_ratios: jax.Array,
    td_lambda: float,
    gae_lambda: float,
    entropy_normalising_constant: float,
) -> PlayerTargets:
    cat_vf_support = jnp.asarray(CAT_VF_SUPPORT, dtype=jnp.float32)
    player_valid = jnp.logical_not(traj.player_transitions.env_output.done)  # (T, B)

    # --- V-Trace IMPALA Variables ---
    rho_t = jnp.minimum(1.0, importance_sampling_ratios)
    c_t = jnp.minimum(1.0, importance_sampling_ratios)

    # --- 1. Extract and Scale Base Values & Rewards ---
    # Value
    player_reward = traj.player_transitions.env_output.win_reward  # (T, B, 3)
    player_value_probs = jnp.exp(target_pred.value_head.log_probs)
    n_bins = player_value_probs.shape[-1]

    # Entropy
    num_valid_actions = traj.player_transitions.env_output.action_mask.sum((-2, -1))
    kl_reward = target_pred.action_head.entropy - jnp.log(num_valid_actions + 1e-8)
    player_ent_scaled = (
        target_pred.conditional_entropy_head.logits * entropy_normalising_constant
    )

    # Potential
    state_pot = traj.player_transitions.env_output.state_potential
    player_state_potential = jnp.concatenate(
        [jnp.zeros_like(state_pot[:1]), state_pot], axis=0
    )
    potential_reward = player_state_potential[1:] - player_state_potential[:-1]
    player_potential_value = target_pred.potential_value_head.logits

    # --- 2. Concatenate Rewards, Values, and Next Values ---
    # Shape: (T, B, n_bins + 2)
    combined_rewards = jnp.concatenate(
        [player_reward, kl_reward[..., None], potential_reward[..., None]], axis=-1
    )

    combined_values = jnp.concatenate(
        [
            player_value_probs,
            player_ent_scaled[..., None],
            player_potential_value[..., None],
        ],
        axis=-1,
    )

    # Construct the offset for next values, maintaining the zero-padding logic for entropy
    last_values = jnp.concatenate(
        [
            player_value_probs[-1:],
            player_ent_scaled[-1:][..., None],
            player_potential_value[-1:][..., None],
        ],
        axis=-1,
    )

    combined_next_values = (
        jnp.concatenate([combined_values[1:], last_values], axis=0)
        * player_valid[..., None]
    )

    # --- 3. Compute Combined Deltas in one batched operation ---
    combined_deltas = (
        rho_t[..., None] * (combined_rewards + combined_next_values - combined_values)
    ).astype(jnp.float32)

    # --- 4. Compute Advantages Deltas ---
    # Transform the n_bins categorical value deltas back into a scalar for advantages
    player_scalar_delta = (combined_deltas[..., :n_bins] @ cat_vf_support)[..., None]

    combined_adv_deltas = jnp.concatenate(
        [
            player_scalar_delta,  # (T, B, 1)
            combined_deltas[..., n_bins:],  # (T, B, 2) - Entropy and Potential deltas
        ],
        axis=-1,
    )

    # --- 5. Discounts & Batched Segmented Cumsum ---
    returns_discounts = (c_t * td_lambda * player_valid).astype(jnp.float32)
    policy_discounts = (c_t * gae_lambda * player_valid).astype(jnp.float32)

    combined_ret_cumsum = segmented_cumsum(
        combined_deltas, returns_discounts[..., None]
    )
    combined_adv_cumsum = segmented_cumsum(
        combined_adv_deltas, policy_discounts[..., None]
    )

    # --- 6. Split Outputs ---
    win_returns = combined_ret_cumsum[..., :n_bins] + combined_values[..., :n_bins]

    ent_returns = (
        combined_ret_cumsum[..., n_bins] + combined_values[..., n_bins]
    ) / entropy_normalising_constant

    potential_returns = (
        combined_ret_cumsum[..., n_bins + 1] + combined_values[..., n_bins + 1]
    )

    return PlayerTargets(
        win_returns=win_returns,
        win_advantages=combined_adv_cumsum[..., 0],
        ent_returns=ent_returns,
        ent_advantages=combined_adv_cumsum[..., 1],
        potential_returns=potential_returns,
        potential_advantages=combined_adv_cumsum[..., 2],
    )


def compute_builder_targets(
    traj: Trajectory,
    target_pred: BuilderActorOutput,
    importance_sampling_ratios: jax.Array,
    td_lambda: float,
    gae_lambda: float,
    entropy_normalising_constant: float,
) -> BuilderTargets:
    cat_vf_support = jnp.asarray(CAT_VF_SUPPORT, dtype=jnp.float32)
    builder_transitions = traj.builder_transitions

    builder_valid = jnp.logical_not(builder_transitions.env_output.done)  # (T_b, B)
    T_b, B = builder_valid.shape

    # --- V-Trace IMPALA Variables ---
    rho_t = jnp.minimum(1.0, importance_sampling_ratios)
    c_t = jnp.minimum(1.0, importance_sampling_ratios)

    # --- 1. Extract and Scale Base Values & Rewards ---
    # Value
    builder_value_probs = jnp.exp(
        builder_transitions.agent_output.actor_output.value_head.log_probs
    )
    n_bins = builder_value_probs.shape[-1]

    final_reward = traj.player_transitions.env_output.win_reward[-1]  # (B, 3)
    num_valid_steps = builder_valid.astype(jnp.int32).sum(axis=0)  # (B,)

    # Use n_bins directly instead of hardcoding 3 for safety/scalability
    builder_reward = jnp.zeros((T_b, B, n_bins), dtype=jnp.float32)
    safe_idx = jnp.clip(num_valid_steps, 0, T_b - 1)
    batch_idx = jnp.arange(B)
    has_terminal = num_valid_steps < T_b
    builder_reward = builder_reward.at[safe_idx, batch_idx].set(
        final_reward * has_terminal[:, None]
    )

    # Entropy
    builder_log_prob = (
        builder_transitions.agent_output.actor_output.action_head.log_prob
    )
    builder_ent_scaled = (
        builder_transitions.agent_output.actor_output.conditional_entropy_head.logits
        * entropy_normalising_constant
    )
    ent_reward = -builder_log_prob

    # --- 2. Concatenate Rewards, Values, and Next Values ---
    # Shape: (T_b, B, n_bins + 1)
    combined_rewards = jnp.concatenate([builder_reward, ent_reward[..., None]], axis=-1)

    combined_values = jnp.concatenate(
        [builder_value_probs, builder_ent_scaled[..., None]], axis=-1
    )

    # Construct the offset for next values, padding the end of the trajectory
    last_values = jnp.concatenate(
        [builder_value_probs[-1:], jnp.zeros_like(builder_ent_scaled[:1])[..., None]],
        axis=-1,
    )

    combined_next_values = (
        jnp.concatenate([combined_values[1:], last_values], axis=0)
        * builder_valid[..., None]
    )

    # --- 3. Compute Combined Deltas in one batched operation ---
    combined_deltas = (
        rho_t[..., None] * (combined_rewards + combined_next_values - combined_values)
    ).astype(jnp.float32)

    # --- 4. Compute Advantages Deltas ---
    # Transform the n_bins categorical value deltas back into a scalar for advantages
    builder_scalar_delta = (combined_deltas[..., :n_bins] @ cat_vf_support)[..., None]

    combined_adv_deltas = jnp.concatenate(
        [
            builder_scalar_delta,  # (T_b, B, 1)
            combined_deltas[..., n_bins:],  # (T_b, B, 1) - Entropy deltas
        ],
        axis=-1,
    )

    # --- 5. Discounts & Batched Segmented Cumsum ---
    td_lambdas = (c_t * td_lambda * builder_valid).astype(jnp.float32)
    gae_lambdas = (c_t * gae_lambda * builder_valid).astype(jnp.float32)

    combined_ret_cumsum = segmented_cumsum(combined_deltas, td_lambdas[..., None])
    combined_adv_cumsum = segmented_cumsum(combined_adv_deltas, gae_lambdas[..., None])

    # --- 6. Split Outputs ---
    win_returns = combined_ret_cumsum[..., :n_bins] + combined_values[..., :n_bins]

    ent_returns = (
        combined_ret_cumsum[..., n_bins] + combined_values[..., n_bins]
    ) / entropy_normalising_constant

    return BuilderTargets(
        win_returns=win_returns,
        win_advantages=combined_adv_cumsum[..., 0],
        ent_returns=ent_returns,
        ent_advantages=combined_adv_cumsum[..., 1],
    )
