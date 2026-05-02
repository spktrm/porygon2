import jax
import jax.numpy as jnp

from rl.environment.data import CAT_VF_SUPPORT
from rl.environment.interfaces import (
    Batch,
    BuilderActorOutput,
    BuilderTargets,
    PlayerActorOutput,
    PlayerTargets,
    Trajectory,
)
from rl.learner.config import Porygon2LearnerConfig


def vtrace(td_errors: jax.Array, discount_t: jax.Array, c_tm1: jax.Array) -> jax.Array:
    """
    Backward cumulative sum using parallel associative scan.
    Best for very long sequence lengths on GPU/TPU.
    """

    discount_t = discount_t.astype(td_errors.dtype)
    c_tm1 = c_tm1.astype(td_errors.dtype)

    def _body(acc, xs):
        td_error, discount, c = xs
        acc = td_error + discount * c * acc
        return acc, acc

    _, errors = jax.lax.scan(
        _body,
        jnp.zeros_like(td_errors[0]),
        (td_errors, discount_t, c_tm1),
        reverse=True,
    )

    return errors


def compute_player_targets(
    batch: Batch,
    learner_pred: PlayerActorOutput,
    target_pred: PlayerActorOutput,
    isr: jax.Array,
    config: Porygon2LearnerConfig,
) -> PlayerTargets:
    cat_vf_support = jnp.asarray(CAT_VF_SUPPORT, dtype=isr.dtype)

    dones_expanded = jnp.expand_dims(batch.player_transitions.env_output.done, axis=-1)
    mask_expanded = 1 - (jnp.cumsum(dones_expanded, axis=0) - dones_expanded)
    discounts = (1 - dones_expanded) * config.player_gamma * mask_expanded

    alpha = config.player_alpha

    rho_expanded = (1 - alpha) * isr + alpha * jnp.minimum(1.0, isr)
    rho_expanded = jnp.expand_dims(rho_expanded, axis=-1)

    c_expanded = (1 - alpha) * isr + alpha * jnp.minimum(1.0, isr)
    c_expanded = jnp.expand_dims(c_expanded, axis=-1)

    player_reward = batch.player_transitions.env_output.win_reward
    target_value_probs = jnp.exp(target_pred.value_head.log_probs)
    n_bins = target_value_probs.shape[-1]

    heuristic_zeros = jnp.zeros_like(player_reward[..., 0])
    combined_rewards = jnp.concatenate(
        (player_reward, heuristic_zeros[..., None]), axis=-1
    )

    combined_values = jnp.concatenate(
        (
            target_value_probs,
            batch.player_transitions.env_output.state_potential[..., None],
        ),
        axis=-1,
    )
    last_values = combined_values[-1:]

    combined_next_values = jnp.concatenate([combined_values[1:], last_values], axis=0)
    combined_td_errors = (
        rho_expanded
        * mask_expanded
        * (combined_rewards + discounts * combined_next_values - combined_values)
    )

    vtrace_errors = vtrace(
        combined_td_errors, discounts, c_expanded * config.player_lambda
    )

    returns = (vtrace_errors + combined_values) * mask_expanded
    q_bootstrap = jnp.concatenate(
        [
            config.player_lambda * returns[1:]
            + (1 - config.player_lambda) * combined_values[1:],
            combined_values[-1:],
        ],
        axis=0,
    )
    q_estimate = combined_rewards + discounts * q_bootstrap
    pg_advantages = mask_expanded * rho_expanded * (q_estimate - combined_values)

    combined_advantage = (
        pg_advantages[..., :n_bins] @ cat_vf_support
        + config.player_potential_reward_scale * pg_advantages[..., n_bins]
    )

    win_returns = returns[..., :n_bins]
    win_returns = jnp.maximum(win_returns, 0.0)
    norm_factor = win_returns.sum(axis=-1, keepdims=True)
    win_returns = win_returns / norm_factor.clip(min=1e-8)

    return PlayerTargets(
        win_returns=win_returns,
        advantages=combined_advantage,
        mask=jnp.squeeze(mask_expanded, axis=-1).astype(jnp.bool_),
        win_returns_norm_factor=norm_factor,
    )


def compute_builder_targets(
    traj: Trajectory,
    target_pred: BuilderActorOutput,
    importance_sampling_ratios: jax.Array,
    lambda_: float,
    entropy_normalising_constant: float,
) -> BuilderTargets:
    cat_vf_support = jnp.asarray(
        CAT_VF_SUPPORT, dtype=target_pred.value_head.log_probs.dtype
    )
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
    builder_reward = jnp.zeros((T_b, B, n_bins), dtype=builder_value_probs.dtype)
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
    combined_td_errors = rho_t[..., None] * (
        combined_rewards + combined_next_values - combined_values
    )

    # --- 5. Discounts & Batched Segmented Cumsum ---
    vtrace_errors = vtrace(
        combined_td_errors, builder_valid[..., None], c_t[..., None] * lambda_
    )
    returns = vtrace_errors + combined_values
    q_bootstrap = jnp.concatenate(
        [
            lambda_ * returns[1:] + (1 - lambda_) * combined_values[1:],
            combined_values[-1:],
        ],
        axis=0,
    )
    q_estimate = combined_rewards + builder_valid[..., None] * q_bootstrap
    pg_advantages = rho_t[..., None] * (q_estimate - combined_values)

    # --- 6. Split Outputs ---
    win_returns = returns[..., :n_bins]
    ent_returns = returns[..., n_bins]

    return BuilderTargets(
        win_returns=win_returns,
        win_advantages=pg_advantages[..., :n_bins] @ cat_vf_support,
        ent_returns=ent_returns,
        ent_advantages=pg_advantages[..., n_bins],
    )
