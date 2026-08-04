import jax
import jax.numpy as jnp

from rl.environment.data import CAT_VF_SUPPORT
from rl.environment.interfaces import (
    Batch,
    BuilderActorOutput,
    BuilderTargets,
    PlayerTargets,
    Trajectory,
)
from rl.online.config import Porygon2LearnerConfig


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
    value_log_probs: jax.Array,
    isr: jax.Array,
    config: Porygon2LearnerConfig,
) -> tuple[PlayerTargets, dict[str, jax.Array]]:
    """Computes v-trace returns and advantages on the win/loss channel.

    PBRS/potential shaping retired (Aug 2026): the shaped-advantage era's
    channel machinery lived here; the win channel is now the sole reward.
    Multi-gamma auxiliary value targets (compute_aux_value_targets)
    supply the dense representation-shaping signal instead.

    IMPACT-style: ``value_log_probs`` are the *fast* EMA target's predictions
    and ``isr = pi_target/mu`` its ratio to the behavior policy, so v-trace
    estimates the target policy's values with off-policy correction — stable
    under replay reuse because the fast target tracks the learner within ~1k
    steps.
    """
    cat_vf_support = jnp.asarray(CAT_VF_SUPPORT, dtype=isr.dtype)

    dones_expanded = jnp.expand_dims(batch.player_transitions.env_output.done, axis=-1)
    mask_expanded = 1 - (jnp.cumsum(dones_expanded, axis=0) - dones_expanded)
    discount_t = (1 - dones_expanded) * config.player_gamma * mask_expanded

    alpha = config.player_alpha

    rho_t = (1 - alpha) * isr + alpha * jnp.minimum(1.0, isr)
    rho_t = jnp.expand_dims(rho_t, axis=-1)

    c_t = (1 - alpha) * isr + alpha * jnp.minimum(1.0, isr)
    c_t = jnp.expand_dims(c_t, axis=-1)

    r_t = batch.player_transitions.env_output.win_reward

    v_tm1 = jnp.exp(value_log_probs)
    last_values = v_tm1[-1:]
    v_t = jnp.concatenate([v_tm1[1:], last_values], axis=0)

    td_errors = rho_t * mask_expanded * (r_t + discount_t * v_t - v_tm1)

    lambda_ = jnp.asarray(config.player_lambda, dtype=isr.dtype)
    errors = vtrace(td_errors, discount_t, c_t * lambda_)

    targets_tm1 = (errors + v_tm1) * mask_expanded
    q_bootstrap = jnp.concatenate(
        [
            lambda_ * targets_tm1[1:] + (1 - lambda_) * v_tm1[1:],
            v_t[-1:],
        ],
        axis=0,
    )
    q_estimate = r_t + discount_t * q_bootstrap

    pg_advantages = rho_t * (q_estimate - v_tm1)

    win_advantages = pg_advantages @ cat_vf_support
    win_returns = targets_tm1

    value_mask = jnp.squeeze(mask_expanded, axis=-1).astype(jnp.bool_)

    t_length, batch_size, *_ = batch.player_transitions.env_output.action_mask.shape
    num_actions = batch.player_transitions.env_output.action_mask.reshape(
        t_length, batch_size, -1
    ).sum(axis=-1)
    policy_mask = (
        value_mask
        & jnp.logical_not(batch.player_transitions.env_output.done)
        & (num_actions > 1)
    )

    # Off-policyness of the replayed batch: normalised effective sample
    # size of the raw importance ratios (1 = fully on-policy; low means the
    # truncated estimator is living off a few samples) and the fraction of
    # steps where the v-trace ρ/c truncation at 1 is active. Both feed the
    # replay-ratio controller diagnostics alongside the actor KL.
    isr_mean = isr.mean(where=policy_mask)
    isr_sq_mean = jnp.square(isr).mean(where=policy_mask)

    channel_logs = {
        "player_isr_ess": isr_mean * isr_mean / (isr_sq_mean + 1e-8),
        "player_rho_clip_frac": (isr > 1.0).mean(where=policy_mask),
        "player_win_adv_std": win_advantages.std(where=policy_mask),
    }

    return (
        PlayerTargets(
            win_returns=win_returns,
            advantages=win_advantages,
            policy_mask=policy_mask,
            value_mask=value_mask,
        ),
        channel_logs,
    )


def compute_aux_value_targets(
    batch: Batch,
    aux_value_log_probs: jax.Array,
    isr: jax.Array,
    config: Porygon2LearnerConfig,
) -> jax.Array:
    """Per-lambda v-trace distribution targets for the multi-lambda
    auxiliary value heads.

    Mirrors compute_player_targets bin-space v-trace at the main gamma,
    vectorised over config.player_aux_lambdas, with each lambda
    bootstrapping from its OWN head's readout. With terminal-only reward
    a gamma spectrum degenerates (gamma^45 kills the signal), so the aux
    spectrum varies the bias/variance of the TARGET instead: lambda=1 is
    the Monte Carlo anchor (its gap to the main lambda=0.99 head is a
    direct bootstrap-bias readout), low lambda leans on the critic. The
    spectrum is deliberately independent of config.player_lambda, which
    the mixture bandit (rl/online/bandit.py) varies per window — aux
    target semantics must not drift with the live arm. Targets only —
    the aux heads never produce advantages; the policy reads the main
    head exclusively.

    aux_value_log_probs: (T, B, K, n_bins) from the fast EMA target.
    Returns (T, B, K, n_bins) distribution targets.
    """
    lambdas = jnp.asarray(config.player_aux_lambdas, dtype=isr.dtype)

    dones = jnp.expand_dims(
        batch.player_transitions.env_output.done, axis=(-2, -1)
    )  # (T, B, 1, 1)
    mask_expanded = 1 - (jnp.cumsum(dones, axis=0) - dones)
    discount_t = (1 - dones) * config.player_gamma * mask_expanded

    alpha = config.player_alpha
    rho_t = (1 - alpha) * isr + alpha * jnp.minimum(1.0, isr)
    rho_t = jnp.expand_dims(rho_t, axis=(-2, -1))
    c_t = rho_t

    r_t = batch.player_transitions.env_output.win_reward[:, :, None, :]

    v_tm1 = jnp.exp(aux_value_log_probs)  # (T, B, K, n_bins)
    v_t = jnp.concatenate([v_tm1[1:], v_tm1[-1:]], axis=0)

    td_errors = rho_t * mask_expanded * (r_t + discount_t * v_t - v_tm1)

    errors = vtrace(td_errors, discount_t, c_t * lambdas[None, None, :, None])

    return (errors + v_tm1) * mask_expanded


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
