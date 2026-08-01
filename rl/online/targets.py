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
    state_potential: jax.Array,
    announced_potential: jax.Array,
    potential_target_adv_share: jax.Array,
    config: Porygon2LearnerConfig,
) -> tuple[PlayerTargets, dict[str, jax.Array]]:
    """Computes v-trace returns and advantages over stacked reward channels.

    Also returns a logs dict of per-channel advantage diagnostics: the win
    and potential channel stds, their correlation (high correlation means
    the potential channel is a rescaled copy of the outcome signal and adds
    no credit-assignment information), and the potential channel's share of
    the combined advantage magnitude.

    Channels: [0:n_bins] categorical win reward, [n_bins] learned potential
    (``state_potential``, the frozen offline critic's Φ(s) in [-1, 1];
    terminal-only reward). The potential channel's coefficient is solved
    per batch so its share of the combined advantage magnitude equals
    ``potential_target_adv_share``.

    Dice excision (``config.potential_dice_excised``): every appearance of
    the potential channel's NEXT-step value — in the TD errors and in the
    bootstrap — is replaced by ``announced_potential`` at t+1 (Φ_ann: the
    critic read with both players' turn choices revealed but chance
    unresolved), while the subtracted current-step value and the terminal
    reward stay realised. Each one-step term becomes γ·Φ_ann(t+1) − Φ(t):
    same conditional expectation as realised PBRS (Φ_ann = E[Φ |
    announcement], tower property), but the channel stops paying the agent
    for crits, misses and damage rolls. With the flag off, this function is
    bit-identical to the pre-excision code and ``announced_potential`` is
    ignored.

    IMPACT-style: ``value_log_probs`` are the *fast* EMA target's predictions
    and ``isr = pi_target/mu`` its ratio to the behavior policy, so v-trace
    estimates the target policy's values with off-policy correction — stable
    under replay reuse because the fast target tracks the learner within ~1k
    steps.
    """
    cat_vf_support = jnp.asarray(CAT_VF_SUPPORT, dtype=isr.dtype)

    dones = batch.player_transitions.env_output.done
    state_potential = state_potential.astype(isr.dtype)

    dones_expanded = jnp.expand_dims(batch.player_transitions.env_output.done, axis=-1)
    mask_expanded = 1 - (jnp.cumsum(dones_expanded, axis=0) - dones_expanded)
    discount_t = (1 - dones_expanded) * config.player_gamma * mask_expanded

    alpha = config.player_alpha

    rho_t = (1 - alpha) * isr + alpha * jnp.minimum(1.0, isr)
    rho_t = jnp.expand_dims(rho_t, axis=-1)

    c_t = (1 - alpha) * isr + alpha * jnp.minimum(1.0, isr)
    c_t = jnp.expand_dims(c_t, axis=-1)

    player_reward = batch.player_transitions.env_output.win_reward

    terminal_potential_reward = jnp.where(dones, state_potential, 0.0)
    r_t = jnp.concatenate(
        (player_reward, terminal_potential_reward[..., None]), axis=-1
    )

    value_probs = jnp.exp(value_log_probs)

    n_bins = value_probs.shape[-1]
    v_tm1 = jnp.concatenate((value_probs, state_potential[..., None]), axis=-1)
    last_values = v_tm1[-1:]

    v_t = jnp.concatenate([v_tm1[1:], last_values], axis=0)
    if config.potential_dice_excised:
        # Dice excision: the potential channel's next-step value is the
        # ANNOUNCED Φ_ann(t+1), so its TD is γ·Φ_ann(t+1) − Φ(t). The final
        # row's filler repeats the realised value like last_values; it only
        # feeds masked/zero-discount positions.
        announced_potential = announced_potential.astype(isr.dtype)
        potential_next = jnp.concatenate(
            [announced_potential[1:], state_potential[-1:]], axis=0
        )
        v_t = jnp.concatenate([v_t[..., :n_bins], potential_next[..., None]], axis=-1)
    td_errors = rho_t * mask_expanded * (r_t + discount_t * v_t - v_tm1)

    # Per-channel λ: the win channel keeps the long player_lambda horizon;
    # the potential channel uses its own short player_potential_lambda so
    # its advantage stays near the one-step PBRS signal γΦ(s')−Φ(s)
    # instead of telescoping into a copy of the outcome signal.
    lambda_ = jnp.concatenate(
        [
            jnp.full((n_bins,), config.player_lambda),
            jnp.array([config.player_potential_lambda]),
        ]
    ).astype(isr.dtype)

    errors = vtrace(td_errors, discount_t, c_t * lambda_)

    targets_tm1 = (errors + v_tm1) * mask_expanded
    q_bootstrap = jnp.concatenate(
        [
            lambda_ * targets_tm1[1:] + (1 - lambda_) * v_tm1[1:],
            v_t[-1:],
        ],
        axis=0,
    )
    if config.potential_dice_excised:
        # The realised Φ(t+1) also enters the bootstrap — directly through
        # the (1−λ)·v_tm1[1:] term and inside targets_tm1[1:]'s value base
        # (λ·targets[1:] + (1−λ)·v[1:] = λ·errors[1:] + v[1:] within the
        # mask). Substituting the announced next value in that identity
        # excises both at once; the errors are already excised above.
        potential_q_bootstrap = jnp.concatenate(
            [
                config.player_potential_lambda * errors[1:, ..., n_bins]
                + potential_next[:-1],
                potential_next[-1:],
            ],
            axis=0,
        )
        q_bootstrap = jnp.concatenate(
            [q_bootstrap[..., :n_bins], potential_q_bootstrap[..., None]], axis=-1
        )
    q_estimate = r_t + discount_t * q_bootstrap

    pg_advantages = rho_t * (q_estimate - v_tm1)

    win_advantages = pg_advantages[..., :n_bins] @ cat_vf_support
    potential_advantages = pg_advantages[..., n_bins]

    win_returns = targets_tm1[..., :n_bins]

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

    win_adv_mean = win_advantages.mean(where=policy_mask)
    pot_adv_mean = potential_advantages.mean(where=policy_mask)
    win_adv_std = win_advantages.std(where=policy_mask)
    pot_adv_std = potential_advantages.std(where=policy_mask)
    adv_cov = (
        (win_advantages - win_adv_mean) * (potential_advantages - pot_adv_mean)
    ).mean(where=policy_mask)

    # Solve the channel coefficient from the target share s of combined
    # advantage magnitude: coef·σ_pot = s/(1−s)·σ_win. The stds don't
    # depend on the coef, so this is a direct solve, not a feedback loop.
    share = jnp.clip(potential_target_adv_share, 0.0, 0.99).astype(isr.dtype)
    potential_advantage_coef = jnp.minimum(
        share / (1.0 - share) * win_adv_std / (pot_adv_std + 1e-8),
        config.player_potential_coef_max,
    )
    combined_advantage = (
        win_advantages + potential_advantage_coef * potential_advantages
    )

    # Off-policyness of the replayed batch: normalised effective sample
    # size of the raw importance ratios (1 = fully on-policy; low means the
    # truncated estimator is living off a few samples) and the fraction of
    # steps where the v-trace ρ/c truncation at 1 is active. Both feed the
    # replay-ratio controller diagnostics alongside the actor KL.
    isr_mean = isr.mean(where=policy_mask)
    isr_sq_mean = jnp.square(isr).mean(where=policy_mask)

    scaled_pot_adv_std = potential_advantage_coef * pot_adv_std
    channel_logs = {
        "player_isr_ess": isr_mean * isr_mean / (isr_sq_mean + 1e-8),
        "player_rho_clip_frac": (isr > 1.0).mean(where=policy_mask),
        "player_win_adv_std": win_adv_std,
        "player_potential_adv_std": pot_adv_std,
        "player_potential_adv_coef": potential_advantage_coef,
        "player_potential_win_adv_corr": adv_cov / (win_adv_std * pot_adv_std + 1e-8),
        "player_potential_adv_share": scaled_pot_adv_std
        / (win_adv_std + scaled_pot_adv_std + 1e-8),
        # Fraction of steps where the potential channel flips the advantage
        # sign: shaping can only change which actions get pushed up vs down
        # where this is nonzero, so ~0 means the channel decorates
        # already-correct advantages and cannot move the policy.
        "player_potential_adv_sign_flip": (
            jnp.sign(combined_advantage) != jnp.sign(win_advantages)
        ).mean(where=policy_mask),
    }

    return (
        PlayerTargets(
            win_returns=win_returns,
            advantages=combined_advantage,
            policy_mask=policy_mask,
            value_mask=value_mask,
        ),
        channel_logs,
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
