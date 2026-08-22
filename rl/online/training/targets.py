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
    """Computes v-trace VALUE targets on the win/loss channel.

    Policy advantages are gone: the single-action PG and UPGO terms were
    removed 2026-08-21 (LESSONS.md 3), so nothing consumes them. What is
    left here feeds the value head and the off-policyness diagnostics.

    PBRS/potential shaping retired (Aug 2026): the shaped-advantage era's
    channel machinery lived here; the win channel is now the sole reward.

    IMPACT-style: ``value_log_probs`` are the *fast* EMA target's predictions
    and ``isr = pi_target/mu`` its ratio to the behavior policy, so v-trace
    estimates the target policy's values with off-policy correction — stable
    under replay reuse because the fast target tracks the learner within ~1k
    steps.

    config.player_lambda (0.8, AlphaStar's TD(lambda) value) shapes the
    value targets.
    """
    dones_expanded = jnp.expand_dims(batch.player_transitions.env_output.done, axis=-1)
    mask_expanded = 1 - (jnp.cumsum(dones_expanded, axis=0) - dones_expanded)
    discount_t = (1 - dones_expanded) * config.player_gamma * mask_expanded

    # Truncated importance weights, AlphaStar/IMPALA: clipped IS only.
    # rho and c are the SAME quantity here — the two were separate
    # expressions behind a player_alpha blend between raw and clipped IS,
    # a dial nothing ever moved off 1.0 (removed 2026-08-21), so this is
    # one min() instead of four multiplies and two adds.
    truncated_isr = jnp.expand_dims(jnp.minimum(1.0, isr), axis=-1)
    rho_t = truncated_isr
    c_t = truncated_isr

    r_t = batch.player_transitions.env_output.win_reward

    v_tm1 = jnp.exp(value_log_probs)
    last_values = v_tm1[-1:]
    v_t = jnp.concatenate([v_tm1[1:], last_values], axis=0)

    td_errors = rho_t * mask_expanded * (r_t + discount_t * v_t - v_tm1)

    td_lambda = jnp.asarray(config.player_lambda, dtype=isr.dtype)
    errors = vtrace(td_errors, discount_t, c_t * td_lambda)
    targets_tm1 = (errors + v_tm1) * mask_expanded

    win_returns = targets_tm1

    value_mask = jnp.squeeze(mask_expanded, axis=-1).astype(jnp.bool_)
    # Chunked unrolls: a chunk's final row is bootstrap-only — it anchors
    # the recursions above (v_t / UPGO's init carry read its value) but
    # takes no loss here, because chunks overlap by one row and that same
    # step trains as row 0 of the NEXT chunk. Exception: a done row on the
    # final position is the game's own terminal row (no next chunk) and
    # keeps its value target (= the terminal reward). policy_mask below
    # inherits this through value_mask; so does the Q-CE mask in
    # train_step.
    is_final_row = (
        jnp.arange(value_mask.shape[0])[:, None] == value_mask.shape[0] - 1
    )
    value_mask = value_mask & (
        ~is_final_row | batch.player_transitions.env_output.done.astype(jnp.bool_)
    )

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
    }

    return (
        PlayerTargets(
            win_returns=win_returns,
            policy_mask=policy_mask,
            value_mask=value_mask,
        ),
        channel_logs,
    )


def two_hot(scalar: jax.Array, support: jax.Array) -> jax.Array:
    """Project scalars onto a categorical support as the standard two-hot
    distribution: all mass on the two bins bracketing the value, split by
    linear interpolation. Values are clipped to the support's range."""
    scalar = jnp.clip(scalar, support[0], support[-1])
    upper_idx = jnp.clip(
        jnp.searchsorted(support, scalar, side="left"), 1, support.shape[0] - 1
    )
    lower = support[upper_idx - 1]
    upper = support[upper_idx]
    w_upper = (scalar - lower) / jnp.maximum(upper - lower, 1e-8)
    n_bins = support.shape[0]
    return jax.nn.one_hot(upper_idx - 1, n_bins) * (1.0 - w_upper[..., None]) + (
        jax.nn.one_hot(upper_idx, n_bins) * w_upper[..., None]
    )


def compute_q_targets(
    batch: Batch,
    q_logits: jax.Array,
    target_log_policy: jax.Array,
    isr: jax.Array,
    config: Porygon2LearnerConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Retrace(lambda) targets for the Q critic (docs/q-critic-plan.md),
    computed from the privileged Q_all rung; the Q_private rung trains by
    CE against the same labels learner-side.

    Scalar-space recursion with a two-hot projection
    back onto CAT_VF_SUPPORT for the CE loss. Expectation bootstrap —
    delta_t uses V(s_{t+1}) = sum_a pi(a|s_{t+1}) E[Q(s_{t+1}, a)], never
    a max — so the target stays sound against a mixed-strategy opponent
    and free of argmax overestimation. The correction product starts at
    t+1 (Retrace, vs v-trace's t): a_t is given, only the continuation is
    off-policy — so delta_t itself carries no rho factor, and the
    recursion is E_t = delta_t + gamma_t * c_{t+1} * E_{t+1} with
    c = player_q_lambda * min(1, pi_target/mu). min(1, .) tolerates
    arbitrary behaviour policies: replay reuse, and the exploration
    ladder's epsilon-mixed games (config.explore_game_prob),
    whose recorded mu IS the mixed distribution.

    q_logits / target_log_policy come from the fast EMA target network —
    the same IMPACT reasoning as the v-trace reference policy. Everything
    runs in f32: value readouts are precision-critical under the bf16
    training policy — a bf16 scan carry crashed the 2026-08-13 session
    (LESSONS.md 2).

    Returns (q_target_probs, retrace_g, q_all, v_exp, q_taken):
      q_target_probs (T, B, n_bins) — CE labels for the taken action;
      retrace_g      (T, B)         — the scalar Retrace returns (R2 diag,
                                      and the Q-boosting advantage's
                                      multistep numerator);
      q_all          (T, B, A)      — target net per-action E[Q];
      v_exp          (T, B)         — its policy expectation (the
                                      Q-boosting baseline), and
      q_taken        (T, B)         — E[Q] of the taken action (the
                                      onestep Q-boosting variant).
    q_all also feeds player_q_switch_move_gap / player_q_ev_gap.
    """
    support = jnp.asarray(CAT_VF_SUPPORT, dtype=jnp.float32)

    dones = batch.player_transitions.env_output.done  # (T, B)
    mask = (1 - (jnp.cumsum(dones, axis=0) - dones)).astype(jnp.float32)
    discount_t = (1 - dones) * config.player_gamma * mask
    discount_t = discount_t.astype(jnp.float32)

    q_probs = jax.nn.softmax(q_logits.astype(jnp.float32), axis=-1)
    q_all = q_probs @ support  # (T, B, A)

    action_mask = batch.player_transitions.env_output.action_mask
    flat_action_mask = action_mask.reshape(*q_all.shape)
    # Renormalised over legal cells: the policy head already zeroes
    # illegal mass, this just guards E[Q] against numerical dust there.
    pi = jnp.exp(target_log_policy.astype(jnp.float32)) * flat_action_mask
    pi = pi / jnp.maximum(pi.sum(axis=-1, keepdims=True), 1e-8)
    v_exp = (pi * q_all).sum(axis=-1)  # (T, B)

    action_index = (
        batch.player_transitions.agent_output.actor_output.action_head.action_index
    )
    q_taken = jnp.take_along_axis(q_all, action_index[..., None], axis=-1).squeeze(-1)

    r_t = (batch.player_transitions.env_output.win_reward @ support).astype(jnp.float32)

    # Terminal anchor: rewards land on the terminal OBSERVATION row (the
    # done step, where no action is taken and the Q head never trains), so
    # that row's state value is exactly its stored reward. Bootstrap the
    # last acted step on r rather than on the Q readout's uncalibrated
    # terminal estimate, and zero the terminal row's own delta — the
    # outcome enters the recursion through the bootstrap, exactly once,
    # with no reference to the terminal row's meaningless action_index.
    v_boot = jnp.where(dones.astype(bool), r_t, v_exp)
    v_next = jnp.concatenate([v_boot[1:], v_boot[-1:]], axis=0)
    td_errors = (
        jnp.where(dones.astype(bool), 0.0, r_t + discount_t * v_next - q_taken) * mask
    )

    c_t = config.player_q_lambda * jnp.minimum(1.0, isr.astype(jnp.float32))
    # Shift left: the recursion's trace factor is c_{t+1} (Retrace), and
    # the final step has no continuation to correct.
    c_next = jnp.concatenate([c_t[1:], jnp.zeros_like(c_t[-1:])], axis=0)
    errors = vtrace(td_errors, discount_t, c_next)

    retrace_g = jnp.clip(q_taken + errors, support[0], support[-1]) * mask
    q_target_probs = two_hot(retrace_g, support)
    return q_target_probs, retrace_g, q_all, v_exp, q_taken


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
