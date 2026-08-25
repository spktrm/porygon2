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
    adv_taken: jax.Array,
    config: Porygon2LearnerConfig,
) -> tuple[PlayerTargets, dict[str, jax.Array]]:
    """Computes Retrace VALUE targets on the win/loss channel.

    Policy advantages are gone: the single-action PG and UPGO terms were
    removed 2026-08-21 (CLAUDE.md 3), so nothing consumes them. What is
    left here feeds the value head and the off-policyness diagnostics.

    PBRS/potential shaping retired (Aug 2026): the shaped-advantage era's
    channel machinery lived here; the win channel is now the sole reward.

    IMPACT-style: ``value_log_probs`` are the *fast* EMA target's predictions
    and ``isr = pi_target/mu`` its ratio to the behavior policy, so v-trace
    estimates the target policy's values with off-policy correction — stable
    under replay reuse because the fast target tracks the learner within ~1k
    steps.

    ``adv_taken`` is the target critic's advantage at the action actually
    taken, which turns the estimator into Retrace over the COMPOSED Q — see
    the baseline comment below.

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

    # Retrace over the COMPOSED Q (2026-08-25): the subtracted baseline is
    # Q(s, a) = V(s) + A(s, a), not V(s) alone. E_pi[Q(s', .)] = V(s')
    # EXACTLY by heads.compose_q's centring, so the bootstrap is untouched
    # and only the baseline moves — and E_pi[A] = 0 means the doubly-robust
    # correction adds nothing back, it only takes the taken cell out.
    #
    # What that buys: with the baseline at V, the truncated rho attenuates
    # the WHOLE residual, including the action-conditional part the critic
    # already predicts. rho-bar = 1 therefore targets min(pi, mu)
    # renormalised — it pulls the value toward the BEHAVIOUR policy exactly
    # on the rows where pi has fallen below mu, which is the collapse's own
    # signature (LESSONS 6, player_isr_below1_switch_voluntary). With the
    # baseline at Q the model term enters at full pi weight and rho
    # attenuates only the environment noise: a bias reduction as much as a
    # variance one.
    #
    # A is zero-init (flat-at-init contract), so this is the identity at
    # launch and adv_taken = 0 is the exact revert.
    support = jnp.asarray(CAT_VF_SUPPORT, dtype=v_tm1.dtype)
    baseline_shift = advantage_shift(v_tm1, adv_taken.astype(v_tm1.dtype), support)

    td_errors = (
        rho_t * mask_expanded * (r_t + discount_t * v_t - v_tm1 - baseline_shift)
    )

    td_lambda = jnp.asarray(config.player_lambda, dtype=isr.dtype)
    errors = vtrace(td_errors, discount_t, c_t * td_lambda)
    targets_tm1 = (errors + v_tm1) * mask_expanded

    win_returns = targets_tm1

    value_mask = jnp.squeeze(mask_expanded, axis=-1).astype(jnp.bool_)
    # Chunked unrolls: a chunk's final row is bootstrap-only — it anchors
    # the recursions above (v_t reads its value) but
    # takes no loss here, because chunks overlap by one row and that same
    # step trains as row 0 of the NEXT chunk. Exception: a done row on the
    # final position is the game's own terminal row (no next chunk) and
    # keeps its value target (= the terminal reward). policy_mask below
    # inherits this through value_mask; so does the Q-CE mask in
    # train_step.
    is_final_row = jnp.arange(value_mask.shape[0])[:, None] == value_mask.shape[0] - 1
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


def advantage_shift(
    value_probs: jax.Array, adv: jax.Array, support: jax.Array
) -> jax.Array:
    """Signed measure over the value atoms carrying first moment ``adv`` and
    zero total mass: two_hot(m + adv) - two_hot(m) at m = E[value_probs].

    Adding it to a value distribution shifts that distribution's mean by
    exactly ``adv`` (up to the support's clipping) and is the identity at
    adv = 0. A scalar cannot enter the recursion any other way: the value
    loss is a cross-entropy whose LABEL is a vector over the atoms, and the
    recursion accumulates vectors to build it. Signed is fine — the whole
    recursion is already an accumulation of target-minus-value differences,
    and softmax_cross_entropy never required non-negative labels.

    Two-hot rather than a fixed direction like adv*(-1/2, 0, +1/2): the
    latter always moves mass between the EXTREME atoms, so a slightly worse
    action at a near-won state reads as "more likely a loss" instead of
    "more likely a draw"."""
    mean = value_probs @ support
    return two_hot(mean + adv, support) - two_hot(mean, support)


def reference_kl(
    log_policy: jax.Array, reg_log_policy: jax.Array, legal_mask: jax.Array
) -> jax.Array:
    """KL(pi || pi_reg) per state over legal cells, f32 — the expected
    reference penalty E_pi[log(pi/pi_reg)] the policy objective pays. Both
    log-policies are
    full-support learner-side readouts (illegal cells hold junk, masked)."""
    lp = log_policy.astype(jnp.float32)
    lr = reg_log_policy.astype(jnp.float32)
    pi = jnp.exp(lp) * legal_mask
    pi = pi / jnp.maximum(pi.sum(axis=-1, keepdims=True), 1e-8)
    return jnp.where(legal_mask, pi * (lp - lr), 0.0).sum(axis=-1)


def reference_penalty(
    log_policy: jax.Array,
    reg_log_policy: jax.Array,
    legal_mask: jax.Array,
    eta: float,
    eta_ent: float,
) -> tuple[jax.Array, jax.Array]:
    """The two analytic per-cell penalties the policy objective pays,
    returned split so the panels can tell them apart: eta*(log pi(a) - log
    pi_reg(a)) against the iteratively refined reference, and
    eta_ent*log pi(a) against uniform. The caller subtracts both from the
    critic's ADVANTAGE — rnad.py's learning_output, with A in place of the
    single-sample v-trace estimate (V is a per-row constant the NeuRD
    centring removes exactly).

    The second IS an entropy bonus. KL(pi || uniform) = sum_a pi(a) log
    pi(a) + log N, so under NeuRD — which reads the advantage per cell and
    applies it with no pi prefactor — it is just the first penalty with
    pi_reg = uniform, and the log N constant cancels in the re-centring at
    the call site. That is why it is written here rather than as a loss
    term: a differentiated entropy (or a reverse-KL magnet) carries a pi
    prefactor and cannot refill a modality that has already starved, which
    is what four magnet retunes measured (LESSONS 4). Together the two are
    a reference geometrically mixed toward uniform.

    Both grow without bound as pi(a) -> 0, which is the restoring force."""
    lp = log_policy.astype(jnp.float32)
    lr = reg_log_policy.astype(jnp.float32)
    ref = jnp.where(legal_mask, eta * (lp - lr), 0.0)
    ent = jnp.where(legal_mask, eta_ent * lp, 0.0)
    return ref, ent


def compute_q_onestep_targets(
    batch: Batch, v_target: jax.Array, config: Porygon2LearnerConfig
) -> jax.Array:
    """TD(0) labels for the residual Q critic: y_t = r_t + gamma * V(s_{t+1})
    from the TARGET net's win-value head (plain Q^pi — no Retrace trace,
    no reference-policy transform; the reference penalty lives only in the
    NeuRD advantage, targets.reference_penalty). Replaced the Retrace
    recursion 2026-08-23: the one-step label has ~33x less variance than
    the outcome chain for the action axis to be learnt against, and its
    state component is exactly what V already carries, so the residual
    A has only the action part left to fit.

    Terminal anchor as before: rewards land on the terminal OBSERVATION
    row, so that row's bootstrap is exactly its stored reward and its own
    label is 0 (never trained — q_mask excludes done rows). Rows past a
    chunk's first done (terminal-copy padding) read 0. f32 throughout.
    """
    support = jnp.asarray(CAT_VF_SUPPORT, dtype=jnp.float32)
    dones = batch.player_transitions.env_output.done
    mask = (1 - (jnp.cumsum(dones, axis=0) - dones)).astype(jnp.float32)
    done_b = dones.astype(bool)
    discount_t = (1 - dones).astype(jnp.float32) * config.player_gamma * mask
    r_t = (batch.player_transitions.env_output.win_reward @ support).astype(jnp.float32)
    v_boot = jnp.where(done_b, r_t, v_target.astype(jnp.float32))
    v_next = jnp.concatenate([v_boot[1:], v_boot[-1:]], axis=0)
    return jnp.where(done_b, 0.0, r_t + discount_t * v_next) * mask


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
