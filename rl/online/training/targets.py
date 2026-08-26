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
    """Computes Retrace VALUE targets on the win/loss channel plus the
    plain v-trace POLICY advantage the PPO surrogate reads (2026-08-26 —
    the single-action advantage returned after five days away — the
    all-action logit-force era in between read the advantage head).

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
    dones = batch.player_transitions.env_output.done
    mask = (1 - (jnp.cumsum(dones, axis=0) - dones)).astype(jnp.float32)
    discount_t = (1 - dones).astype(jnp.float32) * config.player_gamma * mask

    # Truncated importance weights, AlphaStar/IMPALA: clipped IS only.
    # rho and c are the SAME quantity here — the two were separate
    # expressions behind a player_alpha blend between raw and clipped IS,
    # a dial nothing ever moved off 1.0 (removed 2026-08-21), so this is
    # one min() instead of four multiplies and two adds.
    truncated_isr = jnp.minimum(1.0, isr).astype(jnp.float32)
    rho_t = truncated_isr
    c_t = truncated_isr

    # Scalar-space recursion (2026-08-26). The recursion used to run
    # per-atom in distribution space, accumulating signed measures
    # (advantage_shift, r_t as a distribution) into a CE label that was
    # NOT a probability distribution — negative components, mass != 1.
    # softmax CE tolerates that algebraically, but the label stopped
    # meaning "a distribution over outcomes". The scalar form is the
    # same estimator (v-trace is linear, so @ support commutes with the
    # recursion) projected once through two_hot at the end, so the label
    # is always on the simplex and the recursion is one channel instead
    # of n_bins. f32 throughout (LESSONS 2: value recursions run and
    # return f32).
    support = jnp.asarray(CAT_VF_SUPPORT, dtype=jnp.float32)
    r_t = batch.player_transitions.env_output.win_reward.astype(jnp.float32) @ support

    v_tm1 = jnp.exp(value_log_probs.astype(jnp.float32)) @ support
    v_t = jnp.concatenate([v_tm1[1:], v_tm1[-1:]], axis=0)

    # Retrace over the COMPOSED Q (2026-08-25): the subtracted baseline is
    # Q(s, a) = V(s) + A(s, a), not V(s) alone — in scalar space the
    # baseline shift IS adv_taken. E_pi[Q(s', .)] = V(s') EXACTLY by
    # heads.compose_q's centring, so the bootstrap is untouched and only
    # the baseline moves — and E_pi[A] = 0 means the doubly-robust
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
    td_errors = (
        rho_t * mask * (r_t + discount_t * v_t - v_tm1 - adv_taken.astype(jnp.float32))
    )

    errors = vtrace(td_errors, discount_t, c_t * config.player_lambda)
    scalar_returns = (errors + v_tm1) * mask

    # Off-mask rows stay inert zero vectors; every masked row is a proper
    # two-hot distribution (two_hot clips to the support range).
    win_returns = two_hot(scalar_returns, support) * mask[..., None]

    # Policy advantage (2026-08-26): a PLAIN v-trace pass over the V
    # readout — no Retrace baseline shift — feeding the PPO surrogate's
    # taken-action advantage. Same construction as the builder's
    # pg_advantages: q_estimate bootstraps on the lambda-mixed v-trace
    # value of the NEXT step, so the outcome enters exactly once (a done
    # row's discount is 0 and its estimate is the terminal reward itself;
    # done rows are excluded by policy_mask anyway). rho truncates the
    # off-policy weight as everywhere else; f32 like the value recursion.
    td_plain = rho_t * mask * (r_t + discount_t * v_t - v_tm1)
    vtrace_v = vtrace(td_plain, discount_t, c_t * config.player_lambda) + v_tm1
    q_bootstrap = jnp.concatenate(
        [
            config.player_lambda * vtrace_v[1:]
            + (1 - config.player_lambda) * v_tm1[1:],
            v_tm1[-1:],
        ],
        axis=0,
    )
    q_estimate = r_t + discount_t * q_bootstrap
    pg_advantages = rho_t * (q_estimate - v_tm1) * mask

    value_mask = mask.astype(jnp.bool_)
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
            pg_advantages=pg_advantages,
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


# Cap on the advantage tilt's per-cell log-mass contribution: e^3 ~ 20x is
# plenty of reordering headroom, and one wild A outlier cannot own p*. The
# GRADIENT pi - p* is bounded by 1 regardless — this bounds the TARGET's
# concentration, not the force.
_SUPPORT_TILT_CLIP = 3.0


def support_target(
    reg_log_policy: jax.Array,
    legal_mask: jax.Array,
    temperature: float,
    tilt_logits: jax.Array | None = None,
) -> jax.Array:
    """log p* over legal cells: the support anchor's target distribution,
    p* ~ pi_reg ** (1/T) * exp(tilt), f32, stop-gradiented. Written once here
    so the loss (`support_kl`) and the anchor telemetry read the SAME
    construction.

    With no tilt this is the phase-1/2 anchor: a monotone power transform of
    the frozen reference — same ranking, it never says WHICH cell. The tilt
    is where per-cell knowledge enters (phase 3 passes sg(A_target) / tau):
    p* becomes MPO's E-step target (arXiv:1806.06920) with pi_reg as prior,
    so restored mass lands on the cells the critic ranks highest instead of
    flattening the modality. The tilt is stop-gradiented and clipped, and
    tilt contributions are capped at +-_SUPPORT_TILT_CLIP per cell.

    Masked with -1e9 rather than -inf: `0 * -inf` poisons the vjp on rows
    whose targets are all-zero.
    """
    lr = reg_log_policy.astype(jnp.float32)
    target_logits = lr / temperature
    if tilt_logits is not None:
        clipped_tilt = jnp.clip(
            jax.lax.stop_gradient(tilt_logits.astype(jnp.float32)),
            -_SUPPORT_TILT_CLIP,
            _SUPPORT_TILT_CLIP,
        )
        target_logits = target_logits + clipped_tilt
    scaled = jnp.where(legal_mask, target_logits, -1e9)
    return jax.lax.stop_gradient(jax.nn.log_softmax(scaled, axis=-1))


def support_kl(
    log_policy: jax.Array,
    reg_log_policy: jax.Array,
    legal_mask: jax.Array,
    temperature: float,
    tilt_logits: jax.Array | None = None,
) -> jax.Array:
    """KL(p* || pi) per state over legal cells, f32, where p* is the FROZEN
    reference raised to temperature T and optionally advantage-tilted:
    p* ~ pi_reg ** (1/T) * exp(tilt) (see `support_target`).

    The support-preserving half of the reference pair, and the mirror image of
    `reference_kl` above in the only way that matters. Differentiated through
    pi alone, so the per-logit gradient is exactly

        d/dy_b  KL(p_T || pi)  =  pi(b) - p_T(b)

    which carries NO pi prefactor, is bounded by 1 for every pi including
    pi -> 0, and sums to zero over legal cells. `reference_kl`'s KL(pi ||
    pi_reg) has gradient pi(b) * (log(pi(b)/pi_reg(b)) - KL): the pi prefactor
    makes its force vanish on a starved cell, so it is mode-SEEKING and by
    construction indifferent to a dropped modality. This one is mode-covering
    and can refill one.

    The two knobs do different jobs. Direction buys mass-independence — even at
    T = 1 the gradient is pi(b) - pi_reg(b). T decides where the anchor SITS:
    pi_reg is a snapshot of an already-collapsing policy, so T = 1 pins the
    policy at the reference's own support (a brake), while T > 1 re-inflates
    the reference's tail through a monotone power transform — same ranking,
    more mass on starved cells — putting the anchor above the current policy
    and making it restorative. The tilt is the third knob and carries the
    only thing neither of those can: WHICH cell (phase 3).
    """
    lp = log_policy.astype(jnp.float32)
    log_p_t = support_target(reg_log_policy, legal_mask, temperature, tilt_logits)
    p_t = jnp.where(legal_mask, jnp.exp(log_p_t), 0.0)
    return jnp.where(legal_mask, p_t * (log_p_t - lp), 0.0).sum(axis=-1)


def compute_q_onestep_targets(
    batch: Batch, v_target: jax.Array, config: Porygon2LearnerConfig
) -> jax.Array:
    """TD(0) labels for the residual Q critic: y_t = r_t + gamma * V(s_{t+1})
    from the TARGET net's win-value head (plain Q^pi — no Retrace trace,
    no reference-policy transform; the reference enters the POLICY
    objective only, as the differentiated magnet). Replaced the Retrace
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
