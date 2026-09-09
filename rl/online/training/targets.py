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
from rl.model.utils import prune_log_policy
from rl.online.config import Porygon2LearnerConfig
from rl.utils import average


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


def thresholded_target_ratio(
    target_log_policy: jax.Array,
    behaviour_log_prob: jax.Array,
    action_index: jax.Array,
    legal_mask: jax.Array,
    threshold: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The v-trace ratio pi_target(a) / mu(a) at the taken action, with the
    target policy THRESHOLDED first (rl/model/utils.py prune_log_policy --
    the same operation the `thresholded` eval slot samples from): a taken
    action the target has dropped below `threshold` gets ratio 0, so
    v-trace discards the row. DeepNash's `FineTuning` placement
    (rnad.py:798 post-processes pi and hands it to v_trace as
    merged_policy; acting_policy and the policy loss's pi stay raw): this
    is the ONLY place the thresholded distribution enters the learner.
    Variance control on the target estimator, not a policy force.

    Returns (ratio, raw ratio, taken-action-kept mask, fraction of the row's
    legal cells removed). threshold 0.0 makes ratio == raw ratio bit for
    bit; the raw ratio is what the telemetry twins read.
    """
    pruned = prune_log_policy(target_log_policy, legal_mask, threshold)
    index = action_index[..., None]
    taken_pruned = jnp.take_along_axis(pruned, index, axis=-1)[..., 0]
    taken_raw = jnp.take_along_axis(target_log_policy, index, axis=-1)[..., 0]
    removed = legal_mask & (pruned <= jnp.finfo(pruned.dtype).min)
    kept_taken = taken_pruned > jnp.finfo(pruned.dtype).min
    ratio_raw = jnp.exp(taken_raw - behaviour_log_prob)
    ratio = jnp.where(kept_taken, jnp.exp(taken_pruned - behaviour_log_prob), 0.0)
    removed_legal_fraction = removed.sum(axis=-1) / jnp.maximum(
        legal_mask.sum(axis=-1), 1
    )
    return ratio, ratio_raw, kept_taken, removed_legal_fraction


def trace_run_length(continues: jax.Array) -> jax.Array:
    """Per row, how many consecutive rows from it (itself included) keep
    the v-trace continuation alive -- `continues` is (T, B) True where the
    trace passes through the row. A thresholded ratio zeroing one row cuts
    every earlier row's run at it; the raw ratio never does, so the two
    runs side by side are the realised cost of the threshold."""

    def _body(run, alive):
        run = jnp.where(alive, 1.0 + run, 0.0)
        return run, run

    _, runs = jax.lax.scan(
        _body, jnp.zeros(continues.shape[1:], jnp.float32), continues, reverse=True
    )
    return runs


def compute_player_targets(
    batch: Batch,
    value_log_probs: jax.Array,
    isr: jax.Array,
    config: Porygon2LearnerConfig,
    isr_raw: jax.Array | None = None,
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

    config.player_lambda (0.8, AlphaStar's TD(lambda) value) shapes the
    value targets.

    ``isr`` is the THRESHOLDED ratio (thresholded_target_ratio) and
    ``isr_raw`` the raw one, read only by the telemetry twins so the ESS,
    clip-fraction and trace-length panels keep a comparable series across
    the restart that introduced the threshold; None means "the same".
    """
    if isr_raw is None:
        isr_raw = isr
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

    # Plain v-trace: the subtracted baseline is V(s). Between 2026-08-25 and
    # 2026-08-29 it was the COMPOSED Q = V(s) + A(s, a), i.e. this residual
    # also subtracted the target critic's advantage at the taken action, so
    # rho attenuated only environment noise rather than the whole residual.
    # That went with the advantage head; the file's own note promised
    # adv_taken = 0 was the exact revert, and this is it.
    td_errors = rho_t * mask * (r_t + discount_t * v_t - v_tm1)

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

    num_actions = batch.player_transitions.env_output.action_mask.sum(axis=-1)
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
    channel_logs = {}
    for suffix, ratio in (("", isr), ("_raw", isr_raw)):
        ratio_mean = ratio.mean(where=policy_mask)
        ratio_sq_mean = jnp.square(ratio).mean(where=policy_mask)
        channel_logs[f"player_isr_ess{suffix}"] = (
            ratio_mean * ratio_mean / (ratio_sq_mean + 1e-8)
        )
        channel_logs[f"player_rho_clip_frac{suffix}"] = (ratio > 1.0).mean(
            where=policy_mask
        )
        # Realised trace length: rows the continuation survives from each
        # policy row -- the game's end (discount 0) ends it for both, a
        # zeroed ratio ends it for the thresholded one only.
        channel_logs[f"player_trace_len_mean{suffix}"] = average(
            trace_run_length((ratio > 0.0) & (discount_t > 0.0)), policy_mask
        )

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
