import jax
import jax.numpy as jnp

from rl.environment.data import CAT_VF_SUPPORT, MAX_RATIO_TOKEN
from rl.environment.interfaces import (
    Batch,
    BuilderActorOutput,
    BuilderTargets,
    PlayerTargets,
    Trajectory,
)
from rl.environment.protos.features_pb2 import InfoFeature
from rl.model.utils import prune_log_policy
from rl.online.config import Porygon2LearnerConfig
from rl.online.training.telemetry import ratio_ess_and_tail
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
    """Offline comparison of pruned and raw policy/behaviour ratios.

    Returns both ratios, the taken-action-kept mask and the fraction of
    legal actions removed. Training uses the raw learner policy directly.
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


def unit_potential(env_output) -> jax.Array:
    """The service's position potential (INFO_FEATURE__STATE_POTENTIAL,
    int16 at MAX_RATIO_TOKEN) as a unit-scale float: the human-replay
    outcome fit, in [-1, 1] win-loss units (service/src/server/
    position_potential.ts)."""
    potential = env_output.info[..., InfoFeature.INFO_FEATURE__STATE_POTENTIAL]
    return potential.astype(jnp.float32) / MAX_RATIO_TOKEN


def scalar_vtrace(
    reward: jax.Array,
    value: jax.Array,
    discount_t: jax.Array,
    mask: jax.Array,
    rho_t: jax.Array,
    c_t: jax.Array,
    lambda_: float,
) -> tuple[jax.Array, jax.Array]:
    """Detached V-trace labels and once-weighted actor advantages in f32.

    Lambda shortens trace continuation only. The actor bootstraps directly
    on the next corrected value. A bootstrap-only final row has mask zero
    while its supplied value remains available to the preceding row.
    """
    reward, value, discount_t, mask, rho_t, c_t = jax.tree.map(
        lambda array: array.astype(jnp.float32),
        (reward, value, discount_t, mask, rho_t, c_t),
    )
    value_next = jnp.concatenate([value[1:], value[-1:]], axis=0)
    td_errors = rho_t * mask * (reward + discount_t * value_next - value)
    vtrace_value = vtrace(td_errors, discount_t, c_t * lambda_) + value
    returns = vtrace_value * mask
    q_bootstrap = jnp.concatenate([vtrace_value[1:], value[-1:]], axis=0)
    q_estimate = reward + discount_t * q_bootstrap
    advantages = rho_t * (q_estimate - value) * mask
    return jax.lax.stop_gradient(returns), jax.lax.stop_gradient(advantages)


def compute_player_targets(
    batch: Batch,
    value_log_probs: jax.Array,
    isr: jax.Array,
    config: Porygon2LearnerConfig,
    potential_values: jax.Array | None = None,
) -> tuple[PlayerTargets, dict[str, jax.Array]]:
    """Current-policy V-trace using raw learner/behaviour action ratios.

    Both importance weights are truncated at one. Values may come from
    either critic, but policy ratios always use deployable observations.
    Returned labels and advantages are detached from the learner.
    """
    dones = batch.player_transitions.env_output.done
    mask = (1 - (jnp.cumsum(dones, axis=0) - dones)).astype(jnp.float32)
    discount_t = (1 - dones).astype(jnp.float32) * config.player_gamma * mask

    rho_t = jnp.minimum(1.0, isr).astype(jnp.float32)
    # Terminal rows carry outcomes, not sampled actions to importance-weight.
    rho_t = jnp.where(dones, 1.0, rho_t)
    c_t = rho_t

    # Overlapping chunks train the shared row only in the following chunk.
    is_final_row = jnp.arange(mask.shape[0])[:, None] == mask.shape[0] - 1
    value_mask = mask.astype(jnp.bool_) & (~is_final_row | dones)
    target_mask = value_mask.astype(jnp.float32)

    support = jnp.asarray(CAT_VF_SUPPORT, dtype=jnp.float32)
    r_t = batch.player_transitions.env_output.win_reward.astype(jnp.float32) @ support

    v_tm1 = jnp.exp(value_log_probs.astype(jnp.float32)) @ support

    scalar_returns, pg_advantages = scalar_vtrace(
        r_t, v_tm1, discount_t, target_mask, rho_t, c_t, config.player_lambda
    )

    win_returns = two_hot(scalar_returns, support) * target_mask[..., None]

    # The potential channel's exact value is -Psi under every policy.
    # Its live critic cancels shaping once fitted; terminal potentials and
    # values are zero so they cannot leak into earlier trace residuals.
    potential_returns = ()
    potential_advantages = ()
    if potential_values is not None:
        strength = config.player_potential_strength
        if strength <= 0:
            raise ValueError("potential_values need player_potential_strength > 0")
        live = mask * (1 - dones).astype(jnp.float32)
        psi = strength * unit_potential(batch.player_transitions.env_output) * live
        psi_next = jnp.concatenate([psi[1:], psi[-1:]], axis=0)
        channel_returns, potential_advantages = scalar_vtrace(
            discount_t * psi_next - psi,
            strength * potential_values.astype(jnp.float32) * live,
            discount_t,
            target_mask,
            rho_t,
            c_t,
            config.player_lambda,
        )
        potential_returns = channel_returns / strength
        pg_advantages = pg_advantages + potential_advantages

    num_actions = batch.player_transitions.env_output.action_mask.sum(axis=-1)
    policy_mask = (
        value_mask
        & jnp.logical_not(batch.player_transitions.env_output.done)
        & (num_actions > 1)
    )

    ratio_ess, clipped_fraction = ratio_ess_and_tail(isr, policy_mask, 1.0)
    channel_logs = {
        "player_isr_ess": ratio_ess,
        "player_rho_clip_frac": clipped_fraction,
        "player_trace_len_mean": average(
            trace_run_length((isr > 0.0) & (discount_t > 0.0)), policy_mask
        ),
    }

    return (
        PlayerTargets(
            win_returns=win_returns,
            pg_advantages=pg_advantages,
            policy_mask=policy_mask,
            value_mask=value_mask,
            potential_returns=potential_returns,
            potential_advantages=potential_advantages,
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
    """KL(pi || stop(pi_reg)) over legal actions, per state in f32."""

    def normalise_legal(values):
        # Re-normalise after promotion: bf16 log-probabilities can lose unit mass.
        masked = jnp.where(
            legal_mask, values.astype(jnp.float32), jnp.finfo(jnp.float32).min
        )
        return jax.nn.log_softmax(masked, axis=-1)

    policy_log_probs = normalise_legal(log_policy)
    reference_log_probs = normalise_legal(jax.lax.stop_gradient(reg_log_policy))
    policy_probs = jnp.where(legal_mask, jnp.exp(policy_log_probs), 0.0)
    log_ratio = jnp.where(
        legal_mask & (policy_probs > 0.0),
        policy_log_probs - reference_log_probs,
        0.0,
    )
    return (policy_probs * log_ratio).sum(axis=-1)


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

    builder_valid = jnp.logical_not(builder_transitions.env_output.done)
    T_b, B = builder_valid.shape

    rho_t = jnp.minimum(1.0, importance_sampling_ratios)
    c_t = jnp.minimum(1.0, importance_sampling_ratios)

    builder_value_probs = jnp.exp(
        builder_transitions.agent_output.actor_output.value_head.log_probs
    )
    n_bins = builder_value_probs.shape[-1]

    final_reward = traj.player_transitions.env_output.win_reward[-1]
    num_valid_steps = builder_valid.astype(jnp.int32).sum(axis=0)

    builder_reward = jnp.zeros((T_b, B, n_bins), dtype=builder_value_probs.dtype)
    safe_idx = jnp.clip(num_valid_steps, 0, T_b - 1)
    batch_idx = jnp.arange(B)
    has_terminal = num_valid_steps < T_b
    builder_reward = builder_reward.at[safe_idx, batch_idx].set(
        final_reward * has_terminal[:, None]
    )

    builder_log_prob = (
        builder_transitions.agent_output.actor_output.action_head.log_prob
    )
    builder_ent_scaled = (
        builder_transitions.agent_output.actor_output.conditional_entropy_head.logits
        * entropy_normalising_constant
    )
    ent_reward = -builder_log_prob

    combined_rewards = jnp.concatenate([builder_reward, ent_reward[..., None]], axis=-1)

    combined_values = jnp.concatenate(
        [builder_value_probs, builder_ent_scaled[..., None]], axis=-1
    )

    last_values = jnp.concatenate(
        [builder_value_probs[-1:], jnp.zeros_like(builder_ent_scaled[:1])[..., None]],
        axis=-1,
    )

    combined_next_values = (
        jnp.concatenate([combined_values[1:], last_values], axis=0)
        * builder_valid[..., None]
    )

    combined_td_errors = rho_t[..., None] * (
        combined_rewards + combined_next_values - combined_values
    )

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

    win_returns = returns[..., :n_bins]
    ent_returns = returns[..., n_bins]

    return BuilderTargets(
        win_returns=win_returns,
        win_advantages=pg_advantages[..., :n_bins] @ cat_vf_support,
        ent_returns=ent_returns,
        ent_advantages=pg_advantages[..., n_bins],
    )
