import jax
import jax.numpy as jnp

from rl.environment.data import CAT_VF_SUPPORT
from rl.environment.interfaces import BuilderTargets, PlayerTargets, Trajectory
from rl.learner.loss import approx_backward_kl


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
    off_policy_correction: jax.Array,
    td_lambda: float,
    gae_lambda: float,
    entropy_normalising_constant: float,
) -> PlayerTargets:
    """Compute TD(λ) returns and GAE advantages for the player trajectory.

    JAX/JIT compatible.  Operates on batched data with shape (T, B, ...).
    """
    cat_vf_support = jnp.asarray(CAT_VF_SUPPORT, dtype=jnp.float32)

    player_valid = jnp.logical_not(traj.player_transitions.env_output.done)  # (T, B)
    player_reward = traj.player_transitions.env_output.win_reward.astype(
        jnp.float32
    )  # (T, B, 3)
    player_value_probs = jnp.exp(
        traj.player_transitions.agent_output.actor_output.value_head.log_probs.astype(
            jnp.float32
        )
    )  # (T, B, 3)

    player_next_value_probs = jnp.concatenate(
        [player_value_probs[1:], player_value_probs[-1:]], axis=0
    )
    player_value_target = (
        player_reward + player_next_value_probs * player_valid[..., None]
    )
    player_value_delta = (player_value_target - player_value_probs).astype(jnp.float32)
    player_scalar_delta = player_value_delta @ cat_vf_support

    returns_discounts = (off_policy_correction * td_lambda * player_valid).astype(
        jnp.float32
    )  # (T, B)
    policy_discounts = (off_policy_correction * gae_lambda * player_valid).astype(
        jnp.float32
    )  # (T, B)

    returns = (
        segmented_cumsum(player_value_delta, returns_discounts[..., None])
        + player_value_probs
    )  # (T, B, 3)
    advantages = segmented_cumsum(player_scalar_delta, policy_discounts)  # (T, B)

    # Entropy reward (SAC-style)
    player_log_prob = (
        traj.player_transitions.agent_output.actor_output.action_head.log_prob
    )
    player_ent_pred = (
        traj.player_transitions.agent_output.actor_output.conditional_entropy_head.logits
    )

    player_ent_scaled = player_ent_pred * entropy_normalising_constant  # (T, B)
    next_player_ent_scaled = (
        jnp.concatenate(
            [player_ent_scaled[1:], jnp.zeros_like(player_ent_scaled[:1])], axis=0
        )
        * player_valid
    )

    num_valid_actions = traj.player_transitions.env_output.action_mask.sum((-2, -1))
    uniform_log_prob = -jnp.log(num_valid_actions + 1e-8)
    magnet_log_ratio = player_log_prob - uniform_log_prob  # (T, B)
    magnet_ratio = jnp.exp(magnet_log_ratio)
    kl_reward = approx_backward_kl(
        policy_ratio=magnet_ratio, log_policy_ratio=magnet_log_ratio
    )  # (T, B)

    player_ent_delta = (-kl_reward + next_player_ent_scaled - player_ent_scaled).astype(
        jnp.float32
    )

    ent_returns = (
        segmented_cumsum(player_ent_delta, returns_discounts) + player_ent_scaled
    ) / entropy_normalising_constant  # (T, B)

    raw_ent_advantages = segmented_cumsum(player_ent_delta, policy_discounts)  # (T, B)

    # Potential-based shaping
    state_pot = traj.player_transitions.env_output.state_potential.astype(jnp.float32)
    player_state_potential = jnp.concatenate(
        [jnp.zeros_like(state_pot[:1]), state_pot], axis=0
    )  # (T+1, B)

    player_potential_value = traj.player_transitions.agent_output.actor_output.potential_value_head.logits.astype(
        jnp.float32
    )

    potential_reward = (
        player_state_potential[1:] - player_state_potential[:-1]
    )  # (T, B)

    player_next_potential_value = (
        jnp.concatenate(
            [player_potential_value[1:], player_potential_value[-1:]], axis=0
        )
        * player_valid
    )

    player_potential_delta = (
        potential_reward + player_next_potential_value - player_potential_value
    ).astype(jnp.float32)

    potential_returns = (
        segmented_cumsum(player_potential_delta, returns_discounts)
        + player_potential_value
    )
    potential_advantages = segmented_cumsum(player_potential_delta, policy_discounts)

    return PlayerTargets(
        win_returns=returns,
        win_advantages=advantages,
        ent_advantages=raw_ent_advantages,
        ent_returns=ent_returns,
        potential_returns=potential_returns,
        potential_advantages=potential_advantages,
    )


def compute_builder_targets(
    traj: Trajectory,
    off_policy_correction: jax.Array,
    td_lambda: float,
    gae_lambda: float,
    entropy_normalising_constant: float,
) -> BuilderTargets:
    """Compute TD(λ) returns and GAE advantages for the builder trajectory.

    JAX/JIT compatible.  Operates on batched data with shape (T_b, B, ...).
    The builder reward is derived from the player's final win/loss/tie reward
    stored in ``traj.player_final_reward``.
    """
    cat_vf_support = jnp.asarray(CAT_VF_SUPPORT, dtype=jnp.float32)
    builder_transitions = traj.builder_transitions

    builder_valid = jnp.logical_not(builder_transitions.env_output.done)  # (T_b, B)
    T_b = builder_valid.shape[0]
    B = builder_valid.shape[1]

    builder_value_probs = jnp.exp(
        builder_transitions.agent_output.actor_output.value_head.log_probs
    )
    builder_log_prob = (
        builder_transitions.agent_output.actor_output.action_head.log_prob
    )
    builder_ent_pred = (
        builder_transitions.agent_output.actor_output.conditional_entropy_head.logits
    )

    # Terminal reward from the player trajectory.
    final_reward = traj.player_final_reward.astype(jnp.float32)  # (B, 3)

    # Place the final reward at the first terminal position per batch element.
    num_valid_steps = builder_valid.astype(jnp.int32).sum(axis=0)  # (B,)
    builder_reward = jnp.zeros((T_b, B, 3), dtype=jnp.float32)
    safe_idx = jnp.clip(num_valid_steps, 0, T_b - 1)
    batch_idx = jnp.arange(B)
    has_terminal = num_valid_steps < T_b
    builder_reward = builder_reward.at[safe_idx, batch_idx].set(
        final_reward * has_terminal[:, None]
    )

    # Entropy delta
    builder_ent_scaled = builder_ent_pred * entropy_normalising_constant  # (T_b, B)
    next_builder_ent_scaled = jnp.concatenate(
        [builder_ent_scaled[1:], jnp.zeros_like(builder_ent_scaled[:1])], axis=0
    )
    builder_ent_target = -builder_log_prob + next_builder_ent_scaled * builder_valid
    builder_ent_delta = (builder_ent_target - builder_ent_scaled).astype(jnp.float32)

    # Value computation
    builder_next_value_probs = jnp.concatenate(
        [builder_value_probs[1:], builder_value_probs[-1:]], axis=0
    )
    builder_value_target = (
        builder_reward + builder_next_value_probs * builder_valid[..., None]
    )
    builder_value_delta = (builder_value_target - builder_value_probs).astype(
        jnp.float32
    )

    td_lambdas = (off_policy_correction * td_lambda * builder_valid).astype(
        jnp.float32
    )  # (T_b, B)
    gae_lambdas = (off_policy_correction * gae_lambda * builder_valid).astype(
        jnp.float32
    )  # (T_b, B)

    returns = (
        segmented_cumsum(builder_value_delta, td_lambdas[..., None])
        + builder_value_probs
    )  # (T_b, B, 3)
    win_advantages = (
        segmented_cumsum(builder_value_delta, gae_lambdas[..., None]) @ cat_vf_support
    )  # (T_b, B)
    ent_returns = (
        segmented_cumsum(builder_ent_delta, td_lambdas) + builder_ent_scaled
    ) / entropy_normalising_constant  # (T_b, B)
    raw_ent_advantages = segmented_cumsum(builder_ent_delta, gae_lambdas)  # (T_b, B)

    return BuilderTargets(
        win_returns=returns,
        win_advantages=win_advantages,
        ent_advantages=raw_ent_advantages,
        ent_returns=ent_returns,
    )
