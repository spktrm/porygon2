import jax
import jax.numpy as jnp

from rl.environment.data import CAT_VF_SUPPORT
from rl.environment.interfaces import BuilderTargets, PlayerTargets, Trajectory


def jax_segmented_cumsum(x: jnp.ndarray, discount: jnp.ndarray) -> jnp.ndarray:
    """
    Parallel implementation of your @torch.jit.script loop.
    Replaces O(T) sequential loop with O(log T) associative scan.
    """

    def binary_op(right, left):
        # The scan operator for: y[t] = x[t] + discount[t] * y[t+1]
        val_l, disc_l = left
        val_r, disc_r = right
        return val_l + disc_l * val_r, disc_l * disc_r

    # Flip to treat the 'future' as the prefix for the scan
    vals, _ = jax.lax.associative_scan(binary_op, (x[::-1], discount[::-1]))
    return vals[::-1]


@jax.jit(static_argnames=["td_lambda", "gae_lambda"])
def compute_player_targets(
    traj: Trajectory,
    td_lambda: float,
    gae_lambda: float,
) -> PlayerTargets:
    """Compute TD(λ) returns and GAE advantages for the player trajectory.

    Called once when the trajectory is added to the replay buffer so that
    these targets do not need to be recomputed on every training step.
    """
    cat_vf_support = CAT_VF_SUPPORT

    player_valid = jnp.logical_not(traj.player_transitions.env_output.done)  # (T,)
    player_reward = traj.player_transitions.env_output.win_reward.astype(
        jnp.float32
    )  # (T, 3)
    player_value_probs = jnp.exp(
        traj.player_transitions.agent_output.actor_output.value_head.log_probs.astype(
            jnp.float32
        )
    )  # (T, 3)

    player_next_value_probs = jnp.concatenate(
        [player_value_probs[1:], player_value_probs[-1:]], axis=0
    )
    player_value_target = (
        player_reward + player_next_value_probs * player_valid[..., None]
    )
    player_value_delta = player_value_target - player_value_probs  # (T, 3)
    player_scalar_delta = player_value_delta @ cat_vf_support  # (T,)

    td_lambdas = (td_lambda * player_valid).astype(jnp.float32)  # (T,)
    gae_lambdas = (gae_lambda * player_valid).astype(jnp.float32)  # (T,)

    returns = (
        jax_segmented_cumsum(player_value_delta, td_lambdas[..., None])
        + player_value_probs
    )  # (T, 3)
    advantages = jax_segmented_cumsum(player_scalar_delta, gae_lambdas)  # (T,)

    return PlayerTargets(
        returns=returns.astype(jnp.float32),
        advantages=advantages.astype(jnp.float32),
    )


@jax.jit(static_argnames=["td_lambda", "gae_lambda", "entropy_normalising_constant"])
def compute_builder_targets(
    traj: Trajectory,
    td_lambda: float,
    gae_lambda: float,
    entropy_normalising_constant: float,
) -> BuilderTargets:
    """Compute TD(λ) returns and GAE advantages for the builder trajectory.

    The builder reward is derived from the player's final win/loss/tie reward,
    so the full Trajectory (containing both builder and player data) is required.
    The entropy temperature, which changes over training, is intentionally *not*
    applied here; raw_ent_advantages must be scaled in train_step.
    """
    cat_vf_support = CAT_VF_SUPPORT
    builder_transitions = traj.builder_transitions

    builder_valid = jnp.logical_not(builder_transitions.env_output.done)  # (T_b,)
    builder_valid.shape[0]

    builder_value_probs = jnp.exp(
        builder_transitions.agent_output.actor_output.value_head.log_probs.astype(
            jnp.float32
        )
    )  # (T_b, 3)
    builder_log_prob = (
        builder_transitions.agent_output.actor_output.action_head.log_prob.astype(
            jnp.float32
        )
    )  # (T_b,)
    builder_ent_pred = builder_transitions.agent_output.actor_output.conditional_entropy_head.logits.astype(
        jnp.float32
    )  # (T_b,)

    # Place the final player reward at the first terminal position of the builder.
    final_reward = traj.player_transitions.env_output.win_reward[-1].astype(
        jnp.float32
    )  # (3,)

    builder_reward = (
        jax.nn.one_hot(builder_valid.sum(axis=0), builder_valid.shape[0], axis=0)[
            ..., None
        ]
        * final_reward
    )

    # Entropy delta: NLL + discounted future entropy - current entropy prediction.
    builder_ent_scaled = builder_ent_pred * entropy_normalising_constant  # (T_b,)
    next_builder_ent_scaled = (
        jnp.concatenate(
            [builder_ent_scaled[1:], jnp.zeros_like(builder_ent_scaled[:1])], axis=0
        )
        * builder_valid
    )
    builder_nll = -builder_log_prob  # (T_b,)
    builder_ent_delta = builder_nll + next_builder_ent_scaled - builder_ent_scaled

    # Value computation.
    builder_next_value_probs = jnp.concatenate(
        [builder_value_probs[1:], builder_value_probs[-1:]], axis=0
    )
    builder_value_target = (
        builder_reward + builder_next_value_probs * builder_valid[..., None]
    )
    builder_value_delta = builder_value_target - builder_value_probs  # (T_b, 3)

    td_lambdas = (td_lambda * builder_valid).astype(jnp.float32)  # (T_b,)
    gae_lambdas = (gae_lambda * builder_valid).astype(jnp.float32)  # (T_b,)

    returns = (
        jax_segmented_cumsum(builder_value_delta, td_lambdas[..., None])
        + builder_value_probs
    )  # (T_b, 3)
    win_advantages = (
        jax_segmented_cumsum(builder_value_delta, gae_lambdas[..., None])
        @ cat_vf_support
    )  # (T_b,)
    ent_returns = (
        jax_segmented_cumsum(builder_ent_delta, td_lambdas) + builder_ent_scaled
    ) / entropy_normalising_constant  # (T_b,)
    raw_ent_advantages = jax_segmented_cumsum(builder_ent_delta, gae_lambdas)  # (T_b,)

    return BuilderTargets(
        returns=returns.astype(jnp.float32),
        win_advantages=win_advantages.astype(jnp.float32),
        raw_ent_advantages=raw_ent_advantages.astype(jnp.float32),
        ent_returns=ent_returns.astype(jnp.float32),
    )
