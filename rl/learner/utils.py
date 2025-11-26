import functools
from typing import Any, Dict

import chex
import jax
import jax.numpy as jnp

from rl.environment.interfaces import BuilderTrajectory
from rl.environment.protos.features_pb2 import FieldFeature
from rl.environment.protos.service_pb2 import WildCardEnum
from rl.learner.config import PlayerLearnerConfig


def renormalize(loss: jax.Array, mask: jax.Array) -> jax.Array:
    """The `normalization` is the number of steps over which loss is computed."""
    chex.assert_equal_shape((loss, mask))
    loss = jnp.sum(loss * mask)
    normalization = jnp.sum(mask)
    return loss / (normalization + (normalization == 0.0))


@functools.partial(jax.jit, static_argnames=("config"))
def collect_batch_telemetry_data(
    batch: BuilderTrajectory, config: PlayerLearnerConfig
) -> Dict[str, Any]:
    builder_valid = jnp.bitwise_not(batch.builder_transitions.env_output.done)
    builder_lengths = builder_valid.sum(0)

    player_valid = jnp.bitwise_not(batch.transitions.env_output.done)
    player_lengths = player_valid.sum(0)

    history_lengths = batch.history.field[..., FieldFeature.FIELD_FEATURE__VALID].sum(0)

    can_move = batch.transitions.env_output.action_mask[..., :4].any(-1)
    can_switch = batch.transitions.env_output.action_mask[..., 4:].any(-1)
    can_act = can_move & can_switch & player_valid

    action_index = batch.transitions.agent_output.actor_output.action_head.action_index
    wildcard_index = (
        batch.transitions.agent_output.actor_output.wildcard_head.action_index
    )
    did_move = (action_index < 4) & can_move
    did_switch = (action_index >= 4) & can_switch
    move_ratio = renormalize(did_move, can_act)
    switch_ratio = renormalize(did_switch, can_act)

    wildcard_turn = jnp.where(
        did_move & (wildcard_index != WildCardEnum.WILD_CARD_ENUM__CAN_NORMAL),
        jnp.arange(player_valid.shape[0], dtype=jnp.int32)[:, None],
        player_valid.shape[0],
    ).min(axis=0)

    final_reward = batch.transitions.env_output.win_reward[-1]

    return dict(
        player_trajectory_length_mean=player_lengths.mean(),
        player_trajectory_length_min=player_lengths.min(),
        player_trajectory_length_max=player_lengths.max(),
        builder_trajectory_length_mean=builder_lengths.mean(),
        builder_trajectory_length_min=builder_lengths.min(),
        builder_trajectory_length_max=builder_lengths.max(),
        builder_species_reward_sum=batch.builder_transitions.env_output.cum_species_reward[
            -1
        ].mean(),
        builder_teammate_reward_sum=batch.builder_transitions.env_output.cum_teammate_reward[
            -1
        ].mean(),
        history_lengths_mean=history_lengths.mean(),
        move_ratio=move_ratio,
        switch_ratio=switch_ratio,
        wildcard_turn=wildcard_turn.mean(),
        reward_mean=final_reward.mean(),
        early_finish_rate=(jnp.abs(final_reward) < 1).astype(jnp.float32).mean(),
    )


def calculate_r2(
    value_prediction: jax.Array,
    value_target: jax.Array,
    mask: jax.Array = None,
    eps: float = 1e-8,
) -> jax.Array:
    """Calculate the R-squared (coefficient of determination) value."""

    if mask is None:
        mask = jnp.ones_like(value_prediction)

    # Calculate residual sum of squares (SS_residual)
    ss_residual = jnp.sum((value_target - value_prediction) ** 2, where=mask)

    # Calculate total sum of squares (SS_total)
    mean_target = jnp.mean(value_target, where=mask)
    ss_total = jnp.sum((value_target - mean_target) ** 2, where=mask)

    return 1 - (ss_residual / (ss_total + eps))
