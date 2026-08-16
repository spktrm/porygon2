from typing import Any, TypeVar

import chex
import jax
import jax.numpy as jnp

from rl.environment.data import (
    ALLY_SWITCH_INDICES,
    CAT_VF_SUPPORT,
    NUM_PACKED_SET_FEATURES,
    RESERVE_ENTITY_INDICES,
)
from rl.environment.interfaces import Trajectory
from rl.environment.protos.features_pb2 import (
    FieldFeature,
    InfoFeature,
    PackedSetFeature,
)
from rl.environment.protos.service_pb2 import ActionEnum
from rl.online.config import Porygon2LearnerConfig

T = TypeVar("T")


def promote_map(tree: T, dtype) -> T:
    return jax.tree.map(lambda x: x.astype(dtype), tree)


def renormalize(loss: jax.Array, mask: jax.Array) -> jax.Array:
    """The `normalization` is the number of steps over which loss is computed."""
    chex.assert_equal_shape((loss, mask))
    loss = jnp.sum(loss * mask)
    normalization = jnp.sum(mask)
    return loss / (normalization + (normalization == 0.0))


def collect_batch_telemetry_data(
    batch: Trajectory, config: Porygon2LearnerConfig
) -> dict[str, Any]:
    done = batch.player_transitions.env_output.done
    player_valid = 1 - (jnp.cumsum(done, axis=0) - done)
    player_lengths = player_valid.sum(0)

    history_lengths = batch.player_history.field[
        ..., FieldFeature.FIELD_FEATURE__VALID
    ].sum(0)

    can_move = batch.player_transitions.env_output.action_mask[
        ...,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_1 : ActionEnum.ACTION_ENUM__ALLY_2_MOVE_4_WILDCARD
        + 1,
        :,
    ].any((-2, -1))
    can_wildcard = (
        batch.player_transitions.env_output.action_mask[
            ...,
            ActionEnum.ACTION_ENUM__ALLY_1_MOVE_1_WILDCARD : ActionEnum.ACTION_ENUM__ALLY_1_MOVE_4_WILDCARD
            + 1,
            :,
        ].any((-2, -1))
    ) | (
        batch.player_transitions.env_output.action_mask[
            ...,
            ActionEnum.ACTION_ENUM__ALLY_2_MOVE_1_WILDCARD : ActionEnum.ACTION_ENUM__ALLY_2_MOVE_4_WILDCARD
            + 1,
            :,
        ].any((-2, -1))
    )
    can_switch = batch.player_transitions.env_output.action_mask[
        ...,
        ALLY_SWITCH_INDICES,
        :,
    ].any((-2, -1))
    can_act = can_move & can_switch & player_valid

    src_action_index = (
        batch.player_transitions.agent_output.actor_output.action_head.src_index
    )
    tgt_action_index = (
        batch.player_transitions.agent_output.actor_output.action_head.tgt_index
    )
    did_move = (
        (src_action_index >= ActionEnum.ACTION_ENUM__ALLY_1_MOVE_1)
        & (src_action_index <= ActionEnum.ACTION_ENUM__ALLY_2_MOVE_4_WILDCARD)
        & can_move
    )
    did_wildcard = (
        (
            (src_action_index >= ActionEnum.ACTION_ENUM__ALLY_1_MOVE_1_WILDCARD)
            & (src_action_index <= ActionEnum.ACTION_ENUM__ALLY_1_MOVE_4_WILDCARD)
        )
        | (
            (src_action_index >= ActionEnum.ACTION_ENUM__ALLY_2_MOVE_1_WILDCARD)
            & (src_action_index <= ActionEnum.ACTION_ENUM__ALLY_2_MOVE_4_WILDCARD)
        )
    ) & can_move
    did_switch = (
        (src_action_index[..., None] == ALLY_SWITCH_INDICES[None, None]).any(axis=-1)
        & (tgt_action_index[..., None] == RESERVE_ENTITY_INDICES[None, None]).any(
            axis=-1
        )
        & can_switch
    )
    move_ratio = renormalize(did_move, can_act)
    switch_ratio = renormalize(did_switch, can_act)

    wildcard_turn = jnp.where(
        did_move & did_wildcard,
        jnp.arange(player_valid.shape[0], dtype=jnp.int32)[:, None],
        player_valid.shape[0],
    ).min(axis=0)

    final_reward = batch.player_transitions.env_output.win_reward[-1]
    player_value_expectation = (
        batch.player_transitions.agent_output.actor_output.value_head.expectation
    )
    early_valid_length = 5

    # Chunked unrolls: only a game's terminal chunk carries the outcome at
    # win_reward[-1] — outcome-derived stats read those columns only, and
    # "early game" means early REQUESTS, not a chunk's first rows.
    is_terminal_chunk = done.any(axis=0).astype(player_valid.dtype)  # (B,)
    request_counts = batch.player_transitions.env_output.info[
        ..., InfoFeature.INFO_FEATURE__REQUEST_COUNT
    ]
    early_rows = (request_counts < early_valid_length).astype(player_valid.dtype)

    # History-window coverage: fraction of in-game rows whose request
    # precedes the trailing window's first valid token — those rows read
    # the h0 initial state instead of real context. Only a FULL window can
    # have dropped tokens; a part-filled one covers the game from its
    # start. Sustained non-zero here means player_history_length is too
    # small for player_chunk_length.
    field_valid = (
        batch.player_history.field[..., FieldFeature.FIELD_FEATURE__VALID] > 0
    )
    field_requests = batch.player_history.field[
        ..., FieldFeature.FIELD_FEATURE__REQUEST_COUNT
    ]
    window_first_request = jnp.where(
        field_valid, field_requests, jnp.iinfo(jnp.int32).max
    ).min(axis=0)
    window_full = history_lengths >= config.player_history_length
    history_underrun = (
        (request_counts < window_first_request[None, :])
        & window_full[None, :]
        & (player_valid > 0)
    )

    telemetry = dict(
        player_trajectory_length_mean=player_lengths.mean(),
        player_trajectory_length_min=player_lengths.min(),
        player_trajectory_length_max=player_lengths.max(),
        player_trajectory_shape=player_valid.shape[0],
        history_lengths_mean=history_lengths.mean(),
        move_ratio=move_ratio,
        switch_ratio=switch_ratio,
        wildcard_turn=wildcard_turn.mean(),
        player_chunk_terminal_frac=is_terminal_chunk.mean(),
        player_chunk_history_underrun=renormalize(
            history_underrun.astype(player_valid.dtype), player_valid
        ),
        # Whole-game length, read off the terminal chunk's done row (its
        # REQUEST_COUNT/TURN are game totals). The only game-length signal
        # since chunking made trajectory_length chunk-local — and the
        # distribution to watch now that the service's 96-request force-tie
        # is gone (chunked-unrolls change, 2026-08-16).
        game_length_requests_mean=renormalize(
            (request_counts * done).max(axis=0).astype(jnp.float32),
            is_terminal_chunk,
        ),
        game_length_requests_max=jnp.where(
            is_terminal_chunk.any(),
            (request_counts * done).max(axis=0).max(),
            0,
        ),
        game_length_turns_mean=renormalize(
            (
                batch.player_transitions.env_output.info[
                    ..., InfoFeature.INFO_FEATURE__TURN
                ]
                * done
            )
            .max(axis=0)
            .astype(jnp.float32),
            is_terminal_chunk,
        ),
        reward_mean=renormalize(final_reward @ CAT_VF_SUPPORT, is_terminal_chunk),
        value_expectation_mean=renormalize(player_value_expectation, player_valid),
        value_expectation_early_mean=renormalize(
            player_value_expectation, player_valid * early_rows
        ),
        early_finish_rate=renormalize(
            (jnp.abs(final_reward @ CAT_VF_SUPPORT) < 1).astype(jnp.float32),
            is_terminal_chunk,
        ),
    )

    if config.smogon_format != "randombattle":
        builder_valid = jnp.bitwise_not(batch.builder_transitions.env_output.done)
        builder_lengths = builder_valid.sum(0)

        team_tokens = batch.builder_history.packed_team_member_tokens.reshape(
            -1,
            NUM_PACKED_SET_FEATURES,
            batch.builder_history.packed_team_member_tokens.shape[1],
        )
        team_evs = team_tokens[
            :,
            PackedSetFeature.PACKED_SET_FEATURE__HP_EV : PackedSetFeature.PACKED_SET_FEATURE__SPE_EV
            + 1,
        ]
        ev_prob = team_evs / 128
        ev_entropy = -jnp.sum(ev_prob * jnp.log(ev_prob + 1e-8), axis=-1).mean()

        ev_reward = batch.builder_transitions.env_output.ev_reward[-1].mean()

        telemetry.update(
            dict(
                builder_trajectory_length_mean=builder_lengths.mean(),
                builder_trajectory_length_min=builder_lengths.min(),
                builder_trajectory_length_max=builder_lengths.max(),
                builder_ev_entropy=ev_entropy,
                builder_ev_reward=ev_reward,
            )
        )

    return telemetry


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
