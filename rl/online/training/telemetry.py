"""Learner-side telemetry helpers: R2, batch composition counts, and
the dtype promotion the loss path uses.
"""

from typing import Any, NamedTuple, TypeVar

import chex
import jax
import jax.numpy as jnp
import optax

from rl.environment.data import (
    ALLY_SWITCH_INDICES,
    CAT_VF_SUPPORT,
    FLAT_MODALITY_MASK,
    NUM_MODALITY_FEATURES,
    NUM_PACKED_SET_FEATURES,
    RESERVE_ENTITY_INDICES,
)
from rl.environment.interfaces import Trajectory
from rl.environment.protos.features_pb2 import (
    FieldFeature,
    InfoFeature,
    PackedSetFeature,
)
from rl.environment.protos.service_pb2 import ActionEnum, ModalityEnum
from rl.online.config import Porygon2LearnerConfig

T = TypeVar("T")


def promote_map(tree: T, dtype) -> T:
    # Masks stay bool: a bf16 mask makes `average`'s denominator a bf16
    # count, exact only up to 256 rows.
    return jax.tree.map(
        lambda x: x if jnp.issubdtype(x.dtype, jnp.bool_) else x.astype(dtype),
        tree,
    )


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
    field_valid = batch.player_history.field[..., FieldFeature.FIELD_FEATURE__VALID] > 0
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

    ss_residual = jnp.sum((value_target - value_prediction) ** 2, where=mask)

    mean_target = jnp.mean(value_target, where=mask)
    ss_total = jnp.sum((value_target - mean_target) ** 2, where=mask)

    return 1 - (ss_residual / (ss_total + eps))


# Matched-V bins for the critic-offset panels: FIXED edges (static shapes,
# no data-derived quantiles inside the jit). Equal-width over the
# CAT_VF_SUPPORT range; the offline reference used V-quantiles, whose
# outer bins map onto these outer two at this checkpoint age.
MATCHED_V_EDGES = (-1.0, -0.6, -0.2, 0.2, 0.6, 1.0 + 1e-6)


def _get(tree, path):
    for key in path:
        tree = tree[key]
    return tree


# The action readout's leaves, and what each must DO.
#
# These panels are not decoration. The dx65cpwp micro runaway lived entirely
# in head params -- micro_local_tgt 0.0028 -> 0.070 rms, the adapter out
# kernel 0.0058 -> 0.105 -- and was invisible on wandb; diagnosing it needed
# checkpoint forensics. The head that grew those numbers is gone, but the
# flat readout has its own way to fail and it is the same shape: the bilinear
# is a two-factor product with ONE zero-init factor, and CLAUDE.md 13 records
# a learned grid behind a zero-init scale sitting at lecun init for 60k steps.
#
# Expected at init and what to watch:
#   query        0, must leave 0 within ~200 steps (its gradient is a rank-1
#                outer product of live rows, so it moves at step 1)
#   key          lecun 0.0625 at fan-in 256; its gradient is proportional to
#                query, so it is frozen for exactly one step and must then
#                drift. Still 0.0625 at 2k = the stall.
#   local_src    0, must leave 0 from step 1. This is also where a per-
#   local_tgt    MODALITY force lives now that the macro head is gone
#                (modality is a function of the src half), so a flat
#                local_src beside a failing entropy_macro floor is the
#                signal to promote it to an MLP.
#   switch/other 0, single-factor, must leave 0 from step 1.
#
# src/tgt stay SPLIT deliberately: the 7.5x tgt-over-src growth asymmetry --
# a tgt column is read by every legal move cell of a row -- was itself the
# dx65cpwp diagnostic.
_ACTION_HEAD_LEAVES = {
    "player_pointer_query_rms": (("action_head", "query", "kernel"),),
    "player_pointer_key_rms": (("action_head", "key", "kernel"),),
    "player_pointer_local_src_rms": (("action_head", "local_src", "kernel"),),
    "player_pointer_local_tgt_rms": (("action_head", "local_tgt", "kernel"),),
    "player_switch_head_rms": (("action_head", "switch", "kernel"),),
    "player_other_head_rms": (("action_head", "other", "kernel"),),
}
# Trunk leaves carry a leading axis of cfg.trunk.num_blocks (nn.scan stacks
# them), so an rms over the whole leaf is the across-block mean by
# construction -- which is what we want: a per-block panel would be six lines
# saying the same thing until one block diverges, and the rms catches that.
_TRUNK_LEAVES = {
    "player_trunk_attn_out_rms": (
        ("encoder", "trunk", "blocks", "attention", "out_proj", "kernel"),
    ),
    "player_trunk_mlp_out_rms": (
        ("encoder", "trunk", "blocks", "ffw", "Dense_1", "kernel"),
    ),
}
_GRAD_SUBTREES = {
    "player_action_head_grad_norm": ("action_head",),
    "player_trunk_grad_norm": ("encoder", "trunk"),
}


def head_param_telemetry(params, grads) -> dict[str, jax.Array]:
    """Learner-side readouts of the action readout and the trunk actually
    learning: rms of each head leaf against its known init, and pre-clip
    grad norms per subtree. `params`/`grads` are the flax variable dicts
    (top-level "params" collection)."""
    p, g = params["params"], grads["params"]
    logs = {}
    for key, paths in {**_ACTION_HEAD_LEAVES, **_TRUNK_LEAVES}.items():
        leaves = [jnp.asarray(_get(p, path), jnp.float32) for path in paths]
        logs[key] = jnp.mean(
            jnp.stack([jnp.sqrt(jnp.mean(jnp.square(x))) for x in leaves])
        )
    for key, path in _GRAD_SUBTREES.items():
        logs[key] = optax.global_norm(_get(g, path))
    return logs


def masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
    """Mean over mask, NaN when the mask is empty (wandb skips NaN points;
    a 0.0 would read as a measurement — the player_q_calibration_r2_fresh
    lesson of 2026-08-23)."""
    return jnp.where(mask.any(), jnp.mean(x, where=mask), jnp.nan)


def masked_var(x: jax.Array, mask: jax.Array) -> jax.Array:
    m = jnp.mean(x, where=mask)
    return jnp.where(mask.sum() >= 2, jnp.mean(jnp.square(x - m), where=mask), jnp.nan)


def masked_r2(pred: jax.Array, target: jax.Array, mask: jax.Array) -> jax.Array:
    """R² that is NaN, not -1e8, when the target is (near-)constant on the
    slice: a 4-chunk batch whose valid outcomes are all +1 has ss_total 0,
    and calculate_r2's eps then produced -5e8 rows that dominated every
    mean on the first Step-2 run."""
    m = mask.sum() >= 2
    mean_t = jnp.mean(target, where=mask)
    ss_total = jnp.sum(jnp.square(target - mean_t), where=mask)
    return jnp.where(m & (ss_total > 1e-4), calculate_r2(pred, target, mask), jnp.nan)


class ActionAxisMasks(NamedTuple):
    """The switch/move row and cell predicates, derived ONCE.

    This block used to be written out three times — twice in train_step
    (the Q diagnostics and again inside player_loss_fn) and once here — and
    the copies had DRIFTED. `has_both` required a legal switch AND a legal
    MOVE; the policy-loss copy required a legal switch and any legal NON-switch,
    which also admits WILDCARD / OTHER / TARGET cells. Both called
    themselves "a switch and a non-switch are both legal", so the
    `player_policy_*` and `player_q_*` families were scoped to different row
    populations while CLAUDE.md 3's decision rule reads one against the
    other (`absadv_ratio` against `player_q_switch_target_frac`).

    Unified 2026-08-25 on the STRICT reading: a stay/switch decision only
    means something when staying and attacking is actually available, so a
    row offering {switch, pass} is not a real choice. This narrows the
    policy slice; the Q slice is unchanged.

    Row predicates are returned bare — each consumer combines them with its
    own row mask (`acted_mask` for the outcome panels, `policy_mask` for
    the policy loss), because those differ deliberately: policy_mask drops
    forced single-option rows, acted_mask keeps them.
    """

    switch_cells: jax.Array
    move_cells: jax.Array
    valid_switch: jax.Array
    valid_move: jax.Array
    has_switch: jax.Array
    has_move: jax.Array
    has_both: jax.Array
    taken_switch: jax.Array
    # Factorised-objective predicates (2026-08-27). taken_modality is the
    # M-way modality index of the taken action; num_legal_modalities and
    # taken_modality_count feed the per-level row masks: the macro bracket
    # acts only where >= 2 modalities are live (a broader predicate than
    # the strict has_both above — WILDCARD/OTHER count as real macro
    # alternatives), the micro bracket only where the TAKEN modality has
    # >= 2 legal cells (a singleton's conditional is deterministic: its
    # ratio is exactly 1 and its entropy exactly 0).
    taken_modality: jax.Array
    num_legal_modalities: jax.Array
    taken_modality_count: jax.Array


def action_axis_masks(
    flat_action_mask: jax.Array, action_index: jax.Array
) -> ActionAxisMasks:
    """See ActionAxisMasks. `has_both` is THE real-choice predicate."""
    flat_modality = jnp.asarray(FLAT_MODALITY_MASK)
    switch_cells = flat_modality == ModalityEnum.MODALITY_ENUM__SWITCH
    move_cells = flat_modality == ModalityEnum.MODALITY_ENUM__MOVE
    valid_switch = flat_action_mask & switch_cells
    valid_move = flat_action_mask & move_cells
    has_switch = valid_switch.any(axis=-1)
    has_move = valid_move.any(axis=-1)
    modality_oh = jax.nn.one_hot(flat_modality, NUM_MODALITY_FEATURES, dtype=jnp.int32)
    legal_per_modality = (flat_action_mask[..., None] * modality_oh).sum(axis=-2)
    taken_modality = jnp.take(flat_modality, action_index)
    return ActionAxisMasks(
        switch_cells=switch_cells,
        move_cells=move_cells,
        valid_switch=valid_switch,
        valid_move=valid_move,
        has_switch=has_switch,
        has_move=has_move,
        has_both=has_switch & has_move,
        taken_switch=jnp.take(switch_cells, action_index),
        taken_modality=taken_modality,
        num_legal_modalities=(legal_per_modality > 0).sum(axis=-1),
        taken_modality_count=jnp.take_along_axis(
            legal_per_modality, taken_modality[..., None], axis=-1
        ).squeeze(-1),
    )


def critic_outcome_telemetry(
    *,
    game_outcome: jax.Array,
    game_length: jax.Array,
    game_step_offset: jax.Array,
    v_target: jax.Array,
    flat_action_mask: jax.Array,
    masks: ActionAxisMasks,
    acted_mask: jax.Array,
    value_mask: jax.Array,
) -> dict[str, jax.Array]:
    """Step-1 panels of docs/critic-weakness-analysis.md — the per-row
    JOINT statistics wandb's pooled means could not give, computed from
    the completed-game outcome carried on every chunk (Trajectory.
    game_outcome). Shapes: game_* (1, B); v_target / acted_mask /
    value_mask (T, B); flat_action_mask (T, B, A). Every panel is NaN,
    not 0, when its slice is empty in this batch.

    The label-variance panels and the CRITIC half of the matched-V table
    retired with the advantage head on 2026-08-29, and the one-step-label
    panels (v_onestep_r2, q_target_edge_frac) with the last of the Q
    machinery on 2026-08-30. What is left never needed either: the
    REALISED outcome gap, a property of the games, not of any critic.
    - mv_bin{i}_*: matched-V table on real-choice rows (a move and a
      switch both legal), binned by the target V head's own V(s):
      realised outcome of voluntary switches vs moves, and counts
      (per-batch n is small; SE comes from n summed over a window).
    - v_outcome_r2_{all,early,mid,late,prev_switch,prev_move}: the V
      head against the realised outcome (offline reference 0.265),
      split by game phase and by whether the PREVIOUS row's action was
      a switch (row 0 of a chunk has no local predecessor: excluded).
    - {vol,forced}_switch_rows / chunk_vol_switch_frac: row- and
      storage-level voluntary-switch supply (renamed off the
      player_q_support_* prefix 2026-08-30).
    """
    f32 = jnp.float32
    T = v_target.shape[0]
    G = jnp.broadcast_to(game_outcome.astype(f32), v_target.shape)
    valid_g = jnp.isfinite(G)
    G = jnp.where(valid_g, G, 0.0)
    v_target = v_target.astype(f32)

    vol_mask = acted_mask & masks.taken_switch & masks.has_move
    forced_mask = acted_mask & masks.taken_switch & jnp.logical_not(masks.has_move)

    logs: dict[str, jax.Array] = {}
    rows = acted_mask & valid_g & masks.has_both
    for i, (lo, hi) in enumerate(zip(MATCHED_V_EDGES[:-1], MATCHED_V_EDGES[1:])):
        b = rows & (v_target >= lo) & (v_target < hi)
        bv = b & masks.taken_switch
        bm = b & jnp.logical_not(masks.taken_switch)
        g_vol = masked_mean(G, bv)
        g_move = masked_mean(G, bm)
        logs[f"player_mv_bin{i}_n_vol"] = bv.sum().astype(f32)
        logs[f"player_mv_bin{i}_n_move"] = bm.sum().astype(f32)
        logs[f"player_mv_bin{i}_g_vol"] = g_vol
        logs[f"player_mv_bin{i}_g_move"] = g_move
        logs[f"player_mv_bin{i}_gap_realised"] = g_vol - g_move
    logs["player_mv_pooled_gap_realised"] = masked_mean(
        G, rows & masks.taken_switch
    ) - masked_mean(G, rows & jnp.logical_not(masks.taken_switch))
    logs["player_mv_v_at_vol_switch"] = masked_mean(v_target, rows & masks.taken_switch)
    logs["player_mv_v_at_move"] = masked_mean(
        v_target, rows & jnp.logical_not(masks.taken_switch)
    )

    vm = value_mask & valid_g
    t_idx = jnp.arange(T, dtype=f32)[:, None]
    phase = (game_step_offset.astype(f32) + t_idx) / jnp.maximum(
        game_length.astype(f32), 1.0
    )
    logs["player_v_outcome_r2_all"] = masked_r2(v_target, G, vm)
    logs["player_v_outcome_r2_early"] = masked_r2(v_target, G, vm & (phase < 1 / 3))
    logs["player_v_outcome_r2_mid"] = masked_r2(
        v_target, G, vm & (phase >= 1 / 3) & (phase < 2 / 3)
    )
    logs["player_v_outcome_r2_late"] = masked_r2(v_target, G, vm & (phase >= 2 / 3))

    # Previous-row action, split forced / voluntary: after a FORCED switch
    # (a mon just fainted) V read +0.23 optimistic on the collapsed
    # baseline, while the offline post-VOLUNTARY-switch read was
    # pessimistic — two populations, two panels. Row 0 has no local
    # predecessor and is excluded.
    def shift(x):
        return jnp.concatenate([jnp.zeros_like(x[:1]), x[:-1]], axis=0)

    prev_switch = shift(masks.taken_switch)
    prev_forced = shift(masks.taken_switch & jnp.logical_not(masks.has_move))
    prev_voluntary = shift(masks.taken_switch & masks.has_move)
    known_prev = (t_idx >= 1) & jnp.ones_like(masks.taken_switch)
    for name, m in (
        ("prev_switch", prev_switch),
        ("prev_forced", prev_forced),
        ("prev_voluntary", prev_voluntary),
        ("prev_move", jnp.logical_not(prev_switch)),
    ):
        logs[f"player_v_outcome_r2_{name}"] = masked_r2(
            v_target, G, vm & known_prev & m
        )
        logs[f"player_v_outcome_bias_{name}"] = masked_mean(
            v_target - G, vm & known_prev & m
        )

    logs["player_chunk_vol_switch_frac"] = vol_mask.any(axis=0).astype(f32).mean()
    logs["player_vol_switch_rows"] = vol_mask.sum().astype(f32)
    logs["player_forced_switch_rows"] = forced_mask.sum().astype(f32)
    return logs
