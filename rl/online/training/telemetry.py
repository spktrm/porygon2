"""Learner-side telemetry helpers: R2, batch composition counts, and
the dtype promotion the loss path uses.
"""

from typing import Any, NamedTuple, TypeVar

import chex
import jax
import jax.numpy as jnp
import optax

from rl.environment.data import (
    CAT_VF_SUPPORT,
    CELL_MODALITY_MASK,
    MOVE_CELL_OFFSET,
    NUM_MODALITY_FEATURES,
    NUM_PACKED_SET_FEATURES,
    NUM_SWITCH_CELLS,
    OTHER_CELL_OFFSET,
)
from rl.environment.interfaces import Trajectory
from rl.environment.protos.features_pb2 import (
    FieldFeature,
    InfoFeature,
    PackedSetFeature,
)
from rl.environment.protos.service_pb2 import ModalityEnum
from rl.model.constants import SequenceGroup
from rl.model.state_features import (
    STATE_KERNEL_GROUPS,
    STATE_KERNELS,
    state_kernel_blocks,
)
from rl.online.config import Porygon2LearnerConfig
from rl.utils import average

T = TypeVar("T")


def promote_map(tree: T, dtype) -> T:
    # Masks stay bool: a bf16 mask makes `average`'s denominator a bf16
    # count, exact only up to 256 rows.
    def promote_leaf(leaf):
        if jnp.issubdtype(leaf.dtype, jnp.bool_):
            return leaf
        return leaf.astype(dtype)

    return jax.tree.map(promote_leaf, tree)


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
        ..., MOVE_CELL_OFFSET:OTHER_CELL_OFFSET
    ].any(-1)
    can_switch = batch.player_transitions.env_output.action_mask[
        ..., :NUM_SWITCH_CELLS
    ].any(-1)
    can_act = can_move & can_switch & player_valid

    action_index = (
        batch.player_transitions.agent_output.actor_output.action_head.action_index
    )
    taken_cell_modality = jnp.take(jnp.asarray(CELL_MODALITY_MASK), action_index)
    did_move = (
        (taken_cell_modality == ModalityEnum.MODALITY_ENUM__MOVE)
        | (taken_cell_modality == ModalityEnum.MODALITY_ENUM__WILDCARD)
    ) & can_move
    did_wildcard = (
        taken_cell_modality == ModalityEnum.MODALITY_ENUM__WILDCARD
    ) & can_move
    did_switch = (action_index < NUM_SWITCH_CELLS) & can_switch
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
    is_terminal_chunk = done.any(axis=0).astype(player_valid.dtype)
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
        # since chunking made trajectory_length chunk-local, and the
        # distribution to watch: games run to their natural length.
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


def potential_telemetry(
    potential: jax.Array,
    potential_advantages: jax.Array | tuple,
    pg_advantages: jax.Array,
    policy_mask: jax.Array,
    value_mask: jax.Array,
    voluntary_switch_mask: jax.Array,
    move_mask: jax.Array,
) -> dict[str, jax.Array]:
    """The position potential (2026-09-11, unit scale, eta-free) and -- when
    the PBRS channel runs -- its part of the actor's advantage.

    The switch delta Phi(t+1) - Phi(t) on voluntary switches is DESCRIPTIVE:
    self-play need not reproduce the human-replay sign (-0.038 mean). The
    channel's advantage share should FALL as the potential head fits (the
    head's lag is the shaping); a floor is the unfitted part persisting.
    """
    mean = average(potential, value_mask)
    following = jnp.concatenate([potential[1:], potential[-1:]], axis=0)
    has_following = jnp.arange(potential.shape[0])[:, None] < potential.shape[0] - 1
    logs = {
        "player_potential_mean": mean,
        "player_potential_std": jnp.sqrt(
            average(jnp.square(potential - mean), value_mask)
        ),
        "player_potential_switch_delta_mean": average(
            following - potential, voluntary_switch_mask & has_following
        ),
    }
    if isinstance(potential_advantages, tuple):
        return logs

    def centred(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        values = values.astype(jnp.float32)
        deviation = values - average(values, policy_mask)
        return deviation, jnp.sqrt(average(jnp.square(deviation), policy_mask))

    channel = potential_advantages.astype(jnp.float32)
    channel_deviation, channel_std = centred(channel)
    win_deviation, win_std = centred(pg_advantages - channel)
    _, total_std = centred(pg_advantages)
    logs["player_potential_adv_share"] = channel_std / (total_std + 1e-8)
    logs["player_potential_win_adv_corr"] = average(
        channel_deviation * win_deviation, policy_mask
    ) / (channel_std * win_std + 1e-8)
    logs["player_potential_adv_switch"] = average(channel, voluntary_switch_mask)
    logs["player_potential_adv_move"] = average(channel, move_mask)
    return logs


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


def _has(tree, path) -> bool:
    """Whether `path` names a leaf or subtree of the variable dict -- a
    static python question, so a config-gated module (a head built only
    under its coefficient) simply logs nothing."""
    for key in path:
        if not isinstance(tree, dict) or key not in tree:
            return False
        tree = tree[key]
    return True


# The action readout's leaves, and what each must DO.
#
# These panels are not decoration. A head can run away or stall entirely in
# its own params, invisibly on wandb: the bilinear is a two-factor product
# with ONE zero-init factor, and a zero-init factor that never leaves zero
# holds the whole pair at its init.
#
# Expected at init and what to watch:
#   query        0, must leave 0 within ~200 steps (its gradient is a rank-1
#                outer product of live rows, so it moves at step 1)
#   key          lecun 0.0625 at fan-in 256; its gradient is proportional to
#                query, so it is frozen for exactly one step and must then
#                drift. Still 0.0625 at 2k = the stall.
#   move_score   0, must leave 0 from step 1. This is also where a per-
#   move_target_score  MODALITY force lives now that the macro head is gone
#                (modality is a function of the src half), so a flat
#                move_score beside a failing entropy_macro floor is the
#                signal to promote it to an MLP.
#   switch/other 0, single-factor, must leave 0 from step 1.
#
# src/tgt stay SPLIT deliberately: a tgt column is read by every legal move
# cell of a row, so the two grow at different rates and a shared panel would
# hide it.
_ACTION_HEAD_LEAVES = {
    "player_move_query_rms": (("action_head", "move_query", "kernel"),),
    "player_move_key_rms": (("action_head", "move_key", "kernel"),),
    "player_move_score_rms": (("action_head", "move_score", "kernel"),),
    "player_move_target_score_rms": (("action_head", "move_target_score", "kernel"),),
    "player_switch_score_rms": (("action_head", "switch_score", "kernel"),),
    "player_switch_query_rms": (("action_head", "switch_query", "kernel"),),
    "player_switch_key_rms": (("action_head", "switch_key", "kernel"),),
    "player_switch_target_score_rms": (
        ("action_head", "switch_target_score", "kernel"),
    ),
    "player_other_score_rms": (("action_head", "other_score", "kernel"),),
    # The opponent-team term: shared keys (trained every turn through the
    # move block) and the per-block zero-init queries. A switch query
    # flat at init while the move query moves is the sparse switch signal
    # never arriving, the read that separates "routing fixed, credit still
    # the bound" from "routing not fixed" (docs plan 2026-09-18 step 2).
    "player_opponent_key_rms": (
        ("action_head", "opponent_team", "opponent_key", "kernel"),
    ),
    "player_belief_key_rms": (
        ("action_head", "opponent_team", "belief_key", "kernel"),
    ),
    "player_move_opponent_query_rms": (
        ("action_head", "opponent_team", "move_opponent_query", "kernel"),
    ),
    "player_switch_opponent_query_rms": (
        ("action_head", "opponent_team", "switch_opponent_query", "kernel"),
    ),
    "player_move_belief_query_rms": (
        ("action_head", "opponent_team", "move_belief_query", "kernel"),
    ),
    "player_switch_belief_query_rms": (
        ("action_head", "opponent_team", "switch_belief_query", "kernel"),
    ),
    "player_partner_query_rms": (("action_head", "partner_query", "kernel"),),
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
        ("encoder", "trunk", "blocks", "ffw", "down", "kernel"),
    ),
    # The trunk registers, two per tier: RMS-normalised on the way in like
    # every row, so only their DIRECTION reaches the trunk -- rms drift says
    # they train, not what they carry.
    "player_public_register_rms": (("encoder", "public_register_embeddings"),),
    "player_private_register_rms": (("encoder", "private_register_embeddings"),),
    "player_privileged_register_rms": (("encoder", "privileged_register_embeddings"),),
}
_NORM_ENDS = ("input", "output")


_TRUNK_ALPHA_SUBLAYERS = ("attention", "ffw")
_TRUNK_KERNEL_COLUMNS = {
    "player_trunk_kernel_col_norm_attention_q": (
        ("encoder", "trunk", "blocks", "attention", "q_proj", "kernel"),
        -2,
    ),
    "player_trunk_kernel_col_norm_ffw_up": (
        ("encoder", "trunk", "blocks", "ffw", "gate_up", "kernel"),
        -2,
    ),
}


def trunk_alpha_telemetry(param_tree) -> dict[str, jax.Array]:
    """player_trunk_alpha_{attention,ffw}_b<i>: mean over width of each
    block's residual step size under the normalised residual
    (trunk.TrunkBlock); the leaves exist only with the flag on, so the off
    path logs nothing. Per block rather than one rms over the stacked leaf
    (the _TRUNK_LEAVES note): the step sizes drifting apart across depth is
    the reading. player_trunk_kernel_col_norm_*: the mean L2 of two
    representative kernels' embedding-space vectors -- 1 by construction
    under the projection, the spectral-growth read on a plain trunk."""
    logs = {}
    for sublayer in _TRUNK_ALPHA_SUBLAYERS:
        path = ("encoder", "trunk", "blocks", f"{sublayer}_alpha")
        if not _has(param_tree, path):
            continue
        alpha = jnp.asarray(_get(param_tree, path), jnp.float32)
        block_mean = jnp.mean(alpha, axis=-1)
        for block_index in range(alpha.shape[0]):
            logs[f"player_trunk_alpha_{sublayer}_b{block_index}"] = block_mean[
                block_index
            ]
    for key, (path, axis) in _TRUNK_KERNEL_COLUMNS.items():
        if not _has(param_tree, path):
            continue
        kernel = jnp.asarray(_get(param_tree, path), jnp.float32)
        logs[key] = jnp.mean(jnp.linalg.norm(kernel, axis=axis))
    return logs


def norm_scale_telemetry(param_tree) -> dict[str, jax.Array]:
    """player_{input,output}_norm_scale_rms_<group>: rms of each SequenceGroup's
    row of the channel scale of the norm at either end of the trunk
    (modules.SequenceNormalisation; zero at init, effective scale 1 + it). A
    group's row drifting from 0 is the model re-sizing that group against the
    others: at the input the only way the pre-norm disparity the norm removed
    can come back, at the output the magnitude the heads read the group at."""
    logs = {}
    for end in _NORM_ENDS:
        path = ("encoder", f"{end}_normalisation", "group_scale")
        if not _has(param_tree, path):
            continue
        scale = jnp.asarray(_get(param_tree, path), jnp.float32)
        row_rms = jnp.sqrt(jnp.mean(jnp.square(scale), axis=-1))
        for group in SequenceGroup:
            logs[f"player_{end}_norm_scale_rms_{group.name.lower()}"] = row_rms[
                int(group)
            ]
    return logs


_HISTORY_LEAVES = {
    "player_history_step_attn_out_rms": (
        (
            "encoder",
            "history_encoder",
            "sequence_step",
            "attention",
            "out_proj",
            "kernel",
        ),
    ),
    "player_history_step_attn_qk_rms": (
        ("encoder", "history_encoder", "sequence_step", "attention", "query", "kernel"),
        ("encoder", "history_encoder", "sequence_step", "attention", "key", "kernel"),
    ),
    "player_history_slot_write_gate_rms": (
        (
            "encoder",
            "history_encoder",
            "sequence_step",
            "entity_cell",
            "gate",
            "kernel",
        ),
    ),
}
_GRAD_SUBTREES = {
    "player_action_head_grad_norm": ("action_head",),
    # The gradient into the deployable value head (its real-row CE); read
    # beside player_loss_v_win / player_value_head_r2.
    "player_value_head_grad_norm": ("value_head",),
    "player_trunk_grad_norm": ("encoder", "trunk"),
    "player_history_step_attn_grad_norm": (
        "encoder",
        "history_encoder",
        "sequence_step",
        "attention",
    ),
}


# A target column is read by every legal move cell of a row, making
# key/target_score projections a high-gain route for policy regularisation.
_APPLIED_DELTA_LEAVES = {
    "player_applied_delta_rms_switch_query": (
        ("action_head", "switch_query", "kernel"),
    ),
    "player_applied_delta_rms_switch_target_score": (
        ("action_head", "switch_target_score", "kernel"),
    ),
    "player_applied_delta_rms_move_query": (("action_head", "move_query", "kernel"),),
    "player_applied_delta_rms_move_key": (("action_head", "move_key", "kernel"),),
    "player_applied_delta_rms_move_target_score": (
        ("action_head", "move_target_score", "kernel"),
    ),
}


def _rms_panels(table: dict[str, tuple], tree, read_leaf) -> dict[str, jax.Array]:
    """One panel per `table` entry: the mean over its leaf paths of each
    leaf's rms, `read_leaf(path)` the f32 array a path names. An entry with
    a path absent from `tree` logs nothing (a config-gated module)."""
    logs = {}
    for key, paths in table.items():
        if not all(_has(tree, path) for path in paths):
            continue
        logs[key] = jnp.mean(
            jnp.stack(
                [jnp.sqrt(jnp.mean(jnp.square(read_leaf(path)))) for path in paths]
            )
        )
    return logs


def applied_delta_telemetry(prev_params, params) -> dict[str, jax.Array]:
    """rms of the update Adam ACTUALLY applied to each leaf in
    _APPLIED_DELTA_LEAVES (post-clip, post the non-finite revert): the
    gradient norm says what was asked, this says what moved. `prev_params`
    / `params` are the flax variable dicts before and after the update."""
    before, after = prev_params["params"], params["params"]

    def applied_delta(path):
        return jnp.asarray(_get(after, path), jnp.float32) - jnp.asarray(
            _get(before, path), jnp.float32
        )

    return _rms_panels(_APPLIED_DELTA_LEAVES, after, applied_delta)


def head_param_telemetry(params, grads) -> dict[str, jax.Array]:
    """Learner-side readouts of the action readout and the trunk actually
    learning: rms of each head leaf against its known init, and pre-clip
    grad norms per subtree. `params`/`grads` are the flax variable dicts
    (top-level "params" collection)."""
    param_tree, grad_tree = params["params"], grads["params"]

    def leaf(path):
        return jnp.asarray(_get(param_tree, path), jnp.float32)

    logs = _rms_panels(
        {
            **_ACTION_HEAD_LEAVES,
            **_TRUNK_LEAVES,
            **_HISTORY_LEAVES,
        },
        param_tree,
        leaf,
    )
    for key, path in _GRAD_SUBTREES.items():
        if not _has(grad_tree, path):
            continue
        logs[key] = optax.global_norm(_get(grad_tree, path))
    logs.update(state_kernel_telemetry(param_tree))
    logs.update(norm_scale_telemetry(param_tree))
    logs.update(trunk_alpha_telemetry(param_tree))
    return logs


def ratio_ess_and_tail(
    ratio: jax.Array, mask: jax.Array, tail_line: float
) -> tuple[jax.Array, jax.Array]:
    """Normalised effective sample size of importance ratios over the masked
    rows (1 = fully on-policy; low means the estimator is living off a few
    samples) and the fraction of rows above `tail_line`."""
    ratio_mean = average(ratio, mask)
    ratio_sq_mean = average(jnp.square(ratio), mask)
    return ratio_mean * ratio_mean / (ratio_sq_mean + 1e-8), average(
        ratio > tail_line, mask
    )


def state_kernel_telemetry(params) -> dict[str, jax.Array]:
    """`player_state_kernel_rms_{hp,status,boosts,other}`: rms of the three
    state linears' kernels over the input rows of each coarse feature
    group, pooled across the kernels (a kernel without the group -- no
    boosts on the private path -- contributes nothing): what share of a
    state linear reads hp against status, boosts and the rest."""
    blocks = state_kernel_blocks()
    groups = list(STATE_KERNEL_GROUPS) + ["other"]
    logs = {}
    for group in groups:
        rows = []
        for kernel in STATE_KERNELS:
            weight = jnp.asarray(params["encoder"][kernel]["kernel"], jnp.float32)
            for block in blocks[kernel][group]:
                rows.append(weight[block].reshape(-1))
        logs[f"player_state_kernel_rms_{group}"] = jnp.sqrt(
            jnp.mean(jnp.square(jnp.concatenate(rows)))
        )
    return logs


def masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
    """Mean over mask, NaN when the mask is empty (wandb skips NaN points;
    a 0.0 would read as a measurement)."""
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
    populations while LESSONS.md 3's decision rule reads one against the
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
    # taken_modality is the M-way modality index of the taken action;
    # num_legal_modalities and taken_modality_count gate the observer
    # entropy averages only (train_step's macro_valid / micro_valid) —
    # no loss reads them.
    taken_modality: jax.Array
    num_legal_modalities: jax.Array
    taken_modality_count: jax.Array


def action_axis_masks(
    flat_action_mask: jax.Array, action_index: jax.Array
) -> ActionAxisMasks:
    """See ActionAxisMasks. `has_both` is THE real-choice predicate."""
    flat_modality = jnp.asarray(CELL_MODALITY_MASK)
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

    # Previous-row action, split forced / voluntary: a forced switch (a mon
    # just fainted) and a voluntary one are two populations with opposite
    # critic biases, so they get two panels. Row 0 has no local
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
