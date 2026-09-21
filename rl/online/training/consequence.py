"""The consequence model's share of the learner update: its targets, its two
losses and its panels, from the learner forward's `consequence_inputs`.

The heads run as an apply of their own (`consequence_fn`) on those inputs, so
the noise enters here and the learner's other forwards need no rng. Gradient
reaches the trunk through the inputs, scaled by
`player_consequence_trunk_grad`: 0 is an observer of the trunk (the heads
still learn), 1 is full shaping. The target side is always a stop-gradient.
"""

import jax
import jax.numpy as jnp

from rl.environment.interfaces import ConsequenceInputs, ConsequenceOutput
from rl.environment.protos.features_pb2 import InfoFeature
from rl.model.consequence import (
    CONSEQUENCE_GROUP_IDS,
    CONSEQUENCE_GROUPS,
    public_row_alignment,
)
from rl.model.constants import NUM_PUBLIC_SLOTS
from rl.online.config import Porygon2LearnerConfig
from rl.online.training.loss import consequence_energy_loss, consequence_mean_loss
from rl.utils import average

_PUBLIC_ORDER = slice(
    InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0,
    InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0 + NUM_PUBLIC_SLOTS,
)
# Behaviour-probability bins of the taken action: prediction error on rarely
# taken actions is the support a deploy-time rule would have to respect.
PROPENSITY_EDGES = (0.0, 0.05, 0.2, 0.5)


def consequence_targets(
    state_rows: jax.Array,
    state_valid: jax.Array,
    info: jax.Array,
    acted_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Each of the 16 rows now and at the next request, (T, B, 16, D) in f32,
    and the (T, B, 16) mask of rows the pair is defined for: an action was
    taken at t (never true on a chunk's final row, so the self-paired last
    step is inert), the row is valid at t and where its identity went at t+1.
    Both ends are stop-gradients -- a live `now` would make every loss here a
    slowness force on the public rows."""
    rows = jax.lax.stop_gradient(state_rows).astype(jnp.float32)
    next_rows = jnp.concatenate((rows[1:], rows[-1:]), axis=0)
    next_valid = jnp.concatenate((state_valid[1:], state_valid[-1:]), axis=0)
    order = info[..., _PUBLIC_ORDER]
    next_order = jnp.concatenate((order[1:], order[-1:]), axis=0)
    next_index, matched = jax.vmap(jax.vmap(public_row_alignment))(order, next_order)
    gathered = jnp.take_along_axis(next_rows, next_index[..., None], axis=2)
    gathered_valid = jnp.take_along_axis(next_valid, next_index, axis=2)
    mask = acted_mask[..., None] & state_valid & gathered_valid & matched
    return rows, gathered, mask


def live_trunk_gradient(
    step_count: jax.Array, config: Porygon2LearnerConfig
) -> jax.Array:
    """`player_consequence_trunk_grad` under its linear ramp, as a traced
    scalar of the learner's step count."""
    progress = (
        step_count.astype(jnp.float32) - config.player_consequence_ramp_start_step
    ) / max(config.player_consequence_ramp_steps, 1)
    return config.player_consequence_trunk_grad * jnp.clip(progress, 0.0, 1.0)


def scale_trunk_gradient(
    inputs: ConsequenceInputs, scale: float | jax.Array
) -> ConsequenceInputs:
    """The same values with `scale` times the gradient into the trunk."""

    def scaled(rows: jax.Array) -> jax.Array:
        held = jax.lax.stop_gradient(rows)
        return held + scale * (rows - held)

    return inputs.replace(
        state_rows=scaled(inputs.state_rows),
        source_row=scaled(inputs.source_row),
        target_row=scaled(inputs.target_row),
        cls_row=scaled(inputs.cls_row),
    )


def consequence_terms(
    outputs: ConsequenceOutput,
    inputs: ConsequenceInputs,
    info: jax.Array,
    acted_mask: jax.Array,
    real_value: jax.Array,
    behaviour_log_prob: jax.Array,
    config: Porygon2LearnerConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """The weighted loss and the panels. `outputs` leaves lead with the draw
    axis; `real_value` is the state value head's expectation on the REAL rows,
    (T, B)."""
    now, real_next, mask = consequence_targets(
        inputs.state_rows, inputs.state_valid, info, acted_mask
    )
    group_ids = jnp.asarray(CONSEQUENCE_GROUP_IDS)
    scales = jnp.asarray(config.player_consequence_scales, jnp.float32)
    targets = (now, real_next, mask, group_ids, scales)
    mean = consequence_mean_loss(outputs.mean[0], *targets)
    state_only = consequence_mean_loss(outputs.state_only_mean[0], *targets)
    sampler = consequence_energy_loss(outputs.sample, *targets)
    loss = (
        config.player_consequence_mean_coef * (mean["loss"] + state_only["loss"])
        + config.player_consequence_sampler_coef * sampler["loss"]
    )

    logs = dict(
        player_consequence_mean_loss=mean["loss"],
        player_consequence_state_only_loss=state_only["loss"],
        player_consequence_sampler_loss=sampler["loss"],
        player_consequence_transitions=mask.any(axis=-1).sum(dtype=jnp.int32),
    )
    predicted_square = jnp.square(outputs.mean[0].astype(jnp.float32) - now).sum(-1)
    true_square = jnp.square(real_next - now).sum(-1)
    for index, name in enumerate(CONSEQUENCE_GROUPS):
        in_group = mask & (group_ids == index)
        logs[f"player_consequence_mean_loss_{name}"] = mean["ratio"][index]
        logs[f"player_consequence_sampler_loss_{name}"] = sampler["ratio"][index]
        # What copying the current row scores under the same FIXED scale: it
        # rises if the trunk inflates its rows' step-to-step change.
        logs[f"player_consequence_copy_loss_{name}"] = mean["copy_ratio"][index]
        logs[f"player_consequence_copy_energy_{name}"] = sampler["copy_ratio"][index]
        # Positive = the action's own rows predict what the state alone cannot.
        logs[f"player_consequence_action_gain_{name}"] = (
            state_only["ratio"][index] - mean["ratio"][index]
        )
        logs[f"player_consequence_spread_over_skill_{name}"] = sampler[
            "spread_over_skill"
        ][index]
        logs[f"player_consequence_change_norm_{name}"] = sampler["change_norm"][index]
        logs[f"player_consequence_predicted_rms_ratio_{name}"] = jnp.sqrt(
            average(predicted_square, in_group)
            / jnp.maximum(average(true_square, in_group), 1e-12)
        )

    # What a deploy-time read would see: the value of the PREDICTED value row
    # against the value of the real next one, for the action actually taken.
    # The copy predictor's gap is |V(t+1) - V(t)| by construction.
    next_value = jnp.concatenate((real_value[1:], real_value[-1:]), axis=0)
    value_mask = mask[..., -1]
    sampled_gap = jnp.abs(outputs.sample_value.mean(axis=0) - next_value)
    logs["player_consequence_value_gap_sampler"] = average(sampled_gap, value_mask)
    logs["player_consequence_value_gap_mean"] = average(
        jnp.abs(outputs.mean_value[0] - next_value), value_mask
    )
    logs["player_consequence_value_gap_copy"] = average(
        jnp.abs(real_value - next_value), value_mask
    )
    propensity = jnp.exp(behaviour_log_prob.astype(jnp.float32))
    for low, high in zip(PROPENSITY_EDGES, PROPENSITY_EDGES[1:] + (jnp.inf,)):
        in_bin = value_mask & (propensity >= low) & (propensity < high)
        logs[f"player_consequence_value_gap_sampler_p{low:g}"] = average(
            sampled_gap, in_bin
        )
    return loss, logs
