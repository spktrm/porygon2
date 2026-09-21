"""Proper-scoring losses for fixed-meaning outcomes from self-play."""

import jax
import jax.numpy as jnp
import optax

from rl.environment.consequence_labels import HP_CHANGE_SUPPORT, NUM_HP_CHANGE_BINS
from rl.model.constants import NUM_PUBLIC_SLOTS
from rl.utils import average


def observable_gradient(step_count, config):
    progress = (
        step_count.astype(jnp.float32) - config.player_observable_ramp_start_step
    ) / max(config.player_observable_ramp_steps, 1)
    return config.player_observable_shared_grad * jnp.clip(progress, 0, 1)


def scale_features(features, scale):
    held = jax.lax.stop_gradient(features)
    scale = jnp.asarray(scale, dtype=features.dtype)
    return held + scale * (features - held)


def observable_terms(logits, targets):
    logits = logits.astype(jnp.float32)
    execution = logits[..., 0]
    entity = logits[..., 1:].reshape(
        logits.shape[:-1] + (NUM_PUBLIC_SLOTS, NUM_HP_CHANGE_BINS + 1)
    )
    hp_logits = entity[..., :NUM_HP_CHANGE_BINS]
    faint_logits = entity[..., -1]
    support = jnp.asarray(HP_CHANGE_SUPPORT, jnp.float32)
    hp_target = jnp.maximum(1 - jnp.abs(targets.hp_change[..., None] - support) * 10, 0)
    hp_target = hp_target / jnp.maximum(hp_target.sum(-1, keepdims=True), 1e-8)
    execution_loss = average(
        optax.sigmoid_binary_cross_entropy(execution, targets.executed),
        targets.execution_valid,
    )
    hp_loss = average(
        optax.softmax_cross_entropy(hp_logits, hp_target), targets.hp_valid
    )
    faint_loss = average(
        optax.sigmoid_binary_cross_entropy(faint_logits, targets.fainted),
        targets.faint_valid,
    )
    present = jnp.stack(
        (
            targets.execution_valid.any(),
            targets.hp_valid.any(),
            targets.faint_valid.any(),
        )
    )
    loss = average(jnp.stack((execution_loss, hp_loss, faint_loss)), present)
    hp_prediction = jax.nn.softmax(hp_logits) @ support
    logs = dict(
        player_observable_loss=loss,
        player_observable_execution_loss=execution_loss,
        player_observable_hp_loss=hp_loss,
        player_observable_faint_loss=faint_loss,
        player_observable_hp_mae=average(
            jnp.abs(hp_prediction - targets.hp_change), targets.hp_valid
        ),
        player_observable_hp_copy_mae=average(
            jnp.abs(targets.hp_change), targets.hp_valid
        ),
        player_observable_execution_brier=average(
            jnp.square(jax.nn.sigmoid(execution) - targets.executed),
            targets.execution_valid,
        ),
        player_observable_faint_brier=average(
            jnp.square(jax.nn.sigmoid(faint_logits) - targets.fainted),
            targets.faint_valid,
        ),
    )
    for name, prediction, label, valid in (
        ("execution", execution > 0, targets.executed, targets.execution_valid),
        ("faint", faint_logits > 0, targets.fainted, targets.faint_valid),
    ):
        logs[f"player_observable_{name}_count"] = valid.sum()
        logs[f"player_observable_{name}_positive_count"] = (valid & label).sum()
        logs[f"player_observable_{name}_positive_recall"] = average(
            prediction, valid & label
        )
        logs[f"player_observable_{name}_negative_recall"] = average(
            ~prediction, valid & ~label
        )
    logs["player_observable_hp_count"] = targets.hp_valid.sum()
    return loss, logs
