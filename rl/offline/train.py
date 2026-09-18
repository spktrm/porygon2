"""The offline trainer: the player model's PUBLIC path -- encoder + trunk
on the 55 public rows, the public critic and (unless --no-world-model) the
event world model -- as one module whose submodule names are the player
model's, so `merge_params` resumes every leaf by path at a learner
relaunch. The encoder is overlaid from a learner checkpoint and frozen
unless `joint`; the critic and the world model train on the replay export.

    env/bin/python -u rl/offline/train.py --trunk-ckpt ckpts/gen9/ckpt_XXXXXXXX
    env/bin/python -u rl/offline/train.py --trunk-ckpt ... --no-world-model
"""

import argparse
import dataclasses
import itertools
import json
import os
import time
from collections.abc import Iterator
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ml_collections import ConfigDict

import wandb
from rl import checkpoint as checkpoint_lib
from rl.environment.event_labels import EventKind, EventLabels
from rl.environment.interfaces import EventStates
from rl.model import world_model as wm
from rl.model.config import get_player_model_config
from rl.model.constants import PUBLIC_CLS_LOCAL_ROW
from rl.model.encoder import SIDE_MINE, SIDE_OPPONENT, Encoder, active_slot_rows
from rl.model.heads import CategoricalValueLogitHead
from rl.model.utils import get_num_params
from rl.offline.config import Porygon2OfflineConfig
from rl.offline.dataset import ReplayBatch, load_replay_store
from rl.online.artifact import merge_params

Params = dict


class TrainerState(NamedTuple):
    params: Params
    opt_state: optax.OptState
    scale: jax.Array  # (9,) EMA RMS difference per public group
    step: jax.Array


class BatchTerms(NamedTuple):
    """Sums and counts over one trajectory, pooled over the batch by the
    caller -- every metric is a ratio of pooled sums, never a mean of
    per-trajectory ratios."""

    sums: dict
    counts: dict


def trainer_model_config(
    joint: bool,
    world_model: bool,
    history_recurrence: str = "loop",
    trunk_normalised_residual: bool = True,
) -> ConfigDict:
    cfg = get_player_model_config(generation=9, train=True)
    cfg.encoder.public_only = True
    cfg.encoder.history_recurrence = history_recurrence
    cfg.encoder.trunk.normalised_residual = trunk_normalised_residual
    cfg.world_model.enabled = world_model
    cfg.world_model.joint = joint
    return cfg


class OfflineTrainer(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = Encoder(self.cfg.encoder)
        self.public_value_head = CategoricalValueLogitHead(self.cfg.public_value_head)
        if self.cfg.world_model.enabled:
            self.world_model = wm.EventWorldModel(
                self.cfg.world_model, name="world_model"
            )

    def events(self, batch: ReplayBatch) -> EventStates:
        return self.encoder.encode_events(batch.packed_history, batch.history)

    def trajectory_terms(
        self,
        batch: ReplayBatch,
        scale: jax.Array,
        rng: jax.Array,
        with_samples: bool,
        num_samples: int,
    ) -> BatchTerms:
        """Every loss term and metric numerator/denominator for ONE
        trajectory (leaves without the batch axis)."""
        events = self.events(batch)
        labels: EventLabels = batch.labels
        states = events.states
        if not self.cfg.world_model.joint:
            states = jax.lax.stop_gradient(states)
        sums = {}
        counts = {}

        def add(name, value, mask):
            sums[name] = jnp.sum(jnp.where(mask, value, 0.0))
            counts[name] = jnp.sum(mask.astype(jnp.float32))

        self.critic_terms(add, counts, states, events.step_valid, labels, batch)
        if self.cfg.world_model.enabled:
            self.world_model_terms(
                add,
                sums,
                counts,
                events,
                states,
                labels,
                batch,
                scale,
                rng,
                with_samples,
                num_samples,
            )
        return BatchTerms(sums=sums, counts=counts)

    def critic_terms(self, add, counts, states, step_valid, labels, batch) -> None:
        """The public critic on real states, every valid step."""
        value = self.public_value_head(states[:, PUBLIC_CLS_LOCAL_ROW])
        outcome_bin = jnp.argmax(batch.win_reward)
        value_nll = -value.log_probs[:, outcome_bin]
        outcome_value = jnp.asarray([-1.0, 0.0, 1.0])[outcome_bin]
        add("loss_public_value", value_nll, step_valid)
        add("value_residual", jnp.square(value.expectation - outcome_value), step_valid)
        add(
            "value_residual_boundary",
            jnp.square(value.expectation - outcome_value),
            step_valid & labels.new_turn,
        )
        add(
            "value_residual_midturn",
            jnp.square(value.expectation - outcome_value),
            step_valid & ~labels.new_turn,
        )
        add(
            "outcome_value", jnp.full_like(value.expectation, outcome_value), step_valid
        )
        add(
            "outcome_value_sq",
            jnp.full_like(value.expectation, outcome_value**2),
            step_valid,
        )
        counts["value_residual_boundary_n"] = counts["value_residual_boundary"]
        counts["value_residual_midturn_n"] = counts["value_residual_midturn"]

    def world_model_terms(
        self,
        add,
        sums,
        counts,
        events: EventStates,
        states: jax.Array,
        labels: EventLabels,
        batch: ReplayBatch,
        scale: jax.Array,
        rng: jax.Array,
        with_samples: bool,
        num_samples: int,
    ) -> None:
        num_steps = states.shape[0]
        events.step_valid
        pair_valid = labels.valid & ~labels.terminal
        next_states = jnp.roll(states, -1, axis=0)

        ally_slots, ally_found = active_slot_rows(
            events.slot_valid, events.public_sides, events.public_positions, SIDE_MINE
        )
        enemy_slots, enemy_found = active_slot_rows(
            events.slot_valid,
            events.public_sides,
            events.public_positions,
            SIDE_OPPONENT,
        )
        ally_slots = jnp.where(ally_found, ally_slots, -1)
        enemy_slots = jnp.where(enemy_found, enemy_slots, -1)
        row_mask = jax.vmap(wm.update_rows)(
            labels.touched, labels.field_touched, ally_slots, enemy_slots
        )
        tokens = wm.EventTokens(
            kind=labels.kind,
            actor=labels.actor,
            move=labels.move,
            target=labels.target,
        )
        actor_is_mine = labels.actor_side == SIDE_MINE
        noise_key, time_key, sample_key = jax.random.split(rng, 3)
        noise = jax.random.normal(noise_key, states.shape, jnp.float32)
        time = jax.random.uniform(time_key, (num_steps,), jnp.float32, 0.0, 0.999)
        terms = jax.vmap(
            self.world_model.step_terms, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, 0, 0)
        )(
            states,
            events.row_valid,
            next_states,
            labels.declared_kind,
            labels.declared_arg,
            tokens,
            actor_is_mine,
            row_mask,
            scale,
            noise,
            time,
        )
        touched_bits = jnp.concatenate(
            [labels.touched, labels.field_touched[:, None]], axis=-1
        )
        nll = jax.vmap(wm.grammar_nll)(
            terms.logits, tokens, touched_bits, labels.new_turn, labels.num_revealed
        )  # (H, 5)
        # A label outside the grammar mask (a move event whose move token is
        # unknown, a target never revealed) scores -log 0: it leaves the
        # loss and is counted instead.
        actor_legal = jax.vmap(lambda kind, n, actor: wm.actor_mask(kind, n)[actor])(
            labels.kind, labels.num_revealed, labels.actor
        )
        move_legal = jax.vmap(lambda kind, move: wm.move_mask(kind)[move])(
            labels.kind, labels.move
        )
        target_legal = jax.vmap(lambda n, target: wm.target_mask(n)[target])(
            labels.num_revealed, labels.target
        )
        label_legal = jnp.stack(
            [
                jnp.ones_like(actor_legal),
                actor_legal,
                move_legal,
                target_legal,
                jnp.ones_like(actor_legal),
            ],
            axis=-1,
        )
        position_valid = terms.grammar_valid & labels.valid[:, None]
        for index, name in enumerate(("actor", "move", "target")):
            add(
                f"label_illegal_{name}",
                (~label_legal[:, index + 1]).astype(jnp.float32),
                position_valid[:, index + 1],
            )
        position_valid = position_valid & label_legal

        for index, name in enumerate(("kind", "actor", "move", "target", "touched")):
            add(f"loss_{name}", nll[:, index], position_valid[:, index])
        # Only the positions the grammar asks for count: a switch event has
        # no move, and its move logit at the sentinel reads -log 0.
        event_nll = jnp.where(position_valid, nll, 0.0).sum(-1)
        add("nats_per_token", event_nll, labels.valid)
        counts["nats_per_token"] = position_valid.astype(jnp.float32).sum()
        for kind in EventKind:
            add(
                f"nll_kind_{kind.name.lower()}",
                event_nll,
                labels.valid & (labels.kind == kind),
            )
        actor_hit = terms.logits.actor.argmax(-1) == labels.actor
        move_hit = terms.logits.move.argmax(-1) == labels.move
        for side_name, side in (("mine", True), ("theirs", False)):
            side_mask = position_valid[:, 1] & (actor_is_mine == side)
            add(f"actor_acc_{side_name}", actor_hit.astype(jnp.float32), side_mask)
            move_side = position_valid[:, 2] & (actor_is_mine == side)
            for reveal_name, revealed in (("revealed", True), ("unrevealed", False)):
                add(
                    f"move_acc_{side_name}_{reveal_name}",
                    move_hit.astype(jnp.float32),
                    move_side & (labels.move_previously_revealed == revealed),
                )
        touched_pred = terms.logits.touched > 0
        touched_true = touched_bits
        add(
            "touched_tp",
            (touched_pred & touched_true).astype(jnp.float32).sum(-1),
            position_valid[:, 4],
        )
        add(
            "touched_fp",
            (touched_pred & ~touched_true).astype(jnp.float32).sum(-1),
            position_valid[:, 4],
        )
        add(
            "touched_fn",
            (~touched_pred & touched_true).astype(jnp.float32).sum(-1),
            position_valid[:, 4],
        )
        add(
            "touched_count",
            touched_true.astype(jnp.float32).sum(-1),
            position_valid[:, 4],
        )
        new_turn_hit = (terms.logits.new_turn > 0) == labels.new_turn
        add("new_turn_acc", new_turn_hit.astype(jnp.float32), position_valid[:, 0])

        # The flow and its control, in the unit-scale space of the difference.
        group_scale = scale[jnp.asarray(wm.LOCAL_GROUP_IDS)]
        x1_energy = terms.delta_energy / jnp.square(group_scale)[None]
        update_valid = row_mask & pair_valid[:, None]
        # Per-row sums over the steps, (55,): the batch and the eval pool
        # them further, so no history bucket shows in a leaf's shape.
        all_valid = pair_valid[:, None] & jnp.ones_like(row_mask)
        sums["flow_error"] = jnp.where(update_valid, terms.flow_error, 0.0).sum(0)
        # The mean step covers EVERY row: the update rows as the flow's
        # matched control, the rest as the residual imagine() applies.
        sums["mean_error"] = jnp.where(all_valid, terms.mean_error, 0.0).sum(0)
        sums["mean_error_update"] = jnp.where(update_valid, terms.mean_error, 0.0).sum(
            0
        )
        sums["x1_energy"] = jnp.where(update_valid, x1_energy, 0.0).sum(0)
        sums["x1_energy_all"] = jnp.where(all_valid, x1_energy, 0.0).sum(0)
        sums["all_rows"] = all_valid.astype(jnp.float32).sum(0)
        sums["delta_energy_update"] = jnp.where(
            update_valid, terms.delta_energy, 0.0
        ).sum(0)
        sums["delta_energy_untouched"] = jnp.where(
            ~row_mask & pair_valid[:, None], terms.delta_energy, 0.0
        ).sum(0)
        sums["update_rows"] = update_valid.astype(jnp.float32).sum(0)
        for kind in EventKind:
            kind_mask = pair_valid & (labels.kind == kind)
            sums[f"delta_energy_untouched_{kind.name.lower()}"] = jnp.where(
                ~row_mask & kind_mask[:, None], terms.delta_energy, 0.0
            ).sum(0)
            sums[f"delta_energy_all_{kind.name.lower()}"] = jnp.where(
                kind_mask[:, None], terms.delta_energy, 0.0
            ).sum(0)

        # The terminal outcome head, at the real terminal step only.
        outcome_bin = jnp.argmax(batch.win_reward)
        terminal_nll = -jax.nn.log_softmax(terms.terminal_logits, axis=-1)[
            :, outcome_bin
        ]
        add("loss_terminal", terminal_nll, labels.terminal)

        if with_samples:
            # Imagined next states from the teacher-forced tokens; the value
            # of each sample against the value of the real next state.
            frozen_head = self.public_value_head.clone()
            frozen_params = jax.lax.stop_gradient(
                self.public_value_head.variables["params"]
            )
            real_next = frozen_head.apply(
                {"params": frozen_params}, next_states[:, PUBLIC_CLS_LOCAL_ROW]
            ).expectation
            keys = jax.random.split(sample_key, num_samples)

            def imagine_all(key):
                step_keys = jax.random.split(key, num_steps)
                imagined = jax.vmap(
                    self.world_model.imagine, in_axes=(0, 0, 0, None, 0)
                )(states, tokens, row_mask, scale, step_keys)
                return frozen_head.apply(
                    {"params": frozen_params}, imagined[:, PUBLIC_CLS_LOCAL_ROW]
                ).expectation

            sampled = jax.vmap(imagine_all)(keys)  # (S, H)
            sample_mean = sampled.mean(0)
            add("imagined_residual", jnp.square(sample_mean - real_next), pair_valid)
            add(
                "imagined_residual_faint",
                jnp.square(sample_mean - real_next),
                pair_valid & (labels.kind == EventKind.FAINT),
            )
            add(
                "imagined_residual_nonfaint",
                jnp.square(sample_mean - real_next),
                pair_valid & (labels.kind != EventKind.FAINT),
            )
            add("real_next_value", real_next, pair_valid)
            add("real_next_value_sq", jnp.square(real_next), pair_valid)
            # CRPS of the empirical sample distribution against the real value.
            abs_err = jnp.abs(sampled - real_next[None]).mean(0)
            spread = jnp.abs(sampled[:, None] - sampled[None, :]).mean((0, 1))
            add("value_crps", abs_err - 0.5 * spread, pair_valid)
            add("value_crps_mean_control", jnp.abs(sample_mean - real_next), pair_valid)


def pooled_metrics(pooled: BatchTerms, scale: jax.Array, floor: float) -> dict:
    """Ratios of pooled sums; the flow and control losses through the
    per-group normalisation (copy = 1, exact = 0)."""
    sums, counts = pooled.sums, pooled.counts
    metrics = {}
    scalar_keys = [key for key, value in sums.items() if jnp.ndim(value) == 0]
    for key in scalar_keys:
        metrics[key] = sums[key] / jnp.maximum(counts[key], 1.0)
    n = jnp.maximum(counts["outcome_value"], 1.0)
    outcome_var = sums["outcome_value_sq"] / n - jnp.square(sums["outcome_value"] / n)
    metrics["public_value_r2"] = 1.0 - metrics["value_residual"] / jnp.maximum(
        outcome_var, 1e-8
    )
    metrics["public_value_r2_boundary"] = 1.0 - metrics[
        "value_residual_boundary"
    ] / jnp.maximum(outcome_var, 1e-8)
    metrics["public_value_r2_midturn"] = 1.0 - metrics[
        "value_residual_midturn"
    ] / jnp.maximum(outcome_var, 1e-8)
    if "flow_error" in sums:
        world_model_metrics(metrics, sums, counts, scale, floor)
    return metrics


def world_model_metrics(metrics: dict, sums: dict, counts: dict, scale, floor) -> None:
    update_rows = sums["update_rows"] > 0
    flow_loss, flow_groups = wm.pooled_group_loss(
        sums["flow_error"], sums["x1_energy"], update_rows, floor
    )
    mean_loss, mean_groups = wm.pooled_group_loss(
        sums["mean_error"], sums["x1_energy_all"], sums["all_rows"] > 0, floor
    )
    control_loss, _ = wm.pooled_group_loss(
        sums["mean_error_update"], sums["x1_energy"], update_rows, floor
    )
    metrics["loss_flow"] = flow_loss
    metrics["loss_mean_control"] = control_loss
    metrics["loss_mean_all_rows"] = mean_loss
    for index, name in enumerate(wm.LOCAL_GROUP_NAMES):
        metrics[f"flow_loss_{name}"] = flow_groups[index]
        metrics[f"mean_loss_{name}"] = mean_groups[index]
        metrics[f"delta_scale_{name}"] = scale[index]
    total_energy = (
        sums["delta_energy_update"].sum() + sums["delta_energy_untouched"].sum()
    )
    metrics["untouched_delta_frac"] = sums[
        "delta_energy_untouched"
    ].sum() / jnp.maximum(total_energy, 1e-8)
    for kind in EventKind:
        name = kind.name.lower()
        metrics[f"untouched_delta_frac_{name}"] = sums[
            f"delta_energy_untouched_{name}"
        ].sum() / jnp.maximum(sums[f"delta_energy_all_{name}"].sum(), 1e-8)
    precision = sums["touched_tp"] / jnp.maximum(
        sums["touched_tp"] + sums["touched_fp"], 1.0
    )
    recall = sums["touched_tp"] / jnp.maximum(
        sums["touched_tp"] + sums["touched_fn"], 1.0
    )
    metrics["touched_f1"] = (
        2 * precision * recall / jnp.maximum(precision + recall, 1e-8)
    )
    if "imagined_residual" in sums:
        m = jnp.maximum(counts["real_next_value"], 1.0)
        next_var = sums["real_next_value_sq"] / m - jnp.square(
            sums["real_next_value"] / m
        )
        metrics["imagined_value_r2"] = 1.0 - metrics["imagined_residual"] / jnp.maximum(
            next_var, 1e-8
        )
        metrics["imagined_value_r2_faint"] = 1.0 - metrics[
            "imagined_residual_faint"
        ] / jnp.maximum(next_var, 1e-8)
        metrics["imagined_value_r2_nonfaint"] = 1.0 - metrics[
            "imagined_residual_nonfaint"
        ] / jnp.maximum(next_var, 1e-8)


def loss_weights(config: Porygon2OfflineConfig) -> dict[str, float]:
    weights = {}
    if config.world_model:
        weights.update(
            loss_kind=config.kind_loss_weight,
            loss_actor=config.actor_loss_weight,
            loss_move=config.move_loss_weight,
            loss_target=config.target_loss_weight,
            loss_touched=config.touched_loss_weight,
            loss_flow=config.flow_loss_weight,
            loss_mean_all_rows=config.mean_loss_weight,
            loss_terminal=config.terminal_loss_weight,
        )
    weights["loss_public_value"] = config.public_value_loss_weight
    return weights


def total_loss(config: Porygon2OfflineConfig, metrics: dict) -> jax.Array:
    total = None
    for name, weight in loss_weights(config).items():
        if total is None:
            total = weight * metrics[name]
        else:
            total = total + weight * metrics[name]
    return total


def batch_terms(
    model, params, batch: ReplayBatch, scale, rng, with_samples, num_samples
):
    keys = jax.random.split(rng, batch.win_reward.shape[0])

    def one(args):
        trajectory, key = args
        return model.apply(
            {"params": params},
            trajectory,
            scale,
            key,
            with_samples,
            num_samples,
            method=OfflineTrainer.trajectory_terms,
        )

    # Sequential over trajectories (a scan, rematerialised): a vmap over 8
    # trajectories x 512 steps asked for 4.8 GB in one buffer.
    per_trajectory = jax.lax.map(jax.checkpoint(one), (batch, keys))
    return jax.tree.map(lambda x: x.sum(0), per_trajectory)


def make_train_step(config: Porygon2OfflineConfig, model, optimiser, model_cfg):
    floor = model_cfg.world_model.scale_floor

    @jax.jit
    def train_step(state: TrainerState, batch: ReplayBatch, rng):
        def loss_fn(params):
            pooled = batch_terms(model, params, batch, state.scale, rng, False, 0)
            metrics = pooled_metrics(pooled, state.scale, floor)
            return total_loss(config, metrics), (metrics, pooled)

        (loss, (metrics, pooled)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        updates, opt_state = optimiser.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        scale = state.scale
        metrics["loss"] = loss
        metrics["gradient_norm"] = optax.global_norm(grads)
        if config.world_model:
            batch_scale = wm.group_delta_scale(
                pooled.sums["delta_energy_update"]
                / jnp.maximum(pooled.sums["update_rows"], 1.0),
                pooled.sums["update_rows"] > 0,
                floor,
            )
            scale = (
                config.scale_momentum * state.scale
                + (1 - config.scale_momentum) * batch_scale
            )
            metrics["out_proj_rms"] = jnp.sqrt(
                jnp.mean(
                    jnp.square(params["world_model"]["flow"]["out_proj"]["kernel"])
                )
            )
        return TrainerState(params, opt_state, scale, state.step + 1), metrics

    return train_step


def make_eval_step(config: Porygon2OfflineConfig, model, model_cfg):
    floor = model_cfg.world_model.scale_floor

    @jax.jit
    def eval_step(state: TrainerState, batch: ReplayBatch, rng):
        pooled = batch_terms(
            model, state.params, batch, state.scale, rng, True, config.eval_samples
        )
        return pooled

    def evaluate(state, batches: Iterator[ReplayBatch], rng) -> dict[str, float]:
        pooled = None
        for index, batch in enumerate(batches):
            terms = jax.device_get(
                eval_step(state, batch, jax.random.fold_in(rng, index))
            )
            if pooled is None:
                pooled = terms
            else:
                pooled = jax.tree.map(lambda a, b: a + b, pooled, terms)
        if pooled is None:
            return {}
        metrics = pooled_metrics(
            BatchTerms(sums=pooled.sums, counts=pooled.counts), state.scale, floor
        )
        return {f"eval_{key}": float(value) for key, value in metrics.items()}

    return evaluate


def param_labels(params: Params, joint: bool) -> Params:
    trained = {"public_value_head"}
    if "world_model" in params:
        trained.add("world_model")
    if joint:
        trained.add("encoder")
    labels = {}
    for key, value in params.items():
        if key in trained:
            labels[key] = jax.tree.map(lambda _: "train", value)
        else:
            labels[key] = jax.tree.map(lambda _: "frozen", value)
    if "world_model" in params:
        # The flow's scale is written by the trainer's EMA, never by a gradient.
        labels["world_model"]["delta_scale"] = "frozen"
    return labels


def without_subtrees(loaded: Params, subtrees: tuple[str, ...]) -> Params:
    """`loaded` with each "a/b" subtree removed, so overlay_whole leaves the
    fresh init there instead of refusing a shape that no longer matches
    (a history encoder of the other recurrence form)."""
    loaded = dict(loaded)
    for subtree in subtrees:
        keys = subtree.split("/")
        node = loaded
        for key in keys[:-1]:
            node[key] = dict(node[key])
            node = node[key]
        node.pop(keys[-1])
    return loaded


def overlay_whole(
    params: Params, loaded: Params, source: str, fresh_subtrees: tuple[str, ...] = ()
) -> Params:
    """merge_params, refusing a loaded subtree that did not land leaf for
    leaf: a missing or reshaped leaf under a loaded top-level key means the
    checkpoint is not this model's, not a resume across a change -- except
    under a `fresh_subtrees` entry, which stays at init by request."""
    merged, kept_fresh, dropped = merge_params(params, loaded)
    misses = [
        path
        for path in kept_fresh
        if path.split("/")[1] in loaded
        and not any(
            path == f"/{subtree}" or path.startswith(f"/{subtree}/")
            for subtree in fresh_subtrees
        )
    ]
    if misses:
        raise ValueError(
            f"{source}: {len(misses)} leaves did not overlay: {misses[:5]}"
        )
    if dropped:
        print(f"{source}: {len(dropped)} checkpoint-only subtrees not carried")
    return merged


def with_delta_scale(params: Params, scale: jax.Array) -> Params:
    if "world_model" not in params:
        return params
    params = dict(params)
    params["world_model"] = dict(params["world_model"], delta_scale=scale)
    return params


def save_artifact(
    config: Porygon2OfflineConfig,
    state: TrainerState,
    step: int,
    shard_manifest: dict,
    best: bool = False,
) -> str:
    if best:
        ckpt_name = "ckpt_best"
    else:
        ckpt_name = f"ckpt_{step:08}"
    save_path = os.path.abspath(
        os.path.join(config.artifact_root, config.format_id, ckpt_name)
    )
    checkpoint_lib.save_train_state(
        save_path,
        config,
        dict(
            # The learner's layout: the variables dict, so merge_params and
            # the harness read the artifact exactly as a learner checkpoint.
            params={"params": with_delta_scale(state.params, state.scale)},
            scalars=dict(step_count=step, delta_scale=np.asarray(state.scale)),
        ),
        builder_state_components={},
        league_bytes=None,
    )
    with open(os.path.join(save_path, "manifest.json"), "w") as f:
        json.dump(
            dict(
                kind="offline",
                format_id=config.format_id,
                step=step,
                trunk_ckpt=config.trunk_ckpt,
                world_model=config.world_model,
                joint=config.joint,
                public_only=True,
                export_commit=shard_manifest.get("export_commit"),
                num_history=shard_manifest.get("num_history"),
            ),
            f,
            indent=2,
        )
    return save_path


def parse_args() -> tuple[Porygon2OfflineConfig, int, bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trunk-ckpt", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--joint", action="store_true")
    parser.add_argument(
        "--world-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="off = the critic alone: public_value_head on every event state",
    )
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-history-steps", type=int, default=None)
    parser.add_argument(
        "--history-recurrence", choices=["loop", "stacked"], default=None
    )
    parser.add_argument(
        "--fresh-subtrees",
        action="append",
        default=[],
        help="a param subtree kept at its fresh init, e.g. encoder/history_encoder",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="wandb disabled")
    args = parser.parse_args()
    overrides = dict(
        trunk_ckpt=args.trunk_ckpt,
        resume_from=args.resume_from,
        joint=args.joint,
        world_model=args.world_model,
        fresh_subtrees=tuple(args.fresh_subtrees),
    )
    for name in (
        "num_steps",
        "batch_size",
        "learning_rate",
        "max_history_steps",
        "history_recurrence",
    ):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    return (
        dataclasses.replace(Porygon2OfflineConfig(), **overrides),
        args.seed,
        args.debug,
    )


def main() -> None:
    config, seed, debug = parse_args()
    if debug:
        os.environ["WANDB_MODE"] = "disabled"
    model_cfg = trainer_model_config(
        config.joint,
        config.world_model,
        config.history_recurrence,
        config.trunk_normalised_residual,
    )
    model = OfflineTrainer(model_cfg)
    dataset = load_replay_store(config)
    shard_manifest = dataset.manifest
    print(f"{len(dataset)} trajectories, {len(dataset.train_games)} training games")
    first_batch = next(dataset.eval_batches())
    print("Initialising (traces the public encoder once)...")
    init_batch = jax.tree.map(lambda x: jnp.asarray(x[0]), first_batch)
    params = jax.jit(
        lambda key, batch: model.init(
            key,
            batch,
            jnp.ones(wm.NUM_PUBLIC_GROUPS),
            key,
            False,
            0,
            method=OfflineTrainer.trajectory_terms,
        )
    )(jax.random.key(seed), init_batch)["params"]
    # The learner's component is the variables dict, {"params": tree}.
    restored = checkpoint_lib.load_component(config.trunk_ckpt, "player", "params")[
        "params"
    ]
    restored = without_subtrees(restored, config.fresh_subtrees)
    params = overlay_whole(
        params,
        {key: restored[key] for key in ("encoder", "public_value_head")},
        config.trunk_ckpt,
        config.fresh_subtrees,
    )
    scale = jnp.ones(wm.NUM_PUBLIC_GROUPS, jnp.float32)
    if config.resume_from is not None:
        params = overlay_whole(
            params,
            checkpoint_lib.load_component(config.resume_from, "player", "params")[
                "params"
            ],
            config.resume_from,
        )
        scalars = checkpoint_lib.load_component(config.resume_from, "player", "scalars")
        if "delta_scale" in scalars:
            scale = jnp.asarray(scalars["delta_scale"], jnp.float32)
    schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=config.num_steps,
        alpha=config.lr_final_fraction,
    )
    optimiser = optax.multi_transform(
        {
            "train": optax.chain(
                optax.clip_by_global_norm(config.clip_gradient),
                optax.adamw(
                    learning_rate=schedule,
                    b1=config.adam.b1,
                    b2=config.adam.b2,
                    eps=config.adam.eps,
                    weight_decay=config.adam.weight_decay,
                ),
            ),
            "frozen": optax.set_to_zero(),
        },
        param_labels(params, config.joint),
    )
    state = TrainerState(params, optimiser.init(params), scale, jnp.asarray(0))
    train_step = make_train_step(config, model, optimiser, model_cfg)
    evaluate = make_eval_step(config, model, model_cfg)
    wandb.init(
        project="pokemon-rl-offline",
        name=f"offline-{config.format_id}-{config.history_recurrence}-s{seed}",
        config=dict(
            offline_config=dataclasses.asdict(config),
            num_params=get_num_params(params),
        ),
    )
    rng = jax.random.key(seed + 1)
    best_eval = float("inf")
    start = time.monotonic()
    batches = dataset.train_batches(seed=seed)
    for step, batch in enumerate(itertools.islice(batches, config.num_steps), start=1):
        rng, step_key = jax.random.split(rng)
        batch = jax.tree.map(jnp.asarray, batch)
        state, metrics = train_step(state, batch, step_key)
        if step == 1:
            jax.block_until_ready(metrics)
            print(f"First step done in {time.monotonic() - start:.1f}s")
        if step % config.log_interval_steps == 0:
            logs = {key: float(value) for key, value in jax.device_get(metrics).items()}
            logs["step"] = step
            wandb.log(logs, step=step)
            print(
                f"step {step} loss {logs['loss']:.3f} "
                f"public value {logs['loss_public_value']:.3f} "
                f"nats/token {logs.get('nats_per_token', float('nan')):.3f} "
                f"flow {logs.get('loss_flow', float('nan')):.3f} "
                f"mean {logs.get('loss_mean_control', float('nan')):.3f}"
            )
        if step % config.eval_interval_steps == 0:
            eval_batches = (
                jax.tree.map(jnp.asarray, b)
                for b in itertools.islice(dataset.eval_batches(), config.eval_batches)
            )
            eval_metrics = evaluate(state, eval_batches, jax.random.fold_in(rng, step))
            wandb.log(eval_metrics, step=step)
            print(
                f"eval step {step}: public value R2 "
                f"{eval_metrics.get('eval_public_value_r2', float('nan')):.3f} "
                f"nats/token {eval_metrics.get('eval_nats_per_token', float('nan')):.3f} "
                f"flow {eval_metrics.get('eval_loss_flow', float('nan')):.3f} "
                f"imagined R2 {eval_metrics.get('eval_imagined_value_r2', float('nan')):.3f}"
            )
            if config.world_model:
                score = eval_metrics.get(
                    "eval_loss_flow", float("inf")
                ) + eval_metrics.get("eval_nats_per_token", float("inf"))
            else:
                score = eval_metrics.get("eval_loss_public_value", float("inf"))
            if score < best_eval:
                best_eval = score
                save_artifact(config, state, step, shard_manifest, best=True)
        if step % config.save_interval_steps == 0:
            print(f"Saved {save_artifact(config, state, step, shard_manifest)}")
    save_artifact(config, state, config.num_steps, shard_manifest)
    wandb.finish()


if __name__ == "__main__":
    main()
