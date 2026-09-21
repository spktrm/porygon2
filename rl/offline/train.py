"""The offline trainer: the player model's PUBLIC path -- encoder + trunk
on the 55 public rows and the public critic -- as one module whose
submodule names are the player model's, so `merge_params` resumes every
leaf by path at a learner relaunch. The encoder is overlaid from a learner
checkpoint and frozen unless `joint`; the critic trains on the replay export.

    env/bin/python -u rl/offline/train.py --trunk-ckpt ckpts/gen9/ckpt_XXXXXXXX
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
import optax
from ml_collections import ConfigDict

import wandb
from rl import checkpoint as checkpoint_lib
from rl.environment.event_labels import EventLabels
from rl.environment.interfaces import EventStates
from rl.model.config import get_player_model_config
from rl.model.constants import PUBLIC_CLS_LOCAL_ROW
from rl.model.encoder import Encoder
from rl.model.heads import CategoricalValueLogitHead
from rl.model.utils import get_num_params
from rl.offline.config import Porygon2OfflineConfig
from rl.offline.dataset import ReplayBatch, load_replay_store
from rl.online.artifact import merge_params

Params = dict


class TrainerState(NamedTuple):
    params: Params
    opt_state: optax.OptState
    step: jax.Array


class BatchTerms(NamedTuple):
    """Sums and counts over one trajectory, pooled over the batch by the
    caller -- every metric is a ratio of pooled sums, never a mean of
    per-trajectory ratios."""

    sums: dict
    counts: dict


def trainer_model_config(trunk_normalised_residual: bool = True) -> ConfigDict:
    cfg = get_player_model_config(generation=9, train=True)
    cfg.encoder.public_only = True
    cfg.encoder.trunk.normalised_residual = trunk_normalised_residual
    return cfg


def critic_buckets(
    step_valid: jax.Array, new_turn: jax.Array, order: jax.Array, num_valid: jax.Array
) -> dict[str, jax.Array]:
    """The step subsets the public critic is read over, keyed by metric
    suffix ("" = every valid step). `order` is the valid step's 1-based
    position in the window."""
    buckets = {
        "": step_valid,
        "_boundary": step_valid & new_turn,
        "_midturn": step_valid & ~new_turn,
        "_first_half": step_valid & (order <= 0.5 * num_valid),
        "_second_half": step_valid & (order > 0.5 * num_valid),
        "_final": step_valid & (order == num_valid),
    }
    return buckets


class OfflineTrainer(nn.Module):
    cfg: ConfigDict
    joint: bool = False

    def setup(self):
        self.encoder = Encoder(self.cfg.encoder)
        self.public_value_head = CategoricalValueLogitHead(self.cfg.public_value_head)

    def events(self, batch: ReplayBatch) -> EventStates:
        return self.encoder.encode_events(batch.packed_history, batch.history)

    def trajectory_terms(self, batch: ReplayBatch) -> BatchTerms:
        """Every loss term and metric numerator/denominator for ONE
        trajectory (leaves without the batch axis)."""
        events = self.events(batch)
        labels: EventLabels = batch.labels
        states = events.states
        if not self.joint:
            states = jax.lax.stop_gradient(states)
        sums = {}
        counts = {}

        def add(name, value, mask):
            sums[name] = jnp.sum(jnp.where(mask, value, 0.0))
            counts[name] = jnp.sum(mask.astype(jnp.float32))

        self.critic_terms(add, states, events.step_valid, labels, batch)
        return BatchTerms(sums=sums, counts=counts)

    def critic_terms(self, add, states, step_valid, labels, batch) -> None:
        """The public critic on real states, every valid step."""
        value = self.public_value_head(states[:, PUBLIC_CLS_LOCAL_ROW])
        outcome_bin = jnp.argmax(batch.win_reward)
        value_nll = -value.log_probs[:, outcome_bin]
        outcome_value = jnp.asarray([-1.0, 0.0, 1.0])[outcome_bin]
        add("loss_public_value", value_nll, step_valid)
        squared_error = jnp.square(value.expectation - outcome_value)
        # The window is the game's tail, so "first half" is the first half
        # of the window: the whole game when it fits in max_history_steps.
        order = jnp.cumsum(step_valid.astype(jnp.float32))
        num_valid = order[-1]
        for bucket, mask in critic_buckets(
            step_valid, labels.new_turn, order, num_valid
        ).items():
            add(f"value_residual{bucket}", squared_error, mask)
            add(
                f"outcome_value{bucket}",
                jnp.full_like(squared_error, outcome_value),
                mask,
            )
            add(
                f"outcome_value_sq{bucket}",
                jnp.full_like(squared_error, outcome_value**2),
                mask,
            )


def pooled_metrics(pooled: BatchTerms) -> dict:
    """Ratios of pooled sums."""
    sums, counts = pooled.sums, pooled.counts
    metrics = {}
    scalar_keys = [key for key, value in sums.items() if jnp.ndim(value) == 0]
    for key in scalar_keys:
        metrics[key] = sums[key] / jnp.maximum(counts[key], 1.0)
    buckets = [
        key.removeprefix("value_residual")
        for key in sums
        if key.startswith("value_residual")
    ]
    for bucket in buckets:
        outcome_var = metrics[f"outcome_value_sq{bucket}"] - jnp.square(
            metrics[f"outcome_value{bucket}"]
        )
        metrics[f"public_value_r2{bucket}"] = 1.0 - metrics[
            f"value_residual{bucket}"
        ] / jnp.maximum(outcome_var, 1e-8)
    return metrics


def loss_weights(config: Porygon2OfflineConfig) -> dict[str, float]:
    return {"loss_public_value": config.public_value_loss_weight}


def total_loss(config: Porygon2OfflineConfig, metrics: dict) -> jax.Array:
    total = None
    for name, weight in loss_weights(config).items():
        if total is None:
            total = weight * metrics[name]
        else:
            total = total + weight * metrics[name]
    return total


def batch_terms(model, params, batch: ReplayBatch):
    def one(trajectory):
        return model.apply(
            {"params": params}, trajectory, method=OfflineTrainer.trajectory_terms
        )

    # Sequential over trajectories (a scan, rematerialised): a vmap over 8
    # trajectories x 512 steps asked for 4.8 GB in one buffer.
    per_trajectory = jax.lax.map(jax.checkpoint(one), batch)
    return jax.tree.map(lambda x: x.sum(0), per_trajectory)


def make_train_step(config: Porygon2OfflineConfig, model, optimiser):
    @jax.jit
    def train_step(state: TrainerState, batch: ReplayBatch):
        def loss_fn(params):
            metrics = pooled_metrics(batch_terms(model, params, batch))
            return total_loss(config, metrics), metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, opt_state = optimiser.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        metrics["loss"] = loss
        metrics["gradient_norm"] = optax.global_norm(grads)
        return TrainerState(params, opt_state, state.step + 1), metrics

    return train_step


def make_eval_step(model):
    @jax.jit
    def eval_step(state: TrainerState, batch: ReplayBatch):
        return batch_terms(model, state.params, batch)

    def evaluate(state, batches: Iterator[ReplayBatch]) -> dict[str, float]:
        pooled = None
        for batch in batches:
            terms = jax.device_get(eval_step(state, batch))
            if pooled is None:
                pooled = terms
            else:
                pooled = jax.tree.map(lambda a, b: a + b, pooled, terms)
        if pooled is None:
            return {}
        metrics = pooled_metrics(BatchTerms(sums=pooled.sums, counts=pooled.counts))
        return {f"eval_{key}": float(value) for key, value in metrics.items()}

    return evaluate


def param_labels(params: Params, joint: bool) -> Params:
    trained = {"public_value_head"}
    if joint:
        trained.add("encoder")
    labels = {}
    for key, value in params.items():
        if key in trained:
            labels[key] = jax.tree.map(lambda _: "train", value)
        else:
            labels[key] = jax.tree.map(lambda _: "frozen", value)
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
    merged, kept_fresh, dropped, _ = merge_params(params, loaded)
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
            params={"params": state.params},
            scalars=dict(step_count=step),
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
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-history-steps", type=int, default=None)
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
        fresh_subtrees=tuple(args.fresh_subtrees),
    )
    for name in (
        "num_steps",
        "batch_size",
        "learning_rate",
        "max_history_steps",
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
    model = OfflineTrainer(
        trainer_model_config(config.trunk_normalised_residual), joint=config.joint
    )
    dataset = load_replay_store(config)
    shard_manifest = dataset.manifest
    print(f"{len(dataset)} trajectories, {len(dataset.train_games)} training games")
    first_batch = next(dataset.eval_batches())
    print("Initialising (traces the public encoder once)...")
    init_batch = jax.tree.map(lambda x: jnp.asarray(x[0]), first_batch)
    params = jax.jit(
        lambda key, batch: model.init(
            key, batch, method=OfflineTrainer.trajectory_terms
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
    if config.resume_from is not None:
        params = overlay_whole(
            params,
            checkpoint_lib.load_component(config.resume_from, "player", "params")[
                "params"
            ],
            config.resume_from,
        )
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
    state = TrainerState(params, optimiser.init(params), jnp.asarray(0))
    train_step = make_train_step(config, model, optimiser)
    evaluate = make_eval_step(model)
    wandb.init(
        project="pokemon-rl-offline",
        name=f"offline-{config.format_id}-s{seed}",
        config=dict(
            offline_config=dataclasses.asdict(config),
            num_params=get_num_params(params),
        ),
    )
    best_eval = float("inf")
    start = time.monotonic()
    batches = dataset.train_batches(seed=seed)
    for step, batch in enumerate(itertools.islice(batches, config.num_steps), start=1):
        batch = jax.tree.map(jnp.asarray, batch)
        state, metrics = train_step(state, batch)
        if step == 1:
            jax.block_until_ready(metrics)
            print(f"First step done in {time.monotonic() - start:.1f}s")
        if step % config.log_interval_steps == 0:
            logs = {key: float(value) for key, value in jax.device_get(metrics).items()}
            logs["step"] = step
            wandb.log(logs, step=step)
            print(
                f"step {step} loss {logs['loss']:.3f} "
                f"public value {logs['loss_public_value']:.3f}"
            )
        if step % config.eval_interval_steps == 0:
            eval_batches = (
                jax.tree.map(jnp.asarray, b)
                for b in itertools.islice(dataset.eval_batches(), config.eval_batches)
            )
            eval_metrics = evaluate(state, eval_batches)
            wandb.log(eval_metrics, step=step)
            print(
                f"eval step {step}: public value R2 "
                f"{eval_metrics.get('eval_public_value_r2', float('nan')):.3f} "
                f"(final {eval_metrics.get('eval_public_value_r2_final', float('nan')):.3f})"
            )
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
