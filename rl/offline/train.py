"""Offline critic training on Pokemon Showdown replays.

Trains the player model's encoder + categorical value head to predict the
final game outcome from replay states (Monte-Carlo regression over the
public view), and saves artifacts in the RL checkpoint layout so the RL
learner can consume them directly (see rl/offline/artifact.py).

Usage:
    python -m rl.offline.train [--dataset-dir replays/shards] [--debug] ...
"""

import argparse
import itertools
import json
import os
import time
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

import wandb
from rl.environment.utils import get_ex_trajectory
from rl.learner import checkpoint as checkpoint_lib
from rl.model.config import get_player_model_config
from rl.model.utils import Params, get_num_params
from rl.offline.config import Porygon2OfflineConfig, get_offline_config
from rl.offline.dataset import MAX_MARGIN, OfflineBatch, OfflineDataset, prefetch
from rl.offline.model import MARGIN_SUPPORT, Porygon2OfflineCritic


def _value_mask(dones: jax.Array) -> jax.Array:
    """Valid steps: everything up to and including the FIRST done step.
    Written as an equality (not the learner's `1 - (cumsum - dones)`) so it
    stays in {0, 1} even if a trajectory somehow carries repeated dones."""
    dones = dones.astype(jnp.int32)
    return ((jnp.cumsum(dones, axis=0) - dones) == 0).astype(jnp.float32)


def _metrics_from_logits(
    logits: jax.Array,
    labels: jax.Array,
    mask: jax.Array,
    label_smoothing: float = 0.0,
) -> dict[str, jax.Array]:
    logits = logits.astype(jnp.float32)
    num_bins = logits.shape[-1]
    labels = jnp.broadcast_to(labels[None], logits.shape)
    if label_smoothing:
        smoothed = labels * (1.0 - label_smoothing) + label_smoothing / num_bins
    else:
        smoothed = labels
    ce = optax.softmax_cross_entropy(logits=logits, labels=smoothed)
    denom = mask.sum().clip(min=1.0)
    loss = (ce * mask).sum() / denom

    support = jnp.asarray(MARGIN_SUPPORT, dtype=jnp.float32)
    expectation = jax.nn.softmax(logits, axis=-1) @ support  # in [-1, 1]
    true_margin = labels.argmax(axis=-1).astype(jnp.int32) - MAX_MARGIN
    # Sign accuracy: comparable to the previous win/loss accuracy.
    correct = (expectation > 0) == (true_margin > 0)
    accuracy = (correct * mask).sum() / denom
    # What the margin head buys beyond sign: mean |error| in mons.
    margin_mae = (jnp.abs(expectation * MAX_MARGIN - true_margin) * mask).sum() / denom
    # Late-game diagnostic: the last valid step of each trajectory is
    # near-decisive, so sign accuracy here should climb toward ~0.9 quickly
    # if the pathway is wired correctly — long before the trajectory-average
    # loss visibly moves (early-game states are irreducible coin-flips).
    last_idx = jnp.maximum(mask.sum(axis=0).astype(jnp.int32) - 1, 0)
    batch_idx = jnp.arange(logits.shape[1])
    accuracy_last_step = correct[last_idx, batch_idx].mean()
    # Degeneracy canary: masked std of the expected margin. A model that
    # has collapsed to a constant (input-independent) prediction shows ~0
    # here while accuracy tracks batch label composition.
    exp_mean = (expectation * mask).sum() / denom
    margin_std = jnp.sqrt(((expectation - exp_mean) ** 2 * mask).sum() / denom)
    return dict(
        loss=loss,
        accuracy=accuracy,
        accuracy_last_step=accuracy_last_step,
        margin_mae=margin_mae,
        margin_std=margin_std,
        num_valid_steps=mask.sum(),
    )


def make_train_step(config: Porygon2OfflineConfig):
    @jax.jit
    def train_step(state: train_state.TrainState, batch: OfflineBatch):
        def loss_fn(params):
            value_head = state.apply_fn(params, batch.actor_input)
            mask = _value_mask(batch.actor_input.env.done)
            metrics = _metrics_from_logits(
                value_head.logits,
                batch.labels,
                mask,
                label_smoothing=config.label_smoothing,
            )
            return metrics["loss"], metrics

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        metrics["gradient_norm"] = optax.global_norm(grads)
        return state.apply_gradients(grads=grads), metrics

    return train_step


def make_eval_step():
    @jax.jit
    def eval_step(state: train_state.TrainState, batch: OfflineBatch):
        value_head = state.apply_fn(state.params, batch.actor_input)
        mask = _value_mask(batch.actor_input.env.done)
        return _metrics_from_logits(value_head.logits, batch.labels, mask)

    return eval_step


def evaluate(
    eval_step, state: train_state.TrainState, batches: Iterator[OfflineBatch]
) -> dict[str, float]:
    all_metrics = [jax.device_get(eval_step(state, batch)) for batch in batches]
    if not all_metrics:
        return {}
    weights = np.array([m["num_valid_steps"] for m in all_metrics])
    return {
        f"eval_{key}": float(np.average([m[key] for m in all_metrics], weights=weights))
        for key in (
            "loss",
            "accuracy",
            "accuracy_last_step",
            "margin_mae",
            "margin_std",
        )
    }


def save_artifact(
    config: Porygon2OfflineConfig,
    params: Params,
    step: int,
    best: bool = False,
) -> str:
    ckpt_name = "ckpt_best" if best else f"ckpt_{step:08}"
    format_dir = config.format_id
    if config.ensemble_index >= 0:
        format_dir = f"{config.format_id}-ens{config.ensemble_index}"
    save_path = os.path.abspath(
        os.path.join(config.artifact_root, format_dir, ckpt_name)
    )
    player_components = dict(
        params=params,
        scalars=dict(step_count=step),
    )
    checkpoint_lib.save_train_state(
        save_path,
        config,
        player_components,
        builder_state_components={},
        league_bytes=None,
    )
    # Manifest so downstream consumers know what this critic saw in
    # training. The observation is public-view only: private_team and
    # my_moveset were all-unspecified in every training example.
    with open(os.path.join(save_path, "manifest.json"), "w") as f:
        json.dump(
            dict(
                kind="offline_critic",
                format_id=config.format_id,
                public_view_only=True,
                step=step,
            ),
            f,
            indent=2,
        )
    return save_path


# CLI-overridable config fields and their argparse types. `adam` stays
# code-only; booleans are omitted on purpose.
_CLI_FIELDS: dict[str, type] = dict(
    generation=int,
    smogon_format=str,
    dataset_dir=str,
    holdout_modulus=int,
    shuffle_buffer_size=int,
    batch_size=int,
    min_history_length=int,
    min_trajectory_bucket=int,
    max_trajectory_length=int,
    num_steps=int,
    learning_rate=float,
    lr_final_fraction=float,
    label_smoothing=float,
    clip_gradient=float,
    log_interval_steps=int,
    eval_interval_steps=int,
    eval_batches=int,
    save_interval_steps=int,
    artifact_root=str,
    resume_from=str,
    ensemble_index=int,
    num_ensemble_splits=int,
    eval_gate_scale=float,
)


def parse_args() -> tuple[Porygon2OfflineConfig, int, bool]:
    config = get_offline_config()
    parser = argparse.ArgumentParser(description=__doc__)
    for name, arg_type in _CLI_FIELDS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=arg_type)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Train all num_ensemble_splits members simultaneously "
        "(vmapped; equivalent to running --ensemble-index 0..K-1)",
    )
    args = parser.parse_args()
    overrides = {
        name: getattr(args, name)
        for name in _CLI_FIELDS
        if getattr(args, name) is not None
    }
    if args.ensemble:
        overrides["train_ensemble"] = True
    config = config.replace(**overrides)
    if config.train_ensemble and config.ensemble_index >= 0:
        parser.error(
            "--ensemble trains every member; drop --ensemble-index "
            "(use --ensemble-index alone to retrain one member)"
        )
    if config.train_ensemble and config.resume_from is not None:
        parser.error(
            "--resume-from is per-member; retrain a single member "
            "with --ensemble-index instead"
        )
    return config, args.seed, args.debug


def save_member_artifact(
    config: Porygon2OfflineConfig,
    stacked_params: Params,
    member: int,
    step: int,
    best: bool = False,
) -> str:
    """Slices member ``member`` out of the stacked param tree and saves it
    in the standard per-member layout ({format_id}-ens{member}/...), so
    consumption via load_critic_params is identical to separately trained
    members."""
    member_params = jax.tree.map(lambda x: x[member], stacked_params)
    return save_artifact(
        config.replace(ensemble_index=member), member_params, step, best=best
    )


def evaluate_ensemble(
    eval_step, params: Params, batches: Iterator[OfflineBatch]
) -> tuple[dict[str, float], np.ndarray | None]:
    """Shared-holdout eval of every member plus gate diagnostics. Returns
    (wandb logs, per-member eval losses for best-checkpoint tracking)."""
    member_rows, gate_rows = [], []
    for batch in batches:
        per_member, gate = jax.device_get(eval_step(params, batch))
        member_rows.append(per_member)
        gate_rows.append(gate)
    if not member_rows:
        return {}, None
    weights = np.array([g["num_valid_steps"] for g in gate_rows])
    logs: dict[str, float] = {}
    member_losses = None
    for key in ("loss", "accuracy", "accuracy_last_step", "margin_mae"):
        values = np.average(
            np.stack([m[key] for m in member_rows]), axis=0, weights=weights
        )  # (K,)
        logs[f"eval_{key}_mean"] = float(values.mean())
        for k, value in enumerate(values):
            logs[f"eval_{key}_m{k}"] = float(value)
        if key == "loss":
            member_losses = values
    for key in ("gate_accuracy", "gate_member_std", "gate_abs_phi"):
        logs[f"eval_{key}"] = float(
            np.average([g[key] for g in gate_rows], weights=weights)
        )
    return logs, member_losses


def run_ensemble(config: Porygon2OfflineConfig, seed: int):
    """Trains all ``num_ensemble_splits`` members simultaneously.

    Pure parallelism over a leading member axis: stacked params and
    optimizer state, a vmapped member train step — per-member losses and
    gradients, never averaged across members, since member independence is
    exactly what the RL-side uncertainty gate measures — and one shard
    pass feeding every member (each record parsed once, routed by its
    ensemble bucket, identical splits to --ensemble-index runs). Shared-
    holdout evals score every member on the same batches and log live
    gate diagnostics (member std, gated sign accuracy) that separate runs
    only reveal after all members finish.
    """
    num_members = config.num_ensemble_splits
    model_config = get_player_model_config(config.generation, train=False)
    model = Porygon2OfflineCritic(model_config)
    # Same axis convention as the RL learner: leaves (T, B, ...), batch
    # mapped on axis 1. The member axis is vmapped outside of this.
    apply_fn = jax.vmap(model.apply, in_axes=(None, 1), out_axes=1)

    print(f"Initializing {num_members} members (traces the full encoder)...")
    ex_actor_input = jax.tree.map(jnp.asarray, get_ex_trajectory())
    member_keys = jax.random.split(jax.random.key(seed), num_members)
    params = jax.vmap(lambda key: model.init(key, ex_actor_input))(member_keys)

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.clip_gradient),
        optax.adamw(
            learning_rate=optax.cosine_decay_schedule(
                init_value=config.learning_rate,
                decay_steps=config.num_steps,
                alpha=config.lr_final_fraction,
            ),
            b1=config.adam.b1,
            b2=config.adam.b2,
            eps=config.adam.eps,
            weight_decay=config.adam.weight_decay,
        ),
    )
    opt_state = jax.vmap(optimizer.init)(params)

    def member_train_step(params, opt_state, batch: OfflineBatch):
        def loss_fn(p):
            value_head = apply_fn(p, batch.actor_input)
            mask = _value_mask(batch.actor_input.env.done)
            metrics = _metrics_from_logits(
                value_head.logits,
                batch.labels,
                mask,
                label_smoothing=config.label_smoothing,
            )
            return metrics["loss"], metrics

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        metrics["gradient_norm"] = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, metrics

    train_step = jax.jit(jax.vmap(member_train_step), donate_argnums=(0, 1))

    @jax.jit
    def eval_step(params, batch: OfflineBatch):
        # Holdout batches carry no member axis: every member scores the
        # same states, so member metrics are directly comparable and the
        # gate can be measured live.
        logits = jax.vmap(lambda p: apply_fn(p, batch.actor_input).logits)(params)
        mask = _value_mask(batch.actor_input.env.done)
        per_member = jax.vmap(
            lambda member_logits: _metrics_from_logits(
                member_logits, batch.labels, mask
            )
        )(logits)
        support = jnp.asarray(MARGIN_SUPPORT, dtype=jnp.float32)
        phi = jax.nn.softmax(logits.astype(jnp.float32), axis=-1) @ support
        std = phi.std(axis=0)
        gated = phi.mean(axis=0) * jnp.exp(-config.eval_gate_scale * std)
        true_margin = batch.labels.argmax(axis=-1).astype(jnp.int32) - MAX_MARGIN
        denom = mask.sum().clip(min=1.0)
        gated_correct = (gated > 0) == (true_margin > 0)[None]
        gate = dict(
            gate_accuracy=(gated_correct * mask).sum() / denom,
            gate_member_std=(std * mask).sum() / denom,
            gate_abs_phi=(jnp.abs(gated) * mask).sum() / denom,
            num_valid_steps=mask.sum(),
        )
        return per_member, gate

    dataset = OfflineDataset(config)
    wandb_run = wandb.init(
        project="pokemon-rl-offline",
        config=dict(
            offline_config=config,
            num_params=get_num_params(jax.tree.map(lambda x: x[0], params)),
            ensemble_size=num_members,
        ),
    )

    best_eval_loss = np.full(num_members, np.inf)
    batches = prefetch(dataset.train_batches_ensemble(seed=seed))
    print(
        f"Filling {num_members} member shuffle buffers "
        f"({config.shuffle_buffer_size // 2} games each) and JIT-compiling "
        "the first step — one-time startup cost..."
    )
    start_time = time.monotonic()
    for step, batch in enumerate(itertools.islice(batches, config.num_steps), start=1):
        dispatch_start = time.monotonic()
        params, opt_state, metrics = train_step(params, opt_state, batch)
        dispatch_time = time.monotonic() - dispatch_start
        if step == 1:
            jax.block_until_ready(metrics)
            print(f"First step done in {time.monotonic() - start_time:.1f}s.")
        elif dispatch_time > 2.0:
            print(
                f"step {step}: one-off recompile for new batch shape "
                f"(T={batch.actor_input.env.done.shape[1]}, "
                f"history={batch.actor_input.history.field.shape[1]}) "
                f"took {dispatch_time:.0f}s"
            )

        if step % config.log_interval_steps == 0:
            m = {k: np.asarray(v) for k, v in jax.device_get(metrics).items()}
            logs: dict[str, float] = {"step": step}
            for key, values in m.items():
                logs[f"{key}_mean"] = float(values.mean())
                for k in range(num_members):
                    logs[f"{key}_m{k}"] = float(values[k])
            wandb_run.log(logs, step=step)
            losses = " ".join(f"{v:.3f}" for v in m["loss"])
            print(
                f"step {step} | loss [{losses}] | "
                f"acc {m['accuracy'].mean():.3f} | "
                f"last-step acc {m['accuracy_last_step'].mean():.3f} | "
                f"grad norm {m['gradient_norm'].mean():.2e}"
            )
        if step % config.eval_interval_steps == 0:
            eval_logs, member_losses = evaluate_ensemble(
                eval_step,
                params,
                itertools.islice(dataset.eval_batches(), config.eval_batches),
            )
            if eval_logs:
                wandb_run.log(eval_logs, step=step)
                print(
                    f"step {step} | eval loss {eval_logs['eval_loss_mean']:.4f} "
                    f"| eval last-step acc "
                    f"{eval_logs['eval_accuracy_last_step_mean']:.3f} "
                    f"| gate acc {eval_logs['eval_gate_accuracy']:.3f} "
                    f"| member std {eval_logs['eval_gate_member_std']:.3f}"
                )
                for k in range(num_members):
                    if member_losses[k] < best_eval_loss[k]:
                        best_eval_loss[k] = member_losses[k]
                        best_path = save_member_artifact(
                            config, params, k, step, best=True
                        )
                        print(
                            f"member {k}: new best eval loss "
                            f"{member_losses[k]:.4f} — saved {best_path}"
                        )
        if step % config.save_interval_steps == 0:
            for k in range(num_members):
                save_member_artifact(config, params, k, step)
            print(f"Saved {num_members} member artifacts at step {step}")

    for k in range(num_members):
        save_member_artifact(config, params, k, config.num_steps)
    print(f"Done. Saved final artifacts for all {num_members} members.")


def main():
    config, seed, debug = parse_args()
    if debug:
        os.environ["WANDB_MODE"] = "disabled"

    if config.train_ensemble:
        run_ensemble(config, seed)
        return

    model_config = get_player_model_config(config.generation, train=False)
    model = Porygon2OfflineCritic(model_config)

    print("Initializing model (traces the full encoder — takes a minute)...")
    ex_actor_input = jax.tree.map(jnp.asarray, get_ex_trajectory())
    params = model.init(jax.random.key(seed), ex_actor_input)
    if config.resume_from is not None:
        params = checkpoint_lib.load_component(config.resume_from, "player", "params")
        print(f"Resumed params from {config.resume_from}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.clip_gradient),
        optax.adamw(
            learning_rate=optax.cosine_decay_schedule(
                init_value=config.learning_rate,
                decay_steps=config.num_steps,
                alpha=config.lr_final_fraction,
            ),
            b1=config.adam.b1,
            b2=config.adam.b2,
            eps=config.adam.eps,
            weight_decay=config.adam.weight_decay,
        ),
    )
    state = train_state.TrainState.create(
        # Same axis convention as the RL learner: leaves are (T, B, ...),
        # batch mapped on axis 1.
        apply_fn=jax.vmap(model.apply, in_axes=(None, 1), out_axes=1),
        params=params,
        tx=optimizer,
    )

    dataset = OfflineDataset(config)
    wandb_run = wandb.init(
        project="pokemon-rl-offline",
        config=dict(
            offline_config=config,
            num_params=get_num_params(state.params),
        ),
    )

    train_step = make_train_step(config)
    eval_step = make_eval_step()

    best_eval_loss = float("inf")
    batches = prefetch(dataset.train_batches(seed=seed))
    print(
        f"Filling shuffle buffer ({config.shuffle_buffer_size} trajectories) "
        "and JIT-compiling the first step — one-time startup cost..."
    )
    start_time = time.monotonic()
    last_save_path = None
    for step, batch in enumerate(itertools.islice(batches, config.num_steps), start=1):
        dispatch_start = time.monotonic()
        state, metrics = train_step(state, batch)
        # jit compiles synchronously at dispatch, so a slow dispatch on an
        # already-warm loop means this batch hit a new shape bucket.
        dispatch_time = time.monotonic() - dispatch_start
        if step == 1:
            jax.block_until_ready(metrics)
            print(
                f"First step done in {time.monotonic() - start_time:.1f}s. "
                f"Logs print every {config.log_interval_steps} steps. Early "
                "steps stall whenever a new (time, history) batch shape "
                "triggers a one-off recompile — announced below as they "
                "happen, then it's warm."
            )
        elif dispatch_time > 2.0:
            time_bucket = batch.actor_input.env.done.shape[0]
            history_bucket = batch.actor_input.history.field.shape[0]
            print(
                f"step {step}: one-off recompile for new batch shape "
                f"(T={time_bucket}, history={history_bucket}) "
                f"took {dispatch_time:.0f}s"
            )

        if step % config.log_interval_steps == 0:
            logs = {k: float(v) for k, v in jax.device_get(metrics).items()}
            logs["step"] = step
            wandb_run.log(logs, step=step)
            print(
                f"step {step} | loss {logs['loss']:.4f} | "
                f"acc {logs['accuracy']:.3f} | "
                f"last-step acc {logs['accuracy_last_step']:.3f} | "
                f"margin mae {logs['margin_mae']:.2f} | "
                f"margin std {logs['margin_std']:.2e} | "
                f"grad norm {logs['gradient_norm']:.2e}"
            )
        if step % config.eval_interval_steps == 0:
            eval_metrics = evaluate(
                eval_step,
                state,
                itertools.islice(dataset.eval_batches(), config.eval_batches),
            )
            if eval_metrics:
                wandb_run.log(eval_metrics, step=step)
                print(
                    f"step {step} | eval loss {eval_metrics['eval_loss']:.4f} "
                    f"| eval acc {eval_metrics['eval_accuracy']:.3f} "
                    f"| eval last-step acc "
                    f"{eval_metrics['eval_accuracy_last_step']:.3f}"
                )
                # A later plateau must never cost us an earlier peak.
                if eval_metrics["eval_loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["eval_loss"]
                    best_path = save_artifact(config, state.params, step, best=True)
                    print(
                        f"New best eval loss {best_eval_loss:.4f} — "
                        f"saved {best_path}"
                    )
        if step % config.save_interval_steps == 0:
            last_save_path = save_artifact(config, state.params, step)
            print(f"Saved artifact to {last_save_path}")

    last_save_path = save_artifact(config, state.params, config.num_steps)
    print(f"Done. Final artifact: {last_save_path}")


if __name__ == "__main__":
    main()
