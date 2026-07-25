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
from rl.model.utils import get_num_params
from rl.offline.config import Porygon2OfflineConfig, get_offline_config
from rl.offline.dataset import OfflineBatch, OfflineDataset, prefetch
from rl.offline.model import Porygon2OfflineCritic


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
    labels = jnp.broadcast_to(labels[None], logits.shape)
    if label_smoothing:
        smoothed = labels * (1.0 - label_smoothing) + label_smoothing / 3.0
    else:
        smoothed = labels
    ce = optax.softmax_cross_entropy(logits=logits, labels=smoothed)
    denom = mask.sum().clip(min=1.0)
    loss = (ce * mask).sum() / denom
    correct = logits.argmax(axis=-1) == labels.argmax(axis=-1)
    accuracy = (correct * mask).sum() / denom
    # Late-game diagnostic: the last valid step of each trajectory is
    # near-decisive, so accuracy here should climb toward ~0.9 quickly if
    # the history pathway is wired correctly — long before the trajectory-
    # average loss visibly moves (early-game states are irreducible
    # coin-flips that dominate the mean).
    last_idx = jnp.maximum(mask.sum(axis=0).astype(jnp.int32) - 1, 0)
    batch_idx = jnp.arange(logits.shape[1])
    accuracy_last_step = correct[last_idx, batch_idx].mean()
    # Degeneracy canary: masked std of the win-loss logit margin. A model
    # that has collapsed to a constant (input-independent) prediction shows
    # ~0 here while loss sits at ln2 and accuracy tracks batch label
    # composition — catch it at step 100, not after a full run.
    margin = logits[..., 2] - logits[..., 0]
    margin_mean = (margin * mask).sum() / denom
    margin_std = jnp.sqrt(((margin - margin_mean) ** 2 * mask).sum() / denom)
    return dict(
        loss=loss,
        accuracy=accuracy,
        accuracy_last_step=accuracy_last_step,
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
        for key in ("loss", "accuracy", "accuracy_last_step", "margin_std")
    }


def save_artifact(
    config: Porygon2OfflineConfig, state: train_state.TrainState, step: int
) -> str:
    save_path = os.path.abspath(
        os.path.join(config.artifact_root, config.format_id, f"ckpt_{step:08}")
    )
    player_components = dict(
        params=state.params,
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
    label_smoothing=float,
    clip_gradient=float,
    log_interval_steps=int,
    eval_interval_steps=int,
    eval_batches=int,
    save_interval_steps=int,
    artifact_root=str,
    resume_from=str,
)


def parse_args() -> tuple[Porygon2OfflineConfig, int, bool]:
    config = get_offline_config()
    parser = argparse.ArgumentParser(description=__doc__)
    for name, arg_type in _CLI_FIELDS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=arg_type)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()
    overrides = {
        name: getattr(args, name)
        for name in _CLI_FIELDS
        if getattr(args, name) is not None
    }
    return config.replace(**overrides), args.seed, args.debug


def main():
    config, seed, debug = parse_args()
    if debug:
        os.environ["WANDB_MODE"] = "disabled"

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
            learning_rate=config.learning_rate,
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
                    f"| eval acc {eval_metrics['eval_accuracy']:.3f}"
                )
        if step % config.save_interval_steps == 0:
            last_save_path = save_artifact(config, state, step)
            print(f"Saved artifact to {last_save_path}")

    last_save_path = save_artifact(config, state, config.num_steps)
    print(f"Done. Final artifact: {last_save_path}")


if __name__ == "__main__":
    main()
