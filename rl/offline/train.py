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
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training import train_state

from rl.environment.utils import get_ex_trajectory
from rl.learner import checkpoint as checkpoint_lib
from rl.model.config import get_player_model_config
from rl.model.utils import get_num_params
from rl.offline.config import Porygon2OfflineConfig, get_offline_config
from rl.offline.dataset import OfflineBatch, OfflineDataset, prefetch
from rl.offline.model import Porygon2OfflineCritic


def _value_mask(dones: jax.Array) -> jax.Array:
    """Valid steps: everything up to and including the terminal step, the
    same convention as the RL learner's value mask."""
    return (1 - (jnp.cumsum(dones, axis=0) - dones)).astype(jnp.float32)


def _metrics_from_logits(
    logits: jax.Array, labels: jax.Array, mask: jax.Array
) -> dict[str, jax.Array]:
    logits = logits.astype(jnp.float32)
    labels = jnp.broadcast_to(labels[None], logits.shape)
    ce = optax.softmax_cross_entropy(logits=logits, labels=labels)
    denom = mask.sum().clip(min=1.0)
    loss = (ce * mask).sum() / denom
    accuracy = (
        (logits.argmax(axis=-1) == labels.argmax(axis=-1)) * mask
    ).sum() / denom
    return dict(loss=loss, accuracy=accuracy, num_valid_steps=mask.sum())


def make_train_step(config: Porygon2OfflineConfig):
    @jax.jit
    def train_step(state: train_state.TrainState, batch: OfflineBatch):
        def loss_fn(params):
            value_head = state.apply_fn(params, batch.actor_input)
            mask = _value_mask(batch.actor_input.env.done)
            metrics = _metrics_from_logits(value_head.logits, batch.labels, mask)
            return metrics["loss"], metrics

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
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
    all_metrics = [
        jax.device_get(eval_step(state, batch)) for batch in batches
    ]
    if not all_metrics:
        return {}
    weights = np.array([m["num_valid_steps"] for m in all_metrics])
    return {
        f"eval_{key}": float(
            np.average([m[key] for m in all_metrics], weights=weights)
        )
        for key in ("loss", "accuracy")
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
    parser.add_argument(
        "--debug", action="store_true", help="Disable wandb logging"
    )
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

    ex_actor_input = jax.tree.map(jnp.asarray, get_ex_trajectory())
    params = model.init(jax.random.key(seed), ex_actor_input)
    if config.resume_from is not None:
        params = checkpoint_lib.load_component(
            config.resume_from, "player", "params"
        )
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
    last_save_path = None
    for step, batch in enumerate(
        itertools.islice(batches, config.num_steps), start=1
    ):
        state, metrics = train_step(state, batch)

        if step % config.log_interval_steps == 0:
            logs = {k: float(v) for k, v in jax.device_get(metrics).items()}
            logs["step"] = step
            wandb_run.log(logs, step=step)
            print(
                f"step {step} | loss {logs['loss']:.4f} | "
                f"acc {logs['accuracy']:.3f}"
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