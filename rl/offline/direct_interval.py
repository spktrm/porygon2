"""Bounded direct next-value probes on frozen unilateral interval features."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("WANDB_MODE", "disabled")

import argparse
import json
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rl.model.config import get_player_model_config
from rl.model.interval_transition import DirectIntervalValue
from rl.offline import harness
from rl.offline.interval_data import evaluation_partitions
from rl.offline.train_interval import (
    game_bootstrap,
    interval_optimiser,
    prepare_output,
    run_arm,
    train_read_subset,
)
from rl.online.config import Porygon2LearnerConfig


def select_checkpoint(records):
    """Choose lowest development SSE relative to copy; earliest exact tie."""
    candidates = [record for record in records if record.get("split") == "validation"]
    if not candidates:
        raise ValueError("Checkpoint selection requires validation reads")
    return min(
        candidates, key=lambda record: (-record["prior_delta_gain"], record["step"])
    )


def make_evaluator(model, cfg):
    support = jnp.asarray(cfg.v_head.category_values, jnp.float32)
    actions = jnp.eye(cfg.transition.action_classes)

    def evaluate_one(params, rows, root_log_probs, probabilities, taken_cell):
        if model.action_representation == "rows":
            logits = model.apply({"params": params}, rows, taken_cell, root_log_probs)
            mixture = jax.nn.softmax(logits, axis=-1)
        elif model.action_conditioned:
            logits = model.apply({"params": params}, rows, actions, root_log_probs)
            mixture = probabilities @ jax.nn.softmax(logits, axis=-1)
        else:
            logits = model.apply({"params": params}, rows, actions[0], root_log_probs)
            mixture = jax.nn.softmax(logits, axis=-1)
        residual = logits - root_log_probs
        return (
            mixture @ support,
            jnp.sqrt(jnp.mean(residual**2)),
            jnp.max(jnp.abs(residual)),
        )

    return jax.jit(jax.vmap(evaluate_one, in_axes=(None, 0, 0, 0, 0)))


def read_direct(evaluate, params, arrays, indices, cfg, output, split, step):
    outputs = []
    for offset in range(0, len(indices), 32):
        selected = indices[offset : offset + 32]
        count = len(selected)
        selected = np.pad(selected, (0, 32 - count), mode="edge")
        prediction, residual_rms, residual_max = evaluate(
            params,
            jnp.asarray(arrays["rows"][selected], cfg.dtype),
            jnp.asarray(arrays["root_log_probs"][selected]),
            jnp.asarray(arrays["action_probs"][selected]),
            jnp.asarray(arrays["taken_cell"][selected]),
        )
        outputs.append(
            np.stack((prediction, residual_rms, residual_max), axis=-1)[:count]
        )
    outputs = np.concatenate(outputs)
    if not np.isfinite(outputs).all():
        raise FloatingPointError("Non-finite direct evaluation")
    error = (outputs[:, 0] - arrays["target_value"][indices]) ** 2
    energy = (arrays["root_value"][indices] - arrays["target_value"][indices]) ** 2
    result = {
        "step": step,
        "split": split,
        "prior_delta_gain": float(1 - error.sum() / max(energy.sum(), 1e-12)),
        "prior_gain95": game_bootstrap(error, energy, arrays["game"][indices]),
        "residual_rms": float(outputs[:, 1].mean()),
        "residual_max": float(outputs[:, 2].max()),
    }
    np.savez_compressed(
        Path(output) / f"{split}-{step:06d}.npz",
        error=error,
        energy=energy,
        predicted_value=outputs[:, 0],
        game=arrays["game"][indices],
        indices=indices,
    )
    return result


def train_direct(
    args,
    arrays,
    source,
    cfg,
    run_cfg,
    action_conditioned,
    action_representation="latent",
):
    output = prepare_output(Path(args.out))
    train_indices, validation_indices, final_indices = evaluation_partitions(arrays)
    if len(final_indices):
        raise ValueError("Development training must not receive final-test games")
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Training and validation games are required")
    if action_representation == "latent":
        arrays.setdefault("taken_cell", np.zeros(len(arrays["rows"]), dtype=np.int32))
    model = DirectIntervalValue(
        cfg.transition,
        len(cfg.v_head.category_values),
        cfg.dtype,
        action_conditioned,
        action_representation,
    )
    first = train_indices[0]
    if action_representation == "rows":
        initial_action = jnp.asarray(arrays["taken_cell"][first])
    else:
        initial_action = jax.nn.one_hot(0, cfg.transition.action_classes)
    variables = jax.jit(model.init)(
        jax.random.key(args.seed),
        jnp.asarray(arrays["rows"][first], cfg.dtype),
        initial_action,
        jnp.asarray(arrays["root_log_probs"][first]),
    )
    params = variables["params"]
    params["row_read"] = jax.tree.map(
        lambda leaf: jnp.array(leaf, copy=True),
        source["params"]["transition"]["row_read"],
    )
    if action_representation == "rows":
        params["action_projection"] = jax.tree.map(
            lambda leaf: jnp.array(leaf, copy=True),
            source["params"]["transition"]["action_encoder"]["query_proj"],
        )
    else:
        params["action_table"] = jnp.array(
            source["params"]["transition"]["action_table"], copy=True
        )
    optimiser = interval_optimiser(run_cfg.player_learning_rate)
    state = optimiser.init(params)
    manifest = {
        **vars(args),
        "action_conditioned": action_conditioned,
        "action_representation": action_representation,
        "parameters": sum(leaf.size for leaf in jax.tree.leaves(params)),
        "objective": "successor frozen critic distribution CE; centred logit residual from copy",
        "learning_rate": run_cfg.player_learning_rate,
        "train_games": len(np.unique(arrays["game"][train_indices])),
        "validation_games": len(np.unique(arrays["game"][validation_indices])),
        "train_intervals": len(train_indices),
        "action_expectation": "exact latent marginal or deterministic observed action rows",
    }

    def loss_fn(parameters, batch, key):
        if action_representation == "rows":
            actions = batch["taken_cell"]
        else:
            action_key, _ = jax.random.split(key)
            actions = jax.nn.one_hot(
                jax.random.categorical(action_key, jnp.log(batch["action_probs"])),
                cfg.transition.action_classes,
            )
        logits = jax.vmap(model.apply, in_axes=(None, 0, 0, 0))(
            {"params": parameters}, batch["rows"], actions, batch["root_log_probs"]
        )
        return optax.softmax_cross_entropy(logits, batch["target_probs"]).mean()

    @jax.jit
    def update(parameters, optimiser_state, batch, key):
        loss, gradient = jax.value_and_grad(loss_fn)(parameters, batch, key)
        updates, optimiser_state = optimiser.update(
            gradient, optimiser_state, parameters
        )
        return (
            optax.apply_updates(parameters, updates),
            optimiser_state,
            {"loss": loss, "gradient_norm": optax.global_norm(gradient)},
        )

    evaluate = make_evaluator(model, cfg)
    train_read = train_read_subset(train_indices, args.train_eval_limit, args.seed)

    def make_batch(indices):
        batch = {
            name: jnp.asarray(arrays[name][indices])
            for name in (
                "rows",
                "action_probs",
                "root_log_probs",
                "target_probs",
                "taken_cell",
            )
        }
        batch["rows"] = batch["rows"].astype(cfg.dtype)
        return batch

    def read_progress(step, params, seen):
        result = read_direct(
            evaluate,
            params,
            arrays,
            validation_indices,
            cfg,
            output,
            "validation",
            step,
        )
        result["train_read"] = read_direct(
            evaluate, params, arrays, train_read, cfg, output, "train", step
        )
        return result

    _, records = run_arm(
        output,
        output.name,
        manifest,
        params,
        state,
        update,
        make_batch,
        read_progress,
        train_indices,
        len(arrays["game"]),
        args.steps,
        args.batch_size,
        args.seed,
        set(args.eval_steps),
        log_every=0,
    )
    chosen = select_checkpoint(records)
    (output / "selected.json").write_text(json.dumps(chosen, indent=2))
    jax.clear_caches()
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--action-conditioned", action="store_true")
    parser.add_argument(
        "--action-representation", choices=("latent", "rows"), default="latent"
    )
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--eval-steps", type=int, nargs="+", default=(0, 25000))
    parser.add_argument("--train-eval-limit", type=int, default=1663)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    with np.load(args.data, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    source = harness.load_params(args.checkpoint)
    cfg = get_player_model_config(9, train=True)
    run_cfg = Porygon2LearnerConfig()
    train_direct(
        args,
        arrays,
        source,
        cfg,
        run_cfg,
        args.action_conditioned,
        args.action_representation,
    )


if __name__ == "__main__":
    main()
