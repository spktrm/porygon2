"""Bounded direct next-value probes on frozen unilateral interval features."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("WANDB_MODE", "disabled")

import json
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rl.model.constants import CLS_ROW
from rl.model.heads import CategoricalValueLogitHead
from rl.model.interval_transition import DirectIntervalValue
from rl.offline.interval_data import evaluation_partitions
from rl.offline.train_interval import game_bootstrap


def select_checkpoint(records):
    """Choose lowest development SSE relative to copy; earliest exact tie."""
    candidates = [record for record in records if record.get("split") == "validation"]
    if not candidates:
        raise ValueError("Checkpoint selection requires validation reads")
    return min(
        candidates, key=lambda record: (-record["prior_delta_gain"], record["step"])
    )


def prepare_targets(arrays, source, cfg):
    """Only the loss/evaluator receives successor critic outputs."""
    critic = CategoricalValueLogitHead(cfg.v_head)
    variables = {"params": source["params"]["v_head"]}
    apply = jax.jit(lambda rows: critic.apply(variables, rows).log_probs)
    for name, field in [("root_log_probs", "rows"), ("target_log_probs", "next_rows")]:
        outputs = []
        for offset in range(0, len(arrays[field]), 512):
            rows = arrays[field][offset : offset + 512, CLS_ROW]
            count = len(rows)
            rows = np.pad(rows, ((0, 512 - count), (0, 0)), mode="edge")
            outputs.append(np.asarray(apply(jnp.asarray(rows, cfg.dtype)))[:count])
        arrays[name] = np.concatenate(outputs)
    arrays["target_probs"] = np.exp(arrays["target_log_probs"])
    support = np.asarray(cfg.v_head.category_values, dtype=np.float32)
    arrays["root_value"] = np.exp(arrays["root_log_probs"]) @ support
    arrays["target_value"] = arrays["target_probs"] @ support


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
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.jsonl").exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
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
    optimiser = optax.chain(
        optax.clip_by_global_norm(10), optax.adam(run_cfg.player_learning_rate)
    )
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
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))

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
            loss,
            optax.global_norm(gradient),
        )

    evaluate = make_evaluator(model, cfg)
    rng = np.random.default_rng(args.seed)
    train_read = np.random.default_rng(args.seed + 2).choice(
        train_indices, 1663, replace=False
    )
    records = []
    with (output / "metrics.jsonl").open("w") as stream:
        for step in range(args.steps + 1):
            if step:
                indices = rng.choice(train_indices, args.batch_size)
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
                params, state, loss, gradient_norm = update(
                    params,
                    state,
                    batch,
                    jax.random.fold_in(jax.random.key(args.seed), step),
                )
                if not np.isfinite(float(loss)) or not np.isfinite(
                    float(gradient_norm)
                ):
                    raise FloatingPointError(f"Non-finite direct update {step}")
            if step in args.eval_steps:
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
                if step:
                    result["loss"] = float(loss)
                    result["gradient_norm"] = float(gradient_norm)
                records.append(result)
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                print(output.name, json.dumps(result), flush=True)
                destination = output / f"state-{step:06d}.msgpack"
                temporary = destination.with_suffix(f".tmp-{os.getpid()}")
                temporary.write_bytes(
                    flax.serialization.to_bytes(
                        {"params": params, "optimiser": state, "step": step}
                    )
                )
                temporary.replace(destination)
    chosen = select_checkpoint(records)
    (output / "selected.json").write_text(json.dumps(chosen, indent=2))
    jax.clear_caches()
    return chosen
