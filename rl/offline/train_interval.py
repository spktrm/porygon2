"""Matched offline interval ablation on a frozen policy and game-level split.

Legacy fine-tunes the original combined-chance predictor. Combined and history
share exactly the same conditional-code architecture, initial parameters and
batches; only first-posterior successor visibility differs. The fixed critic is
a diagnostic/teacher, not an estimated counterfactual ground truth. No policy,
league, live checkpoint or production learner is updated.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("WANDB_MODE", "disabled")

import argparse
import json
import logging
import time
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rl.model.categoricals import unimix_probs
from rl.model.config import get_player_model_config
from rl.model.constants import (
    CLS_ROW,
    LEARNER_ONLY_GROUPS,
    POLICY_READABLE_ROWS,
    SEQUENCE_GROUP_IDS,
    SequenceGroup,
)
from rl.model.heads import CategoricalValueLogitHead
from rl.model.interval_transition import IntervalTransition, warm_start_decoder
from rl.offline import harness
from rl.offline.interval_data import evaluation_partitions
from rl.online.config import Porygon2LearnerConfig
from rl.online.training.train_step import DYNAMICS_SCALE_FLOOR


def consistency_terms(rows, next_rows, imagined, valid):
    """Production's nonempty-group copy-relative loss on adjacent intervals."""
    errors = jnp.sum(jnp.square(imagined.astype(jnp.float32) - next_rows), axis=-1)
    energy = jnp.sum(jnp.square(next_rows - rows), axis=-1)
    group_ids = jnp.asarray(SEQUENCE_GROUP_IDS[POLICY_READABLE_ROWS])
    losses = []
    present = []
    for group in SequenceGroup:
        if group in LEARNER_ONLY_GROUPS:
            continue
        mask = valid & (group_ids == group)
        count = mask.sum()
        denominator = jnp.maximum(count, 1)
        mean_error = jnp.sum(jnp.where(mask, errors, 0)) / denominator
        mean_energy = jnp.sum(jnp.where(mask, energy, 0)) / denominator
        losses.append(mean_error / jnp.maximum(mean_energy, DYNAMICS_SCALE_FLOOR))
        present.append(count > 0)
    present = jnp.stack(present)
    return jnp.sum(jnp.where(present, jnp.stack(losses), 0)) / jnp.maximum(
        present.sum(), 1
    )


def balanced_kl(prior_logits, posterior_logits, dyn_coef, rep_coef, free_nats):
    prior = unimix_probs(prior_logits)
    posterior = unimix_probs(posterior_logits)
    target = jax.lax.stop_gradient(posterior)
    dynamics = jnp.sum(target * (jnp.log(target) - jnp.log(prior)), axis=(-2, -1))
    representation = jnp.sum(
        posterior * (jnp.log(posterior) - jnp.log(jax.lax.stop_gradient(prior))),
        axis=(-2, -1),
    )
    loss = dyn_coef * jnp.maximum(dynamics, free_nats) + rep_coef * jnp.maximum(
        representation, free_nats
    )
    return loss.mean(), dynamics.mean()


def game_bootstrap(error, energy, games, seed=0, replicates=2000):
    """Whole-game uncertainty; sum before dividing, including both perspectives."""
    identities = np.unique(games)
    totals = np.asarray(
        [
            [error[games == identity].sum(), energy[games == identity].sum()]
            for identity in identities
        ]
    )
    rng = np.random.default_rng(seed)
    selected = rng.integers(len(totals), size=(replicates, len(totals)))
    sums = totals[selected].sum(axis=1)
    gains = 1 - sums[:, 0] / np.maximum(sums[:, 1], 1e-12)
    return [float(value) for value in np.quantile(gains, [0.025, 0.975])]


def train_arm(args, arrays, source, cfg, run_cfg, evidence):
    output = Path(args.output) / evidence
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.jsonl").exists():
        raise FileExistsError(f"Refusing to overwrite experiment {output}")
    model = IntervalTransition(cfg.transition, cfg.dtype, evidence)
    critic = CategoricalValueLogitHead(cfg.v_head)
    critic_params = {"params": source["params"]["v_head"]}
    train_indices, heldout_indices, final_indices = evaluation_partitions(arrays)
    if not len(train_indices) or not len(heldout_indices):
        raise ValueError("Both training and held-out games are required")
    first = train_indices[0]
    init_action = jax.nn.one_hot(
        np.argmax(arrays["action_probs"][first]), cfg.transition.action_classes
    )
    initial = jax.jit(model.init)(
        jax.random.key(args.seed),
        jnp.asarray(arrays["rows"][first], cfg.dtype),
        init_action,
        jnp.asarray(arrays["next_rows"][first], cfg.dtype),
        jax.random.key(args.seed),
    )
    params, copied = warm_start_decoder(
        initial["params"], source["params"]["transition"]
    )
    optimiser = optax.chain(
        optax.clip_by_global_norm(10.0), optax.adam(run_cfg.player_learning_rate)
    )
    optimiser_state = optimiser.init(params)
    parameter_count = sum(leaf.size for leaf in jax.tree.leaves(params))
    manifest = {
        "arm": evidence,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "parameters": parameter_count,
        "copied_leaves": copied,
        "steps": args.steps,
        "train_intervals": len(train_indices),
        "heldout_intervals": len(heldout_indices),
        "train_games": len(np.unique(arrays["game"][train_indices])),
        "heldout_games": len(np.unique(arrays["game"][heldout_indices])),
        "final_test_games": len(np.unique(arrays["game"][final_indices])),
        "data": args.data,
        "eval_steps": args.eval_steps,
        "eval_every": args.eval_every,
        "train_eval_limit": args.train_eval_limit,
        "batch_size": args.batch_size,
        "prior_samples": args.prior_samples,
        "learning_rate": run_cfg.player_learning_rate,
        "consistency_coef": run_cfg.player_transition_cons_coef,
        "kl_dyn_coef": run_cfg.player_transition_dyn_coef,
        "kl_rep_coef": run_cfg.player_transition_rep_coef,
        "free_nats": run_cfg.player_transition_free_nats,
        "objective": "one-step consistency + fixed-next-critic distribution CE + balanced KL",
        "differences_from_production": "frozen actor/action encoder/critic; one-step; no generator, grounding or outcome fitting; isolated Adam",
        "behaviour_semantics": "unidentified; history rows mix contextual effects, not opponent submissions",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def loss_fn(parameters, batch, rng):
        action_key, posterior_key = jax.random.split(rng)
        actions = jax.nn.one_hot(
            jax.random.categorical(action_key, jnp.log(batch["action_probs"])),
            cfg.transition.action_classes,
        )
        prediction = jax.vmap(
            lambda rows, action, successor, key: model.apply(
                {"params": parameters}, rows, action, successor, key
            )
        )(
            batch["rows"],
            actions,
            batch["next_rows"],
            jax.random.split(posterior_key, args.batch_size),
        )
        consistency = consistency_terms(
            batch["rows"].astype(jnp.float32),
            batch["next_rows"].astype(jnp.float32),
            prediction.rows,
            batch["valid"],
        )
        target = jax.lax.stop_gradient(
            jnp.exp(
                critic.apply(critic_params, batch["next_rows"][:, CLS_ROW]).log_probs
            )
        )
        logits = critic.apply(critic_params, prediction.rows[:, CLS_ROW]).logits
        value_loss = optax.softmax_cross_entropy(logits, target).mean()
        kl_loss, kl = balanced_kl(
            prediction.prior_logits,
            prediction.posterior_logits,
            run_cfg.player_transition_dyn_coef,
            run_cfg.player_transition_rep_coef,
            run_cfg.player_transition_free_nats,
        )
        loss = run_cfg.player_transition_cons_coef * consistency + value_loss + kl_loss
        return loss, {"consistency": consistency, "value_ce": value_loss, "kl": kl}

    @jax.jit
    def update(parameters, state, batch, rng):
        (loss, logs), gradient = jax.value_and_grad(loss_fn, has_aux=True)(
            parameters, batch, rng
        )
        updates, state = optimiser.update(gradient, state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        return (
            parameters,
            state,
            {**logs, "loss": loss, "gradient_norm": optax.global_norm(gradient)},
        )

    def evaluate_one(parameters, rows, successor, action_probs, rng):
        root_value = critic.apply(critic_params, rows[CLS_ROW]).expectation
        target_value = critic.apply(critic_params, successor[CLS_ROW]).expectation

        def sample(key):
            action_key, chance_key = jax.random.split(key)
            action = jax.nn.one_hot(
                jax.random.categorical(action_key, jnp.log(action_probs)),
                cfg.transition.action_classes,
            )
            imagined = model.apply(
                {"params": parameters},
                rows,
                action,
                chance_key,
                method=model.sample_prior,
            )
            return critic.apply(critic_params, imagined[CLS_ROW]).expectation

        values = jax.lax.map(
            sample, jax.random.split(rng, args.prior_samples), batch_size=4
        )
        posterior_action_key, posterior_key = jax.random.split(rng)
        action = jax.nn.one_hot(
            jax.random.categorical(posterior_action_key, jnp.log(action_probs)),
            cfg.transition.action_classes,
        )
        posterior = model.apply(
            {"params": parameters}, rows, action, successor, posterior_key
        )
        posterior_value = critic.apply(
            critic_params, posterior.rows[CLS_ROW]
        ).expectation
        return jnp.stack(
            (
                (values.mean() - target_value) ** 2,
                (root_value - target_value) ** 2,
                (posterior_value - target_value) ** 2,
                values.std(),
            )
        )

    evaluate = jax.jit(
        lambda parameters, rows, successors, probabilities, keys: jax.lax.map(
            lambda inputs: evaluate_one(parameters, *inputs),
            (rows, successors, probabilities, keys),
        )
    )

    def read_split(step, evaluation_indices, split):
        parts = []
        for offset in range(0, len(evaluation_indices), args.batch_size):
            selected = evaluation_indices[offset : offset + args.batch_size]
            valid_count = len(selected)
            selected = np.pad(selected, (0, args.batch_size - valid_count), mode="edge")
            keys = jax.vmap(
                lambda index: jax.random.fold_in(jax.random.key(args.seed + 1), index)
            )(jnp.asarray(selected))
            read = evaluate(
                params,
                jnp.asarray(arrays["rows"][selected], cfg.dtype),
                jnp.asarray(arrays["next_rows"][selected], cfg.dtype),
                jnp.asarray(arrays["action_probs"][selected]),
                keys,
            )
            parts.append(np.asarray(read)[:valid_count])
        reads = np.concatenate(parts)
        if not np.isfinite(reads).all():
            raise FloatingPointError("Non-finite evaluation diagnostics")
        result = {
            "step": step,
            "split": split,
            "prior_delta_gain": float(
                1 - reads[:, 0].sum() / max(reads[:, 1].sum(), 1e-12)
            ),
            "posterior_delta_gain": float(
                1 - reads[:, 2].sum() / max(reads[:, 1].sum(), 1e-12)
            ),
            "prior_sigma": float(reads[:, 3].mean()),
            "prior_gain95": game_bootstrap(
                reads[:, 0], reads[:, 1], arrays["game"][evaluation_indices]
            ),
        }
        if split == "validation":
            filename = f"evaluation-{step:06d}.npz"
        else:
            filename = f"{split}-{step:06d}.npz"
        np.savez_compressed(
            output / filename,
            reads=reads,
            game=arrays["game"][evaluation_indices],
            indices=evaluation_indices,
        )
        return result

    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    diagnostic_rng = np.random.default_rng(args.seed + 2)
    train_read_indices = diagnostic_rng.choice(
        train_indices, min(args.train_eval_limit, len(train_indices)), replace=False
    )
    seen = np.zeros(len(arrays["game"]), dtype=bool)

    def read_progress(step):
        result = read_split(step, heldout_indices, "validation")
        if len(train_read_indices):
            result["train_read"] = read_split(step, train_read_indices, "train")
        result["sampled_passes"] = step * args.batch_size / len(train_indices)
        result["unique_sampled_intervals"] = int(seen.sum())
        result["unique_sampled_games"] = len(np.unique(arrays["game"][seen]))
        return result

    with (output / "metrics.jsonl").open("w") as stream:
        initial_read = read_progress(0)
        stream.write(json.dumps(initial_read) + "\n")
        stream.flush()
        print(evidence, json.dumps(initial_read), flush=True)
        for step in range(1, args.steps + 1):
            indices = rng.choice(train_indices, args.batch_size)
            seen[indices] = True
            batch = {
                name: jnp.asarray(arrays[name][indices])
                for name in ("rows", "next_rows", "valid", "action_probs")
            }
            batch["rows"] = batch["rows"].astype(cfg.dtype)
            batch["next_rows"] = batch["next_rows"].astype(cfg.dtype)
            params, optimiser_state, logs = update(
                params,
                optimiser_state,
                batch,
                jax.random.fold_in(jax.random.key(args.seed), step),
            )
            numbers = {name: float(value) for name, value in logs.items()}
            if not all(np.isfinite(value) for value in numbers.values()):
                raise FloatingPointError(f"Non-finite update at {step}: {numbers}")
            if step % 100 == 0:
                stream.write(json.dumps({"step": step, **numbers}) + "\n")
                stream.flush()
                print(evidence, step, json.dumps(numbers), flush=True)
            if (
                step % args.eval_every == 0
                or step in args.eval_steps
                or step == args.steps
            ):
                result = {
                    **read_progress(step),
                    **numbers,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                print(evidence, json.dumps(result), flush=True)
                destination = output / f"state-{step:06d}.msgpack"
                temporary = destination.with_suffix(f".tmp-{os.getpid()}")
                temporary.write_bytes(
                    flax.serialization.to_bytes(
                        {"params": params, "optimiser": optimiser_state, "step": step}
                    )
                )
                temporary.replace(destination)
        if len(final_indices):
            final_read = read_split(args.steps, final_indices, "final_test")
            stream.write(json.dumps(final_read) + "\n")
            stream.flush()
            print(evidence, json.dumps(final_read), flush=True)
    jax.clear_caches()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("legacy", "combined", "history"),
        default=("legacy", "combined", "history"),
    )
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--eval-steps", type=int, nargs="*", default=())
    parser.add_argument("--train-eval-limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prior-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Interval training requires GPU")
    manifest = json.loads(Path(args.data).with_name("features.json").read_text())
    if manifest["checkpoint"] != args.checkpoint:
        raise ValueError(
            "Frozen features and critic must come from the same checkpoint"
        )
    collection = json.loads(Path(manifest["collection"]).read_text())
    if collection["purpose"] != "research_training" or collection["is_eval"]:
        raise ValueError("Evaluation data is ineligible for training")
    with np.load(args.data, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    source = harness.load_params(args.checkpoint)
    cfg = get_player_model_config(9, train=True)
    run_cfg = Porygon2LearnerConfig()
    for evidence in args.arms:
        train_arm(args, arrays, source, cfg, run_cfg, evidence)


if __name__ == "__main__":
    main()
