"""Matched supervised switch readouts on cached, frozen representations.

Extraction expects the tactical heuristic cohort: one side per game. Labels
come from type_probe; no supervised signal is returned to the player model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax


def extract(args):
    from rl.model.config import get_player_model_config
    from rl.model.constants import CLS_ROW, PRIVATE_ROWS, PUBLIC_ROWS
    from rl.model.player_model import get_player_model
    from rl.offline import harness
    from rl.offline.separation_probe import (
        _assembled_and_encoded_fn,
        actor_input_of,
    )
    from rl.offline.type_probe import _OPP_ROW, TypeTables, label_switch_batch
    from rl.online.training.batching import stack_batch

    sides = harness.load(args.games_pkl)
    provenance_path = Path(args.games_pkl).with_name("provenance.json")
    provenance = json.loads(provenance_path.read_text())
    if len(sides) != provenance["games"]:
        raise ValueError("Expected one recorded side per heuristic game")
    chunks = []
    game_ids = []
    for game_index, side in enumerate(sides):
        chunks.extend(side)
        game_ids.extend([game_index] * len(side))
    net = get_player_model(get_player_model_config(9, train=True))
    variables = jax.device_put(harness.load_params(args.ckpt))
    apply_both = jax.jit(
        jax.vmap(
            lambda params, inputs: net.apply(
                params, inputs, method=_assembled_and_encoded_fn
            ),
            in_axes=(None, 1),
            out_axes=1,
        )
    )
    tables = TypeTables("data/data")
    records = {
        name: []
        for name in (
            "candidate_pre",
            "candidate_post",
            "opponent_pre",
            "opponent_post",
            "cls_post",
            "offensive",
            "defensive",
            "cand_type",
            "opp_type",
            "game",
            "step",
            "slot",
            "alive",
        )
    }
    for start in range(0, len(chunks), args.batch):
        group = chunks[start : start + args.batch]
        # Keep the extraction batch shape fixed, including the final batch.
        padded = group + [group[-1]] * (args.batch - len(group))
        stacked = stack_batch(padded)
        assembled, encoded = apply_both(variables, actor_input_of(stacked))
        assembled = np.asarray(assembled, dtype=np.float32)
        encoded = np.asarray(encoded, dtype=np.float32)
        env = stacked.player_transitions.env_output
        done = np.asarray(env.done, bool)
        steps = (np.cumsum(done, axis=0) == 0) & ~done
        steps[:, len(group) :] = False
        # The last row of a nonterminal chunk is bootstrap-only.
        steps[-1, : len(group)] = False
        for record in label_switch_batch(tables, env, steps):
            time_index, batch_index = record["t"], record["b"]
            candidate_row = PRIVATE_ROWS.start + record["slot"]
            opponent_row = PUBLIC_ROWS.start + _OPP_ROW
            for name, bank, row in (
                ("candidate_pre", assembled, candidate_row),
                ("candidate_post", encoded, candidate_row),
                ("opponent_pre", assembled, opponent_row),
                ("opponent_post", encoded, opponent_row),
                ("cls_post", encoded, CLS_ROW),
            ):
                records[name].append(bank[time_index, batch_index, row])
            for name in (
                "offensive",
                "defensive",
                "cand_type",
                "opp_type",
                "slot",
                "alive",
            ):
                records[name].append(record[name])
            chunk_index = start + batch_index
            offset = int(np.asarray(chunks[chunk_index].game_step_offset).item())
            records["game"].append(game_ids[chunk_index])
            records["step"].append(offset + time_index)
        print(f"extracted {start + len(group)}/{len(chunks)} chunks", flush=True)
    arrays = {name: np.asarray(values) for name, values in records.items()}
    keys = np.stack([arrays[name] for name in ("game", "step", "slot")], axis=1)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("Duplicate game/step/candidate records")
    arrays["metadata"] = np.asarray(
        json.dumps(
            {
                "checkpoint": args.ckpt,
                "parameters": "target_params",
                "cohort": args.games_pkl,
                "cohort_provenance": provenance,
                "games": len(sides),
                "extraction": "nonterminal acted rows; bootstrap overlap removed",
            }
        )
    )
    np.savez_compressed(args.cache, **arrays)
    print(f"saved {len(keys)} cells to {args.cache}", flush=True)


def logits(params, left, right):
    result = left @ params["linear"] + params["bias"]
    if "query" in params:
        query = jnp.einsum("nd,dcr->ncr", left, params["query"])
        key = jnp.einsum("nd,dcr->ncr", right, params["key"])
        result = result + jnp.sum(query * key, axis=-1) / 8.0
        result = result + right @ params["partner_linear"]
    return result


def initial_params(width, classes, paired, seed):
    rng = np.random.default_rng(seed)
    params = {
        "linear": jnp.zeros((width, classes)),
        "bias": jnp.zeros(classes),
    }
    if paired:
        params["query"] = jnp.zeros((width, classes, 64))
        params["key"] = jnp.asarray(
            rng.normal(size=(width, classes, 64)).astype(np.float32) / np.sqrt(width)
        )
        params["partner_linear"] = jnp.zeros((width, classes))
    return params


optimiser = optax.adam(1e-3)


@jax.jit
def epoch(params, state, left, right, labels, batches, regularisation):
    def update(carry, indices):
        current, opt_state = carry

        def objective(weights):
            predictions = logits(weights, left[indices], right[indices])
            loss = optax.softmax_cross_entropy_with_integer_labels(
                predictions, labels[indices]
            ).mean()
            penalty = sum(
                jnp.sum(value**2) for name, value in weights.items() if name != "bias"
            )
            return loss + regularisation * penalty

        gradients = jax.grad(objective)(current)
        updates, opt_state = optimiser.update(gradients, opt_state, current)
        current = optax.apply_updates(current, updates)
        return (current, opt_state), None

    (params, state), _ = jax.lax.scan(update, (params, state), batches)
    return params, state


@jax.jit
def cross_entropy(params, left, right, labels):
    return optax.softmax_cross_entropy_with_integer_labels(
        logits(params, left, right), labels
    ).mean()


predict = jax.jit(logits)


def metrics(predictions, labels):
    chosen = predictions.argmax(axis=1)
    recalls = []
    for label in np.unique(labels):
        recalls.append(float(np.mean(chosen[labels == label] == label)))
    shifted = predictions - predictions.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    expectation = probabilities @ np.arange(predictions.shape[1])
    correlation = 0.0
    if expectation.std() > 1e-8 and labels.std() > 0:
        correlation = float(np.corrcoef(expectation, labels)[0, 1])
    return {
        "accuracy": float(np.mean(chosen == labels)),
        "balanced_accuracy": float(np.mean(recalls)),
        "cross_entropy": float(
            -np.log(
                np.maximum(probabilities[np.arange(len(labels)), labels], 1e-30)
            ).mean()
        ),
        "ordinal_r": correlation,
        "recall": recalls,
    }


def controls(args):
    from rl.offline.separation_probe import _ridge_predict

    cache = np.load(args.cache)
    results = []
    for seed in range(3):
        games = np.random.default_rng(seed).permutation(np.unique(cache["game"]))
        train_games = games[: int(len(games) * 0.6)]
        test_games = games[int(len(games) * 0.8) :]
        selected = np.isin(cache["game"], np.concatenate([train_games, test_games]))
        train = np.isin(cache["game"][selected], train_games)
        for feature, label in (
            ("candidate_pre", "cand_type"),
            ("candidate_post", "cand_type"),
            ("opponent_pre", "opp_type"),
            ("opponent_post", "opp_type"),
            ("cls_post", "opp_type"),
        ):
            labels = cache[label][selected]
            targets = np.eye(int(cache[label].max()) + 1)[labels]
            predictions, held = _ridge_predict(
                cache[feature][selected].astype(np.float64), targets, train
            )
            results.append(
                {
                    "seed": seed,
                    "feature": feature,
                    "label": label,
                    "accuracy": float(np.mean(predictions.argmax(1) == held.argmax(1))),
                }
            )
    # A balanced interaction-only task: neither operand predicts the class.
    rng = np.random.default_rng(71)
    left_type = rng.integers(0, 4, size=4096)
    right_type = rng.integers(0, 4, size=4096)
    left = jnp.asarray(np.eye(4, dtype=np.float32)[left_type])
    right = jnp.asarray(np.eye(4, dtype=np.float32)[right_type])
    labels = jnp.asarray((left_type + right_type) % 4)
    synthetic = {}
    for paired in (False, True):
        params = initial_params(4, 4, paired, 0)
        state = optimiser.init(params)
        for epoch_index in range(100):
            batches = rng.permutation(3072).reshape(-1, 512)
            params, state = epoch(
                params, state, left[:3072], right[:3072], labels[:3072], batches, 0.0
            )
        synthetic[str(paired)] = metrics(
            np.asarray(predict(params, left[3072:], right[3072:])),
            np.asarray(labels[3072:]),
        )
    if synthetic["True"]["accuracy"] < 0.95 or synthetic["False"]["accuracy"] > 0.35:
        raise AssertionError("Interaction positive/negative control failed")
    output = {"type_controls": results, "synthetic_interaction": synthetic}
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


def fit_readout(left, right, labels, masks, args, seed):
    """Shared frozen readout fit; selection uses validation games only."""
    train, validation, test = masks
    paired = right is not None
    features = []
    for values in (left, right):
        if values is None:
            features.append(np.zeros_like(left))
        else:
            mean = values[train].mean(axis=0)
            scale = values[train].std(axis=0) + 1e-6
            features.append((values - mean) / scale)
    left, right = features
    splits = [
        tuple(jnp.asarray(values[mask]) for values in (left, right, labels))
        for mask in masks
    ]
    best_loss = np.inf
    best_params = None
    best_details = None
    for regularisation in args.l2:
        params = initial_params(left.shape[1], 4, paired, seed)
        state = optimiser.init(params)
        rng = np.random.default_rng(seed + 100)
        for epoch_index in range(args.epochs):
            indices = rng.permutation(train.sum())
            indices = indices[: len(indices) // 512 * 512].reshape(-1, 512)
            params, state = epoch(params, state, *splits[0], indices, regularisation)
            if (epoch_index + 1) % args.validation_every == 0:
                loss = float(cross_entropy(params, *splits[1]))
                if loss < best_loss:
                    best_loss = loss
                    best_params = jax.tree.map(np.asarray, params)
                    best_details = {"l2": regularisation, "epoch": epoch_index + 1}
    return {
        "selection": best_details,
        "validation_ce": best_loss,
        "test": metrics(np.asarray(predict(best_params, *splits[2][:2])), labels[test]),
        "train": metrics(
            np.asarray(predict(best_params, *splits[0][:2])), labels[train]
        ),
    }


def fit(args):
    cache = np.load(args.cache)
    output = {"metadata": json.loads(str(cache["metadata"])), "results": []}
    output["protocol"] = {
        "split": "60/20/20 percent whole games, seeds 0,1,2",
        "selection": "lowest validation cross entropy over epochs and L2 grid",
        "l2_grid": args.l2,
        "epochs": args.epochs,
        "validation_every": args.validation_every,
        "optimiser": "Adam lr=0.001, batch=512",
        "rank_per_class": 64,
        "features": "train-only coordinate standardisation; frozen f32 features",
    }
    arms = [
        ("candidate_post", "candidate_post", None),
        ("candidate_pre", "candidate_pre", None),
        ("pair_post", "candidate_post", "opponent_post"),
        ("pair_cls", "candidate_post", "cls_post"),
        ("pair_pre", "candidate_pre", "opponent_pre"),
    ]
    games = np.unique(cache["game"])
    for seed in range(3):
        permutation = np.random.default_rng(seed).permutation(games)
        train_end = int(len(games) * 0.6)
        valid_end = int(len(games) * 0.8)
        masks = [
            np.isin(cache["game"], group)
            for group in (
                permutation[:train_end],
                permutation[train_end:valid_end],
                permutation[valid_end:],
            )
        ]
        train, validation, test = masks
        for label_name in ("offensive", "defensive"):
            labels = cache[label_name].astype(np.int32)
            majority = int(np.bincount(labels[train], minlength=4).argmax())
            for arm, left_name, right_name in arms:
                if right_name is None:
                    right = None
                else:
                    right = cache[right_name]
                result = fit_readout(cache[left_name], right, labels, masks, args, seed)
                result.update(
                    {
                        "seed": seed,
                        "label": label_name,
                        "arm": arm,
                        "majority_accuracy": float(np.mean(labels[test] == majority)),
                        "test_cells": int(test.sum()),
                        "test_games": permutation[valid_end:].tolist(),
                    }
                )
                output["results"].append(result)
                Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
                print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("extract", "fit", "controls"))
    parser.add_argument("--games-pkl", default="runtime/tactical-cohort/games.pkl")
    parser.add_argument("--ckpt", default="ckpts/gen9/ckpt_02339569")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--cache", default="runtime/type-probe-switch/frozen_02339569.npz"
    )
    parser.add_argument(
        "--output", default="runtime/type-probe-switch/learned_02339569.json"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--l2", type=float, nargs="+", default=[0.0, 1e-5, 1e-3, 1e-2, 1e-1]
    )
    parser.add_argument("--validation-every", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    if args.mode == "extract":
        extract(args)
    elif args.mode == "controls":
        controls(args)
    else:
        fit(args)


if __name__ == "__main__":
    main()
