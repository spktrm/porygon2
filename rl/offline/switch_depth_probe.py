"""Depth accessibility and isolated intermediate-supervision experiments.

All labels and game splits follow switch_readout_probe. Only cached trunk
inputs are used for interventions; production parameters are never written.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rl.model.config import get_player_model_config
from rl.model.constants import PRIVATE_ROWS, PUBLIC_ROWS, SEQUENCE_READ_MASK
from rl.model.trunk import TrunkBlock
from rl.offline.switch_readout_probe import fit_readout, initial_params, logits, metrics
from rl.offline.type_probe import _OPP_ROW

ROOT = Path("runtime/type-probe-switch/depth")
CHECKPOINT = "ckpts/gen9/ckpt_02339569"
LABELS = ("offensive", "defensive")


def trunk_depths(params, sequence, valid):
    """Apply the actual checkpoint blocks; return depths 0 through 6."""
    block = TrunkBlock(get_player_model_config(9, train=True).encoder.trunk)
    read_mask = jnp.asarray(SEQUENCE_READ_MASK)

    def step(current, block_params):
        (updated, _), _ = jax.vmap(
            lambda rows, present: block.apply(
                {"params": block_params}, (rows, present), read_mask
            )
        )(current, valid)
        return updated, updated

    _, depths = jax.lax.scan(
        jax.checkpoint(step), sequence.astype(jnp.bfloat16), params
    )
    return jnp.concatenate([sequence[None].astype(jnp.bfloat16), depths], axis=0)


apply_depths = jax.jit(trunk_depths)


def extract(args):
    from rl.model.player_model import get_player_model
    from rl.offline import harness
    from rl.offline.separation_probe import actor_input_of
    from rl.offline.type_probe import TypeTables, label_switch_batch
    from rl.online.training.batching import stack_batch

    reference = np.load("runtime/type-probe-switch/frozen_02339569.npz")
    lookup = {
        tuple(key): index
        for index, key in enumerate(
            np.stack([reference[name] for name in ("game", "step", "slot")], axis=1)
        )
    }
    sides = harness.load("runtime/tactical-cohort/games.pkl")
    chunks, games = [], []
    for game_index, side in enumerate(sides):
        chunks.extend(side)
        games.extend([game_index] * len(side))
    net = get_player_model(get_player_model_config(9, train=True))
    variables = harness.load_params(CHECKPOINT)
    trunk_params = variables["params"]["encoder"]["trunk"]["blocks"]

    def assembled(module, actor_input):
        return module.encoder.assembled_sequence(
            actor_input.env, actor_input.packed_history, actor_input.history
        )

    assemble = jax.jit(
        jax.vmap(
            lambda params, inputs: net.apply(params, inputs, method=assembled),
            in_axes=(None, 1),
            out_axes=1,
        )
    )
    variables = jax.device_put(variables)
    tables = TypeTables("data/data")
    sequences, validity, state_games = [], [], []
    state_indices = np.full(len(lookup), -1, dtype=np.int32)
    candidate = np.empty((7, len(lookup), 256), np.float32)
    opponent = np.empty_like(candidate)
    for start in range(0, len(chunks), args.batch):
        group = chunks[start : start + args.batch]
        padded = group + [group[-1]] * (args.batch - len(group))
        stacked = stack_batch(padded)
        rows, valid = assemble(variables, actor_input_of(stacked))
        rows, valid = np.asarray(rows), np.asarray(valid)
        env = stacked.player_transitions.env_output
        done = np.asarray(env.done, bool)
        steps = np.cumsum(done, axis=0) == 0
        steps[:, len(group) :] = False
        steps[-1, : len(group)] = False
        labels = label_switch_batch(tables, env, steps)
        local_states = {}
        local_rows, local_valid = [], []
        for record in labels:
            time_index, batch_index = record["t"], record["b"]
            state_key = (time_index, batch_index)
            if state_key not in local_states:
                local_states[state_key] = len(local_rows)
                local_rows.append(rows[time_index, batch_index])
                local_valid.append(valid[time_index, batch_index])
                state_games.append(games[start + batch_index])
        if not local_rows:
            continue
        # Bounded state batches avoid a new compile for every chunk's record count.
        depth_parts = []
        for offset in range(0, len(local_rows), 32):
            batch_rows = np.asarray(local_rows[offset : offset + 32])
            batch_valid = np.asarray(local_valid[offset : offset + 32])
            count = len(batch_rows)
            batch_rows = np.pad(batch_rows, ((0, 32 - count), (0, 0), (0, 0)))
            batch_valid = np.pad(batch_valid, ((0, 32 - count), (0, 0)))
            depth_parts.append(
                np.asarray(
                    apply_depths(trunk_params, batch_rows, batch_valid), np.float32
                )[:, :count]
            )
        depths = np.concatenate(depth_parts, axis=1)
        for record in labels:
            time_index, batch_index = record["t"], record["b"]
            offset = int(np.asarray(group[batch_index].game_step_offset).item())
            key = (games[start + batch_index], offset + time_index, record["slot"])
            record_index = lookup[key]
            state_index = local_states[(time_index, batch_index)]
            state_indices[record_index] = len(sequences) + state_index
            candidate[:, record_index] = depths[
                :, state_index, PRIVATE_ROWS.start + record["slot"]
            ]
            opponent[:, record_index] = depths[
                :, state_index, PUBLIC_ROWS.start + _OPP_ROW
            ]
        sequences.extend(local_rows)
        validity.extend(local_valid)
        print(f"depth extraction {start + len(group)}/{len(chunks)}", flush=True)
    if np.any(state_indices < 0):
        raise AssertionError("Missing reference records")
    errors = {}
    for name, values, depth in (
        ("candidate_pre", candidate, 0),
        ("candidate_post", candidate, 6),
        ("opponent_pre", opponent, 0),
        ("opponent_post", opponent, 6),
    ):
        errors[name] = float(np.max(np.abs(values[depth] - reference[name])))
        np.testing.assert_allclose(values[depth], reference[name], rtol=0.02, atol=0.02)
    metadata = {
        "endpoint_max_abs_error": errors,
        "states": len(sequences),
        "cells": len(lookup),
    }
    np.savez_compressed(
        ROOT / "states.npz",
        sequence=np.asarray(sequences, np.float32),
        valid=np.asarray(validity),
        game=np.asarray(state_games),
        record_state=state_indices,
    )
    np.savez_compressed(ROOT / "frozen.npz", candidate=candidate, opponent=opponent)
    (ROOT / "extraction.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (ROOT / "original-trunk.pkl").open("wb") as handle:
        pickle.dump(jax.tree.map(np.asarray, trunk_params), handle)
    print(json.dumps(metadata), flush=True)


def split_masks(games, seed):
    permutation = np.random.default_rng(seed).permutation(np.unique(games))
    train_end = int(len(permutation) * 0.6)
    valid_end = int(len(permutation) * 0.8)
    return [
        np.isin(games, group)
        for group in (
            permutation[:train_end],
            permutation[train_end:valid_end],
            permutation[valid_end:],
        )
    ]


def depth_controls():
    from rl.model.constants import POLICY_READABLE_ROWS
    from rl.offline.separation_probe import _ridge_predict

    reference = np.load("runtime/type-probe-switch/frozen_02339569.npz")
    results = []
    for arm in ("frozen", "final", "intermediate"):
        for seed in range(3):
            if arm == "frozen":
                path = ROOT / "frozen.npz"
            else:
                path = ROOT / f"{arm}-{seed}-features.npz"
            if not path.exists():
                continue
            features = np.load(path)
            train, validation, test = split_masks(reference["game"], seed)
            selected = train | test
            for depth in range(7):
                for operand, label in (
                    ("candidate", "cand_type"),
                    ("opponent", "opp_type"),
                ):
                    labels = reference[label][selected]
                    targets = np.eye(int(reference[label].max()) + 1)[labels]
                    predictions, held = _ridge_predict(
                        features[operand][depth, selected].astype(np.float64),
                        targets,
                        train[selected],
                    )
                    results.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "depth": depth,
                            "operand": operand,
                            "accuracy": float(
                                np.mean(predictions.argmax(1) == held.argmax(1))
                            ),
                        }
                    )
    states = np.load(ROOT / "states.npz")
    sequence = jnp.asarray(states["sequence"][:2])
    valid = jnp.asarray(states["valid"][:2])
    with (ROOT / "original-trunk.pkl").open("rb") as handle:
        params = pickle.load(handle)
    original = np.asarray(apply_depths(params, sequence, valid), np.float32)
    private = np.setdiff1d(np.arange(sequence.shape[1]), POLICY_READABLE_ROWS)
    perturbed = sequence.at[:, private].add(37.5)
    changed = np.asarray(apply_depths(params, perturbed, valid), np.float32)
    np.testing.assert_array_equal(
        original[:, :, POLICY_READABLE_ROWS], changed[:, :, POLICY_READABLE_ROWS]
    )
    public_change = sequence.at[:, PUBLIC_ROWS.start + _OPP_ROW].add(3.0)
    public_changed = np.asarray(apply_depths(params, public_change, valid), np.float32)
    public_delta = float(
        np.max(
            np.abs(original[-1, :, PRIVATE_ROWS] - public_changed[-1, :, PRIVATE_ROWS])
        )
    )
    if public_delta <= 0:
        raise AssertionError("Public perturbation positive control is inert")
    gradient = jax.jit(
        jax.grad(
            lambda weights: jnp.square(
                trunk_depths(weights, sequence, valid)[2, :, PRIVATE_ROWS].astype(
                    jnp.float32
                )
            ).mean()
        )
    )(params)
    norms = [
        float(
            np.sqrt(
                sum(
                    np.square(np.asarray(leaf)[depth]).sum()
                    for leaf in jax.tree.leaves(gradient)
                )
            )
        )
        for depth in range(6)
    ]
    if not all(norm > 0 for norm in norms[:2]) or not all(
        norm == 0 for norm in norms[2:]
    ):
        raise AssertionError("Intermediate gradient does not stop at its own depth")
    output = {
        "primary_type": results,
        "privileged_policy_delta": 0.0,
        "public_candidate_delta": public_delta,
        "depth2_loss_gradient_norm_by_block": norms,
    }
    (ROOT / "controls.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "control_rows": len(results),
                "privileged_policy_delta": 0.0,
                "public_candidate_delta": public_delta,
                "depth2_gradient_norms": norms,
            }
        ),
        flush=True,
    )


def curve(args, arm="frozen", seeds=range(3)):
    reference = np.load("runtime/type-probe-switch/frozen_02339569.npz")
    output_path = ROOT / f"curve-{arm}.json"
    if output_path.exists():
        output = json.loads(output_path.read_text())
    else:
        output = {"arm": arm, "results": []}
    for seed in seeds:
        if arm == "frozen":
            features = np.load(ROOT / "frozen.npz")
        else:
            features = np.load(ROOT / f"{arm}-{seed}-features.npz")
        masks = split_masks(reference["game"], seed)
        for depth in range(7):
            for label_name in LABELS:
                for paired in (False, True):
                    key = (seed, depth, label_name, paired)
                    if any(
                        (entry["seed"], entry["depth"], entry["label"], entry["paired"])
                        == key
                        for entry in output["results"]
                    ):
                        continue
                    if arm != "frozen" and depth == 0:
                        baseline = json.loads((ROOT / "curve-frozen.json").read_text())
                        result = next(
                            entry
                            for entry in baseline["results"]
                            if (
                                entry["seed"],
                                entry["depth"],
                                entry["label"],
                                entry["paired"],
                            )
                            == key
                        )
                        output["results"].append(result)
                        output_path.write_text(json.dumps(output, indent=2) + "\n")
                        continue
                    if paired:
                        right = features["opponent"][depth]
                    else:
                        right = None
                    result = fit_readout(
                        features["candidate"][depth],
                        right,
                        reference[label_name].astype(np.int32),
                        masks,
                        args,
                        seed,
                    )
                    result.update(
                        seed=seed, depth=depth, label=label_name, paired=paired
                    )
                    output["results"].append(result)
                    output_path.write_text(json.dumps(output, indent=2) + "\n")
                    print(json.dumps({"arm": arm, **result}), flush=True)


def intervention_optimiser(trunk_rate):
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(
            {"trunk": optax.adam(trunk_rate), "heads": optax.adam(1e-3)},
            {"trunk": "trunk", "heads": "heads"},
        ),
    )


def supervised_predictions(params, sequence, valid, means, scales):
    depths = trunk_depths(params["trunk"], sequence, valid)[
        jnp.asarray([2, 4, 6])
    ].astype(jnp.float32)
    left = depths[:, :, PRIVATE_ROWS]
    right = depths[:, :, PUBLIC_ROWS.start + _OPP_ROW, None, :]
    right = jnp.broadcast_to(right, left.shape)
    left = (left - means[:, 0, None, None, :]) / scales[:, 0, None, None, :]
    right = (right - means[:, 1, None, None, :]) / scales[:, 1, None, None, :]

    def depth_head(head_params, candidate, opponent):
        def label_head(label_params):
            prediction = logits(
                label_params, candidate.reshape(-1, 256), opponent.reshape(-1, 256)
            )
            return prediction.reshape(candidate.shape[0], candidate.shape[1], 4)

        return jax.vmap(label_head)(head_params)

    return jax.vmap(depth_head)(params["heads"], left, right)


def supervised_objective(params, sequence, valid, labels, mask, means, scales, mode):
    predictions = supervised_predictions(params, sequence, valid, means, scales)
    if mode == "final":
        selected = jnp.asarray([2])
    else:
        selected = jnp.asarray([0, 1, 2])
    predictions = predictions[selected]
    targets = jnp.broadcast_to(labels.transpose(2, 0, 1)[None], predictions.shape[:-1])
    losses = optax.softmax_cross_entropy_with_integer_labels(predictions, targets)
    cross_entropy = jnp.sum(losses * mask[None, None]) / (
        len(selected) * 2 * jnp.maximum(mask.sum(), 1)
    )
    penalty = sum(
        jnp.sum(value[selected] ** 2)
        for name, value in params["heads"].items()
        if name != "bias"
    )
    return cross_entropy + 0.01 * penalty / (len(selected) * 2)


@functools.partial(jax.jit, static_argnames=("mode", "trunk_rate"))
def supervised_step(
    params, state, sequence, valid, labels, mask, means, scales, mode, trunk_rate
):
    loss, gradients = jax.value_and_grad(
        lambda weights: supervised_objective(
            weights, sequence, valid, labels, mask, means, scales, mode
        )
    )(params)
    optimiser = intervention_optimiser(trunk_rate)
    updates, state = optimiser.update(gradients, state, params)
    return optax.apply_updates(params, updates), state, loss


supervised_predict = jax.jit(supervised_predictions)


def evaluate_heads(params, data, indices, means, scales):
    predictions = []
    labels = []
    for start in range(0, len(indices), 32):
        selected = indices[start : start + 32]
        count = len(selected)
        selected = np.pad(selected, (0, 32 - count), mode="edge")
        values = np.asarray(
            supervised_predict(
                params,
                data["sequence"][selected],
                data["valid"][selected],
                means,
                scales,
            )
        )
        mask = data["mask"][selected[:count]]
        predictions.append(values[-1, :, :count][:, mask])
        labels.append(data["labels"][selected[:count]][mask])
    predictions = np.concatenate(predictions, axis=1)
    labels = np.concatenate(labels, axis=0)
    return {
        name: metrics(predictions[label_index], labels[:, label_index])
        for label_index, name in enumerate(LABELS)
    }


def selected_features(params, data, arm, seed):
    reference = np.load("runtime/type-probe-switch/frozen_02339569.npz")
    record_states = data["record_state"]
    candidate = np.empty((7, len(record_states), 256), np.float32)
    opponent = np.empty_like(candidate)
    for start in range(0, len(data["sequence"]), 32):
        end = min(start + 32, len(data["sequence"]))
        selected = np.arange(start, end)
        selected = np.pad(selected, (0, 32 - len(selected)), mode="edge")
        depths = np.asarray(
            apply_depths(params, data["sequence"][selected], data["valid"][selected]),
            np.float32,
        )
        records = np.flatnonzero((record_states >= start) & (record_states < end))
        local = record_states[records] - start
        candidate[:, records] = depths[
            :, local, PRIVATE_ROWS.start + reference["slot"][records]
        ]
        opponent[:, records] = depths[:, local, PUBLIC_ROWS.start + _OPP_ROW]
    np.savez_compressed(
        ROOT / f"{arm}-{seed}-features.npz", candidate=candidate, opponent=opponent
    )


def intervene(args):
    reference = np.load("runtime/type-probe-switch/frozen_02339569.npz")
    frozen = np.load(ROOT / "frozen.npz")
    with (ROOT / "original-trunk.pkl").open("rb") as handle:
        original = pickle.load(handle)
    data = dict(np.load(ROOT / "states.npz"))
    data["labels"] = np.zeros((len(data["sequence"]), 6, 2), np.int32)
    data["mask"] = np.zeros((len(data["sequence"]), 6), bool)
    data["mask"][data["record_state"], reference["slot"]] = True
    for label_index, name in enumerate(LABELS):
        data["labels"][data["record_state"], reference["slot"], label_index] = (
            reference[name]
        )
    output_path = ROOT / "interventions.json"
    if output_path.exists():
        output = json.loads(output_path.read_text())
    else:
        output = {"results": []}
    for seed in range(3):
        state_masks = split_masks(data["game"], seed)
        train_records = split_masks(reference["game"], seed)[0]
        means, scales = [], []
        for depth in (2, 4, 6):
            operands = [
                frozen[name][depth, train_records] for name in ("candidate", "opponent")
            ]
            means.append([values.mean(axis=0) for values in operands])
            scales.append([values.std(axis=0) + 1e-6 for values in operands])
        means, scales = np.asarray(means), np.asarray(scales)
        train_indices = np.flatnonzero(state_masks[0])
        validation_indices = np.flatnonzero(state_masks[1])
        for arm in ("final", "intermediate"):
            if any(
                row["seed"] == seed and row["arm"] == arm for row in output["results"]
            ):
                continue
            head_depths = []
            for depth in (2, 4, 6):
                label_heads = [
                    initial_params(256, 4, True, seed + 1000 * depth + label_index)
                    for label_index in range(2)
                ]
                head_depths.append(
                    jax.tree.map(lambda *values: jnp.stack(values), *label_heads)
                )
            heads = jax.tree.map(lambda *values: jnp.stack(values), *head_depths)
            best_loss = np.inf
            best = None
            selected_settings = None
            history = []
            for trunk_rate in (1e-5, 1e-4):
                params = {"trunk": jax.tree.map(jnp.asarray, original), "heads": heads}
                state = intervention_optimiser(trunk_rate).init(params)
                rng = np.random.default_rng(seed + 100)
                for epoch_index in range(args.trunk_epochs):
                    order = rng.permutation(train_indices)
                    order = order[: len(order) // 32 * 32].reshape(-1, 32)
                    for selected in order:
                        params, state, loss = supervised_step(
                            params,
                            state,
                            data["sequence"][selected],
                            data["valid"][selected],
                            data["labels"][selected],
                            data["mask"][selected],
                            means,
                            scales,
                            arm,
                            trunk_rate,
                        )
                    validation = evaluate_heads(
                        params, data, validation_indices, means, scales
                    )
                    validation_ce = np.mean(
                        [validation[name]["cross_entropy"] for name in LABELS]
                    )
                    if not np.isfinite(validation_ce) or not np.isfinite(float(loss)):
                        raise FloatingPointError("Non-finite supervised intervention")
                    history.append(
                        {
                            "lr": trunk_rate,
                            "epoch": epoch_index + 1,
                            "validation": validation,
                            "last_batch_objective": float(loss),
                        }
                    )
                    if validation_ce < best_loss:
                        best_loss = validation_ce
                        best = jax.tree.map(np.asarray, params)
                        selected_settings = {"lr": trunk_rate, "epoch": epoch_index + 1}
                    print(
                        json.dumps({"arm": arm, "seed": seed, **history[-1]}),
                        flush=True,
                    )
            with (ROOT / f"{arm}-{seed}-selected.pkl").open("wb") as handle:
                pickle.dump(best, handle)
            selected_features(best["trunk"], data, arm, seed)
            result = {
                "seed": seed,
                "arm": arm,
                "selection": selected_settings,
                "validation_ce": best_loss,
                "history": history,
                "test": evaluate_heads(
                    best, data, np.flatnonzero(state_masks[2]), means, scales
                ),
                "train": evaluate_heads(best, data, train_indices, means, scales),
            }
            output["results"].append(result)
            output_path.write_text(json.dumps(output, indent=2) + "\n")
    print("interventions complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("extract", "curve", "intervene", "refit", "controls")
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--trunk-epochs", type=int, default=20)
    parser.add_argument(
        "--arm", choices=("frozen", "final", "intermediate"), default="frozen"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--validation-every", type=int, default=1)
    parser.add_argument(
        "--l2", type=float, nargs="+", default=[0.0, 1e-5, 1e-3, 0.01, 0.1]
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    for logger_name in (
        "jax",
        "jax._src.compiler",
        "jax._src.dispatch",
        "jax._src.interpreters.partial_eval",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.mode == "extract":
        extract(args)
    elif args.mode == "intervene":
        intervene(args)
    elif args.mode == "refit":
        for arm in ("final", "intermediate"):
            curve(args, arm)
    elif args.mode == "controls":
        depth_controls()
    else:
        curve(args, args.arm, args.seeds)


if __name__ == "__main__":
    main()
