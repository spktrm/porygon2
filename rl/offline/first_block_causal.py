"""Frozen first-block routing interventions; never writes player parameters."""

import json
import logging
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from rl.model.config import get_player_model_config
from rl.model.constants import (
    CLS_ROW,
    PRIVATE_ROWS,
    PUBLIC_ROWS,
    SEQUENCE_READ_MASK,
    SEQUENCE_SLICES,
    SequenceGroup,
)
from rl.model.trunk import TrunkBlock
from rl.offline.switch_depth_probe import ROOT as DEPTH_ROOT
from rl.offline.switch_depth_probe import split_masks
from rl.offline.switch_readout_probe import fit_readout
from rl.offline.type_probe import _OPP_ROW

ROOT = Path("runtime/type-probe-switch/first-block-causal")
ARMS = ("baseline", "info", "cls", "info_cls", "field", "mass_control", "skip")


def apply_intervention(params, sequence, valid, arm):
    block = TrunkBlock(get_player_model_config(9, train=True).encoder.trunk)
    original_mask = jnp.asarray(SEQUENCE_READ_MASK)
    info_rows = SEQUENCE_SLICES[SequenceGroup.INFO]
    sources = jnp.zeros(sequence.shape[-2], bool).at[CLS_ROW].set(True)
    sources = sources.at[info_rows].set(True)
    changed_mask = original_mask
    if arm in ("info", "info_cls"):
        changed_mask = changed_mask.at[PRIVATE_ROWS, info_rows].set(False)
    if arm in ("cls", "info_cls"):
        changed_mask = changed_mask.at[PRIVATE_ROWS, CLS_ROW].set(False)
    if arm == "field":
        changed_mask = changed_mask.at[
            PRIVATE_ROWS, SEQUENCE_SLICES[SequenceGroup.FIELD]
        ].set(False)

    def first(rows, present, block_params, block_index):
        def intercept(next_fun, args, kwargs, context):
            output = next_fun(*args, **kwargs)
            module = context.module
            if module.name == "attention" and context.method_name == "__call__":
                weights = module.get_variable("intermediates", "attn_weights")[0]
                mass = (weights * sources).sum(-1).mean(0)
                scale = (
                    jnp.ones(rows.shape[0])
                    .at[PRIVATE_ROWS]
                    .set(1 - jnp.where(block_index == 0, mass[PRIVATE_ROWS], 0))
                )
                output = output * scale[:, None].astype(output.dtype)
            return output

        def forward():
            return block.apply(
                {"params": block_params},
                (rows, present),
                jnp.where(block_index == 0, changed_mask, original_mask),
                mutable=["intermediates"],
            )

        if arm == "mass_control":
            with nn.intercept_methods(intercept):
                ((updated, _), _), captured = forward()
        else:
            ((updated, _), _), captured = forward()
        if arm == "skip":
            updated = jnp.where(block_index == 0, rows, updated)
        weights = captured["intermediates"]["attention"]["attn_weights"][0]
        mass = (weights * sources).sum(-1).mean(0)
        return updated, mass

    def step(current, inputs):
        block_params, block_index = inputs
        updated, stats = jax.vmap(first, in_axes=(0, 0, None, None))(
            current, valid, block_params, block_index
        )
        return updated, (updated, stats)

    _, (depths, statistics) = jax.lax.scan(
        step, sequence.astype(jnp.bfloat16), (params, jnp.arange(6))
    )
    return depths[jnp.asarray([0, 5])], statistics[0]


def main():
    if os.environ.get("COLLECT_INTERMEDIATES") != "1":
        raise RuntimeError("Set COLLECT_INTERMEDIATES=1 before import")
    if jax.default_backend() != "gpu":
        raise RuntimeError("Requires an idle GPU")
    ROOT.mkdir(parents=True, exist_ok=True)
    logging.getLogger("jax").setLevel(logging.ERROR)
    data = dict(np.load(DEPTH_ROOT / "states.npz"))
    reference = dict(np.load("runtime/type-probe-switch/frozen_02339569.npz"))
    with (DEPTH_ROOT / "original-trunk.pkl").open("rb") as handle:
        params = pickle.load(handle)
    baseline = np.load(DEPTH_ROOT / "frozen.npz")
    results_path = ROOT / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
    else:
        results = []
    args = SimpleNamespace(
        epochs=100, validation_every=1, l2=[0, 1e-5, 0.001, 0.01, 0.1]
    )
    for arm in ARMS:
        feature_path = ROOT / f"{arm}.npz"
        if not feature_path.exists():
            reader = jax.jit(
                lambda weights, rows, present: apply_intervention(
                    weights, rows, present, arm
                )
            )
            candidate = np.zeros(
                (2, len(reference["slot"]), data["sequence"].shape[-1]), np.float32
            )
            opponent = np.zeros_like(candidate)
            statistics = np.zeros((len(reference["slot"]),), np.float32)
            for start in range(0, len(data["sequence"]), 32):
                end = min(start + 32, len(data["sequence"]))
                selected = np.pad(
                    np.arange(start, end), (0, 32 - (end - start)), mode="edge"
                )
                depths, stats = reader(
                    params, data["sequence"][selected], data["valid"][selected]
                )
                depths = np.asarray(depths, np.float32)
                records = np.flatnonzero(
                    (data["record_state"] >= start) & (data["record_state"] < end)
                )
                local = data["record_state"][records] - start
                sheet = PRIVATE_ROWS.start + reference["slot"][records]
                candidate[:, records] = depths[:, local, sheet]
                opponent[:, records] = depths[:, local, PUBLIC_ROWS.start + _OPP_ROW]
                statistics[records] = np.asarray(stats)[local, sheet]
            if arm == "baseline":
                np.testing.assert_array_equal(candidate, baseline["candidate"][[1, 6]])
                np.testing.assert_array_equal(opponent, baseline["opponent"][[1, 6]])
            np.savez_compressed(
                feature_path,
                candidate=candidate,
                opponent=opponent,
                statistics=statistics,
            )
            print(f"extracted {arm}", flush=True)
        features = dict(np.load(feature_path))
        for seed in range(3):
            for depth_index, depth in enumerate((1, 6)):
                for label in ("offensive", "defensive"):
                    key = dict(arm=arm, seed=seed, depth=depth, label=label)
                    if any(
                        all(entry[name] == value for name, value in key.items())
                        for entry in results
                    ):
                        continue
                    if arm == "baseline":
                        previous = json.loads(
                            (DEPTH_ROOT / "curve-frozen.json").read_text()
                        )["results"]
                        result = next(
                            entry.copy()
                            for entry in previous
                            if entry["seed"] == seed
                            and entry["depth"] == depth
                            and entry["label"] == label
                            and entry["paired"]
                        )
                    else:
                        result = fit_readout(
                            features["candidate"][depth_index],
                            features["opponent"][depth_index],
                            reference[label].astype(np.int32),
                            split_masks(reference["game"], seed),
                            args,
                            seed,
                        )
                    result.update(key)
                    results.append(result)
                    results_path.write_text(json.dumps(results, indent=2) + "\n")
                    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
