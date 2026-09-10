"""Frozen-checkpoint sensitivity to the opt-in group input normaliser."""

import json
import logging
import pickle
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from rl.model.constants import (
    NUM_SEQUENCE_GROUPS,
    PRIVATE_ROWS,
    PUBLIC_ROWS,
    SEQUENCE_GROUP_IDS,
)
from rl.model.modules import SequenceInputNormalisation
from rl.offline.switch_depth_probe import ROOT as DEPTH_ROOT
from rl.offline.switch_depth_probe import split_masks, trunk_depths
from rl.offline.switch_readout_probe import fit_readout
from rl.offline.type_probe import _OPP_ROW

ROOT = Path("runtime/type-probe-switch/input-normalisation")


def normalised_depths(params, sequence, valid):
    normaliser = SequenceInputNormalisation(num_groups=NUM_SEQUENCE_GROUPS)
    normalised = normaliser.apply(
        {
            "params": {
                "group_scale": jnp.zeros((NUM_SEQUENCE_GROUPS, sequence.shape[-1]))
            }
        },
        sequence.astype(jnp.bfloat16),
        valid,
        jnp.asarray(SEQUENCE_GROUP_IDS),
    )
    return trunk_depths(params, normalised, valid)[jnp.asarray([0, 1, 6])]


def main():
    if jax.default_backend() != "gpu":
        raise RuntimeError("Requires an idle GPU")
    logging.getLogger("jax").setLevel(logging.ERROR)
    ROOT.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(DEPTH_ROOT / "states.npz"))
    reference = dict(np.load("runtime/type-probe-switch/frozen_02339569.npz"))
    with (DEPTH_ROOT / "original-trunk.pkl").open("rb") as handle:
        params = pickle.load(handle)
    reader = jax.jit(normalised_depths)
    feature_path = ROOT / "features.npz"
    if not feature_path.exists():
        candidate = np.zeros(
            (3, len(reference["slot"]), data["sequence"].shape[-1]), np.float32
        )
        opponent = np.zeros_like(candidate)
        for start in range(0, len(data["sequence"]), 32):
            end = min(start + 32, len(data["sequence"]))
            selected = np.pad(np.arange(start, end), (0, 32 - end + start), mode="edge")
            depths = np.asarray(
                reader(params, data["sequence"][selected], data["valid"][selected]),
                np.float32,
            )
            records = np.flatnonzero(
                (data["record_state"] >= start) & (data["record_state"] < end)
            )
            local = data["record_state"][records] - start
            candidate[:, records] = depths[
                :, local, PRIVATE_ROWS.start + reference["slot"][records]
            ]
            opponent[:, records] = depths[:, local, PUBLIC_ROWS.start + _OPP_ROW]
        np.savez_compressed(feature_path, candidate=candidate, opponent=opponent)
    features = dict(np.load(feature_path))
    args = SimpleNamespace(
        epochs=100, validation_every=1, l2=[0, 1e-5, 0.001, 0.01, 0.1]
    )
    results_path = ROOT / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
    else:
        results = []
    for seed in range(3):
        for depth_index, depth in enumerate((0, 1, 6)):
            for label in ("offensive", "defensive"):
                key = dict(seed=seed, depth=depth, label=label)
                if any(
                    all(entry[name] == value for name, value in key.items())
                    for entry in results
                ):
                    continue
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
