"""Separate descriptive pass so value reconstruction cannot change ablation numerics."""

import json
import logging
import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from rl.model.config import get_player_model_config
from rl.model.constants import CLS_ROW, PRIVATE_ROWS, SEQUENCE_SLICES, SequenceGroup
from rl.model.modules import RMSNorm
from rl.offline.attention_routes import capture, groups_and_rows
from rl.offline.switch_depth_probe import apply_depths


def main():
    if os.environ.get("COLLECT_INTERMEDIATES") != "1":
        raise RuntimeError("Set COLLECT_INTERMEDIATES=1 before import")
    if jax.default_backend() != "gpu":
        raise RuntimeError("Requires an idle GPU")
    root = Path("runtime/type-probe-switch/first-block-causal")
    logging.getLogger("jax").setLevel(logging.ERROR)
    data = dict(np.load("runtime/type-probe-switch/depth/states.npz"))
    reference = dict(np.load("runtime/type-probe-switch/frozen_02339569.npz"))
    with open("runtime/type-probe-switch/depth/original-trunk.pkl", "rb") as handle:
        params = pickle.load(handle)
    config = get_player_model_config(9, train=True).encoder.trunk
    first_params = jax.tree.map(lambda value: value[0], params)

    @jax.jit
    def value_projection(rows):
        normalised = RMSNorm().apply(
            {"params": first_params["RMSNorm_1"]}, rows.astype(jnp.bfloat16)
        )
        projection_params = first_params["attention"]["v_proj"]
        return nn.Dense(
            config.num_heads * config.v_size,
            use_bias="bias" in projection_params,
            dtype=jnp.bfloat16,
        ).apply({"params": projection_params}, normalised)

    reader = jax.jit(capture)
    norms = np.zeros((7, len(data["sequence"]), data["sequence"].shape[1]), np.float32)
    contributions = np.zeros((len(reference["slot"]), 5), np.float32)
    sources = np.zeros(data["sequence"].shape[1], bool)
    sources[CLS_ROW] = True
    sources[SEQUENCE_SLICES[SequenceGroup.INFO]] = True
    kernel = np.asarray(
        jnp.asarray(first_params["attention"]["out_proj"]["kernel"], jnp.bfloat16),
        np.float32,
    )
    for start in range(0, len(data["sequence"]), 32):
        end = min(start + 32, len(data["sequence"]))
        selected = np.pad(np.arange(start, end), (0, 32 - end + start), mode="edge")
        rows = data["sequence"][selected]
        valid = data["valid"][selected]
        depths = np.asarray(apply_depths(params, rows, valid), np.float32)
        norms[:, start:end] = np.linalg.norm(depths[:, : end - start], axis=-1)
        final, weights = reader(params, rows, valid)
        np.testing.assert_array_equal(np.asarray(final, np.float32), depths[-1])
        weights = np.asarray(weights[0], np.float32)
        values = np.asarray(value_projection(rows), np.float32).reshape(
            32, rows.shape[1], config.num_heads, config.v_size
        )
        route = (
            np.einsum("bhqt,bthd,t->bqhd", weights, values, sources).reshape(
                32, rows.shape[1], config.num_heads * config.v_size
            )
            @ kernel
        )
        total = (
            np.einsum("bhqt,bthd->bqhd", weights, values).reshape(
                32, rows.shape[1], config.num_heads * config.v_size
            )
            @ kernel
        )
        if "bias" in first_params["attention"]["out_proj"]:
            total += np.asarray(first_params["attention"]["out_proj"]["bias"])
        records = np.flatnonzero(
            (data["record_state"] >= start) & (data["record_state"] < end)
        )
        local = data["record_state"][records] - start
        sheet = PRIVATE_ROWS.start + reference["slot"][records]
        contributions[records] = np.stack(
            [
                np.linalg.norm(route[local, sheet], axis=-1),
                np.linalg.norm(total[local, sheet], axis=-1),
                norms[0, data["record_state"][records], sheet],
                np.linalg.norm(
                    depths[1, local, sheet] - depths[0, local, sheet], axis=-1
                ),
                np.linalg.norm(
                    depths[1, local, sheet]
                    - depths[0, local, sheet]
                    - total[local, sheet],
                    axis=-1,
                ),
            ],
            axis=-1,
        )
    np.savez_compressed(
        root / "diagnostics.npz", norms=norms, contributions=contributions
    )
    groups, _ = groups_and_rows()
    summary = {"token_norms": [], "contributions": {}}
    for depth in range(7):
        values = norms[depth][data["valid"]]
        overall_median = float(np.median(values))
        for group in groups:
            values = norms[depth][:, group["rows"]][data["valid"][:, group["rows"]]]
            if len(values):
                summary["token_norms"].append(
                    dict(
                        depth=depth,
                        group=group["name"],
                        median=float(np.median(values)),
                        p95=float(np.quantile(values, 0.95)),
                        maximum=float(values.max()),
                        overall_median=overall_median,
                    )
                )
    for name, values in [
        ("route_norm", contributions[:, 0]),
        ("attention_norm", contributions[:, 1]),
        ("residual_norm", contributions[:, 2]),
        ("block_update_norm", contributions[:, 3]),
        ("approx_mlp_norm", contributions[:, 4]),
        (
            "route_over_attention",
            contributions[:, 0] / np.maximum(contributions[:, 1], 1e-12),
        ),
        (
            "route_over_residual",
            contributions[:, 0] / np.maximum(contributions[:, 2], 1e-12),
        ),
    ]:
        summary["contributions"][name] = dict(
            mean=float(values.mean()),
            median=float(np.median(values)),
            p95=float(np.quantile(values, 0.95)),
        )
    (root / "diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["contributions"]))


if __name__ == "__main__":
    main()
