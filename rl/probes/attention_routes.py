"""Read six-layer attention from the frozen switch cohort for visualisation.

Requires COLLECT_INTERMEDIATES=1 before importing the model. All inference
uses the production TrunkBlock and the cached, unmodified trunk inputs.
"""

from __future__ import annotations

import base64
import json
import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rl.model.config import get_player_model_config
from rl.model.constants import (
    HISTORY_ENTITY_ROWS,
    NUM_SEQUENCE_ROWS,
    POLICY_READABLE_ROWS,
    PRIVATE_ROWS,
    PUBLIC_ROWS,
    SEQUENCE_LAYOUT,
    SEQUENCE_READ_MASK,
    SEQUENCE_SLICES,
)
from rl.model.trunk import TrunkBlock

ROOT = Path("runtime/type-probe-switch/attention")
DEPTH_ROOT = Path("runtime/type-probe-switch/depth")


def groups_and_rows():
    groups = []
    names = {
        "CLS": "Policy / value CLS",
        "PUBLIC_ENTITY": "Public entities",
        "PRIVATE_ENTITY": "Own private sheets",
        "MOVE_SLOT": "Own moves",
        "TARGET_SLOT": "Action targets",
        "FIELD": "Field now",
        "HISTORY_FIELD": "Field history",
        "PREV_ACTION": "Previous action",
        "INFO": "Request info",
        "OPP_PRIVATE_ENTITY": "Opponent truth",
        "VALUE_CLS": "Privileged value CLS",
        "HISTORY_ENTITY": "Entity history",
    }
    for group, count in SEQUENCE_LAYOUT:
        rows = SEQUENCE_SLICES[group]
        if group.name == "PUBLIC_ENTITY":
            for label, offsets in (
                ("Own active", range(0, 1)),
                ("Own public bench", range(1, 6)),
                ("Opponent active", range(6, 7)),
                ("Opponent public bench", range(7, 12)),
            ):
                groups.append(
                    {"name": label, "rows": [rows.start + offset for offset in offsets]}
                )
        elif group.name == "HISTORY_ENTITY":
            groups.append(
                {
                    "name": "Own entity history",
                    "rows": list(range(rows.start, rows.start + count // 2)),
                }
            )
            groups.append(
                {
                    "name": "Opponent history",
                    "rows": list(range(rows.start + count // 2, rows.stop)),
                }
            )
        else:
            groups.append(
                {"name": names[group.name], "rows": list(range(rows.start, rows.stop))}
            )
    row_info = [None] * NUM_SEQUENCE_ROWS
    for group_index, group in enumerate(groups):
        group["privileged"] = not any(
            row in POLICY_READABLE_ROWS for row in group["rows"]
        )
        for slot, row in enumerate(group["rows"]):
            label = group["name"]
            if len(group["rows"]) > 1:
                label = f"{label} {slot + 1}"
            row_info[row] = {
                "name": label,
                "group": group_index,
                "slot": slot,
                "privileged": group["privileged"],
            }
    return groups, row_info


def capture(params, sequence, valid):
    block = TrunkBlock(get_player_model_config(9, train=True).encoder.trunk)
    read_mask = jnp.asarray(SEQUENCE_READ_MASK)

    def step(current, block_params):
        ((updated, _), _), captured = jax.vmap(
            lambda rows, present: block.apply(
                {"params": block_params},
                (rows, present),
                read_mask,
                mutable=["intermediates"],
            )
        )(current, valid)
        weights = captured["intermediates"]["attention"]["attn_weights"][0]
        return updated, weights

    final, weights = jax.lax.scan(step, sequence.astype(jnp.bfloat16), params)
    return final, weights


def encoded(values, dtype):
    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("ascii")


def package(weights, counts, valid_keys, groups, label, token_labels, legal):
    # Head-mean is computed from FULL matrices before the sparse token export.
    weights = np.concatenate([weights.mean(axis=1, keepdims=True), weights], axis=1)
    indices = np.argsort(-weights, axis=-1)[..., :8]
    strongest = np.take_along_axis(weights, indices, axis=-1)
    group_weights = np.zeros((*weights.shape[:2], len(groups), len(groups)), np.float64)
    group_counts = []
    for query_group, group in enumerate(groups):
        query_rows = group["rows"]
        denominator = counts[query_rows].sum()
        group_counts.append(float(denominator))
        for key_group, sources in enumerate(groups):
            values = weights[..., query_rows, :][..., sources["rows"]].sum(-1)
            group_weights[..., query_group, key_group] = (
                values * counts[query_rows]
            ).sum(-1) / max(denominator, 1)
    return {
        "label": label,
        "tokens": token_labels,
        "legal": legal,
        "queryCounts": np.asarray(counts).tolist(),
        "validKeyFrequency": np.round(
            np.asarray(valid_keys, dtype=np.float64), 5
        ).tolist(),
        "groupCounts": group_counts,
        "keys": encoded(indices, "u1"),
        "weights": encoded(np.rint(strongest * 65535), "<u2"),
        "groups": encoded(np.rint(group_weights * 65535), "<u2"),
    }


def sample_metadata(indices, states, reference, row_info):
    from rl.environment.protos.features_pb2 import (
        EntityPrivateNodeFeature,
        EntityRevealedNodeFeature,
        MovesetFeature,
    )
    from rl.offline import harness

    enum_data = json.loads(Path("data/data/data.json").read_text())
    species = {value: name for name, value in enum_data["species"].items()}
    moves = {value: name for name, value in enum_data["moves"].items()}
    sides = harness.load("runtime/tactical-cohort/games.pkl")
    result = {}
    for state_index in indices:
        record_index = int(np.flatnonzero(states["record_state"] == state_index)[0])
        game = int(reference["game"][record_index])
        step = int(reference["step"][record_index])
        for chunk in reversed(sides[game]):
            offset = int(np.asarray(chunk.game_step_offset).item())
            if offset <= step:
                break
        local = step - offset
        env = chunk.player_transitions.env_output
        private = np.asarray(env.private_team)[local]
        revealed = np.asarray(env.revealed_team)[local]
        moveset = np.asarray(env.my_moveset)[local]
        labels = [row["name"] for row in row_info]
        for slot in range(private.shape[0]):
            name = species.get(
                int(
                    private[
                        slot,
                        EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SPECIES,
                    ]
                ),
                "unknown",
            )
            labels[PRIVATE_ROWS.start + slot] = f"Sheet {slot + 1}: {name}"
        for slot in range(revealed.shape[0]):
            name = species.get(
                int(
                    revealed[
                        slot,
                        EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES,
                    ]
                ),
                "unknown",
            )
            labels[PUBLIC_ROWS.start + slot] += f": {name}"
            labels[HISTORY_ENTITY_ROWS.start + slot] += f": {name}"
        from rl.model.constants import MOVE_ROWS

        for slot in range(min(moveset.shape[0], MOVE_ROWS.stop - MOVE_ROWS.start)):
            move = moves.get(
                int(moveset[slot, MovesetFeature.MOVESET_FEATURE__MOVE_ID]), "unknown"
            )
            labels[MOVE_ROWS.start + slot] += f": {move}"
        legal = np.flatnonzero(
            np.asarray(env.action_mask)[local, : PRIVATE_ROWS.stop - PRIVATE_ROWS.start]
        ).tolist()
        label = f"Game {game} · decision {step} · {labels[PUBLIC_ROWS.start].split(': ')[-1]} / {labels[PUBLIC_ROWS.start + 6].split(': ')[-1]}"
        result[int(state_index)] = {
            "label": label,
            "labels": labels,
            "legal": legal,
            "game": game,
            "step": step,
        }
    return result


def main():
    if os.environ.get("COLLECT_INTERMEDIATES") != "1":
        raise RuntimeError("Run with COLLECT_INTERMEDIATES=1 before model import")
    if jax.default_backend() != "gpu":
        raise RuntimeError("Attention extraction requires the free GPU")
    ROOT.mkdir(parents=True, exist_ok=True)
    states = dict(np.load(DEPTH_ROOT / "states.npz"))
    reference = np.load("runtime/type-probe-switch/frozen_02339569.npz")
    with (DEPTH_ROOT / "original-trunk.pkl").open("rb") as handle:
        params = pickle.load(handle)
    groups, row_info = groups_and_rows()
    rng = np.random.default_rng(910)
    selected_games = rng.choice(np.unique(states["game"]), 6, replace=False)
    selected_states = [
        int(rng.choice(np.flatnonzero(states["game"] == game)))
        for game in selected_games
    ]
    metadata = sample_metadata(selected_states, states, reference, row_info)
    reader = jax.jit(capture)
    counts = states["valid"].sum(0)
    totals = np.zeros((6, 4, NUM_SEQUENCE_ROWS, NUM_SEQUENCE_ROWS), np.float64)
    samples = {}
    maximum_sum_error = 0.0
    maximum_leak = 0.0
    endpoint_error = 0.0
    legal_masks = np.zeros((len(states["sequence"]), NUM_SEQUENCE_ROWS), bool)
    legal_masks[states["record_state"], PRIVATE_ROWS.start + reference["slot"]] = True
    legal_group_mass = np.zeros((6, 4, len(groups)), np.float64)
    entropy_sum = np.zeros((6, 4), np.float64)
    forbidden = ~np.asarray(SEQUENCE_READ_MASK)
    cached_final = reference["candidate_post"]
    for start in range(0, len(states["sequence"]), 32):
        end = min(start + 32, len(states["sequence"]))
        selected = np.arange(start, end)
        count = len(selected)
        selected = np.pad(selected, (0, 32 - count), mode="edge")
        final, weights = reader(
            params, states["sequence"][selected], states["valid"][selected]
        )
        final = np.asarray(final, np.float32)[:count]
        weights = np.asarray(weights, np.float64)[:, :count]
        sums = weights.sum(-1)
        valid = states["valid"][start:end]
        maximum_sum_error = max(
            maximum_sum_error, float(np.max(np.abs(sums - valid[None, :, None])))
        )
        maximum_leak = max(maximum_leak, float(np.max(np.abs(weights[..., forbidden]))))
        weights = weights / np.maximum(sums[..., None], 1e-12)
        totals += weights.sum(1)
        records = np.flatnonzero(
            (states["record_state"] >= start) & (states["record_state"] < end)
        )
        matched = final[
            states["record_state"][records] - start,
            PRIVATE_ROWS.start + reference["slot"][records],
        ]
        endpoint_error = max(
            endpoint_error, float(np.max(np.abs(matched - cached_final[records])))
        )
        query_mask = legal_masks[start:end]
        entropy = -(weights * np.log(np.maximum(weights, 1e-30))).sum(-1)
        entropy_sum += (entropy * query_mask[None, :, None]).sum((1, 3))
        for group_index, group in enumerate(groups):
            mass = weights[..., group["rows"]].sum(-1)
            legal_group_mass[..., group_index] += (
                mass * query_mask[None, :, None]
            ).sum((1, 3))
        for state_index in selected_states:
            if start <= state_index < end:
                samples[state_index] = weights[:, state_index - start]
        if start % 320 == 0:
            print(f"attention {end}/{len(states['sequence'])} states", flush=True)
    if maximum_leak != 0 or maximum_sum_error > 0.03 or endpoint_error > 0.1:
        raise AssertionError((maximum_leak, maximum_sum_error, endpoint_error))
    mean = totals / np.maximum(counts, 1)[None, None, :, None]
    datasets = [
        package(
            mean,
            counts,
            states["valid"].mean(0),
            groups,
            f"Cohort mean · {len(states['sequence']):,} states",
            [row["name"] for row in row_info],
            [],
        )
    ]
    for state_index in selected_states:
        detail = metadata[state_index]
        datasets.append(
            package(
                samples[state_index],
                states["valid"][state_index].astype(int),
                states["valid"][state_index],
                groups,
                detail["label"],
                detail["labels"],
                detail["legal"],
            )
        )
    compact = {
        "checkpoint": "02339569 · EMA",
        "states": len(states["sequence"]),
        "cells": len(reference["slot"]),
        "layers": 6,
        "heads": 4,
        "rows": row_info,
        "groups": groups,
        "datasets": datasets,
        "topK": 8,
        "sampleSeed": 910,
    }
    summary = {
        "checkpoint": "ckpt_02339569",
        "parameters": "target_params",
        "states": len(states["sequence"]),
        "legal_switch_cells": int(legal_masks.sum()),
        "groups": groups,
        "legal_switch_group_mass": (legal_group_mass / legal_masks.sum()).tolist(),
        "legal_switch_entropy_nats": (entropy_sum / legal_masks.sum()).tolist(),
        "max_raw_row_sum_error": maximum_sum_error,
        "max_forbidden_attention": maximum_leak,
        "max_final_candidate_delta": endpoint_error,
        "samples": metadata,
    }
    (ROOT / "visual-data.json").write_text(json.dumps(compact, separators=(",", ":")))
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        ROOT / "full-attention.npz",
        mean=mean,
        counts=counts,
        **{
            f"state_{state_index}": samples[state_index]
            for state_index in selected_states
        },
    )
    print(
        json.dumps(
            {
                name: summary[name]
                for name in (
                    "states",
                    "legal_switch_cells",
                    "max_raw_row_sum_error",
                    "max_forbidden_attention",
                    "max_final_candidate_delta",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
