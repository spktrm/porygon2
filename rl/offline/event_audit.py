"""Reads the re-exported replay shards through the event labels and prints
the numbers the world-model plan pre-registers: the KIND histogram, events
per turn, the fraction of games whose early events fell outside the
terminal history window, the fraction of own decisions the log cannot
name, and the snapshot lag -- how often a slot's latest cache row at a
turn boundary disagrees with the |turn| state's public row (the cache
snapshots an entity on its FIRST touch inside an edge).

    env/bin/python rl/offline/event_audit.py --records 2000
"""

import argparse
import collections
import json
import os

import numpy as np

from constants import NUM_HISTORY
from rl.environment import event_labels
from rl.environment.protos.features_pb2 import (
    EntityPublicNodeFeature,
    FieldFeature,
    InfoFeature,
)
from rl.environment.protos.service_pb2 import EnvironmentBatch
from rl.environment.utils import process_state
from rl.model.constants import NUM_PUBLIC_SLOTS
from rl.offline.shards import check_shard_manifest, iter_shard_payloads

_LAG_FEATURES = np.array(
    [
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO,
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__STATUS,
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED,
    ]
)


def audit_trajectory(trajectory, stats: dict) -> None:
    final = process_state(trajectory.states[-1])
    field = np.asarray(final.history.field)
    packed = final.packed_history
    events = event_labels.step_events(
        field,
        np.asarray(packed.edge_cache),
        np.asarray(packed.public_cache),
        np.asarray(packed.revealed_cache),
    )
    out = event_labels.event_labels(
        field,
        np.asarray(packed.edge_cache),
        np.asarray(packed.public_cache),
        np.asarray(packed.revealed_cache),
    )
    valid = events.valid
    stats["games"] += 1
    stats["steps"] += int(valid.sum())
    for kind in event_labels.EventKind:
        stats["kind"][kind.name] += int((out.kind[out.valid] == kind).sum())
    turns = events.turn[valid]
    if turns.size:
        stats["events_per_turn"].extend(np.bincount(turns - turns.min()).tolist())
    first_index = field[valid, FieldFeature.FIELD_FEATURE__INDEX]
    if first_index.size and first_index.min() > 0:
        stats["truncated_games"] += 1
    own = (events.actor_side == event_labels.SIDE_MINE) & valid
    decisions = out.new_turn[out.valid] | out.boundary[out.valid]
    stats["decisions"] += int(decisions.sum())
    stats["own_events"] += int(own.sum())
    stats["declared_unknown"] += int(
        ((out.declared_kind == event_labels.DeclaredKind.UNKNOWN) & out.valid).sum()
    )
    stats["own_cant_no_move"] += int(
        (own & (events.kind == event_labels.EventKind.CANT) & ~events.move_valid).sum()
    )
    if int(valid.sum()) >= NUM_HISTORY:
        stats["full_window_games"] += 1

    # Snapshot lag: each |turn| state vs the slots' latest cache rows as of
    # that request count.
    public_cache = np.asarray(packed.public_cache)
    step_index = np.arange(field.shape[0])
    for state in trajectory.states[:-1]:
        env = process_state(state, with_history=False).env
        request_count = int(env.info[InfoFeature.INFO_FEATURE__REQUEST_COUNT])
        order = env.info[
            InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0 : InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11
            + 1
        ]
        ok = valid & (events.request_count <= request_count)
        if not ok.any():
            continue
        last_step = int(np.where(ok, step_index, -1).max())
        last_row = events.last_row[last_step]
        for row_index, slot in enumerate(order):
            if slot < 0 or slot >= NUM_PUBLIC_SLOTS or last_row[slot] < 0:
                continue
            cached = public_cache[last_row[slot], _LAG_FEATURES]
            truth = env.public_team[row_index, _LAG_FEATURES]
            stats["lag_pairs"] += 1
            stats["lag_mismatch"] += int((cached != truth).any())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", default="replays/shards/gen9randombattle")
    parser.add_argument("--records", type=int, default=2000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    manifest = check_shard_manifest(args.shard_dir)
    stats = collections.Counter()
    stats["kind"] = collections.Counter()
    stats["events_per_turn"] = []
    seen = 0
    shards = sorted(
        os.path.join(args.shard_dir, f)
        for f in os.listdir(args.shard_dir)
        if f.endswith(".bin")
    )
    for shard in shards:
        for payload in iter_shard_payloads(shard):
            batch = EnvironmentBatch()
            batch.ParseFromString(payload)
            for trajectory in batch.trajectories:
                audit_trajectory(trajectory, stats)
            seen += 1
            if seen >= args.records:
                break
        if seen >= args.records:
            break
    per_turn = np.asarray(stats.pop("events_per_turn"), dtype=np.float64)
    kinds = stats.pop("kind")
    summary = {
        "export_commit": manifest.get("export_commit"),
        "records": seen,
        "games": stats["games"],
        "steps": stats["steps"],
        "kind_histogram": dict(kinds),
        "events_per_turn_mean": float(per_turn.mean()),
        "events_per_turn_p90": float(np.percentile(per_turn, 90)),
        "events_per_turn_p99": float(np.percentile(per_turn, 99)),
        "window_truncated_frac": stats["truncated_games"] / max(1, stats["games"]),
        "full_window_games": stats["full_window_games"],
        "declared_unknown_frac": stats["declared_unknown"] / max(1, stats["steps"]),
        "own_cant_no_move_frac": stats["own_cant_no_move"]
        / max(1, stats["own_events"]),
        "decisions_per_game": stats["decisions"] / max(1, stats["games"]),
        "snapshot_lag_frac": stats["lag_mismatch"] / max(1, stats["lag_pairs"]),
        "snapshot_lag_pairs": stats["lag_pairs"],
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
