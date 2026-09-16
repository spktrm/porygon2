"""The offline export's decoded event stream, on disk beside the records.

`replays/shards/<format>/` holds the service's export: one protobuf
record per replay. `convert` decodes each trajectory's terminal state
once (the whole public event stream) and derives the per-step event
labels, writing the unpadded stream (CSR: concatenated field / public /
revealed / edge rows plus offsets, labels per step, outcome, holdout flag
and record id) to `replays/shards/<format>/decoded/`; `EventStreamStore`
serves it from memory, so a batch is a slice, a pad and a stack instead
of a protobuf parse and a label derivation on the training thread.

    env/bin/python rl/offline/event_stream.py --workers 16
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from collections.abc import Iterator

import numpy as np

from constants import NUM_HISTORY
from rl.environment.event_labels import NO_SLOT, EventKind, EventLabels, relevant_edges
from rl.environment.interfaces import PlayerHistoryOutput, PlayerPackedHistoryOutput
from rl.environment.protos.features_pb2 import FieldFeature
from rl.offline.shards import check_shard_manifest, is_holdout, iter_shard_payloads
from rl.offline.world_model_data import (
    WorldModelBatch,
    WorldModelExample,
    collate,
    record_to_examples,
)

LABEL_FIELDS = tuple(EventLabels.__dataclass_fields__)
DECODED = "decoded"


def decoded_dir(source_dir: str) -> str:
    return os.path.join(source_dir, DECODED)


LABEL_PAD = {
    "kind": int(EventKind.RESIDUAL),
    "actor": NO_SLOT,
    "target": NO_SLOT,
    "actor_side": -1,
}
MANIFEST = "manifest.json"


def _trim(example: WorldModelExample) -> dict:
    field = np.asarray(example.history.field)
    steps = int((field[:, FieldFeature.FIELD_FEATURE__VALID] > 0).sum())
    # Rows the steps reference, not the species-sentinel count
    # (CLAUDE.md known-open: that heuristic can undercount live rows).
    relevant, edge_mask = relevant_edges(field[:steps])
    rows = int(np.where(edge_mask, relevant, -1).max()) + 1
    packed = example.packed_history
    out = {
        "field": field[:steps].astype(np.int16),
        "public": np.asarray(packed.public_cache)[:rows].astype(np.int16),
        "revealed": np.asarray(packed.revealed_cache)[:rows].astype(np.int16),
        "edge": np.asarray(packed.edge_cache)[:rows].astype(np.int16),
        "win_reward": np.asarray(example.win_reward, np.float32),
    }
    for name in LABEL_FIELDS:
        value = np.asarray(getattr(example.labels, name))
        if value.dtype == np.bool_:
            out[f"label_{name}"] = value[:steps]
        else:
            out[f"label_{name}"] = value[:steps].astype(np.int32)
    return out


def _convert_range(args) -> str:
    shard, start, end, part, out_dir, holdout_modulus = args
    pieces: list[dict] = []
    meta = {"holdout": [], "record": [], "side": []}
    for index, payload in enumerate(iter_shard_payloads(shard)):
        if index < start:
            continue
        if index >= end:
            break
        examples = record_to_examples(payload)
        for side, example in enumerate(examples):
            pieces.append(_trim(example))
            meta["holdout"].append(is_holdout(shard, index, holdout_modulus))
            meta["record"].append(index)
            meta["side"].append(side)
    stem = os.path.splitext(os.path.basename(shard))[0]
    path = os.path.join(out_dir, f"{stem}-part{part:02d}.npz")
    if not pieces:
        np.savez(path, empty=np.zeros(0))
        return path
    arrays = {}
    for key in pieces[0]:
        arrays[key] = np.concatenate([piece[key] for piece in pieces], axis=0)
    arrays["step_offsets"] = np.concatenate(
        [[0], np.cumsum([piece["field"].shape[0] for piece in pieces])]
    ).astype(np.int64)
    arrays["row_offsets"] = np.concatenate(
        [[0], np.cumsum([piece["public"].shape[0] for piece in pieces])]
    ).astype(np.int64)
    arrays["win_reward"] = np.stack([piece["win_reward"] for piece in pieces])
    for name, values in meta.items():
        arrays[name] = np.asarray(values)
    np.savez(path, **arrays)
    return path


def convert(
    source_dir: str, out_dir: str, workers: int, holdout_modulus: int, parts: int
) -> None:
    manifest = check_shard_manifest(source_dir)
    shards = sorted(
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if f.endswith(".bin")
    )
    os.makedirs(out_dir, exist_ok=True)
    per_shard = manifest["num_replays"] // len(shards) + 1
    tasks = []
    for shard in shards:
        span = per_shard // parts + 1
        for part in range(parts):
            tasks.append(
                (shard, part * span, (part + 1) * span, part, out_dir, holdout_modulus)
            )
    start = time.monotonic()
    with mp.get_context("spawn").Pool(workers) as pool:
        for done, path in enumerate(pool.imap_unordered(_convert_range, tasks), 1):
            print(
                f"{done}/{len(tasks)} {path} ({time.monotonic() - start:.0f}s)",
                flush=True,
            )
    with open(os.path.join(out_dir, MANIFEST), "w") as f:
        json.dump(
            dict(
                source=source_dir,
                export_commit=manifest["export_commit"],
                num_history=manifest["num_history"],
                feature_counts=manifest["feature_counts"],
                holdout_modulus=holdout_modulus,
                label_fields=list(LABEL_FIELDS),
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            ),
            f,
            indent=2,
        )


class EventStreamStore:
    """Every decoded trajectory in memory (3 GB for the corpus), served as
    the padded WorldModelBatch the trainer consumes. Both perspectives of
    a game share a batch; the holdout split is the converter's."""

    def __init__(self, config):
        self.config = config
        source_dir = os.path.join(config.dataset_dir, config.format_id)
        shard_dir = decoded_dir(source_dir)
        if not os.path.isdir(shard_dir):
            raise FileNotFoundError(
                f"No decoded event stream at {shard_dir} -- run "
                f"rl/offline/event_stream.py once over the export."
            )
        with open(os.path.join(shard_dir, MANIFEST)) as f:
            self.manifest = json.load(f)
        source_manifest = check_shard_manifest(source_dir)
        if source_manifest["export_commit"] != self.manifest["export_commit"]:
            raise ValueError(
                f"{shard_dir} was decoded from export {self.manifest['export_commit']}, "
                f"the records are {source_manifest['export_commit']} -- reconvert."
            )
        self.parts = []
        for name in sorted(os.listdir(shard_dir)):
            if not name.endswith(".npz"):
                continue
            part = np.load(os.path.join(shard_dir, name))
            if "empty" in part:
                continue
            self.parts.append({key: part[key] for key in part.files})
        self.index = [
            (part_index, local)
            for part_index, part in enumerate(self.parts)
            for local in range(part["holdout"].shape[0])
        ]
        # Games: consecutive trajectories of one record.
        self.games: dict[tuple[int, int], list[int]] = {}
        for global_index, (part_index, local) in enumerate(self.index):
            record = int(self.parts[part_index]["record"][local])
            self.games.setdefault((part_index, record), []).append(global_index)
        self.train_games = [
            members
            for (part_index, _), members in self.games.items()
            if not self.parts[part_index]["holdout"][self.index[members[0]][1]]
        ]
        self.holdout = [
            global_index
            for global_index, (part_index, local) in enumerate(self.index)
            if self.parts[part_index]["holdout"][local]
        ]

    def __len__(self) -> int:
        return len(self.index)

    def example(self, global_index: int) -> WorldModelExample:
        part_index, local = self.index[global_index]
        part = self.parts[part_index]
        step_lo, step_hi = part["step_offsets"][local : local + 2]
        row_lo, row_hi = part["row_offsets"][local : local + 2]
        max_rows = 2 * NUM_HISTORY

        def pad_steps(values, fill=0):
            padded = np.full((NUM_HISTORY,) + values.shape[1:], fill, values.dtype)
            padded[: step_hi - step_lo] = values[step_lo:step_hi]
            return padded

        def pad_rows(values):
            padded = np.zeros((max_rows,) + values.shape[1:], values.dtype)
            padded[: row_hi - row_lo] = values[row_lo:row_hi]
            return padded

        # Padded steps carry the sentinels event_labels writes there, so a
        # stored example is the on-the-fly example bit for bit.
        label_arrays = {}
        for name in LABEL_FIELDS:
            values = part[f"label_{name}"]
            fill = LABEL_PAD.get(name, 0)
            if name == "num_revealed" and step_hi > step_lo:
                # A running maximum: the label module carries the final
                # count over the padded steps.
                fill = values[step_hi - 1]
            label_arrays[name] = pad_steps(values, fill)
        labels = EventLabels(**label_arrays)
        return WorldModelExample(
            packed_history=PlayerPackedHistoryOutput(
                public_cache=pad_rows(part["public"]).astype(np.int32),
                revealed_cache=pad_rows(part["revealed"]).astype(np.int32),
                edge_cache=pad_rows(part["edge"]).astype(np.int32),
            ),
            history=PlayerHistoryOutput(
                field=pad_steps(part["field"]).astype(np.int32)
            ),
            labels=labels,
            win_reward=part["win_reward"][local],
        )

    def train_batches(self, seed: int = 0) -> Iterator[WorldModelBatch]:
        rng = random.Random(seed)
        games_per_batch = max(1, self.config.batch_size // 2)
        while True:
            order = list(range(len(self.train_games)))
            rng.shuffle(order)
            for start in range(0, len(order) - games_per_batch + 1, games_per_batch):
                members = [
                    index
                    for game in order[start : start + games_per_batch]
                    for index in self.train_games[game]
                ][: self.config.batch_size]
                yield collate(
                    [self.example(index) for index in members],
                    self.config.min_history_length,
                )

    def eval_batches(self) -> Iterator[WorldModelBatch]:
        batch_size = self.config.batch_size
        for start in range(0, len(self.holdout) - batch_size + 1, batch_size):
            yield collate(
                [
                    self.example(index)
                    for index in self.holdout[start : start + batch_size]
                ],
                self.config.min_history_length,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="replays/shards/gen9randombattle")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--parts", type=int, default=4, help="parts per source shard")
    parser.add_argument("--holdout-modulus", type=int, default=20)
    args = parser.parse_args()
    convert(
        args.source,
        decoded_dir(args.source),
        args.workers,
        args.holdout_modulus,
        args.parts,
    )


if __name__ == "__main__":
    main()
