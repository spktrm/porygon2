"""The replay export in memory: one example per trajectory (one perspective
of one replay) -- the terminal state's packed caches and field history (the
whole event stream, up to NUM_HISTORY steps), the outcome and the per-step
event labels. The per-|turn| states are not decoded; every consumer reads
the stream. `load_replay_store` decodes the shards once at startup across a
process pool (the export is the only form on disk), and `ReplayStore`
serves padded batches from the unpadded CSR arrays. Both perspectives of a
game travel together (a mirrored pair is a free side-swap control) and the
holdout split is per game.
"""

import contextlib
import multiprocessing as mp
import os
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import field

import chex
import jax
import numpy as np
from jaxtyping import ArrayLike

from constants import NUM_HISTORY
from rl.environment.event_labels import (
    NO_SLOT,
    EventKind,
    EventLabels,
    event_labels,
    relevant_edges,
)
from rl.environment.interfaces import PlayerHistoryOutput, PlayerPackedHistoryOutput
from rl.environment.protos.features_pb2 import FieldFeature
from rl.environment.protos.service_pb2 import EnvironmentBatch, EnvironmentTrajectory
from rl.environment.utils import (
    clip_history,
    clip_history_windows_tail,
    clip_packed_history,
    process_state,
)
from rl.offline.shards import (
    check_shard_manifest,
    is_holdout,
    iter_shard_payloads,
    list_shards,
    record_offsets,
)

LABEL_FIELDS = tuple(EventLabels.__dataclass_fields__)
LABEL_PAD = {
    "kind": int(EventKind.RESIDUAL),
    "actor": NO_SLOT,
    "target": NO_SLOT,
    "actor_side": -1,
}
# Decode tasks per shard: only the pool's granularity, never a result.
PARTS_PER_SHARD = 4
_CHILD_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}


@chex.dataclass
class ReplayExample:
    packed_history: PlayerPackedHistoryOutput = field(
        default_factory=PlayerPackedHistoryOutput
    )
    history: PlayerHistoryOutput = field(default_factory=PlayerHistoryOutput)
    labels: EventLabels = field(default_factory=EventLabels)
    win_reward: ArrayLike = ()  # (3,) loss / tie / win


@chex.dataclass
class ReplayBatch:
    """Leaves are (B, ...): the trajectory axis first, the history axis
    second, clipped to one geometric bucket across the batch."""

    packed_history: PlayerPackedHistoryOutput = field(
        default_factory=PlayerPackedHistoryOutput
    )
    history: PlayerHistoryOutput = field(default_factory=PlayerHistoryOutput)
    labels: EventLabels = field(default_factory=EventLabels)
    win_reward: ArrayLike = ()  # (B, 3)


def _labelled(
    history: PlayerHistoryOutput,
    packed: PlayerPackedHistoryOutput,
    win_reward: np.ndarray,
    max_history_steps: int,
) -> ReplayExample | None:
    """The tail window BEFORE the labels: the clip rebases the packed row
    indices, and the labels read them."""
    if max_history_steps < history.field.shape[0]:
        history, packed = clip_history_windows_tail(history, packed, max_history_steps)
    labels = event_labels(
        np.asarray(history.field),
        np.asarray(packed.edge_cache),
        np.asarray(packed.public_cache),
        np.asarray(packed.revealed_cache),
    )
    if not labels.valid.any():
        return None
    return ReplayExample(
        packed_history=packed, history=history, labels=labels, win_reward=win_reward
    )


def trajectory_to_example(
    trajectory: EnvironmentTrajectory, max_history_steps: int = NUM_HISTORY
) -> ReplayExample | None:
    if len(trajectory.states) < 2:
        return None
    final = process_state(trajectory.states[-1])
    if not final.env.done or final.env.win_reward.sum() == 0:
        return None
    return _labelled(
        final.history,
        final.packed_history,
        np.asarray(final.env.win_reward, np.float32),
        max_history_steps,
    )


def record_to_examples(
    payload: bytes, max_history_steps: int = NUM_HISTORY
) -> list[ReplayExample]:
    batch = EnvironmentBatch()
    batch.ParseFromString(payload)
    examples = []
    for trajectory in batch.trajectories:
        example = trajectory_to_example(trajectory, max_history_steps)
        if example is not None:
            examples.append(example)
    return examples


def collate(examples: Sequence[ReplayExample], min_history_length: int) -> ReplayBatch:
    def stack(*xs):
        return np.stack(xs, axis=0)

    packed = jax.tree.map(stack, *[e.packed_history for e in examples])
    history = jax.tree.map(stack, *[e.history for e in examples])
    labels = jax.tree.map(stack, *[e.labels for e in examples])
    # The history clip windows the field axis to a bucket over the batch;
    # every (H,) label follows the same window. clip_* read the length off
    # the leaves' second axis, so the leaves are (B, H, ...) here.
    packed = clip_packed_history(
        jax.tree.map(lambda x: np.moveaxis(x, 0, 1), packed),
        min_length=min_history_length,
    )
    history = clip_history(
        jax.tree.map(lambda x: np.moveaxis(x, 0, 1), history),
        min_length=min_history_length,
    )
    field_length = history.field.shape[0]
    packed = jax.tree.map(lambda x: np.moveaxis(x, 0, 1), packed)
    history = jax.tree.map(lambda x: np.moveaxis(x, 0, 1), history)
    labels = jax.tree.map(lambda x: x[:, :field_length], labels)
    return ReplayBatch(
        packed_history=packed,
        history=history,
        labels=labels,
        win_reward=np.stack([e.win_reward for e in examples], axis=0),
    )


def _trim(example: ReplayExample) -> dict:
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


def _decode_range(args) -> dict | None:
    """One pool task: the records [start, end) of a shard, decoded to the
    unpadded event stream in CSR form (rows concatenated, offsets per
    trajectory). Numpy only -- the workers never touch the accelerator."""
    shard, start, end, start_offset, holdout_modulus = args
    pieces: list[dict] = []
    meta = {"holdout": [], "record": [], "side": []}
    payloads = iter_shard_payloads(shard, start_offset)
    for index, payload in enumerate(payloads, start=start):
        if index >= end:
            break
        examples = record_to_examples(payload)
        for side, example in enumerate(examples):
            pieces.append(_trim(example))
            meta["holdout"].append(is_holdout(shard, index, holdout_modulus))
            meta["record"].append(index)
            meta["side"].append(side)
    if not pieces:
        return None
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
    return arrays


def decode_tasks(shards: Sequence[str], holdout_modulus: int) -> list[tuple]:
    tasks = []
    for shard in shards:
        offsets = record_offsets(shard)
        span = len(offsets) // PARTS_PER_SHARD + 1
        for start in range(0, len(offsets), span):
            tasks.append(
                (shard, start, start + span, int(offsets[start]), holdout_modulus)
            )
    return tasks


class ReplayStore:
    """Every decoded trajectory in memory (3 GB for the corpus), served as
    the padded ReplayBatch the trainer consumes. Both perspectives of a
    game share a batch; the global order is the shards' record order."""

    def __init__(self, config, parts: Sequence[dict], manifest: dict):
        self.config = config
        self.manifest = manifest
        self.parts = [part for part in parts if part is not None]
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

    def example(self, global_index: int) -> ReplayExample:
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

        packed = PlayerPackedHistoryOutput(
            public_cache=pad_rows(part["public"]).astype(np.int32),
            revealed_cache=pad_rows(part["revealed"]).astype(np.int32),
            edge_cache=pad_rows(part["edge"]).astype(np.int32),
        )
        history = PlayerHistoryOutput(field=pad_steps(part["field"]).astype(np.int32))
        win_reward = part["win_reward"][local]
        if self.config.max_history_steps < NUM_HISTORY:
            # The consumer's resolution, chosen at load: the labels are
            # re-derived on the window exactly as the decode would have.
            return _labelled(history, packed, win_reward, self.config.max_history_steps)
        # Padded steps carry the sentinels event_labels writes there, so a
        # stored example is the decoded example bit for bit.
        label_arrays = {}
        for name in LABEL_FIELDS:
            values = part[f"label_{name}"]
            fill = LABEL_PAD.get(name, 0)
            if name == "num_revealed" and step_hi > step_lo:
                # A running maximum: the label module carries the final
                # count over the padded steps.
                fill = values[step_hi - 1]
            label_arrays[name] = pad_steps(values, fill)
        return ReplayExample(
            packed_history=packed,
            history=history,
            labels=EventLabels(**label_arrays),
            win_reward=win_reward,
        )

    def train_batches(self, seed: int = 0) -> Iterator[ReplayBatch]:
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

    def eval_batches(self) -> Iterator[ReplayBatch]:
        batch_size = self.config.batch_size
        for start in range(0, len(self.holdout) - batch_size + 1, batch_size):
            yield collate(
                [
                    self.example(index)
                    for index in self.holdout[start : start + batch_size]
                ],
                self.config.min_history_length,
            )


@contextlib.contextmanager
def _no_accelerator_for_children():
    """Spawned workers inherit this environment and import the model
    package, which builds a device array at import: with the GPU visible
    every worker would initialise CUDA beside the parent's allocation and
    fail. The decode is numpy; the workers run on the CPU."""
    saved = {key: os.environ.get(key) for key in _CHILD_ENVIRONMENT}
    os.environ.update(_CHILD_ENVIRONMENT)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                del os.environ[key]
            else:
                os.environ[key] = value


def load_replay_store(config) -> ReplayStore:
    """Decode the export once, across `config.decode_workers` processes.
    Ordered map: the store's global index is the shards' record order.
    Spawned workers re-import the caller's main module, so a script that
    calls this must guard its entry point (`if __name__ == "__main__"`);
    unguarded, every worker re-runs the script and the pool respawns
    them without end."""
    shard_dir = config.shard_dir()
    manifest = check_shard_manifest(shard_dir)
    tasks = decode_tasks(list_shards(shard_dir), config.holdout_modulus)
    started = time.monotonic()
    with _no_accelerator_for_children():
        with mp.get_context("spawn").Pool(config.decode_workers) as pool:
            parts = pool.map(_decode_range, tasks)
    store = ReplayStore(config, parts, manifest)
    print(
        f"{len(store)} trajectories decoded from {shard_dir} in "
        f"{time.monotonic() - started:.0f}s ({len(tasks)} tasks, "
        f"{config.decode_workers} workers)"
    )
    return store
