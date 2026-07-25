"""Reads replay shards produced by service/src/scripts/offline.ts.

A shard file is a flat sequence of records, each:

    [uint32 little-endian payload length][EnvironmentTrajectory proto bytes]

one record per (replay, perspective). The final state of each trajectory
carries the game outcome in its info buffer (win_reward one-hot), exactly
like a live self-play trajectory, so rl.environment.utils.process_state is
reused unchanged.
"""

import hashlib
import os
import queue
import struct
import threading
from collections.abc import Iterator, Sequence

import chex
import jax
import numpy as np

from rl.environment.interfaces import PlayerActorInput
from rl.environment.protos.service_pb2 import EnvironmentTrajectory
from rl.environment.utils import (
    clip_history,
    clip_packed_history,
    geometric_bucket,
    process_state,
)
from rl.offline.config import Porygon2OfflineConfig

_LENGTH_STRUCT = struct.Struct("<I")


@chex.dataclass(frozen=True)
class OfflineExample:
    actor_input: PlayerActorInput  # env leaves (T, ...), histories unbatched
    label: chex.ArrayLike  # (3,) one-hot [loss, tie, win]


@chex.dataclass(frozen=True)
class OfflineBatch:
    actor_input: PlayerActorInput  # leaves (T, B, ...)
    labels: chex.ArrayLike  # (B, 3)


def list_shards(config: Porygon2OfflineConfig) -> list[str]:
    shard_dir = os.path.join(config.dataset_dir, config.format_id)
    if not os.path.isdir(shard_dir):
        raise FileNotFoundError(
            f"No shard directory at {shard_dir} — run the offline exporter "
            f"(service/src/scripts/offline.ts) first."
        )
    shards = sorted(
        os.path.join(shard_dir, f)
        for f in os.listdir(shard_dir)
        if f.endswith(".bin")
    )
    if not shards:
        raise FileNotFoundError(f"No .bin shards in {shard_dir}")
    return shards


def iter_shard_payloads(shard_path: str) -> Iterator[bytes]:
    with open(shard_path, "rb") as f:
        while True:
            header = f.read(_LENGTH_STRUCT.size)
            if len(header) < _LENGTH_STRUCT.size:
                return
            (length,) = _LENGTH_STRUCT.unpack(header)
            payload = f.read(length)
            if len(payload) < length:
                return  # truncated tail (interrupted exporter) — drop it
            yield payload


def _is_holdout(shard_path: str, record_index: int, holdout_modulus: int) -> bool:
    key = f"{os.path.basename(shard_path)}:{record_index}".encode()
    digest = hashlib.md5(key).digest()
    return int.from_bytes(digest[:4], "little") % holdout_modulus == 0


def trajectory_to_example(payload: bytes) -> OfflineExample | None:
    trajectory = EnvironmentTrajectory.FromString(payload)
    if len(trajectory.states) < 2:
        return None
    steps = [process_state(state) for state in trajectory.states]
    final = steps[-1]
    label = final.env.win_reward
    # A trajectory without a decided outcome (no |win|/|tie| in the log,
    # e.g. a forfeited-by-disconnect fragment) carries no training signal.
    if not final.env.done or label.sum() == 0:
        return None
    env = jax.tree.map(lambda *xs: np.stack(xs), *[s.env for s in steps])
    return OfflineExample(
        actor_input=PlayerActorInput(
            env=env,
            packed_history=final.packed_history,
            history=final.history,
        ),
        label=label,
    )


def _pad_env_to(env, length: int):
    """Pads the time axis by repeating the terminal step with done=1, the
    same convention as rl.environment.utils.get_ex_batch, so the loss mask
    (built from cumsum(done)) zeroes the padding."""
    t = env.done.shape[0]
    if t >= length:
        return jax.tree.map(lambda x: x[:length], env)
    last = jax.tree.map(lambda x: x[-1:], env)
    last = last.replace(done=np.ones_like(last.done))
    pad = jax.tree.map(lambda x: np.repeat(x, length - t, axis=0), last)
    return jax.tree.map(lambda a, b: np.concatenate([a, b], axis=0), env, pad)


def collate(
    examples: Sequence[OfflineExample],
    config: Porygon2OfflineConfig,
) -> OfflineBatch:
    max_t = max(e.actor_input.env.done.shape[0] for e in examples)
    bucket_t = geometric_bucket(
        max_t, lo=config.min_trajectory_bucket, hi=config.max_trajectory_length
    )
    envs = [_pad_env_to(e.actor_input.env, bucket_t) for e in examples]
    env = jax.tree.map(lambda *xs: np.stack(xs, axis=1), *envs)
    packed_history = jax.tree.map(
        lambda *xs: np.stack(xs, axis=1),
        *[e.actor_input.packed_history for e in examples],
    )
    history = jax.tree.map(
        lambda *xs: np.stack(xs, axis=1),
        *[e.actor_input.history for e in examples],
    )
    packed_history = clip_packed_history(
        packed_history, min_length=config.min_history_length
    )
    history = clip_history(history, min_length=config.min_history_length)
    labels = np.stack([e.label for e in examples], axis=0)
    return OfflineBatch(
        actor_input=PlayerActorInput(
            env=env, packed_history=packed_history, history=history
        ),
        labels=labels,
    )


class OfflineDataset:
    """Streaming dataset over replay shards with a shuffle buffer and a
    deterministic hash-based train/eval split."""

    def __init__(self, config: Porygon2OfflineConfig):
        self.config = config
        self.shards = list_shards(config)

    def _iter_examples(self, holdout: bool) -> Iterator[OfflineExample]:
        for shard in self.shards:
            for index, payload in enumerate(iter_shard_payloads(shard)):
                if _is_holdout(shard, index, self.config.holdout_modulus) != holdout:
                    continue
                example = trajectory_to_example(payload)
                if example is not None:
                    yield example

    def train_batches(self, seed: int = 0) -> Iterator[OfflineBatch]:
        """Infinite iterator: reshuffles shard order each epoch and mixes
        examples through a shuffle buffer."""
        rng = np.random.default_rng(seed)
        buffer: list[OfflineExample] = []
        while True:
            rng.shuffle(self.shards)
            for example in self._iter_examples(holdout=False):
                buffer.append(example)
                if len(buffer) < max(
                    self.config.shuffle_buffer_size, self.config.batch_size
                ):
                    continue
                picks = rng.choice(
                    len(buffer), size=self.config.batch_size, replace=False
                )
                batch = [buffer[i] for i in picks]
                for i in sorted(picks, reverse=True):
                    buffer.pop(i)
                yield collate(batch, self.config)
            # Flush what remains at epoch end so small datasets still train.
            while len(buffer) >= self.config.batch_size:
                batch = [buffer.pop() for _ in range(self.config.batch_size)]
                yield collate(batch, self.config)

    def eval_batches(self) -> Iterator[OfflineBatch]:
        """Single pass over the holdout split."""
        batch: list[OfflineExample] = []
        for example in self._iter_examples(holdout=True):
            batch.append(example)
            if len(batch) == self.config.batch_size:
                yield collate(batch, self.config)
                batch = []


def prefetch(
    iterator: Iterator[OfflineBatch], buffer_size: int = 4
) -> Iterator[OfflineBatch]:
    """Runs proto parsing + collation in a background thread."""
    q: "queue.Queue" = queue.Queue(maxsize=buffer_size)
    sentinel = object()

    def _worker():
        try:
            for item in iterator:
                q.put(item)
        finally:
            q.put(sentinel)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    while True:
        item = q.get()
        if item is sentinel:
            return
        yield item