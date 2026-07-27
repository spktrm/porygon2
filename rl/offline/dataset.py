"""Reads replay shards produced by service/src/scripts/offline.ts.

A shard file is a flat sequence of records, each:

    [uint32 little-endian payload length][EnvironmentBatch proto bytes]

one record per replay (both perspectives). Labels are 13-bin one-hots over
the final alive-mon margin, derived from the terminal state's caches and
sign-clamped to the recorded result; rl.environment.utils.process_state is
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
from jaxtyping import ArrayLike

from rl.environment.interfaces import PlayerActorInput
from rl.environment.protos.features_pb2 import (
    EntityEdgeFeature,
    EntityPublicNodeFeature,
)
from rl.environment.protos.service_pb2 import (
    EnvironmentBatch,
    EnvironmentTrajectory,
)
from rl.environment.utils import (
    clip_history,
    clip_packed_history,
    geometric_bucket,
    process_state,
)
from rl.offline.config import Porygon2OfflineConfig

_LENGTH_STRUCT = struct.Struct("<I")

# Margin bins: final alive-mon differential in [-6, +6], 13 classes.
MAX_MARGIN = 6
NUM_MARGIN_BINS = 2 * MAX_MARGIN + 1


@chex.dataclass(frozen=True)
class OfflineExample:
    actor_input: PlayerActorInput  # env leaves (T, ...), histories unbatched
    label: ArrayLike  # (NUM_MARGIN_BINS,) one-hot over final margin


@chex.dataclass(frozen=True)
class OfflineBatch:
    actor_input: PlayerActorInput  # leaves (T, B, ...)
    labels: ArrayLike  # (B, NUM_MARGIN_BINS)


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


def _ensemble_bucket(shard_path: str, record_index: int, num_splits: int) -> int:
    """Disjoint per-game split for ensemble members (salted independently
    of the holdout hash). All members share the same holdout set."""
    key = f"ens:{os.path.basename(shard_path)}:{record_index}".encode()
    digest = hashlib.md5(key).digest()
    return int.from_bytes(digest[:4], "little") % num_splits


def _final_margin(final: PlayerActorInput, win_reward: np.ndarray) -> int:
    """Final alive-mon differential from the terminal cache (every fainted
    mon is revealed, so alive = 6 - faints exactly). The sign is clamped to
    the recorded result: a mid-game forfeit can leave the winner behind on
    mons, and the result is the ground truth."""
    public = np.asarray(final.packed_history.public_cache)
    edges = np.asarray(final.packed_history.edge_cache)
    real_rows = np.nonzero(public.any(axis=1))[0]
    slots = edges[:, EntityEdgeFeature.ENTITY_EDGE_FEATURE__ENTITY_IDX]
    last_row_per_slot: dict[int, int] = {}
    for row in real_rows:
        slot = int(slots[row])
        if 0 <= slot < 12:
            last_row_per_slot[slot] = int(row)
    my_faints = opp_faints = 0
    for row in last_row_per_slot.values():
        fainted = int(
            public[row, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED]
        )
        side = int(
            public[row, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE]
        )
        if side == 1:
            my_faints += fainted
        else:
            opp_faints += fainted
    alive_diff = opp_faints - my_faints
    if win_reward[2] == 1:
        return max(min(alive_diff, MAX_MARGIN), 1)
    if win_reward[0] == 1:
        return min(max(alive_diff, -MAX_MARGIN), -1)
    return 0


def record_to_examples(payload: bytes) -> list[OfflineExample]:
    """One shard record = one replay (EnvironmentBatch holding both
    perspectives). Grouping them keeps a game and its mirrored,
    label-flipped twin on the same side of the train/eval split."""
    batch = EnvironmentBatch.FromString(payload)
    examples = [trajectory_to_example(t) for t in batch.trajectories]
    return [e for e in examples if e is not None]


def trajectory_to_example(
    trajectory: EnvironmentTrajectory,
) -> OfflineExample | None:
    if len(trajectory.states) < 2:
        return None
    # Only the final state's (large) history caches are kept per
    # trajectory, so skip materialising them for every other state.
    steps = [
        process_state(state, with_history=False)
        for state in trajectory.states[:-1]
    ]
    steps.append(process_state(trajectory.states[-1]))
    final = steps[-1]
    label = final.env.win_reward
    # A trajectory without a decided outcome (no |win|/|tie| in the log,
    # e.g. a forfeited-by-disconnect fragment) carries no training signal.
    if not final.env.done or label.sum() == 0:
        return None
    margin = _final_margin(final, label)
    margin_onehot = np.zeros(NUM_MARGIN_BINS, dtype=np.float32)
    margin_onehot[margin + MAX_MARGIN] = 1.0
    env = jax.tree.map(lambda *xs: np.stack(xs), *[s.env for s in steps])
    return OfflineExample(
        actor_input=PlayerActorInput(
            env=env,
            packed_history=final.packed_history,
            history=final.history,
        ),
        label=margin_onehot,
    )


def _pad_env_to(env, length: int):
    """Pads the time axis by repeating the terminal step with done=0 — the
    RL actor's padding convention (see rl/actor/player_actor.py): exactly
    one done=1 per trajectory, so cumsum(done)-based masks zero everything
    after the terminal step."""
    t = env.done.shape[0]
    if t >= length:
        return jax.tree.map(lambda x: x[:length], env)
    last = jax.tree.map(lambda x: x[-1:], env)
    last = last.replace(done=np.zeros_like(last.done))
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

    def _iter_records(
        self, holdout: bool, pairs_only: bool = False
    ) -> Iterator[list[OfflineExample]]:
        for shard in self.shards:
            for index, payload in enumerate(iter_shard_payloads(shard)):
                # Records are whole replays, so this split is per game.
                if _is_holdout(shard, index, self.config.holdout_modulus) != holdout:
                    continue
                # Ensemble members train on disjoint games but share the
                # holdout, so member disagreement on eval states measures
                # epistemic uncertainty, not split luck.
                if (
                    not holdout
                    and self.config.ensemble_index >= 0
                    and _ensemble_bucket(
                        shard, index, self.config.num_ensemble_splits
                    )
                    != self.config.ensemble_index
                ):
                    continue
                examples = record_to_examples(payload)
                if pairs_only and len(examples) != 2:
                    continue
                if examples:
                    yield examples

    def _iter_examples(self, holdout: bool) -> Iterator[OfflineExample]:
        for record in self._iter_records(holdout):
            yield from record

    def train_batches(self, seed: int = 0) -> Iterator[OfflineBatch]:
        """Infinite pair-aware iterator: both perspectives of a game always
        share a batch.

        Mirrored perspective pairs have identical non-side features and
        opposite labels, so within a batch they cancel every gradient
        direction EXCEPT the side-differenced signal the critic must
        learn. With unpaired batches, spurious in-batch correlations offer
        easier descent directions that cancel across batches, and training
        random-walks at the constant symmetric predictor (loss pinned at
        ln 2) — observed empirically on a full 50k-step run.
        """
        rng = np.random.default_rng(seed)
        games_per_batch = max(1, self.config.batch_size // 2)
        buffer: list[list[OfflineExample]] = []
        while True:
            rng.shuffle(self.shards)
            for record in self._iter_records(holdout=False, pairs_only=True):
                buffer.append(record)
                if len(buffer) < max(
                    self.config.shuffle_buffer_size // 2, games_per_batch
                ):
                    continue
                picks = rng.choice(len(buffer), size=games_per_batch, replace=False)
                batch = [example for i in picks for example in buffer[i]]
                for i in sorted(picks, reverse=True):
                    buffer.pop(i)
                yield collate(batch[: self.config.batch_size], self.config)
            # Flush what remains at epoch end so small datasets still train.
            while len(buffer) >= games_per_batch:
                records = [buffer.pop() for _ in range(games_per_batch)]
                batch = [example for record in records for example in record]
                yield collate(batch[: self.config.batch_size], self.config)

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
    """Runs proto parsing + collation in a background thread. Exceptions in
    the worker are re-raised in the consumer — a data error must crash the
    run loudly, not silently end it early."""
    q: "queue.Queue" = queue.Queue(maxsize=buffer_size)
    sentinel = object()
    error: list[BaseException] = []

    def _worker():
        try:
            for item in iterator:
                q.put(item)
        except BaseException as e:  # noqa: BLE001 — re-raised in consumer
            error.append(e)
        finally:
            q.put(sentinel)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    while True:
        item = q.get()
        if item is sentinel:
            if error:
                raise error[0]
            return
        yield item