"""Replay shards -> event world model batches.

One example per trajectory (one perspective of one replay): the terminal
state's packed caches and field history (the whole event stream, up to
NUM_HISTORY steps), the outcome, and the per-step event labels. The
per-|turn| states are not decoded -- the world model reads the stream
only. Both perspectives of a game travel together (a mirrored pair is a
free side-swap control) and the holdout split is per game, as in
rl/offline/dataset.py.
"""

import random
from collections.abc import Iterator, Sequence
from dataclasses import field

import chex
import jax
import numpy as np
from jaxtyping import ArrayLike

from rl.environment.interfaces import PlayerHistoryOutput, PlayerPackedHistoryOutput
from rl.environment.protos.service_pb2 import EnvironmentBatch, EnvironmentTrajectory
from rl.environment.utils import (
    clip_history,
    clip_history_windows_tail,
    clip_packed_history,
    process_state,
)
from rl.offline.dataset import _is_holdout, iter_shard_payloads, list_shards
from rl.offline.event_labels import EventLabels, event_labels


@chex.dataclass
class WorldModelExample:
    packed_history: PlayerPackedHistoryOutput = field(
        default_factory=PlayerPackedHistoryOutput
    )
    history: PlayerHistoryOutput = field(default_factory=PlayerHistoryOutput)
    labels: EventLabels = field(default_factory=EventLabels)
    win_reward: ArrayLike = ()  # (3,) loss / tie / win


@chex.dataclass
class WorldModelBatch:
    """Leaves are (B, ...): the trajectory axis first, the history axis
    second, clipped to one geometric bucket across the batch."""

    packed_history: PlayerPackedHistoryOutput = field(
        default_factory=PlayerPackedHistoryOutput
    )
    history: PlayerHistoryOutput = field(default_factory=PlayerHistoryOutput)
    labels: EventLabels = field(default_factory=EventLabels)
    win_reward: ArrayLike = ()  # (B, 3)


def trajectory_to_example(
    trajectory: EnvironmentTrajectory, max_history_steps: int | None = None
) -> WorldModelExample | None:
    if len(trajectory.states) < 2:
        return None
    final = process_state(trajectory.states[-1])
    if not final.env.done or final.env.win_reward.sum() == 0:
        return None
    history = final.history
    packed = final.packed_history
    if max_history_steps is not None and max_history_steps < history.field.shape[0]:
        # Tail window BEFORE the labels: the clip rebases the packed row
        # indices, and the labels read them.
        history, packed = clip_history_windows_tail(history, packed, max_history_steps)
    labels = event_labels(
        np.asarray(history.field),
        np.asarray(packed.edge_cache),
        np.asarray(packed.public_cache),
        np.asarray(packed.revealed_cache),
    )
    if not labels.valid.any():
        return None
    return WorldModelExample(
        packed_history=packed,
        history=history,
        labels=labels,
        win_reward=np.asarray(final.env.win_reward, np.float32),
    )


def record_to_examples(
    payload: bytes, max_history_steps: int | None = None
) -> list[WorldModelExample]:
    batch = EnvironmentBatch()
    batch.ParseFromString(payload)
    examples = []
    for trajectory in batch.trajectories:
        example = trajectory_to_example(trajectory, max_history_steps)
        if example is not None:
            examples.append(example)
    return examples


def collate(
    examples: Sequence[WorldModelExample], min_history_length: int
) -> WorldModelBatch:
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
    return WorldModelBatch(
        packed_history=packed,
        history=history,
        labels=labels,
        win_reward=np.stack([e.win_reward for e in examples], axis=0),
    )


class WorldModelDataset:
    def __init__(self, config):
        self.config = config
        self.shards = list_shards(config)

    def _iter_records(self, holdout: bool) -> Iterator[list[WorldModelExample]]:
        for shard in self.shards:
            for index, payload in enumerate(iter_shard_payloads(shard)):
                if _is_holdout(shard, index, self.config.holdout_modulus) != holdout:
                    continue
                examples = record_to_examples(payload, self.config.max_history_steps)
                if examples:
                    yield examples

    def train_batches(self, seed: int = 0) -> Iterator[WorldModelBatch]:
        """Infinite, pair-aware: both perspectives of a game share a batch;
        a shuffle buffer of records reorders games within an epoch."""
        rng = random.Random(seed)
        batch_size = self.config.batch_size
        while True:
            buffer: list[list[WorldModelExample]] = []
            pending: list[WorldModelExample] = []
            for record in self._iter_records(holdout=False):
                buffer.append(record)
                if len(buffer) < self.config.shuffle_buffer_size:
                    continue
                pending.extend(buffer.pop(rng.randrange(len(buffer))))
                while len(pending) >= batch_size:
                    yield collate(pending[:batch_size], self.config.min_history_length)
                    pending = pending[batch_size:]
            rng.shuffle(buffer)
            for record in buffer:
                pending.extend(record)
                while len(pending) >= batch_size:
                    yield collate(pending[:batch_size], self.config.min_history_length)
                    pending = pending[batch_size:]

    def eval_batches(self) -> Iterator[WorldModelBatch]:
        pending: list[WorldModelExample] = []
        for record in self._iter_records(holdout=True):
            pending.extend(record)
            while len(pending) >= self.config.batch_size:
                yield collate(
                    pending[: self.config.batch_size], self.config.min_history_length
                )
                pending = pending[self.config.batch_size :]
