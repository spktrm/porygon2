"""The in-memory store reproduces the decoded example: decode one record
range of the real replay shards through the pool task, load it into a
store, and compare every array of both perspectives with
`trajectory_to_example` on the same record."""

import os

import numpy as np
import pytest

from rl.environment.event_labels import EventLabels
from rl.environment.protos.service_pb2 import EnvironmentBatch
from rl.offline import dataset
from rl.offline.config import Porygon2OfflineConfig
from rl.offline.shards import check_shard_manifest, iter_shard_payloads, list_shards

SOURCE = "replays/shards/gen9randombattle"
# Trajectories the corpus decodes to, by export commit (the pool test).
CORPUS_TRAJECTORIES = {"cc999c1979e635e5cde515e24da2f4f4b5556a90": 98_512}


def assert_same_example(stored, reference) -> None:
    for name in EventLabels.__dataclass_fields__:
        np.testing.assert_array_equal(
            np.asarray(getattr(stored.labels, name)),
            np.asarray(getattr(reference.labels, name)),
            err_msg=name,
        )
    np.testing.assert_array_equal(stored.history.field, reference.history.field)
    for name in ("public_cache", "revealed_cache", "edge_cache"):
        np.testing.assert_array_equal(
            getattr(stored.packed_history, name),
            getattr(reference.packed_history, name),
        )
    np.testing.assert_array_equal(stored.win_reward, reference.win_reward)


@pytest.mark.skipif(not os.path.isdir(SOURCE), reason="replay shards not on this box")
def test_store_reproduces_the_decoded_example() -> None:
    shard = list_shards(SOURCE)[0]
    config = Porygon2OfflineConfig(batch_size=2)
    part = dataset._decode_range((shard, 0, 3, 0, config.holdout_modulus))
    store = dataset.ReplayStore(config, [part], check_shard_manifest(SOURCE))
    batch = EnvironmentBatch()
    batch.ParseFromString(next(iter_shard_payloads(shard)))
    for side in (0, 1):
        assert_same_example(
            store.example(side), dataset.trajectory_to_example(batch.trajectories[side])
        )
    # Both perspectives of a record share a game; a batch holds both.
    first = next(store.train_batches(0))
    assert first.history.field.shape[0] == 2
    members = store.train_games[0]
    assert {int(part["record"][store.index[m][1]]) for m in members} == {
        int(part["record"][store.index[members[0]][1]])
    }
    # A load-time window re-derives the labels on the tail exactly as the
    # decode would.
    windowed = dataset.ReplayStore(
        Porygon2OfflineConfig(batch_size=2, max_history_steps=64), [part], {}
    )
    assert_same_example(
        windowed.example(0),
        dataset.trajectory_to_example(batch.trajectories[0], max_history_steps=64),
    )


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(SOURCE), reason="replay shards not on this box")
def test_pool_decodes_the_whole_corpus() -> None:
    manifest = check_shard_manifest(SOURCE)
    expected = CORPUS_TRAJECTORIES.get(manifest["export_commit"])
    if expected is None:
        pytest.skip(f"corpus count not recorded for {manifest['export_commit']}")
    store = dataset.load_replay_store(Porygon2OfflineConfig(decode_workers=4))
    assert len(store) == expected
    assert len(store.holdout) + sum(len(g) for g in store.train_games) == expected
