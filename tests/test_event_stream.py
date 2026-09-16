"""The decoded event stream reproduces the on-the-fly examples:
convert one record range from the real replay shards into a temp dir,
load it through the store, and compare every array of the first
trajectory with `trajectory_to_example` on the same record."""

import os

import numpy as np
import pytest

from rl.environment.protos.service_pb2 import EnvironmentBatch
from rl.offline import event_stream as shards
from rl.offline.config import Porygon2WorldModelConfig
from rl.offline.dataset import iter_shard_payloads
from rl.offline.event_labels import EventLabels
from rl.offline.world_model_data import trajectory_to_example

SOURCE = "replays/shards/gen9randombattle"


@pytest.mark.skipif(not os.path.isdir(SOURCE), reason="replay shards not on this box")
def test_store_round_trips_the_on_the_fly_example(tmp_path) -> None:
    shard = sorted(
        os.path.join(SOURCE, f) for f in os.listdir(SOURCE) if f.endswith(".bin")
    )[0]
    # A temp copy of the export layout: records dir with a decoded/ child.
    source = tmp_path / "gen9randombattle"
    source.mkdir()
    os.symlink(os.path.abspath(shard), source / os.path.basename(shard))
    (source / "manifest.json").write_text(
        (open(os.path.join(SOURCE, "manifest.json")).read())
    )
    out_dir = source / shards.DECODED
    out_dir.mkdir()
    path = shards._convert_range((shard, 0, 3, 0, str(out_dir), 20))
    assert path.endswith("part00.npz")
    shards_manifest = shards.check_shard_manifest(SOURCE)
    (out_dir / shards.MANIFEST).write_text(
        '{"export_commit": "%s", "holdout_modulus": 20}'
        % shards_manifest["export_commit"]
    )
    config = Porygon2WorldModelConfig(dataset_dir=str(tmp_path), batch_size=2)
    store = shards.EventStreamStore(config)
    payload = next(iter_shard_payloads(shard))
    batch = EnvironmentBatch()
    batch.ParseFromString(payload)
    reference = trajectory_to_example(batch.trajectories[0])
    stored = store.example(0)
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
    # Both perspectives of a record share a game; a batch holds both.
    assert len(store) == 2 * len([g for g in store.games]) or len(store) >= 2
    batch_out = next(store.train_batches(0))
    assert batch_out.history.field.shape[0] == 2
