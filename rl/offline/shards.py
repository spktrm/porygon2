"""The replay export on disk: replays/shards/<format>/, written by
service/src/scripts/offline.ts. A shard file is a flat sequence of records,
each

    [uint32 little-endian payload length][EnvironmentBatch proto bytes]

one record per replay carrying both perspectives. Every consumer reads the
shards through this module; nothing derived is stored beside them.
"""

import hashlib
import json
import os
import struct
from collections.abc import Iterator

import numpy as np

from rl.environment.data import (
    NUM_ENTITY_EDGE_FEATURES,
    NUM_ENTITY_PUBLIC_FEATURES,
    NUM_ENTITY_REVEALED_FEATURES,
    NUM_FIELD_FEATURES,
)
from rl.environment.protos.features_pb2 import InfoFeature

_LENGTH_STRUCT = struct.Struct("<I")


def check_shard_manifest(shard_dir: str) -> dict:
    """A shard's feature layout must be the one this code was built
    against: the July 2026 export decoded without error after the
    action-mask and feature-count changes and read garbage. The exporter
    writes the counts; an export that predates them is refused outright."""
    manifest_path = os.path.join(shard_dir, "manifest.json")
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"No manifest at {manifest_path}") from error
    expected = {
        "public": NUM_ENTITY_PUBLIC_FEATURES,
        "revealed": NUM_ENTITY_REVEALED_FEATURES,
        "edge": NUM_ENTITY_EDGE_FEATURES,
        "field": NUM_FIELD_FEATURES,
        "info": len(InfoFeature.keys()),
    }
    counts = manifest.get("feature_counts")
    if "export_commit" not in manifest or counts != expected:
        raise ValueError(
            f"Stale shards at {shard_dir}: manifest feature counts {counts} vs "
            f"{expected} (export_commit {manifest.get('export_commit')}) -- "
            f"re-export with service/src/scripts/offline.ts."
        )
    return manifest


def list_shards(shard_dir: str) -> list[str]:
    if not os.path.isdir(shard_dir):
        raise FileNotFoundError(
            f"No shard directory at {shard_dir} — run the offline exporter "
            f"(service/src/scripts/offline.ts) first."
        )
    check_shard_manifest(shard_dir)
    shards = sorted(
        os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith(".bin")
    )
    if not shards:
        raise FileNotFoundError(f"No .bin shards in {shard_dir}")
    return shards


def iter_shard_payloads(shard_path: str, start_offset: int = 0) -> Iterator[bytes]:
    with open(shard_path, "rb") as f:
        f.seek(start_offset)
        while True:
            header = f.read(_LENGTH_STRUCT.size)
            if len(header) < _LENGTH_STRUCT.size:
                return
            (length,) = _LENGTH_STRUCT.unpack(header)
            payload = f.read(length)
            if len(payload) < length:
                return  # truncated tail (interrupted exporter) — drop it
            yield payload


def record_offsets(shard_path: str) -> np.ndarray:
    """Byte offset of every complete record, header-only pass."""
    offsets = []
    size = os.path.getsize(shard_path)
    with open(shard_path, "rb") as f:
        while True:
            offset = f.tell()
            header = f.read(_LENGTH_STRUCT.size)
            if len(header) < _LENGTH_STRUCT.size:
                break
            (length,) = _LENGTH_STRUCT.unpack(header)
            if offset + _LENGTH_STRUCT.size + length > size:
                break
            offsets.append(offset)
            f.seek(length, os.SEEK_CUR)
    return np.asarray(offsets, dtype=np.int64)


def is_holdout(shard_path: str, record_index: int, holdout_modulus: int) -> bool:
    key = f"{os.path.basename(shard_path)}:{record_index}".encode()
    digest = hashlib.md5(key).digest()
    return int.from_bytes(digest[:4], "little") % holdout_modulus == 0
