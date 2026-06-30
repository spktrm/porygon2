"""Sharded checkpoint I/O.

A checkpoint is a *directory* (``ckpt_{step:08}``) in which every logical
component is stored as its own file:

    ckpt_00012345/
        meta                      # {format_version, learner_config}
        player/params
        player/target_params      # ema params — loadable without opt_state
        player/opt_state
        player/alpha_params
        player/alpha_opt_state
        player/scalars            # step_count, frame_count, ema_adv_*
        builder/params
        builder/target_params
        builder/opt_state
        builder/scalars
        league                    # league.serialize() bytes (refs + stats)

Storing components separately means an opponent can be materialised by reading
only ``player/target_params`` (+ ``builder/target_params``) — the large
optimiser state is never touched. This is the foundation the disk-backed
league builds on.
"""

from __future__ import annotations

import os
import re
from typing import Any

import cloudpickle as pickle
import jax

FORMAT_VERSION = 1

_CKPT_DIR_RE = re.compile(r"ckpt_(\d+)$")


def _dump(path: str, obj: Any) -> None:
    # Pull arrays back to host so checkpoints don't pin device memory and stay
    # portable across device topologies.
    obj = jax.device_get(obj)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    # Atomic rename so a reader never observes a half-written component.
    os.replace(tmp_path, path)


def _load(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _ckpt_dir(root: str, step: int) -> str:
    return os.path.join(root, f"ckpt_{step:08}")


def save_train_state(
    ckpt_dir: str,
    learner_config: Any,
    player_state_components: dict[str, Any],
    builder_state_components: dict[str, Any],
    league_bytes: bytes,
) -> str:
    """Write a full train state as a folder of per-component files.

    ``*_state_components`` are plain dicts so this module stays decoupled from
    the concrete TrainState classes. Keys map directly to file names; the
    ``scalars`` key (if present) is written as a single file.
    """
    player_dir = os.path.join(ckpt_dir, "player")
    builder_dir = os.path.join(ckpt_dir, "builder")
    os.makedirs(player_dir, exist_ok=True)
    os.makedirs(builder_dir, exist_ok=True)

    _dump(
        os.path.join(ckpt_dir, "meta"),
        dict(format_version=FORMAT_VERSION, learner_config=learner_config),
    )
    for name, value in player_state_components.items():
        _dump(os.path.join(player_dir, name), value)
    for name, value in builder_state_components.items():
        _dump(os.path.join(builder_dir, name), value)

    # League bytes are already serialised (refs + stats only); store verbatim.
    _dump(os.path.join(ckpt_dir, "league"), league_bytes)
    return ckpt_dir


def save_param_snapshot(
    snapshot_dir: str,
    player_components: dict[str, Any],
    builder_components: dict[str, Any],
) -> str:
    """Write a params-only snapshot (no optimiser state) for a league opponent.

    Uses the same ``player/`` + ``builder/`` layout as a full checkpoint, so
    ``load_component(snapshot_dir, "player", "params")`` works identically.
    """
    player_dir = os.path.join(snapshot_dir, "player")
    builder_dir = os.path.join(snapshot_dir, "builder")
    os.makedirs(player_dir, exist_ok=True)
    os.makedirs(builder_dir, exist_ok=True)
    for name, value in player_components.items():
        _dump(os.path.join(player_dir, name), value)
    for name, value in builder_components.items():
        _dump(os.path.join(builder_dir, name), value)
    return snapshot_dir


def load_component(ckpt_dir: str, who: str, name: str) -> Any:
    """Load a single component, e.g. ``load_component(d, "player", "target_params")``.

    Reads exactly one file — the optimiser state is never deserialised unless
    explicitly requested.
    """
    return _load(os.path.join(ckpt_dir, who, name))


def has_component(ckpt_dir: str, who: str, name: str) -> bool:
    return os.path.exists(os.path.join(ckpt_dir, who, name))


def load_league_bytes(ckpt_dir: str) -> bytes | None:
    path = os.path.join(ckpt_dir, "league")
    if not os.path.exists(path):
        return None
    return _load(path)


def load_full(ckpt_dir: str) -> dict[str, Any]:
    """Rebuild the legacy ``ckpt_data`` shape for full-restore code paths."""

    def _read_dir(who: str) -> dict[str, Any]:
        d = os.path.join(ckpt_dir, who)
        out: dict[str, Any] = {}
        for name in os.listdir(d):
            out[name] = _load(os.path.join(d, name))
        return out

    return dict(
        meta=_load(os.path.join(ckpt_dir, "meta")),
        player_state=_read_dir("player"),
        builder_state=_read_dir("builder"),
        league=load_league_bytes(ckpt_dir),
    )


def list_ckpt_dirs(root: str) -> list[tuple[int, str]]:
    """Return ``(step, path)`` for every checkpoint folder under ``root``, sorted."""
    if not os.path.exists(root):
        return []
    found: list[tuple[int, str]] = []
    for name in os.listdir(root):
        match = _CKPT_DIR_RE.match(name)
        path = os.path.join(root, name)
        if match and os.path.isdir(path):
            found.append((int(match.group(1)), path))
    found.sort(key=lambda x: x[0])
    return found


def most_recent_ckpt_dir(root: str) -> str | None:
    dirs = list_ckpt_dirs(root)
    return dirs[-1][1] if dirs else None
