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
        controllers               # host-side controller/plasticity state
        scheduler                 # block-sequential scheduler state (which
                                  # population owned the GPU, rotation index,
                                  # and which populations/ entries are live)
        populations/{name}/       # a live exploiter population's own full
            player/... builder/   # resumable state, same layout as main's
            host                  # host-side counters (fork_step, budget
                                  # anchor) that live outside the TrainState
            controllers

Storing components separately means an opponent can be materialised by reading
only ``player/target_params`` (+ ``builder/target_params``) — the large
optimiser state is never touched. This is the foundation the disk-backed
league builds on.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any

import cloudpickle as pickle
import jax

FORMAT_VERSION = 1

_CKPT_DIR_RE = re.compile(r"ckpt_(\d+)$")


def _dump(path: str, obj: Any) -> None:
    # Pull arrays back to host so checkpoints don't pin device memory and stay
    # portable across device topologies.
    obj = jax.device_get(obj)
    # Writer-unique tmp name: the periodic checkpoint worker and the OOM
    # guard's emergency save can race on the SAME step directory (observed
    # 2026-08-14 at ckpt_00020000) — with a shared "<path>.tmp" the loser's
    # os.replace finds its tmp already consumed and crashes the save it was
    # supposed to guarantee. Both writers dump identical state for that
    # step, so last-rename-wins per component is fully consistent.
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    # Atomic rename so a reader never observes a half-written component.
    os.replace(tmp_path, path)


def _load(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _component_names(d: str) -> list[str]:
    """Component filenames in a checkpoint dir, EXCLUDING writer scratch:
    a process killed mid-_dump leaves a "<name>.tmp.<pid>.<tid>" file
    behind (the atomic-rename never ran), and a directory listing that
    picks it up feeds a truncated pickle to the loader — the 2026-08-15
    resume failure. The completed component beside it is the real one."""
    return [n for n in os.listdir(d) if ".tmp." not in n]


def _ckpt_dir(root: str, step: int) -> str:
    return os.path.join(root, f"ckpt_{step:08}")


def save_train_state(
    ckpt_dir: str,
    learner_config: Any,
    player_state_components: dict[str, Any],
    builder_state_components: dict[str, Any],
    league_bytes: bytes,
    controller_bytes: bytes | None = None,
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
    # Host-side controller + plasticity state: not parameters, but training
    # dynamics that a resume must not silently reset (a forgotten
    # plasticity recovery clears the perturbation cooldown; a forgotten
    # lambda controller re-anneals from scratch).
    if controller_bytes is not None:
        _dump(os.path.join(ckpt_dir, "controllers"), controller_bytes)
    return ckpt_dir


def save_scheduler_state(ckpt_dir: str, scheduler: dict) -> None:
    """Block-sequential scheduler state: {"active": ..., "rotation_idx": ...,
    "populations": [names]}. The "populations" list is the source of truth
    for which populations/{name}/ subdirs are live in THIS write — a reader
    must ignore any subdir not listed, since repeated writes into the same
    ckpt dir (main's step is frozen during an exploiter block) can leave a
    stale subdir behind after that population's block ended."""
    _dump(os.path.join(ckpt_dir, "scheduler"), scheduler)


def load_scheduler_state(ckpt_dir: str) -> dict | None:
    """None for checkpoints written before exploiter-phase resume existed
    (those resume with a main window, exactly the old behaviour)."""
    path = os.path.join(ckpt_dir, "scheduler")
    if not os.path.exists(path):
        return None
    return _load(path)


def save_population_state(
    ckpt_dir: str,
    name: str,
    player_state_components: dict[str, Any],
    builder_state_components: dict[str, Any],
    host: dict[str, Any],
    controller_bytes: bytes | None = None,
) -> None:
    """Write one exploiter population's full resumable state under
    populations/{name}/, mirroring the main checkpoint's player/ + builder/
    layout so the same component readers work on both."""
    base = os.path.join(ckpt_dir, "populations", name)
    player_dir = os.path.join(base, "player")
    builder_dir = os.path.join(base, "builder")
    os.makedirs(player_dir, exist_ok=True)
    os.makedirs(builder_dir, exist_ok=True)
    for comp_name, value in player_state_components.items():
        _dump(os.path.join(player_dir, comp_name), value)
    for comp_name, value in builder_state_components.items():
        _dump(os.path.join(builder_dir, comp_name), value)
    _dump(os.path.join(base, "host"), host)
    if controller_bytes is not None:
        _dump(os.path.join(base, "controllers"), controller_bytes)


def load_population_state(ckpt_dir: str, name: str) -> dict[str, Any] | None:
    """One population's saved state, or None if it isn't in this checkpoint."""
    base = os.path.join(ckpt_dir, "populations", name)
    if not os.path.isdir(base):
        return None

    def _read_dir(who: str) -> dict[str, Any]:
        d = os.path.join(base, who)
        return {n: _load(os.path.join(d, n)) for n in _component_names(d)}

    host_path = os.path.join(base, "host")
    controllers_path = os.path.join(base, "controllers")
    return dict(
        player_state=_read_dir("player"),
        builder_state=_read_dir("builder"),
        host=_load(host_path) if os.path.exists(host_path) else {},
        controllers=(
            _load(controllers_path) if os.path.exists(controllers_path) else None
        ),
    )


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


def load_controller_bytes(ckpt_dir: str) -> bytes | None:
    """Controller/plasticity state; None for checkpoints written before it
    was persisted (those resume with freshly initialised controllers)."""
    path = os.path.join(ckpt_dir, "controllers")
    if not os.path.exists(path):
        return None
    return _load(path)


def load_full(ckpt_dir: str) -> dict[str, Any]:
    """Rebuild the legacy ``ckpt_data`` shape for full-restore code paths."""

    def _read_dir(who: str) -> dict[str, Any]:
        d = os.path.join(ckpt_dir, who)
        return {n: _load(os.path.join(d, n)) for n in _component_names(d)}

    try:
        meta = _load(os.path.join(ckpt_dir, "meta"))
    except Exception:
        # meta["learner_config"] is provenance-only — no restore path
        # actually reads it back (the live config always comes fresh from
        # get_learner_config(), never from a checkpoint; verified no
        # caller reads ckpt_data["meta"]). Unpickling it still runs
        # eagerly here, though, so a config dataclass schema change since
        # this checkpoint was written (a field renamed/removed — exactly
        # what a redesign like the 2026-08-12 three-population one does)
        # must not fail an otherwise-healthy resume of the actual training
        # state. Same "never fatal for inessential state" posture as
        # Learner.restore_controller_state.
        meta = None

    return dict(
        meta=meta,
        player_state=_read_dir("player"),
        builder_state=_read_dir("builder"),
        league=load_league_bytes(ckpt_dir),
        controllers=load_controller_bytes(ckpt_dir),
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
