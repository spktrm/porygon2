"""Read unilateral request intervals without joining players or importing JAX.

History indices refer to the chunk's original field/packed window. Callers must
not independently slice packed caches. These are observed history steps, not a
count of opponent submissions, and may include our own actions or chance.
"""

import hashlib
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from rl.environment.protos.features_pb2 import FieldFeature, InfoFeature


@dataclass(frozen=True)
class RequestInterval:
    source_row: int
    successor_row: int
    game_step: int
    action_index: int
    source_request: int
    successor_request: int
    terminal: bool
    request_type: int
    num_active: int
    has_previous_action: bool
    successor_has_previous_action: bool
    history_indices: tuple[int, ...]
    history_prefix_retained: bool


def game_split(game_id: str, *, seed: int = 0, heldout_fraction: float = 0.2) -> str:
    """Stable game-level split; both perspectives must supply the same ID.

    Legacy trajectory objects do not carry game IDs or evaluation provenance.
    Callers must provide these from their collection manifest; never infer an
    identity from private teams or independently split individual chunks.
    """
    if not game_id or not 0.0 <= heldout_fraction <= 1.0:
        raise ValueError("A nonempty game ID and fraction in [0, 1] are required")
    digest = hashlib.sha256(f"{seed}:{game_id}".encode()).digest()
    quantile = int.from_bytes(digest[:8], "big") / 2**64
    if quantile < heldout_fraction:
        return "heldout"
    return "train"


def evaluation_partitions(arrays):
    """Keep entire games disjoint across training, validation and final test."""
    heldout = np.asarray(arrays["heldout"], dtype=bool)
    final = np.asarray(arrays.get("final_test", np.zeros_like(heldout)), dtype=bool)
    eligible = np.asarray(arrays.get("train_eligible", ~heldout), dtype=bool)
    games = np.asarray(arrays["game"])
    if not (heldout.shape == final.shape == eligible.shape == games.shape):
        raise ValueError("Partition masks must align with game identities")
    if np.any(final & ~heldout) or np.any(eligible & heldout):
        raise ValueError("Final test must be held out and training must be disjoint")
    labels = eligible.astype(int) + 2 * (heldout & ~final) + 3 * final
    for game in np.unique(games):
        if len(np.unique(labels[games == game])) != 1:
            raise ValueError(f"Game crosses evaluation partitions: {game}")
    return (
        np.flatnonzero(eligible),
        np.flatnonzero(heldout & ~final),
        np.flatnonzero(final),
    )


def iter_intervals(chunk) -> Iterator[RequestInterval]:
    """Yield real adjacent own decisions, excluding bootstrap-only and pad rows.

    Same-request microsteps are retained, including previous choices in doubles.
    Terminal successors are included, but terminal rows never supply actions.
    ``history_prefix_retained`` only certifies that the retained history starts
    no later than this request; it does not certify service alignment or absence
    of later history rewrites. Missing coverage must not mean opponent inactivity.
    """
    transitions = chunk.player_transitions
    environment = transitions.env_output
    info = np.asarray(environment.info)
    done = np.asarray(environment.done).reshape(-1).astype(bool)
    actions = np.asarray(
        transitions.agent_output.actor_output.action_head.action_index
    ).reshape(-1)
    legal = np.asarray(environment.action_mask).astype(bool)
    if info.ndim != 2 or done.size != info.shape[0] or actions.size != done.size:
        raise ValueError("Expected one unbatched chunk with aligned decision rows")
    if legal.ndim != 2 or legal.shape[0] != done.size:
        raise ValueError("Expected an action mask for each decision row")
    offset_values = np.asarray(chunk.game_step_offset).reshape(-1)
    if offset_values.size:
        offset = int(offset_values[0])
    else:
        raise ValueError("Game step offsets are required for overlap-safe identities")
    length_values = np.asarray(chunk.game_length).reshape(-1)
    if length_values.size:
        real_rows = min(done.size, int(length_values[0]) - offset)
    else:
        real_rows = done.size
    terminal_rows = np.flatnonzero(done[:real_rows])
    if terminal_rows.size:
        real_rows = int(terminal_rows[0]) + 1

    history = np.asarray(chunk.player_history.field)
    if history.ndim == 2 and history.shape[0]:
        history_valid = history[:, FieldFeature.FIELD_FEATURE__VALID].astype(bool)
        history_requests = history[:, FieldFeature.FIELD_FEATURE__REQUEST_COUNT]
    else:
        history_valid = np.zeros(0, dtype=bool)
        history_requests = np.zeros(0, dtype=np.int32)
    request_column = InfoFeature.INFO_FEATURE__REQUEST_COUNT
    previous_column = InfoFeature.INFO_FEATURE__HAS_PREV_ACTION
    for source_row in range(max(real_rows - 1, 0)):
        successor_row = source_row + 1
        action_index = int(actions[source_row])
        if (
            not 0 <= action_index < legal.shape[1]
            or not legal[source_row, action_index]
        ):
            raise ValueError(f"Illegal submitted action at row {source_row}")
        source_request = int(info[source_row, request_column])
        successor_request = int(info[successor_row, request_column])
        if successor_request < source_request:
            raise ValueError("Request counters must not decrease within a chunk")
        interval_steps = history_valid & (history_requests > source_request)
        interval_steps &= history_requests <= successor_request
        yield RequestInterval(
            source_row=source_row,
            successor_row=successor_row,
            game_step=offset + source_row,
            action_index=action_index,
            source_request=source_request,
            successor_request=successor_request,
            terminal=bool(done[successor_row]),
            request_type=int(info[source_row, InfoFeature.INFO_FEATURE__REQUEST_TYPE]),
            num_active=int(info[source_row, InfoFeature.INFO_FEATURE__NUM_ACTIVE]),
            has_previous_action=bool(info[source_row, previous_column]),
            successor_has_previous_action=bool(info[successor_row, previous_column]),
            history_indices=tuple(np.flatnonzero(interval_steps).tolist()),
            history_prefix_retained=bool(
                np.any(history_valid & (history_requests <= source_request))
            ),
        )
