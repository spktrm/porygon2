"""Host-side batch assembly: the static shape lattice and the stacker.

Kept out of the learner because these are pure functions over trajectories
— the shape arithmetic is the part worth testing on its own, and the OOM
history that produced it (LESSONS.md 1) makes it the part worth reading
without 2,000 lines of orchestration around it.
"""

import logging

import jax
import numpy as np


from rl.environment.interfaces import (
    Batch,
    Trajectory,
)
from rl.environment.protos.enums_pb2 import SpeciesEnum
from rl.environment.protos.features_pb2 import (
    EntityRevealedNodeFeature,
    FieldFeature,
)

logger = logging.getLogger(__name__)

# Why a snapshot was added to the league. "dominant" is the healthy path
# (the agent beat its own history); "overdue" means only the frame budget

def _chunk_required_shape(traj: Trajectory) -> tuple[int, int]:
    """Smallest (chunk_rows, history_rows) this chunk fits LOSSLESSLY.

    T: rows up to and including the done row. Trailing padding rows are
    copies of the terminal step with done zeroed (PlayerActor.make_chunk),
    so trimming them changes nothing any cumsum-done mask or [-1] outcome
    read sees — the row surviving at [-1] carries the same terminal-step
    content. A mid-game chunk has no done row and requires full length.

    H: the stored window is already tail-clipped and REBASED to packed
    row 0 (clip_history_windows_tail at the actor), so keeping every
    valid field step needs only history_rows >= valid steps and
    2 * history_rows >= valid packed rows — under which a re-clip
    degenerates to slicing zero padding (no rebase, nothing dropped).
    """
    done = np.asarray(traj.player_transitions.env_output.done)
    done_rows = done.reshape(done.shape[0], -1).any(axis=-1)
    t_req = int(done_rows.argmax()) + 1 if done_rows.any() else done.shape[0]
    field = np.asarray(traj.player_history.field)
    valid_steps = int(field[:, FieldFeature.FIELD_FEATURE__VALID].sum())
    packed_valid = int(
        (
            np.asarray(traj.player_packed_history.revealed_cache)[
                ..., EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
            ]
            != SpeciesEnum.SPECIES_ENUM___UNSPECIFIED
        ).sum()
    )
    h_req = max(valid_steps, -(-packed_valid // 2), 1)
    return t_req, h_req


def _trim_to_lattice(
    batch: list[Trajectory], lattice: tuple[tuple[int, int], ...]
) -> list[Trajectory]:
    """Slices every chunk's T/H-leading axes down to the first lattice
    combo that fits the batch's content losslessly (see
    _chunk_required_shape). The lattice is a CHAIN ascending in both dims
    whose last entry is the full stored shape, so a fitting combo always
    exists and selecting it is a max + linear scan. Slicing only — no
    padding, no rebase, no data-derived shapes: the set of shapes XLA can
    ever see is exactly the enumerated lattice."""
    if len(lattice) <= 1:
        return batch
    t_req = h_req = 1
    for traj in batch:
        t_c, h_c = _chunk_required_shape(traj)
        t_req = max(t_req, t_c)
        h_req = max(h_req, h_c)
    t_out, h_out = lattice[-1]
    for t_c, h_c in lattice:
        if t_c >= t_req and h_c >= h_req:
            t_out, h_out = t_c, h_c
            break
    if (t_out, h_out) == lattice[-1]:
        return batch
    return [
        traj.replace(
            player_transitions=jax.tree.map(
                lambda x: x[:t_out], traj.player_transitions
            ),
            player_history=jax.tree.map(lambda x: x[:h_out], traj.player_history),
            player_packed_history=jax.tree.map(
                lambda x: x[: 2 * h_out], traj.player_packed_history
            ),
        )
        for traj in batch
    ]


def _or_empty(x):
    """Trajectory side fields default to () (an empty pytree, so tree.map
    stacks it to ()) when the actor did not attach them — keep that
    sentinel rather than an empty array so consumers can test with
    isinstance(x, tuple), as they do for reuse_count."""
    return () if isinstance(x, tuple) else x


def stack_batch(
    batch: list[Trajectory],
    rng_key: jax.Array = None,
    lattice: tuple[tuple[int, int], ...] = (),
) -> Batch:
    """Stacks a list of fixed-shape trajectory chunks into a Batch.

    Chunked unrolls (2026-08-16) made every stored trajectory exactly
    (player_chunk_length, player_history_length)-shaped at the actor
    (PlayerActor.unroll), so the geometric shared-bucket machinery that
    used to live here — one clip level per batch, sized by the batch's
    longest game — is gone, and with it the whole family of
    _TRAIN_STEP_JIT shape variants it generated (each a separately
    compiled executable with its own workspace: the first top-bucket
    batch of a session, arriving once games ran long enough, is what
    OOM'd sessions 1786537634 and 1786712180).

    The static shape LATTICE (2026-08-20, config.player_shape_lattice) is
    the bounded successor: batches are trimmed to the first of a fixed,
    enumerated chain of combos that fits their content losslessly, and
    every combo is precompiled at startup (Learner._precompile_lattice) —
    the failure mode above was the surprise LATE compile of a data-derived
    shape, not the existence of a second executable."""
    batch = _trim_to_lattice(batch, tuple(lattice))
    stacked_trajectory: Trajectory = jax.tree.map(
        lambda *xs: np.stack(xs, axis=1), *batch
    )

    return Batch(
        builder_transitions=stacked_trajectory.builder_transitions,
        builder_history=stacked_trajectory.builder_history,
        player_transitions=stacked_trajectory.player_transitions,
        player_packed_history=stacked_trajectory.player_packed_history,
        player_history=stacked_trajectory.player_history,
        reuse_count=(
            ()
            if isinstance(stacked_trajectory.reuse_count, tuple)
            else stacked_trajectory.reuse_count
        ),
        game_outcome=_or_empty(stacked_trajectory.game_outcome),
        game_length=_or_empty(stacked_trajectory.game_length),
        game_step_offset=_or_empty(stacked_trajectory.game_step_offset),
        rng_key=rng_key,
    )
