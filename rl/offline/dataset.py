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

from rl.environment.data import NUM_MOVES as MOVE_VOCAB_SIZE
from rl.environment.interfaces import PlayerActorInput
from rl.environment.protos.enums_pb2 import MovesEnum
from rl.environment.protos.features_pb2 import (
    EntityEdgeFeature,
    EntityPublicNodeFeature,
    EntityRevealedNodeFeature,
    InfoFeature,
)
from rl.environment.protos.service_pb2 import EnvironmentBatch, EnvironmentTrajectory
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

# Aux survival target: y = discount**(steps to this mon's next faint), 0 if
# it never faints, binned uniformly over [0, 1]. Bins live in y-space, not
# turn-space — the only semantic hyperparameter is the discount itself.
NUM_SURVIVAL_BINS = 16
# Matches Porygon2OfflineConfig.survival_discount; used for the raw
# (config-less) view so visualiser batches stay well-formed.
DEFAULT_SURVIVAL_DISCOUNT = 0.9
# Matches Porygon2OfflineConfig.unseen_discount (see there for semantics).
DEFAULT_UNSEEN_DISCOUNT = 0.95

# History-pathway stable entity slots (both sides), as in the encoder.
NUM_SLOTS = 12

# Next-action head classes: the full move vocab plus a "never moves again"
# class (faints / game plays out before this mon acts again).
ACTION_NONE_CLASS = MOVE_VOCAB_SIZE
NUM_ACTION_CLASSES = MOVE_VOCAB_SIZE + 1

# Up to 4 revealed move slots per mon in the public view.
NUM_MOVE_SLOTS = 4

# Sentinels that mean "no revealed move in this slot". ___UNK matters most:
# the exporter initialises every revealed-team move slot to it
# (service/src/server/state.ts::getUnkRevealedPokemon), so treating it as a
# real move makes every mon look fully revealed from its first appearance
# and starves all three action/set aux heads of targets.
_INVALID_MOVE_IDS = np.array(
    [
        MovesEnum.MOVES_ENUM___UNSPECIFIED,
        MovesEnum.MOVES_ENUM___NULL,
        MovesEnum.MOVES_ENUM___PAD,
        MovesEnum.MOVES_ENUM___UNK,
    ]
)


@chex.dataclass(frozen=True)
class OfflineExample:
    actor_input: PlayerActorInput  # env leaves (T, ...), histories unbatched
    label: ArrayLike  # (NUM_MARGIN_BINS,) one-hot over final margin
    # Allowed-bins mask per (step, slot): one-hot where the faint step was
    # observed, an interval mask where right-censored. Loss is
    # -log(predicted mass inside the mask) — exact rows reduce to CE.
    survival_target: ArrayLike  # (T, NUM_SLOTS, NUM_SURVIVAL_BINS)
    survival_mask: ArrayLike  # (T, NUM_SLOTS) — revealed & currently alive
    # Next-action aux: the id of the next move this slot's mon executes
    # after step t (any distance into the future), ACTION_NONE_CLASS when
    # the replay played out without it acting again, -1 = no supervision
    # (unrevealed/fainted slot, or right-censored by an early ending).
    action_target: ArrayLike  # (T, NUM_SLOTS) int32
    # 1.0 where the labelled next action is that move's FIRST use — i.e.
    # the model is being asked to predict a move it has never seen. The
    # loss slice over these rows is the anticipation learning curve.
    action_is_reveal: ArrayLike  # (T, NUM_SLOTS)
    # Unseen-move hazard aux: allowed-bins masks over y =
    # unseen_discount**(requests until this mon next uses a move that is
    # unrevealed as of step t); same censoring scheme as survival_target.
    unseen_target: ArrayLike  # (T, NUM_SLOTS, NUM_SURVIVAL_BINS)
    unseen_mask: ArrayLike  # (T, NUM_SLOTS)
    # Set-prediction aux (eventually-revealed moves), shipped compactly:
    # per slot the move ids revealed by game end (-1 pad), the step each
    # was first revealed at (T when padded), and whether the mon's full
    # move count was eventually observed (making non-set moves certain
    # negatives). Positives at step t are the entries with reveal_step > t.
    set_eventual: ArrayLike  # (NUM_SLOTS, NUM_MOVE_SLOTS) int32
    set_reveal_step: ArrayLike  # (NUM_SLOTS, NUM_MOVE_SLOTS) int32
    set_fully_observed: ArrayLike  # (NUM_SLOTS,)
    set_mask: ArrayLike  # (T, NUM_SLOTS) — revealed, alive, set not yet full


@chex.dataclass(frozen=True)
class OfflineBatch:
    actor_input: PlayerActorInput  # leaves (T, B, ...)
    labels: ArrayLike  # (B, NUM_MARGIN_BINS)
    survival_targets: ArrayLike  # (T, B, NUM_SLOTS, NUM_SURVIVAL_BINS)
    survival_masks: ArrayLike  # (T, B, NUM_SLOTS)
    action_targets: ArrayLike  # (T, B, NUM_SLOTS) int32
    action_is_reveal: ArrayLike  # (T, B, NUM_SLOTS)
    unseen_targets: ArrayLike  # (T, B, NUM_SLOTS, NUM_SURVIVAL_BINS)
    unseen_masks: ArrayLike  # (T, B, NUM_SLOTS)
    set_eventual: ArrayLike  # (B, NUM_SLOTS, NUM_MOVE_SLOTS) int32
    set_reveal_steps: ArrayLike  # (B, NUM_SLOTS, NUM_MOVE_SLOTS) int32
    set_fully_observed: ArrayLike  # (B, NUM_SLOTS)
    set_masks: ArrayLike  # (T, B, NUM_SLOTS)


def list_shards(config: Porygon2OfflineConfig) -> list[str]:
    shard_dir = os.path.join(config.dataset_dir, config.format_id)
    if not os.path.isdir(shard_dir):
        raise FileNotFoundError(
            f"No shard directory at {shard_dir} — run the offline exporter "
            f"(service/src/scripts/offline.ts) first."
        )
    shards = sorted(
        os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith(".bin")
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


def _final_margin(
    final: PlayerActorInput, win_reward: np.ndarray
) -> tuple[int, str, int]:
    """Final alive-mon differential from the terminal cache (every fainted
    mon is revealed, so alive = 6 - faints exactly), plus how the game
    ended. The sign is clamped to the recorded result: a mid-game forfeit
    can leave the winner behind on mons, and the result is the ground truth.

    Endings (measured on 50k rated gen9randombattle games, July 2026):
    - "played_out" (~48%): the loser's six mons all fainted — exact margin.
    - "conceded" (~41%): forfeit/timeout with the winner ahead on mons —
      the margin is the count at concession, a compressed lower bound on
      the played-out margin (concessions cluster at 1-3, played-out games
      reach 4-6 far more often).
    - "clamped" (~11%): forfeit/timeout with the winner NOT ahead (rage
      quit / timer / disconnect) — the position contradicts the result, so
      the ±1 margin is pure label noise.
    - "tie": rare, margin 0.

    Also returns the |margin| cap: the winner's alive-mon count at game
    end. Mons never return (Revival Blessing aside), so no played-out
    continuation of a conceded game could have exceeded that margin — the
    censored label spreads only up to it, never to ±MAX_MARGIN.
    """
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
        margin = max(min(alive_diff, MAX_MARGIN), 1)
        cap = min(6 - my_faints, MAX_MARGIN)
        if alive_diff <= 0:
            return margin, "clamped", cap
        return margin, ("played_out" if opp_faints == 6 else "conceded"), cap
    if win_reward[0] == 1:
        margin = min(max(alive_diff, -MAX_MARGIN), -1)
        cap = min(6 - opp_faints, MAX_MARGIN)
        if alive_diff >= 0:
            return margin, "clamped", cap
        return margin, ("played_out" if my_faints == 6 else "conceded"), cap
    return 0, "tie", 0


def _margin_label(
    margin: int, ending: str, censor_decay: float, margin_cap: int
) -> np.ndarray:
    """One-hot margin label, except conceded games with censoring on: a
    concession at |margin| mons down right-censors the played-out margin
    (it would have been at least that), so the mass spreads geometrically
    over the bins from the observed margin up to ``margin_cap`` — the
    winner's alive count, the hardest possible played-out margin from that
    position. The mode stays at the observed bin, and the two perspectives
    of a game remain exact mirror flips of each other (the cap is
    perspective-independent), so pair-aware batching is undisturbed."""
    label = np.zeros(NUM_MARGIN_BINS, dtype=np.float32)
    if ending == "conceded" and censor_decay > 0.0 and abs(margin) < margin_cap:
        sign = 1 if margin > 0 else -1
        weight = 1.0
        for k in range(abs(margin), margin_cap + 1):
            label[sign * k + MAX_MARGIN] = weight
            weight *= censor_decay
        label /= label.sum()
    else:
        label[margin + MAX_MARGIN] = 1.0
    return label


def _slot_view(env) -> dict[str, np.ndarray]:
    """Scatters the per-step, per-side public/revealed team rows into the
    history pathway's stable entity-slot indexing (revelation order across
    both sides): the per-step public team rows are per-side and re-sorted
    actives-first, so PUBLIC_ORDER (row i of the public team holds the mon
    in slot public_order[i], -1 for unrevealed fillers) scatters per-step
    features back to slots — the same alignment the RL trunk uses to
    gather history rows."""
    info = np.asarray(env.info)
    public_team = np.asarray(env.public_team)
    revealed_team = np.asarray(env.revealed_team)
    num_steps = info.shape[0]
    order = info[
        :,
        InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0 : InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11
        + 1,
    ].astype(np.int64)
    row_valid = (order >= 0) & (order < NUM_SLOTS)
    move_id_features = np.array(
        [
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID0,
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID1,
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID2,
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID3,
        ]
    )
    move_pp_features = np.array(
        [
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP0,
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP1,
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP2,
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP3,
        ]
    )
    revealed = np.zeros((num_steps, NUM_SLOTS), dtype=bool)
    fainted = np.zeros((num_steps, NUM_SLOTS), dtype=bool)
    move_ids = np.full(
        (num_steps, NUM_SLOTS, NUM_MOVE_SLOTS),
        MovesEnum.MOVES_ENUM___UNSPECIFIED,
        dtype=np.int64,
    )
    move_pp = np.zeros((num_steps, NUM_SLOTS, NUM_MOVE_SLOTS), dtype=np.int64)
    num_moves = np.zeros((num_steps, NUM_SLOTS), dtype=np.int64)
    step_idx = np.arange(num_steps)
    for i in range(NUM_SLOTS):  # one write per (step, row): no scatter races
        rows = row_valid[:, i]
        t = step_idx[rows]
        s = order[rows, i]
        revealed[t, s] = True
        fainted[t, s] = (
            public_team[
                rows, i, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED
            ]
            > 0
        )
        move_ids[t, s] = revealed_team[rows, i][:, move_id_features]
        move_pp[t, s] = public_team[rows, i][:, move_pp_features]
        num_moves[t, s] = public_team[
            rows, i, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__NUM_MOVES
        ]
    return dict(
        revealed=revealed,
        fainted=fainted,
        move_ids=move_ids,
        move_pp=move_pp,
        num_moves=num_moves,
    )


def _discounted_bin_targets(
    dist: np.ndarray, exact_zero: np.ndarray, ending: str, discount: float
) -> np.ndarray:
    """Allowed-bins masks over NUM_SURVIVAL_BINS uniform bins in [0, 1] for
    y = discount**dist:
    - finite dist (event witnessed) -> one-hot at bin(y);
    - exact_zero rows (event structurally impossible, or witnessed absent
      through a played-out ending) -> exact y = 0;
    - otherwise, when the game ended early (concession/timeout/tie on
      timer) -> right-censored: the replay only witnessed "did not happen
      before the end", so every bin with y <= discount**(T_last - t) is
      allowed and the loss constrains nothing below that bound. At the
      terminal step the bound is 1 (no information, zero loss) and it
      tightens exponentially for earlier states."""
    num_steps = dist.shape[0]
    observed = np.isfinite(dist)
    y = np.where(observed, discount ** np.where(observed, dist, 0.0), 0.0)
    y = np.where(exact_zero, 0.0, y)
    bin_idx = np.minimum(
        (y * NUM_SURVIVAL_BINS).astype(np.int64), NUM_SURVIVAL_BINS - 1
    )
    target = np.eye(NUM_SURVIVAL_BINS, dtype=np.float32)[bin_idx]
    if ending != "played_out":
        step_idx = np.arange(num_steps)
        bound = discount ** (num_steps - 1 - step_idx)  # (T,)
        bound_bin = np.minimum(
            (bound * NUM_SURVIVAL_BINS).astype(np.int64), NUM_SURVIVAL_BINS - 1
        )
        allowed = (
            np.arange(NUM_SURVIVAL_BINS)[None, None, :] <= bound_bin[:, None, None]
        ).astype(np.float32)
        known = observed | exact_zero
        target = np.where(
            known[..., None], target, np.broadcast_to(allowed, target.shape)
        )
    return target


def _survival_targets(
    view: dict[str, np.ndarray], ending: str, discount: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-slot discounted survival targets for the aux head.

    Target variable per (step t, slot): y = discount**(steps to the slot's
    next faint), 0 if it never faints (handles Revival Blessing: distance is
    to the NEXT faint), encoded via _discounted_bin_targets (no dies-in-x
    window). The loss mask is (revealed & not currently fainted): the head
    predicts the future of live, visible mons only.
    """
    revealed, fainted = view["revealed"], view["fainted"]
    num_steps = fainted.shape[0]

    # Steps to next faint, per slot; inf = never faints in the replay.
    dist = np.full((num_steps, NUM_SLOTS), np.inf)
    run = np.full(NUM_SLOTS, np.inf)
    for t in range(num_steps - 1, -1, -1):
        run = np.where(fainted[t], 0.0, run + 1.0)
        dist[t] = run

    target = _discounted_bin_targets(
        dist, np.zeros_like(fainted), ending, discount
    )
    mask = (revealed & ~fainted).astype(np.float32)
    return target, mask


def _action_targets(
    view: dict[str, np.ndarray], ending: str, unseen_discount: float
) -> dict[str, np.ndarray]:
    """Self-supervised action/set targets, all derived from the replay's
    own event stream (no reveal-ontology beyond what the grammar records):

    - action_target: per (step t, slot) the id of the next move the mon
      executes after t. Usage events are recovered as (first appearance of
      a move id in the revealed rows — in randbats a move is revealed BY
      its first use) ∪ (a PP decrement on an already-revealed move slot).
      Played-out games label "never acts again" with ACTION_NONE_CLASS;
      early endings right-censor it (-1, no loss). To predict this well
      before the reveal, the encoder must hold a posterior over the mon's
      unseen slots — the set prior anticipation is built from.
    - action_is_reveal: 1 where that label is a first use, i.e. the model
      is asked to name a move it has never seen — the anticipation slice.
    - unseen_target/mask: allowed-bins masks over y = unseen_discount**
      (requests until the mon next uses a move unrevealed as of t), the
      hazard that carries "an unseen move is coming" at a longer horizon
      than the survival head's imminent-doom scale. Mons whose full move
      count is already revealed get an EXACT y = 0 regardless of ending —
      the event is structurally impossible, which teaches "fully revealed
      = no latent surprise" for free.
    - set_*: compact positives/negatives for the eventually-revealed set
      head. Positives at step t are moves revealed later than t (certain
      members of the set); negatives exist only for mons whose whole move
      count was eventually observed (positive-unlabelled otherwise: an
      unrevealed move may well be carried, so it earns no loss).
    """
    revealed, fainted = view["revealed"], view["fainted"]
    move_ids, move_pp = view["move_ids"], view["move_pp"]
    num_steps = revealed.shape[0]

    valid_move = ~np.isin(move_ids, _INVALID_MOVE_IDS)  # (T, S, 4)
    # First step each (slot, move-slot) is revealed at; T when never.
    any_reveal = valid_move.any(axis=0)  # (S, 4)
    first_reveal = np.where(any_reveal, valid_move.argmax(axis=0), num_steps)
    slot_idx = np.arange(NUM_SLOTS)[:, None]
    move_slot_idx = np.arange(NUM_MOVE_SLOTS)[None, :]
    eventual = np.where(
        any_reveal,
        move_ids[first_reveal.clip(max=num_steps - 1), slot_idx, move_slot_idx],
        -1,
    ).astype(np.int32)

    revealed_count = valid_move.sum(axis=-1)  # (T, S)
    num_moves = view["num_moves"].max(axis=0)  # (S,) constant per mon
    num_moves = np.where(num_moves > 0, num_moves, NUM_MOVE_SLOTS).clip(
        max=NUM_MOVE_SLOTS
    )

    # Usage events per (t, slot, move-slot): the reveal itself, or a rise
    # in the slot's used-PP counter (the exporter stores PP USED, scaled
    # 0..31 — see service/src/server/state.ts — so usage increments it;
    # Pressure double-increments still register once; bucket rounding can
    # hide a rare high-max-PP use, but first uses — the
    # anticipation-relevant ones — always reveal).
    step_idx = np.arange(num_steps)
    reveal_event = valid_move & (step_idx[:, None, None] == first_reveal[None])
    pp_used_rise = np.zeros_like(valid_move)
    pp_used_rise[1:] = (
        valid_move[1:]
        & valid_move[:-1]
        & (move_ids[1:] == move_ids[:-1])
        & (move_pp[1:] > move_pp[:-1])
    )
    event = reveal_event | pp_used_rise
    has_event = event.any(axis=-1)  # (T, S)
    event_move_slot = event.argmax(axis=-1)
    used_id = np.where(
        has_event,
        np.take_along_axis(move_ids, event_move_slot[..., None], axis=-1)[..., 0],
        -1,
    )
    used_is_reveal = has_event & np.take_along_axis(
        reveal_event, event_move_slot[..., None], axis=-1
    )[..., 0]

    # Next usage strictly after t, and requests until the next reveal
    # (= first use of a then-unseen move) strictly after t.
    next_id = np.full((num_steps, NUM_SLOTS), -1, dtype=np.int64)
    next_is_reveal = np.zeros((num_steps, NUM_SLOTS), dtype=bool)
    reveal_dist = np.full((num_steps, NUM_SLOTS), np.inf)
    run_id = np.full(NUM_SLOTS, -1, dtype=np.int64)
    run_reveal = np.zeros(NUM_SLOTS, dtype=bool)
    run_dist = np.full(NUM_SLOTS, np.inf)
    reveal_at = reveal_event.any(axis=-1)
    for t in range(num_steps - 1, -1, -1):
        next_id[t] = run_id
        next_is_reveal[t] = run_reveal
        reveal_dist[t] = run_dist + 1.0
        acted = used_id[t] >= 0
        run_id = np.where(acted, used_id[t], run_id)
        run_reveal = np.where(acted, used_is_reveal[t], run_reveal)
        run_dist = np.where(reveal_at[t], 0.0, run_dist + 1.0)

    base = revealed & ~fainted
    action_target = np.where(
        next_id >= 0,
        next_id,
        ACTION_NONE_CLASS if ending == "played_out" else -1,
    )
    action_target = np.where(base, action_target, -1).astype(np.int32)
    action_is_reveal = (next_is_reveal & (action_target >= 0)).astype(np.float32)

    fully_revealed = revealed_count >= num_moves[None]  # (T, S)
    unseen_target = _discounted_bin_targets(
        reveal_dist, fully_revealed, ending, unseen_discount
    )
    unseen_mask = base.astype(np.float32)

    eventual_count = (eventual >= 0).sum(axis=-1)  # (S,)
    return dict(
        action_target=action_target,
        action_is_reveal=action_is_reveal,
        unseen_target=unseen_target,
        unseen_mask=unseen_mask,
        set_eventual=eventual,
        set_reveal_step=first_reveal.astype(np.int32),
        set_fully_observed=(eventual_count >= num_moves).astype(np.float32),
        set_mask=(base & ~fully_revealed).astype(np.float32),
    )


def record_to_examples(
    payload: bytes, config: "Porygon2OfflineConfig | None" = None
) -> list[OfflineExample]:
    """One shard record = one replay (EnvironmentBatch holding both
    perspectives). Grouping them keeps a game and its mirrored,
    label-flipped twin on the same side of the train/eval split.

    With a config, its forfeit policy applies (drop_clamped_forfeits /
    concession_censor_decay); without one, every decided game is kept with
    its exact one-hot label — the raw view (used by the visualiser)."""
    batch = EnvironmentBatch.FromString(payload)
    examples = [trajectory_to_example(t, config) for t in batch.trajectories]
    return [e for e in examples if e is not None]


def trajectory_to_example(
    trajectory: EnvironmentTrajectory,
    config: "Porygon2OfflineConfig | None" = None,
) -> OfflineExample | None:
    if len(trajectory.states) < 2:
        return None
    # Only the final state's (large) history caches are kept per
    # trajectory, so skip materialising them for every other state.
    steps = [
        process_state(state, with_history=False) for state in trajectory.states[:-1]
    ]
    steps.append(process_state(trajectory.states[-1]))
    final = steps[-1]
    label = final.env.win_reward
    # A trajectory without a decided outcome (no |win|/|tie| in the log,
    # e.g. a forfeited-by-disconnect fragment) carries no training signal.
    if not final.env.done or label.sum() == 0:
        return None
    margin, ending, margin_cap = _final_margin(final, label)
    # A forfeit where the "winner" wasn't ahead contradicts the position at
    # every step of the game — deep supervision would train the whole
    # trajectory toward an outcome the states never indicated.
    if config is not None and config.drop_clamped_forfeits and ending == "clamped":
        return None
    censor_decay = config.concession_censor_decay if config is not None else 0.0
    margin_label = _margin_label(margin, ending, censor_decay, margin_cap)
    env = jax.tree.map(lambda *xs: np.stack(xs), *[s.env for s in steps])
    survival_discount = (
        config.survival_discount if config is not None else DEFAULT_SURVIVAL_DISCOUNT
    )
    unseen_discount = (
        config.unseen_discount if config is not None else DEFAULT_UNSEEN_DISCOUNT
    )
    view = _slot_view(env)
    survival_target, survival_mask = _survival_targets(view, ending, survival_discount)
    aux = _action_targets(view, ending, unseen_discount)
    return OfflineExample(
        actor_input=PlayerActorInput(
            env=env,
            packed_history=final.packed_history,
            history=final.history,
        ),
        label=margin_label,
        survival_target=survival_target,
        survival_mask=survival_mask,
        **aux,
    )


def _drop_ratings(record: Sequence[OfflineExample], rng, dropout: float) -> None:
    """With probability ``dropout``, zeroes the rating info features of
    every perspective of one game in place (0 = the unknown bucket).
    Applied per game so mirrored pairs stay exact mirrors."""
    if dropout <= 0.0 or rng.random() >= dropout:
        return
    for example in record:
        info = np.asarray(example.actor_input.env.info)
        info[:, InfoFeature.INFO_FEATURE__MY_RATING] = 0
        info[:, InfoFeature.INFO_FEATURE__OPP_RATING] = 0


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


def _pad_time_zeros(x: np.ndarray, length: int, fill=0) -> np.ndarray:
    """Pads (or truncates) a time-major target array with ``fill`` — padded
    steps carry an all-zero loss mask (or a -1 no-supervision label), so
    they never contribute."""
    x = np.asarray(x)
    if x.shape[0] >= length:
        return x[:length]
    pad = np.full((length - x.shape[0],) + x.shape[1:], fill, dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)


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

    def stack_time(field: str, fill=0) -> np.ndarray:
        return np.stack(
            [_pad_time_zeros(getattr(e, field), bucket_t, fill) for e in examples],
            axis=1,
        )

    return OfflineBatch(
        actor_input=PlayerActorInput(
            env=env, packed_history=packed_history, history=history
        ),
        labels=labels,
        survival_targets=stack_time("survival_target"),
        survival_masks=stack_time("survival_mask"),
        action_targets=stack_time("action_target", fill=-1),
        action_is_reveal=stack_time("action_is_reveal"),
        unseen_targets=stack_time("unseen_target"),
        unseen_masks=stack_time("unseen_mask"),
        set_eventual=np.stack([e.set_eventual for e in examples], axis=0),
        set_reveal_steps=np.stack([e.set_reveal_step for e in examples], axis=0),
        set_fully_observed=np.stack(
            [e.set_fully_observed for e in examples], axis=0
        ),
        set_masks=stack_time("set_mask"),
    )


def collate_ensemble(
    member_examples: Sequence[Sequence[OfflineExample]],
    config: Porygon2OfflineConfig,
) -> OfflineBatch:
    """Collates K members' example lists into one batch with a leading
    member axis: actor-input leaves (K, T, B, ...), labels (K, B, bins).

    A single combined collate over all K·B examples guarantees every
    member shares the same time/history buckets (so the member axis is
    stackable); the batch axis is then regrouped member-major."""
    num_members = len(member_examples)
    batch_size = len(member_examples[0])
    assert all(len(m) == batch_size for m in member_examples)
    combined = collate([e for member in member_examples for e in member], config)

    def regroup(x: np.ndarray, axis: int) -> np.ndarray:
        x = np.asarray(x)
        shape = x.shape
        x = x.reshape(shape[:axis] + (num_members, batch_size) + shape[axis + 1 :])
        return np.moveaxis(x, axis, 0)

    return OfflineBatch(
        actor_input=jax.tree.map(lambda x: regroup(x, 1), combined.actor_input),
        labels=regroup(combined.labels, 0),
        survival_targets=regroup(combined.survival_targets, 1),
        survival_masks=regroup(combined.survival_masks, 1),
        action_targets=regroup(combined.action_targets, 1),
        action_is_reveal=regroup(combined.action_is_reveal, 1),
        unseen_targets=regroup(combined.unseen_targets, 1),
        unseen_masks=regroup(combined.unseen_masks, 1),
        set_eventual=regroup(combined.set_eventual, 0),
        set_reveal_steps=regroup(combined.set_reveal_steps, 0),
        set_fully_observed=regroup(combined.set_fully_observed, 0),
        set_masks=regroup(combined.set_masks, 1),
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
                    and _ensemble_bucket(shard, index, self.config.num_ensemble_splits)
                    != self.config.ensemble_index
                ):
                    continue
                examples = record_to_examples(payload, self.config)
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
                _drop_ratings(record, rng, self.config.rating_dropout)
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

    def train_batches_ensemble(self, seed: int = 0) -> Iterator[OfflineBatch]:
        """Infinite iterator for simultaneous ensemble training: leaves
        carry a leading (num_ensemble_splits,) member axis.

        One shard pass feeds every member — each record is parsed once and
        routed to its member by the same salted hash as --ensemble-index
        runs, so member k trains on exactly the games it would see in a
        separate run. Each member's stream is pair-aware like
        train_batches; a batch is emitted only when every member's buffer
        is filled, keeping per-member shuffle quality equal to a separate
        run (total buffered records are K× a single run's)."""
        config = self.config
        num_members = config.num_ensemble_splits
        rng = np.random.default_rng(seed)
        games_per_batch = max(1, config.batch_size // 2)
        fill = max(config.shuffle_buffer_size // 2, games_per_batch)
        buffers: list[list[list[OfflineExample]]] = [[] for _ in range(num_members)]

        def pop_member_batch(buffer: list[list[OfflineExample]]):
            picks = rng.choice(len(buffer), size=games_per_batch, replace=False)
            batch = [e for i in picks for e in buffer[i]][: config.batch_size]
            for i in sorted(picks, reverse=True):
                buffer.pop(i)
            return batch

        while True:
            rng.shuffle(self.shards)
            for shard in self.shards:
                for index, payload in enumerate(iter_shard_payloads(shard)):
                    if _is_holdout(shard, index, config.holdout_modulus):
                        continue
                    member = _ensemble_bucket(shard, index, num_members)
                    examples = record_to_examples(payload, config)
                    if len(examples) != 2:  # pairs only, as in train_batches
                        continue
                    _drop_ratings(examples, rng, config.rating_dropout)
                    buffers[member].append(examples)
                    if all(len(b) >= fill for b in buffers):
                        yield collate_ensemble(
                            [pop_member_batch(b) for b in buffers], config
                        )
            # Flush what remains at epoch end so small datasets still train.
            while all(len(b) >= games_per_batch for b in buffers):
                yield collate_ensemble([pop_member_batch(b) for b in buffers], config)

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