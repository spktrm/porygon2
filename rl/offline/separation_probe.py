"""Within-modality separation probe — the pre-launch capacity gate.

Two probes over one fixed batch of real self-play states, both answering
"can this ARCHITECTURE tell a modality's candidates apart?" before any
training time is spent (docs: the 2026-08-27 measurement found within-row
switch Q std 0.0196 vs move 0.0374 vs between-row 0.507 on the live
checkpoint — the collapse's representational reading).

Probe A — routing. Swap the SPECIES feature of two legal reserves in one
batch column (species -> embedding is computed at forward time, legality
untouched, history caches join by index not species) and compare per-cell
|delta log pi|: the swapped reserves' own cells vs sibling switch cells.
A shared pooled representation moves siblings as much as the perturbed
cell; genuine per-candidate routing separates them. Read at fresh init with
the action readout's zero-init paths OPENED (else the grid is
exactly 0 and everything is vacuous), and on trained params as-is.

Probe B — overfit separation (THE GATE). Synthetic per-cell labels keyed to
the candidate's IDENTITY (species id for switch/entity-target cells, move
id for move cells) hashed to a standard normal, then CENTRED within each
modality over the row's legal cells and scaled by --label-scale. Centring
removes all state-level and between-modality signal, and identity keying
removes the positional shortcut (a per-cell bias fits position-keyed
labels with zero input reading) — only within-modality discrimination of
the actual candidates can fit these. Direct f32 MSE on the log-policy grid over
the eval cells, full-param Adam, jitted; per-modality within-row Pearson r
and pred-std/label-std every --eval-every steps.

Pre-registered criteria (plan 2026-08-27): the new architecture passes at
switch-group r >= 0.9 AND std ratio >= 0.8 by step 1000; if the CURRENT
architecture also clears that comfortably, the capacity hypothesis is
falsified and the launch does not happen on this justification.

Usage (learner down; games from a pickle, so no service needed):

    env/bin/python -m rl.offline.separation_probe \\
        --games-pkl runtime/discrim_sides_ckpt224773.pkl --probe both
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax

from constants import MAX_RATIO_TOKEN
from rl.environment.data import (
    ALLY_TARGET_INDICES,
    CELL_MODALITY_MASK,
    ENEMY_TARGET_INDICES,
    MOVE_CELL_OFFSET,
    MOVE_INDICES,
    NUM_MOVE_SLOTS,
    NUM_SWITCH_CELLS,
    NUM_TARGET_SLOTS,
    OTHER_CELL_OFFSET,
    TARGET_SLOT_INDICES,
    WILDCARD_MOVE_INDICES,
)
from rl.environment.interfaces import PlayerActorInput, Trajectory
from rl.environment.protos.features_pb2 import (
    EntityPrivateNodeFeature,
    EntityPublicNodeFeature,
    EntityRevealedNodeFeature,
    MovesetFeature,
)
from rl.environment.protos.service_pb2 import ModalityEnum
from rl.model.config import get_player_model_config
from rl.model.constants import (
    ALLY_TARGET_ROWS,
    ENEMY_TARGET_ROWS,
    PRIVATE_ROWS,
    TARGET_ROWS,
)
from rl.model.heads import HeadParams
from rl.model.player_model import get_player_model
from rl.model.utils import open_zero_init_paths
from rl.offline import harness
from rl.online.training.batching import stack_batch

logger = logging.getLogger(__name__)

# Batch selection, moved here from rl/offline/overfit_probe.py when that
# file retired with the Q head on 2026-08-29. This probe was its only
# remaining consumer.
_FLAT = np.asarray(CELL_MODALITY_MASK)
_SWITCH_CELLS = _FLAT == ModalityEnum.MODALITY_ENUM__SWITCH
_MOVE_CELLS = _FLAT == ModalityEnum.MODALITY_ENUM__MOVE


def trainable_rows(done: np.ndarray) -> np.ndarray:
    """The learner's acted_mask along the time axis: rows up to and including
    the first done row (the terminal-copy padding after it repeats the
    terminal row with done=0), minus the bootstrap-only final row unless
    it is the game's own terminal, minus done rows themselves."""
    done = done.astype(bool)
    before_or_at_done = (np.cumsum(done, axis=0) - done) == 0
    rows = before_or_at_done.copy()
    rows[-1] &= done[-1]
    return rows & ~done


def voluntary_switch_rows(chunk: Trajectory) -> int:
    """Rows where a switch was taken with a legal move available, over the
    chunk's trainable rows (not done, not the bootstrap-only final row)."""
    env = chunk.player_transitions.env_output
    flat = np.asarray(env.action_mask)
    idx = np.asarray(
        chunk.player_transitions.agent_output.actor_output.action_head.action_index
    )
    rows = trainable_rows(np.asarray(env.done).astype(bool))
    taken_switch = _SWITCH_CELLS[idx]
    has_move = (flat & _MOVE_CELLS).any(-1)
    return int((rows & taken_switch & has_move).sum())


def pick_batches(chunks: list[Trajectory], batch: int, pool: int = 1):
    """Top `pool * batch` chunks by voluntary-switch rows, dealt round-robin
    into `pool` training batches (so every batch is switch-rich), the next
    `batch` as the held-out batch (also switch-rich, so the held-out
    voluntary panels have rows). Returns (train_batches, held_batch)."""
    ranked = sorted(chunks, key=voluntary_switch_rows, reverse=True)
    if len(ranked) < (pool + 1) * batch:
        raise ValueError(
            f"{len(ranked)} chunks < {(pool + 1) * batch} needed (pool {pool})"
        )
    train = [ranked[i : pool * batch : pool] for i in range(pool)]
    held = ranked[pool * batch : (pool + 1) * batch]
    logger.info(
        "pool %d: batch-0 voluntary rows %s; held-out %s",
        pool,
        [voluntary_switch_rows(c) for c in train[0]],
        [voluntary_switch_rows(c) for c in held],
    )
    return [stack_batch(b) for b in train], stack_batch(held)


_SPECIES_PRIV = EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SPECIES
_SPECIES_REV = EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
_MOVE_ID = MovesetFeature.MOVESET_FEATURE__MOVE_ID
_HP_RATIO = EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO
_FAINTED = EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED
# Entity-backed target cells: ally targets key revealed rows 0/1, enemy
# targets revealed rows 6/7; the learned TARGET_*/PASS slots carry no
# entity and are excluded from identity labels. Keys are BLOCK CELLS (the
# standalone block entry of each target slot).
_target_row_of = {int(slot): row for row, slot in enumerate(TARGET_SLOT_INDICES)}
_TARGET_ENTITY_CELLS = {
    OTHER_CELL_OFFSET + _target_row_of[int(ALLY_TARGET_INDICES[0])]: 0,
    OTHER_CELL_OFFSET + _target_row_of[int(ALLY_TARGET_INDICES[1])]: 1,
    OTHER_CELL_OFFSET + _target_row_of[int(ENEMY_TARGET_INDICES[0])]: 6,
    OTHER_CELL_OFFSET + _target_row_of[int(ENEMY_TARGET_INDICES[1])]: 7,
}
_IS_WILDCARD_SLOT = np.isin(MOVE_INDICES, WILDCARD_MOVE_INDICES)


def cell_identities(env, mode: str = "identity") -> np.ndarray:
    """(T, B, A*A) int64: the candidate identity behind each cell — species
    id for switch cells (battle: tgt reserve; preview: src reserve) and
    entity-backed target cells, move id for regular move cells; -1 where no
    identity applies (wildcard, pass, structural).

    mode "pair" keys every identity-carrying cell by (candidate id,
    OPPONENT ACTIVE species) instead: the label then changes when the
    opponent changes, so a lookup of the candidate alone cannot fit it —
    only RELATIONAL routing (candidate x opponent, the matchup shape the
    deployment task actually needs) generalises. Lookup labels measured
    2026-08-27: both architectures ~0.73-0.75 held-seen — lookup does not
    discriminate them."""
    priv_species = np.asarray(env.private_team)[..., _SPECIES_PRIV]
    rev_species = np.asarray(env.revealed_team)[..., _SPECIES_REV]
    move_ids = np.asarray(env.my_moveset)[..., _MOVE_ID]
    shape = priv_species.shape[:-1] + (len(_FLAT),)
    ids = np.full(shape, -1, dtype=np.int64)
    # Switch cell j IS reserve j in the block space -- one write serves the
    # battle switch and the team-preview lead alike.
    ids[..., :NUM_SWITCH_CELLS] = priv_species
    for move_row in range(NUM_MOVE_SLOTS):
        if _IS_WILDCARD_SLOT[move_row]:
            continue
        base = MOVE_CELL_OFFSET + move_row * NUM_TARGET_SLOTS
        ids[..., base : base + NUM_TARGET_SLOTS] = move_ids[
            ..., move_row : move_row + 1
        ]
    for cell, rev_row in _TARGET_ENTITY_CELLS.items():
        ids[..., cell] = rev_species[..., rev_row]
    if mode == "pair":
        # Opponent side is revealed rows 6-11, actives first — row 6 is
        # the opponent's active. 8192 clears every id enum's range.
        opp_active = rev_species[..., 6:7]
        ids = np.where(ids >= 0, ids * 8192 + opp_active, -1)
    return ids


def hash_normal(ids: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic standard-normal value per identity (Box–Muller over
    two integer hashes) — a fixed, input-representable function of the id."""
    ids64 = ids.astype(np.uint64)
    mod = np.uint64(2**32)
    h1 = (ids64 * np.uint64(2654435761) + np.uint64(seed)) % mod
    h2 = (ids64 * np.uint64(40503) + np.uint64(seed + 1)) % mod
    u1 = (h1.astype(np.float64) + 0.5) / 2**32
    u2 = (h2.astype(np.float64) + 0.5) / 2**32
    return (np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)).astype(np.float32)


# The three eval groups, in report order.
GROUPS = (
    ("switch", _SWITCH_CELLS),
    ("move", _MOVE_CELLS),
    ("target", np.isin(np.arange(len(_FLAT)), list(_TARGET_ENTITY_CELLS))),
)


def build_labels(env, seed: int, label_scale: float, mode: str = "identity"):
    """Per-cell labels y and the eval mask: identity-keyed z, centred
    WITHIN each group over the row's legal identity-carrying cells; rows
    contribute a group only when it has >= 2 such cells."""
    flat_mask = np.asarray(env.action_mask)
    ids = cell_identities(env, mode)
    z = hash_normal(ids, seed)
    y = np.zeros_like(z, dtype=np.float32)
    eval_mask = np.zeros_like(flat_mask, dtype=bool)
    rows = trainable_rows(np.asarray(env.done))
    for _, group_cells in GROUPS:
        cells = flat_mask & group_cells & (ids >= 0) & rows[..., None]
        count = cells.sum(-1, keepdims=True)
        cells &= count >= 2
        mean = np.where(cells, z, 0.0).sum(-1, keepdims=True) / np.maximum(count, 1)
        y = np.where(cells, (z - mean) * label_scale, y)
        eval_mask |= cells
    return jnp.asarray(y), jnp.asarray(eval_mask), ids


def make_apply(net):
    return jax.jit(jax.vmap(net.apply, in_axes=(None, 1, 1, None), out_axes=1))


def actor_input_of(batch):
    pt = batch.player_transitions
    return PlayerActorInput(
        env=pt.env_output,
        packed_history=batch.player_packed_history,
        history=batch.player_history,
    )


def group_masks(eval_mask: np.ndarray, ids: np.ndarray, seen_ids=None):
    """Per-group eval-cell masks; with `seen_ids` (a {group: id-set}),
    restrict to cells whose identity appeared on the TRAINING batch's eval
    cells of the same group. Identity spaces are per-group (species vs move
    ids share integers), so the restriction must be per-group too.

    Why the restriction exists (measured 2026-08-27): held-out labels are
    hashes of identity, so a species never seen in training is UNLEARNABLE
    by any architecture — and randombattle teams barely overlap across
    games (seen-frac 0.267 for switch cells vs 0.793 for move cells on the
    12-game cache), so unrestricted held-out r is overlap-capped and reads
    as an architecture gap that is actually a data artefact."""
    out = {}
    for name, group_cells in GROUPS:
        cells = eval_mask & group_cells
        if seen_ids is not None:
            cells = cells & np.isin(ids, sorted(seen_ids[name]))
        out[name] = cells
    return out


def seen_id_sets(eval_mask: np.ndarray, ids: np.ndarray):
    return {
        name: set(np.unique(ids[eval_mask & group_cells]))
        for name, group_cells in GROUPS
    }


def group_stats(adv: np.ndarray, y: np.ndarray, masks: dict[str, np.ndarray]):
    """Per group: mean within-row Pearson r and pred-std/label-std over
    rows with >= 3 eval cells."""
    out = {}
    for name, cells in masks.items():
        report_rs, report_ratios = [], []
        t_idx, b_idx = np.nonzero(cells.sum(-1) >= 3)
        for t, b in zip(t_idx, b_idx):
            row = cells[t, b]
            a, lab = adv[t, b][row], y[t, b][row]
            label_std = lab.std()
            if label_std < 1e-8:
                continue
            pred_std = a.std()
            report_ratios.append(pred_std / label_std)
            if pred_std > 1e-8:
                report_rs.append(np.corrcoef(a, lab)[0, 1])
            else:
                report_rs.append(0.0)
        if report_rs:
            out[name] = (
                len(report_rs),
                float(np.mean(report_rs)),
                float(np.mean(report_ratios)),
            )
        else:
            out[name] = (0, float("nan"), float("nan"))
    return out


def run_probe_b(
    net,
    variables,
    batch,
    held,
    steps,
    lr,
    eval_every,
    seed,
    label_scale,
    mode,
    split="batch",
):
    """The gate metric is the HELD-OUT read. On a fixed training batch the
    species<->slot assignment is frozen per row, so a full-param overfit can
    memorise per-row-per-slot values through state features without ever
    routing species -> cell — measured 2026-08-27: the shared-stream
    architecture hits train r = 1.000 by step 100. Identity-keyed labels
    generalise to UNSEEN states only through genuine candidate routing
    (same species -> same z everywhere), so held-out r is the capacity
    reading and train r is only the trainability sanity check."""
    apply = make_apply(net)
    actor_input = actor_input_of(batch)
    actor_output = batch.player_transitions.agent_output.actor_output
    y, eval_mask, train_ids = build_labels(
        batch.player_transitions.env_output, seed, label_scale, mode
    )
    held_input = actor_input_of(held)
    held_output = held.player_transitions.agent_output.actor_output
    held_y, held_eval, held_ids = build_labels(
        held.player_transitions.env_output, seed, label_scale, mode
    )
    if split == "rows":
        # Row-level split: hold out a random half of the TRAIN batch's own
        # rows instead of a disjoint game set. (species x opponent-active)
        # pairs essentially never recur across different games' team
        # draws, so a game-level split leaves the pair mode with no
        # scoreable held-seen cells at all; within a game the same matchup
        # persists across evolving states (hp, field, reveals), which is
        # the weakest generalisation that still requires reading the pair.
        # Caveat on record: temporally-adjacent states are similar, so
        # this bar is easier than cross-game — read it as a lower bound.
        row_rng = np.random.default_rng(seed + 7919)
        train_rows = (row_rng.random(np.asarray(eval_mask).shape[:2]) < 0.5)[..., None]
        eval_np = np.asarray(eval_mask)
        train_eval = eval_np & train_rows
        held_eval_np = eval_np & ~train_rows
        train_masks = group_masks(train_eval, train_ids)
        seen = seen_id_sets(train_eval, train_ids)
        held_masks = group_masks(held_eval_np, train_ids)
        held_seen_masks = group_masks(held_eval_np, train_ids, seen_ids=seen)
        held_input, held_output, held_y = actor_input, actor_output, y
        weight = jnp.asarray(train_eval).astype(jnp.float32)
    else:
        train_masks = group_masks(np.asarray(eval_mask), train_ids)
        seen = seen_id_sets(np.asarray(eval_mask), train_ids)
        held_masks = group_masks(np.asarray(held_eval), held_ids)
        held_seen_masks = group_masks(np.asarray(held_eval), held_ids, seen_ids=seen)
        weight = eval_mask.astype(jnp.float32)
    denom = jnp.maximum(weight.sum(), 1.0)

    def loss_fn(params):
        pred = apply(params, actor_input, actor_output, HeadParams())
        adv = pred.action_head.log_policy.astype(jnp.float32)
        return jnp.sum(weight * (adv - y) ** 2) / denom

    opt = optax.adam(lr)
    opt_state = opt.init(variables)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    def evaluate(params, step_no, loss):
        train_pred = apply(params, actor_input, actor_output, HeadParams())
        held_pred = apply(params, held_input, held_output, HeadParams())
        train_adv = np.asarray(train_pred.action_head.log_policy, dtype=np.float32)
        held_adv = np.asarray(held_pred.action_head.log_policy, dtype=np.float32)
        splits = {
            "train": group_stats(train_adv, np.asarray(y), train_masks),
            "held": group_stats(held_adv, np.asarray(held_y), held_masks),
            "held-seen": group_stats(held_adv, np.asarray(held_y), held_seen_masks),
        }
        parts = [f"step {step_no:4d} loss {float(loss):.5f}"]
        for split, stats in splits.items():
            for name, (rows, pearson, ratio) in stats.items():
                parts.append(
                    f"{split}/{name}: rows={rows} r={pearson:.3f} sr={ratio:.3f}"
                )
        print("  ".join(parts), flush=True)
        return splits

    history = [evaluate(variables, 0, loss_fn(variables))]
    loss = jnp.nan
    for step_no in range(1, steps + 1):
        variables, opt_state, loss = step(variables, opt_state)
        if step_no % eval_every == 0:
            history.append(evaluate(variables, step_no, loss))
    # Pre-registered gate (plan 2026-08-27): held-out switch-group r >= 0.9
    # AND std ratio >= 0.8 at any eval. Printed, not enforced — the
    # decision table (baseline fails / new arch passes -> launch) lives
    # with the plan; this line just makes the reading unambiguous.
    # Pre-registered gate, corrected 2026-08-27 to the SEEN-identity slice:
    # held-out switch-group r >= 0.9 AND std ratio >= 0.8 at any eval,
    # scored only on cells whose species appeared in training (unseen
    # hashes are unlearnable by any architecture — see group_masks).
    scored = [
        (s["held-seen"]["switch"][1], s["held-seen"]["switch"][2])
        for s in history
        if s["held-seen"]["switch"][0] > 0
    ]
    if not scored:
        print(
            "probe B gate: no seen-identity held switch rows — no reading", flush=True
        )
        return history
    best_r, best_ratio = max(scored, key=lambda pair: pair[0])
    if best_r >= 0.9 and best_ratio >= 0.8:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    print(
        f"probe B gate (held-seen/switch, best eval): r={best_r:.3f} "
        f"sr={best_ratio:.3f} -> {verdict}",
        flush=True,
    )
    return history


def run_probe_a(net, variables, batch, seed):
    """Species-swap routing probe: per batch column, swap the species of
    two reserves that are both legal battle-switch targets somewhere in the
    column, forward once each way, and report own-cell vs sibling |delta|."""
    apply = make_apply(net)
    env = batch.player_transitions.env_output
    actor_output = batch.player_transitions.agent_output.actor_output
    flat_mask = np.asarray(env.action_mask)
    rows = trainable_rows(np.asarray(env.done))
    rng = np.random.default_rng(seed)

    reserve_cells = [
        np.arange(len(_FLAT)) == reserve for reserve in range(NUM_SWITCH_CELLS)
    ]
    legal_per_reserve = np.stack(
        [(flat_mask & cells).any(-1) & rows for cells in reserve_cells], axis=-1
    )

    swapped_team = np.asarray(env.private_team).copy()
    swap_pairs = []
    for b in range(legal_per_reserve.shape[1]):
        counts = legal_per_reserve[:, b, :].sum(0)
        candidates = np.nonzero(counts > 0)[0]
        if len(candidates) < 2:
            swap_pairs.append(None)
            continue
        first, second = rng.choice(candidates, size=2, replace=False)
        swap_pairs.append((int(first), int(second)))
        col = swapped_team[:, b]
        col[:, [first, second], _SPECIES_PRIV] = col[:, [second, first], _SPECIES_PRIV]
    import dataclasses

    swapped_env = dataclasses.replace(env, private_team=jnp.asarray(swapped_team))

    def action_grid(env_in):
        batch_input = PlayerActorInput(
            env=env_in,
            packed_history=batch.player_packed_history,
            history=batch.player_history,
        )
        pred = apply(variables, batch_input, actor_output, HeadParams())
        return np.asarray(pred.action_head.log_policy, dtype=np.float32)

    delta = np.abs(action_grid(swapped_env) - action_grid(env))
    own_deltas, sibling_deltas, move_deltas = [], [], []
    for b, pair in enumerate(swap_pairs):
        if pair is None:
            continue
        own_cells = reserve_cells[pair[0]] | reserve_cells[pair[1]]
        for t in np.nonzero(rows[:, b])[0]:
            legal = flat_mask[t, b]
            own = legal & _SWITCH_CELLS & own_cells
            sibling = legal & _SWITCH_CELLS & ~own_cells
            if not (own.any() and sibling.any()):
                continue
            own_deltas.append(delta[t, b][own].mean())
            sibling_deltas.append(delta[t, b][sibling].mean())
            moves = legal & _MOVE_CELLS
            if moves.any():
                move_deltas.append(delta[t, b][moves].mean())
    own_mean = float(np.mean(own_deltas))
    sibling_mean = float(np.mean(sibling_deltas))
    print(
        f"probe A: rows={len(own_deltas)}  own|d|={own_mean:.5f}  "
        f"sibling|d|={sibling_mean:.5f}  "
        f"separation={own_mean / max(sibling_mean, 1e-9):.2f}x  "
        f"move-leak|d|={float(np.mean(move_deltas)):.5f}",
        flush=True,
    )


def fresh_variables(net, batch, seed):
    unbatched = jax.tree.map(lambda x: x[:, 0], actor_input_of(batch))
    unbatched_out = jax.tree.map(
        lambda x: x[:, 0], batch.player_transitions.agent_output.actor_output
    )
    return jax.jit(net.init)(
        jax.random.key(seed), unbatched, unbatched_out, HeadParams()
    )


def _ridge_r(features: np.ndarray, labels: np.ndarray, train: np.ndarray, alpha=1.0):
    """Held-out Pearson r of a ridge readout. Train fit is memorisation on a
    fixed batch (2026-08-27 probe lesson 1), so only held-out is reported."""
    held = ~train
    if train.sum() < 32 or held.sum() < 32:
        return float("nan"), int(held.sum())
    x_train = features[train]
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0) + 1e-6
    x_train = (x_train - mean) / scale
    x_held = (features[held] - mean) / scale
    y_train = labels[train] - labels[train].mean()
    gram = x_train.T @ x_train + alpha * np.eye(x_train.shape[1], dtype=np.float64)
    weights = np.linalg.solve(gram, x_train.T @ y_train)
    pred = x_held @ weights
    y_held = labels[held]
    if pred.std() < 1e-9 or y_held.std() < 1e-9:
        return 0.0, int(held.sum())
    return float(np.corrcoef(pred, y_held)[0, 1]), int(held.sum())


def _encode_fn(module, actor_input):
    sequence, _ = module.encoder(
        actor_input.env, actor_input.packed_history, actor_input.history
    )
    return sequence


def _slot_condition_rows(env, batch_index: int):
    """Per (t, b) and reserve slot j: the matched public row, that row's hp
    ratio and fainted flag, and whether switching to j is legal.

    The match is by SPECIES, which is a unique key across a team in
    gen9randombattle. Ambiguous or absent matches are dropped — the probe
    measures routing, not the matcher.
    """
    private = np.asarray(env.private_team)
    revealed = np.asarray(env.revealed_team)
    public = np.asarray(env.public_team)
    action_mask = np.asarray(env.action_mask)
    num_t = private.shape[0]

    rows = []
    for t in range(num_t):
        priv_species = private[t, batch_index, :, _SPECIES_PRIV]
        pub_species = revealed[t, batch_index, :6, _SPECIES_REV]
        equal = priv_species[:, None] == pub_species[None, :]
        matched = equal.sum(axis=1) == 1
        public_row = equal.argmax(axis=1)
        for j in range(6):
            if not matched[j]:
                continue
            row = public[t, batch_index, public_row[j]]
            rows.append(
                dict(
                    t=t,
                    j=j,
                    hp=float(row[_HP_RATIO]) / MAX_RATIO_TOKEN,
                    fainted=float(row[_FAINTED]),
                    legal=bool(action_mask[t, batch_index, j]),
                )
            )
    return rows


def run_probe_c(net, variables, chunks, batch_size: int, seed: int, alpha: float):
    """Does a switch candidate's CURRENT condition reach its RESERVE_j action
    embedding?

    `EntityPrivateNodeFeature` carries no hp / status / fainted / boosts, so
    RESERVE_j's warm start is a static set descriptor and the condition can
    only arrive through the trunk, retrieved from the candidate's PUBLIC row.
    This measures whether it does.

    Three readouts, all ridge on the same trunk, so the comparison is of
    ROUTING and not of readout capacity:

      test     RESERVE_j embedding    -> candidate j's hp / fainted
      control  ALLY_1_SWITCH, ENEMY_1_TARGET -> their OWN mon's hp / fainted
               (slots warm-started from an entity whose tokens include
               HP_RATIO, so the data is local — the positive control that
               proves the probe can read a condition at all)
      floor    RESERVE_j embedding    -> the same labels, shuffled

    Held out by CHUNK, so train and held rows come from different games. The
    labels are continuous / binary rather than identity hashes, so unlike the
    species probe a cross-game split is valid here (2026-08-27 lesson 2).
    """
    apply_encoder = jax.jit(
        jax.vmap(
            lambda variables, actor_input: net.apply(
                variables, actor_input, method=_encode_fn
            ),
            in_axes=(None, 1),
            out_axes=1,
        )
    )
    dev_variables = jax.device_put(variables)

    reserve = {"features": [], "hp": [], "fainted": [], "legal": [], "chunk": []}
    # Controls are SEQUENCE rows that carry the named entity: my active's
    # ally-target row and the opponent active's enemy-target row (the
    # entity-derived target rows). The grid era read the 41-slot action
    # stream here; the flat trunk has no such stream, so the rows are named
    # off rl/model/constants like every head does.
    controls = {
        "ally_1_target": (
            TARGET_ROWS.start + int(ALLY_TARGET_ROWS[0]),
            0,
        ),
        "enemy_1_target": (
            TARGET_ROWS.start + int(ENEMY_TARGET_ROWS[0]),
            6,
        ),
    }
    control_rows = {
        name: {"features": [], "hp": [], "fainted": [], "chunk": []}
        for name in controls
    }

    for start in range(0, len(chunks), batch_size):
        group = chunks[start : start + batch_size]
        stacked = stack_batch(group)
        actor_input = actor_input_of(stacked)
        sequence_rows_out = apply_encoder(dev_variables, actor_input)
        sequence_rows_out = np.asarray(sequence_rows_out, dtype=np.float32)
        env = stacked.player_transitions.env_output
        public = np.asarray(env.public_team)
        for b in range(len(group)):
            chunk_id = start + b
            for entry in _slot_condition_rows(env, b):
                row_index = PRIVATE_ROWS.start + entry["j"]
                reserve["features"].append(sequence_rows_out[entry["t"], b, row_index])
                reserve["hp"].append(entry["hp"])
                reserve["fainted"].append(entry["fainted"])
                reserve["legal"].append(entry["legal"])
                reserve["chunk"].append(chunk_id)
            for name, (row_index, public_row) in controls.items():
                for t in range(sequence_rows_out.shape[0]):
                    row = public[t, b, public_row]
                    control_rows[name]["features"].append(
                        sequence_rows_out[t, b, row_index]
                    )
                    control_rows[name]["hp"].append(
                        float(row[_HP_RATIO]) / MAX_RATIO_TOKEN
                    )
                    control_rows[name]["fainted"].append(float(row[_FAINTED]))
                    control_rows[name]["chunk"].append(chunk_id)

    rng = np.random.default_rng(seed)
    held_chunks = set(
        rng.choice(
            np.arange(len(chunks)), size=max(1, len(chunks) // 3), replace=False
        ).tolist()
    )

    readouts = {
        "RESERVE_j": reserve,
        "ally_1_target (ctl)": control_rows["ally_1_target"],
        "enemy_1_target (ctl)": control_rows["enemy_1_target"],
    }
    for entry in readouts.values():
        for key, value in list(entry.items()):
            entry[key] = np.asarray(value)
        entry["alive"] = entry["hp"] > 0.0

    print("\n=== probe C: does RESERVE_j carry the candidate's condition? ===")
    print(
        "Subsets are matched on the TARGET mon: an unmatched control reads the\n"
        "alive/dead contrast rather than the condition. 'legal' rows are alive by\n"
        "construction, so `fainted` has no variance there and its r is vacuous —\n"
        "read the y-std column before reading any r.\n"
    )
    header = (
        f"{'readout':<22} {'subset':<9} {'label':<8} "
        f"{'held r':>7} {'y std':>7} {'n_held':>7}"
    )
    print(header)
    print("-" * len(header))

    results = {}
    for name, entry in readouts.items():
        subsets = {
            "all": np.ones(len(entry["hp"]), dtype=bool),
            "alive": entry["alive"],
        }
        if "legal" in entry:
            subsets["legal"] = entry["legal"]
        for subset_name, subset in subsets.items():
            if subset.sum() < 128:
                continue
            train = np.array(
                [c not in held_chunks for c in entry["chunk"][subset]], dtype=bool
            )
            for label_name in ("hp", "fainted"):
                labels = entry[label_name][subset].astype(np.float64)
                value, n_held = _ridge_r(
                    entry["features"][subset].astype(np.float64), labels, train, alpha
                )
                held_std = labels[~train].std() if (~train).any() else float("nan")
                print(
                    f"{name:<22} {subset_name:<9} {label_name:<8} "
                    f"{value:>7.3f} {held_std:>7.3f} {n_held:>7}"
                )
                results[f"{name}/{subset_name}/{label_name}"] = value
        train = np.array([c not in held_chunks for c in entry["chunk"]], dtype=bool)
        shuffled = entry["hp"][rng.permutation(len(entry["hp"]))].astype(np.float64)
        value, n_held = _ridge_r(
            entry["features"].astype(np.float64), shuffled, train, alpha
        )
        print(
            f"{name:<22} {'shuffled':<9} {'hp':<8} {value:>7.3f} "
            f"{shuffled[~train].std():>7.3f} {n_held:>7}"
        )
        results[f"{name}/shuffled/hp"] = value
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-pkl", default="runtime/discrim_sides_ckpt224773.pkl")
    parser.add_argument("--ckpt", default=None, help="trained params; default fresh")
    parser.add_argument("--probe", choices=("a", "b", "c", "both"), default="both")
    parser.add_argument(
        "--ridge-alpha", type=float, default=1.0, help="probe C ridge penalty"
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-scale", type=float, default=0.2)
    parser.add_argument(
        "--label-mode", choices=("identity", "pair"), default="identity"
    )
    parser.add_argument("--split", choices=("batch", "rows"), default="batch")
    parser.add_argument("--out", default=None, help="pickle the eval history here")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    chunks = harness.flatten(harness.load(args.games_pkl))
    (batch,), held = pick_batches(chunks, args.batch, pool=1)
    net = get_player_model(get_player_model_config(9, train=True))
    if args.ckpt:
        variables = harness.load_params(args.ckpt)
        opened = variables
        source = args.ckpt
    else:
        variables = fresh_variables(net, batch, args.seed)
        opened = open_zero_init_paths(variables, ("action_head",), seed=args.seed)
        source = "fresh init (action head opened for probe A)"
    print(f"params: {source}", flush=True)

    if args.probe == "c":
        run_probe_c(net, variables, chunks, args.batch, args.seed, args.ridge_alpha)
        return

    if args.probe in ("a", "both"):
        run_probe_a(net, opened, batch, args.seed)
    history = None
    if args.probe in ("b", "both"):
        history = run_probe_b(
            net,
            variables,
            batch,
            held,
            args.steps,
            args.lr,
            args.eval_every,
            args.seed,
            args.label_scale,
            args.label_mode,
            args.split,
        )
    if args.out and history is not None:
        with open(args.out, "wb") as handle:
            pickle.dump(history, handle)


if __name__ == "__main__":
    main()
