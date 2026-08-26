"""Within-modality separation probe — the pre-launch capacity gate.

Two probes over one fixed batch of real self-play states, both answering
"can this ARCHITECTURE tell a modality's candidates apart?" before any
training time is spent (docs: the 2026-08-27 measurement found within-row
switch Q std 0.0196 vs move 0.0374 vs between-row 0.507 on the live
checkpoint — the collapse's representational reading).

Probe A — routing. Swap the SPECIES feature of two legal reserves in one
batch column (species -> embedding is computed at forward time, legality
untouched, history caches join by index not species) and compare per-cell
|delta advantage|: the swapped reserves' own cells vs sibling switch cells.
A shared pooled representation moves siblings as much as the perturbed
cell; genuine per-candidate routing separates them. Read at fresh init with
the advantage head's zero-init paths OPENED (else the composition is
exactly 0 and everything is vacuous), and on trained params as-is.

Probe B — overfit separation (THE GATE). Synthetic per-cell labels keyed to
the candidate's IDENTITY (species id for switch/entity-target cells, move
id for move cells) hashed to a standard normal, then CENTRED within each
modality over the row's legal cells and scaled by --label-scale. Centring
removes all state-level and between-modality signal, and identity keying
removes the positional shortcut (a per-cell bias fits position-keyed
labels with zero input reading) — only within-modality discrimination of
the actual candidates can fit these. Direct f32 MSE on out.advantage over
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

from rl.environment.data import (
    ALLY_TARGET_INDICES,
    ENEMY_TARGET_INDICES,
    FLAT_MODALITY_MASK,
    MOVE_SLOT_INDICES,
    NUM_ACTION_FEATURES,
    RESERVE_ENTITY_INDICES,
)
from rl.environment.interfaces import PlayerActorInput
from rl.environment.protos.features_pb2 import (
    EntityPrivateNodeFeature,
    EntityRevealedNodeFeature,
    MovesetFeature,
)
from rl.environment.protos.service_pb2 import ModalityEnum
from rl.model.config import get_player_model_config
from rl.model.heads import HeadParams
from rl.model.player_model import get_player_model
from rl.model.utils import open_zero_init_paths
from rl.offline import harness
from rl.offline.overfit_probe import pick_batches, trainable_rows

logger = logging.getLogger(__name__)

_A = NUM_ACTION_FEATURES
_CELL_SRC = np.arange(_A * _A) // _A
_CELL_TGT = np.arange(_A * _A) % _A
_FLAT = np.asarray(FLAT_MODALITY_MASK)
_SWITCH_CELLS = _FLAT == ModalityEnum.MODALITY_ENUM__SWITCH
_MOVE_CELLS = _FLAT == ModalityEnum.MODALITY_ENUM__MOVE
_SPECIES_PRIV = EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SPECIES
_SPECIES_REV = EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
_MOVE_ID = MovesetFeature.MOVESET_FEATURE__MOVE_ID
# Entity-backed target cells: ally targets key revealed rows 0/1, enemy
# targets revealed rows 6/7; the learned TARGET_*/PASS slots carry no
# entity and are excluded from identity labels.
_TARGET_ENTITY_SLOTS = {
    int(ALLY_TARGET_INDICES[0]): 0,
    int(ALLY_TARGET_INDICES[1]): 1,
    int(ENEMY_TARGET_INDICES[0]): 6,
    int(ENEMY_TARGET_INDICES[1]): 7,
}
_OTHER = ModalityEnum.MODALITY_ENUM__OTHER


def cell_identities(env) -> np.ndarray:
    """(T, B, A*A) int64: the candidate identity behind each cell — species
    id for switch cells (battle: tgt reserve; preview: src reserve) and
    entity-backed target cells, move id for regular move cells; -1 where no
    identity applies (wildcard, pass, structural)."""
    priv_species = np.asarray(env.private_team)[..., _SPECIES_PRIV]
    rev_species = np.asarray(env.revealed_team)[..., _SPECIES_REV]
    move_ids = np.asarray(env.my_moveset)[..., _MOVE_ID]
    shape = priv_species.shape[:-1] + (_A * _A,)
    ids = np.full(shape, -1, dtype=np.int64)
    # Preview switch cells key the reserve as SRC; battle switch cells key
    # it as TGT and take precedence where both apply.
    for reserve, slot in enumerate(RESERVE_ENTITY_INDICES):
        cells = _SWITCH_CELLS & (_CELL_SRC == slot)
        ids[..., cells] = priv_species[..., reserve : reserve + 1]
    for reserve, slot in enumerate(RESERVE_ENTITY_INDICES):
        cells = _SWITCH_CELLS & (_CELL_TGT == slot)
        ids[..., cells] = priv_species[..., reserve : reserve + 1]
    for row, slot in enumerate(MOVE_SLOT_INDICES):
        cells = _MOVE_CELLS & (_CELL_SRC == slot)
        if cells.any():
            ids[..., cells] = move_ids[..., row : row + 1]
    for slot, rev_row in _TARGET_ENTITY_SLOTS.items():
        cells = (_FLAT == _OTHER) & (_CELL_TGT == slot)
        ids[..., cells] = rev_species[..., rev_row : rev_row + 1]
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
    ("target", (_FLAT == _OTHER) & np.isin(_CELL_TGT, list(_TARGET_ENTITY_SLOTS))),
)


def build_labels(env, seed: int, label_scale: float):
    """Per-cell labels y and the eval mask: identity-keyed z, centred
    WITHIN each group over the row's legal identity-carrying cells; rows
    contribute a group only when it has >= 2 such cells."""
    flat_mask = np.asarray(env.action_mask).reshape(*env.done.shape, -1)
    ids = cell_identities(env)
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
    return jnp.asarray(y), jnp.asarray(eval_mask)


def make_apply(net):
    return jax.jit(jax.vmap(net.apply, in_axes=(None, 1, 1, None), out_axes=1))


def actor_input_of(batch):
    pt = batch.player_transitions
    return PlayerActorInput(
        env=pt.env_output,
        packed_history=batch.player_packed_history,
        history=batch.player_history,
    )


def group_stats(adv: np.ndarray, y: np.ndarray, eval_mask: np.ndarray):
    """Per group: mean within-row Pearson r and pred-std/label-std over
    rows with >= 3 eval cells."""
    out = {}
    for name, group_cells in GROUPS:
        cells = eval_mask & group_cells
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


def run_probe_b(net, variables, batch, held, steps, lr, eval_every, seed, label_scale):
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
    y, eval_mask = build_labels(batch.player_transitions.env_output, seed, label_scale)
    held_input = actor_input_of(held)
    held_output = held.player_transitions.agent_output.actor_output
    held_y, held_eval = build_labels(
        held.player_transitions.env_output, seed, label_scale
    )
    weight = eval_mask.astype(jnp.float32)
    denom = jnp.maximum(weight.sum(), 1.0)

    def loss_fn(params):
        pred = apply(params, actor_input, actor_output, HeadParams())
        adv = pred.advantage.astype(jnp.float32)
        return jnp.sum(weight * (adv - y) ** 2) / denom

    opt = optax.adam(lr)
    opt_state = opt.init(variables)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    def evaluate(params, step_no, loss):
        splits = {}
        for split, (inp, outp, labels, mask) in {
            "train": (actor_input, actor_output, y, eval_mask),
            "held": (held_input, held_output, held_y, held_eval),
        }.items():
            pred = apply(params, inp, outp, HeadParams())
            splits[split] = group_stats(
                np.asarray(pred.advantage, dtype=np.float32),
                np.asarray(labels),
                np.asarray(mask),
            )
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
    scored = [
        (s["held"]["switch"][1], s["held"]["switch"][2])
        for s in history
        if s["held"]["switch"][0] > 0
    ]
    if not scored:
        print("probe B gate: no held-out switch rows — no reading", flush=True)
        return history
    best_r, best_ratio = max(scored, key=lambda pair: pair[0])
    if best_r >= 0.9 and best_ratio >= 0.8:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    print(
        f"probe B gate (held/switch, best eval): r={best_r:.3f} "
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
    flat_mask = np.asarray(env.action_mask).reshape(*env.done.shape, -1)
    rows = trainable_rows(np.asarray(env.done))
    rng = np.random.default_rng(seed)

    reserve_cells = [
        _SWITCH_CELLS & (_CELL_TGT == slot) for slot in RESERVE_ENTITY_INDICES
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

    def advantage(env_in):
        batch_input = PlayerActorInput(
            env=env_in,
            packed_history=batch.player_packed_history,
            history=batch.player_history,
        )
        pred = apply(variables, batch_input, actor_output, HeadParams())
        return np.asarray(pred.advantage, dtype=np.float32)

    delta = np.abs(advantage(swapped_env) - advantage(env))
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-pkl", default="runtime/discrim_sides_ckpt224773.pkl")
    parser.add_argument("--ckpt", default=None, help="trained params; default fresh")
    parser.add_argument("--probe", choices=("a", "b", "both"), default="both")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-scale", type=float, default=0.2)
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
        opened = open_zero_init_paths(variables, ("advantage_head",), seed=args.seed)
        source = "fresh init (advantage head opened for probe A)"
    print(f"params: {source}", flush=True)

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
        )
    if args.out and history is not None:
        with open(args.out, "wb") as handle:
            pickle.dump(history, handle)


if __name__ == "__main__":
    main()
