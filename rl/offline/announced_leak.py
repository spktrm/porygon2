"""Leak check for the announced-state outcome mask (Φ_ann).

Φ_ann for a turn must be a function of the pre-turn history plus the turn's
ANNOUNCEMENTS only — never of the turn's resolved chance. This checks the
whole pathway end to end, in the spirit of rl/offline/causality.py: export a
replay through the real exporter, then perturb the OUTCOME features of one
turn's packed-cache rows — flip a bit in the MINOR_ARG bitmasks (the crit
bit lives there), rescale DAMAGE_RATIO / HEAL_RATIO, and dent the post-event
node snapshots' HP_RATIO — and compare per-member outputs:

- Φ_ann at that turn's own step(s) must be BIT-INVARIANT (the masked
  pathway replaces every outcome column with a constant and reads node
  snapshots at pre-turn values, so the perturbed values never enter the
  program). Any nonzero Δ = a column escaped the mask.
- Φ_ann and Φ at earlier steps must be bit-invariant too (plain causality:
  their gathers stop before the perturbed rows).
- Φ at the perturbed turn's step SHOULD move (sanity: the perturbation is
  real and visible to the unmasked pathway). Later steps of both may move
  legitimately — outcomes of past turns are real history.

Invariance alone is one-sided: a Φ_ann that never reads the turn at all (a
plain copy of the pre-turn state) passes it perfectly. So a second pass
perturbs the ANNOUNCEMENT columns of the same rows — real move tokens
bumped to a different id, other announcement major args relabelled — and
requires Φ_ann at that turn to CHANGE. Zero Δ there means announcements
never reach the announced state (mask too aggressive, request-count
stamping misaligned, or wiring), which would silently gut the decision
term of the skill/luck decomposition.

Usage:
    python -m rl.offline.announced_leak <replay> [--ckpt ...] \
        [--fractions 0.25,0.5,0.75] [--steps 12,30]

<replay> accepts the same forms as rl.offline.visualise (path / id / URL).
Works on any checkpoint — Φ_ann adds no parameters, and mask invariance is
architectural, independent of whether the artifact was trained at announced
points.
"""

import argparse
import functools
import tempfile

import jax
import numpy as np

from rl.environment.protos.enums_pb2 import BattlemajorargsEnum
from rl.environment.protos.features_pb2 import (
    EntityEdgeFeature,
    EntityPublicNodeFeature,
    FieldFeature,
    InfoFeature,
)
from rl.model.history_encoder import _OUTCOME_MAJOR_ARGS, _RELEVANT_ENTITY_FEATURES
from rl.offline.artifact import load_critic_params
from rl.offline.config import get_offline_config
from rl.offline.dataset import _INVALID_MOVE_IDS, collate, record_to_examples
from rl.offline.model import Porygon2OfflineCritic, get_offline_critic
from rl.offline.visualise import (
    _format_generation,
    discover_ckpts,
    export_record,
    resolve_replay,
)


def turn_cache_rows(history_field: np.ndarray, request_count: int) -> np.ndarray:
    """Packed-cache row indices referenced by the history steps of one turn
    (the steps stamped with ``request_count``). history_field: (H, F)."""
    valid = history_field[:, FieldFeature.FIELD_FEATURE__VALID] > 0
    in_turn = valid & (
        history_field[:, FieldFeature.FIELD_FEATURE__REQUEST_COUNT] == request_count
    )
    rows = []
    for step in np.nonzero(in_turn)[0]:
        num = int(history_field[step, FieldFeature.FIELD_FEATURE__NUM_RELEVANT])
        rows.extend(history_field[step, _RELEVANT_ENTITY_FEATURES[:num]].tolist())
    return np.unique(np.asarray(rows, dtype=np.int64))


def perturb_batch(actor_input, example_idx: int, rows: np.ndarray):
    """Returns a copy of the batched actor input with the outcome features
    of ``rows`` (one perspective's packed caches) perturbed: crit-adjacent
    MINOR_ARG bits flipped, damage/heal ratios rescaled, and the post-event
    node snapshots' hp dented. Announcement columns (MAJOR_ARG, MOVE_TOKEN,
    ENTITY_IDX) are untouched, so the announced pathway sees an identical
    program input."""
    edge_cache = np.asarray(actor_input.packed_history.edge_cache).copy()
    public_cache = np.asarray(actor_input.packed_history.public_cache).copy()

    minor0 = EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG0
    damage = EntityEdgeFeature.ENTITY_EDGE_FEATURE__DAMAGE_RATIO
    heal = EntityEdgeFeature.ENTITY_EDGE_FEATURE__HEAL_RATIO
    hp = EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO

    edge_cache[rows, example_idx, minor0] ^= 1  # flip a minor-arg bit
    old_damage = edge_cache[rows, example_idx, damage]
    edge_cache[rows, example_idx, damage] = np.where(
        old_damage > 0, old_damage // 2, 300
    )
    old_heal = edge_cache[rows, example_idx, heal]
    edge_cache[rows, example_idx, heal] = np.where(old_heal > 0, old_heal // 2, 150)
    public_cache[rows, example_idx, hp] //= 2

    return actor_input.replace(
        packed_history=actor_input.packed_history.replace(
            edge_cache=edge_cache, public_cache=public_cache
        )
    )


def announcement_rows(edge_cache: np.ndarray, example_idx: int, rows: np.ndarray):
    """Subset of ``rows`` that are announcement edges (mask_outcome_features
    semantics: a real major arg, not one of the outcome majors)."""
    major = edge_cache[
        rows, example_idx, EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG
    ]
    keep = (major != BattlemajorargsEnum.BATTLEMAJORARGS_ENUM___UNSPECIFIED) & ~np.isin(
        major, _OUTCOME_MAJOR_ARGS
    )
    return rows[keep]


def perturb_announcements(actor_input, example_idx: int, rows: np.ndarray):
    """Returns a copy with the ANNOUNCEMENT content of ``rows`` changed —
    the sensitivity twin of perturb_batch. Real move tokens are bumped to a
    different id (a different one-hot either way, even if the new id is a
    sentinel); rows without a real move token get their major arg swapped
    between switch and move. Φ_ann must respond: these columns are exactly
    what the announced state is allowed to read."""
    edge_cache = np.asarray(actor_input.packed_history.edge_cache).copy()
    move_col = EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN
    major_col = EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG

    tokens = edge_cache[rows, example_idx, move_col]
    real = ~np.isin(tokens, _INVALID_MOVE_IDS)
    edge_cache[rows[real], example_idx, move_col] = tokens[real] + 1
    swap = rows[~real]
    majors = edge_cache[swap, example_idx, major_col]
    edge_cache[swap, example_idx, major_col] = np.where(
        majors == BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__SWITCH,
        BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__MOVE,
        BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__SWITCH,
    )
    return actor_input.replace(
        packed_history=actor_input.packed_history.replace(edge_cache=edge_cache)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", help="replay JSON path, replay id, or replay URL")
    parser.add_argument("--ckpt", action="append", default=None)
    parser.add_argument(
        "--fractions",
        default="0.25,0.5,0.75",
        help="perturbed turns as fractions of the trajectory's step count",
    )
    parser.add_argument(
        "--steps", default=None, help="explicit trajectory step indices, e.g. 12,30"
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        replay, replay_json_path = resolve_replay(args.replay, tmpdir)
        payload, stats = export_record(replay_json_path, tmpdir)

    format_id = replay.get("formatid", "gen9randombattle")
    ckpt_paths = args.ckpt or discover_ckpts(format_id)
    print(f"replay: {replay.get('id')}")
    print(f"checkpoints: {ckpt_paths}")

    config = get_offline_config().replace(generation=_format_generation(format_id))
    params = load_critic_params(ckpt_paths)
    num_members = jax.tree.leaves(params)[0].shape[0]
    model = get_offline_critic(
        config.generation,
        rating_conditioning="rating_embed" in params.get("params", {}),
    )
    apply_fn = jax.jit(
        jax.vmap(
            functools.partial(model.apply, method=Porygon2OfflineCritic.announced),
            in_axes=(None, 1),
            out_axes=1,
        )
    )

    examples = record_to_examples(payload)
    batch = collate(examples, config)
    perspectives = stats["perspectives"]
    anchor = perspectives.index(0) if 0 in perspectives else 0

    def run(actor_input):
        """(K, T) realised and announced Φ for the anchor perspective."""
        phi, ann = [], []
        for k in range(num_members):
            member_params = jax.tree.map(lambda x: x[k], params)  # noqa: B023
            head, ann_head = jax.device_get(apply_fn(member_params, actor_input))
            phi.append(np.asarray(head.expectation, np.float32)[:, anchor])
            ann.append(np.asarray(ann_head.expectation, np.float32)[:, anchor])
        return np.stack(phi), np.stack(ann)

    done = np.asarray(batch.actor_input.env.done[:, anchor]).astype(np.int32)
    num_valid = int(((np.cumsum(done) - done) == 0).sum())
    info = np.asarray(batch.actor_input.env.info[:, anchor])
    request_counts = info[:, InfoFeature.INFO_FEATURE__REQUEST_COUNT]
    history_field = np.asarray(batch.actor_input.history.field[:, anchor])

    if args.steps:
        target_steps = [int(s) for s in args.steps.split(",")]
    else:
        target_steps = [
            round(float(f) * (num_valid - 1)) for f in args.fractions.split(",")
        ]
    # Step 0's "turn" is the initial switch-ins (little to perturb);
    # num_valid - 1 is the terminal state.
    target_steps = sorted({min(max(s, 1), num_valid - 1) for s in target_steps})

    phi_base, ann_base = run(batch.actor_input)

    edge_cache_np = np.asarray(batch.actor_input.packed_history.edge_cache)
    print(
        f"\n{'step':>6} {'rows':>5} {'annRows':>8} {'outcome Δ':>11} "
        f"{'Δ earlier':>10} {'Φ Δ (sanity)':>13} {'announce Δ':>11}"
    )
    worst_leak, sanity_ok, sensitivity_ok = 0.0, True, True
    for step in target_steps:
        request_count = int(request_counts[step])
        rows = turn_cache_rows(history_field, request_count)
        if rows.size == 0:
            print(f"{step:>6} {0:>5}  no history steps for this turn — skipped")
            continue
        at_turn = (request_counts[:num_valid] == request_count).nonzero()[0]
        earlier = (request_counts[:num_valid] < request_count).nonzero()[0]

        # Pass 1 — outcome invariance: perturb resolved-chance features.
        perturbed = perturb_batch(batch.actor_input, anchor, rows)
        phi_pert, ann_pert = run(perturbed)
        ann_delta = np.abs(ann_pert[:, at_turn] - ann_base[:, at_turn]).max()
        earlier_delta = 0.0
        if earlier.size:
            earlier_delta = max(
                np.abs(ann_pert[:, earlier] - ann_base[:, earlier]).max(),
                np.abs(phi_pert[:, earlier] - phi_base[:, earlier]).max(),
            )
        phi_delta = np.abs(phi_pert[:, at_turn] - phi_base[:, at_turn]).max()
        worst_leak = max(worst_leak, float(ann_delta), float(earlier_delta))
        sanity_ok = sanity_ok and phi_delta > 0

        # Pass 2 — announcement sensitivity: perturb what was announced.
        ann_rows = announcement_rows(edge_cache_np, anchor, rows)
        if ann_rows.size:
            sens_input = perturb_announcements(batch.actor_input, anchor, ann_rows)
            _, ann_sens = run(sens_input)
            sens_delta = float(
                np.abs(ann_sens[:, at_turn] - ann_base[:, at_turn]).max()
            )
            sensitivity_ok = sensitivity_ok and sens_delta > 0
            sens_text = f"{sens_delta:>11.2e}"
        else:
            sens_text = f"{'no ann':>11}"
        print(
            f"{step:>6} {rows.size:>5} {ann_rows.size:>8} {ann_delta:>11.2e} "
            f"{earlier_delta:>10.2e} {phi_delta:>13.2e} {sens_text}"
        )

    print()
    if worst_leak == 0.0:
        print(
            "PASS (invariance) — Φ_ann is bit-invariant to the perturbed "
            "outcome features of its own turn (and earlier states are "
            "untouched)."
        )
    else:
        print(
            f"FAIL (invariance) — max Φ_ann Δ {worst_leak:.2e}: an outcome "
            "column reaches the announced state. Prime suspects: a column "
            "missing from the mask in rl/model/history_encoder.py, "
            "post-event node snapshots entering the announced messages, or "
            "the turn's own field rows being fed."
        )
    if sensitivity_ok:
        print(
            "PASS (sensitivity) — Φ_ann responds to changed announcements on "
            "every tested turn that had any, so the announced pathway is "
            "actually reading the turn (invariance alone can't tell a "
            "correct mask from a Φ_ann that never reads the turn at all)."
        )
    else:
        print(
            "FAIL (sensitivity) — Φ_ann did not move when a turn's "
            "announcements were changed: announcements never reach the "
            "announced state. Prime suspects: request-count stamping "
            "misaligned with exported states (the turn-selection einsum "
            "matches no steps), row_is_announcement rejecting real edges, "
            "or the announced advance's valid gate never opening."
        )
    if not sanity_ok:
        print(
            "WARNING — the realised Φ did not move for at least one perturbed "
            "turn: the perturbation may not have touched anything that turn "
            "actually read (try other steps). Invariance above proves nothing "
            "for such turns."
        )


if __name__ == "__main__":
    main()
