"""Probe E -- type effectiveness: does the trunk compute it, and does the
policy act on it?

Motivated by the 2026-09-03 behavioural comparison (ckpt_00240000 vs 2000
human >=1900 replays): the model lands twice the human rate of IMMUNE hits
(0.047-0.051 vs 0.024) and fewer supereffective ones (0.175 vs 0.195). Types ARE on
the wire: `species.npy` / `moves.npy` are multi-hot attribute tables with a
dedicated column per type (verified 2026-09-03), so each operand row carries
its type bits pre-trunk, and the enemy target row the readout multiplies is
built by ADDING the opp active's public row (`_assemble_sequence`). "The state
has it" is therefore literal; the question is whether the trunk and the
bilinear readout COMPUTE the matchup from the two rows. Two halves:

* REPRESENTATION. Ridge readouts, held out by chunk, of the effectiveness
  class {immune, resisted, neutral, supereffective} of each legal damaging
  move against the opponent's active, from the move's own row PRE-trunk (the
  marginal reference: a move row alone knows its type, not the matchup) and
  POST-trunk (the test). Positive controls: the move's TYPE from the post
  move row and the opponent's PRIMARY TYPE from the post opp-active row --
  each operand on its own must be legible or the matchup read is vacuous.
  Shuffled-label floor. A random-projected bilinear over (move row, target
  row) is reported as an accessibility bound only: a bilinear probe can learn
  the type chart by itself, so it is not evidence about the trunk.

* BEHAVIOUR. The policy mass (train=True heads re-run over the stored rows, temp 1.0) per effectiveness
  class, conditional on the legal damaging set holding a BETTER alternative:
  mass on immune moves when a non-immune damaging move is legal, on resisted
  when a neutral-or-better one is, and on the best class overall -- each
  against the uniform-over-legal-damaging share a policy blind to the
  matchup would put there. Higher on the best class / lower on the bad ones
  is good; parity with uniform says the readout does not use the matchup.

Labels: opponent types = tera type if revealed (non-UNK, not stellar), else
the TYPECHANGE bitmask when set, else the species' base types; the revealed
ability's blanket immunities (Levitate etc.) applied when known. Status moves
are not labelled. Ability immunities from an UNREVEALED ability are label
noise, not modelled.

    env/bin/python -m rl.offline.type_probe --games-pkl X.pkl --ckpt ckpts/...
"""

from __future__ import annotations

import argparse
import json
import logging

import jax
import numpy as np

from rl.environment.data import TARGET_SLOT_INDICES, WILDCARD_MOVE_INDICES
from rl.environment.protos.features_pb2 import (
    EntityPublicNodeFeature,
    EntityRevealedNodeFeature,
    MovesetFeature,
)
from rl.model.config import get_player_model_config
from rl.model.constants import (
    _BANK_MOVE_OFFSET,
    CELL_BANK_SRC,
    MOVE_ROWS,
    OPP_ACTIVE_PUBLIC_ROWS,
    PUBLIC_ROWS,
    TARGET_ROWS,
)
from rl.model.heads import HeadParams
from rl.model.player_model import get_player_model
from rl.offline import harness
from rl.offline.separation_probe import (
    _assembled_and_encoded_fn,
    _ridge_accuracy,
    _ridge_r,
    actor_input_of,
    make_apply,
)
from rl.online.training.batching import stack_batch

logger = logging.getLogger(__name__)

CLASS_NAMES = ("immune", "resisted", "neutral", "supereffective")
IMMUNE, RESISTED, NEUTRAL, SUPEREFFECTIVE = range(4)
NUM_REGULAR_MOVES = 4
NUM_TARGETS = len(TARGET_SLOT_INDICES)
# Regular slot k's wildcard (tera) shadow is move row k + 4 (MOVE_INDICES order).
WILDCARD_ROW_OFFSET = len(WILDCARD_MOVE_INDICES) // 2

_SPECIES = EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
_ABILITY = EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ABILITY
_TERA = EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__TERA_TYPE
_TYPECHANGE = (
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE0,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE1,
)
_MOVE_ID = MovesetFeature.MOVESET_FEATURE__MOVE_ID
_OPP_ROW = int(OPP_ACTIVE_PUBLIC_ROWS[0])
# Move cells are row-major (slot, target) from the first move cell.
MOVE_CELL_START = int(np.nonzero(CELL_BANK_SRC == _BANK_MOVE_OFFSET)[0][0])

# Blanket type immunities granted by an ability (gen 9, revealed only).
_ABILITY_IMMUNITY = {
    "levitate": "Ground",
    "eartheater": "Ground",
    "flashfire": "Fire",
    "wellbakedbody": "Fire",
    "voltabsorb": "Electric",
    "lightningrod": "Electric",
    "motordrive": "Electric",
    "waterabsorb": "Water",
    "stormdrain": "Water",
    "dryskin": "Water",
    "sapsipper": "Grass",
    "windrider": "Flying",
}


class TypeTables:
    """Enum index -> dex facts, from the same data the encoder's tables are
    built from, plus the gen9 chart (`chart[atk][def]` in {-9 immune, -1, 0, 1})."""

    def __init__(self, data_dir: str, chart_path: str):
        with open(f"{data_dir}/data.json") as handle:
            enums = json.load(handle)
        with open(f"{data_dir}/gen9/moves.json") as handle:
            moves = {move["id"]: move for move in json.load(handle)}
        with open(f"{data_dir}/gen9/species.json") as handle:
            species = {mon["id"]: mon for mon in json.load(handle)}
        with open(chart_path) as handle:
            self.chart = json.load(handle)
        self.type_names = sorted(self.chart)
        self.type_index = {name: i for i, name in enumerate(self.type_names)}
        # typechart enum value -> capitalised name (the chart's key space)
        self.type_of_enum = {
            value: name.capitalize()
            for name, value in enums["typechart"].items()
            if not name.startswith("_")
        }
        self.move_of_enum = {
            value: moves[name]
            for name, value in enums["moves"].items()
            if name in moves
        }
        self.species_types = {
            value: species[name]["types"]
            for name, value in enums["species"].items()
            if name in species
        }
        self.ability_immunity = {
            value: _ABILITY_IMMUNITY[name]
            for name, value in enums["abilities"].items()
            if name in _ABILITY_IMMUNITY
        }

    def effectiveness(self, attack: str, defend: list[str], immune_to) -> int:
        if attack == immune_to:
            return IMMUNE
        total = 0
        for defend_type in defend:
            factor = self.chart[attack][defend_type]
            if factor == -9:
                return IMMUNE
            total += factor
        if total < 0:
            return RESISTED
        if total > 0:
            return SUPEREFFECTIVE
        return NEUTRAL


def opponent_types(tables: TypeTables, revealed_row, public_row):
    """The opponent active's CURRENT types, or None when its species is unknown."""
    base = tables.species_types.get(int(revealed_row[_SPECIES]))
    if base is None:
        return None
    tera = tables.type_of_enum.get(int(revealed_row[_TERA]))
    if tera is not None and tera != "Stellar":
        return [tera]
    bits = (int(public_row[_TYPECHANGE[0]]) & 0xFFFF) | (
        (int(public_row[_TYPECHANGE[1]]) & 0xFFFF) << 16
    )
    if bits:
        changed = [
            tables.type_of_enum[value]
            for value in tables.type_of_enum
            if bits & (1 << value)
        ]
        if changed:
            return changed
    return base


def slot_cells(slot: int) -> np.ndarray:
    return CELL_BANK_SRC == _BANK_MOVE_OFFSET + slot


def label_batch(tables: TypeTables, env, steps: np.ndarray):
    """One record per (step, legal damaging regular move): indices, class,
    move type, opp primary type, the legal target row, and the policy mass
    on the move (regular + wildcard cells)."""
    mask = np.asarray(env.action_mask, bool)
    moveset = np.asarray(env.my_moveset)
    revealed = np.asarray(env.revealed_team)
    public = np.asarray(env.public_team)
    records = []
    for time_index, batch_index in zip(*np.nonzero(steps)):
        row_mask = mask[time_index, batch_index]
        defend = opponent_types(
            tables,
            revealed[time_index, batch_index, _OPP_ROW],
            public[time_index, batch_index, _OPP_ROW],
        )
        if defend is None:
            continue
        immune_to = tables.ability_immunity.get(
            int(revealed[time_index, batch_index, _OPP_ROW, _ABILITY])
        )
        for slot in range(NUM_REGULAR_MOVES):
            regular = slot_cells(slot) & row_mask
            wildcard = slot_cells(slot + WILDCARD_ROW_OFFSET) & row_mask
            if not regular.any():
                continue
            move = tables.move_of_enum.get(
                int(moveset[time_index, batch_index, slot, _MOVE_ID])
            )
            if move is None or move["category"] == "Status":
                continue
            records.append(
                dict(
                    t=int(time_index),
                    b=int(batch_index),
                    slot=slot,
                    klass=tables.effectiveness(move["type"], defend, immune_to),
                    move_type=tables.type_index[move["type"]],
                    opp_type=tables.type_index[defend[0]],
                    target=int(
                        (np.nonzero(regular)[0][0] - MOVE_CELL_START) % NUM_TARGETS
                    ),
                    cells=regular | wildcard,
                )
            )
    return records


def one_hot(values: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros((len(values), size), dtype=np.float64)
    out[np.arange(len(values)), values] = 1.0
    return out


def run_probe_e(
    net, variables, chunks, batch_size, seed, alpha, tables, projection_dim
):
    apply_both = jax.jit(
        jax.vmap(
            lambda variables, actor_input: net.apply(
                variables, actor_input, method=_assembled_and_encoded_fn
            ),
            in_axes=(None, 1),
            out_axes=1,
        )
    )
    # The actor path stores no log_policy (train=False); re-run the
    # learner-side heads over the stored rows for the full-support policy.
    apply_heads = make_apply(net)
    dev_variables = jax.device_put(variables)
    rng = np.random.default_rng(seed)
    projection = None

    readouts = {
        "move row (pre-trunk)": [],
        "move row (post-trunk)": [],
        "target row (post-trunk)": [],
        "opp active row (post-trunk)": [],
        f"bilinear move x target (post, rp{projection_dim})": [],
    }
    klass, move_type, opp_type, chunk_of = [], [], [], []
    # behaviour: per step, the legal damaging classes and the mass on each
    step_records = []
    for start in range(0, len(chunks), batch_size):
        group = chunks[start : start + batch_size]
        stacked = stack_batch(group)
        actor_input = actor_input_of(stacked)
        assembled, encoded = apply_both(dev_variables, actor_input)
        assembled = np.asarray(assembled, dtype=np.float32)
        encoded = np.asarray(encoded, dtype=np.float32)
        pt = stacked.player_transitions
        env = pt.env_output
        done = np.asarray(env.done)
        steps = (np.cumsum(done, axis=0) - done) == 0
        pred = apply_heads(
            dev_variables, actor_input, pt.agent_output.actor_output, HeadParams()
        )
        policy = np.exp(harness.decode_log_policy(pred, env.action_mask)) * np.asarray(
            env.action_mask, bool
        )
        records = label_batch(tables, env, steps)
        if projection is None:
            width = encoded.shape[-1]
            projection = rng.standard_normal((2, width, projection_dim)) / np.sqrt(
                width
            )
        by_step: dict[tuple[int, int], list] = {}
        for record in records:
            time_index, batch_index, slot = record["t"], record["b"], record["slot"]
            move_row = MOVE_ROWS.start + slot
            target_row = TARGET_ROWS.start + record["target"]
            pre = assembled[time_index, batch_index, move_row]
            post = encoded[time_index, batch_index, move_row]
            post_target = encoded[time_index, batch_index, target_row]
            readouts["move row (pre-trunk)"].append(pre)
            readouts["move row (post-trunk)"].append(post)
            readouts["target row (post-trunk)"].append(post_target)
            readouts["opp active row (post-trunk)"].append(
                encoded[time_index, batch_index, PUBLIC_ROWS.start + _OPP_ROW]
            )
            left = post @ projection[0]
            right = post_target @ projection[1]
            readouts[f"bilinear move x target (post, rp{projection_dim})"].append(
                np.outer(left, right).ravel()
            )
            klass.append(record["klass"])
            move_type.append(record["move_type"])
            opp_type.append(record["opp_type"])
            chunk_of.append(start + batch_index)
            mass = float(policy[time_index, batch_index][record["cells"]].sum())
            by_step.setdefault((time_index, batch_index), []).append(
                (record["klass"], mass)
            )
        step_records.extend(by_step.values())
        logger.info(
            "probe e: %d/%d chunks, %d move records",
            start + len(group),
            len(chunks),
            len(klass),
        )

    klass = np.asarray(klass)
    move_type = np.asarray(move_type)
    opp_type = np.asarray(opp_type)
    chunk_of = np.asarray(chunk_of)
    held_chunks = rng.permutation(len(chunks))[: len(chunks) // 3]
    train = ~np.isin(chunk_of, held_chunks)
    num_types = len(tables.type_names)

    print(
        f"\nrecords: {len(klass)} legal damaging moves over {len(step_records)} steps"
    )
    print(
        "class shares: "
        + ", ".join(
            f"{name} {np.mean(klass == i):.3f}" for i, name in enumerate(CLASS_NAMES)
        )
    )
    print(
        "\nREPRESENTATION -- held-out-by-chunk ridge, effectiveness class of "
        "(my move, their active). acc = argmax accuracy over 4 classes; r = "
        "corr on the ordinal class. Chance acc = majority share."
    )
    print(f"  majority-class floor acc {np.bincount(klass).max() / len(klass):.3f}")
    class_onehot = one_hot(klass, 4)
    shuffled = rng.permutation(klass)
    for name, rows in readouts.items():
        features = np.asarray(rows, dtype=np.float64)
        acc, n_held = _ridge_accuracy(features, class_onehot, train, alpha)
        corr, _ = _ridge_r(features, klass.astype(np.float64), train, alpha)
        floor, _ = _ridge_r(features, shuffled.astype(np.float64), train, alpha)
        print(
            f"  {name:48s} acc {acc:.3f}  r {corr:+.3f}  shuffled r {floor:+.3f}  n_held {n_held}"
        )
    print("\n  positive controls (each operand alone):")
    for name in ("move row (pre-trunk)", "move row (post-trunk)"):
        acc, _ = _ridge_accuracy(
            np.asarray(readouts[name], np.float64),
            one_hot(move_type, num_types),
            train,
            alpha,
        )
        print(
            f"  {name:48s} MOVE TYPE acc {acc:.3f}  (chance {np.bincount(move_type).max() / len(move_type):.3f})"
        )
    for name in (
        "move row (post-trunk)",
        "target row (post-trunk)",
        "opp active row (post-trunk)",
    ):
        acc, _ = _ridge_accuracy(
            np.asarray(readouts[name], np.float64),
            one_hot(opp_type, num_types),
            train,
            alpha,
        )
        print(
            f"  {name:48s} OPP PRIMARY TYPE acc {acc:.3f}  (chance {np.bincount(opp_type).max() / len(opp_type):.3f})"
        )

    print(
        "\nBEHAVIOUR -- actor policy mass (temp 1.0) among legal DAMAGING moves, "
        "renormalised over them, on steps where a better class is also legal. "
        "'uniform' = the share a matchup-blind policy would give. Lower than "
        "uniform on immune/resisted and higher on best is good."
    )
    conditions = {
        "immune, non-immune legal": (lambda c: c == IMMUNE, lambda c: c > IMMUNE),
        "resisted, neutral+ legal": (lambda c: c == RESISTED, lambda c: c > RESISTED),
        "supereffective, weaker legal": (
            lambda c: c == SUPEREFFECTIVE,
            lambda c: c < SUPEREFFECTIVE,
        ),
    }
    for name, (is_class, has_alternative) in conditions.items():
        shares, uniform, n_steps = [], [], 0
        for moves in step_records:
            classes = np.array([klass_ for klass_, _ in moves])
            masses = np.array([mass for _, mass in moves])
            if not (is_class(classes).any() and has_alternative(classes).any()):
                continue
            total = masses.sum()
            if total <= 1e-6:
                continue
            shares.append(masses[is_class(classes)].sum() / total)
            uniform.append(is_class(classes).mean())
            n_steps += 1
        print(
            f"  {name:32s} policy {np.mean(shares):.3f}  uniform {np.mean(uniform):.3f}  n_steps {n_steps}"
        )
    best_shares, best_uniform, n_best = [], [], 0
    for moves in step_records:
        classes = np.array([klass_ for klass_, _ in moves])
        masses = np.array([mass for _, mass in moves])
        if len(set(classes.tolist())) < 2 or masses.sum() <= 1e-6:
            continue
        best = classes == classes.max()
        best_shares.append(masses[best].sum() / masses.sum())
        best_uniform.append(best.mean())
        n_best += 1
    print(
        f"  {'best class present, mixed row':32s} policy {np.mean(best_shares):.3f}  uniform {np.mean(best_uniform):.3f}  n_steps {n_best}"
    )
    print(
        "  (damaging-move mass only; switch/status mass excluded from the "
        "denominator so this reads WHICH attack, not WHETHER to attack)"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-pkl", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-dir", default="data/data")
    parser.add_argument(
        "--chart", required=True, help="typechart json: chart[atk][def]"
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--projection-dim", type=int, default=24)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    chunks = harness.flatten(harness.load(args.games_pkl))
    net = get_player_model(get_player_model_config(9, train=True))
    variables = harness.load_params(args.ckpt)
    tables = TypeTables(args.data_dir, args.chart)
    run_probe_e(
        net,
        variables,
        chunks,
        args.batch,
        args.seed,
        args.ridge_alpha,
        tables,
        args.projection_dim,
    )


if __name__ == "__main__":
    main()
