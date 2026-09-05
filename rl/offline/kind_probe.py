"""Kind confounding through the trunk (2026-09-03).

The trunk is one sequence of ~10 row KINDS (public entities, my sheet,
move slots, target slots, history, field, ...) through one attention and
ONE shared MLP, told apart only by additive group/row biases. The
readout consumes the rows AFTER six blocks, and the type ceiling read
post-trunk rows as LESS legible than the assembled input (0.50 vs 0.60)
while `player_trunk_row_participation` fell 7.1 -> 4.5 over irqeetfg.
This asks, per block, whether that is the trunk squeezing the kinds into
a shared subspace:

  1. OWN legibility -- a ridge readout of each row's own wire attribute
     (species types, hp, move type) from the row at every block, held out
     by chunk. Falling through the blocks = the trunk overwriting a row's
     own content with what it read from other rows.
  2. KIND identity -- ridge one-hot classification of the row's kind from
     the row alone; trivially ~1.0 at the input (the group bias), so a
     drop is the kinds losing their separating direction.
  3. SUBSPACE overlap -- the fraction of kind A's centred variance that
     lies inside kind B's top-k principal subspace, at the input and the
     final block. Rising overlap with falling own legibility is the
     confounding shape; rising overlap with legibility held is the kinds
     sharing a basis harmlessly.

Needs the residual sow, so COLLECT_INTERMEDIATES is set before rl.model
is imported. Runs on ANY architecture with the same row groups, so the
same script reads the sum-pool lineage against this one.

    env/bin/python rl/offline/kind_probe.py --ckpt ckpts/gen9/ckpt_00260000 \\
        --games-pkl runtime/lineage_games_ckpt182000.pkl
"""

import os

os.environ["COLLECT_INTERMEDIATES"] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import logging  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from rl.environment.protos.features_pb2 import (  # noqa: E402
    EntityPrivateNodeFeature,
    EntityPublicNodeFeature,
    EntityRevealedNodeFeature,
    MovesetFeature,
)
from rl.model.config import get_player_model_config  # noqa: E402
from rl.model.constants import (  # noqa: E402
    ENEMY_TARGET_ROWS,
    HISTORY_ENTITY_ROWS,
    MOVE_ROWS,
    PRIVATE_ROWS,
    PUBLIC_ROWS,
    TARGET_ROWS,
)
from rl.model.player_model import get_player_model  # noqa: E402
from rl.offline import harness  # noqa: E402
from rl.offline.separation_probe import (  # noqa: E402
    _ridge_accuracy,
    _ridge_predict,
    _ridge_r,
    actor_input_of,
    fresh_variables,
)
from rl.offline.trunk_homogeneity import _sequences, valid_steps  # noqa: E402
from rl.offline.type_probe import _OPP_ROW, TypeTables, opponent_types  # noqa: E402
from rl.online.training.batching import stack_batch  # noqa: E402

logger = logging.getLogger(__name__)

_REVEALED_SPECIES = EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
_PUBLIC_HP = EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO
_PUBLIC_ACTIVE = EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE
_PRIVATE_SPECIES = EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SPECIES
_PRIVATE_HP = EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__HP_RATIO
_MOVE_ID = MovesetFeature.MOVESET_FEATURE__MOVE_ID
_MOVE_PP = MovesetFeature.MOVESET_FEATURE__PP
NUM_REGULAR_MOVES = 4
KINDS = ("public", "private", "move", "target", "opp_active", "history")


def make_reader(net):
    """chunk batch -> residual stack (T, B, blocks + 1, rows, dim) f32;
    block 0 is the assembled input."""

    def per_chunk(variables, actor_input):
        (assembled, _), mutated = net.apply(
            variables, actor_input, method=_sequences, mutable=["intermediates"]
        )
        residual = mutated["intermediates"]["encoder"]["trunk"]["blocks"]["residual"][0]
        return jnp.concatenate([assembled.astype(jnp.float32)[:, None], residual], 1)

    return jax.jit(jax.vmap(per_chunk, in_axes=(None, 1), out_axes=1))


def _type_multi_hot(tables: TypeTables, types) -> np.ndarray:
    out = np.zeros(len(tables.type_names), dtype=np.float64)
    for name in types:
        out[tables.type_index[name]] = 1.0
    return out


def collect(net, variables, tables, chunks, batch_size, steps_per_chunk, rng):
    """Per kind: rows (n, blocks + 1, dim), labels {name: (n, k)}, chunk ids."""
    read = make_reader(net)
    dev_variables = jax.device_put(variables)
    rows = {kind: [] for kind in KINDS}
    labels = {kind: {} for kind in KINDS}
    chunk_ids = {kind: [] for kind in KINDS}

    def add(kind, chunk_id, vector, **named):
        rows[kind].append(vector)
        chunk_ids[kind].append(chunk_id)
        for name, value in named.items():
            labels[kind].setdefault(name, []).append(value)

    for start in range(0, len(chunks), batch_size):
        batch = stack_batch(chunks[start : start + batch_size])
        env = batch.player_transitions.env_output
        stack = np.asarray(read(dev_variables, actor_input_of(batch)))
        valid = valid_steps(np.asarray(env.done))
        revealed = np.asarray(env.revealed_team)
        public = np.asarray(env.public_team)
        private = np.asarray(env.private_team)
        moveset = np.asarray(env.my_moveset)
        for batch_index in range(stack.shape[1]):
            chunk_id = start + batch_index
            steps = np.nonzero(valid[:, batch_index])[0]
            if len(steps) == 0:
                continue
            steps = rng.choice(steps, min(steps_per_chunk, len(steps)), replace=False)
            for time_index in steps:
                sequence = stack[time_index, batch_index]
                live = np.linalg.norm(sequence[0], axis=-1) > 0
                # Public rows + their history rows: species types, hp, active.
                for slot, row in enumerate(
                    range(*PUBLIC_ROWS.indices(sequence.shape[1]))
                ):
                    types = tables.species_types.get(
                        int(revealed[time_index, batch_index, slot, _REVEALED_SPECIES])
                    )
                    if types is None or not live[row]:
                        continue
                    public_row = public[time_index, batch_index, slot]
                    named = dict(
                        types=_type_multi_hot(tables, types),
                        hp=np.float64(public_row[_PUBLIC_HP]),
                        active=np.float64(public_row[_PUBLIC_ACTIVE]),
                    )
                    add("public", chunk_id, sequence[:, row], **named)
                    history_row = HISTORY_ENTITY_ROWS.start + slot
                    if live[history_row]:
                        add("history", chunk_id, sequence[:, history_row], **named)
                # My sheet: species types, hp.
                for slot, row in enumerate(
                    range(*PRIVATE_ROWS.indices(sequence.shape[1]))
                ):
                    private_row = private[time_index, batch_index, slot]
                    types = tables.species_types.get(int(private_row[_PRIVATE_SPECIES]))
                    if types is None or not live[row]:
                        continue
                    add(
                        "private",
                        chunk_id,
                        sequence[:, row],
                        types=_type_multi_hot(tables, types),
                        hp=np.float64(private_row[_PRIVATE_HP]),
                    )
                # My active's regular move rows: move type, base power, pp.
                for slot in range(NUM_REGULAR_MOVES):
                    row = MOVE_ROWS.start + slot
                    move_row = moveset[time_index, batch_index, slot]
                    move = tables.move_of_enum.get(int(move_row[_MOVE_ID]))
                    if move is None or not live[row]:
                        continue
                    add(
                        "move",
                        chunk_id,
                        sequence[:, row],
                        type=_type_multi_hot(tables, [move["type"]]),
                        power=np.float64(move.get("basePower", 0)),
                        pp=np.float64(move_row[_MOVE_PP]),
                    )
                # The singles target row: the opponent active's CURRENT types.
                row = TARGET_ROWS.start + int(ENEMY_TARGET_ROWS[0])
                defend = opponent_types(
                    tables,
                    revealed[time_index, batch_index, _OPP_ROW],
                    public[time_index, batch_index, _OPP_ROW],
                )
                if defend is not None and live[row]:
                    add(
                        "target",
                        chunk_id,
                        sequence[:, row],
                        types=_type_multi_hot(tables, defend),
                    )
                    # The public row the target row is built from, at the
                    # same steps: the n-matched calibration for the read above.
                    add(
                        "opp_active",
                        chunk_id,
                        sequence[:, PUBLIC_ROWS.start + _OPP_ROW],
                        types=_type_multi_hot(tables, defend),
                    )
    out = {}
    for kind in KINDS:
        if not rows[kind]:
            continue
        out[kind] = (
            np.stack(rows[kind]).astype(np.float64),
            {name: np.stack(values) for name, values in labels[kind].items()},
            np.asarray(chunk_ids[kind]),
        )
    return out


def _type_hit(features, multi_hot, train):
    """Held-out 'argmax of the readout is one of the row's types'."""
    fit = _ridge_predict(features, multi_hot, train)
    if fit is None:
        return float("nan"), 0
    pred, y_held = fit
    hit = y_held[np.arange(len(pred)), pred.argmax(-1)] > 0
    return float(hit.mean()), len(pred)


def own_legibility(collected, held_fold: int):
    """{(kind, label): (blocks + 1,)} held-out readings, plus n."""
    table = {}
    for kind, (rows, labels, chunk_ids) in collected.items():
        train = chunk_ids % 3 != held_fold
        for name, values in labels.items():
            readings = []
            for block in range(rows.shape[1]):
                features = rows[:, block]
                if name in ("types", "type"):
                    reading, n_held = _type_hit(features, values, train)
                else:
                    reading, n_held = _ridge_r(features, values, train)
                readings.append(reading)
            table[(kind, name)] = (np.asarray(readings), n_held)
    return table


def kind_identity(collected, held_fold: int, rng, per_kind: int):
    """Held-out accuracy of kind-from-row, per block, on a kind-balanced sample."""
    features = []
    one_hot = []
    chunk_ids = []
    kinds = list(collected)
    for index, kind in enumerate(kinds):
        rows, _, ids = collected[kind]
        pick = rng.choice(len(rows), min(per_kind, len(rows)), replace=False)
        features.append(rows[pick])
        chunk_ids.append(ids[pick])
        hot = np.zeros((len(pick), len(kinds)))
        hot[:, index] = 1.0
        one_hot.append(hot)
    features = np.concatenate(features)
    one_hot = np.concatenate(one_hot)
    train = np.concatenate(chunk_ids) % 3 != held_fold
    return np.asarray(
        [
            _ridge_accuracy(features[:, block], one_hot, train)[0]
            for block in range(features.shape[1])
        ]
    )


def subspace_overlap(collected, block: int, rank: int):
    """overlap[a, b] = fraction of kind a's centred variance inside kind b's
    top-`rank` principal subspace. The diagonal is each kind's own top-rank
    variance share (how concentrated it is)."""
    kinds = list(collected)
    bases = {}
    centred = {}
    for kind in kinds:
        rows = collected[kind][0][:, block]
        rows = rows - rows.mean(0)
        centred[kind] = rows
        _, _, vt = np.linalg.svd(rows, full_matrices=False)
        bases[kind] = vt[:rank]
    overlap = np.zeros((len(kinds), len(kinds)))
    for i, kind_a in enumerate(kinds):
        total = np.square(centred[kind_a]).sum()
        for j, kind_b in enumerate(kinds):
            projected = centred[kind_a] @ bases[kind_b].T
            overlap[i, j] = np.square(projected).sum() / max(total, 1e-12)
    return kinds, overlap


def print_legibility(table):
    blocks = len(next(iter(table.values()))[0])
    print("\nown legibility (held-out; types/type = argmax-hit, else Pearson r)")
    print(
        f"{'kind':>8} {'label':>7} {'n':>6}  "
        + "  ".join(f"{b:>6}" for b in ["input"] + [str(b) for b in range(1, blocks)])
    )
    for (kind, name), (readings, n_held) in table.items():
        print(
            f"{kind:>8} {name:>7} {n_held:>6}  "
            + "  ".join(f"{r:6.3f}" for r in readings)
        )


def print_overlap(kinds, overlap, label):
    print(
        f"\nsubspace overlap @ {label} (row kind's variance inside column kind's top-k subspace)"
    )
    print(f"{'':>8} " + " ".join(f"{k:>8}" for k in kinds))
    for kind, row in zip(kinds, overlap):
        print(f"{kind:>8} " + " ".join(f"{v:8.3f}" for v in row))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-pkl", default="runtime/lineage_games_ckpt182000.pkl")
    parser.add_argument("--ckpt", default=None, help="trained params; default fresh")
    parser.add_argument("--data-dir", default="data/data")
    parser.add_argument("--chart", default="rl/offline/typechart.json")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps-per-chunk", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(args.seed)

    chunks = harness.flatten(harness.load(args.games_pkl))
    tables = TypeTables(args.data_dir, args.chart)
    net = get_player_model(get_player_model_config(9, train=True))
    if args.ckpt:
        variables = harness.load_params(args.ckpt)
        source = args.ckpt
    else:
        variables = fresh_variables(net, stack_batch(chunks[: args.batch]), args.seed)
        source = "fresh init"
    print(f"params: {source}; {len(chunks)} chunks", flush=True)
    collected = collect(
        net, variables, tables, chunks, args.batch, args.steps_per_chunk, rng
    )
    for kind, (rows, _, _) in collected.items():
        print(f"{kind}: {len(rows)} rows")
    print_legibility(own_legibility(collected, held_fold=0))
    identity = kind_identity(collected, 0, rng, per_kind=3000)
    print(
        "\nkind identity (held-out acc, kind-balanced): "
        + "  ".join(f"{a:.3f}" for a in identity)
    )
    for block in (0, -1):
        kinds, overlap = subspace_overlap(collected, block, args.rank)
        label = "input" if block == 0 else "final"
        print_overlap(kinds, overlap, f"{label}, rank {args.rank}")


if __name__ == "__main__":
    main()
