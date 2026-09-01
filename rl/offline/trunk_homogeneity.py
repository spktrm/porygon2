"""Trunk row homogeneity per block, offline (lineage-instrumentation step c).

Six ungated pre-norm blocks over 80 rows whose readouts are all PER ROW:
if the rows converge on one direction (Noci et al. 2022, rank collapse) the
existing panels read it as "entropy at ceiling while the pointer params
grow" -- the phase-1 support-anchor shape -- so nothing live can tell the
two apart. This reads `row_homogeneity` (rl/model/trunk.py) on the
assembled input (block 0) and after every trunk block, over the whole
sequence and per SequenceGroup, on real batches; the live panel
(`player_trunk_row_*`, final block only) must reproduce its final-block row.

Needs the residual sow, so COLLECT_INTERMEDIATES is set here before
rl.model is imported (the flag is read at import).

    env/bin/python rl/offline/trunk_homogeneity.py \\
        --games-pkl runtime/lineage_games_ckpt182000.pkl [--ckpt ckpts/gen9/ckpt_00182000]
"""

import os

os.environ["COLLECT_INTERMEDIATES"] = "1"

import argparse  # noqa: E402
import logging  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from rl.model.config import get_player_model_config  # noqa: E402
from rl.model.constants import SEQUENCE_LAYOUT, SEQUENCE_SLICES  # noqa: E402
from rl.model.player_model import get_player_model  # noqa: E402
from rl.model.trunk import row_homogeneity  # noqa: E402
from rl.offline import harness  # noqa: E402
from rl.offline.separation_probe import actor_input_of, fresh_variables  # noqa: E402
from rl.online.training.batching import stack_batch  # noqa: E402

logger = logging.getLogger(__name__)

# Groups with one row have no pair to compare.
GROUPS = [group for group, rows in SEQUENCE_LAYOUT if rows > 1]


def _sequences(module, actor_input):
    """(assembled input, trunk output) for one chunk; the per-block
    residuals ride out in the "intermediates" collection."""
    encoder = module.encoder
    assembled, _ = encoder.assembled_sequence(
        actor_input.env, actor_input.packed_history, actor_input.history
    )
    trunk_out, _ = encoder(
        actor_input.env, actor_input.packed_history, actor_input.history
    )
    return assembled, trunk_out


def valid_steps(done: np.ndarray) -> np.ndarray:
    """Rows up to and including the done row (the learner's cumsum mask)."""
    return (np.cumsum(done, axis=0) - done) == 0


def make_reader(net):
    """chunk batch -> {"all"|group: (cosine, participation)}, each
    (blocks + 1, T, B): block 0 is the assembled input."""

    def per_chunk(variables, actor_input):
        (assembled, trunk_out), mutated = net.apply(
            variables, actor_input, method=_sequences, mutable=["intermediates"]
        )
        residual = mutated["intermediates"]["encoder"]["trunk"]["blocks"]["residual"][0]
        # (T, blocks + 1, rows, dim)
        stack = jnp.concatenate([assembled.astype(jnp.float32)[:, None], residual], 1)
        readings = {"all": row_homogeneity(stack)}
        for group in GROUPS:
            readings[group.name] = row_homogeneity(stack[:, :, SEQUENCE_SLICES[group]])
        return readings, trunk_out

    batched = jax.jit(jax.vmap(per_chunk, in_axes=(None, 1), out_axes=(2, 1)))
    direct = jax.jit(jax.vmap(row_homogeneity, in_axes=1, out_axes=1))

    def read(variables, batch):
        readings, trunk_out = batched(variables, actor_input_of(batch))
        # (T, blocks + 1, B) -> (blocks + 1, T, B)
        readings = jax.tree.map(lambda x: np.asarray(x).transpose(1, 0, 2), readings)
        return readings, np.asarray(direct(trunk_out)[1])

    return read


def run(net, variables, chunks, batch_size: int):
    read = make_reader(net)
    dev_variables = jax.device_put(variables)
    sums = {}
    counts = {}
    checked = False
    for start in range(0, len(chunks), batch_size):
        batch = stack_batch(chunks[start : start + batch_size])
        readings, direct_participation = read(dev_variables, batch)
        valid = valid_steps(np.asarray(batch.player_transitions.env_output.done))
        if not checked:
            # The sown final block IS the trunk's output -- the read's own
            # positive control against a stale or mis-indexed collection.
            _, final_participation = readings["all"]
            assert np.allclose(
                final_participation[-1], direct_participation, equal_nan=True
            )
            checked = True
        for name, (cosine, participation) in readings.items():
            for key, values in (("cosine", cosine), ("participation", participation)):
                finite = np.isfinite(values) & valid[None]
                sums.setdefault((name, key), np.zeros(values.shape[0]))
                counts.setdefault((name, key), np.zeros(values.shape[0]))
                sums[(name, key)] += np.where(finite, values, 0.0).sum((1, 2))
                counts[(name, key)] += finite.sum((1, 2))
    return {key: sums[key] / np.maximum(counts[key], 1) for key in sums}, counts


def print_table(means, counts, key: str):
    # Groups whose rows are never valid on this data drop out (PREV_ACTION
    # is the within-turn sub-decision context, so singles never fills it).
    names = [
        name
        for name in ["all"] + [group.name for group in GROUPS]
        if counts[(name, key)].max() > 0
    ]
    blocks = len(means[("all", key)])
    print(f"\n{key} (mean over {int(counts[('all', key)].max())} valid steps)")
    print("block  " + "  ".join(f"{name[:12]:>12}" for name in names))
    for block in range(blocks):
        label = "input" if block == 0 else str(block)
        print(
            f"{label:>5}  "
            + "  ".join(f"{means[(name, key)][block]:12.3f}" for name in names)
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-pkl", default="runtime/lineage_games_ckpt182000.pkl")
    parser.add_argument("--ckpt", default=None, help="trained params; default fresh")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    chunks = harness.flatten(harness.load(args.games_pkl))
    net = get_player_model(get_player_model_config(9, train=True))
    if args.ckpt:
        variables = harness.load_params(args.ckpt)
        source = args.ckpt
    else:
        variables = fresh_variables(net, stack_batch(chunks[: args.batch]), args.seed)
        source = "fresh init"
    print(f"params: {source}; {len(chunks)} chunks", flush=True)
    means, counts = run(net, variables, chunks, args.batch)
    print_table(means, counts, "cosine")
    print_table(means, counts, "participation")


if __name__ == "__main__":
    main()
