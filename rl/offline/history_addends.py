"""History-row addend rms, offline (lineage-instrumentation step d1).

A HISTORY_ENTITY row is `gru_state + node_snapshot + entity_index_tag[i]`
(rl/model/encoder.py `_assemble_sequence`): the tag is the join key that
lets the trunk match a diary to its board row, and nothing in training
measures whether it survives the sum. The GRU state is tanh-bounded; the
pooled snapshot is not. This reads the rms of each addend over the
order-valid rows, on fresh init (is the key drowned from step 0?) and on a
checkpoint (has `entity_index_tag` grown into it?). Pre-registered alarm:
tag rms / (states + snapshots) rms < 0.05.

    env/bin/python rl/offline/history_addends.py [--ckpt ckpts/gen9/ckpt_00182000]
"""

import argparse
import logging

import jax
import jax.numpy as jnp
import numpy as np

from rl.environment.protos.features_pb2 import InfoFeature
from rl.model.config import get_player_model_config
from rl.model.constants import NUM_PUBLIC_SLOTS
from rl.model.player_model import get_player_model
from rl.offline import harness
from rl.offline.separation_probe import actor_input_of, fresh_variables
from rl.offline.trunk_homogeneity import valid_steps
from rl.online.training.batching import stack_batch

logger = logging.getLogger(__name__)

ADDENDS = ("states", "snapshots", "tag", "states+snapshots")


def _addends(module, actor_input):
    """Per (t, public row): rms of each addend and the order-valid mask."""
    encoder = module.encoder
    row_states, order_valid, _, snapshot_rows = encoder._history_inputs(
        actor_input.env, actor_input.packed_history, actor_input.history
    )
    public_order = actor_input.env.info[
        ...,
        InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0 : InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11
        + 1,
    ]
    tag = jnp.take(
        encoder.entity_index_tag, jnp.clip(public_order + 1, 0, NUM_PUBLIC_SLOTS), 0
    )
    states = row_states.astype(jnp.float32)
    snapshots = snapshot_rows.astype(jnp.float32)

    def rms(values):
        return jnp.sqrt(jnp.mean(jnp.square(values), -1))

    return {
        "states": rms(states),
        "snapshots": rms(snapshots),
        "tag": rms(tag.astype(jnp.float32)),
        "states+snapshots": rms(states + snapshots),
    }, order_valid


def run(net, variables, chunks, batch_size: int):
    read = jax.jit(
        jax.vmap(
            lambda variables, actor_input: net.apply(
                variables, actor_input, method=_addends
            ),
            in_axes=(None, 1),
            out_axes=1,
        )
    )
    dev_variables = jax.device_put(variables)
    sums = {name: 0.0 for name in ADDENDS}
    count = 0
    for start in range(0, len(chunks), batch_size):
        batch = stack_batch(chunks[start : start + batch_size])
        readings, order_valid = read(dev_variables, actor_input_of(batch))
        valid = valid_steps(np.asarray(batch.player_transitions.env_output.done))
        rows = np.asarray(order_valid) & valid[..., None]
        count += rows.sum()
        for name in ADDENDS:
            sums[name] += float(np.asarray(readings[name])[rows].sum())
    means = {name: sums[name] / max(count, 1) for name in ADDENDS}
    print(f"mean rms over {int(count)} order-valid history rows")
    for name in ADDENDS:
        print(f"  {name:>18}  {means[name]:.4f}")
    ratio = means["tag"] / max(means["states+snapshots"], 1e-9)
    print(f"  tag / (states+snapshots) = {ratio:.4f}  (alarm < 0.05)")
    return means


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
    run(net, variables, chunks, args.batch)


if __name__ == "__main__":
    main()
