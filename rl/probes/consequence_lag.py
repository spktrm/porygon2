"""Is the change the consequence model predicts a real state change, or a
component the trunk writes to be predictable?

For the fixed-slot predicted rows (the 3 FIELD rows and STATE_VALUE_CLS) over
consecutive requests of a game: the lag-1 change norm, the lag-2 change norm,
and the cosine between successive changes. A row that ALTERNATES has lag-2
changes much smaller than lag-1 and successive changes pointing opposite ways
(cosine near -1); a row tracking the battle has lag-2 >= lag-1 and a cosine
near 0. The entity rows are read only where the public order did not move.

    env/bin/python -m rl.probes.consequence_lag \\
        --root runtime/tactical-cohort-20260920 --checkpoint ckpts/gen9/ckpt_N
"""

import argparse
import json
import logging
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

from rl.environment.protos.features_pb2 import InfoFeature
from rl.environment.utils import acted_rows
from rl.model.consequence import CONSEQUENCE_GROUP_IDS, CONSEQUENCE_GROUPS
from rl.model.constants import NUM_PUBLIC_SLOTS
from rl.model.heads import HeadParams
from rl.model.player_model import get_player_model
from rl.offline import harness
from rl.online.artifact import player_model_config_for
from rl.online.config import get_learner_config
from rl.online.training.batching import stack_batch
from rl.probes.separation_probe import actor_input_of

logger = logging.getLogger(__name__)
_ORDER = slice(
    InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0,
    InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0 + NUM_PUBLIC_SLOTS,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--chunks", type=int, default=160)
    parser.add_argument("--batch", type=int, default=4)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    network = get_player_model(player_model_config_for(get_learner_config()))
    variables = jax.device_put(harness.load_params(arguments.checkpoint))
    forward = jax.jit(jax.vmap(network.apply, in_axes=(None, 1, 1, None), out_axes=1))
    sides = harness.load(f"{arguments.root}/games.pkl")
    chunks = [chunk for side in sides for chunk in side][: arguments.chunks]

    sums = {
        name: dict(lag1=0.0, lag2=0.0, cosine=0.0, row=0.0, count=0)
        for name in CONSEQUENCE_GROUPS
    }
    for start in range(0, len(chunks) - arguments.batch + 1, arguments.batch):
        batch = stack_batch(chunks[start : start + arguments.batch])
        transitions = batch.player_transitions
        prediction = forward(
            variables,
            actor_input_of(batch),
            transitions.agent_output.actor_output,
            HeadParams(),
        )
        rows = np.asarray(prediction.consequence_inputs.state_rows, np.float32)
        valid = np.asarray(prediction.consequence_inputs.state_valid)
        info = np.asarray(transitions.env_output.info)
        acted = acted_rows(np.asarray(transitions.env_output.done))
        # Three consecutive real requests with every row in place throughout.
        steady = acted[:-2] & acted[1:-1]
        same_order = (info[:-2, :, _ORDER] == info[1:-1, :, _ORDER]).all(-1) & (
            info[1:-1, :, _ORDER] == info[2:, :, _ORDER]
        ).all(-1)
        first = rows[1:-1] - rows[:-2]
        second = rows[2:] - rows[1:-1]
        lag2 = rows[2:] - rows[:-2]
        usable = (steady & same_order)[..., None] & valid[:-2] & valid[1:-1] & valid[2:]
        cosine = (first * second).sum(-1) / np.maximum(
            np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1), 1e-9
        )
        for index, name in enumerate(CONSEQUENCE_GROUPS):
            chosen = usable & (CONSEQUENCE_GROUP_IDS == index)
            entry = sums[name]
            entry["lag1"] += float(np.linalg.norm(first, axis=-1)[chosen].sum())
            entry["lag2"] += float(np.linalg.norm(lag2, axis=-1)[chosen].sum())
            entry["cosine"] += float(cosine[chosen].sum())
            entry["row"] += float(np.linalg.norm(rows[:-2], axis=-1)[chosen].sum())
            entry["count"] += int(chosen.sum())
    result = {}
    for name, entry in sums.items():
        count = max(entry["count"], 1)
        result[name] = dict(
            rows=entry["count"],
            row_norm=entry["row"] / count,
            lag1_change=entry["lag1"] / count,
            lag2_change=entry["lag2"] / count,
            lag2_over_lag1=entry["lag2"] / max(entry["lag1"], 1e-9),
            successive_change_cosine=entry["cosine"] / count,
        )
    print(json.dumps(dict(checkpoint=arguments.checkpoint, **result), indent=1))


if __name__ == "__main__":
    main()
