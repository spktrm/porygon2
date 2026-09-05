"""The trunk's sows are real: `COLLECT_INTERMEDIATES=1` captures every
block's attention weights and residual stream, stacked on the block axis.

From a1c18ed to 2026-09-02 the trunk's nn.scan lifted only `params`, so
the attention sow inside it was a silent no-op and scripts/attn_probe.py
reported "intermediates captured: 0" -- flax drops a sow whose collection
is not lifted through every transform above it, without a warning. The
flag is read at import (rl/model/modules.py), so each arm of this test is
a fresh interpreter; the no-flag arm is the control proving the gate is
closed in training.
"""

import json
import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

_SCRIPT = """
import json
import jax
import numpy as np
from rl.environment.utils import get_ex_player_step
from rl.model.config import get_player_model_config
from rl.model.heads import HeadParams
from rl.model.player_model import get_player_model

net = get_player_model(get_player_model_config(generation=9, train=True))
actor_input, actor_output = jax.tree.map(lambda x: x[:, 0], get_ex_player_step())
params = jax.jit(net.init)(jax.random.key(0), actor_input, actor_output, HeadParams())


def encode(module, actor_input):
    sequence, *_ = module.encoder(
        actor_input.env, actor_input.packed_history, actor_input.history
    )
    return sequence


@jax.jit
def run(params):
    sequence, mutated = net.apply(
        params, actor_input, method=encode, mutable=["intermediates"]
    )
    return sequence, mutated


sequence, mutated = run(params)
report = {"has_intermediates": "intermediates" in mutated}
if report["has_intermediates"]:
    blocks = mutated["intermediates"]["encoder"]["trunk"]["blocks"]
    weights = blocks["attention"]["attn_weights"][0]
    residual = blocks["residual"][0]
    report["attn_shape"] = list(weights.shape)
    report["residual_shape"] = list(residual.shape)
    final = np.asarray(residual[:, -1], dtype=np.float32)
    report["final_block_max_diff"] = float(
        np.abs(final - np.asarray(sequence, dtype=np.float32)).max()
    )
print("REPORT " + json.dumps(report))
"""


def _run(collect: bool) -> dict:
    env = dict(os.environ)
    env["COLLECT_INTERMEDIATES"] = "1" if collect else "0"
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    lines = [line for line in proc.stdout.splitlines() if line.startswith("REPORT ")]
    assert len(lines) == 1, proc.stdout[-2000:]
    return json.loads(lines[0][len("REPORT ") :])


def test_trunk_sows_are_captured_per_block():
    from rl.model.config import get_player_model_config
    from rl.model.constants import NUM_SEQUENCE_ROWS

    cfg = get_player_model_config(generation=9, train=True)
    report = _run(collect=True)
    assert report["has_intermediates"]
    time, blocks, heads, n_q, n_k = report["attn_shape"]
    assert blocks == cfg.encoder.trunk.num_blocks
    assert heads == cfg.encoder.trunk.num_heads
    assert (n_q, n_k) == (NUM_SEQUENCE_ROWS, NUM_SEQUENCE_ROWS)
    assert report["residual_shape"] == [
        time,
        blocks,
        NUM_SEQUENCE_ROWS,
        cfg.encoder.trunk.model_size,
    ]
    # The last block's sown residual IS the trunk's output.
    assert report["final_block_max_diff"] == 0.0


def test_training_forward_collects_nothing():
    report = _run(collect=False)
    assert not report["has_intermediates"]
