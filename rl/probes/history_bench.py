"""Actor-forward wall-clock per history recurrence form (LESSONS 09-02's
bench, kept): the actor network (train=False, bf16) on the ex.bin request
broadcast to B requests at history length H, median of `--repeats` timed
calls after a warm-up. Fresh params -- the timing is the program's, not the
weights'. Run with nothing else on the GPU.

    env/bin/python -m rl.probes.history_bench --out runtime/ablation-history/bench.json
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from rl.environment.utils import clip_history_windows_tail, get_ex_player_step
from rl.model.heads import HeadParams
from rl.model.player_model import get_player_model
from rl.online.artifact import player_model_config_for
from rl.online.config import get_learner_config


def timed_forward(form: str, history_length: int, batch: int, repeats: int) -> float:
    config = player_model_config_for(get_learner_config(), train=False)
    config.encoder.history_recurrence = form
    network = get_player_model(config)
    example_input, example_output = get_ex_player_step()
    # The game's LAST request (the longest history), as one request of one
    # game: env / actor_output (T, 1, ...) -> (1, ...); the history axes are
    # (H, 1, ...) -> (H, ...), then tail-clipped to `history_length`.
    env = jax.tree.map(lambda x: x[-1:, 0], example_input.env)
    actor_output = jax.tree.map(lambda x: x[-1:, 0], example_output)
    history, packed = clip_history_windows_tail(
        jax.tree.map(lambda x: x[:, 0], example_input.history),
        jax.tree.map(lambda x: x[:, 0], example_input.packed_history),
        history_length,
    )
    actor_input = example_input.replace(env=env, history=history, packed_history=packed)
    actor_input, actor_output = jax.tree.map(
        lambda x: jnp.broadcast_to(x[None], (batch,) + x.shape),
        (actor_input, actor_output),
    )
    params = jax.jit(jax.vmap(network.init, in_axes=(None, 0, 0, None)))(
        jax.random.key(0), actor_input, actor_output, HeadParams()
    )
    params = jax.tree.map(lambda x: x[0], params)
    apply = jax.jit(jax.vmap(network.apply, in_axes=(None, 0, 0, None)))
    for _ in range(3):
        jax.block_until_ready(apply(params, actor_input, actor_output, HeadParams()))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(apply(params, actor_input, actor_output, HeadParams()))
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forms", nargs="+", default=["loop", "stacked"])
    parser.add_argument("--history", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--batch", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)
    report = {}
    for form in arguments.forms:
        for history_length in arguments.history:
            for batch in arguments.batch:
                key = f"{form}/H{history_length}/B{batch}"
                report[key] = timed_forward(
                    form, history_length, batch, arguments.repeats
                )
                print(f"{key:>20} {report[key]:8.2f} ms")
    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    Path(arguments.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
