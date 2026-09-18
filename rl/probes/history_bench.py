"""Actor-forward wall-clock per history recurrence form (LESSONS 09-02's
bench, kept): the actor network (train=False, bf16) on the ex.bin request
broadcast to B requests at history length H, median of `--repeats` timed
calls after a warm-up. Fresh params -- the timing is the program's, not the
weights'. Run with nothing else on the GPU; `kill -USR1` dumps the stacks.

    env/bin/python -m rl.probes.history_bench --out runtime/ablation-history/bench.json
"""

import argparse
import faulthandler
import json
import os
import signal
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


def last_request(history_length: int):
    """The game's LAST request (the longest history) as one request of one
    game: env / actor_output (T, 1, ...) -> (1, ...); the history axes
    (H, 1, ...) -> (H, ...), tail-clipped to `history_length`."""
    example_input, example_output = get_ex_player_step()
    env = jax.tree.map(lambda x: x[-1:, 0], example_input.env)
    actor_output = jax.tree.map(lambda x: x[-1:, 0], example_output)
    history, packed = clip_history_windows_tail(
        jax.tree.map(lambda x: x[:, 0], example_input.history),
        jax.tree.map(lambda x: x[:, 0], example_input.packed_history),
        history_length,
    )
    actor_input = example_input.replace(env=env, history=history, packed_history=packed)
    return actor_input, actor_output


def bench_form(form: str, history_lengths, batches, repeats: int) -> dict:
    config = player_model_config_for(get_learner_config(), train=False)
    config.encoder.history_recurrence = form
    network = get_player_model(config)
    actor_input, actor_output = last_request(history_lengths[0])
    started = time.perf_counter()
    params = jax.jit(network.init)(
        jax.random.key(0), actor_input, actor_output, HeadParams()
    )
    jax.block_until_ready(params)
    print(f"{form}: init {time.perf_counter() - started:.1f}s", flush=True)

    def one(params, key, actor_input, actor_output):
        return network.apply(
            params, actor_input, actor_output, HeadParams(), rngs={"sampling": key}
        )

    apply = jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0)))
    report = {}
    for history_length in history_lengths:
        actor_input, actor_output = last_request(history_length)
        for batch in batches:
            batched = jax.tree.map(
                lambda x: jnp.broadcast_to(x[None], (batch,) + x.shape),
                (actor_input, actor_output),
            )
            keys = jax.random.split(jax.random.key(0), batch)
            started = time.perf_counter()
            for _ in range(3):
                jax.block_until_ready(apply(params, keys, *batched))
            compile_seconds = time.perf_counter() - started
            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                jax.block_until_ready(apply(params, keys, *batched))
                samples.append((time.perf_counter() - started) * 1e3)
            key = f"{form}/H{history_length}/B{batch}"
            report[key] = statistics.median(samples)
            print(
                f"{key:>20} {report[key]:8.2f} ms (compile {compile_seconds:.1f}s)",
                flush=True,
            )
    return report


def main(argv=None):
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forms", nargs="+", default=["loop", "stacked"])
    parser.add_argument("--history", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--batch", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)
    report = {}
    for form in arguments.forms:
        report.update(
            bench_form(form, arguments.history, arguments.batch, arguments.repeats)
        )
    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    Path(arguments.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
