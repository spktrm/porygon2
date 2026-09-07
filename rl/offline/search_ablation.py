"""Fixed-checkpoint search pilot; each invocation runs one arm in isolation.

Actor RNG seeds are fixed, but the live service does not expose simulator/team
seeds: game outcomes across arms are independent, not paired. Report this limit
rather than applying paired confidence intervals to unrelated battles.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from rl.offline import harness  # noqa: E402


def summarise_game(side):
    record = {"outcome": harness.outcome(side), "steps": 0}
    metrics = {}
    for chunk in side:
        env = chunk.player_transitions.env_output
        real_rows = int(chunk.game_length[0] - chunk.game_step_offset[0])
        acted = (np.arange(env.done.shape[0]) < real_rows) & ~np.asarray(env.done, bool)
        # Overlapping final rows bootstrap the next chunk; they are not actions.
        acted[-1] = False
        record["steps"] += int(acted.sum())
        search = chunk.player_transitions.agent_output.actor_output.search
        for name in (
            "root_kl",
            "root_value_gap",
            "legal_truncated",
            "candidate_retained_mass",
            "candidate_occupied",
            "mcts_model_calls",
            "mcts_depth_reached",
        ):
            value = getattr(search, name)
            if not isinstance(value, tuple):
                metrics.setdefault(name, []).extend(np.asarray(value)[acted].tolist())
    for name, values in metrics.items():
        if values:
            record[name] = float(np.mean(values))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--arm",
        choices=("plain", "expectimax1", "expectimax2", "mcts1", "mcts2"),
        required=True,
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--chance-samples", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    mode = args.arm.rstrip("12")
    if args.arm.endswith("2"):
        depth = 2
    else:
        depth = 1
    variables = harness.load_params(args.checkpoint)
    start = time.perf_counter()
    sides = harness.play_games(
        variables,
        args.games,
        pairs=1,
        tag=f"ablation-{args.arm}-{args.seed}",
        seed=args.seed,
        opponent="heuristic",
        search_mode=mode,
        search_depth=depth,
        simulations=args.simulations,
        chance_samples=args.chance_samples,
        temperature=1.0,
        device=args.device,
        deadline_s=7200,
    )
    elapsed = time.perf_counter() - start
    games = [summarise_game(side) for side in sides]
    result = {
        "checkpoint": args.checkpoint,
        "parameters": "target_params",
        "arm": args.arm,
        "temperature": 1.0,
        "device": args.device,
        "simulations": args.simulations,
        "chance_samples": args.chance_samples,
        "requested_games": args.games,
        "completed_games": len(games),
        "elapsed_seconds_including_compile": elapsed,
        "simulator_seeds_paired": False,
        "games": games,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(
        json.dumps({key: value for key, value in result.items() if key != "games"}),
        flush=True,
    )
    if len(games) != args.games:
        raise RuntimeError("Incomplete ablation arm; see failed/abandoned log counts")


if __name__ == "__main__":
    main()
