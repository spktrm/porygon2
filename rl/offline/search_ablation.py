"""Fixed-checkpoint search read (plan Step 4): one arm per invocation --
plain, search (depth-1 event rollouts) or blind (the same rollouts, no
bonus) -- against the service's SimpleHeuristic. Actor RNG seeds are
fixed, but the service does not expose simulator/team seeds, so the arms'
games are independent, not paired: report Wilson intervals per arm and a
two-proportion test, never a paired interval.

    PS_SERVICE_URI=ws://localhost:8081 env/bin/python rl/offline/search_ablation.py \\
        --checkpoint ckpts/gen9/ckpt_XXXXXXXX --world-model ckpts/world_model/gen9randombattle/ckpt_best \\
        --arm search --games 400 --seed 123 --out runtime/search-read/search.json
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from rl.environment.utils import acted_rows
from rl.offline import harness


def summarise_game(side) -> dict:
    record = {"outcome": harness.outcome(side), "steps": 0}
    metrics: dict[str, list[float]] = {}
    for chunk in side:
        env = chunk.player_transitions.env_output
        acted = acted_rows(env.done)
        record["steps"] += int(acted.sum())
        actor_output = chunk.player_transitions.agent_output.actor_output
        for name in ("search_root_kl", "search_bonus_gap", "search_overflow"):
            value = getattr(actor_output, name)
            if not isinstance(value, tuple):
                metrics.setdefault(name, []).extend(
                    np.asarray(value, np.float32)[acted].tolist()
                )
    for name, values in metrics.items():
        if values:
            record[name] = float(np.mean(values))
    return record


def wilson(wins: int, games: int) -> tuple[float, float]:
    if games == 0:
        return (0.0, 1.0)
    z = 1.96
    p = wins / games
    denominator = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denominator
    half = z * np.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return (float(centre - half), float(centre + half))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--world-model", default=None)
    parser.add_argument("--arm", choices=("plain", "search", "blind"), required=True)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.arm == "plain":
        params = harness.load_params(args.checkpoint)
        search_arm = None
    else:
        if args.world_model is None:
            raise ValueError("--world-model is required for the search and blind arms")
        params = harness.load_search_params(args.checkpoint, args.world_model)
        search_arm = args.arm
    start = time.perf_counter()
    sides = harness.play_games(
        params,
        args.games,
        pairs=args.pairs,
        tag=f"searchread-{args.arm}-{args.seed}",
        seed=args.seed,
        opponent="heuristic",
        temperature=1.0,
        device=args.device,
        deadline_s=7200,
        search_arm=search_arm,
    )
    elapsed = time.perf_counter() - start
    games = [summarise_game(side) for side in sides]
    wins = sum(1 for game in games if game["outcome"] > 0)
    result = {
        "checkpoint": args.checkpoint,
        "world_model": args.world_model,
        "arm": args.arm,
        "requested_games": args.games,
        "completed_games": len(games),
        "wins": wins,
        "winrate": wins / max(1, len(games)),
        "winrate_wilson95": wilson(wins, len(games)),
        "elapsed_seconds_including_compile": elapsed,
        "simulator_seeds_paired": False,
        "games": games,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(
        json.dumps({key: value for key, value in result.items() if key != "games"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
