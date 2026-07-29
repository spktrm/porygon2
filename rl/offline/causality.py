"""Causality check: does the offline critic's Φ at turn k change when the
game's future is deleted?

Every trajectory step shares one terminal history cache (the O(T) shard
convention), so per-step causality rests on the request-count gather
stamps AND on every other cache read being masked down to the step. This
verifies the whole pipeline end to end, with no assumptions about where a
leak could hide: truncate the raw log just after |turn|k (appending a
synthetic |win| line so the exporter accepts the game — it lands after
every compared state, so it cannot affect them), export both versions
through the real exporter, and compare per-member Φ step by step on the
shared prefix (states for turns 1..k exist identically in both).

The two logs are byte-identical up to |turn|k, so:

- Δ at float-noise level (different padding buckets change XLA reduction
  shapes): the pipeline is causal. An "incredibly confident early Φ" is
  then a calibration problem, not a leak.
- Systematic Δ: future information reaches earlier states through the
  shared cache — and equally a train/serve skew, since live caches never
  contain the future.

Usage:
    python -m rl.offline.causality <replay> [--ckpt ...] \
        [--fractions 0.25,0.5,0.75] [--turns 12,30]

<replay> accepts the same forms as rl.offline.visualize (path / id / URL).
"""

import argparse
import json
import os
import tempfile

import numpy as np

from rl.offline.visualise import (
    CriticRunner,
    discover_ckpts,
    export_record,
    resolve_replay,
)

PASS_MAX_DELTA = 5e-3
WARN_MAX_DELTA = 5e-2


def truncate_log(log: str, turn: int, winner: str) -> str:
    """Everything up to and including the |turn|<turn> line, plus a
    synthetic |win| so the exporter accepts the game. The exporter emits
    the turn-k state upon reading the |turn|k line, before it ever sees
    the appended win, so all compared states are untouched by it."""
    out = []
    for line in log.split("\n"):
        out.append(line)
        if line.startswith("|turn|") and line.split("|")[2] == str(turn):
            out.append(f"|win|{winner}")
            return "\n".join(out)
    raise ValueError(f"log has no |turn|{turn} line")


def perspective_phi(runner: CriticRunner, payload: bytes, stats: dict) -> np.ndarray:
    """Per-member Φ for the p1 perspective, valid steps only: (K, n)."""
    outputs, _ = runner.run(payload)
    perspectives = stats["perspectives"]
    idx = perspectives.index(0) if 0 in perspectives else 0
    valid = int(outputs["mask"][:, idx].sum())
    return outputs["phi"][:, :valid, idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", help="replay JSON path, replay id, or replay URL")
    parser.add_argument("--ckpt", action="append", default=None)
    parser.add_argument(
        "--fractions",
        default="0.25,0.5,0.75",
        help="truncation points as fractions of the game's turn count",
    )
    parser.add_argument(
        "--turns", default=None, help="explicit truncation turns, e.g. 12,30"
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        replay, replay_json_path = resolve_replay(args.replay, tmpdir)
        num_turns = sum(
            1 for line in replay["log"].split("\n") if line.startswith("|turn|")
        )
        if args.turns:
            turns = [int(t) for t in args.turns.split(",")]
        else:
            turns = [round(float(f) * num_turns) for f in args.fractions.split(",")]
        # k = num_turns would compare the full game to itself; k < 2 leaves
        # fewer than two states and the exporter drops the trajectory.
        turns = sorted({min(max(k, 2), num_turns - 1) for k in turns})

        ckpt_paths = args.ckpt or discover_ckpts(
            replay.get("formatid", "gen9randombattle")
        )
        print(f"replay: {replay.get('id')} ({num_turns} turns)")
        print(f"checkpoints: {ckpt_paths}")
        runner = CriticRunner(ckpt_paths)

        full_payload, full_stats = export_record(replay_json_path, tmpdir)
        phi_full = perspective_phi(runner, full_payload, full_stats)  # (K, n)

        worst = 0.0
        print(
            f"\n{'trunc @':>8} {'steps':>6} {'max |Δφ|':>10} "
            f"{'mean |Δφ|':>10} {'worst turn':>11}"
        )
        for k in turns:
            truncated = dict(
                replay,
                log=truncate_log(replay["log"], k, replay["players"][0]),
            )
            trunc_path = os.path.join(tmpdir, f"trunc_{k}.json")
            with open(trunc_path, "w") as f:
                json.dump(truncated, f)
            payload, stats = export_record(trunc_path, tmpdir)
            phi_trunc = perspective_phi(runner, payload, stats)
            # Compare the turn states 1..k only — the truncated export's
            # final state is the synthetic terminal and has no counterpart.
            steps = min(k, phi_trunc.shape[1] - 1, phi_full.shape[1])
            delta = np.abs(phi_trunc[:, :steps] - phi_full[:, :steps])
            worst_turn = int(np.unravel_index(delta.argmax(), delta.shape)[1]) + 1
            worst = max(worst, float(delta.max()))
            print(
                f"{'turn ' + str(k):>8} {steps:>6} {delta.max():>10.2e} "
                f"{delta.mean():>10.2e} {worst_turn:>11}"
            )

    print()
    if worst < PASS_MAX_DELTA:
        print(
            f"PASS — max |Δφ| {worst:.2e}: states are unchanged by deleting "
            "the future (differences at this level are padding-bucket float "
            "noise). The pipeline is causal; confident early Φ is a "
            "calibration/overfit question, not leakage."
        )
    elif worst < WARN_MAX_DELTA:
        print(
            f"BORDERLINE — max |Δφ| {worst:.2e}: larger than float noise "
            "usually is. Rerun on a few more replays; a consistent gap this "
            "size deserves a look at what reads the cache unmasked."
        )
    else:
        print(
            f"FAIL — max |Δφ| {worst:.2e}: earlier states change when the "
            "future is deleted, so future information reaches them through "
            "the shared terminal cache (and live play, whose caches never "
            "contain the future, is off-distribution for this critic). "
            "Prime suspect: reads of packed_history that bypass the "
            "request-count gather — e.g. pooling masks built from "
            "history_slot_sides over the terminal cache's slot rows."
        )


if __name__ == "__main__":
    main()
