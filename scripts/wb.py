"""Compact wandb queries — terse output by design, for cheap in-terminal
checks (and to keep agent context small). Named runs/metrics only; never
enumerates entities or projects.

Usage (always via env/bin/python):
    scripts/wb.py latest [-n 3]                      # newest runs, one line each
    scripts/wb.py summary [--run ID] KEY [KEY ...]   # selected summary keys
    scripts/wb.py metric NAME [--run ID] [--last 15] # downsampled metric tail
    scripts/wb.py compare RUN_A RUN_B KEY [KEY ...]  # side-by-side summaries

--run accepts a run id or display name; omitted = newest running run
(else newest run). Default entity/project: jtwin/pokemon-rl.
"""

import argparse
import sys

import wandb

ENTITY = "jtwin"
PROJECT = "pokemon-rl"


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def api_path():
    return f"{ENTITY}/{PROJECT}"


def newest_runs(api, count):
    return list(api.runs(api_path(), order="-created_at", per_page=max(count, 1)))[
        :count
    ]


def resolve_run(api, ref):
    if ref:
        try:
            return api.run(f"{api_path()}/{ref}")
        except wandb.errors.CommError:
            matches = list(
                api.runs(api_path(), filters={"display_name": ref}, per_page=1)
            )
            if not matches:
                sys.exit(f"no run with id or name '{ref}'")
            return matches[0]
    candidates = newest_runs(api, 6)
    running = [run for run in candidates if run.state == "running"]
    for pool in (running, candidates):
        for run in pool:
            # Prefer the main population's run — the default question is
            # almost always "how is main doing".
            if run.name.endswith("-main"):
                return run
        if pool:
            return pool[0]
    sys.exit("no runs found")


def cmd_latest(api, args):
    for run in newest_runs(api, args.n):
        step = run.summary.get("training_step", run.summary.get("_step", "?"))
        print(f"{run.state:9} {run.name:40.40} {run.id} step={fmt(step)}")


def cmd_summary(api, args):
    run = resolve_run(api, args.run)
    print(f"# {run.name} ({run.id}, {run.state})")
    for key in args.keys:
        print(f"{key} = {fmt(run.summary.get(key, '<absent>'))}")


def cmd_metric(api, args):
    run = resolve_run(api, args.run)
    rows = run.history(keys=[args.name], samples=args.last, pandas=False)
    print(f"# {run.name} ({run.id}) {args.name}, {len(rows)} samples")
    for row in rows:
        print(f"{row.get('_step', '?')}\t{fmt(row.get(args.name))}")


def cmd_compare(api, args):
    run_a = resolve_run(api, args.run_a)
    run_b = resolve_run(api, args.run_b)
    print(f"# {'key':32} {run_a.name:>16.16} {run_b.name:>16.16}")
    for key in args.keys:
        left = fmt(run_a.summary.get(key, "<absent>"))
        right = fmt(run_b.summary.get(key, "<absent>"))
        print(f"{key:34} {left:>16} {right:>16}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_latest = sub.add_parser("latest")
    p_latest.add_argument("-n", type=int, default=3)

    p_summary = sub.add_parser("summary")
    p_summary.add_argument("keys", nargs="+")
    p_summary.add_argument("--run", default=None)

    p_metric = sub.add_parser("metric")
    p_metric.add_argument("name")
    p_metric.add_argument("--run", default=None)
    p_metric.add_argument("--last", type=int, default=15)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("run_a")
    p_compare.add_argument("run_b")
    p_compare.add_argument("keys", nargs="+")

    args = parser.parse_args()
    api = wandb.Api(timeout=60)
    {
        "latest": cmd_latest,
        "summary": cmd_summary,
        "metric": cmd_metric,
        "compare": cmd_compare,
    }[args.cmd](api, args)


if __name__ == "__main__":
    main()
