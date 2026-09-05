"""Behavioural statistics off Showdown protocol logs — one parser for
every population, so a human replay (omniscient log), a model self-play
game and a model-vs-heuristic game (both p1's log from the service's
BATTLE_LOG_DIR dump) are measured with the same code.

    env/bin/python rl/offline/battle_stats.py \\
        --human replays/data/gen9randombattle --human-min-rating 1900 \\
        --logs /path/to/selfplay_logs --logs /path/to/heuristic_logs

A "population" is a directory (or the human replay set) plus which side(s)
to score; per side per game the parser walks the turns tracking each
side's active and its hp fraction, and classifies every observed action:
a chosen move (by category off @pkmn/dex) or a switch, the latter as
lead / forced by a faint / pivot (selfSwitch move) / eject-class effect /
drag / VOLUNTARY. Only voluntary switches are decisions. Numbers are
pooled over decisions unless the row says per game.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
from dataclasses import dataclass, field

import numpy as np

SERVICE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "service")

_MOVES_EXPORT = """
const {Dex} = require("@pkmn/dex");
const d = Dex.forGen(9);
const out = {};
for (const m of d.moves.all()) {
  out[m.name] = {
    cat: m.category, type: m.type, bp: m.basePower,
    protect: !!(m.stallingMove), heal: !!(m.heal || (m.flags && m.flags.heal)),
    setup: !!(m.boosts || (m.self && m.self.boosts)) && m.category === "Status",
    hazard: !!m.sideCondition && m.target === "foeSide",
    selfSwitch: !!m.selfSwitch, status: !!m.status,
  };
}
process.stdout.write(JSON.stringify(out));
"""


def load_moves() -> dict:
    raw = subprocess.check_output(["node", "-e", _MOVES_EXPORT], cwd=SERVICE_DIR)
    return json.loads(raw)


_HP_RE = re.compile(r"^(\d+)/(\d+)")
_EJECT_MARKERS = ("Eject Button", "Eject Pack", "Emergency Exit", "Wimp Out")

HP_BUCKETS = ((0.0, 0.34), (0.34, 0.67), (0.67, 1.01))


def hp_fraction(text: str) -> float:
    if text.startswith("0 fnt") or text.startswith("0/"):
        return 0.0
    match = _HP_RE.match(text)
    if match is None:
        return float("nan")
    return int(match.group(1)) / max(int(match.group(2)), 1)


@dataclass
class SideStats:
    moves: collections.Counter = field(default_factory=collections.Counter)
    attack_outcomes: collections.Counter = field(default_factory=collections.Counter)
    attack_damage: list[float] = field(default_factory=list)
    switch_kinds: collections.Counter = field(default_factory=collections.Counter)
    switch_out_hp: list[float] = field(default_factory=list)
    switch_in_hp: list[float] = field(default_factory=list)
    # (own hp bucket) -> [moves, voluntary switches]
    decisions_by_hp: dict = field(
        default_factory=lambda: collections.defaultdict(lambda: [0, 0])
    )
    tera_turn: float = float("nan")
    fainted: int = 0
    won: bool = False


@dataclass
class GameStats:
    turns: int
    sides: dict[str, SideStats]


def parse_log(lines: list[str], moves: dict) -> GameStats | None:
    sides = {"p1": SideStats(), "p2": SideStats()}
    names = {}
    active_hp = {"p1": float("nan"), "p2": float("nan")}
    needs_replacement = {"p1": False, "p2": False}
    pending_switch = {"p1": None, "p2": None}
    turn = 0
    winner = None
    last_attack = None  # (side, target hp before) while an attack resolves

    def side_of(ident: str) -> str:
        return ident[:2]

    for line in lines:
        if not line.startswith("|"):
            continue
        parts = line.split("|")[1:]
        cmd = parts[0]
        args = parts[1:]
        if cmd == "player" and len(args) >= 2:
            names[args[0]] = args[1]
        elif cmd == "turn":
            turn = int(args[0])
            pending_switch = {"p1": None, "p2": None}
            last_attack = None
        elif cmd in ("switch", "drag"):
            side = side_of(args[0])
            stats = sides[side]
            out_hp = active_hp[side]
            if cmd == "drag":
                kind = "drag"
            elif turn == 0:
                kind = "lead"
            elif needs_replacement[side]:
                kind = "faint"
            elif pending_switch[side] is not None:
                kind = pending_switch[side]
            else:
                kind = "voluntary"
                stats.switch_out_hp.append(out_hp)
                bucket = hp_bucket(out_hp)
                if bucket is not None:
                    stats.decisions_by_hp[bucket][1] += 1
            stats.switch_kinds[kind] += 1
            in_hp = hp_fraction(args[2]) if len(args) > 2 else float("nan")
            if kind == "voluntary":
                stats.switch_in_hp.append(in_hp)
            active_hp[side] = in_hp
            needs_replacement[side] = False
            pending_switch[side] = None
            last_attack = None
        elif cmd == "move":
            if any(a.startswith("[from]") for a in args[3:]):
                continue
            side = side_of(args[0])
            stats = sides[side]
            move = moves.get(args[1])
            if move is None:
                stats.moves["unknown"] += 1
                last_attack = None
                continue
            bucket = hp_bucket(active_hp[side])
            if bucket is not None:
                stats.decisions_by_hp[bucket][0] += 1
            if move["selfSwitch"]:
                pending_switch[side] = "pivot"
            if move["cat"] == "Status":
                if move["setup"]:
                    stats.moves["setup"] += 1
                elif move["hazard"]:
                    stats.moves["hazard"] += 1
                elif move["heal"]:
                    stats.moves["recovery"] += 1
                elif move["protect"]:
                    stats.moves["protect"] += 1
                elif move["status"]:
                    stats.moves["status_infliction"] += 1
                else:
                    stats.moves["status_other"] += 1
                last_attack = None
            else:
                stats.moves["attack"] += 1
                stats.attack_outcomes["total"] += 1
                if any(a == "[miss]" for a in args):
                    stats.attack_outcomes["miss"] += 1
                    last_attack = None
                else:
                    opp = "p2" if side == "p1" else "p1"
                    last_attack = (side, opp, active_hp[opp], False)
        elif cmd in ("-supereffective", "-resisted", "-immune"):
            if last_attack is not None and side_of(args[0]) == last_attack[1]:
                sides[last_attack[0]].attack_outcomes[cmd[1:]] += 1
                if cmd == "-immune":
                    sides[last_attack[0]].attack_damage.append(0.0)
                    last_attack = None
        elif cmd == "-miss":
            if last_attack is not None and side_of(args[0]) == last_attack[0]:
                sides[last_attack[0]].attack_outcomes["miss"] += 1
                sides[last_attack[0]].attack_damage.append(0.0)
                last_attack = None
        elif cmd in ("-damage", "-heal", "-sethp"):
            side = side_of(args[0])
            new_hp = hp_fraction(args[1]) if len(args) > 1 else float("nan")
            residual = any(a.startswith("[from]") for a in args[2:])
            if (
                last_attack is not None
                and cmd == "-damage"
                and side == last_attack[1]
                and not residual
                and not last_attack[3]
            ):
                before = last_attack[2]
                if np.isfinite(before) and np.isfinite(new_hp):
                    stats = sides[last_attack[0]]
                    stats.attack_damage.append(max(before - new_hp, 0.0))
                    if new_hp == 0.0:
                        stats.attack_outcomes["ko"] += 1
                # first hit only (multi-hit moves land several -damage lines)
                last_attack = (last_attack[0], last_attack[1], before, True)
            if np.isfinite(new_hp):
                active_hp[side] = new_hp
        elif cmd == "faint":
            side = side_of(args[0])
            sides[side].fainted += 1
            needs_replacement[side] = True
            active_hp[side] = 0.0
        elif cmd == "-terastallize":
            side = side_of(args[0])
            sides[side].tera_turn = turn
        elif cmd in ("-enditem", "-activate"):
            if len(args) > 1 and any(m in line for m in _EJECT_MARKERS):
                pending_switch[side_of(args[0])] = "eject"
        elif cmd == "win":
            winner = args[0]
        elif cmd == "tie":
            winner = None

    if turn == 0:
        return None
    for side, stats in sides.items():
        stats.won = winner is not None and names.get(side) == winner
    return GameStats(turns=turn, sides=sides)


def hp_bucket(hp: float) -> int | None:
    if not np.isfinite(hp) or hp <= 0.0:
        return None
    for index, (lo, hi) in enumerate(HP_BUCKETS):
        if lo <= hp < hi:
            return index
    return None


def summarise(games: list[GameStats], sides_wanted) -> dict[str, float]:
    moves = collections.Counter()
    outcomes = collections.Counter()
    kinds = collections.Counter()
    damage, out_hp, in_hp, tera_rel = [], [], [], []
    by_hp = collections.defaultdict(lambda: [0, 0])
    remaining_win, remaining_lose = [], []
    tera_games = 0
    n_sides = 0
    for game in games:
        for side in sides_wanted:
            stats = game.sides[side]
            n_sides += 1
            moves.update(stats.moves)
            outcomes.update(stats.attack_outcomes)
            kinds.update(stats.switch_kinds)
            damage.extend(stats.attack_damage)
            out_hp.extend(h for h in stats.switch_out_hp if np.isfinite(h))
            in_hp.extend(h for h in stats.switch_in_hp if np.isfinite(h))
            for bucket, counts in stats.decisions_by_hp.items():
                by_hp[bucket][0] += counts[0]
                by_hp[bucket][1] += counts[1]
            if np.isfinite(stats.tera_turn):
                tera_games += 1
                tera_rel.append(stats.tera_turn / max(game.turns, 1))
            if stats.won:
                remaining_win.append(6 - stats.fainted)
            else:
                remaining_lose.append(6 - stats.fainted)
    n_moves = sum(moves.values())
    vol = kinds["voluntary"]
    decisions = n_moves + vol
    attacks = max(outcomes["total"], 1)
    row = {
        "games": len(games),
        "turns/game": np.mean([g.turns for g in games]),
        "decisions/side": decisions / max(n_sides, 1),
        "vol_switch_rate": vol / max(decisions, 1),
        "switch_kinds vol/faint/pivot/eject/drag": (
            f"{vol}/{kinds['faint']}/{kinds['pivot']}/{kinds['eject']}/{kinds['drag']}"
        ),
        "vol_switch_out_hp": np.mean(out_hp) if out_hp else float("nan"),
        "vol_switch_in_hp": np.mean(in_hp) if in_hp else float("nan"),
    }
    for index, (lo, hi) in enumerate(HP_BUCKETS):
        counts = by_hp[index]
        total = counts[0] + counts[1]
        row[f"vol_switch_rate own_hp[{lo:.2f},{min(hi, 1.0):.2f})"] = counts[1] / max(
            total, 1
        )
    for key in (
        "attack",
        "setup",
        "hazard",
        "recovery",
        "protect",
        "status_infliction",
        "status_other",
    ):
        row[f"move_share {key}"] = moves[key] / max(n_moves, 1)
    for key in ("supereffective", "resisted", "immune", "miss", "ko"):
        row[f"attack {key}"] = outcomes[key] / attacks
    row["attack dmg (frac of target max)"] = np.mean(damage) if damage else float("nan")
    row["attack dmg==0 share"] = (
        float(np.mean(np.array(damage) == 0.0)) if damage else float("nan")
    )
    row["tera games"] = tera_games / max(n_sides, 1)
    row["tera turn / game turns"] = np.mean(tera_rel) if tera_rel else float("nan")
    row["mons left (winner)"] = (
        np.mean(remaining_win) if remaining_win else float("nan")
    )
    row["mons left (loser)"] = (
        np.mean(remaining_lose) if remaining_lose else float("nan")
    )
    return row


def read_human(path: str, min_rating: int, limit: int) -> list[list[str]]:
    logs = []
    for name in sorted(os.listdir(path)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(path, name)) as handle:
            log = json.load(handle)["log"]
        ratings = re.findall(r"\|player\|p[12]\|[^|\n]*\|[^|\n]*\|(\d+)", log)
        if len(ratings) < 2 or min(int(r) for r in ratings) < min_rating:
            continue
        logs.append(log.split("\n"))
        if len(logs) >= limit:
            break
    return logs


def read_dir(path: str) -> list[list[str]]:
    logs = []
    for name in sorted(os.listdir(path)):
        if name.endswith(".log"):
            with open(os.path.join(path, name)) as handle:
                logs.append(handle.read().split("\n"))
    return logs


def print_table(columns: dict[str, dict[str, float]]) -> None:
    keys = list(next(iter(columns.values())).keys())
    width = max(len(k) for k in keys) + 2
    print("".ljust(width) + "".join(name.rjust(22) for name in columns))
    for key in keys:
        cells = []
        for row in columns.values():
            value = row[key]
            if isinstance(value, str):
                cells.append(value.rjust(22))
            elif isinstance(value, (int, np.integer)):
                cells.append(f"{value}".rjust(22))
            else:
                cells.append(f"{value:.3f}".rjust(22))
        print(key.ljust(width) + "".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", default=None)
    parser.add_argument("--human-min-rating", type=int, default=1900)
    parser.add_argument("--human-limit", type=int, default=2000)
    parser.add_argument(
        "--logs",
        action="append",
        default=[],
        help="NAME=DIR[:p1|p2|both] — a directory of dumped logs and which side to score",
    )
    args = parser.parse_args()
    moves = load_moves()
    columns = {}
    if args.human:
        games = [
            g
            for g in (
                parse_log(lines, moves)
                for lines in read_human(
                    args.human, args.human_min_rating, args.human_limit
                )
            )
            if g is not None
        ]
        columns[f"human>={args.human_min_rating}"] = summarise(games, ("p1", "p2"))
    for spec in args.logs:
        name, rest = spec.split("=", 1)
        if ":" in rest:
            path, side = rest.rsplit(":", 1)
        else:
            path, side = rest, "both"
        if side == "both":
            sides = ("p1", "p2")
        else:
            sides = (side,)
        games = [
            g
            for g in (parse_log(lines, moves) for lines in read_dir(path))
            if g is not None
        ]
        columns[name] = summarise(games, sides)
    print_table(columns)


if __name__ == "__main__":
    main()
