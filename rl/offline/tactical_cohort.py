"""The frozen tactical cohort — the ONE metric that judges the 2026-09-09
action-support set as a whole (descriptive, never attributive).

`ineffective_confident_mass`: over held-out game contexts, the policy's
probability mass on legal damaging moves that are immediately ineffective
against the revealed opponent (immune by type, or by a REVEALED ability —
labelled separately), in states where a legal damaging alternative exists.
A random miss is not an immunity failure; a forced choice, a choice lock
or an Encore (no alternative damaging move) is excluded from the primary
count by construction. Whole-game bootstrap intervals, since states within
a game are not independent.

Two commands. `collect` plays the cohort ONCE against the service's
SimpleHeuristic at T=.5 with fixed seeds and pickles it — fixed forever
after, so every checkpoint is read on the same contexts (the
runtime/priority-audit-01861967 recipe). Point PS_SERVICE_URI at a
service started with BATTLE_LOG_DIR set so the simulator's own `-immune`
lines confirm the taken-move events. `read` runs a checkpoint's target
params over the cohort and emits the metric at T=1 and T=.5.

    env/bin/python -m rl.offline.tactical_cohort collect --checkpoint ckpts/gen9/ckpt_N
    env/bin/python -m rl.offline.tactical_cohort read --checkpoint ckpts/gen9/ckpt_M \\
        --out runtime/tactical-cohort/read-ckpt_M.json
"""

import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

from rl.environment.protos.features_pb2 import (  # noqa: E402
    EntityRevealedNodeFeature,
    InfoFeature,
    MovesetFeature,
)
from rl.environment.utils import acted_rows  # noqa: E402
from rl.model.constants import (  # noqa: E402
    _BANK_MOVE_OFFSET,
    CELL_BANK_SRC,
    OPP_ACTIVE_PUBLIC_ROWS,
)
from rl.offline import harness  # noqa: E402
from rl.offline.type_probe import (  # noqa: E402
    IMMUNE,
    TypeTables,
    label_batch,
    opponent_types,
)

logger = logging.getLogger(__name__)

ROOT = Path("runtime/tactical-cohort")
GAMES_PATH = ROOT / "games.pkl"
LOGS_DIR = ROOT / "logs"
CHART_PATH = "runtime/priority-audit-01861967/typechart.json"
TAG = "tactical-cohort"
_ABILITY = EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ABILITY
_MOVE_ID = MovesetFeature.MOVESET_FEATURE__MOVE_ID
_TURN = InfoFeature.INFO_FEATURE__TURN
_OPP_ROW = int(OPP_ACTIVE_PUBLIC_ROWS[0])
# Moves whose type is decided at run time, and abilities that retype or
# bypass immunities: the chart cannot label them, so they are left out.
DYNAMIC_MOVES = {
    "terablast",
    "weatherball",
    "terrainpulse",
    "revelationdance",
    "judgment",
    "multiattack",
    "ivycudgel",
    "ragingbull",
    "technoblast",
    "naturalgift",
    "hiddenpower",
    "flyingpress",
    "freezedry",
}
DYNAMIC_ABILITIES = {
    "aerilate",
    "pixilate",
    "refrigerate",
    "galvanize",
    "normalize",
    "liquidvoice",
    "scrappy",
    "mindseye",
    "moldbreaker",
    "teravolt",
    "turboblaze",
}


def immunity_events(path: Path):
    """The simulator's own `-immune` lines against p2 following a p1 move,
    per turn (runtime/priority-audit-01861967/policy_probe.py's reader)."""
    events = []
    turn = 0
    move = None
    for number, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split("|")
        if len(parts) < 3:
            continue
        command = parts[1]
        if command == "turn":
            turn = int(parts[2])
            move = None
        elif command in ("switch", "drag", "replace", "-terastallize", "-transform"):
            move = None
        elif command == "move":
            move = (parts[2][:2], parts[3], number)
        elif command == "-damage" and move is not None:
            if parts[2][:2] != move[0] and not any(
                part.startswith("[from]") for part in parts[4:]
            ):
                move = None
        elif command == "-immune" and move is not None:
            if move[0] == "p1" and parts[2][:2] == "p2":
                events.append(dict(turn=turn, move=move[1]))
    return events


def collect(arguments):
    ROOT.mkdir(parents=True, exist_ok=True)
    if GAMES_PATH.exists() and not arguments.overwrite:
        raise SystemExit(
            f"{GAMES_PATH} exists; the cohort is frozen (--overwrite to rebuild)"
        )
    parameters = harness.load_params(arguments.checkpoint)
    sides = harness.play_games(
        parameters,
        arguments.games,
        pairs=arguments.pairs,
        tag=TAG,
        seed=arguments.seed,
        opponent="heuristic",
        temperature=0.5,
        device=arguments.device,
    )
    harness.dump(sides, str(GAMES_PATH))
    (ROOT / "provenance.json").write_text(
        json.dumps(
            dict(
                checkpoint=arguments.checkpoint,
                parameters="target_params",
                games=len(sides),
                requested=arguments.games,
                temperature=0.5,
                seed=arguments.seed,
                wins=sum(harness.outcome(side) > 0 for side in sides),
            ),
            indent=2,
        )
    )
    logger.info("cohort frozen: %d games at %s", len(sides), GAMES_PATH)


def _sharpen(log_policy, legal, temperature):
    scaled = np.where(legal, log_policy / temperature, -1e9)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    policy = np.exp(scaled) * legal
    return policy / np.maximum(policy.sum(axis=-1, keepdims=True), 1e-30)


def read(arguments):
    sides = harness.load(str(GAMES_PATH))
    parameters = harness.load_params(arguments.checkpoint)
    tables = TypeTables("data/data", CHART_PATH)
    enums = json.loads(Path("data/data/data.json").read_text())
    ability_names = {value: name for name, value in enums["abilities"].items()}
    chunks = []
    game_of_chunk = []
    for game_id, side in enumerate(sides):
        chunks.extend(side)
        game_of_chunk.extend([game_id] * len(side))
    events = {}
    if LOGS_DIR.exists():
        for game_id in range(len(sides)):
            paths = list(LOGS_DIR.glob(f"*{TAG}-g{game_id}_*"))
            if len(paths) == 1:
                events[game_id] = immunity_events(paths[0])

    temperatures = (1.0, 0.5)
    # Per state: game, immune mass at each temperature, the immunity reason.
    states = []
    confirmed = []
    index = 0
    for prediction, batch in harness.forward(parameters, chunks, batch=4):
        env = batch.player_transitions.env_output
        legal = np.asarray(env.action_mask, bool)
        log_policy = np.asarray(prediction.action_head.log_policy, np.float32)
        policies = {
            temperature: _sharpen(log_policy, legal, temperature)
            for temperature in temperatures
        }
        acted = acted_rows(env.done)
        records = label_batch(tables, env, acted)
        by_state = {}
        for record in records:
            move = tables.move_of_enum[
                int(env.my_moveset[record["t"], record["b"], record["slot"], _MOVE_ID])
            ]
            ability = ability_names.get(
                int(env.revealed_team[record["t"], record["b"], _OPP_ROW, _ABILITY])
            )
            if move["id"] in DYNAMIC_MOVES or ability in DYNAMIC_ABILITIES:
                continue
            defend = opponent_types(
                tables,
                env.revealed_team[record["t"], record["b"], _OPP_ROW],
                env.public_team[record["t"], record["b"], _OPP_ROW],
            )
            by_type = tables.effectiveness(move["type"], defend, None) == IMMUNE
            if record["klass"] == IMMUNE and by_type:
                record["reason"] = "type"
            elif record["klass"] == IMMUNE:
                record["reason"] = "revealed_ability"
            else:
                record["reason"] = None
            by_state.setdefault((record["t"], record["b"]), []).append(record)
        for (time_index, column), moves in by_state.items():
            immune = [record for record in moves if record["klass"] == IMMUNE]
            effective = [record for record in moves if record["klass"] > IMMUNE]
            if not immune or not effective:
                continue
            cells = np.zeros(legal.shape[-1], bool)
            for record in immune:
                cells |= record["cells"]
            state = dict(
                game=game_of_chunk[index + column],
                turn=int(env.info[time_index, column, _TURN]),
                reasons=sorted({record["reason"] for record in immune}),
                legal_cells=int(legal[time_index, column].sum()),
            )
            for temperature in temperatures:
                state[f"mass_t{temperature}"] = float(
                    policies[temperature][time_index, column][cells].sum()
                )
            states.append(state)
        # Simulator-confirmed: the taken move drew `-immune` this turn.
        taken = np.asarray(
            batch.player_transitions.agent_output.actor_output.action_head.action_index
        )
        recorded_log_prob = np.asarray(
            batch.player_transitions.agent_output.actor_output.action_head.log_prob
        )
        for time_index, column in zip(*np.nonzero(acted)):
            game_id = game_of_chunk[index + column]
            if game_id not in events:
                continue
            source = (
                int(CELL_BANK_SRC[int(taken[time_index, column])]) - _BANK_MOVE_OFFSET
            )
            if not 0 <= source < 16:
                continue
            move = tables.move_of_enum.get(
                int(env.my_moveset[time_index, column, source, _MOVE_ID])
            )
            if move is None or move["category"] == "Status":
                continue
            turn = int(env.info[time_index, column, _TURN])
            if any(
                event["turn"] == turn and event["move"] == move["name"]
                for event in events[game_id]
            ):
                confirmed.append(
                    dict(
                        game=game_id,
                        turn=turn,
                        move=move["name"],
                        recorded_probability=float(
                            np.exp(recorded_log_prob[time_index, column])
                        ),
                    )
                )
        index += legal.shape[1]

    result = dict(
        checkpoint=arguments.checkpoint,
        games=len(sides),
        states=len(states),
        confirmed_immune_actions=len(confirmed),
        confirmed_recorded_probability_mean=(
            float(np.mean([event["recorded_probability"] for event in confirmed]))
            if confirmed
            else None
        ),
    )
    rng = np.random.default_rng(arguments.seed)
    game_ids = np.asarray([state["game"] for state in states])
    unique_games = np.unique(game_ids)
    for temperature in temperatures:
        masses = np.asarray([state[f"mass_t{temperature}"] for state in states])
        key = f"ineffective_confident_mass_t{temperature}"
        result[key] = float(masses.mean()) if len(masses) else None
        if len(unique_games) > 1:
            samples = []
            for _ in range(arguments.bootstrap):
                drawn = rng.choice(unique_games, size=len(unique_games), replace=True)
                counts = {}
                for game in drawn:
                    counts[game] = counts.get(game, 0) + 1
                weights = np.asarray([counts.get(game, 0) for game in game_ids])
                if weights.sum() == 0:
                    continue
                samples.append(float((masses * weights).sum() / weights.sum()))
            result[f"{key}_ci95"] = [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ]
        for reason in ("type", "revealed_ability"):
            selected = np.asarray([reason in state["reasons"] for state in states])
            if selected.any():
                result[f"{key}_{reason}"] = float(masses[selected].mean())
                result[f"states_{reason}"] = int(selected.sum())
    if arguments.out:
        Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.out).write_text(
            json.dumps(
                dict(result, state_records=states, confirmed=confirmed), indent=2
            )
        )
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collector = commands.add_parser("collect")
    collector.add_argument("--checkpoint", required=True)
    collector.add_argument("--games", type=int, default=240)
    collector.add_argument("--pairs", type=int, default=4)
    collector.add_argument("--seed", type=int, default=909)
    collector.add_argument("--device", default="gpu")
    collector.add_argument("--overwrite", action="store_true")
    reader = commands.add_parser("read")
    reader.add_argument("--checkpoint", required=True)
    reader.add_argument("--out", default=None)
    reader.add_argument("--seed", type=int, default=0)
    reader.add_argument("--bootstrap", type=int, default=1000)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if arguments.command == "collect":
        collect(arguments)
    else:
        read(arguments)


if __name__ == "__main__":
    main()
