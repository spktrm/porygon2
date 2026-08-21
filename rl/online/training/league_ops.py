"""League bookkeeping: the checkpoint-pacing gate, snapshot publication,
and the payoff-table readouts.

Free functions over (run_state, league, config) — none of this needs the
Learner, and the pacing gate in particular is the piece worth testing on
its own (tests/test_learner_gate.py).
"""

import logging
import os

import jax
import numpy as np
import wandb

from rl import checkpoint
from rl.environment.data import STOI
from rl.model.utils import ParamsContainer
from rl.online.config import Porygon2LearnerConfig
from rl.online.league import LIVE_KEYS, MAIN_KEY, League, PlayerRef
from rl.online.training.run_state import AddReason, RunState

logger = logging.getLogger(__name__)


# (_measure_exploitability/_update_exploit_controller/_apply_exploit_
# scale removed 2026-08-14 with the ExploitabilityController — the
# worst-matchup win-rate signal still exists in _should_add_new_player's
# "dominant" gate; it just doesn't actuate anything anymore.)

def should_add_new_player(
    run_state: RunState, league: League, config: Porygon2LearnerConfig
) -> AddReason | None:
    """Returns why a snapshot should join the league, or None to skip.
    main only."""
    # Pacing is measured against main's OWN last checkpoint (AlphaStar
    # MainPlayer.ready_to_checkpoint: steps since the last checkpoint step),
    # not the league's newest entry — a foreign-origin publication
    # would otherwise become "latest" permanently (its offset key wins
    # max()) with a frame count that never advances, firing an overdue
    # add on every league-management tick.
    latest = league.get_latest_player(origin="main")
    current = league.get_live(MAIN_KEY)

    latest_frames = latest.player_frame_count if latest is not None else 0
    frames_passed = int(current.player_frame_count - latest_frames)

    if frames_passed < config.add_player_min_frames:
        return None

    historical_players = [
        v for k, v in league.players.items() if k not in LIVE_KEYS
    ]

    if not historical_players:
        if (
            int(run_state.player_state.step_count)
            > config.minimum_historical_player_steps
        ):
            return "initial"
        return None

    win_rates = league.get_winrate((current, historical_players))

    if win_rates.min() > 0.7:
        return "dominant"
    if frames_passed >= config.add_player_max_frames:
        return "overdue"
    return None

def create_params_container(run_state: RunState) -> ParamsContainer:
    return ParamsContainer(
        player_frame_count=jax.device_get(run_state.player_state.frame_count).item(),
        builder_frame_count=jax.device_get(run_state.builder_state.frame_count).item(),
        step_count=MAIN_KEY,
        player_params=jax.device_get(run_state.player_state.params),
        builder_params=jax.device_get(run_state.builder_state.params),
    )

def add_player_to_league(
    run_state: RunState,
    league: League,
    config: Porygon2LearnerConfig,
    step: int,
    origin: str = "main",
):
    """Persist the current params as an opponent snapshot and register
    a ref. Only the params files are written (no optimiser state); the
    league holds the lightweight ref and materialises the params
    lazily when this player is actually drawn as an opponent."""
    league_step = step
    players_root = f"./ckpts/gen{config.generation}/players"
    snapshot_dir = os.path.abspath(f"{players_root}/p_{league_step:08}")
    checkpoint.save_param_snapshot(
        snapshot_dir,
        player_components=dict(
            params=jax.device_get(run_state.player_state.params),
            target_params=jax.device_get(run_state.player_state.target_params),
        ),
        builder_components=dict(
            params=jax.device_get(run_state.builder_state.params),
            target_params=jax.device_get(run_state.builder_state.target_params),
        ),
    )
    league.add_player(
        PlayerRef(
            step_count=league_step,
            snapshot_dir=snapshot_dir,
            player_frame_count=jax.device_get(run_state.player_state.frame_count).item(),
            builder_frame_count=jax.device_get(
                run_state.builder_state.frame_count
            ).item(),
            player_key="params",
            builder_key="params",
            origin=origin,
        )
    )

def get_usage_counts(run_state: RunState):
    result = {}
    for key, counts in [
        ("species", run_state.player_replay._species_counts),
        ("items", run_state.player_replay._item_counts),
        ("abilities", run_state.player_replay._ability_counts),
        ("moves", run_state.player_replay._move_counts),
    ]:
        names = list(STOI[key])
        table = wandb.Table(columns=[key, "usage"])
        for name, count in zip(names, counts):
            table.add_data(name, count)
        result[f"{key}_usage"] = table
    return result

def winrate_tracked_opponents(league: League) -> list[PlayerRef]:
    """Every historical league member."""
    return [v for k, v in league.players.items() if k not in LIVE_KEYS]

def ref_label(ref: PlayerRef) -> str:
    """Payoff-table label: the snapshot's own step count."""
    return f"{ref.step_count}"

def get_league_winrates(league: League) -> dict:
    current = league.get_live(MAIN_KEY)
    others = winrate_tracked_opponents(league)
    if not others:
        return {}
    win_rates = league.get_winrate((current, others))
    # Origin-labelled keys ("league_main_v_ME-1834_winrate") still
    # match scripts/wandb_views.py's ^league_main_v_.*_winrate$ panel
    # regex.
    return {
        f"league_main_v_{ref_label(others[i])}_winrate": wr
        for i, wr in enumerate(win_rates)
    }

def get_league_winrate_heatmap(league: League) -> dict:
    """Full pairwise win-rate matrix over the whole shared payoff
    table: live main and every historical snapshot (when they
    exist), and every historical snapshot with an origin-labelled
    row — logged through a custom Vega-Lite chart preset
    (jtwin/league-payoff-heatmap-v10, registered once via
    scripts/register_wandb_charts.py) instead of hijacking wandb's
    confusion-matrix preset: proper axis titles (player/opponent, not
    Actual/Predicted), a red/gold/green win-rate colour band per
    cell, and a text label per cell. The colour is a chain of
    condition/value tests on winrate with NO field bound directly to
    the colour channel — every version that bound colour to a table
    field (scale.range, scale.scheme+domain+clamp, a literal
    per-cell hex column with scale: null) rendered as either an
    unrelated colour or one flat colour for every cell in wandb's
    actual custom-chart panel, confirmed via wandb's own GraphQL API
    (spec stored correctly) and a neutral Vega-Lite renderer (spec
    renders correctly outside wandb) — so wandb's Vega2 runtime does
    not honour a field-bound colour channel here. Condition/value
    (no field) is the one pattern proven to render correctly (the
    text mark's black/white choice used exactly this pattern the
    whole time). Interactive (hover shows exact values), no
    matplotlib figure render on the train-loop thread, no image
    upload per log. row_idx/col_idx carry insertion order so the
    chart's ordinal axes sort by league structure rather than
    wandb's default alphabetical sort. A pair that has never actually
    played just shows the table's prior."""
    current = league.get_live(MAIN_KEY)
    others = winrate_tracked_opponents(league)
    if not others:
        return {}

    all_players = [current] + others
    labels = ["main (live)"] + [ref_label(p) for p in others]
    matrix = np.asarray(league.get_winrate((all_players, all_players)))

    table = wandb.Table(
        columns=["row", "row_idx", "col", "col_idx", "winrate"],
        data=[
            [row, i, col, j, float(matrix[i, j])]
            for i, row in enumerate(labels)
            for j, col in enumerate(labels)
        ],
    )
    chart = wandb.plot_table(
        "jtwin/league-payoff-heatmap-v10",
        table,
        fields={
            "row": "row",
            "row_idx": "row_idx",
            "col": "col",
            "col_idx": "col_idx",
            "winrate": "winrate",
        },
        string_fields={
            "title": "league payoff table (row beats column)"
        },
    )
    return {"league_winrate_heatmap": chart}
