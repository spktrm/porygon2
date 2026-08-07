"""Bradley-Terry rating telemetry for the league's frozen snapshot pool.

Strength is a Bradley-Terry rating fit on the league's per-pair
win/draw/loss table (draws count half; the table is per-pair EMA-decayed
by League.update_payoff, which lags the rating by roughly the decay
memory — the signal is smoothed, not point-in-time). BT is identified
only up to an offset, and the snapshot pool composition changes over
time, so consecutive fits would need aligning by the mean rating shift of
snapshots common to both fits if compared window-to-window — snapshots
are frozen, so any common shift is scale artefact.

Before the pool is rateable (fewer than min_rated_opponents snapshots
with min_games_per_opponent effective games against main) there is no
signal: rating_logs reports bandit_rating_valid=0 rather than a number.

Historical note: this module used to also hold LambdaBandit, a
discounted-UCB bandit that retuned the main v-trace lambda from the
per-window aligned rating gain. Retired in favour of the lambda
gap-controller (rl/online/controllers.py) plus the exploitability
controller: the bandit paid an extra exploration tax (it must sometimes
hold an arm it suspects is worse just to keep the uncertainty estimate
honest) on top of the rating signal's own latency (hundreds of games per
point), which made it slower to react than either replacement. The BT
fit and its telemetry below remain — they are the only self-play-pure
absolute-strength signal (mirror games are 50% by symmetry, and scripted
baselines must never steer training), logged every window regardless of
which controller is driving.
"""

import numpy as np

from rl.online.league import MAIN_KEY, League


def bt_ratings(
    keys: list[int],
    wins: dict,
    draws: dict,
    games: dict,
    num_iters: int = 200,
) -> dict[int, float]:
    """Bradley-Terry ratings (natural-log units) from a pairwise table.

    keys: player ids; entries of wins/draws/games are keyed (i, j).
    Draws count as half a win each way. One pseudo-draw is added to every
    observed pairing so an undefeated player gets a finite rating. Fitted
    with Hunter's MM updates; returns {} if fewer than two players have
    any games.
    """
    idx = {k: n for n, k in enumerate(keys)}
    size = len(keys)
    w = np.zeros((size, size))
    n = np.zeros((size, size))
    for i in keys:
        for j in keys:
            if i == j:
                continue
            nij = 0.5 * (games.get((i, j), 0.0) + games.get((j, i), 0.0))
            if nij <= 0:
                continue
            wij = wins.get((i, j), 0.0) + 0.5 * draws.get((i, j), 0.0)
            # One pseudo-draw per observed pairing (split across the two
            # directed entries this loop visits).
            w[idx[i], idx[j]] = wij + 0.25
            n[idx[i], idx[j]] = nij + 0.5

    played = (n.sum(axis=1) > 0.5) | (n.sum(axis=0) > 0.5)
    if played.sum() < 2:
        return {}

    pi = np.ones(size)
    w_row = w.sum(axis=1)
    for _ in range(num_iters):
        denom = np.where(n > 0, n / np.maximum(pi[:, None] + pi[None, :], 1e-12), 0.0)
        pi_new = w_row / np.maximum(denom.sum(axis=1), 1e-12)
        pi_new = np.where(played, pi_new, 1.0)
        # Renormalise each sweep for numerical stability; the caller
        # aligns scales across fits anyway.
        pi = pi_new / np.exp(np.log(np.maximum(pi_new, 1e-12)).mean())

    return {k: float(np.log(max(pi[idx[k]], 1e-12))) for k in keys if played[idx[k]]}


def rating_logs(
    league: League,
    min_games_per_opponent: float,
    min_rated_opponents: int,
) -> dict[str, float]:
    """BT-rating telemetry without any arm machinery — used when the
    bandit is disabled so the strength-vs-frozen-pool signal (the only
    self-play-pure absolute progress measure) stays on the dashboard."""
    with league.lock:
        snap_keys = [
            s
            for s in league.players.keys()
            if s != MAIN_KEY
            and league.games.get((MAIN_KEY, s), 0.0)
            + league.games.get((s, MAIN_KEY), 0.0)
            >= min_games_per_opponent
        ]
        wins = dict(league.wins)
        draws = dict(league.draws)
        games = dict(league.games)

    if len(snap_keys) < min_rated_opponents:
        return {"bandit_rating_valid": 0.0}
    ratings = bt_ratings([MAIN_KEY] + snap_keys, wins, draws, games)
    if MAIN_KEY not in ratings:
        return {"bandit_rating_valid": 0.0}
    logs = {
        "bandit_rating_valid": 1.0,
        "bandit_bt_rating": ratings[MAIN_KEY],
        "bandit_rated_opponents": float(len(ratings) - 1),
    }
    logs.update(_rating_exploitability_logs(ratings, wins, draws, games))
    return logs


def _rating_exploitability_logs(
    ratings: dict[int, float], wins: dict, draws: dict, games: dict
) -> dict[str, float]:
    """Auditor metrics for the adaptivity controller — logged, never
    controlled on (they need hundreds of games per point, far slower than
    the covariance loop).

    An under-regularised policy is EXPLOITABLE, which shows up two ways
    in the payoff table: main's worst matchup drifts toward (or below)
    even while its mean stays healthy, and the population stops being a
    transitive strength ladder. The BT model assumes transitivity, so the
    mean absolute gap between its predicted winrates and the observed
    ones is a non-transitivity (rock-paper-scissors) index.
    """
    obs, pred = [], []
    for s, rating in ratings.items():
        if s == MAIN_KEY:
            continue
        n = games.get((MAIN_KEY, s), 0.0) + games.get((s, MAIN_KEY), 0.0)
        if n <= 0:
            continue
        w = wins.get((MAIN_KEY, s), 0.0) + 0.5 * draws.get((MAIN_KEY, s), 0.0)
        obs.append(w / (0.5 * n))
        # Bradley-Terry: P(main beats s) = pi_main / (pi_main + pi_s).
        pred.append(1.0 / (1.0 + float(np.exp(rating - ratings[MAIN_KEY]))))
    if not obs:
        return {}
    obs_arr = np.asarray(obs, dtype=float)
    return {
        "league_main_winrate_min": float(obs_arr.min()),
        "league_main_winrate_mean": float(obs_arr.mean()),
        # Healthy dominance keeps this small; a big spread means one
        # opponent has found a hole.
        "league_winrate_spread": float(obs_arr.mean() - obs_arr.min()),
        "league_bt_residual": float(
            np.abs(obs_arr - np.asarray(pred, dtype=float)).mean()
        ),
    }

