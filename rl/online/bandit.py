"""Learning-progress bandit over the policy-target mixture.

A non-stationary multi-armed bandit that retunes the main v-trace lambda
online. Every window it measures how much the main player's strength rose
against the league's FROZEN snapshots — the only self-play-pure progress
signal available (mirror games are 50% by symmetry, and scripted
baselines must never steer training) — credits that gain to the arm that
trained the window, and picks the next arm by discounted UCB (Garivier &
Moulines: arm values drift as training progresses, so old evidence is
discounted away rather than averaged forever).

Strength is a Bradley-Terry rating fit on the league's per-pair
win/draw/loss table (draws count half; the table is per-pair EMA-decayed
by League.update_payoff, which lags the rating by roughly the decay
memory — the reward is smoothed, not point-in-time). BT is identified
only up to an offset, and the snapshot pool composition changes over
time, so consecutive fits are aligned by the mean rating shift of the
snapshots common to both fits — snapshots are frozen, so any common
shift is scale artefact, and what survives alignment is main's real
movement.

Before the pool is rateable (fewer than min_rated_opponents snapshots
with min_games_per_opponent effective games against main) there is no
reward: the bandit idles on its current arm and re-baselines at the next
valid fit, so a signal gap is never mis-credited to whichever arm
happened to be live.
"""

import pickle

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
    logs.update(_exploitability_logs(ratings, wins, draws, games))
    return logs


def _exploitability_logs(
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


class LambdaBandit:
    """Discounted-UCB bandit whose arms are main-target lambda values."""

    def __init__(
        self,
        arms: tuple[float, ...],
        default_arm: int,
        ucb_c: float,
        discount: float,
        min_games_per_opponent: float,
        min_rated_opponents: int,
    ):
        self.arms = tuple(arms)
        self.ucb_c = ucb_c
        self.discount = discount
        self.min_games_per_opponent = min_games_per_opponent
        self.min_rated_opponents = min_rated_opponents

        self.current_arm = default_arm
        self.counts = np.zeros(len(arms))
        self.sums = np.zeros(len(arms))
        # Baseline from the previous VALID fit: raw ratings (own scale)
        # and main's raw rating within them. None = no baseline; the next
        # valid fit only re-baselines and pays no reward.
        self.prev_ratings: dict[int, float] | None = None
        self.prev_main: float | None = None

    # --- persistence (piggybacks on the league pickle) ----------------------

    def serialize(self) -> bytes:
        return pickle.dumps(
            dict(
                arms=self.arms,
                current_arm=self.current_arm,
                counts=self.counts,
                sums=self.sums,
                prev_ratings=self.prev_ratings,
                prev_main=self.prev_main,
            )
        )

    def restore(self, data: bytes) -> None:
        state = pickle.loads(data)
        if tuple(state["arms"]) != self.arms:
            # Arm set changed across the restart — stale statistics would
            # be credited to the wrong lambdas, so start the controller
            # fresh (the rating baseline is arm-independent and kept).
            self.prev_ratings = state["prev_ratings"]
            self.prev_main = state["prev_main"]
            return
        self.current_arm = int(state["current_arm"])
        self.counts = np.asarray(state["counts"], dtype=float)
        self.sums = np.asarray(state["sums"], dtype=float)
        self.prev_ratings = state["prev_ratings"]
        self.prev_main = state["prev_main"]

    # --- one window boundary -------------------------------------------------

    def _rateable_snapshot_keys(self, league: League) -> list[int]:
        keys = []
        for s in league.players.keys():
            if s == MAIN_KEY:
                continue
            eff = league.games.get((MAIN_KEY, s), 0.0) + league.games.get(
                (s, MAIN_KEY), 0.0
            )
            if eff >= self.min_games_per_opponent:
                keys.append(s)
        return keys

    def update(self, league: League) -> dict[str, float]:
        """Close the current window: reward the live arm with the aligned
        BT-rating gain since the last valid fit, then pick the next arm.
        Called under no lock; takes the league lock only to copy stats.
        """
        with league.lock:
            snap_keys = self._rateable_snapshot_keys(league)
            wins = dict(league.wins)
            draws = dict(league.draws)
            games = dict(league.games)

        logs = {
            "bandit_lambda": self.arms[self.current_arm],
            "bandit_arm": float(self.current_arm),
            "bandit_rating_valid": 0.0,
        }

        if len(snap_keys) < self.min_rated_opponents:
            # Unrateable pool (cold start / pure mirror): no reward exists.
            # Drop the baseline so the eventual first valid window is not
            # credited with progress made during the gap.
            self.prev_ratings = None
            self.prev_main = None
            return self._log_arm_stats(logs)

        ratings = bt_ratings([MAIN_KEY] + snap_keys, wins, draws, games)
        if MAIN_KEY not in ratings:
            self.prev_ratings = None
            self.prev_main = None
            return self._log_arm_stats(logs)

        logs["bandit_rating_valid"] = 1.0
        logs["bandit_bt_rating"] = ratings[MAIN_KEY]
        logs["bandit_rated_opponents"] = float(len(ratings) - 1)

        if self.prev_ratings is not None and self.prev_main is not None:
            common = [s for s in ratings if s != MAIN_KEY and s in self.prev_ratings]
            if common:
                # Frozen snapshots move only through scale drift; the mean
                # shift over common snapshots is that drift.
                offset = float(
                    np.mean([self.prev_ratings[s] - ratings[s] for s in common])
                )
                reward = (ratings[MAIN_KEY] + offset) - self.prev_main
                logs["bandit_window_reward"] = reward

                self.counts *= self.discount
                self.sums *= self.discount
                self.counts[self.current_arm] += 1.0
                self.sums[self.current_arm] += reward

        # Re-baseline in this fit's own scale.
        self.prev_ratings = {k: v for k, v in ratings.items() if k != MAIN_KEY}
        self.prev_main = ratings[MAIN_KEY]

        # Discounted UCB: untried (or fully decayed) arms first.
        untried = np.flatnonzero(self.counts < 1e-6)
        if untried.size:
            self.current_arm = int(untried[0])
        else:
            total = self.counts.sum()
            ucb = self.sums / self.counts + self.ucb_c * np.sqrt(
                np.log(total + 1.0) / self.counts
            )
            self.current_arm = int(np.argmax(ucb))

        logs["bandit_next_lambda"] = self.arms[self.current_arm]
        return self._log_arm_stats(logs)

    def _log_arm_stats(self, logs: dict[str, float]) -> dict[str, float]:
        """Per-arm stats AFTER this window's reward and selection, so the
        logged counts reflect the update the same row reports."""
        for i in range(len(self.arms)):
            logs[f"bandit_arm{i}_count"] = float(self.counts[i])
            logs[f"bandit_arm{i}_mean"] = float(
                self.sums[i] / self.counts[i] if self.counts[i] > 0 else 0.0
            )
        return logs
