import collections
import threading
from typing import Literal, NamedTuple

import cloudpickle as pickle
import numpy as np

from rl.learner import checkpoint
from rl.model.utils import ParamsContainer

_psfp_weightings = {
    "variance": lambda x: x * (1 - x),
    "linear": lambda x: 1 - x,
    "linear_capped": lambda x: np.minimum(0.5, 1 - x),
    "squared": lambda x: (1 - x) ** 2,
    "inverse_squared": lambda x: x**2,
}

PsfpWeighting = Literal[
    "variance", "linear", "linear_capped", "squared", "inverse_squared"
]


def pfsp(win_rates: np.ndarray, weighting: PsfpWeighting = "squared") -> np.ndarray:
    fn = _psfp_weightings[weighting]
    probs = fn(np.asarray(win_rates))
    norm = probs.sum()
    if norm < 1e-10:
        return np.ones_like(win_rates) / len(win_rates)
    return probs / norm


MAIN_KEY = -1


class PlayerRef(NamedTuple):
    """Lightweight, picklable handle to a historical opponent.

    Holds only metadata + a pointer to the on-disk snapshot folder; the param
    trees are loaded lazily (and cached) when the player is actually played.
    """

    step_count: int
    snapshot_dir: str
    player_frame_count: int
    builder_frame_count: int
    player_key: str = "params"
    builder_key: str = "params"


class League:
    """Disk-backed league.

    All historical opponents live on disk as sharded snapshots; the league
    keeps only ``PlayerRef`` metadata + win/draw/loss stats resident (tiny).
    Materialised param trees are held in a bounded LRU-style cache governed by
    a UCB retention rule: when the cache is full, evict the opponent the main
    player already beats reliably *and* has sampled enough — i.e. the one with
    the lowest ``(1 - main_winrate) + c * sqrt(ln N / (n + 1))``. Under-sampled
    or still-challenging opponents are kept hot.

    The ``main`` player is the live learner and always stays in memory.
    """

    def __init__(
        self,
        main_player: ParamsContainer | None,
        players: list[PlayerRef],
        league_size: int = 16,
        tau: float = 1e-3,
        cache_size: int = 16,
        ucb_c: float = 1.0,
    ):
        if tau <= 0:
            raise ValueError("Tau must be positive.")

        self.league_size = league_size
        self.decay = tau
        self.cache_size = cache_size
        self.ucb_c = ucb_c
        self.lock = threading.Lock()

        self.main = main_player
        self.players: dict[int, PlayerRef] = {p.step_count: p for p in players}
        self.wins = collections.defaultdict(lambda: 0)
        self.draws = collections.defaultdict(lambda: 0)
        self.losses = collections.defaultdict(lambda: 0)
        self.games = collections.defaultdict(lambda: 0)

        # step_count -> materialised ParamsContainer (bounded, UCB-evicted).
        self._cache: "collections.OrderedDict[int, ParamsContainer]" = (
            collections.OrderedDict()
        )

    def print_players(self):
        print("League initialized with num players: ", len(self.players.keys()))
        for k, v in self.players.items():
            print("  Player: ", k, repr(v))

    def serialize(self) -> bytes:
        # Only refs + stats — never param trees.
        return pickle.dumps(
            dict(
                players=self.players,
                wins=self.wins,
                draws=self.draws,
                losses=self.losses,
                games=self.games,
                max_players=self.league_size,
                decay=self.decay,
                cache_size=self.cache_size,
                ucb_c=self.ucb_c,
            )
        )

    @classmethod
    def deserialize(cls, data: bytes) -> "League":
        state = pickle.loads(data)
        players: dict[int, PlayerRef] = state["players"]
        league = cls(
            main_player=None,  # set by the caller via update_main_player
            players=list(players.values()),
            league_size=state["max_players"],
            tau=state["decay"],
            cache_size=state.get("cache_size", 16),
            ucb_c=state.get("ucb_c", 1.0),
        )
        league.wins = state["wins"]
        league.draws = state["draws"]
        league.losses = state["losses"]
        league.games = state["games"]
        league.print_players()
        return league

    # --- lazy materialisation + UCB-managed cache ---------------------------

    def materialize(self, ref: PlayerRef) -> ParamsContainer:
        """Return a ParamsContainer for ``ref``, loading from disk on a miss."""
        with self.lock:
            cached = self._cache.get(ref.step_count)
            if cached is not None:
                self._cache.move_to_end(ref.step_count)
                return cached

        # Load outside the lock — only the params files, never opt_state.
        player_params = checkpoint.load_component(
            ref.snapshot_dir, "player", ref.player_key
        )
        builder_params = checkpoint.load_component(
            ref.snapshot_dir, "builder", ref.builder_key
        )
        container = ParamsContainer(
            step_count=ref.step_count,
            player_frame_count=ref.player_frame_count,
            builder_frame_count=ref.builder_frame_count,
            player_params=player_params,
            builder_params=builder_params,
        )

        with self.lock:
            self._cache[ref.step_count] = container
            self._cache.move_to_end(ref.step_count)
            self._evict_if_needed()
        return container

    def _evict_if_needed(self):
        """Evict lowest-UCB opponents until the cache fits. Caller holds lock."""
        while len(self._cache) > self.cache_size:
            victim = min(self._cache.keys(), key=self._retention_score)
            del self._cache[victim]

    def _opponent_games(self, step: int) -> float:
        if self.main is None:
            return 0.0
        m = self.main.step_count
        return self.games[m, step] + self.games[step, m]

    def _retention_score(self, step: int) -> float:
        """Higher = keep hot. Challenge to main + UCB exploration bonus."""
        main_step = self.main.step_count if self.main is not None else MAIN_KEY
        challenge = 1.0 - self._win_rate_by_steps(main_step, step)
        n = self._opponent_games(step)
        total = sum(self._opponent_games(s) for s in self._cache) or 1.0
        bonus = self.ucb_c * float(np.sqrt(np.log(total + 1.0) / (n + 1.0)))
        return challenge + bonus

    # --- selection (metadata only) ------------------------------------------

    def get_latest_player(self) -> PlayerRef | None:
        with self.lock:
            if not self.players:
                return None
            return self.players[max(self.players.keys())]

    def get_winrate(
        self,
        match: tuple[
            ParamsContainer | PlayerRef | list,
            ParamsContainer | PlayerRef | list,
        ],
    ) -> np.ndarray:
        home, away = match

        if not isinstance(home, list):
            home = [home]
        if not isinstance(away, list):
            away = [away]

        win_rates = np.array([[self._win_rate(h, a) for a in away] for h in home])
        if win_rates.shape[0] == 1 or win_rates.shape[1] == 1:
            win_rates = win_rates.reshape(-1)

        return win_rates

    def _win_rate(self, sender: ParamsContainer, receiver: ParamsContainer) -> float:
        return self._win_rate_by_steps(sender.step_count, receiver.step_count)

    def _win_rate_by_steps(self, home: int, away: int) -> float:
        numer = self.wins[home, away] + 0.5 * self.draws[home, away] + 0.5
        denom = self.games[home, away] + 1
        return numer / denom

    # --- mutation ------------------------------------------------------------

    def add_player(self, ref: PlayerRef):
        with self.lock:
            self.players[ref.step_count] = ref

    def update_main_player(self, main_player: ParamsContainer):
        with self.lock:
            self.main = main_player

    def update_payoff(
        self, sender: ParamsContainer, receiver: ParamsContainer, payoff: float
    ):
        with self.lock:
            home = sender.step_count
            away = receiver.step_count

            # Ignore updates for players that may have been removed
            if home != MAIN_KEY and home not in self.players:
                return
            if away != MAIN_KEY and away not in self.players:
                return

            for stats in (self.games, self.wins, self.draws, self.losses):
                stats[home, away] *= 1 - self.decay
                stats[away, home] *= 1 - self.decay

            self.games[home, away] += 1
            self.games[away, home] += 1

            if payoff > 0:
                self.wins[home, away] += 1
                self.losses[away, home] += 1
            elif payoff == 0:
                self.draws[home, away] += 1
                self.draws[away, home] += 1
            else:
                self.losses[home, away] += 1
                self.wins[away, home] += 1

    def get_main_player(self) -> ParamsContainer:
        with self.lock:
            if self.main is None:
                raise RuntimeError("Main player has not been set on the league.")
            return self.main


def main():
    print(pfsp(np.array([0.1, 0.5, 0.9, 0.9]), weighting="inverse_squared"))


if __name__ == "__main__":
    main()
