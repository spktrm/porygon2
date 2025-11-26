import collections
import threading
from typing import Literal

import cloudpickle as pickle
import numpy as np

from rl.model.utils import ParamsContainer

_psfp_weightings = {
    "variance": lambda x: x * (1 - x),
    "linear": lambda x: 1 - x,
    "linear_capped": lambda x: np.minimum(0.5, 1 - x),
    "squared": lambda x: (1 - x) ** 2,
}


PsfpWeighting = Literal["variance", "linear", "linear_capped", "squared"]


def pfsp(win_rates: np.ndarray, weighting: PsfpWeighting = "squared") -> np.ndarray:
    fn = _psfp_weightings[weighting]
    probs = fn(np.asarray(win_rates))
    norm = probs.sum()
    if norm < 1e-10:
        return np.ones_like(win_rates) / len(win_rates)
    return probs / norm


MAIN_KEY = -1


class League:
    def __init__(
        self,
        main_player: ParamsContainer,
        players: list[ParamsContainer],
        league_size: int = 16,
        tau: float = 1e-3,
    ):
        if tau <= 0:
            raise ValueError("Tau must be positive.")

        self.league_size = league_size
        self.decay = tau
        self.lock = threading.Lock()

        self.players = {p.get_key(): p for p in players}
        self.players[MAIN_KEY] = main_player

        self.wins = collections.defaultdict(lambda: 0)
        self.draws = collections.defaultdict(lambda: 0)
        self.losses = collections.defaultdict(lambda: 0)
        self.games = collections.defaultdict(lambda: 0)

    def print_players(self):
        print("League initialized with num players: ", len(self.players.keys()))
        for k, v in self.players.items():
            print("  Player: ", k, repr(v))

    def serialize(self) -> bytes:
        return pickle.dumps(
            dict(
                players=self.players,
                wins=self.wins,
                draws=self.draws,
                losses=self.losses,
                games=self.games,
                max_players=self.league_size,
                decay=self.decay,
            )
        )

    @classmethod
    def deserialize(cls, data: bytes) -> "League":
        state = pickle.loads(data)
        players: dict[int, ParamsContainer] = state["players"]
        main_player = players.pop(MAIN_KEY, players)
        league = cls(
            main_player,
            list(players.values()),
            state["max_players"],
            state["decay"],
        )
        league.wins = state["wins"]
        league.draws = state["draws"]
        league.losses = state["losses"]
        league.games = state["games"]
        league.print_players()
        return league

    def get_latest_player(self) -> ParamsContainer:
        with self.lock:
            latest_step = max(self.players.keys())
            return self.players[latest_step]

    def get_main_player(self) -> ParamsContainer:
        with self.lock:
            return self.players[MAIN_KEY]

    def _pfsp_branch(
        self, exclude_main: bool = False, weighting: PsfpWeighting = "squared"
    ) -> ParamsContainer:
        keys = list(self.players.keys())
        if exclude_main:
            keys = [k for k in keys if k != MAIN_KEY]
        win_rates = np.array(
            [self._win_rate(self.players[MAIN_KEY], self.players[k]) for k in keys]
        )
        probs = pfsp(win_rates, weighting=weighting)
        chosen_key = np.random.choice(keys, p=probs)
        return self.players[chosen_key]

    def get_opponent(
        self, exclude_main: bool = False, weighting: PsfpWeighting = "squared"
    ) -> tuple[ParamsContainer, bool]:
        coin_toss = np.random.random()

        if coin_toss < 0.5:
            with self.lock:
                return self._pfsp_branch(exclude_main, weighting), False

        return self.get_main_player(), True

    def _win_rate(self, sender: ParamsContainer, receiver: ParamsContainer) -> float:
        home = sender.get_key()
        away = receiver.get_key()

        numer = self.wins[home, away] + 0.5 * self.draws[home, away] + 0.5
        denom = self.games[home, away] + 1

        return numer / denom

    def get_winrate(
        self,
        match: tuple[
            ParamsContainer | list[ParamsContainer],
            ParamsContainer | list[ParamsContainer],
        ],
    ) -> np.ndarray:
        home, away = match

        if isinstance(home, ParamsContainer):
            home = [home]
        if isinstance(away, ParamsContainer):
            away = [away]

        win_rates = np.array([[self._win_rate(h, a) for a in away] for h in home])
        if win_rates.shape[0] == 1 or win_rates.shape[1] == 1:
            win_rates = win_rates.reshape(-1)

        return win_rates

    def add_player(self, player: ParamsContainer):
        self.players[player.get_key()] = player

    def update_player(self, key: int, player: ParamsContainer):
        with self.lock:
            self.players[key] = player

    def update_main_player(self, main_player: ParamsContainer):
        self.update_player(MAIN_KEY, main_player)

    def update_payoff(
        self, sender: ParamsContainer, receiver: ParamsContainer, payoff: float
    ):
        with self.lock:
            # Ignore updates for players that may have been removed
            home = sender.get_key()
            away = receiver.get_key()

            if home not in self.players or away not in self.players:
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


def main():
    print(pfsp(np.array([0.1, 0.5, 0.9, 0.9]), weighting="linear_capped"))


if __name__ == "__main__":
    main()
