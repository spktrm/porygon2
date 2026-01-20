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


LATEST_KEY = -1


def pfsp(win_rates: np.ndarray, weighting: PsfpWeighting = "squared") -> np.ndarray:
    fn = _psfp_weightings[weighting]
    probs = fn(np.asarray(win_rates))
    norm = probs.sum()
    if norm < 1e-10:
        return np.ones_like(win_rates) / len(win_rates)
    return probs / norm


class League:
    def __init__(
        self,
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

        self.wins = collections.defaultdict(lambda: 0)
        self.draws = collections.defaultdict(lambda: 0)
        self.losses = collections.defaultdict(lambda: 0)
        self.games = collections.defaultdict(lambda: 0)

    def print_players(self):
        print("League initialized with num players: ", len(self.players.keys()))
        for player_key, player in self.players.items():
            print("  Player: ", player_key, player.get_key())

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
        league = cls(
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

    def get_latest_player(self):
        with self.lock:
            if not self.players:
                raise ValueError("No players in the league.")
            latest_player = max(self.players.values(), key=lambda p: p.step_count)
            return latest_player

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

    def _win_rate(self, sender: ParamsContainer, receiver: ParamsContainer) -> float:
        home = sender.get_key()
        away = receiver.get_key()

        numer = self.wins[home, away] + 0.5 * self.draws[home, away] + 0.5
        denom = self.games[home, away] + 1

        return numer / denom

    def add_player(self, player: ParamsContainer):
        # self.remove_weakest_players()
        self.players[player.get_key()] = player

    def update_player(self, key: str, player: ParamsContainer):
        with self.lock:
            self.players[key] = player

    def get_player(self, key: str) -> ParamsContainer:
        with self.lock:
            player = self.players.get(key)
            if player is None:
                raise KeyError(
                    f"Player with key {key} not found in the {list(self.players.keys())}."
                )
            return player

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
    print(pfsp(np.array([0.1, 0.5, 0.9, 0.9]), weighting="squared"))


if __name__ == "__main__":
    main()
