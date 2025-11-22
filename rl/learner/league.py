import collections
import threading

import cloudpickle as pickle
import numpy as np

from rl.model.utils import ParamsContainer

_psfp_weightings = {
    "variance": lambda x: x * (1 - x),
    "linear": lambda x: 1 - x,
    "linear_capped": lambda x: np.minimum(0.5, 1 - x),
    "squared": lambda x: (1 - x) ** 2,
}


def pfsp(win_rates: np.ndarray, weighting: str = "squared") -> np.ndarray:
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

        self.players = {p.step_count: p for p in players}
        self.players[MAIN_KEY] = main_player
        self.wins = collections.defaultdict(lambda: 0)
        self.draws = collections.defaultdict(lambda: 0)
        self.losses = collections.defaultdict(lambda: 0)
        self.games = collections.defaultdict(lambda: 0)
        
        # Track controlled matches where teams are fixed
        self.controlled_wins = collections.defaultdict(lambda: 0)
        self.controlled_draws = collections.defaultdict(lambda: 0)
        self.controlled_losses = collections.defaultdict(lambda: 0)
        self.controlled_games = collections.defaultdict(lambda: 0)

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
                controlled_wins=self.controlled_wins,
                controlled_draws=self.controlled_draws,
                controlled_losses=self.controlled_losses,
                controlled_games=self.controlled_games,
                max_players=self.league_size,
                decay=self.decay,
            )
        )

    @classmethod
    def deserialize(cls, data: bytes) -> "League":
        state = pickle.loads(data)
        players = state["players"]
        main_player = players.pop(
            MAIN_KEY, max(players.values(), key=lambda p: p.step_count)
        )
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
        league.controlled_wins = state.get("controlled_wins", collections.defaultdict(lambda: 0))
        league.controlled_draws = state.get("controlled_draws", collections.defaultdict(lambda: 0))
        league.controlled_losses = state.get("controlled_losses", collections.defaultdict(lambda: 0))
        league.controlled_games = state.get("controlled_games", collections.defaultdict(lambda: 0))
        league.print_players()
        return league

    def get_latest_player(self) -> ParamsContainer:
        with self.lock:
            latest_step = max(self.players.keys())
            return self.players[latest_step]

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
        home = sender.step_count
        away = receiver.step_count

        numer = self.wins[home, away] + 0.5 * self.draws[home, away] + 0.5
        denom = self.games[home, away] + 1

        return numer / denom

    def _controlled_win_rate(self, sender: ParamsContainer, receiver: ParamsContainer) -> float:
        """Get win rate for controlled matches (fixed/swapped teams)."""
        home = sender.step_count
        away = receiver.step_count

        numer = self.controlled_wins[home, away] + 0.5 * self.controlled_draws[home, away] + 0.5
        denom = self.controlled_games[home, away] + 1

        return numer / denom

    def get_controlled_winrate(
        self,
        match: tuple[
            ParamsContainer | list[ParamsContainer],
            ParamsContainer | list[ParamsContainer],
        ],
    ) -> np.ndarray:
        """Get win rates for controlled matches where teams are fixed."""
        home, away = match

        if isinstance(home, ParamsContainer):
            home = [home]
        if isinstance(away, ParamsContainer):
            away = [away]

        win_rates = np.array([[self._controlled_win_rate(h, a) for a in away] for h in home])
        if win_rates.shape[0] == 1 or win_rates.shape[1] == 1:
            win_rates = win_rates.reshape(-1)

        return win_rates

    def add_player(self, player: ParamsContainer):
        # self.remove_weakest_players()
        self.players[player.step_count] = player

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
            home = sender.step_count
            away = receiver.step_count

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

    def update_controlled_payoff(
        self, sender: ParamsContainer, receiver: ParamsContainer, payoff: float
    ):
        """Update payoff for controlled matches where teams are fixed."""
        with self.lock:
            # Ignore updates for players that may have been removed
            home = sender.step_count
            away = receiver.step_count

            if home not in self.players or away not in self.players:
                return

            for stats in (self.controlled_games, self.controlled_wins, self.controlled_draws, self.controlled_losses):
                stats[home, away] *= 1 - self.decay
                stats[away, home] *= 1 - self.decay

            self.controlled_games[home, away] += 1
            self.controlled_games[away, home] += 1

            if payoff > 0:
                self.controlled_wins[home, away] += 1
                self.controlled_losses[away, home] += 1
            elif payoff == 0:
                self.controlled_draws[home, away] += 1
                self.controlled_draws[away, home] += 1
            else:
                self.controlled_losses[home, away] += 1
                self.controlled_wins[away, home] += 1

    def remove_weakest_players(self) -> ParamsContainer:
        with self.lock:
            historical_players = [v for k, v in self.players.items() if k != MAIN_KEY]
            win_rates = self.get_winrate((self.players[MAIN_KEY], historical_players))

            indices_to_remove = np.argwhere(win_rates >= 0.9).reshape(-1)
            for idx in indices_to_remove:
                weakest_player = historical_players[idx].step_count
                print("Removing player with step count: ", weakest_player)
                self.players.pop(weakest_player)

                keys_to_pop = []
                for home, away in self.wins.keys():
                    if weakest_player in (home, away):
                        keys_to_pop.append((home, away))
                for stats in (self.games, self.wins, self.draws, self.losses):
                    for key in keys_to_pop:
                        stats.pop(key)

    def get_main_player(self) -> ParamsContainer:
        with self.lock:
            return self.players[MAIN_KEY]


def main():
    print(pfsp(np.array([0.1, 0.5, 0.9, 0.9]), weighting="linear_capped"))


if __name__ == "__main__":
    main()
