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
        league.print_players()
        return league

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

        if self.games[home, away] == 0:
            return 0.5

        return (self.wins[home, away] + -0.5 * self.draws[home, away]) / self.games[
            home, away
        ]

    def add_player(self, player: ParamsContainer):
        while len(self.players) >= self.league_size:
            self.remove_weakest_player()
        self.players[player.step_count] = player

    def get_h2h_winrate(self):
        num_players = len(self.players)
        if num_players == 0:
            return []

        # Build the win-rate matrix efficiently
        win_rates = np.full((num_players, num_players), 0.5)
        player_to_idx = {player: idx for idx, player in enumerate(self.players.keys())}
        idx_to_player = {idx: player for player, idx in player_to_idx.items()}

        seen = set()
        if len(self.games) > 0:
            for (p1, p2), games_played in self.games.items():
                seen_matchups = [(p1, p2), (p2, p1)]
                if any(m in seen for m in seen_matchups):
                    continue
                seen.update(seen_matchups)

                if games_played > 0:
                    idx1 = player_to_idx.get(p1)
                    idx2 = player_to_idx.get(p2)

                    if idx1 is not None and idx2 is not None:
                        winrate = (
                            self.wins[p1, p2] + 0.5 * self.draws[p1, p2]
                        ) / games_played
                        win_rates[idx1, idx2] = winrate
                        win_rates[idx2, idx1] = winrate

        return win_rates, player_to_idx, idx_to_player

    def get_player_rankings(self) -> tuple[np.ndarray, dict[int, int]]:
        """
        Calculates the average win rate for all players.

        Returns:
            A list of (score, player) tuples, sorted from strongest to weakest.
        """
        num_players = len(self.players)
        winrates, _, idx_to_player = self.get_h2h_winrate()

        # Calculate average scores
        np.fill_diagonal(winrates, 0)
        avg_scores = np.sum(winrates, axis=-1) / max(1, num_players - 1)

        # Handle case where a player has played no games (avg_score will be NaN)
        avg_scores = np.nan_to_num(avg_scores, nan=0.5)

        return avg_scores, idx_to_player

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

            self.games[home, away] += 1

            if payoff > 0:
                self.wins[home, away] += payoff
            elif payoff == 0:
                self.draws[home, away] += payoff
            else:
                self.losses[home, away] += payoff

    def remove_weakest_player(self) -> ParamsContainer:
        with self.lock:
            win_rates, player_to_idx, idx_to_player = self.get_h2h_winrate()

            while True:
                # Remove player with worst win rate v main player
                # This is the player that the main agent has the highest win rate against
                weakest_idx = np.argmax(win_rates[player_to_idx[MAIN_KEY]]).item()
                weakest_player = idx_to_player[weakest_idx]
                if weakest_player == MAIN_KEY:
                    win_rates[weakest_idx] = -np.inf
                    continue
                break

            keys_to_pop = []
            for home, away in self.wins.keys():
                if weakest_player in (home, away):
                    keys_to_pop.append((home, away))
            for stats in (self.games, self.wins, self.draws, self.losses):
                for key in keys_to_pop:
                    stats.pop(key)

            print("Removing player with step count: ", weakest_player)
            return self.players.pop(weakest_player)

    def get_player(self, key: int) -> ParamsContainer:
        with self.lock:
            return self.players[key]

    def get_main_player(self) -> ParamsContainer:
        with self.lock:
            return self.players[MAIN_KEY]

    def get_best_player(self) -> ParamsContainer:
        with self.lock:
            scores, idx_to_player = self.get_player_rankings()
            strongest_idx = np.argmax(scores).item()
            strongest_player = idx_to_player[strongest_idx]
            return self.players[strongest_player]

    def sample_player(self) -> ParamsContainer:
        with self.lock:
            historical = list(self.players.keys())
            # Sample a random player
            return self.players[np.random.choice(historical)]

    def sample_opponent(self, player: ParamsContainer) -> ParamsContainer:
        with self.lock:
            historical_keys = list(self.players.keys())
            historical = [self.players[k] for k in historical_keys]
            win_rates = self.get_winrate((player, historical))

            # Sample the player with worst win rate against the given player
            return self.players[
                np.random.choice(
                    historical_keys, p=pfsp(win_rates, weighting="squared")
                )
            ]


def main():
    print(pfsp(np.array([0.1, 0.5, 0.9]), weighting="linear_capped"))


if __name__ == "__main__":
    main()
