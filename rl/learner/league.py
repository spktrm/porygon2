import collections
import uuid
from typing import Any

import cloudpickle
import jax
import numpy as np

from rl.model.utils import ParamsContainer


MatchRequest = tuple[int, bool]  # (opponent_id, is_historical)


def pfsp(win_rates: np.ndarray, weighting: str = "linear") -> np.ndarray:
    """Prioritized Fictitious Self-Play weighting."""
    weightings = {
        "variance": lambda x: x * (1 - x),
        "linear": lambda x: 1 - x,
        "linear_capped": lambda x: np.minimum(0.5, 1 - x),
        "squared": lambda x: (1 - x) ** 2,
    }
    fn = weightings.get(weighting, weightings["linear"])
    probs = fn(np.asarray(win_rates))
    norm = probs.sum()
    if norm < 1e-10:
        return np.ones_like(win_rates) / len(win_rates)
    return probs / norm


def remove_monotonic_suffix(
    win_rates: np.ndarray, lora_ids: list[int]
) -> tuple[np.ndarray, list[int]]:
    """Truncates history if win rates stop improving."""
    if not win_rates.size:
        return win_rates, lora_ids

    for i in range(len(win_rates) - 1, 0, -1):
        if win_rates[i - 1] < win_rates[i]:
            return win_rates[: i + 1], lora_ids[: i + 1]

    return np.array([]), []


class Payoff:
    """
    Tracks win rates using String IDs.
    Lightweight and safe to pickle.
    """

    def __init__(self, decay: float = 0.99):
        self._player_metadata: dict[int, dict[str, Any]] = {}
        self._wins = collections.defaultdict(int)
        self._draws = collections.defaultdict(int)
        self._losses = collections.defaultdict(int)
        self._games = collections.defaultdict(int)
        self._decay = decay

    def _win_rate(self, _home_id: int, _away_id: int) -> float:
        if self._games[_home_id, _away_id] == 0:
            return 0.5
        return (
            self._wins[_home_id, _away_id] + 0.5 * self._draws[_home_id, _away_id]
        ) / self._games[_home_id, _away_id]

    def __getitem__(self, match: tuple[int, int | list[int]]) -> np.ndarray:
        home_id, away_ids = match

        if isinstance(away_ids, str):
            away_ids = [away_ids]

        win_rates = np.array([[self._win_rate(home_id, a_id) for a_id in away_ids]])
        if win_rates.shape[0] == 1 or win_rates.shape[1] == 1:
            win_rates = win_rates.reshape(-1)
        return win_rates

    def update(self, home_id: int, away_id: int, result: str) -> None:
        for stats in (self._games, self._wins, self._draws, self._losses):
            stats[home_id, away_id] *= self._decay
            stats[away_id, home_id] *= self._decay

        self._games[home_id, away_id] += 1
        self._games[away_id, home_id] += 1

        if result == "win":
            self._wins[home_id, away_id] += 1
            self._losses[away_id, home_id] += 1
        elif result == "draw":
            self._draws[home_id, away_id] += 1
            self._draws[away_id, home_id] += 1
        else:
            self._wins[away_id, home_id] += 1
            self._losses[home_id, away_id] += 1

    def add_player(
        self, lora_id: int, type_name: str, parent_lora_id: None | int = None
    ) -> None:
        self._player_metadata[lora_id] = {
            "type": type_name,
            "parent_lora_id": parent_lora_id,
        }

    def get_potential_opponents(
        self, type_filter: None | str = None, parent_filter: None | int = None
    ) -> list[int]:
        candidates = []
        for pid, meta in self._player_metadata.items():
            if type_filter and meta["type"] != type_filter:
                continue
            if parent_filter and meta["parent_lora_id"] != parent_filter:
                continue
            candidates.append(pid)
        return candidates


class Player:
    def __init__(
        self,
        payoff: Payoff,
        lora_id: int,
        parent_lora_id: int | None = None,
    ):
        self._payoff = payoff
        self._lora_id = lora_id
        self._parent_lora_id = parent_lora_id

    @property
    def lora_id(self) -> int:
        return self._lora_id

    @property
    def parent_lora_id(self) -> int | None:
        return self._parent_lora_id

    def ready_to_checkpoint(self) -> bool:
        return False

    def checkpoint(self) -> "Player":
        raise NotImplementedError

    def get_match(self) -> tuple[None | int, bool]:
        raise NotImplementedError


class HistoricalPlayer(Player):
    def __init__(
        self,
        payoff: Payoff,
        lora_id: int,
        parent_lora_id: int | None = None,
    ):
        super().__init__(payoff, lora_id, parent_lora_id)

    def get_match(self) -> tuple[None | int, bool]:
        raise ValueError("Historical players do not request matches.")


class MainPlayer(Player):
    def __init__(self, lora_id: int, payoff: Payoff):
        super().__init__(payoff, lora_id)

    def _pfsp_branch(self) -> MatchRequest:
        historical_ids = self._payoff.get_potential_opponents(
            type_filter=HistoricalPlayer.__name__
        )
        if not historical_ids:
            return None, True
        win_rates = self._payoff[self.lora_id, historical_ids]
        return (
            np.random.choice(historical_ids, p=pfsp(win_rates, weighting="squared")),
            True,
        )

    def _selfplay_branch(self, opponent_lora_id: int) -> MatchRequest:
        if self._payoff[self.lora_id, opponent_lora_id] > 0.3:
            return opponent_lora_id, False

        historical_ids = self._payoff.get_potential_opponents(
            type_filter=HistoricalPlayer.__name__, parent_filter=opponent_lora_id
        )
        if not historical_ids:
            return opponent_lora_id, False

        win_rates = self._payoff[self.player_id, historical_ids]
        return (
            np.random.choice(historical_ids, p=pfsp(win_rates, weighting="variance")),
            True,
        )

    def _verification_branch(self, opponent_lora_id: int) -> None | MatchRequest:
        # Check MainExploiters
        exploiter_ids = self._payoff.get_potential_opponents(
            type_filter=MainExploiter.__name__
        )
        exp_historical = []
        for exploiter_id in exploiter_ids:
            exp_historical.extend(
                self._payoff.get_potential_opponents(
                    type_filter=HistoricalPlayer.__name__,
                    parent_filter=exploiter_id,
                )
            )
        if exp_historical:
            win_rates = self._payoff[self.lora_id, exp_historical]
            if len(win_rates) and win_rates.min() < 0.3:
                return (
                    np.random.choice(
                        exp_historical, p=pfsp(win_rates, weighting="squared")
                    ),
                    True,
                )

        # Check Forgetting
        historical_ids = self._payoff.get_potential_opponents(
            type_filter=HistoricalPlayer.__name__, parent_filter=opponent_lora_id
        )
        if historical_ids:
            win_rates = self._payoff[self.lora_id, historical_ids]
            win_rates, historical_ids = remove_monotonic_suffix(
                win_rates, historical_ids
            )
            if len(win_rates) and win_rates.min() < 0.7:
                return (
                    np.random.choice(
                        historical_ids, p=pfsp(win_rates, weighting="squared")
                    ),
                    True,
                )
        return None

    def get_match(self) -> MatchRequest:
        coin_toss = np.random.random()
        if coin_toss < 0.5:
            match = self._pfsp_branch()
            if match and match[0]:
                return match

        league_ids = self._payoff.get_potential_opponents()
        opponent_id = np.random.choice(league_ids)

        if coin_toss < 0.5 + 0.15:
            req = self._verification_branch(opponent_id)
            if req:
                return req

        return self._selfplay_branch(opponent_id)

    def ready_to_checkpoint(self) -> bool:
        historical = self._payoff.get_potential_opponents(
            type_filter=HistoricalPlayer.__name__
        )
        win_rates = self._payoff[self.player_id, historical]
        return win_rates.min() > 0.7


class MainExploiter(Player):
    def __init__(self, lora_id: int, payoff: Payoff):
        super().__init__(payoff, lora_id)

    def get_match(self) -> MatchRequest:
        main_agent_ids = self._payoff.get_potential_opponents(
            type_filter=MainPlayer.__name__
        )
        opponent_lora_id = np.random.choice(main_agent_ids)

        if self._payoff[self.lora_id, opponent_lora_id] > 0.1:
            return opponent_lora_id, True

        historical_ids = self._payoff.get_potential_opponents(
            type_filter=HistoricalPlayer.__name__, parent_filter=opponent_lora_id
        )
        if not historical_ids:
            return opponent_lora_id, True

        win_rates = self._payoff[self.lora_id, historical_ids]
        return (
            np.random.choice(historical_ids, p=pfsp(win_rates, weighting="variance")),
            True,
        )

    def ready_to_checkpoint(self) -> bool:
        main_agents = self._payoff.get_potential_opponents(
            type_filter=MainPlayer.__name__
        )
        win_rates = self._payoff[self.lora_id, main_agents]
        return win_rates.min() > 0.7


class LeagueExploiter(Player):
    def __init__(self, lora_id: int, payoff: Payoff):
        super().__init__(payoff, lora_id)

    def get_match(self) -> MatchRequest:
        historical = self._payoff.get_potential_opponents(
            type_filter=HistoricalPlayer.__name__
        )
        if not historical:
            return None, True  # Wait for history

        win_rates = self._payoff[self.lora_id, historical]
        return (
            np.random.choice(historical, p=pfsp(win_rates, weighting="linear_capped")),
            True,
        )

    def ready_to_checkpoint(self) -> bool:
        historical = self._payoff.get_potential_opponents(
            type_filter=HistoricalPlayer.__name__
        )
        win_rates = self._payoff[self.lora_id, historical]
        return win_rates.min() > 0.7


class League:
    def __init__(self):
        self._payoff = Payoff()
        self._learning_agents: list[Player] = []
        self._player_registry: dict[int, Player] = {}

    def add_player(self, player: Player) -> None:
        # Register metadata in Payoff
        self._payoff.add_player(
            player.lora_id, type(player).__name__, player.parent_lora_id
        )

        # Register object in League
        self._player_registry[player.lora_id] = player

        # Add to training loop if it's not a static historical player
        if not isinstance(player, HistoricalPlayer):
            self._learning_agents.append(player)

    def update_result(self, home_lora_id: int, away_lora_id: int, result: str) -> None:
        self._payoff.update(home_lora_id, away_lora_id, result)
