from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from rl.environment.data import STOI


def _esp_k(odds: np.ndarray, k: int) -> float:
    """
    Compute the k-th elementary symmetric polynomial e_k(odds) with DP.
    odds: shape (m,)
    Returns a scalar e_k.
    """
    # DP over degrees using 1D array: E[d] = e_d of processed prefix
    E = np.zeros(k + 1, dtype=np.float64)
    E[0] = 1.0
    for o in odds:
        # update descending to avoid overwriting dependencies
        upto = min(k, np.count_nonzero(E) - 1 + 1)  # small micro-opt
        for d in range(upto, 0, -1):
            E[d] += o * E[d - 1]
    return float(E[k])


def _esp_suffix_table(odds: np.ndarray, k: int) -> np.ndarray:
    """
    Build suffix ESP table S[i, d] = e_d(odds[i:]) for d=0..k and i=0..m.
    Useful for conditional Poisson sampling decisions.
    """
    m = odds.shape[0]
    S = np.zeros((m + 1, k + 1), dtype=np.float64)
    S[m, 0] = 1.0  # e_0 of empty set is 1
    for i in range(m - 1, -1, -1):
        S[i, 0] = 1.0
        oi = odds[i]
        # e_d on suffix i uses S[i+1]
        # S[i, d] = S[i+1, d] + oi * S[i+1, d-1]
        for d in range(1, k + 1):
            S[i, d] = S[i + 1, d] + oi * S[i + 1, d - 1]
    return S


def _sample_unordered_k_from_marginals(
    moves: List[str], probs: np.ndarray, k: int, rng: np.random.Generator
) -> Tuple[Tuple[str, ...], float]:
    """
    Conditional independent-Bernoulli model:
    - Let odds_i = p_i / (1 - p_i)
    - P(S) ∝ ∏_{i∈S} odds_i, with |S|=k.
    - Exact normalization is e_k(odds).
    Returns (sorted_moves_tuple, exact_probability_under_moves_model).
    """
    assert probs.ndim == 1 and len(moves) == probs.shape[0]
    assert 0 < k <= len(moves)

    # Clip for numeric safety
    p = np.clip(probs.astype(np.float64), 1e-12, 1 - 1e-12)
    odds = p / (1.0 - p)

    # Precompute suffix ESP for sampling-by-conditioning
    S = _esp_suffix_table(odds, k)
    e_k_total = S[0, k]

    chosen_idx = []
    remain_to_pick = k
    m = len(moves)

    for i in range(m):
        if remain_to_pick == 0:
            break
        # probability to include i given need=remain_to_pick, using:
        # P(include i | need r) = (odds[i] * e_{r-1}(odds[i+1:])) / e_r(odds[i:])
        num = odds[i] * S[i + 1, remain_to_pick - 1]
        den = S[i, remain_to_pick]
        if den <= 0:
            # numeric fallback: if den is zero (shouldn't happen with clamped probs), skip
            take_prob = 0.0
        else:
            take_prob = num / den

        if rng.random() < take_prob:
            chosen_idx.append(i)
            remain_to_pick -= 1

    # If numerics under-select (extremely rare), fill greedily among remaining highest odds
    if remain_to_pick > 0:
        remaining = [j for j in range(m) if j not in chosen_idx]
        # sort by odds desc
        remaining.sort(key=lambda j: odds[j], reverse=True)
        chosen_idx.extend(remaining[:remain_to_pick])

    chosen_idx = tuple(chosen_idx)
    chosen_moves = tuple(sorted([moves[i] for i in chosen_idx]))
    # exact unordered probability for chosen set under the model:
    # P(S) = (prod odds_i) / e_k_total
    logp = float(np.sum(np.log(odds[list(chosen_idx)]))) - math.log(e_k_total)
    return chosen_moves, math.exp(logp)


# -----------------------------
# Data containers
# -----------------------------


@dataclass(frozen=True)
class CategoricalSpace:
    names: np.ndarray  # shape (n,)
    probs: np.ndarray  # shape (n,), sum ≈ 1

    @staticmethod
    def from_dict(d: Dict[str, float]) -> "CategoricalSpace":
        items = list(d.items())
        names = np.array([k for k, _ in items], dtype=object)
        probs = np.array([v for _, v in items], dtype=np.float64)
        # Normalize defensively
        s = probs.sum()
        if s <= 0:
            raise ValueError("All probabilities are zero in a categorical space.")
        probs = probs / s
        # Sort descending for efficient top-mass truncation
        order = np.argsort(-probs)
        return CategoricalSpace(names=names[order], probs=probs[order])

    def truncate_to_mass(self, mass: float) -> "CategoricalSpace":
        """
        Keep smallest prefix reaching at least `mass` cumulative probability.
        """
        if not (0 < mass <= 1):
            raise ValueError("mass must be in (0, 1].")
        c = np.cumsum(self.probs)
        k = int(np.searchsorted(c, mass, side="left")) + 1
        return CategoricalSpace(self.names[:k].copy(), self.probs[:k].copy())


# -----------------------------
# Main sampler
# -----------------------------


class PokemonSetSampler:
    """
    Given a species datum (dict shaped as in the prompt), supports:
      - sample(n): random IID samples from the exact (factorized) model
      - top_pct(P, N): enumerate the top P% (cumulative mass) up to N sets via k-best product heap
    Model assumptions (standard and efficient):
      - Items, abilities, spreads, tera types are independent categorical features (using provided marginals).
      - Moves use conditional independent-Bernoulli with |S|=4, treating movesets as unordered sets.
    """

    def __init__(
        self, species: Dict[str, Any], rng: Optional[np.random.Generator] = None
    ):
        self.data = species
        self.rng = rng if rng is not None else np.random.default_rng()

        # Build categorical spaces
        self.abilities = CategoricalSpace.from_dict(self.data["abilities"])
        self.items = CategoricalSpace.from_dict(self.data["items"])
        self.tera = CategoricalSpace.from_dict(
            self.data.get("teraTypes", {"Nothing": 1.0})
        )
        self.spreads = CategoricalSpace.from_dict(self.data["spreads"])

        # Moves: dictionary of marginal inclusion probabilities
        moves = self.data["moves"]
        moves.pop("Nothing", None)  # remove "Nothing" if present
        mv_items = list(moves.items())
        self.move_names = np.array([k for k, _ in mv_items], dtype=object)
        self.move_probs = np.array([v for _, v in mv_items], dtype=np.float64)
        # Guardrails
        self.move_probs = np.clip(self.move_probs, 1e-12, 1 - 1e-12)

        # Precompute moves ESP normalization for k=4 (unordered 4-sets)
        self.k_moves = 4
        self._odds = self.move_probs / (1.0 - self.move_probs)
        self._e4 = _esp_k(self._odds, self.k_moves)

    # ---------- Core probability math ----------

    def moveset_logprob(self, moveset: Tuple[str, ...]) -> float:
        """
        Exact unordered moveset log-probability under conditional-Bernoulli(|S|=4) model.
        """
        idx = [int(np.where(self.move_names == m)[0][0]) for m in moveset]
        lp = float(np.sum(np.log(self._odds[idx])) - math.log(self._e4))
        return lp

    def fullset_logprob(
        self,
        item: str,
        ability: str,
        spread: str,
        teratype: str,
        moveset: Tuple[str, ...],
    ) -> float:
        """
        Log-probability of the full set as product of independent factors and moveset prob.
        """

        def cat_logp(space: CategoricalSpace, name: str) -> float:
            # find index (spaces are small; vector search is fine)
            idx = np.where(space.names == name)[0]
            if idx.size == 0:
                raise ValueError(f"{name} not in categorical space.")
            p = space.probs[idx[0]]
            return float(np.log(p))

        return (
            cat_logp(self.items, item)
            + cat_logp(self.abilities, ability)
            + cat_logp(self.spreads, spread)
            + cat_logp(self.tera, teratype)
            + self.moveset_logprob(moveset)
        )

    # ---------- Random sampling ----------

    def sample(self, n: int = 1) -> List[Dict[str, Any]]:
        """
        IID samples from the model; movesets are unordered and deduplicated within this batch.
        Returns a list of dicts with exact probabilities (not frequency estimates).
        """
        out = []
        seen = set()
        for _ in range(n):
            # Sample moveset
            moveset, pm = _sample_unordered_k_from_marginals(
                moves=list(self.move_names),
                probs=self.move_probs,
                k=self.k_moves,
                rng=self.rng,
            )

            # Sample other categories
            def pick(space: CategoricalSpace) -> str:
                idx = self.rng.choice(space.names.size, p=space.probs)
                return str(space.names[idx])

            item = pick(self.items)
            ability = pick(self.abilities)
            spread = pick(self.spreads)
            tera = pick(self.tera)

            key = (item, ability, spread, tera, moveset)
            if key in seen:
                # ensure uniqueness inside batch; if collision, re-draw lightweightly
                # (collisions are rare; bounded loop)
                retries = 0
                while key in seen and retries < 10:
                    moveset, pm = _sample_unordered_k_from_marginals(
                        moves=list(self.move_names),
                        probs=self.move_probs,
                        k=self.k_moves,
                        rng=self.rng,
                    )
                    item = pick(self.items)
                    ability = pick(self.abilities)
                    spread = pick(self.spreads)
                    tera = pick(self.tera)
                    key = (item, ability, spread, tera, moveset)
                    retries += 1
                if key in seen:
                    continue  # skip if we somehow still collided
            seen.add(key)

            logp = self.fullset_logprob(item, ability, spread, tera, moveset)
            out.append(
                {
                    "item": item,
                    "ability": ability,
                    "spread": spread,
                    "teraType": tera,
                    "moves": list(moveset),  # canonical sorted order
                    "prob": float(math.exp(logp)),
                }
            )
        return out

    # ---------- Top-P% (up to N) enumeration ----------

    def _top_mass_space(self, space: CategoricalSpace, mass: float) -> CategoricalSpace:
        return space.truncate_to_mass(mass)

    def _top_movesets_by_score(
        self, thresh: float, K: int
    ) -> List[Tuple[Tuple[str, ...], float]]:
        """
        Build candidate moves by taking the top-L moves by marginal p, scoring 4-combos
        using exact unordered probability (constant normalization) => product of odds.
        Returns top-K unique movesets and their exact probabilities under the moves model.
        """
        # Choose top-L by p
        L = (self.move_probs >= thresh).sum()

        order = np.argsort(-self.move_probs)
        idxL = order[:L]
        namesL = self.move_names[idxL]
        oddsL = self._odds[idxL]

        # Enumerate all C(L,4) combos if manageable, else grow L adaptively.
        # For typical move counts (~30–40) picking L=16 keeps C(16,4)=1820.
        # Score by sum(log odds) => rank; probability uses full e4 normalization (on all moves).

        log_odds = np.log(oddsL)
        combo_scores = []
        for comb in combinations(range(L), 4):
            s = float(log_odds[list(comb)].sum())
            combo_scores.append((s, comb))
        # top-K by score
        combo_scores.sort(reverse=True)
        sel = combo_scores[:K]
        out = []
        for s, comb in sel:
            chosen_names = tuple(sorted([str(namesL[i]) for i in comb]))
            # exact prob uses full e4 on all moves (not just top-L)
            # P(S) = exp(sum(log_odds_S) - log(e4_total))
            p = math.exp(s - math.log(self._e4))
            out.append((chosen_names, p))
        # dedupe (should already be unique)
        return out

    def top_pct(
        self, P: float, N: int, cat_mass: float = 0.995
    ) -> List[Dict[str, Any]]:
        """
        Return the most likely sets covering at least P (0..1) cumulative probability,
        up to at most N sets. Uses:
          - Truncation of categories to `cat_mass` cumulative mass each
          - Top-K movesets generated from top L_moves moves (default 16)
          - K-best product heap to enumerate largest products first
        Probabilities reported are the exact model probabilities (product of factors),
        not empirical frequencies.

        Note: Because we truncate categories and moves, if P is extremely high (e.g., >0.9999),
        increase cat_mass and/or L_moves.
        """
        if not (0 < P <= 1):
            raise ValueError("P must be in (0,1].")
        if N <= 0:
            return []

        # Truncate factor spaces to keep combinatorics small while covering most mass
        A = self._top_mass_space(self.abilities, cat_mass)
        I = self._top_mass_space(self.items, cat_mass)
        T = self._top_mass_space(self.tera, cat_mass)
        S = self._top_mass_space(self.spreads, cat_mass)

        # Movesets: choose K so that the product space is manageable; crude cap:
        # Aim roughly so that A*I*T*S*mK ~ few hundred thousand at most.
        est_target = 150_000
        denom = max(1, A.names.size * I.names.size * T.names.size * S.names.size)
        mK = max(64, min(2000, est_target // denom))
        top_moves = self._top_movesets_by_score(thresh=1 - cat_mass, K=mK)
        M_names = np.array([m for (m, _) in top_moves], dtype=object)
        M_probs = np.array([p for (_, p) in top_moves], dtype=np.float64)

        # Normalize the truncated moves mass to its true mass share (since top_moves is truncated),
        # but when computing the fullset probability we still use *exact* moveset probs (already exact);
        # truncation only affects whether we can reach the requested P mass.
        # Build arrays (sorted desc already)
        aP, iP, tP, sP, mP = A.probs, I.probs, T.probs, S.probs, M_probs

        # K-best product enumeration over 5 axes via max-heap.
        # State = indices (ia, ii, it, is, im). Product = aP[ia]*iP[ii]*tP[it]*sP[is]*mP[im].
        # We push neighbors by incrementing one coordinate at a time.
        axes = [aP, iP, tP, sP, mP]
        sizes = [len(aP), len(iP), len(tP), len(sP), len(mP)]
        if any(sz == 0 for sz in sizes):
            return []

        # Start at all zeros
        start = (0, 0, 0, 0, 0)
        start_log = sum(float(np.log(ax[0])) for ax in axes)
        heap: List[Tuple[float, Tuple[int, int, int, int, int]]] = []
        heapq.heappush(heap, (-start_log, start))
        visited = {start}

        out: List[Dict[str, Any]] = []
        cum_mass = 0.0

        while heap and len(out) < N and cum_mass < P:
            neg_logp, idxs = heapq.heappop(heap)
            ia, ii, it, is_, im = idxs
            prob = math.exp(-neg_logp)

            # Build the concrete set
            item = str(I.names[ii])
            ability = str(A.names[ia])
            spread = str(S.names[is_])
            tera = str(T.names[it])
            moveset = tuple(M_names[im])  # already sorted tuple of names

            out.append(
                {
                    "item": item,
                    "ability": ability,
                    "spread": spread,
                    "teraType": tera,
                    "moves": list(moveset),
                    "prob": prob,
                }
            )
            cum_mass += prob

            # Push neighbors
            for dim in range(5):
                lst = list(idxs)
                lst[dim] += 1
                if lst[dim] < sizes[dim]:
                    nxt = tuple(lst)
                    if nxt not in visited:
                        visited.add(nxt)
                        # new log prob = current + log(next_axis[next_idx]) - log(current_axis[current_idx])
                        delta = float(
                            np.log(axes[dim][lst[dim]])
                            - np.log(axes[dim][lst[dim] - 1])
                        )
                        heapq.heappush(heap, (neg_logp - delta, nxt))

        return out


import re
from typing import Any, Dict, Sequence, Tuple


def to_id(value: Any) -> str:
    """
    Rough equivalent of Pokémon Showdown's toID:
    - stringify
    - lowercase
    - strip non [a-z0-9]
    """
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def parse_spread_for_nature_and_evs(spread: Any) -> Tuple[str, str]:
    """
    Placeholder that mirrors the JS helper the original code uses.
    Adjust/replace with your real logic.
    Accepts either a dict like {'nature': 'Jolly', 'evs': '4/252/0/0/0/252'}
    or anything else (returns empty strings).
    """
    nature, evs = spread.split(":")
    return nature, evs.replace("/", ",")  # ensure string


def format_set_line(sampled_set: Dict[str, Any], opts: Dict[str, Any]) -> str:
    """
    Python port of the TypeScript function:
    - sampled_set is expected to have keys like 'moves', 'item', 'ability', 'spread', 'teraType'
    - opts should include 'species' and may include the rest (see below)
    """
    # Required
    species_name = opts["species"]  # required, like the TS version

    # Optionals with defaults
    nickname = opts.get("species", "")
    gender = opts.get("gender", "")
    shiny = opts.get("shiny", None)
    level = opts.get("level", None)
    happiness = opts.get("happiness", None)
    pokeball = opts.get("pokeball", "")
    hidden_power_type = opts.get("hiddenPowerType", "")
    gigantamax = opts.get("gigantamax", None)
    dynamax_level = opts.get("dynamaxLevel", None)
    ivs = opts.get("ivs", "31,31,31,31,31,31")
    move_separator = opts.get("moveSeparator", ",")

    nature, evs = parse_spread_for_nature_and_evs(sampled_set.get("spread"))

    moves: Sequence[Any] = sampled_set.get("moves", [])
    if len(moves) < 4:
        moves += [""] * (4 - len(moves))
    MOVES = move_separator.join(to_id(m) for m in moves)

    SHINY = "" if shiny is None else ("S" if shiny else "")
    LEVEL = "" if level is None else str(level)
    HAPPINESS = "" if happiness is None else str(happiness)
    GMAX = "" if gigantamax is None else ("G" if gigantamax else "")
    DYN = "" if dynamax_level is None else str(dynamax_level)
    item = sampled_set.get("item")
    if item == "Nothing":
        item = ""

    pipe_fields = [
        to_id(nickname),  # NICKNAME
        to_id(species_name),  # SPECIES
        to_id(item),  # ITEM
        to_id(sampled_set.get("ability")),  # ABILITY
        MOVES,  # MOVES
        nature,  # NATURE
        evs,  # EVS
        gender,  # GENDER
        ivs,  # IVS
        SHINY,  # SHINY
        LEVEL,  # LEVEL
        HAPPINESS,  # HAPPINESS
    ]

    teratype = sampled_set.get("teraType", "")
    if teratype == "Nothing":
        teratype = ""
    comma_tail = [
        pokeball,  # POKEBALL
        hidden_power_type,  # HIDDENPOWERTYPE
        GMAX,  # GIGANTAMAX
        DYN,  # DYNAMAXLEVEL
        teratype,  # TERATYPE
    ]

    return f"{'|'.join(pipe_fields)},{','.join(comma_tail)}"


def get_stats_url(generation: int, smogon_format: str) -> str:
    try:
        return requests.get(
            f"https://raw.githubusercontent.com/pkmn/smogon/refs/heads/main/data/stats/gen{generation}{smogon_format}.json"
        ).json()
    except:
        return None


ALL_FORMATS = ["ubers", "ou", "uu", "ru", "nu", "pu", "zu"]


def main():

    for generation in range(9, 0, -1):
        all_formats = {f: {s: [] for s in STOI["species"]} for f in ALL_FORMATS}
        only_format = {f: {s: [] for s in STOI["species"]} for f in ALL_FORMATS}

        for smogon_format in reversed(ALL_FORMATS):
            data = get_stats_url(generation, smogon_format)
            if data is None:
                continue

            for species in data["pokemon"]:
                sampler = PokemonSetSampler(data["pokemon"][species])
                many_samples = sampler.top_pct(0.95, 1024)
                packed_sets = [
                    format_set_line(s, {"species": species}) for s in many_samples
                ]

                species_key = to_id(species)

                for f in reversed(ALL_FORMATS):
                    if f == smogon_format:
                        all_formats[smogon_format][species_key] += packed_sets
                        break
                    all_formats[smogon_format][species_key] += all_formats[f][
                        species_key
                    ]

                only_format[smogon_format][species_key] += packed_sets

                print(species, smogon_format, len(packed_sets))

            with open(
                f"data/data/gen{generation}/{smogon_format}_all_formats.json", "w"
            ) as f:
                for sf in all_formats:
                    for s in all_formats[sf]:
                        all_formats[sf][s] = list(set(all_formats[sf][s]))
                json.dump(all_formats[smogon_format], f)

            with open(
                f"data/data/gen{generation}/{smogon_format}_only_format.json", "w"
            ) as f:
                for sf in only_format:
                    for s in only_format[sf]:
                        only_format[sf][s] = list(set(only_format[sf][s]))
                json.dump(only_format[smogon_format], f)


if __name__ == "__main__":
    main()
