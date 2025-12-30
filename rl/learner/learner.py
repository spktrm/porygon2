from __future__ import annotations

import heapq
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

# -----------------------------------------------------------------------------
# Scikit-Learn Imports
# -----------------------------------------------------------------------------
try:
    import warnings

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
except ImportError:
    raise ImportError(
        "This script requires scikit-learn. Please run: pip install scikit-learn"
    )

# -----------------------------------------------------------------------------
# Math & Probability Helpers
# -----------------------------------------------------------------------------


def _esp_k(odds: np.ndarray, k: int) -> float:
    E = np.zeros(k + 1, dtype=np.float64)
    E[0] = 1.0
    for o in odds:
        upto = min(k, np.count_nonzero(E) - 1 + 1)
        for d in range(upto, 0, -1):
            E[d] += o * E[d - 1]
    return float(E[k])


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CategoricalSpace:
    names: np.ndarray
    probs: np.ndarray

    @staticmethod
    def from_dict(d: Dict[str, float]) -> "CategoricalSpace":
        items = list(d.items())
        names = np.array([k for k, _ in items], dtype=object)
        probs = np.array([v for _, v in items], dtype=np.float64)
        s = probs.sum()
        if s <= 0:
            return CategoricalSpace(names=np.array([]), probs=np.array([]))
        probs = probs / s
        order = np.argsort(-probs)
        return CategoricalSpace(names=names[order], probs=probs[order])

    def truncate_to_mass(self, mass: float) -> "CategoricalSpace":
        if self.probs.size == 0:
            return self
        c = np.cumsum(self.probs)
        k = int(np.searchsorted(c, mass, side="left")) + 1
        return CategoricalSpace(self.names[:k].copy(), self.probs[:k].copy())


# -----------------------------------------------------------------------------
# Sampler Logic
# -----------------------------------------------------------------------------


class PokemonSetSampler:
    def __init__(
        self, species: Dict[str, Any], rng: Optional[np.random.Generator] = None
    ):
        self.data = species
        self.rng = rng if rng is not None else np.random.default_rng()

        self.abilities = CategoricalSpace.from_dict(self.data.get("abilities", {}))
        self.items = CategoricalSpace.from_dict(self.data.get("items", {}))
        self.tera = CategoricalSpace.from_dict(
            self.data.get("teraTypes", {"Nothing": 1.0})
        )
        self.spreads = CategoricalSpace.from_dict(self.data.get("spreads", {}))

        moves = self.data.get("moves", {})
        moves.pop("Nothing", None)
        mv_items = list(moves.items())

        if not mv_items:
            self.move_names = np.array([], dtype=object)
            self.move_probs = np.array([], dtype=np.float64)
            self._odds = np.array([])
            self._e4 = 1.0
            self.k_moves = 0
        else:
            self.move_names = np.array([k for k, _ in mv_items], dtype=object)
            self.move_probs = np.array([v for _, v in mv_items], dtype=np.float64)
            self.move_probs = np.clip(self.move_probs, 1e-12, 1 - 1e-12)
            self.k_moves = min(4, len(self.move_names))
            self._odds = self.move_probs / (1.0 - self.move_probs)
            self._e4 = _esp_k(self._odds, self.k_moves)

    def _top_movesets_by_score(
        self, thresh: float, K: int
    ) -> List[Tuple[Tuple[str, ...], float]]:
        if self.k_moves == 0:
            return [(tuple(), 1.0)]
        L = (self.move_probs >= thresh).sum()
        L = max(L, self.k_moves)

        # Optimization: Limit search space for combinations
        L = min(L, 30)

        order = np.argsort(-self.move_probs)
        idxL = order[:L]
        namesL = self.move_names[idxL]
        oddsL = self._odds[idxL]
        log_odds = np.log(oddsL)

        combo_scores = []
        for comb in combinations(range(L), self.k_moves):
            s = float(log_odds[list(comb)].sum())
            combo_scores.append((s, comb))

        combo_scores.sort(reverse=True)
        sel = combo_scores[:K]
        out = []
        for s, comb in sel:
            chosen_names = tuple(sorted([str(namesL[i]) for i in comb]))
            p = math.exp(s - math.log(self._e4))
            out.append((chosen_names, p))
        return out

    def _generate_raw_pool(self, N: int) -> List[Dict[str, Any]]:
        if self.abilities.probs.size == 0 or self.spreads.probs.size == 0:
            return []

        A = self.abilities.truncate_to_mass(0.999)
        I = self.items.truncate_to_mass(0.999)
        T = self.tera.truncate_to_mass(0.999)
        S = self.spreads.truncate_to_mass(0.999)

        denom = max(1, A.names.size * I.names.size * T.names.size * S.names.size)
        mK = max(64, min(4000, 150_000 // denom))

        top_moves = self._top_movesets_by_score(thresh=0.005, K=mK)
        if not top_moves:
            return []
        M_names = np.array([m for (m, _) in top_moves], dtype=object)
        M_probs = np.array([p for (_, p) in top_moves], dtype=np.float64)

        axes = [A.probs, I.probs, T.probs, S.probs, M_probs]
        sizes = [len(ax) for ax in axes]
        if any(sz == 0 for sz in sizes):
            return []

        start = (0, 0, 0, 0, 0)
        start_log = sum(float(np.log(ax[0])) for ax in axes)
        heap = [(-start_log, start)]
        visited = {start}
        pool = []

        while heap and len(pool) < N:
            neg_logp, idxs = heapq.heappop(heap)
            ia, ii, it, is_, im = idxs
            prob = math.exp(-neg_logp)

            pool.append(
                {
                    "item": str(I.names[ii]),
                    "ability": str(A.names[ia]),
                    "spread": str(S.names[is_]),
                    "teraType": str(T.names[it]),
                    "moves": list(M_names[im]),
                    "prob": prob,
                }
            )

            for dim in range(5):
                lst = list(idxs)
                lst[dim] += 1
                if lst[dim] < sizes[dim]:
                    nxt = tuple(lst)
                    if nxt not in visited:
                        visited.add(nxt)
                        delta = float(
                            np.log(axes[dim][lst[dim]])
                            - np.log(axes[dim][lst[dim] - 1])
                        )
                        heapq.heappush(heap, (neg_logp - delta, nxt))
        return pool

    def _find_optimal_k(self, X: np.ndarray, max_k: int) -> int:
        candidate_ks = np.unique(np.linspace(2, max_k, num=5, dtype=int))
        candidate_ks = candidate_ks[candidate_ks > 1]

        best_k = 2
        best_score = -1.0

        for k in candidate_ks:
            km = MiniBatchKMeans(
                n_clusters=k, random_state=42, n_init=3, batch_size=256
            )
            labels = km.fit_predict(X)

            try:
                score = silhouette_score(X, labels, sample_size=500)
            except ValueError:
                score = -1.0

            if score > best_score:
                best_score = score
                best_k = k

        return int(best_k)

    def _parse_spread_features(
        self, spread_str: str
    ) -> Tuple[str, List[float], List[float]]:
        """
        Parses a spread string (e.g., 'Adamant:252/0/0/252/4/0') into components.
        Returns: (Nature, EVs[6], IVs[6])
        """
        parts = spread_str.split(":")
        nature = parts[0]
        if len(parts) > 1:
            nums = parts[1].split("/")
            # Parse EVs
            evs = []
            for n in nums:
                try:
                    evs.append(float(n))
                except ValueError:
                    evs.append(0.0)
            # Pad to 6
            while len(evs) < 6:
                evs.append(0.0)
        else:
            evs = [0.0] * 6

        # Standard stats data usually implies 31 IVs unless specified in specific formats (which this parser simplifies)
        # We return 31s for IVs as default
        ivs = [31.0] * 6

        return nature, evs, ivs

    def sample_kmeans(
        self, total_slots: int = 1024, variance_threshold: float = 0.95
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

        # Adaptive sampling size
        n_m = len(self.move_names)
        n_i = len(self.items.names)
        complexity = n_m * n_i

        if complexity < 500:
            N_adapt = 1500
        elif complexity < 5000:
            N_adapt = 3500
        else:
            N_adapt = 6500

        raw_pool = self._generate_raw_pool(N=N_adapt)

        stats = {"clusters": 0, "unique": 0, "mass": 0.0, "total_sets": 0}

        if not raw_pool:
            return [], stats

        # ---------------------------------------------------------------------
        # Feature Encoding
        # ---------------------------------------------------------------------

        # 1. Moves (Multi-Label)
        mlb = MultiLabelBinarizer()
        moves_matrix = mlb.fit_transform([p["moves"] for p in raw_pool])

        # 2. Simple Categoricals (One-Hot)
        def encode_col(key):
            lb = LabelBinarizer()
            col_data = [p.get(key, "Nothing") for p in raw_pool]
            return lb.fit_transform(col_data)

        item_mat = encode_col("item")
        ability_mat = encode_col("ability")

        # 3. Complex Spread Encoding (Numerical Split)
        natures = []
        ev_matrix = []
        iv_matrix = []

        for p in raw_pool:
            spr = p.get("spread", "")
            nat, evs, ivs = self._parse_spread_features(spr)
            natures.append(nat)

            # Normalize EVs by 512, IVs by 31
            ev_matrix.append([x / 512.0 for x in evs])
            iv_matrix.append([x / 31.0 for x in ivs])

        lb_nature = LabelBinarizer()
        nature_mat = lb_nature.fit_transform(natures)
        ev_mat_np = np.array(ev_matrix, dtype=np.float32)
        iv_mat_np = np.array(iv_matrix, dtype=np.float32)

        # Concatenate all features
        # [Moves | Items | Abilities | Nature | EVs | IVs]
        X = np.hstack(
            (moves_matrix, item_mat, ability_mat, nature_mat, ev_mat_np, iv_mat_np)
        )

        # ---------------------------------------------------------------------
        # Clustering
        # ---------------------------------------------------------------------
        n_samples = X.shape[0]
        unique_rows = np.unique(X, axis=0).shape[0]

        n_components = min(n_samples, X.shape[1], 128)
        if n_components > 0:
            pca = PCA(n_components=n_components, random_state=42)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X

        max_possible_k = min(unique_rows, int(math.sqrt(n_samples)))

        if max_possible_k < 2:
            k_clusters = 1
        else:
            k_clusters = self._find_optimal_k(X_reduced, max_possible_k)

        if k_clusters > 1:
            kmeans = MiniBatchKMeans(
                n_clusters=k_clusters, random_state=42, n_init=3, batch_size=256
            )
            labels = kmeans.fit_predict(X_reduced)
        else:
            labels = np.zeros(n_samples, dtype=int)

        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(raw_pool[i])

        final_sets = []

        for _, cluster_sets in clusters.items():
            cluster_sets.sort(key=lambda x: x["prob"], reverse=True)

            total_cluster_mass = sum(s["prob"] for s in cluster_sets)
            target_mass = total_cluster_mass * variance_threshold

            current_captured = 0.0

            for s in cluster_sets:
                final_sets.append(s)
                current_captured += s["prob"]
                if current_captured >= target_mass:
                    break

        if len(final_sets) > total_slots:
            final_sets.sort(key=lambda x: x["prob"], reverse=True)
            final_sets = final_sets[:total_slots]

        unique_hashes = set()
        captured_mass = 0.0
        for s in final_sets:
            h = (s["item"], s["ability"], s["spread"], tuple(s["moves"]))
            if h not in unique_hashes:
                unique_hashes.add(h)
                captured_mass += s["prob"]

        stats.update(
            {
                "clusters": len(clusters),
                "unique": len(unique_hashes),
                "mass": min(1.0, captured_mass),
                "total_sets": len(final_sets),
            }
        )

        return final_sets, stats


def to_id(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def parse_spread_for_nature_and_evs(spread: Any) -> Tuple[str, str]:
    if not isinstance(spread, str) or ":" not in spread:
        return "", "0,0,0,0,0,0"
    nature, evs = spread.split(":")
    return nature, evs.replace("/", ",")


def format_set_line(sampled_set: Dict[str, Any], opts: Dict[str, Any]) -> str:
    species_name = opts["species"]
    moves = sampled_set.get("moves", [])
    if len(moves) < 4:
        moves += [""] * (4 - len(moves))

    move_sep = opts.get("moveSeparator", ",")
    MOVES = move_sep.join(to_id(m) for m in moves)

    nature, evs = parse_spread_for_nature_and_evs(sampled_set.get("spread"))

    item = sampled_set.get("item", "")
    if item == "Nothing":
        item = ""
    tera = sampled_set.get("teraType", "")
    if tera == "Nothing":
        tera = ""

    pipe_fields = [
        to_id(opts.get("species", "")),
        to_id(species_name),
        to_id(item),
        to_id(sampled_set.get("ability")),
        MOVES,
        nature,
        evs,
        "",
        "31,31,31,31,31,31",
        "",
        "",
        "",
    ]
    comma_tail = ["", "", "", "", tera]
    return f"{'|'.join(pipe_fields)},{','.join(comma_tail)}"


def get_stats_url(generation: int, smogon_format: str) -> Optional[Dict]:
    url = f"https://raw.githubusercontent.com/pkmn/smogon/refs/heads/main/data/stats/gen{generation}{smogon_format}.json"
    try:
        resp = requests.get(url)
        return resp.json() if resp.status_code == 200 else None
    except:
        return None


# -----------------------------------------------------------------------------
# Worker Function
# -----------------------------------------------------------------------------


def process_species(args: Tuple) -> Tuple[str, List[str], Dict[str, Any]]:
    species, stats, slots = args
    sampler = PokemonSetSampler(stats)

    many_samples, stats_info = sampler.sample_kmeans(
        total_slots=slots, variance_threshold=0.95
    )

    packed_sets = [format_set_line(s, {"species": species}) for s in many_samples]
    packed_sets = list(dict.fromkeys(packed_sets))

    return species, packed_sets, stats_info


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

ALL_FORMATS = ["ubers", "ou", "uu", "ru", "nu", "pu", "zu"]


def main():
    # Reverted to ThreadPoolExecutor as requested
    MAX_WORKERS = os.cpu_count() or 4

    for generation in range(9, 10):
        all_formats = {f: defaultdict(list) for f in ALL_FORMATS}
        only_format = {f: defaultdict(list) for f in ALL_FORMATS}

        for smogon_format in reversed(ALL_FORMATS):
            print(f"\n--- Gen {generation} {smogon_format} ---")
            data = get_stats_url(generation, smogon_format)
            if not data:
                continue

            pokemon_data = data.get("pokemon", data)

            tasks = []
            for species, stats in pokemon_data.items():
                tasks.append((species, stats, 1024))

            # ThreadPoolExecutor used here
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_species = {
                    executor.submit(process_species, t): t[0] for t in tasks
                }

                for future in as_completed(future_to_species):
                    try:
                        species, packed_sets, stats = future.result()

                        print(
                            f"[{species:<12}] K-Means: {stats['clusters']:<3} | "
                            f"Sets: {stats['total_sets']:<4} | "
                            f"Mass: {stats['mass']:.4f}"
                        )

                        species_key = to_id(species)
                        only_format[smogon_format][species_key].extend(packed_sets)

                        for f in reversed(ALL_FORMATS):
                            if f == smogon_format:
                                all_formats[smogon_format][species_key].extend(
                                    packed_sets
                                )
                                break
                            all_formats[smogon_format][species_key].extend(
                                all_formats[f][species_key]
                            )

                    except Exception as e:
                        sp_name = future_to_species[future]
                        print(f"Error processing {sp_name}: {e}")

            os.makedirs(f"data/data/gen{generation}", exist_ok=True)

            with open(
                f"data/data/gen{generation}/{smogon_format}_all_formats.json", "w"
            ) as f:
                clean_all = {
                    k: list(dict.fromkeys(v))
                    for k, v in all_formats[smogon_format].items()
                }
                json.dump(clean_all, f, indent=2)

            with open(
                f"data/data/gen{generation}/{smogon_format}_only_format.json", "w"
            ) as f:
                clean_only = {
                    k: list(dict.fromkeys(v))
                    for k, v in only_format[smogon_format].items()
                }
                json.dump(clean_only, f, indent=2)


if __name__ == "__main__":
    main()
