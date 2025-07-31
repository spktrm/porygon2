import itertools
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests


def toid(string: str) -> str:
    """Convert any label to Pokémon Showdown's id format (lowercase, A-Z 0-9 and _)."""
    return "".join(c for c in string if c.isalnum() or c == "_").lower()


STAT_ORDER = ["hp", "atk", "def", "spa", "spd", "spe"]


def stats_dict_to_list(d: Dict[str, int], default: int) -> List[int]:
    """Re-order a stats dict (EVs / IVs) to an HP-Atk-Def-SpA-SpD-Spe list."""
    stats = [default] * 6
    if not d:
        return stats

    normalise = {
        "spatk": "spa",
        "spa": "spa",
        "spattack": "spa",
        "spdef": "spd",
        "spd": "spd",
        "spe": "spe",
        "speed": "spe",
    }

    if isinstance(d, list):
        d = d[0]

    for k, v in d.items():
        key = normalise.get(k.lower(), k.lower())
        if key in STAT_ORDER:
            stats[STAT_ORDER.index(key)] = v
    return stats


def stats_to_csv(stats: List[int], default: int) -> str:
    """Compress stats list by omitting default values and trailing commas."""
    if all(v == default for v in stats):
        return ""
    return ",".join("" if v == default else str(v) for v in stats)


def expand_option(opt: Any) -> List[str]:
    """Return *opt* as a list (if it's already a list, unchanged)."""
    if isinstance(opt, list):
        return opt
    return [opt]


def product_lists(lists: List[List[str]]):
    for prod in itertools.product(*lists):
        yield list(prod)


def generate_packed_sets(source: Dict[str, Any], generation: str) -> List[str]:
    rows: List[str] = []

    with open(f"data/data/{generation}/species.json", "r") as f:
        data = json.load(f)
    species_df = pd.json_normalize(data)

    ability_cols = [c for c in species_df.columns if c.startswith("abilities")]

    for species_name, setdict in source.items():
        row = species_df.loc[species_df["id"] == toid(species_name)]
        spid = toid(species_name)
        nickname = spid  # identical => species field blank
        species_field = ""

        for _set_name, setdata in setdict.items():
            if len(ability_cols) > 0:
                if setdata.get("ability") is None:
                    setdata["ability"] = (
                        row[ability_cols].dropna(axis=1).values.squeeze().tolist()
                    )

            # Move slots (each slot either a str or list[str])
            move_slots: List[List[str]] = []
            for slot in setdata.get("moves", []):
                move_slots.append([toid(m) for m in expand_option(slot)])
            if not move_slots:
                move_slots = [[""]]

            # Combinatorial options: item, nature, ability
            item_opts = [toid(i) for i in expand_option(setdata.get("item", ""))] or [
                ""
            ]
            nature_opts = [
                "" if toid(n) == "serious" else toid(n)
                for n in expand_option(setdata.get("nature", ""))
            ] or [""]
            ability_opts = [
                toid(a) for a in expand_option(setdata.get("ability", ""))
            ] or [""]

            # EV/IV compression
            evs_csv = stats_to_csv(stats_dict_to_list(setdata.get("evs", {}), 0), 0)
            ivs_csv = stats_to_csv(stats_dict_to_list(setdata.get("ivs", {}), 31), 31)

            # Fields fixed/blank for Smogon dex exports (Gen1-9, no shiny/level/etc.)
            gender = shiny = level = happiness = ""
            cluster_suffix = ""  # all cluster sub-fields blank => omit

            for moves in product_lists(move_slots):
                moves_csv = ",".join(moves)
                for item in item_opts:
                    for nat in nature_opts:
                        for abil in ability_opts:
                            rows.append(
                                "|".join(
                                    [
                                        nickname,
                                        species_field,
                                        item,
                                        abil,
                                        moves_csv,
                                        nat,
                                        evs_csv,
                                        gender,
                                        ivs_csv,
                                        shiny,
                                        level,
                                        cluster_suffix,
                                    ]
                                )
                            )
    return rows


BASE_URL = "https://pkmn.github.io/smogon/data/sets/"
DEST_DIR = Path("data/data/")
DEST_DIR.mkdir(exist_ok=True)


def list_remote_jsons() -> List[str]:
    """Scrape the directory listing to collect every *.json file path."""
    tiers = ["ubers", "ou", "uu", "uubl", "ru", "rubl", "nu", "nubl", "pu", "publ"]
    return [
        f"gen{generation}{tier}.json" for generation in range(1, 10) for tier in tiers
    ]


def main():
    print("Discovering available gens/tiers …")
    json_files = list_remote_jsons()
    if not json_files:
        raise RuntimeError("No JSON sets found at " + BASE_URL)

    print(f"Found {len(json_files)} tier files. Processing …\n")

    for rel_path in json_files:
        url = BASE_URL + rel_path
        print(f"→ {rel_path} … ", end="", flush=True)

        try:
            data = requests.get(url, timeout=60).json()
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            continue

        packed_rows = generate_packed_sets(data, rel_path[:4])
        out_file = DEST_DIR / f"{Path(rel_path).stem}_packed.json"
        out_file.write_text(json.dumps(packed_rows, indent=2))

        print(f"{len(packed_rows):,} sets saved → {out_file}")


if __name__ == "__main__":
    main()
