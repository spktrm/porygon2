import asyncio
import logging
import os
import numpy as np
import httpx
from typing import TypedDict

import numpy.typing as npt
import pandas as pd
import json

from rl.environment.data import STOI, toid


ALL_FORMATS = [
    "ou",
    # "ubers",
    # "uu",
    # "zu",
    # "pu",
    # "nu",
]
BASE_URL = (
    "https://github.com/pkmn/smogon/raw/refs/heads/main/data/stats/gen{gen}{fmt}.json"
)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FrequencyStats(TypedDict):
    raw: float
    real: float
    weighted: float


class PokemonStats(TypedDict):
    lead: FrequencyStats
    usage: FrequencyStats
    count: int
    weight: float
    viability: tuple[int, int, int, int]
    abilities: dict[str, float]
    items: dict[str, float]
    moves: dict[str, float]
    teraTypes: dict[str, float]
    happinesses: dict[str, float]
    spreads: dict[str, float]  # { "adamant:0/252/0/0/4/252": 0.05, ... }
    teammates: dict[str, float]
    counters: dict[str, tuple[float, float, float]]


class UsageType(TypedDict):
    battles: int
    pokemon: dict[str, PokemonStats]


def _construct_generic_usage(
    data: UsageType, stats_key: str, mapping_key: str
) -> npt.NDArray[np.float32]:
    """Internal helper to map Pokemon stats to a NumPy usage matrix."""

    # Get the target mapping (e.g., STOI["items"])
    target_mapping = STOI[mapping_key]

    # Initialize array: Rows = Species, Cols = Attribute (Items, Moves, etc.)
    usage_array = np.zeros(
        (len(STOI["species"]), len(target_mapping)), dtype=np.float32
    )

    for pokemon, stats in data["pokemon"].items():
        pokemon_id = toid(pokemon)
        if pokemon_id not in STOI["species"]:
            print(
                f"Warning: Pokemon '{pokemon}' not found in STOI['species']. Skipping."
            )
            continue

        row_idx = STOI["species"][pokemon_id]

        for attr_name, usage_val in stats[stats_key].items():
            attr_id = toid(attr_name)
            if attr_id in target_mapping:
                col_idx = target_mapping[attr_id]
                usage_array[row_idx, col_idx] = usage_val

    return usage_array


def construct_ability_usage(data: UsageType) -> npt.NDArray[np.float32]:
    return _construct_generic_usage(data, "abilities", "abilities")


def construct_items_usage(data: UsageType) -> npt.NDArray[np.float32]:
    return _construct_generic_usage(data, "items", "items")


def construct_moves_usage(data: UsageType) -> npt.NDArray[np.float32]:
    return _construct_generic_usage(data, "moves", "moves")


def construct_teratypes_usage(data: UsageType) -> npt.NDArray[np.float32]:
    return _construct_generic_usage(data, "teraTypes", "typechart")


def construct_teammates_usage(data: UsageType) -> npt.NDArray[np.float32]:
    return _construct_generic_usage(data, "teammates", "species")


def construct_gender_usage(
    data: UsageType, pokedex_df: pd.DataFrame
) -> npt.NDArray[np.float32]:
    gender_col = pokedex_df["gender"]

    gender_col_unique = [v for v in gender_col.unique() if v]
    gender_col_sorted = sorted(gender_col_unique)

    usage_array = np.zeros(
        (len(STOI["species"]), len(STOI["gendername"])), dtype=np.float32
    )

    for _, row in pokedex_df.iterrows():
        pokemon_id = toid(row.id)
        if pokemon_id not in STOI["species"]:
            print(
                f"Warning: Pokemon ID '{row.id}' not found in STOI['species']. Skipping."
            )
            continue

        row_idx = STOI["species"][pokemon_id]
        if row.gender:
            usage_array[row_idx, STOI["gendername"][row.gender.lower()]] = 1.0

        for gender_string in ["m", "f"]:
            usage_array[row_idx, STOI["gendername"][gender_string]] = row[
                f"genderRatio.{gender_string.upper()}"
            ]

    return usage_array


def construct_nature_usage(data: UsageType) -> npt.NDArray[np.float32]:
    usage_array = np.zeros(
        (len(STOI["species"]), len(STOI["natures"])), dtype=np.float32
    )

    for pokemon, stats in data["pokemon"].items():
        pokemon_id = toid(pokemon)
        if pokemon_id not in STOI["species"]:
            continue

        row_idx = STOI["species"][pokemon_id]

        for attr_name, usage_val in stats["spreads"].items():
            nature = attr_name.split(":")[0]
            nature_id = toid(nature)
            if nature_id in STOI["natures"]:
                col_idx = STOI["natures"][nature_id]
                usage_array[row_idx, col_idx] += usage_val

    return usage_array


def construct_ev_usage(data: UsageType) -> npt.NDArray[np.float32]:
    usage_array = np.zeros((len(STOI["species"]), 6, 64), dtype=np.float32)

    for pokemon, stats in data["pokemon"].items():
        pokemon_id = toid(pokemon)
        if pokemon_id not in STOI["species"]:
            continue

        row_idx = STOI["species"][pokemon_id]

        for attr_name, usage_val in stats["spreads"].items():
            stat_spread = attr_name.split(":")[1]
            for idx, (stat, value) in enumerate(
                zip(
                    ["hp", "atk", "def", "spa", "spd", "spe"],
                    map(int, stat_spread.split("/")),
                )
            ):
                usage_array[row_idx, idx, int(value / 4)] += usage_val

    return usage_array


def construct_species_usage(data: UsageType) -> npt.NDArray[np.float32]:
    usage_array = np.zeros((len(STOI["species"]),), dtype=np.float32)

    for pokemon, stats in data["pokemon"].items():
        pokemon_id = toid(pokemon)
        if pokemon_id not in STOI["species"]:
            logger.warning(
                f"Pokemon '{pokemon}' not found in STOI['species']. Skipping species usage entry."
            )
            continue

        row_idx = STOI["species"][pokemon_id]
        usage_array[row_idx] = stats["usage"]["real"]

    return usage_array


def save_usage_array(
    stat: str, usage_array: npt.NDArray[np.float32], generation: int, smogon_format: str
):
    for _type in ["mask", "usage"]:
        directory = f"data/data/gen{generation}/{_type}/"
        os.makedirs(directory, exist_ok=True)

        prefix = f"{directory}{stat}_{_type}_{smogon_format}"
        np.save(f"{prefix}.npy", usage_array > 0 if _type == "mask" else usage_array)
    logger.info(f"Saved {stat} data for gen{generation}{smogon_format}")


async def fetch_and_process(
    client: httpx.AsyncClient, generation: int, smogon_format: str
):
    with open(f"data/data/gen{generation}/species.json", "r") as f:
        pokedex_df = pd.json_normalize(json.load(f))

    url = BASE_URL.format(gen=generation, fmt=smogon_format)

    # Simple retry logic
    for attempt in range(3):
        try:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()

            for stat, usage_func in [
                ("species", construct_species_usage),
                ("teammates", construct_teammates_usage),
                ("abilities", construct_ability_usage),
                ("items", construct_items_usage),
                ("moves", construct_moves_usage),
                ("teratypes", construct_teratypes_usage),
                ("ev", construct_ev_usage),
                ("nature", construct_nature_usage),
            ]:
                try:
                    usage_array = usage_func(data)
                    # usage_array = usage_array + ~(usage_array.any(axis=-1)[..., None])
                except Exception as e:
                    logger.error(
                        f"Error processing {stat} for gen{generation}{smogon_format}: {e}"
                    )
                    continue
                else:
                    save_usage_array(stat, usage_array, generation, smogon_format)

            usage_array = construct_gender_usage(data, pokedex_df)
            save_usage_array("gender", usage_array, generation, smogon_format)

            return  # Success

        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.warning(f"Attempt {attempt + 1} failed for {smogon_format}: {e}")
            await asyncio.sleep(1 * (attempt + 1))

    logger.error(
        f"Failed to fetch data for Gen {generation} {smogon_format} after 3 attempts."
    )


async def main():
    # Using a client session for connection pooling
    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = []
        for generation in range(9, 10):
            for smogon_format in ALL_FORMATS:
                tasks.append(fetch_and_process(client, generation, smogon_format))

        logger.info(f"Starting async fetch for {len(tasks)} formats...")
        await asyncio.gather(*tasks)
        logger.info("All tasks completed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
