import numpy as np
import requests

from rl.environment.data import (
    NUM_ABILITIES,
    NUM_ITEMS,
    NUM_MOVES,
    NUM_NATURES,
    NUM_SPECIES,
)

STATS_URL_BASE = (
    "https://raw.githubusercontent.com/pkmn/smogon/refs/heads/main/data/stats/{}.json"
)
SMOGON_FORMATS = ["ou", "uu", "ru", "nu", "pu", "zu"]


def generate_masks(data):

    return


def main():
    for i in range(10, 0, -1):

        ability_mask = np.zeros((NUM_SPECIES, NUM_ABILITIES))
        item_mask = np.zeros((NUM_SPECIES, NUM_ITEMS))
        move_mask = np.zeros((NUM_SPECIES, NUM_MOVES))
        nature_mask = np.zeros((NUM_SPECIES, NUM_NATURES))

        for smogon_format in SMOGON_FORMATS:
            data = requests.get(STATS_URL_BASE.format(f"gen{i}{smogon_format}")).json()
            mask_output = generate_masks(data)

            ability_mask = np.logical_or(ability_mask, mask_output["ability_mask"])
            item_mask = np.logical_or(item_mask, mask_output["item_mask"])
            move_mask = np.logical_or(move_mask, mask_output["move_mask"])
            nature_mask = np.logical_or(nature_mask, mask_output["nature_mask"])

        np.save(f"data/data/gen{i}/ability_mask.npy", ability_mask)
        np.save(f"data/data/gen{i}/item_mask.npy", item_mask)
        np.save(f"data/data/gen{i}/move_mask.npy", move_mask)
        np.save(f"data/data/gen{i}/nature_mask.npy", nature_mask)

    return


if __name__ == "__main__":
    main()
