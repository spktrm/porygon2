import json

import numpy as np
import pandas as pd


def toid(string: str) -> str:
    return "".join(c for c in string if c.isalnum() or c == "_").lower()


class Pokedex:
    def __init__(self, generation: int):
        self.generation = generation

        self.data = self._load_base_data()

        self.species = self._load_species_data()
        self.moves = self._load_moves_data()
        self.abilities = self._load_abilities_data()
        self.items = self._load_items_data()
        self.learnset = self._load_learnset_data()

    def _load_base_data(self):
        with open(f"data/data/data.json", "r") as f:
            return json.load(f)

    def _load_data(self, filename: str):
        with open(f"data/data/gen{self.generation}/{filename}", "r") as f:
            json_data = json.load(f)
        return pd.json_normalize(json_data)

    def _get_index(self, datum: str, key: str):
        value = self.data[datum].get(toid(key), None)
        if value is None:
            raise ValueError(f"Key '{key}' not found in {datum}.")
        return value

    def get_species_index(self, species: str):
        return self._get_index("species", species)

    def get_move_index(self, move: str):
        return self._get_index("moves", move)

    def get_ability_index(self, ability: str):
        return self._get_index("abilities", ability)

    def get_item_index(self, item: str):
        return self._get_index("items", item)

    def _load_species_data(self):
        return self._load_data("species.json")

    def _load_moves_data(self):
        return self._load_data("moves.json")

    def _load_abilities_data(self):
        return self._load_data("abilities.json")

    def _load_items_data(self):
        return self._load_data("items.json")

    def _load_learnset_data(self):
        return self._load_data("learnsets.json")

    def get_tier_species_mask(self, tier: str):
        numbered_species = self.data["species"]
        mask = np.zeros(len(numbered_species), dtype=bool)
        for key, index in numbered_species.items():
            row = self.species[self.species["id"] == key.lower()]
            if row.empty:
                continue
            mask[index] = ((row["tier"] == tier) & (row["tier"] != "Illegal")).item()
        return mask

    def get_item_mask(self):
        numbered_items = self.data["items"]
        mask = np.zeros(len(numbered_items), dtype=bool)
        for key, index in numbered_items.items():
            row = self.items[self.items["id"] == key.lower()]
            if row.empty:
                continue
            desc = row["shortDesc"].item()
            does_nothing = desc == "No competitive use."
            mask[index] = (not does_nothing) & (
                (row["isNonstandard"] != "Future")
                & (row["isNonstandard"] != "Unobtainable")
            ).item()
        return mask

    def get_ability_mask(self):
        numbered_species = self.data["species"]
        numbered_abilities = self.data["abilities"]
        mask = np.zeros((len(numbered_species), len(numbered_abilities)), dtype=bool)
        ability_columns = [i for i in self.species.columns if i.startswith("abilities")]
        for _, row in self.species.iterrows():
            row_index = numbered_species.get(toid(row["id"]))
            for ability in row[ability_columns].tolist():
                if not isinstance(ability, str):
                    continue
                col_index = numbered_abilities.get(toid(ability))
                mask[row_index, col_index] = True
        return mask

    def get_learnset_mask(self):
        numbered_species = self.data["species"]
        numbered_moves = self.data["moves"]
        mask = np.zeros((len(numbered_species), len(numbered_moves)), dtype=bool)
        move_columns = [i for i in self.learnset.columns if i.startswith("learnset")]
        for (_, species_row), (_, learnset_row) in zip(
            self.species.iterrows(), self.learnset.iterrows()
        ):
            row_index = numbered_species.get(toid(species_row["id"]))

            for learned_at, move in zip(
                learnset_row[move_columns].tolist(), move_columns
            ):
                if isinstance(learned_at, float):
                    continue

                value = True
                if isinstance(learned_at, list):
                    value = any(
                        possibilities.startswith(f"{self.generation}")
                        for possibilities in learned_at
                    )

                if value:
                    col_index = numbered_moves.get(toid(move[len("learnset.") :]))

                    if row_index is None or col_index is None:
                        raise ValueError(
                            f"Invalid species or move: {species_row['id']} -> {move}"
                        )
                    mask[row_index, col_index] = value
        return mask


def save_gen_three():
    pokedex = Pokedex(generation=3)

    ability_mask = pokedex.get_ability_mask()
    ability_mask[..., pokedex.get_ability_index("sandveil")] = False
    ability_mask[..., pokedex.get_ability_index("soundproof")] = False

    with open(f"data/data/gen3/OU_ability_mask.npy", "wb") as f:
        np.save(f, ability_mask)

    item_mask = pokedex.get_item_mask()

    with open(f"data/data/gen3/OU_item_mask.npy", "wb") as f:
        np.save(f, item_mask)

    learnset_mask = pokedex.get_learnset_mask()
    learnset_mask[pokedex.get_species_index("smeargle")] = learnset_mask.any(axis=0)
    learnset_mask[
        pokedex.get_species_index("smeargle"), pokedex.get_move_index("ingrain")
    ] = False
    learnset_mask[..., pokedex.get_move_index("assist")] = False
    learnset_mask[..., pokedex.get_move_index("swagger")] = False
    learnset_mask[..., pokedex.get_move_index("doubleteam")] = False
    learnset_mask[..., pokedex.get_move_index("minimize")] = False
    learnset_mask[..., pokedex.get_move_index("horndrill")] = False
    learnset_mask[..., pokedex.get_move_index("guillotine")] = False
    learnset_mask[..., pokedex.get_move_index("sheercold")] = False
    learnset_mask[..., pokedex.get_move_index("fissure")] = False

    with open(f"data/data/gen3/OU_learnset_mask.npy", "wb") as f:
        np.save(f, learnset_mask)

    for tier in ["Uber", "OU", "UU", "RU", "NU", "PU"]:
        mask = pokedex.get_tier_species_mask(tier)

        with open(f"data/data/gen3/{tier}_species_mask.npy", "wb") as f:
            np.save(f, mask)


def main():
    save_gen_three()


if __name__ == "__main__":
    main()
