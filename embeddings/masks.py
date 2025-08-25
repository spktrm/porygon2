import json
from typing import Any, Callable

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
        self.learnset = self._load_learnset_data(do_filter=False)

    def _load_base_data(self):
        with open(f"data/data/data.json", "r") as f:
            return json.load(f)

    def _filter_df(
        self, df: pd.DataFrame, column: str, func: Callable[[Any], bool]
    ) -> pd.DataFrame:
        return df[df[column].map(func)]

    def _load_data(self, filename: str, do_filter: bool = True):
        with open(f"data/data/gen{self.generation}/{filename}", "r") as f:
            json_data = json.load(f)
        df = pd.json_normalize(json_data)
        if do_filter:
            df = self._filter_df(df, "isNonstandard", lambda x: x is None)
        return df

    def _get_index(self, datum: str, key: str):
        if not key.startswith("_"):
            key = toid(key)
        value = self.data[datum].get(key, None)
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

    def _load_species_data(self, do_filter: bool = True):
        return self._load_data("species.json", do_filter=do_filter)

    def _load_moves_data(self, do_filter: bool = True):
        return self._load_data("moves.json", do_filter=do_filter)

    def _load_abilities_data(self, do_filter: bool = True):
        return self._load_data("abilities.json", do_filter=do_filter)

    def _load_items_data(self, do_filter: bool = True):
        return self._load_data("items.json", do_filter=do_filter)

    def _load_learnset_data(self, do_filter: bool = True):
        return self._load_data("learnsets.json", do_filter=do_filter)

    def get_species_mask(self):
        numbered_species = self.data["species"]
        mask = np.zeros(len(numbered_species), dtype=bool)

        species_df = self._filter_df(self.species, "isMega", lambda x: not bool(x))
        species_df = self._filter_df(self.species, "forme", lambda x: x != "Gmax")
        species_df = self._filter_df(self.species, "forme", lambda x: x != "Starter")
        species_df = self._filter_df(
            self.species, "forme", lambda x: x != "Alola-Totem"
        )

        for species_id in species_df["id"].tolist():
            mask[numbered_species.get(species_id)] = True
        return mask

    def get_duplicate_mask(self):
        numbered_species = self.data["species"]
        num_species = len(numbered_species)
        mask = np.zeros((num_species, num_species), dtype=bool)

        one_hot = np.eye(num_species, num_species, dtype=bool)

        for species_id, base_species_id in self.species[
            ["id", "baseSpecies"]
        ].values.tolist():
            species_index = numbered_species.get(species_id)
            row_indices = self.species.id[
                self.species["baseSpecies"] == base_species_id
            ].map(lambda x: numbered_species[x])
            row_mask = one_hot[row_indices].any(axis=0)
            mask[species_index] = row_mask
        return mask

    def get_item_mask(self):
        numbered_species = self.data["species"]
        numbered_items = self.data["items"]

        mask = np.zeros((len(numbered_species), len(numbered_items)), dtype=bool)

        does_nothing_mask = (
            self.items["isPokeball"]
            | self.items["shortDesc"].map(
                lambda x: x == "Evolves certain species of pokemon when used"
            )
            | self.items["shortDesc"].map(lambda x: x.startswith("No competitive use"))
            | self.items["shortDesc"].map(lambda x: x.startswith("Can be revived into"))
            | self.items["shortDesc"].map(lambda x: x.endswith("when traded."))
            | self.items["shortDesc"].map(lambda x: x.startswith("Evolves "))
            | self.items["shortDesc"].map(
                lambda x: x.startswith("Teaches certain Pokemon the move ")
            )
        )
        competitive_use_mask = ~does_nothing_mask
        competitive_items = self.items[competitive_use_mask]

        competitive_mask = np.eye(len(numbered_items))[
            np.array(
                [
                    numbered_items.get(toid(item))
                    for item in competitive_items["id"].tolist()
                ]
            )
        ].sum(axis=0)

        for _, row in self.species.iterrows():
            row_index = numbered_species.get(toid(row["id"]))

            if isinstance(row.requiredItems, list):
                for item in row.requiredItems:
                    item_index = numbered_items.get(toid(item))
                    mask[row_index, item_index] = True

            else:
                mask[row_index] = competitive_mask

        num_extra_tokens = sum([k.startswith("_") for k in numbered_items.keys()])
        mask[numbered_species.get("_UNK"), num_extra_tokens:] = True

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

        num_extra_tokens = sum([k.startswith("_") for k in numbered_abilities.keys()])
        mask[numbered_species.get("_UNK"), num_extra_tokens:] = True

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

        num_extra_tokens = sum([k.startswith("_") for k in numbered_moves.keys()])
        mask[numbered_species.get("_UNK"), num_extra_tokens:] = True

        return mask


def save_gen_three():
    for generation in range(9, 2, -1):
        pokedex = Pokedex(generation=generation)

        for name, mask_arr in [
            ("duplicate", pokedex.get_duplicate_mask()),
            ("ability", pokedex.get_ability_mask()),
            ("item", pokedex.get_item_mask()),
            ("learnset", pokedex.get_learnset_mask()),
            ("species", pokedex.get_species_mask()),
        ]:
            print(
                f"Saving {name} mask for generation {generation}, shape {mask_arr.shape}, valid {mask_arr.any(axis=-1).sum()}"
            )

            with open(f"data/data/gen{generation}/{name}_mask.npy", "wb") as f:
                np.save(f, mask_arr)


def main():
    save_gen_three()


if __name__ == "__main__":
    main()
