import os
import json
import traceback
import umap

import pandas as pd
import numpy as np
import plotly.express as px

from typing import Any, Dict, List, Sequence
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from embeddings.encoders import onehot_encode, z_score_scale
from embeddings.protocols import (
    ABILITIES_PROTOCOLS,
    ITEMS_PROTOCOLS,
    MOVES_PROTOCOLS,
    SPECIES_PROTOCOLS,
    Protocol,
)


def to_id(string: str) -> str:
    return "".join(c for c in string if c.isalnum()).lower()


def get_df(data: List[Dict[str, Any]], sortby: str = None):
    data = [
        d
        for d in data
        if (d.get("tier") != "Illegal")
        and (d.get("isNonstandard") not in ["Future", "Unobtainable"])
    ]

    df = pd.json_normalize(data)
    for mask_fn in [
        lambda: df["isNonstandard"].map(lambda x: x is None),
        lambda: (df["tier"] != "Illegal"),
    ]:
        try:
            mask = mask_fn()
            df = df[mask]
        except Exception:
            # traceback.print_exc()
            # return df
            pass

    cols_to_drop = []
    for column in df.columns:
        if len(df[column].map(lambda x: json.dumps(x)).unique()) <= 1:
            cols_to_drop.append(column)
    df = df.drop(cols_to_drop, axis=1)
    try:
        df.index = df["id"]
    except:
        pass

    if sortby is not None:
        df = df.sort_values("num")
    return df


class NoDataFramesError(Exception):
    pass


def concat_encodings(dataframes: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not dataframes:
        raise NoDataFramesError
    df = pd.concat([e.reset_index(drop=True) for e in dataframes], axis=1)
    df.index = dataframes[0].index
    return df


def get_encodings(df: pd.DataFrame, protocols: List[Protocol], verbose: bool = True):
    feature_vector_dfs = []
    for protoc in protocols:
        func = protoc["func"]
        feature = protoc.get("feature")
        feature_fn = protoc.get("feature_fn")

        if feature_fn is None:
            if feature not in df.columns:
                if verbose:
                    print(f"{feature} not in df")
                continue
            series = df[feature]
            feature_df = func(series)
            if not feature_df.empty:
                feature_vector_dfs.append(feature_df)
        else:
            for feature in df.columns:
                if feature not in df.columns:
                    if verbose:
                        print(f"{feature} not in df")
                    continue
                if feature_fn(feature):
                    series = df[feature]
                    feature_df = func(series)
                    if not feature_df.empty:
                        feature_vector_dfs.append(feature_df)

    concat_df = concat_encodings(feature_vector_dfs)
    concat_df.index = df["name"].map(to_id)

    return concat_df


class VectorContainsNanError(Exception):
    pass


def to_lookup_table(encodings_df: pd.DataFrame, stoi: Dict[str, int]) -> np.ndarray:

    padding_keys = [s for s in stoi.keys() if s.startswith("!") & s.endswith("!")]
    num_padding_keys = len(padding_keys)

    vector_length = encodings_df.shape[-1] + num_padding_keys

    store = np.zeros((len(stoi), vector_length))

    for key, row in encodings_df.iterrows():
        idx = stoi[key]
        nan_col_mask = pd.isna(row.values)
        if nan_col_mask.any():
            bad_columns = np.array(encodings_df.columns)[nan_col_mask]
            raise VectorContainsNanError(f"{key} vector contains nan in {bad_columns}")

        store[idx, num_padding_keys:] = row.values

    for key in padding_keys:
        idx = stoi[key]
        store[idx, idx] = 1

    return store


class GenerationEncodings:
    NAMES = ["abilities", "items", "move", "species", "typechart"]

    def __init__(
        self, gen: int, stoi: Dict[str, int] = None, onehot_id_only: bool = False
    ):
        if stoi is None:
            with open("data/data.json", "r", encoding="utf-8") as f:
                stoi = json.load(f)

        self.onehot_id_only = onehot_id_only
        self.gen = gen
        self.stoi = stoi

        self.gendata = {}
        gen_dir = f"data/data/gen{gen}/"
        for fname in os.listdir(gen_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(gen_dir, fname)
            fname_head, _ = os.path.splitext(fname)
            with open(fpath, "r") as f:
                self.gendata[fname_head] = json.load(f)

        self.typechart_df = get_df(self.gendata["typechart"])

    def get_species_df(self):
        df = get_df(self.gendata["species"])
        ability_columns = [
            column for column in df.columns if column.startswith("abilities")
        ]
        df["abilities"] = df.fillna("").apply(
            lambda row: [
                value for value in [row[column] for column in ability_columns] if value
            ],
            axis=1,
        )
        damage_taken_df = get_encodings(
            self.typechart_df,
            [
                {
                    "feature_fn": lambda x: x.startswith("damageTaken."),
                    "func": lambda x: x,
                }
            ],
        ).replace({0: 1, 1: 2, 2: 0.5, 3: 0})

        type_cols_to_remove = []
        if self.gen < 2:
            type_cols_to_remove += [".Dark", ".Steel"]

        if self.gen < 6:
            type_cols_to_remove += [".Fairy"]

        if type_cols_to_remove:
            damage_taken_df = damage_taken_df.drop(
                [
                    col
                    for col in damage_taken_df.columns
                    if any(col.endswith(suffix) for suffix in type_cols_to_remove)
                ],
                axis=1,
            )

        weakness_data = {k: [] for k in damage_taken_df.columns}
        for _, row in df.iterrows():
            types = row["types"]
            damage_multipliers = np.prod(
                damage_taken_df.loc[[t.lower() for t in types]].values, axis=0
            )
            for tidx, col in enumerate(damage_taken_df.columns):
                weakness_data[col].append(damage_multipliers[tidx])

        weakness_df = pd.DataFrame(weakness_data, index=df.index)
        weakness_df = get_encodings(
            concat_encodings([df, weakness_df]),
            [
                {
                    "feature_fn": lambda x: x.startswith("damageTaken."),
                    "func": onehot_encode,
                },
                {
                    "feature_fn": lambda x: x.startswith("damageTaken."),
                    "func": z_score_scale,
                },
            ],
        )
        encodings = get_encodings(
            concat_encodings([df, weakness_df]),
            (
                [{"feature": "id", "func": onehot_encode}]
                if self.onehot_id_only
                else SPECIES_PROTOCOLS
            ),
        )
        return to_lookup_table(encodings, self.stoi["species"])

    def get_moves_df(self):
        df = get_df(self.gendata["moves"])
        encodings = get_encodings(
            df,
            (
                [{"feature": "id", "func": onehot_encode}]
                if self.onehot_id_only
                else MOVES_PROTOCOLS
            ),
        )
        return to_lookup_table(encodings, self.stoi["moves"])

    def get_abilities_df(self):
        df = get_df(self.gendata["abilities"])
        encodings = get_encodings(
            df,
            (
                [{"feature": "id", "func": onehot_encode}]
                if self.onehot_id_only
                else ABILITIES_PROTOCOLS
            ),
        )
        return to_lookup_table(encodings, self.stoi["abilities"])

    def get_items_df(self):
        df = get_df(self.gendata["items"])
        encodings = get_encodings(
            df,
            (
                [{"feature": "id", "func": onehot_encode}]
                if self.onehot_id_only
                else ITEMS_PROTOCOLS
            ),
        )
        return to_lookup_table(encodings, self.stoi["items"])


def main(plot: bool = False, onehot_id_only: bool = False):
    with open("data/data/data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    name: str
    for gen in range(3, 10):
        print(gen)

        enc = GenerationEncodings(gen, data, onehot_id_only)
        for name, func in [
            ("species", enc.get_species_df),
            ("moves", enc.get_moves_df),
            ("abilities", enc.get_abilities_df),
            ("items", enc.get_items_df),
        ]:
            try:
                encodings_arr = func()
            except NoDataFramesError:
                traceback.print_exc()
                continue
            except VectorContainsNanError as e:
                raise e
            except Exception as e:
                raise e

            mask = abs(encodings_arr).sum(-1) > 0
            # pca = PCA(min(*encodings_arr[mask].shape, 128))
            # encoded = pca.fit_transform(encodings_arr[mask])
            encoded = encodings_arr[mask]
            # encoded = StandardScaler().fit_transform(encoded)
            # encoded = MinMaxScaler((-1, 1)).fit_transform(encoded)

            print(
                (
                    name,
                    # repr(encodings_arr.shape),
                    repr(encoded.shape),
                    encoded.min(),
                    encoded.max(),
                    # pca.explained_variance_ratio_[: pca.n_components_].sum(),
                )
            )

            new = np.zeros((encodings_arr.shape[0], encoded.shape[-1]))
            new[mask] = encoded
            with open(f"data/data/gen{gen}/{name}.npy", "wb") as f:
                np.save(f, new)

            if plot and gen == 3:
                cosine_sim = cosine_similarity(encoded)

                # Step 3: Plot the cosine similarity matrix as a heatmap
                names = np.array(list(enc.stoi[name]))[mask.flatten()]
                df_cosine_sim = pd.DataFrame(cosine_sim, columns=names, index=names)

                fig = px.imshow(
                    df_cosine_sim,
                    text_auto=True,
                    aspect="auto",
                    labels=dict(
                        x="Sample Index", y="Sample Index", color="Cosine Similarity"
                    ),
                    title=f"Gen{gen} {name.capitalize()} Cosine Similarity Heatmap",
                )
                fig.show()


if __name__ == "__main__":
    main(True)
