import functools
import json
import os

import numpy as np

from embeddings.masks import toid
from rl.environment.data import NUM_PACKED_SET_FEATURES, NUM_SPECIES, STOI
from rl.environment.protos.enums_pb2 import SpeciesEnum
from rl.environment.protos.features_pb2 import PackedSetFeature


@functools.lru_cache(maxsize=None)
def encode_packed_set(generation: int, packed_set: str):
    """
    NICKNAME|SPECIES|ITEM|ABILITY|MOVES|NATURE|EVS|GENDER|IVS|SHINY|LEVEL|HAPPINESS,POKEBALL,HIDDENPOWERTYPE,GIGANTAMAX,DYNAMAXLEVEL,TERATYPE
    """
    arr = np.zeros((NUM_PACKED_SET_FEATURES,), dtype=np.int32)

    splits = packed_set.split("|")
    extras = splits[11].split(",")
    moves = splits[4].split(",")[:4]
    if len(moves) != 4:
        moves += ["_NULL"] * (4 - len(moves))

    nickname = splits[0]
    species = splits[1]
    item = splits[2]
    ability = splits[3]
    nature = splits[5]
    evs = [float(v) if v else 0 for v in splits[6].split(",")]
    gender = splits[7]
    ivs = [float(v) if v else 0 for v in splits[8].split(",")] if splits[8] else [0] * 6

    level = splits[10]

    arr[PackedSetFeature.PACKED_SET_FEATURE__SPECIES] = STOI["species"][
        nickname or species
    ]
    arr[PackedSetFeature.PACKED_SET_FEATURE__ITEM] = STOI["items"][item or "_NULL"]
    arr[PackedSetFeature.PACKED_SET_FEATURE__ABILITY] = STOI["abilities"][
        ability or "_NULL"
    ]
    arr[PackedSetFeature.PACKED_SET_FEATURE__NATURE] = STOI["natures"][
        nature.lower() or "_NULL"
    ]
    arr[PackedSetFeature.PACKED_SET_FEATURE__GENDER] = STOI["gendername"][
        gender or "_NULL"
    ]
    arr[PackedSetFeature.PACKED_SET_FEATURE__MOVE1] = STOI["moves"][moves[0] or "_NULL"]
    arr[PackedSetFeature.PACKED_SET_FEATURE__MOVE2] = STOI["moves"][moves[1] or "_NULL"]
    arr[PackedSetFeature.PACKED_SET_FEATURE__MOVE3] = STOI["moves"][moves[2] or "_NULL"]
    arr[PackedSetFeature.PACKED_SET_FEATURE__MOVE4] = STOI["moves"][moves[3] or "_NULL"]

    arr[PackedSetFeature.PACKED_SET_FEATURE__ATK_EV] = evs[1]
    arr[PackedSetFeature.PACKED_SET_FEATURE__DEF_EV] = evs[2]
    arr[PackedSetFeature.PACKED_SET_FEATURE__SPA_EV] = evs[3]
    arr[PackedSetFeature.PACKED_SET_FEATURE__SPD_EV] = evs[4]
    arr[PackedSetFeature.PACKED_SET_FEATURE__SPE_EV] = evs[5]
    arr[PackedSetFeature.PACKED_SET_FEATURE__HP_IV] = ivs[0]
    arr[PackedSetFeature.PACKED_SET_FEATURE__ATK_IV] = ivs[1]
    arr[PackedSetFeature.PACKED_SET_FEATURE__DEF_IV] = ivs[2]
    arr[PackedSetFeature.PACKED_SET_FEATURE__SPA_IV] = ivs[3]
    arr[PackedSetFeature.PACKED_SET_FEATURE__SPD_IV] = ivs[4]
    arr[PackedSetFeature.PACKED_SET_FEATURE__SPE_IV] = ivs[5]

    arr[PackedSetFeature.PACKED_SET_FEATURE__LEVEL] = level or 100

    if len(extras) > 5 and generation == 9:
        teratype = extras[5]
        arr[PackedSetFeature.PACKED_SET_FEATURE__TERATYPE] = STOI["typechart"][
            toid(teratype) or "_NULL"
        ]
    if len(extras) > 2 and generation < 8:
        hiddenpowertype = extras[2]
        arr[PackedSetFeature.PACKED_SET_FEATURE__HIDDENPOWERTYPE] = STOI["typechart"][
            toid(hiddenpowertype) or "_NULL"
        ]
    if len(extras) > 0:
        arr[PackedSetFeature.PACKED_SET_FEATURE__HAPPINESS] = extras[0] or 255

    return arr


def main(max_set_size: int = 1024):
    for generation in range(9, 0, -1):
        packed_set_fpaths = [
            f
            for f in os.listdir(f"data/data/gen{generation}")
            if f.endswith("all_formats.json") or f.endswith("only_format.json")
        ]
        for packed_set_fpath in packed_set_fpaths:
            read_path = f"data/data/gen{generation}/{packed_set_fpath}"

            with open(read_path, "r") as f:
                data = json.load(f)

            packed_set_mask = np.zeros((NUM_SPECIES, max_set_size), dtype=np.bool_)
            packed_set_features = np.zeros(
                (NUM_SPECIES, max_set_size, NUM_PACKED_SET_FEATURES), dtype=np.int32
            )

            packed_set_features[..., PackedSetFeature.PACKED_SET_FEATURE__SPECIES] = (
                SpeciesEnum.SPECIES_ENUM___NULL
            )

            for i, (species, sets) in enumerate(data.items()):
                for j, packed_set in enumerate(sets[:max_set_size]):
                    packed_set_features[i, j] = encode_packed_set(
                        generation, packed_set
                    )
                    packed_set_mask[i, j] = True

            print(
                f"Processed gen{generation}/{packed_set_fpath} with shape {packed_set_features.shape}"
            )

            np.save(read_path.replace(".json", "_mask.npy"), packed_set_mask)
            np.save(read_path.replace(".json", "_features.npy"), packed_set_features)

    return


if __name__ == "__main__":
    main()
