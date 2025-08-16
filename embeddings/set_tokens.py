import json

import numpy as np

from rl.environment.data import NUM_TOKEN_COLUMNS, STOI, TokenColumns


def unpack_set(packed_set: str):
    """
    NICKNAME|SPECIES|ITEM|ABILITY|MOVES|NATURE|EVS|GENDER|IVS|SHINY|LEVEL|HAPPINESS,POKEBALL,HIDDENPOWERTYPE,GIGANTAMAX,DYNAMAXLEVEL,TERATYPE
    """
    splits = packed_set.split("|")
    extras = splits[11].split(",")
    moves = splits[4].split(",")[:4]
    if len(moves) != 4:
        moves += ["_NULL"] * (4 - len(moves))
    unpacked_set = {
        "nickname": splits[0],
        "species": splits[1],
        "item": splits[2],
        "ability": splits[3],
        "moves": moves,
        "nature": splits[5],
        "evs": [float(v) if v else 0 for v in splits[6].split(",")],
        "gender": splits[7],
        "ivs": (
            [float(v) if v else 0 for v in splits[8].split(",")]
            if splits[8]
            else [0] * 6
        ),
        "shiny": splits[9],
        "level": splits[10],
    }
    if len(extras) > 5:
        unpacked_set["teratype"] = extras[5]
    if len(extras) > 4:
        unpacked_set["dynamaxlevel"] = extras[4]
    if len(extras) > 3:
        unpacked_set["gigantamax"] = extras[3]
    if len(extras) > 2:
        unpacked_set["hiddenpowertype"] = extras[2]
    if len(extras) > 1:
        unpacked_set["pokeball"] = extras[1]
    if len(extras) > 0:
        unpacked_set["happiness"] = extras[0]

    return unpacked_set


def main():
    for generation in range(1, 10):
        with open(f"data/data/validated_gen{generation}_packed.json", "r") as f:
            data = json.load(f)

        rows = []
        for packed_set, formats in data.items():
            row = np.zeros(NUM_TOKEN_COLUMNS)
            unpacked_set = unpack_set(packed_set)
            species = unpacked_set["nickname"] or unpacked_set["species"]
            row[TokenColumns.SPECIES.value] = STOI["species"][species]
            row[TokenColumns.ITEM.value] = STOI["items"][
                unpacked_set["item"] or "_NULL"
            ]
            row[TokenColumns.ABILITY.value] = STOI["abilities"][unpacked_set["ability"]]
            row[TokenColumns.MOVE1.value] = STOI["moves"][unpacked_set["moves"][0]]
            row[TokenColumns.MOVE2.value] = STOI["moves"][unpacked_set["moves"][1]]
            row[TokenColumns.MOVE3.value] = STOI["moves"][unpacked_set["moves"][2]]
            row[TokenColumns.MOVE4.value] = STOI["moves"][unpacked_set["moves"][3]]
            if "nature" in unpacked_set:
                row[TokenColumns.NATURE.value] = STOI["natures"][
                    unpacked_set["nature"] or "_NULL"
                ]
            else:
                row[TokenColumns.HIDDENPOWERTYPE.value] = STOI["natures"]["serious"]
            row[TokenColumns.GENDER.value] = STOI["gendername"][
                unpacked_set["gender"] or "_NULL"
            ]
            row[TokenColumns.HP_EV.value] = unpacked_set["evs"][0] or 0
            row[TokenColumns.ATK_EV.value] = unpacked_set["evs"][1] or 0
            row[TokenColumns.DEF_EV.value] = unpacked_set["evs"][2] or 0
            row[TokenColumns.SPA_EV.value] = unpacked_set["evs"][3] or 0
            row[TokenColumns.SPD_EV.value] = unpacked_set["evs"][4] or 0
            row[TokenColumns.SPE_EV.value] = unpacked_set["evs"][5] or 0
            row[TokenColumns.HP_IV.value] = unpacked_set["ivs"][0] or 0
            row[TokenColumns.ATK_IV.value] = unpacked_set["ivs"][1] or 0
            row[TokenColumns.DEF_IV.value] = unpacked_set["ivs"][2] or 0
            row[TokenColumns.SPA_IV.value] = unpacked_set["ivs"][3] or 0
            row[TokenColumns.SPD_IV.value] = unpacked_set["ivs"][4] or 0
            row[TokenColumns.SPE_IV.value] = unpacked_set["ivs"][5] or 0
            if "hiddenpowertype" in unpacked_set:
                row[TokenColumns.HIDDENPOWERTYPE.value] = STOI["typechart"][
                    unpacked_set["hiddenpowertype"]
                ]
            else:
                row[TokenColumns.HIDDENPOWERTYPE.value] = STOI["typechart"]["_NULL"]
            if "teratype" in unpacked_set:
                row[TokenColumns.TERATYPE.value] = STOI["typechart"][
                    unpacked_set["teratype"]
                ]
            else:
                row[TokenColumns.HIDDENPOWERTYPE.value] = STOI["typechart"]["_NULL"]

            rows.append(row)

        rows = np.stack(rows)

        with open(f"data/data/gen{generation}/packed_sets.npy", "wb") as f:
            np.save(f, rows)

    return


if __name__ == "__main__":
    main()
