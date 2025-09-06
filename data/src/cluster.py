import numpy as np

from rl.environment.data import NUM_PACKED_SET_FEATURES, ONEHOT_ENCODERS, STOI
from rl.environment.protos.features_pb2 import PackedSetFeature


def to_numpy(packed_set: str) -> np.ndarray:
    """
    NICKNAME|SPECIES|ITEM|ABILITY|MOVES|NATURE|EVS|GENDER|IVS|SHINY|LEVEL|HAPPINESS,POKEBALL,HIDDENPOWERTYPE,GIGANTAMAX,DYNAMAXLEVEL,TERATYPE
    """
    (
        nickname,
        species,
        item,
        ability,
        moves,
        nature,
        evs,
        gender,
        ivs,
        shiny,
        level,
        extras,
    ) = packed_set.split("|")
    (
        happiness,
        pokeball,
        hiddenpowertype,
        gigantamax,
        dynamaxlevel,
        teratype,
    ) = extras.split(",")
    arr = np.zeros(NUM_PACKED_SET_FEATURES, dtype=np.int32)
    arr[PackedSetFeature.PACKED_SET_FEATURE__SPECIES] = int(STOI["species"][nickname])
    arr[PackedSetFeature.PACKED_SET_FEATURE__ITEM] = int(STOI["items"][item])
    arr[PackedSetFeature.PACKED_SET_FEATURE__ABILITY] = int(STOI["abilities"][ability])
    for i, move in enumerate(moves.split(";")):
        arr[PackedSetFeature.PACKED_SET_FEATURE__MOVE1 + i] = int(STOI["moves"][move])
    arr[PackedSetFeature.PACKED_SET_FEATURE__NATURE] = int(STOI["natures"][nature])
    for i, ev in enumerate(evs.split(",")):
        arr[PackedSetFeature.PACKED_SET_FEATURE__HP_EV + i] = int(ev)
    for i, iv in enumerate(ivs.split(",")):
        arr[PackedSetFeature.PACKED_SET_FEATURE__HP_IV + i] = int(iv)
    arr[PackedSetFeature.PACKED_SET_FEATURE__TERATYPE] = int(
        STOI["typechart"][teratype]
    )
    return arr


def to_embeddings(packed_sets: list[str]) -> np.ndarray:
    encodings = []
    for packed_set in packed_sets:
        encodings.append(to_numpy(packed_set))

    embeddings = ONEHOT_ENCODERS[9]["species"](
        encodings[..., PackedSetFeature.PACKED_SET_FEATURE__SPECIES]
    )

    return np.stack(encodings, axis=0)
