import functools
import json
import os
import traceback

import jax.numpy as jnp
import numpy as np
import requests

from rl.environment.protos.enums_pb2 import (
    AbilitiesEnum,
    BattlemajorargsEnum,
    BattleminorargsEnum,
    BoostsEnum,
    EffectEnum,
    GendernameEnum,
    ItemeffecttypesEnum,
    ItemsEnum,
    LastitemeffecttypesEnum,
    MovesEnum,
    NaturesEnum,
    PseudoweatherEnum,
    SideconditionEnum,
    SpeciesEnum,
    StatusEnum,
    TerrainEnum,
    TypechartEnum,
    VolatilestatusEnum,
    WeatherEnum,
)
from rl.environment.protos.features_pb2 import (
    ActionMaskFeature,
    ActionType,
    EntityEdgeFeature,
    EntityNodeFeature,
    FieldFeature,
    MovesetFeature,
    MovesetHasPP,
    PackedSetFeature,
)
from rl.environment.protos.service_pb2 import EnvironmentTrajectory
from rl.model.modules import PretrainedEmbedding

NUM_GENDERS = len(GendernameEnum.keys())
NUM_STATUS = len(StatusEnum.keys())
# NUM_TYPES = len(TypesEnum.keys())
NUM_VOLATILE_STATUS = len(VolatilestatusEnum.keys())
NUM_TYPECHART = len(TypechartEnum.keys())
NUM_SIDE_CONDITION = len(SideconditionEnum.keys())
NUM_BOOSTS = len(BoostsEnum.keys())
NUM_PSEUDOWEATHER = len(PseudoweatherEnum.keys())
NUM_WEATHER = len(WeatherEnum.keys())
NUM_TERRAIN = len(TerrainEnum.keys())
NUM_SPECIES = len(SpeciesEnum.keys())
NUM_MOVES = len(MovesEnum.keys())
NUM_FROM_SOURCE_EFFECTS = len(EffectEnum.keys())
NUM_ACTION_TYPES = len(ActionType.keys())
NUM_HAS_PP = len(MovesetHasPP.keys())
NUM_ABILITIES = len(AbilitiesEnum.keys())
NUM_ITEMS = len(ItemsEnum.keys())
NUM_MINOR_ARGS = len(BattleminorargsEnum.keys())
NUM_MAJOR_ARGS = len(BattlemajorargsEnum.keys())
NUM_ITEM_EFFECTS = len(ItemeffecttypesEnum.keys())
NUM_NATURES = len(NaturesEnum.keys())
NUM_LAST_ITEM_EFFECTS = len(LastitemeffecttypesEnum.keys())
NUM_EFFECTS = len(EffectEnum.keys())
NUM_MOVE_FEATURES = len(MovesetFeature.keys())
NUM_ENTITY_EDGE_FEATURES = len(EntityEdgeFeature.keys())
NUM_FIELD_FEATURES = len(FieldFeature.keys())
NUM_ENTITY_NODE_FEATURES = len(EntityNodeFeature.keys())
NUM_ACTION_MASK_FEATURES = len(ActionMaskFeature.keys())

NUM_HISTORY = 384

SPIKES_TOKEN = SideconditionEnum.SIDECONDITION_ENUM__SPIKES
TOXIC_SPIKES_TOKEN = SideconditionEnum.SIDECONDITION_ENUM__TOXICSPIKES

MOVESET_ID_FEATURE_IDXS = jnp.array(
    [
        EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEID0,
        EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEID1,
        EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEID2,
        EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEID3,
    ],
    dtype=jnp.int32,
)

MOVESET_PP_FEATURE_IDXS = jnp.array(
    [
        EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEPP0,
        EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEPP1,
        EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEPP2,
        EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEPP3,
    ],
    dtype=jnp.int32,
)


MAX_RATIO_TOKEN = 16384
MAX_BOOST_VALUE = 13


ENTITY_NODE_MAX_VALUES = {
    EntityNodeFeature.ENTITY_NODE_FEATURE__LEVEL: 100,
    EntityNodeFeature.ENTITY_NODE_FEATURE__ACTIVE: 2,
    EntityNodeFeature.ENTITY_NODE_FEATURE__IS_PUBLIC: 2,
    EntityNodeFeature.ENTITY_NODE_FEATURE__SIDE: 2,
    EntityNodeFeature.ENTITY_NODE_FEATURE__HP_RATIO: MAX_RATIO_TOKEN,
    EntityNodeFeature.ENTITY_NODE_FEATURE__GENDER: NUM_GENDERS,
    EntityNodeFeature.ENTITY_NODE_FEATURE__STATUS: NUM_STATUS,
    EntityNodeFeature.ENTITY_NODE_FEATURE__ITEM_EFFECT: NUM_ITEM_EFFECTS,
    EntityNodeFeature.ENTITY_NODE_FEATURE__BEING_CALLED_BACK: 2,
    EntityNodeFeature.ENTITY_NODE_FEATURE__TRAPPED: 2,
    EntityNodeFeature.ENTITY_NODE_FEATURE__NEWLY_SWITCHED: 2,
    EntityNodeFeature.ENTITY_NODE_FEATURE__TOXIC_TURNS: 8,
    EntityNodeFeature.ENTITY_NODE_FEATURE__SLEEP_TURNS: 4,
    EntityNodeFeature.ENTITY_NODE_FEATURE__FAINTED: 2,
    EntityNodeFeature.ENTITY_NODE_FEATURE__NATURE: NUM_NATURES,
    EntityNodeFeature.ENTITY_NODE_FEATURE__TERA_TYPE: NUM_TYPECHART,
    EntityNodeFeature.ENTITY_NODE_FEATURE__TERASTALLIZED: 2,
}

ENTITY_EDGE_MAX_VALUES = {
    EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG: NUM_MAJOR_ARGS,
    EntityEdgeFeature.ENTITY_EDGE_FEATURE__DAMAGE_RATIO: MAX_RATIO_TOKEN,
    EntityEdgeFeature.ENTITY_EDGE_FEATURE__HEAL_RATIO: MAX_RATIO_TOKEN,
    EntityEdgeFeature.ENTITY_EDGE_FEATURE__STATUS_TOKEN: NUM_STATUS,
}


FIELD_MAX_VALUES = {
    FieldFeature.FIELD_FEATURE__WEATHER_ID: NUM_WEATHER,
    FieldFeature.FIELD_FEATURE__WEATHER_MAX_DURATION: 9,
    FieldFeature.FIELD_FEATURE__WEATHER_MIN_DURATION: 9,
    FieldFeature.FIELD_FEATURE__TERRAIN_ID: NUM_TERRAIN,
    FieldFeature.FIELD_FEATURE__TERRAIN_MAX_DURATION: 9,
    FieldFeature.FIELD_FEATURE__TERRAIN_MIN_DURATION: 9,
    FieldFeature.FIELD_FEATURE__PSEUDOWEATHER_ID: NUM_PSEUDOWEATHER,
    FieldFeature.FIELD_FEATURE__PSEUDOWEATHER_MAX_DURATION: 9,
    FieldFeature.FIELD_FEATURE__PSEUDOWEATHER_MIN_DURATION: 9,
    FieldFeature.FIELD_FEATURE__MY_SPIKES: 4,
    FieldFeature.FIELD_FEATURE__OPP_SPIKES: 4,
    FieldFeature.FIELD_FEATURE__MY_TOXIC_SPIKES: 2,
    FieldFeature.FIELD_FEATURE__OPP_TOXIC_SPIKES: 2,
}

ACTION_MAX_VALUES = {
    MovesetFeature.MOVESET_FEATURE__ACTION_TYPE: NUM_ACTION_TYPES,
    MovesetFeature.MOVESET_FEATURE__HAS_PP: NUM_HAS_PP,
    MovesetFeature.MOVESET_FEATURE__PP: 64,
    MovesetFeature.MOVESET_FEATURE__MAXPP: 64,
}

with open("data/data/data.json", "r") as f:
    token_data = json.load(f)


PACKED_SET_MAX_VALUES = {
    PackedSetFeature.PACKED_SET_FEATURE__GENDER: NUM_GENDERS,
    PackedSetFeature.PACKED_SET_FEATURE__NATURE: NUM_NATURES,
    PackedSetFeature.PACKED_SET_FEATURE__HIDDENPOWERTYPE: NUM_TYPECHART,
    PackedSetFeature.PACKED_SET_FEATURE__TERATYPE: NUM_TYPECHART,
}

ITOS = {key.lower(): {v: k for k, v in token_data[key].items()} for key in token_data}
STOI = {key.lower(): {k: v for k, v in token_data[key].items()} for key in token_data}


def toid(string: str) -> str:
    return "".join(c for c in string if c.isalnum() or c == "_").lower()


@functools.lru_cache(maxsize=None)
def do_request(url: str, **kwargs):
    return requests.get(url, **kwargs).json()


def make_species_count(generation: int, smogon_format: str):
    data = do_request(
        f"https://raw.githubusercontent.com/pkmn/smogon/refs/heads/main/data/stats/gen{generation}{smogon_format}.json"
    )
    total_count = sum(d["count"] for d in data["pokemon"].values())
    species_count = np.zeros(NUM_SPECIES, dtype=np.float32)
    for k, v in data["pokemon"].items():
        species_id = STOI["species"].get(toid(k), 0)
        species_count[species_id] = v["count"] / total_count
    return species_count


def make_species_count(generation: int, smogon_format: str):
    data = do_request(
        f"https://raw.githubusercontent.com/pkmn/smogon/refs/heads/main/data/stats/gen{generation}{smogon_format}.json"
    )
    species_count = np.zeros(NUM_SPECIES, dtype=np.float32)
    for k, v in data["pokemon"].items():
        species_id = STOI["species"].get(toid(k), 0)
        species_count[species_id] = v["usage"]["raw"]
    return species_count


def make_teammate_count(generation: int, smogon_format: str):
    data = do_request(
        f"https://raw.githubusercontent.com/pkmn/smogon/refs/heads/main/data/stats/gen{generation}{smogon_format}.json"
    )
    teammate_count = np.zeros((NUM_SPECIES, NUM_SPECIES), dtype=np.float32)
    for k, v in data["pokemon"].items():
        species_id = STOI["species"].get(toid(k), 0)
        for teammate, rew in v["teammates"].items():
            teammate_id = STOI["species"].get(toid(teammate), 0)
            teammate_count[species_id, teammate_id] = rew
    return teammate_count


HUMAN_SPECIES_COUNTS = {
    generation: {
        smogon_format: make_species_count(generation, smogon_format)
        for smogon_format in ["ou"]
    }
    for generation in range(1, 10)
}

HUMAN_TEAMMATE_COUNTS = {
    generation: {
        smogon_format: make_teammate_count(generation, smogon_format)
        for smogon_format in ["ou"]
    }
    for generation in range(1, 10)
}

NUM_PACKED_SET_FEATURES = len(PackedSetFeature.keys())


ONEHOT_DTYPE = jnp.bfloat16

try:
    MASKS = {
        generation: {
            "species": jnp.asarray(
                np.load(
                    f"data/data/gen{generation}/species_mask.npy",
                )
            ).astype(ONEHOT_DTYPE),
            "abilities": jnp.asarray(
                np.load(
                    f"data/data/gen{generation}/ability_mask.npy",
                )
            ).astype(ONEHOT_DTYPE),
            "items": jnp.asarray(
                np.load(
                    f"data/data/gen{generation}/item_mask.npy",
                )
            ).astype(ONEHOT_DTYPE),
            "learnset": jnp.asarray(
                np.load(
                    f"data/data/gen{generation}/learnset_mask.npy",
                )
            ).astype(ONEHOT_DTYPE),
            "duplicate": jnp.asarray(
                np.load(
                    f"data/data/gen{generation}/duplicate_mask.npy",
                )
            ),
        }
        for generation in range(3, 10)
    }

    SET_TOKENS = {
        generation: {
            f"{smogon_format}_{suffix}": jnp.asarray(
                np.load(
                    f"data/data/gen{generation}/{smogon_format}_{suffix}_features.npy",
                )
            )
            for smogon_format in ["ou"]
            for suffix in ["all_formats", "only_format"]
        }
        for generation in range(9, 10)
    }
    SET_MASK = {
        generation: {
            f"{smogon_format}_{suffix}": jnp.asarray(
                np.load(
                    f"data/data/gen{generation}/{smogon_format}_{suffix}_mask.npy",
                )
            )
            for smogon_format in ["ou"]
            for suffix in ["all_formats", "only_format"]
        }
        for generation in range(9, 10)
    }
except:
    traceback.print_exc()

ONEHOT_ENCODERS = {
    generation: {
        "species": PretrainedEmbedding(
            fpath=f"data/data/gen{generation}/species.npy",
            dtype=ONEHOT_DTYPE,
        ),
        "abilities": PretrainedEmbedding(
            fpath=f"data/data/gen{generation}/abilities.npy",
            dtype=ONEHOT_DTYPE,
        ),
        "items": PretrainedEmbedding(
            fpath=f"data/data/gen{generation}/items.npy", dtype=ONEHOT_DTYPE
        ),
        "moves": PretrainedEmbedding(
            fpath=f"data/data/gen{generation}/moves.npy", dtype=ONEHOT_DTYPE
        ),
        "learnset": PretrainedEmbedding(
            fpath=f"data/data/gen{generation}/learnset.npy", dtype=ONEHOT_DTYPE
        ),
    }
    for generation in range(3, 10)
}

with open(os.path.join(os.path.dirname(__file__), "ex.bin"), "rb") as f:
    EX_BUFFER = f.read()


EX_TRAJECTORY = EnvironmentTrajectory.FromString(EX_BUFFER)
