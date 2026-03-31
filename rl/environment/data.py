import json
import os
import traceback

import jax.numpy as jnp
import numpy as np

from constants import NUM_HISTORY
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
    ActionType,
    EntityEdgeFeature,
    EntityPrivateNodeFeature,
    EntityPublicNodeFeature,
    EntityRevealedNodeFeature,
    FieldFeature,
    MovesetFeature,
    MovesetHasPP,
    PackedSetFeature,
)
from rl.environment.protos.service_pb2 import ActionEnum, EnvironmentTrajectory
from rl.model.modules import PretrainedEmbedding, ZeroEmbedding

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
NUM_ENTITY_PRIVATE_FEATURES = len(EntityPrivateNodeFeature.keys())
NUM_ENTITY_PUBLIC_FEATURES = len(EntityPublicNodeFeature.keys())
NUM_ENTITY_REVEALED_FEATURES = len(EntityRevealedNodeFeature.keys())
NUM_ACTION_FEATURES = len(ActionEnum.keys())

SPIKES_TOKEN = SideconditionEnum.SIDECONDITION_ENUM__SPIKES
TOXIC_SPIKES_TOKEN = SideconditionEnum.SIDECONDITION_ENUM__TOXICSPIKES


MAX_RATIO_TOKEN = 16384


ENTITY_PUBLIC_MAX_VALUES = {
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__LEVEL: 100,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE: 3,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE: 2,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO: MAX_RATIO_TOKEN,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__GENDER: NUM_GENDERS,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__STATUS: NUM_STATUS,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ITEM_EFFECT: NUM_ITEM_EFFECTS,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BEING_CALLED_BACK: 2,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TRAPPED: 2,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__NEWLY_SWITCHED: 2,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TOXIC_TURNS: 8,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SLEEP_TURNS: 4,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED: 2,
}

ENTITY_PRIVATE_MAX_VALUES = {
    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__TERA_TYPE: NUM_TYPECHART,
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
    MovesetFeature.MOVESET_FEATURE__DISABLED: 2,
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


VALID_GENERATIONS = [1, 9]


NUM_PACKED_SET_FEATURES = len(PackedSetFeature.keys())


ONEHOT_DTYPE = jnp.bfloat16


CAT_VF_SUPPORT = np.array([-1, -0.1, 1], dtype=np.float32)


def add_pretrained_embedding(generation):
    tables = {}
    for name in ["species", "abilities", "items", "moves", "learnset"]:
        try:
            tables[name] = PretrainedEmbedding(
                fpath=f"data/data/gen{generation}/{name}.npy",
                dtype=ONEHOT_DTYPE,
            )
        except:
            traceback.print_exc()
            tables[name] = ZeroEmbedding(dtype=ONEHOT_DTYPE)
    return tables


ONEHOT_ENCODERS = {
    generation: add_pretrained_embedding(generation) for generation in VALID_GENERATIONS
}

with open(os.path.join(os.path.dirname(__file__), "ex.bin"), "rb") as f:
    EX_BUFFER = f.read()


EX_TRAJECTORY = EnvironmentTrajectory.FromString(EX_BUFFER)
