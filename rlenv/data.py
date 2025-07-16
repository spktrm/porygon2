import os

import jax.numpy as jnp

from rlenv.protos.enums_pb2 import (
    AbilitiesEnum,
    ActionsEnum,
    BattlemajorargsEnum,
    BattleminorargsEnum,
    BoostsEnum,
    EffectEnum,
    GendernameEnum,
    ItemeffecttypesEnum,
    ItemsEnum,
    LastitemeffecttypesEnum,
    MovesEnum,
    PseudoweatherEnum,
    SideconditionEnum,
    SpeciesEnum,
    StatusEnum,
    TerrainEnum,
    TypechartEnum,
    VolatilestatusEnum,
    WeatherEnum,
)
from rlenv.protos.features_pb2 import (
    AbsoluteEdgeFeature,
    ContextFeature,
    EntityFeature,
    MovesetActionTypeEnum,
    MovesetFeature,
    MovesetHasPPEnum,
    RelativeEdgeFeature,
)
from rlenv.protos.state_pb2 import State

with open(os.path.join(os.path.dirname(__file__), "ex"), "rb") as f:
    EX_BUFFER = f.read()

EX_STATE = State.FromString(EX_BUFFER)

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
NUM_ACTIONS = len(ActionsEnum.keys())
NUM_ACTION_TYPES = len(MovesetActionTypeEnum.keys())
NUM_HAS_PP = len(MovesetHasPPEnum.keys())
NUM_ABILITIES = len(AbilitiesEnum.keys())
NUM_ITEMS = len(ItemsEnum.keys())
NUM_MINOR_ARGS = len(BattleminorargsEnum.keys())
NUM_MAJOR_ARGS = len(BattlemajorargsEnum.keys())
NUM_ITEM_EFFECTS = len(ItemeffecttypesEnum.keys())
NUM_LAST_ITEM_EFFECTS = len(LastitemeffecttypesEnum.keys())
NUM_EFFECTS = len(EffectEnum.keys())
NUM_MOVE_FIELDS = len(MovesetFeature.keys())
NUM_RELATIVE_EDGE_FIELDS = len(RelativeEdgeFeature.keys())
NUM_ABSOLUTE_EDGE_FIELDS = len(AbsoluteEdgeFeature.keys())
NUM_ENTITY_FIELDS = len(EntityFeature.keys())
NUM_CONTEXT_FIELDS = len(ContextFeature.keys())

NUM_HISTORY = 384

SPIKES_TOKEN = SideconditionEnum.SIDECONDITION_ENUM__SPIKES
TOXIC_SPIKES_TOKEN = SideconditionEnum.SIDECONDITION_ENUM__TOXICSPIKES

MOVESET_ID_FEATURE_IDXS = jnp.array(
    [
        EntityFeature.ENTITY_FEATURE__MOVEID0,
        EntityFeature.ENTITY_FEATURE__MOVEID1,
        EntityFeature.ENTITY_FEATURE__MOVEID2,
        EntityFeature.ENTITY_FEATURE__MOVEID3,
    ],
    dtype=jnp.int32,
)

MOVESET_PP_FEATURE_IDXS = jnp.array(
    [
        EntityFeature.ENTITY_FEATURE__MOVEPP0,
        EntityFeature.ENTITY_FEATURE__MOVEPP1,
        EntityFeature.ENTITY_FEATURE__MOVEPP2,
        EntityFeature.ENTITY_FEATURE__MOVEPP3,
    ],
    dtype=jnp.int32,
)


MAX_RATIO_TOKEN = 16384
MAX_BOOST_VALUE = 13


ENTITY_MAX_VALUES = {
    EntityFeature.ENTITY_FEATURE__LEVEL: 100,
    EntityFeature.ENTITY_FEATURE__ACTIVE: 2,
    EntityFeature.ENTITY_FEATURE__IS_PUBLIC: 2,
    EntityFeature.ENTITY_FEATURE__SIDE: 2,
    EntityFeature.ENTITY_FEATURE__HP_RATIO: MAX_RATIO_TOKEN,
    EntityFeature.ENTITY_FEATURE__GENDER: NUM_GENDERS,
    EntityFeature.ENTITY_FEATURE__STATUS: NUM_STATUS,
    EntityFeature.ENTITY_FEATURE__ITEM_EFFECT: NUM_ITEM_EFFECTS,
    EntityFeature.ENTITY_FEATURE__BEING_CALLED_BACK: 2,
    EntityFeature.ENTITY_FEATURE__TRAPPED: 2,
    EntityFeature.ENTITY_FEATURE__NEWLY_SWITCHED: 2,
    EntityFeature.ENTITY_FEATURE__TOXIC_TURNS: 8,
    EntityFeature.ENTITY_FEATURE__SLEEP_TURNS: 4,
    EntityFeature.ENTITY_FEATURE__FAINTED: 2,
    EntityFeature.ENTITY_FEATURE__BOOST_ATK_VALUE: MAX_BOOST_VALUE,
    EntityFeature.ENTITY_FEATURE__BOOST_DEF_VALUE: MAX_BOOST_VALUE,
    EntityFeature.ENTITY_FEATURE__BOOST_SPA_VALUE: MAX_BOOST_VALUE,
    EntityFeature.ENTITY_FEATURE__BOOST_SPD_VALUE: MAX_BOOST_VALUE,
    EntityFeature.ENTITY_FEATURE__BOOST_SPE_VALUE: MAX_BOOST_VALUE,
    EntityFeature.ENTITY_FEATURE__BOOST_EVASION_VALUE: MAX_BOOST_VALUE,
    EntityFeature.ENTITY_FEATURE__BOOST_ACCURACY_VALUE: MAX_BOOST_VALUE,
}

RELATIVE_EDGE_MAX_VALUES = {
    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MAJOR_ARG: NUM_MAJOR_ARGS,
    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__DAMAGE_RATIO: MAX_RATIO_TOKEN,
    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__HEAL_RATIO: MAX_RATIO_TOKEN,
    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__STATUS_TOKEN: NUM_STATUS,
    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_ATK_VALUE: MAX_BOOST_VALUE,
    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_DEF_VALUE: MAX_BOOST_VALUE,
    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_SPA_VALUE: MAX_BOOST_VALUE,
    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_SPD_VALUE: MAX_BOOST_VALUE,
    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_SPE_VALUE: MAX_BOOST_VALUE,
    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_EVASION_VALUE: MAX_BOOST_VALUE,
    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_ACCURACY_VALUE: MAX_BOOST_VALUE,
    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SPIKES: 4,
    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__TOXIC_SPIKES: 2,
}


ABSOLUTE_EDGE_MAX_VALUES = {
    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_ID: NUM_WEATHER,
    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_MAX_DURATION: 9,
    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_MIN_DURATION: 9,
    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TERRAIN_ID: NUM_TERRAIN,
    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TERRAIN_MAX_DURATION: 9,
    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TERRAIN_MIN_DURATION: 9,
    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__PSEUDOWEATHER_ID: NUM_PSEUDOWEATHER,
    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__PSEUDOWEATHER_MAX_DURATION: 9,
    AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__PSEUDOWEATHER_MIN_DURATION: 9,
}

ACTION_MAX_VALUES = {
    MovesetFeature.MOVESET_FEATURE__ACTION_TYPE: NUM_ACTION_TYPES,
    MovesetFeature.MOVESET_FEATURE__HAS_PP: NUM_HAS_PP,
    MovesetFeature.MOVESET_FEATURE__PP: 64,
    MovesetFeature.MOVESET_FEATURE__MAXPP: 64,
}


ACTION_STRINGS = {v: k[8:] for k, v in ActionsEnum.items()}
