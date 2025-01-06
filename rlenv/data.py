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
    EdgeFromTypes,
    EdgeTypes,
    FeatureEdge,
    FeatureEntity,
    FeatureMoveset,
    MovesetActionType,
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
NUM_ACTIONS = len(ActionsEnum.keys())
NUM_ACTION_TYPES = len(MovesetActionType.keys())
NUM_ABILITIES = len(AbilitiesEnum.keys())
NUM_ITEMS = len(ItemsEnum.keys())
NUM_MINOR_ARGS = len(BattleminorargsEnum.keys())
NUM_MAJOR_ARGS = len(BattlemajorargsEnum.keys())
NUM_ITEM_EFFECTS = len(ItemeffecttypesEnum.keys())
NUM_LAST_ITEM_EFFECTS = len(LastitemeffecttypesEnum.keys())
NUM_EDGE_FROM_TYPES = len(EdgeFromTypes.keys())
NUM_EDGE_TYPES = len(EdgeTypes.keys())
NUM_EFFECTS = len(EffectEnum.keys())
NUM_MOVE_FIELDS = len(FeatureMoveset.keys())
NUM_EDGE_FIELDS = len(FeatureEdge.keys())
NUM_ENTITY_FIELDS = len(FeatureEntity.keys())

NUM_HISTORY = 32

SPIKES_TOKEN = SideconditionEnum.SIDECONDITION_SPIKES
TOXIC_SPIKES_TOKEN = SideconditionEnum.SIDECONDITION_TOXICSPIKES

MOVESET_ID_FEATURE_IDXS = jnp.array(
    [
        FeatureEntity.ENTITY_MOVEID0,
        FeatureEntity.ENTITY_MOVEID1,
        FeatureEntity.ENTITY_MOVEID2,
        FeatureEntity.ENTITY_MOVEID3,
    ],
    dtype=jnp.int32,
)

MOVESET_PP_FEATURE_IDXS = jnp.array(
    [
        FeatureEntity.ENTITY_MOVEPP0,
        FeatureEntity.ENTITY_MOVEPP1,
        FeatureEntity.ENTITY_MOVEPP2,
        FeatureEntity.ENTITY_MOVEPP3,
    ],
    dtype=jnp.int32,
)


MAX_BOOST_VALUE = 13


ENTITY_MAX_VALUES = {
    FeatureEntity.ENTITY_SIDE: 2,
    FeatureEntity.ENTITY_LEVEL: 100,
    FeatureEntity.ENTITY_HP_RATIO: 31,
    FeatureEntity.ENTITY_GENDER: NUM_GENDERS,
    FeatureEntity.ENTITY_STATUS: NUM_STATUS,
    FeatureEntity.ENTITY_ITEM_EFFECT: NUM_ITEM_EFFECTS,
    FeatureEntity.ENTITY_BEING_CALLED_BACK: 2,
    FeatureEntity.ENTITY_TRAPPED: 2,
    FeatureEntity.ENTITY_NEWLY_SWITCHED: 2,
    FeatureEntity.ENTITY_TOXIC_TURNS: 8,
    FeatureEntity.ENTITY_SLEEP_TURNS: 4,
    FeatureEntity.ENTITY_FAINTED: 2,
    FeatureEntity.ENTITY_ACTIVE: 2,
    FeatureEntity.ENTITY_BOOST_ATK_VALUE: MAX_BOOST_VALUE,
    FeatureEntity.ENTITY_BOOST_DEF_VALUE: MAX_BOOST_VALUE,
    FeatureEntity.ENTITY_BOOST_SPA_VALUE: MAX_BOOST_VALUE,
    FeatureEntity.ENTITY_BOOST_SPD_VALUE: MAX_BOOST_VALUE,
    FeatureEntity.ENTITY_BOOST_SPE_VALUE: MAX_BOOST_VALUE,
    FeatureEntity.ENTITY_BOOST_EVASION_VALUE: MAX_BOOST_VALUE,
    FeatureEntity.ENTITY_BOOST_ACCURACY_VALUE: MAX_BOOST_VALUE,
}

EDGE_MAX_VALUES = {
    FeatureEdge.MAJOR_ARG: NUM_MAJOR_ARGS,
    FeatureEdge.MINOR_ARG: NUM_MINOR_ARGS,
    FeatureEdge.FROM_SOURCE_TOKEN: NUM_EFFECTS,
    FeatureEdge.FROM_TYPE_TOKEN: NUM_EDGE_FROM_TYPES,
    FeatureEdge.EDGE_TYPE_TOKEN: NUM_EDGE_TYPES,
    FeatureEdge.DAMAGE_RATIO: 31,
    FeatureEdge.HEAL_RATIO: 31,
    FeatureEdge.STATUS_TOKEN: NUM_STATUS,
    FeatureEdge.BOOST_ATK_VALUE: MAX_BOOST_VALUE,
    FeatureEdge.BOOST_DEF_VALUE: MAX_BOOST_VALUE,
    FeatureEdge.BOOST_SPA_VALUE: MAX_BOOST_VALUE,
    FeatureEdge.BOOST_SPD_VALUE: MAX_BOOST_VALUE,
    FeatureEdge.BOOST_SPE_VALUE: MAX_BOOST_VALUE,
    FeatureEdge.BOOST_EVASION_VALUE: MAX_BOOST_VALUE,
    FeatureEdge.BOOST_ACCURACY_VALUE: MAX_BOOST_VALUE,
}
