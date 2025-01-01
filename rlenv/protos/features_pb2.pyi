from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

class FeatureEntity(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ENTITY_SPECIES: _ClassVar[FeatureEntity]
    ENTITY_ITEM: _ClassVar[FeatureEntity]
    ENTITY_ITEM_EFFECT: _ClassVar[FeatureEntity]
    ENTITY_ABILITY: _ClassVar[FeatureEntity]
    ENTITY_GENDER: _ClassVar[FeatureEntity]
    ENTITY_ACTIVE: _ClassVar[FeatureEntity]
    ENTITY_FAINTED: _ClassVar[FeatureEntity]
    ENTITY_HP: _ClassVar[FeatureEntity]
    ENTITY_MAXHP: _ClassVar[FeatureEntity]
    ENTITY_STATUS: _ClassVar[FeatureEntity]
    ENTITY_TOXIC_TURNS: _ClassVar[FeatureEntity]
    ENTITY_SLEEP_TURNS: _ClassVar[FeatureEntity]
    ENTITY_BEING_CALLED_BACK: _ClassVar[FeatureEntity]
    ENTITY_TRAPPED: _ClassVar[FeatureEntity]
    ENTITY_NEWLY_SWITCHED: _ClassVar[FeatureEntity]
    ENTITY_LEVEL: _ClassVar[FeatureEntity]
    ENTITY_MOVEID0: _ClassVar[FeatureEntity]
    ENTITY_MOVEID1: _ClassVar[FeatureEntity]
    ENTITY_MOVEID2: _ClassVar[FeatureEntity]
    ENTITY_MOVEID3: _ClassVar[FeatureEntity]
    ENTITY_MOVEPP0: _ClassVar[FeatureEntity]
    ENTITY_MOVEPP1: _ClassVar[FeatureEntity]
    ENTITY_MOVEPP2: _ClassVar[FeatureEntity]
    ENTITY_MOVEPP3: _ClassVar[FeatureEntity]
    ENTITY_HAS_STATUS: _ClassVar[FeatureEntity]
    ENTITY_BOOST_ATK_VALUE: _ClassVar[FeatureEntity]
    ENTITY_BOOST_DEF_VALUE: _ClassVar[FeatureEntity]
    ENTITY_BOOST_SPA_VALUE: _ClassVar[FeatureEntity]
    ENTITY_BOOST_SPD_VALUE: _ClassVar[FeatureEntity]
    ENTITY_BOOST_SPE_VALUE: _ClassVar[FeatureEntity]
    ENTITY_BOOST_ACCURACY_VALUE: _ClassVar[FeatureEntity]
    ENTITY_BOOST_EVASION_VALUE: _ClassVar[FeatureEntity]
    ENTITY_VOLATILES0: _ClassVar[FeatureEntity]
    ENTITY_VOLATILES1: _ClassVar[FeatureEntity]
    ENTITY_VOLATILES2: _ClassVar[FeatureEntity]
    ENTITY_VOLATILES3: _ClassVar[FeatureEntity]
    ENTITY_VOLATILES4: _ClassVar[FeatureEntity]
    ENTITY_VOLATILES5: _ClassVar[FeatureEntity]
    ENTITY_VOLATILES6: _ClassVar[FeatureEntity]
    ENTITY_VOLATILES7: _ClassVar[FeatureEntity]
    ENTITY_VOLATILES8: _ClassVar[FeatureEntity]
    ENTITY_SIDE: _ClassVar[FeatureEntity]
    ENTITY_HP_TOKEN: _ClassVar[FeatureEntity]

class MovesetActionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MOVESET_ACTION_TYPE_MOVE: _ClassVar[MovesetActionType]
    MOVESET_ACTION_TYPE_SWITCH: _ClassVar[MovesetActionType]

class FeatureMoveset(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MOVESET_ACTION_ID: _ClassVar[FeatureMoveset]
    MOVESET_PPUSED: _ClassVar[FeatureMoveset]
    MOVESET_LEGAL: _ClassVar[FeatureMoveset]
    MOVESET_SIDE: _ClassVar[FeatureMoveset]
    MOVESET_ACTION_TYPE: _ClassVar[FeatureMoveset]
    MOVESET_EST_DAMAGE: _ClassVar[FeatureMoveset]
    MOVESET_MOVE_ID: _ClassVar[FeatureMoveset]
    MOVESET_SPECIES_ID: _ClassVar[FeatureMoveset]

class FeatureTurnContext(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VALID: _ClassVar[FeatureTurnContext]
    IS_MY_TURN: _ClassVar[FeatureTurnContext]
    ACTION: _ClassVar[FeatureTurnContext]
    MOVE: _ClassVar[FeatureTurnContext]
    SWITCH_COUNTER: _ClassVar[FeatureTurnContext]
    MOVE_COUNTER: _ClassVar[FeatureTurnContext]
    TURN: _ClassVar[FeatureTurnContext]

class FeatureWeather(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WEATHER_ID: _ClassVar[FeatureWeather]
    MIN_DURATION: _ClassVar[FeatureWeather]
    MAX_DURATION: _ClassVar[FeatureWeather]

class FeatureAdditionalInformation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NUM_FAINTED: _ClassVar[FeatureAdditionalInformation]
    HP_TOTAL: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_PAD: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_UNK: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_BUG: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_DARK: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_DRAGON: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_ELECTRIC: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_FAIRY: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_FIGHTING: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_FIRE: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_FLYING: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_GHOST: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_GRASS: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_GROUND: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_ICE: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_NORMAL: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_POISON: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_PSYCHIC: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_ROCK: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_STEEL: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_STELLAR: _ClassVar[FeatureAdditionalInformation]
    NUM_TYPES_WATER: _ClassVar[FeatureAdditionalInformation]
    TOTAL_POKEMON: _ClassVar[FeatureAdditionalInformation]
    WISHING: _ClassVar[FeatureAdditionalInformation]
    MEMBER0_HP: _ClassVar[FeatureAdditionalInformation]
    MEMBER1_HP: _ClassVar[FeatureAdditionalInformation]
    MEMBER2_HP: _ClassVar[FeatureAdditionalInformation]
    MEMBER3_HP: _ClassVar[FeatureAdditionalInformation]
    MEMBER4_HP: _ClassVar[FeatureAdditionalInformation]
    MEMBER5_HP: _ClassVar[FeatureAdditionalInformation]

class EdgeTypes(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EDGE_TYPE_NONE: _ClassVar[EdgeTypes]
    MOVE_EDGE: _ClassVar[EdgeTypes]
    SWITCH_EDGE: _ClassVar[EdgeTypes]
    EFFECT_EDGE: _ClassVar[EdgeTypes]
    CANT_EDGE: _ClassVar[EdgeTypes]
    EDGE_TYPE_START: _ClassVar[EdgeTypes]

class EdgeFromTypes(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EDGE_FROM_NONE: _ClassVar[EdgeFromTypes]
    EDGE_FROM_ITEM: _ClassVar[EdgeFromTypes]
    EDGE_FROM_EFFECT: _ClassVar[EdgeFromTypes]
    EDGE_FROM_MOVE: _ClassVar[EdgeFromTypes]
    EDGE_FROM_ABILITY: _ClassVar[EdgeFromTypes]
    EDGE_FROM_SIDECONDITION: _ClassVar[EdgeFromTypes]
    EDGE_FROM_STATUS: _ClassVar[EdgeFromTypes]
    EDGE_FROM_WEATHER: _ClassVar[EdgeFromTypes]
    EDGE_FROM_TERRAIN: _ClassVar[EdgeFromTypes]
    EDGE_FROM_PSEUDOWEATHER: _ClassVar[EdgeFromTypes]

class FeatureEdge(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TURN_ORDER_VALUE: _ClassVar[FeatureEdge]
    EDGE_TYPE_TOKEN: _ClassVar[FeatureEdge]
    MAJOR_ARG: _ClassVar[FeatureEdge]
    MINOR_ARG: _ClassVar[FeatureEdge]
    ACTION_TOKEN: _ClassVar[FeatureEdge]
    ITEM_TOKEN: _ClassVar[FeatureEdge]
    ABILITY_TOKEN: _ClassVar[FeatureEdge]
    FROM_TYPE_TOKEN: _ClassVar[FeatureEdge]
    FROM_SOURCE_TOKEN: _ClassVar[FeatureEdge]
    DAMAGE_TOKEN: _ClassVar[FeatureEdge]
    EFFECT_TOKEN: _ClassVar[FeatureEdge]
    BOOST_ATK_VALUE: _ClassVar[FeatureEdge]
    BOOST_DEF_VALUE: _ClassVar[FeatureEdge]
    BOOST_SPA_VALUE: _ClassVar[FeatureEdge]
    BOOST_SPD_VALUE: _ClassVar[FeatureEdge]
    BOOST_SPE_VALUE: _ClassVar[FeatureEdge]
    BOOST_ACCURACY_VALUE: _ClassVar[FeatureEdge]
    BOOST_EVASION_VALUE: _ClassVar[FeatureEdge]
    STATUS_TOKEN: _ClassVar[FeatureEdge]
    EDGE_AFFECTING_SIDE: _ClassVar[FeatureEdge]
    PLAYER_ID: _ClassVar[FeatureEdge]
    REQUEST_COUNT: _ClassVar[FeatureEdge]
    EDGE_VALID: _ClassVar[FeatureEdge]
    EDGE_INDEX: _ClassVar[FeatureEdge]
    TURN_VALUE: _ClassVar[FeatureEdge]
ENTITY_SPECIES: FeatureEntity
ENTITY_ITEM: FeatureEntity
ENTITY_ITEM_EFFECT: FeatureEntity
ENTITY_ABILITY: FeatureEntity
ENTITY_GENDER: FeatureEntity
ENTITY_ACTIVE: FeatureEntity
ENTITY_FAINTED: FeatureEntity
ENTITY_HP: FeatureEntity
ENTITY_MAXHP: FeatureEntity
ENTITY_STATUS: FeatureEntity
ENTITY_TOXIC_TURNS: FeatureEntity
ENTITY_SLEEP_TURNS: FeatureEntity
ENTITY_BEING_CALLED_BACK: FeatureEntity
ENTITY_TRAPPED: FeatureEntity
ENTITY_NEWLY_SWITCHED: FeatureEntity
ENTITY_LEVEL: FeatureEntity
ENTITY_MOVEID0: FeatureEntity
ENTITY_MOVEID1: FeatureEntity
ENTITY_MOVEID2: FeatureEntity
ENTITY_MOVEID3: FeatureEntity
ENTITY_MOVEPP0: FeatureEntity
ENTITY_MOVEPP1: FeatureEntity
ENTITY_MOVEPP2: FeatureEntity
ENTITY_MOVEPP3: FeatureEntity
ENTITY_HAS_STATUS: FeatureEntity
ENTITY_BOOST_ATK_VALUE: FeatureEntity
ENTITY_BOOST_DEF_VALUE: FeatureEntity
ENTITY_BOOST_SPA_VALUE: FeatureEntity
ENTITY_BOOST_SPD_VALUE: FeatureEntity
ENTITY_BOOST_SPE_VALUE: FeatureEntity
ENTITY_BOOST_ACCURACY_VALUE: FeatureEntity
ENTITY_BOOST_EVASION_VALUE: FeatureEntity
ENTITY_VOLATILES0: FeatureEntity
ENTITY_VOLATILES1: FeatureEntity
ENTITY_VOLATILES2: FeatureEntity
ENTITY_VOLATILES3: FeatureEntity
ENTITY_VOLATILES4: FeatureEntity
ENTITY_VOLATILES5: FeatureEntity
ENTITY_VOLATILES6: FeatureEntity
ENTITY_VOLATILES7: FeatureEntity
ENTITY_VOLATILES8: FeatureEntity
ENTITY_SIDE: FeatureEntity
ENTITY_HP_TOKEN: FeatureEntity
MOVESET_ACTION_TYPE_MOVE: MovesetActionType
MOVESET_ACTION_TYPE_SWITCH: MovesetActionType
MOVESET_ACTION_ID: FeatureMoveset
MOVESET_PPUSED: FeatureMoveset
MOVESET_LEGAL: FeatureMoveset
MOVESET_SIDE: FeatureMoveset
MOVESET_ACTION_TYPE: FeatureMoveset
MOVESET_EST_DAMAGE: FeatureMoveset
MOVESET_MOVE_ID: FeatureMoveset
MOVESET_SPECIES_ID: FeatureMoveset
VALID: FeatureTurnContext
IS_MY_TURN: FeatureTurnContext
ACTION: FeatureTurnContext
MOVE: FeatureTurnContext
SWITCH_COUNTER: FeatureTurnContext
MOVE_COUNTER: FeatureTurnContext
TURN: FeatureTurnContext
WEATHER_ID: FeatureWeather
MIN_DURATION: FeatureWeather
MAX_DURATION: FeatureWeather
NUM_FAINTED: FeatureAdditionalInformation
HP_TOTAL: FeatureAdditionalInformation
NUM_TYPES_PAD: FeatureAdditionalInformation
NUM_TYPES_UNK: FeatureAdditionalInformation
NUM_TYPES_BUG: FeatureAdditionalInformation
NUM_TYPES_DARK: FeatureAdditionalInformation
NUM_TYPES_DRAGON: FeatureAdditionalInformation
NUM_TYPES_ELECTRIC: FeatureAdditionalInformation
NUM_TYPES_FAIRY: FeatureAdditionalInformation
NUM_TYPES_FIGHTING: FeatureAdditionalInformation
NUM_TYPES_FIRE: FeatureAdditionalInformation
NUM_TYPES_FLYING: FeatureAdditionalInformation
NUM_TYPES_GHOST: FeatureAdditionalInformation
NUM_TYPES_GRASS: FeatureAdditionalInformation
NUM_TYPES_GROUND: FeatureAdditionalInformation
NUM_TYPES_ICE: FeatureAdditionalInformation
NUM_TYPES_NORMAL: FeatureAdditionalInformation
NUM_TYPES_POISON: FeatureAdditionalInformation
NUM_TYPES_PSYCHIC: FeatureAdditionalInformation
NUM_TYPES_ROCK: FeatureAdditionalInformation
NUM_TYPES_STEEL: FeatureAdditionalInformation
NUM_TYPES_STELLAR: FeatureAdditionalInformation
NUM_TYPES_WATER: FeatureAdditionalInformation
TOTAL_POKEMON: FeatureAdditionalInformation
WISHING: FeatureAdditionalInformation
MEMBER0_HP: FeatureAdditionalInformation
MEMBER1_HP: FeatureAdditionalInformation
MEMBER2_HP: FeatureAdditionalInformation
MEMBER3_HP: FeatureAdditionalInformation
MEMBER4_HP: FeatureAdditionalInformation
MEMBER5_HP: FeatureAdditionalInformation
EDGE_TYPE_NONE: EdgeTypes
MOVE_EDGE: EdgeTypes
SWITCH_EDGE: EdgeTypes
EFFECT_EDGE: EdgeTypes
CANT_EDGE: EdgeTypes
EDGE_TYPE_START: EdgeTypes
EDGE_FROM_NONE: EdgeFromTypes
EDGE_FROM_ITEM: EdgeFromTypes
EDGE_FROM_EFFECT: EdgeFromTypes
EDGE_FROM_MOVE: EdgeFromTypes
EDGE_FROM_ABILITY: EdgeFromTypes
EDGE_FROM_SIDECONDITION: EdgeFromTypes
EDGE_FROM_STATUS: EdgeFromTypes
EDGE_FROM_WEATHER: EdgeFromTypes
EDGE_FROM_TERRAIN: EdgeFromTypes
EDGE_FROM_PSEUDOWEATHER: EdgeFromTypes
TURN_ORDER_VALUE: FeatureEdge
EDGE_TYPE_TOKEN: FeatureEdge
MAJOR_ARG: FeatureEdge
MINOR_ARG: FeatureEdge
ACTION_TOKEN: FeatureEdge
ITEM_TOKEN: FeatureEdge
ABILITY_TOKEN: FeatureEdge
FROM_TYPE_TOKEN: FeatureEdge
FROM_SOURCE_TOKEN: FeatureEdge
DAMAGE_TOKEN: FeatureEdge
EFFECT_TOKEN: FeatureEdge
BOOST_ATK_VALUE: FeatureEdge
BOOST_DEF_VALUE: FeatureEdge
BOOST_SPA_VALUE: FeatureEdge
BOOST_SPD_VALUE: FeatureEdge
BOOST_SPE_VALUE: FeatureEdge
BOOST_ACCURACY_VALUE: FeatureEdge
BOOST_EVASION_VALUE: FeatureEdge
STATUS_TOKEN: FeatureEdge
EDGE_AFFECTING_SIDE: FeatureEdge
PLAYER_ID: FeatureEdge
REQUEST_COUNT: FeatureEdge
EDGE_VALID: FeatureEdge
EDGE_INDEX: FeatureEdge
TURN_VALUE: FeatureEdge
