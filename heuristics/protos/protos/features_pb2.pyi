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
    ENTITY_HP_RATIO: _ClassVar[FeatureEntity]
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
    ENTITY_TYPECHANGE0: _ClassVar[FeatureEntity]
    ENTITY_TYPECHANGE1: _ClassVar[FeatureEntity]

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

class FeatureAbsoluteEdge(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EDGE_TURN_ORDER_VALUE: _ClassVar[FeatureAbsoluteEdge]
    EDGE_TYPE_TOKEN: _ClassVar[FeatureAbsoluteEdge]
    EDGE_WEATHER_ID: _ClassVar[FeatureAbsoluteEdge]
    EDGE_WEATHER_MIN_DURATION: _ClassVar[FeatureAbsoluteEdge]
    EDGE_WEATHER_MAX_DURATION: _ClassVar[FeatureAbsoluteEdge]
    EDGE_TERRAIN_ID: _ClassVar[FeatureAbsoluteEdge]
    EDGE_TERRAIN_MIN_DURATION: _ClassVar[FeatureAbsoluteEdge]
    EDGE_TERRAIN_MAX_DURATION: _ClassVar[FeatureAbsoluteEdge]
    EDGE_PSEUDOWEATHER_ID: _ClassVar[FeatureAbsoluteEdge]
    EDGE_PSEUDOWEATHER_MIN_DURATION: _ClassVar[FeatureAbsoluteEdge]
    EDGE_PSEUDOWEATHER_MAX_DURATION: _ClassVar[FeatureAbsoluteEdge]
    EDGE_REQUEST_COUNT: _ClassVar[FeatureAbsoluteEdge]
    EDGE_VALID: _ClassVar[FeatureAbsoluteEdge]
    EDGE_INDEX: _ClassVar[FeatureAbsoluteEdge]
    EDGE_TURN_VALUE: _ClassVar[FeatureAbsoluteEdge]

class FeatureRelativeEdge(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EDGE_MAJOR_ARG: _ClassVar[FeatureRelativeEdge]
    EDGE_MINOR_ARG0: _ClassVar[FeatureRelativeEdge]
    EDGE_MINOR_ARG1: _ClassVar[FeatureRelativeEdge]
    EDGE_MINOR_ARG2: _ClassVar[FeatureRelativeEdge]
    EDGE_MINOR_ARG3: _ClassVar[FeatureRelativeEdge]
    EDGE_ACTION_TOKEN: _ClassVar[FeatureRelativeEdge]
    EDGE_ITEM_TOKEN: _ClassVar[FeatureRelativeEdge]
    EDGE_ABILITY_TOKEN: _ClassVar[FeatureRelativeEdge]
    EDGE_FROM_TYPE_TOKEN: _ClassVar[FeatureRelativeEdge]
    EDGE_FROM_SOURCE_TOKEN: _ClassVar[FeatureRelativeEdge]
    EDGE_DAMAGE_RATIO: _ClassVar[FeatureRelativeEdge]
    EDGE_HEAL_RATIO: _ClassVar[FeatureRelativeEdge]
    EDGE_EFFECT_TOKEN: _ClassVar[FeatureRelativeEdge]
    EDGE_BOOST_ATK_VALUE: _ClassVar[FeatureRelativeEdge]
    EDGE_BOOST_DEF_VALUE: _ClassVar[FeatureRelativeEdge]
    EDGE_BOOST_SPA_VALUE: _ClassVar[FeatureRelativeEdge]
    EDGE_BOOST_SPD_VALUE: _ClassVar[FeatureRelativeEdge]
    EDGE_BOOST_SPE_VALUE: _ClassVar[FeatureRelativeEdge]
    EDGE_BOOST_ACCURACY_VALUE: _ClassVar[FeatureRelativeEdge]
    EDGE_BOOST_EVASION_VALUE: _ClassVar[FeatureRelativeEdge]
    EDGE_STATUS_TOKEN: _ClassVar[FeatureRelativeEdge]
    EDGE_SIDECONDITIONS0: _ClassVar[FeatureRelativeEdge]
    EDGE_SIDECONDITIONS1: _ClassVar[FeatureRelativeEdge]
    EDGE_TOXIC_SPIKES: _ClassVar[FeatureRelativeEdge]
    EDGE_SPIKES: _ClassVar[FeatureRelativeEdge]
ENTITY_SPECIES: FeatureEntity
ENTITY_ITEM: FeatureEntity
ENTITY_ITEM_EFFECT: FeatureEntity
ENTITY_ABILITY: FeatureEntity
ENTITY_GENDER: FeatureEntity
ENTITY_ACTIVE: FeatureEntity
ENTITY_FAINTED: FeatureEntity
ENTITY_HP: FeatureEntity
ENTITY_MAXHP: FeatureEntity
ENTITY_HP_RATIO: FeatureEntity
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
ENTITY_TYPECHANGE0: FeatureEntity
ENTITY_TYPECHANGE1: FeatureEntity
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
EDGE_TURN_ORDER_VALUE: FeatureAbsoluteEdge
EDGE_TYPE_TOKEN: FeatureAbsoluteEdge
EDGE_WEATHER_ID: FeatureAbsoluteEdge
EDGE_WEATHER_MIN_DURATION: FeatureAbsoluteEdge
EDGE_WEATHER_MAX_DURATION: FeatureAbsoluteEdge
EDGE_TERRAIN_ID: FeatureAbsoluteEdge
EDGE_TERRAIN_MIN_DURATION: FeatureAbsoluteEdge
EDGE_TERRAIN_MAX_DURATION: FeatureAbsoluteEdge
EDGE_PSEUDOWEATHER_ID: FeatureAbsoluteEdge
EDGE_PSEUDOWEATHER_MIN_DURATION: FeatureAbsoluteEdge
EDGE_PSEUDOWEATHER_MAX_DURATION: FeatureAbsoluteEdge
EDGE_REQUEST_COUNT: FeatureAbsoluteEdge
EDGE_VALID: FeatureAbsoluteEdge
EDGE_INDEX: FeatureAbsoluteEdge
EDGE_TURN_VALUE: FeatureAbsoluteEdge
EDGE_MAJOR_ARG: FeatureRelativeEdge
EDGE_MINOR_ARG0: FeatureRelativeEdge
EDGE_MINOR_ARG1: FeatureRelativeEdge
EDGE_MINOR_ARG2: FeatureRelativeEdge
EDGE_MINOR_ARG3: FeatureRelativeEdge
EDGE_ACTION_TOKEN: FeatureRelativeEdge
EDGE_ITEM_TOKEN: FeatureRelativeEdge
EDGE_ABILITY_TOKEN: FeatureRelativeEdge
EDGE_FROM_TYPE_TOKEN: FeatureRelativeEdge
EDGE_FROM_SOURCE_TOKEN: FeatureRelativeEdge
EDGE_DAMAGE_RATIO: FeatureRelativeEdge
EDGE_HEAL_RATIO: FeatureRelativeEdge
EDGE_EFFECT_TOKEN: FeatureRelativeEdge
EDGE_BOOST_ATK_VALUE: FeatureRelativeEdge
EDGE_BOOST_DEF_VALUE: FeatureRelativeEdge
EDGE_BOOST_SPA_VALUE: FeatureRelativeEdge
EDGE_BOOST_SPD_VALUE: FeatureRelativeEdge
EDGE_BOOST_SPE_VALUE: FeatureRelativeEdge
EDGE_BOOST_ACCURACY_VALUE: FeatureRelativeEdge
EDGE_BOOST_EVASION_VALUE: FeatureRelativeEdge
EDGE_STATUS_TOKEN: FeatureRelativeEdge
EDGE_SIDECONDITIONS0: FeatureRelativeEdge
EDGE_SIDECONDITIONS1: FeatureRelativeEdge
EDGE_TOXIC_SPIKES: FeatureRelativeEdge
EDGE_SPIKES: FeatureRelativeEdge
