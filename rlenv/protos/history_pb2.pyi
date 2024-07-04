import pokemon_pb2 as _pokemon_pb2
import enums_pb2 as _enums_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ActionTypeEnum(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    move: _ClassVar[ActionTypeEnum]
    switch: _ClassVar[ActionTypeEnum]
move: ActionTypeEnum
switch: ActionTypeEnum

class Boost(_message.Message):
    __slots__ = ("stat", "value")
    STAT_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    stat: _enums_pb2.BoostsEnum
    value: int
    def __init__(self, stat: _Optional[_Union[_enums_pb2.BoostsEnum, str]] = ..., value: _Optional[int] = ...) -> None: ...

class SideCondition(_message.Message):
    __slots__ = ("type", "value")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    type: _enums_pb2.SideconditionsEnum
    value: int
    def __init__(self, type: _Optional[_Union[_enums_pb2.SideconditionsEnum, str]] = ..., value: _Optional[int] = ...) -> None: ...

class HistorySide(_message.Message):
    __slots__ = ("active", "boosts", "sideConditions", "volatileStatus")
    ACTIVE_FIELD_NUMBER: _ClassVar[int]
    BOOSTS_FIELD_NUMBER: _ClassVar[int]
    SIDECONDITIONS_FIELD_NUMBER: _ClassVar[int]
    VOLATILESTATUS_FIELD_NUMBER: _ClassVar[int]
    active: _pokemon_pb2.Pokemon
    boosts: _containers.RepeatedCompositeFieldContainer[Boost]
    sideConditions: _containers.RepeatedCompositeFieldContainer[SideCondition]
    volatileStatus: _containers.RepeatedScalarFieldContainer[_enums_pb2.VolatilestatusEnum]
    def __init__(self, active: _Optional[_Union[_pokemon_pb2.Pokemon, _Mapping]] = ..., boosts: _Optional[_Iterable[_Union[Boost, _Mapping]]] = ..., sideConditions: _Optional[_Iterable[_Union[SideCondition, _Mapping]]] = ..., volatileStatus: _Optional[_Iterable[_Union[_enums_pb2.VolatilestatusEnum, str]]] = ...) -> None: ...

class HistoryStep(_message.Message):
    __slots__ = ("p1", "p2", "weather", "pseudoweather", "action", "move", "hyphenArgs")
    P1_FIELD_NUMBER: _ClassVar[int]
    P2_FIELD_NUMBER: _ClassVar[int]
    WEATHER_FIELD_NUMBER: _ClassVar[int]
    PSEUDOWEATHER_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    MOVE_FIELD_NUMBER: _ClassVar[int]
    HYPHENARGS_FIELD_NUMBER: _ClassVar[int]
    p1: HistorySide
    p2: HistorySide
    weather: _enums_pb2.WeathersEnum
    pseudoweather: _containers.RepeatedScalarFieldContainer[_enums_pb2.PseudoweatherEnum]
    action: ActionTypeEnum
    move: _enums_pb2.MovesEnum
    hyphenArgs: _containers.RepeatedScalarFieldContainer[_enums_pb2.HyphenargsEnum]
    def __init__(self, p1: _Optional[_Union[HistorySide, _Mapping]] = ..., p2: _Optional[_Union[HistorySide, _Mapping]] = ..., weather: _Optional[_Union[_enums_pb2.WeathersEnum, str]] = ..., pseudoweather: _Optional[_Iterable[_Union[_enums_pb2.PseudoweatherEnum, str]]] = ..., action: _Optional[_Union[ActionTypeEnum, str]] = ..., move: _Optional[_Union[_enums_pb2.MovesEnum, str]] = ..., hyphenArgs: _Optional[_Iterable[_Union[_enums_pb2.HyphenargsEnum, str]]] = ...) -> None: ...
