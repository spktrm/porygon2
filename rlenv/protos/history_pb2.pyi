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
    __slots__ = ("index", "value")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    index: _enums_pb2.BoostsEnum
    value: int
    def __init__(self, index: _Optional[_Union[_enums_pb2.BoostsEnum, str]] = ..., value: _Optional[int] = ...) -> None: ...

class Sidecondition(_message.Message):
    __slots__ = ("index", "value")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    index: _enums_pb2.SideconditionsEnum
    value: int
    def __init__(self, index: _Optional[_Union[_enums_pb2.SideconditionsEnum, str]] = ..., value: _Optional[int] = ...) -> None: ...

class Volatilestatus(_message.Message):
    __slots__ = ("index", "value")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    index: _enums_pb2.VolatilestatusEnum
    value: int
    def __init__(self, index: _Optional[_Union[_enums_pb2.VolatilestatusEnum, str]] = ..., value: _Optional[int] = ...) -> None: ...

class HyphenArg(_message.Message):
    __slots__ = ("index", "value")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    index: _enums_pb2.HyphenargsEnum
    value: bool
    def __init__(self, index: _Optional[_Union[_enums_pb2.HyphenargsEnum, str]] = ..., value: bool = ...) -> None: ...

class HistorySide(_message.Message):
    __slots__ = ("active", "boosts", "sideConditions", "volatileStatus", "hyphenArgs")
    ACTIVE_FIELD_NUMBER: _ClassVar[int]
    BOOSTS_FIELD_NUMBER: _ClassVar[int]
    SIDECONDITIONS_FIELD_NUMBER: _ClassVar[int]
    VOLATILESTATUS_FIELD_NUMBER: _ClassVar[int]
    HYPHENARGS_FIELD_NUMBER: _ClassVar[int]
    active: _pokemon_pb2.Pokemon
    boosts: _containers.RepeatedCompositeFieldContainer[Boost]
    sideConditions: _containers.RepeatedCompositeFieldContainer[Sidecondition]
    volatileStatus: _containers.RepeatedCompositeFieldContainer[Volatilestatus]
    hyphenArgs: _containers.RepeatedCompositeFieldContainer[HyphenArg]
    def __init__(self, active: _Optional[_Union[_pokemon_pb2.Pokemon, _Mapping]] = ..., boosts: _Optional[_Iterable[_Union[Boost, _Mapping]]] = ..., sideConditions: _Optional[_Iterable[_Union[Sidecondition, _Mapping]]] = ..., volatileStatus: _Optional[_Iterable[_Union[Volatilestatus, _Mapping]]] = ..., hyphenArgs: _Optional[_Iterable[_Union[HyphenArg, _Mapping]]] = ...) -> None: ...

class Weather(_message.Message):
    __slots__ = ("index", "minDuration", "maxDuration")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    MINDURATION_FIELD_NUMBER: _ClassVar[int]
    MAXDURATION_FIELD_NUMBER: _ClassVar[int]
    index: _enums_pb2.WeathersEnum
    minDuration: int
    maxDuration: int
    def __init__(self, index: _Optional[_Union[_enums_pb2.WeathersEnum, str]] = ..., minDuration: _Optional[int] = ..., maxDuration: _Optional[int] = ...) -> None: ...

class PseudoWeather(_message.Message):
    __slots__ = ("index", "minDuration", "maxDuration")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    MINDURATION_FIELD_NUMBER: _ClassVar[int]
    MAXDURATION_FIELD_NUMBER: _ClassVar[int]
    index: _enums_pb2.PseudoweatherEnum
    minDuration: int
    maxDuration: int
    def __init__(self, index: _Optional[_Union[_enums_pb2.PseudoweatherEnum, str]] = ..., minDuration: _Optional[int] = ..., maxDuration: _Optional[int] = ...) -> None: ...

class HistoryStep(_message.Message):
    __slots__ = ("p1", "p2", "weather", "pseudoweather", "action", "move", "isMyTurn", "moveCounter", "switchCounter", "turn")
    P1_FIELD_NUMBER: _ClassVar[int]
    P2_FIELD_NUMBER: _ClassVar[int]
    WEATHER_FIELD_NUMBER: _ClassVar[int]
    PSEUDOWEATHER_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    MOVE_FIELD_NUMBER: _ClassVar[int]
    ISMYTURN_FIELD_NUMBER: _ClassVar[int]
    MOVECOUNTER_FIELD_NUMBER: _ClassVar[int]
    SWITCHCOUNTER_FIELD_NUMBER: _ClassVar[int]
    TURN_FIELD_NUMBER: _ClassVar[int]
    p1: HistorySide
    p2: HistorySide
    weather: Weather
    pseudoweather: _containers.RepeatedCompositeFieldContainer[PseudoWeather]
    action: ActionTypeEnum
    move: _enums_pb2.MovesEnum
    isMyTurn: bool
    moveCounter: int
    switchCounter: int
    turn: int
    def __init__(self, p1: _Optional[_Union[HistorySide, _Mapping]] = ..., p2: _Optional[_Union[HistorySide, _Mapping]] = ..., weather: _Optional[_Union[Weather, _Mapping]] = ..., pseudoweather: _Optional[_Iterable[_Union[PseudoWeather, _Mapping]]] = ..., action: _Optional[_Union[ActionTypeEnum, str]] = ..., move: _Optional[_Union[_enums_pb2.MovesEnum, str]] = ..., isMyTurn: bool = ..., moveCounter: _Optional[int] = ..., switchCounter: _Optional[int] = ..., turn: _Optional[int] = ...) -> None: ...
