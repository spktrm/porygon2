import enums_pb2 as _enums_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Move(_message.Message):
    __slots__ = ("moveId", "ppUsed")
    MOVEID_FIELD_NUMBER: _ClassVar[int]
    PPUSED_FIELD_NUMBER: _ClassVar[int]
    moveId: _enums_pb2.MovesEnum
    ppUsed: int
    def __init__(self, moveId: _Optional[_Union[_enums_pb2.MovesEnum, str]] = ..., ppUsed: _Optional[int] = ...) -> None: ...

class Pokemon(_message.Message):
    __slots__ = ("species", "item", "ability", "moveset", "hpRatio", "active", "fainted", "level", "gender")
    SPECIES_FIELD_NUMBER: _ClassVar[int]
    ITEM_FIELD_NUMBER: _ClassVar[int]
    ABILITY_FIELD_NUMBER: _ClassVar[int]
    MOVESET_FIELD_NUMBER: _ClassVar[int]
    HPRATIO_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_FIELD_NUMBER: _ClassVar[int]
    FAINTED_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    GENDER_FIELD_NUMBER: _ClassVar[int]
    species: _enums_pb2.SpeciesEnum
    item: _enums_pb2.ItemsEnum
    ability: _enums_pb2.AbilitiesEnum
    moveset: _containers.RepeatedCompositeFieldContainer[Move]
    hpRatio: float
    active: bool
    fainted: bool
    level: int
    gender: _enums_pb2.GendersEnum
    def __init__(self, species: _Optional[_Union[_enums_pb2.SpeciesEnum, str]] = ..., item: _Optional[_Union[_enums_pb2.ItemsEnum, str]] = ..., ability: _Optional[_Union[_enums_pb2.AbilitiesEnum, str]] = ..., moveset: _Optional[_Iterable[_Union[Move, _Mapping]]] = ..., hpRatio: _Optional[float] = ..., active: bool = ..., fainted: bool = ..., level: _Optional[int] = ..., gender: _Optional[_Union[_enums_pb2.GendersEnum, str]] = ...) -> None: ...
