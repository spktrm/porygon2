import enums_pb2 as _enums_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Pokemon(_message.Message):
    __slots__ = ("species", "item", "ability", "move1Id", "move2Id", "move3Id", "move4Id", "pp1Used", "pp2Used", "pp3Used", "pp4Used", "hpRatio", "active", "fainted", "level", "gender", "itemEffect")
    SPECIES_FIELD_NUMBER: _ClassVar[int]
    ITEM_FIELD_NUMBER: _ClassVar[int]
    ABILITY_FIELD_NUMBER: _ClassVar[int]
    MOVE1ID_FIELD_NUMBER: _ClassVar[int]
    MOVE2ID_FIELD_NUMBER: _ClassVar[int]
    MOVE3ID_FIELD_NUMBER: _ClassVar[int]
    MOVE4ID_FIELD_NUMBER: _ClassVar[int]
    PP1USED_FIELD_NUMBER: _ClassVar[int]
    PP2USED_FIELD_NUMBER: _ClassVar[int]
    PP3USED_FIELD_NUMBER: _ClassVar[int]
    PP4USED_FIELD_NUMBER: _ClassVar[int]
    HPRATIO_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_FIELD_NUMBER: _ClassVar[int]
    FAINTED_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    GENDER_FIELD_NUMBER: _ClassVar[int]
    ITEMEFFECT_FIELD_NUMBER: _ClassVar[int]
    species: _enums_pb2.SpeciesEnum
    item: _enums_pb2.ItemsEnum
    ability: _enums_pb2.AbilitiesEnum
    move1Id: _enums_pb2.MovesEnum
    move2Id: _enums_pb2.MovesEnum
    move3Id: _enums_pb2.MovesEnum
    move4Id: _enums_pb2.MovesEnum
    pp1Used: int
    pp2Used: int
    pp3Used: int
    pp4Used: int
    hpRatio: float
    active: bool
    fainted: bool
    level: int
    gender: _enums_pb2.GendersEnum
    itemEffect: _enums_pb2.ItemeffectEnum
    def __init__(self, species: _Optional[_Union[_enums_pb2.SpeciesEnum, str]] = ..., item: _Optional[_Union[_enums_pb2.ItemsEnum, str]] = ..., ability: _Optional[_Union[_enums_pb2.AbilitiesEnum, str]] = ..., move1Id: _Optional[_Union[_enums_pb2.MovesEnum, str]] = ..., move2Id: _Optional[_Union[_enums_pb2.MovesEnum, str]] = ..., move3Id: _Optional[_Union[_enums_pb2.MovesEnum, str]] = ..., move4Id: _Optional[_Union[_enums_pb2.MovesEnum, str]] = ..., pp1Used: _Optional[int] = ..., pp2Used: _Optional[int] = ..., pp3Used: _Optional[int] = ..., pp4Used: _Optional[int] = ..., hpRatio: _Optional[float] = ..., active: bool = ..., fainted: bool = ..., level: _Optional[int] = ..., gender: _Optional[_Union[_enums_pb2.GendersEnum, str]] = ..., itemEffect: _Optional[_Union[_enums_pb2.ItemeffectEnum, str]] = ...) -> None: ...
