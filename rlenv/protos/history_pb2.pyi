from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class History(_message.Message):
    __slots__ = ("edges", "entities", "sideConditions", "field", "length")
    EDGES_FIELD_NUMBER: _ClassVar[int]
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    SIDECONDITIONS_FIELD_NUMBER: _ClassVar[int]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    edges: bytes
    entities: bytes
    sideConditions: bytes
    field: bytes
    length: int
    def __init__(self, edges: _Optional[bytes] = ..., entities: _Optional[bytes] = ..., sideConditions: _Optional[bytes] = ..., field: _Optional[bytes] = ..., length: _Optional[int] = ...) -> None: ...
