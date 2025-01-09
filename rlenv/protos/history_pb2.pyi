from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class History(_message.Message):
    __slots__ = ("absoluteEdge", "relativeEdges", "entities", "length")
    ABSOLUTEEDGE_FIELD_NUMBER: _ClassVar[int]
    RELATIVEEDGES_FIELD_NUMBER: _ClassVar[int]
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    absoluteEdge: bytes
    relativeEdges: bytes
    entities: bytes
    length: int
    def __init__(self, absoluteEdge: _Optional[bytes] = ..., relativeEdges: _Optional[bytes] = ..., entities: _Optional[bytes] = ..., length: _Optional[int] = ...) -> None: ...
