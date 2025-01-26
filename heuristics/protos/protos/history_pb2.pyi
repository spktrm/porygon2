from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class History(_message.Message):
    __slots__ = ("absolute_edge", "relative_edges", "entities", "length")
    ABSOLUTE_EDGE_FIELD_NUMBER: _ClassVar[int]
    RELATIVE_EDGES_FIELD_NUMBER: _ClassVar[int]
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    absolute_edge: bytes
    relative_edges: bytes
    entities: bytes
    length: int
    def __init__(self, absolute_edge: _Optional[bytes] = ..., relative_edges: _Optional[bytes] = ..., entities: _Optional[bytes] = ..., length: _Optional[int] = ...) -> None: ...
