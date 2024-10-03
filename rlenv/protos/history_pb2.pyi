from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class History(_message.Message):
    __slots__ = ("edges", "nodes", "length")
    EDGES_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    edges: bytes
    nodes: bytes
    length: int
    def __init__(self, edges: _Optional[bytes] = ..., nodes: _Optional[bytes] = ..., length: _Optional[int] = ...) -> None: ...
