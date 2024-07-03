from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Action(_message.Message):
    __slots__ = ("gameId", "playerIndex", "index", "text")
    GAMEID_FIELD_NUMBER: _ClassVar[int]
    PLAYERINDEX_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    gameId: int
    playerIndex: bool
    index: int
    text: str
    def __init__(self, gameId: _Optional[int] = ..., playerIndex: bool = ..., index: _Optional[int] = ..., text: _Optional[str] = ...) -> None: ...
