from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Action(_message.Message):
    __slots__ = ("key", "index", "text", "gameId")
    KEY_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    GAMEID_FIELD_NUMBER: _ClassVar[int]
    key: str
    index: int
    text: str
    gameId: int
    def __init__(self, key: _Optional[str] = ..., index: _Optional[int] = ..., text: _Optional[str] = ..., gameId: _Optional[int] = ...) -> None: ...
