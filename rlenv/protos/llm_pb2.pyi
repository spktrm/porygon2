from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class LLMState(_message.Message):
    __slots__ = ("legalMask", "request", "log", "myTeam", "oppTeam")
    LEGALMASK_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    LOG_FIELD_NUMBER: _ClassVar[int]
    MYTEAM_FIELD_NUMBER: _ClassVar[int]
    OPPTEAM_FIELD_NUMBER: _ClassVar[int]
    legalMask: _containers.RepeatedScalarFieldContainer[bool]
    request: str
    log: str
    myTeam: str
    oppTeam: str
    def __init__(self, legalMask: _Optional[_Iterable[bool]] = ..., request: _Optional[str] = ..., log: _Optional[str] = ..., myTeam: _Optional[str] = ..., oppTeam: _Optional[str] = ...) -> None: ...
