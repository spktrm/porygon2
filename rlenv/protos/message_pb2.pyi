import action_pb2 as _action_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class WorkerMessageType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    START: _ClassVar[WorkerMessageType]
    ACTION: _ClassVar[WorkerMessageType]
START: WorkerMessageType
ACTION: WorkerMessageType

class WorkerMessage(_message.Message):
    __slots__ = ("workerIndex", "messageType", "action", "gameId")
    WORKERINDEX_FIELD_NUMBER: _ClassVar[int]
    MESSAGETYPE_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    GAMEID_FIELD_NUMBER: _ClassVar[int]
    workerIndex: int
    messageType: WorkerMessageType
    action: _action_pb2.Action
    gameId: int
    def __init__(self, workerIndex: _Optional[int] = ..., messageType: _Optional[_Union[WorkerMessageType, str]] = ..., action: _Optional[_Union[_action_pb2.Action, _Mapping]] = ..., gameId: _Optional[int] = ...) -> None: ...
