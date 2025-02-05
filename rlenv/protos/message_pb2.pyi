import action_pb2 as _action_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class WorkerMessageTypeEnum(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WORKER_MESSAGE_TYPE_ENUM___UNSPECIFIED: _ClassVar[WorkerMessageTypeEnum]
    WORKER_MESSAGE_TYPE_ENUM__START: _ClassVar[WorkerMessageTypeEnum]
    WORKER_MESSAGE_TYPE_ENUM__ACTION: _ClassVar[WorkerMessageTypeEnum]
WORKER_MESSAGE_TYPE_ENUM___UNSPECIFIED: WorkerMessageTypeEnum
WORKER_MESSAGE_TYPE_ENUM__START: WorkerMessageTypeEnum
WORKER_MESSAGE_TYPE_ENUM__ACTION: WorkerMessageTypeEnum

class WorkerMessage(_message.Message):
    __slots__ = ("worker_index", "message_type", "action", "game_id")
    WORKER_INDEX_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    GAME_ID_FIELD_NUMBER: _ClassVar[int]
    worker_index: int
    message_type: WorkerMessageTypeEnum
    action: _action_pb2.Action
    game_id: int
    def __init__(self, worker_index: _Optional[int] = ..., message_type: _Optional[_Union[WorkerMessageTypeEnum, str]] = ..., action: _Optional[_Union[_action_pb2.Action, _Mapping]] = ..., game_id: _Optional[int] = ...) -> None: ...
