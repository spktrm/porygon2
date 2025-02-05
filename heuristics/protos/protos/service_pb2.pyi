from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ClientMessage(_message.Message):
    __slots__ = ("player_id", "game_id", "connect", "step", "reset")
    PLAYER_ID_FIELD_NUMBER: _ClassVar[int]
    GAME_ID_FIELD_NUMBER: _ClassVar[int]
    CONNECT_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    RESET_FIELD_NUMBER: _ClassVar[int]
    player_id: int
    game_id: int
    connect: ConnectMessage
    step: StepMessage
    reset: ResetMessage
    def __init__(self, player_id: _Optional[int] = ..., game_id: _Optional[int] = ..., connect: _Optional[_Union[ConnectMessage, _Mapping]] = ..., step: _Optional[_Union[StepMessage, _Mapping]] = ..., reset: _Optional[_Union[ResetMessage, _Mapping]] = ...) -> None: ...

class Action(_message.Message):
    __slots__ = ("rqid", "value", "text")
    RQID_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    rqid: int
    value: int
    text: str
    def __init__(self, rqid: _Optional[int] = ..., value: _Optional[int] = ..., text: _Optional[str] = ...) -> None: ...

class StepMessage(_message.Message):
    __slots__ = ("action",)
    ACTION_FIELD_NUMBER: _ClassVar[int]
    action: Action
    def __init__(self, action: _Optional[_Union[Action, _Mapping]] = ...) -> None: ...

class ResetMessage(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ConnectMessage(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ServerMessage(_message.Message):
    __slots__ = ("game_state", "error")
    GAME_STATE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    game_state: GameState
    error: ErrorMessage
    def __init__(self, game_state: _Optional[_Union[GameState, _Mapping]] = ..., error: _Optional[_Union[ErrorMessage, _Mapping]] = ...) -> None: ...

class GameState(_message.Message):
    __slots__ = ("player_id", "rqid", "state")
    PLAYER_ID_FIELD_NUMBER: _ClassVar[int]
    RQID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    player_id: int
    rqid: int
    state: bytes
    def __init__(self, player_id: _Optional[int] = ..., rqid: _Optional[int] = ..., state: _Optional[bytes] = ...) -> None: ...

class ErrorMessage(_message.Message):
    __slots__ = ("error_message",)
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    error_message: str
    def __init__(self, error_message: _Optional[str] = ...) -> None: ...
