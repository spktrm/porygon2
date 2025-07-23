from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ClientRequest(_message.Message):
    __slots__ = ("step", "reset")
    STEP_FIELD_NUMBER: _ClassVar[int]
    RESET_FIELD_NUMBER: _ClassVar[int]
    step: StepRequest
    reset: ResetRequest
    def __init__(self, step: _Optional[_Union[StepRequest, _Mapping]] = ..., reset: _Optional[_Union[ResetRequest, _Mapping]] = ...) -> None: ...

class StepRequest(_message.Message):
    __slots__ = ("username", "action", "rqid")
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    RQID_FIELD_NUMBER: _ClassVar[int]
    username: str
    action: int
    rqid: int
    def __init__(self, username: _Optional[str] = ..., action: _Optional[int] = ..., rqid: _Optional[int] = ...) -> None: ...

class ResetRequest(_message.Message):
    __slots__ = ("username",)
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    username: str
    def __init__(self, username: _Optional[str] = ...) -> None: ...

class EnvironmentState(_message.Message):
    __slots__ = ("info", "legal_actions", "history_entity_nodes", "history_entity_edges", "history_field", "history_length", "moveset", "public_team", "private_team", "current_context", "rqid")
    INFO_FIELD_NUMBER: _ClassVar[int]
    LEGAL_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    HISTORY_ENTITY_NODES_FIELD_NUMBER: _ClassVar[int]
    HISTORY_ENTITY_EDGES_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_FIELD_NUMBER: _ClassVar[int]
    HISTORY_LENGTH_FIELD_NUMBER: _ClassVar[int]
    MOVESET_FIELD_NUMBER: _ClassVar[int]
    PUBLIC_TEAM_FIELD_NUMBER: _ClassVar[int]
    PRIVATE_TEAM_FIELD_NUMBER: _ClassVar[int]
    CURRENT_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    RQID_FIELD_NUMBER: _ClassVar[int]
    info: bytes
    legal_actions: bytes
    history_entity_nodes: bytes
    history_entity_edges: bytes
    history_field: bytes
    history_length: int
    moveset: bytes
    public_team: bytes
    private_team: bytes
    current_context: bytes
    rqid: int
    def __init__(self, info: _Optional[bytes] = ..., legal_actions: _Optional[bytes] = ..., history_entity_nodes: _Optional[bytes] = ..., history_entity_edges: _Optional[bytes] = ..., history_field: _Optional[bytes] = ..., history_length: _Optional[int] = ..., moveset: _Optional[bytes] = ..., public_team: _Optional[bytes] = ..., private_team: _Optional[bytes] = ..., current_context: _Optional[bytes] = ..., rqid: _Optional[int] = ...) -> None: ...

class EnvironmentResponse(_message.Message):
    __slots__ = ("username", "state")
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    username: str
    state: EnvironmentState
    def __init__(self, username: _Optional[str] = ..., state: _Optional[_Union[EnvironmentState, _Mapping]] = ...) -> None: ...

class ErrorResponse(_message.Message):
    __slots__ = ("trace",)
    TRACE_FIELD_NUMBER: _ClassVar[int]
    trace: str
    def __init__(self, trace: _Optional[str] = ...) -> None: ...

class WorkerRequest(_message.Message):
    __slots__ = ("task_id", "step_request", "reset_request")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    STEP_REQUEST_FIELD_NUMBER: _ClassVar[int]
    RESET_REQUEST_FIELD_NUMBER: _ClassVar[int]
    task_id: int
    step_request: StepRequest
    reset_request: ResetRequest
    def __init__(self, task_id: _Optional[int] = ..., step_request: _Optional[_Union[StepRequest, _Mapping]] = ..., reset_request: _Optional[_Union[ResetRequest, _Mapping]] = ...) -> None: ...

class WorkerResponse(_message.Message):
    __slots__ = ("task_id", "environment_response", "error_response")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    ERROR_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    task_id: int
    environment_response: EnvironmentResponse
    error_response: ErrorResponse
    def __init__(self, task_id: _Optional[int] = ..., environment_response: _Optional[_Union[EnvironmentResponse, _Mapping]] = ..., error_response: _Optional[_Union[ErrorResponse, _Mapping]] = ...) -> None: ...
