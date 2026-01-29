from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ActionEnum(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTION_ENUM___UNSPECIFIED: _ClassVar[ActionEnum]
    ACTION_ENUM__DEFAULT: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_1_MOVE_1: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_1_MOVE_2: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_1_MOVE_3: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_1_MOVE_4: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_1_MOVE_1_WILDCARD: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_1_MOVE_2_WILDCARD: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_1_MOVE_3_WILDCARD: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_1_MOVE_4_WILDCARD: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_2_MOVE_1: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_2_MOVE_2: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_2_MOVE_3: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_2_MOVE_4: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_2_MOVE_1_WILDCARD: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_2_MOVE_2_WILDCARD: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_2_MOVE_3_WILDCARD: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_2_MOVE_4_WILDCARD: _ClassVar[ActionEnum]
    ACTION_ENUM__RESERVE_1: _ClassVar[ActionEnum]
    ACTION_ENUM__RESERVE_2: _ClassVar[ActionEnum]
    ACTION_ENUM__RESERVE_3: _ClassVar[ActionEnum]
    ACTION_ENUM__RESERVE_4: _ClassVar[ActionEnum]
    ACTION_ENUM__RESERVE_5: _ClassVar[ActionEnum]
    ACTION_ENUM__RESERVE_6: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_1: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_2: _ClassVar[ActionEnum]
    ACTION_ENUM__ENEMY_1: _ClassVar[ActionEnum]
    ACTION_ENUM__ENEMY_2: _ClassVar[ActionEnum]
    ACTION_ENUM__TARGET_AUTO: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_1_PASS: _ClassVar[ActionEnum]
    ACTION_ENUM__ALLY_2_PASS: _ClassVar[ActionEnum]
ACTION_ENUM___UNSPECIFIED: ActionEnum
ACTION_ENUM__DEFAULT: ActionEnum
ACTION_ENUM__ALLY_1_MOVE_1: ActionEnum
ACTION_ENUM__ALLY_1_MOVE_2: ActionEnum
ACTION_ENUM__ALLY_1_MOVE_3: ActionEnum
ACTION_ENUM__ALLY_1_MOVE_4: ActionEnum
ACTION_ENUM__ALLY_1_MOVE_1_WILDCARD: ActionEnum
ACTION_ENUM__ALLY_1_MOVE_2_WILDCARD: ActionEnum
ACTION_ENUM__ALLY_1_MOVE_3_WILDCARD: ActionEnum
ACTION_ENUM__ALLY_1_MOVE_4_WILDCARD: ActionEnum
ACTION_ENUM__ALLY_2_MOVE_1: ActionEnum
ACTION_ENUM__ALLY_2_MOVE_2: ActionEnum
ACTION_ENUM__ALLY_2_MOVE_3: ActionEnum
ACTION_ENUM__ALLY_2_MOVE_4: ActionEnum
ACTION_ENUM__ALLY_2_MOVE_1_WILDCARD: ActionEnum
ACTION_ENUM__ALLY_2_MOVE_2_WILDCARD: ActionEnum
ACTION_ENUM__ALLY_2_MOVE_3_WILDCARD: ActionEnum
ACTION_ENUM__ALLY_2_MOVE_4_WILDCARD: ActionEnum
ACTION_ENUM__RESERVE_1: ActionEnum
ACTION_ENUM__RESERVE_2: ActionEnum
ACTION_ENUM__RESERVE_3: ActionEnum
ACTION_ENUM__RESERVE_4: ActionEnum
ACTION_ENUM__RESERVE_5: ActionEnum
ACTION_ENUM__RESERVE_6: ActionEnum
ACTION_ENUM__ALLY_1: ActionEnum
ACTION_ENUM__ALLY_2: ActionEnum
ACTION_ENUM__ENEMY_1: ActionEnum
ACTION_ENUM__ENEMY_2: ActionEnum
ACTION_ENUM__TARGET_AUTO: ActionEnum
ACTION_ENUM__ALLY_1_PASS: ActionEnum
ACTION_ENUM__ALLY_2_PASS: ActionEnum

class ClientRequest(_message.Message):
    __slots__ = ("step", "reset")
    STEP_FIELD_NUMBER: _ClassVar[int]
    RESET_FIELD_NUMBER: _ClassVar[int]
    step: StepRequest
    reset: ResetRequest
    def __init__(self, step: _Optional[_Union[StepRequest, _Mapping]] = ..., reset: _Optional[_Union[ResetRequest, _Mapping]] = ...) -> None: ...

class Action(_message.Message):
    __slots__ = ("src", "tgt")
    SRC_FIELD_NUMBER: _ClassVar[int]
    TGT_FIELD_NUMBER: _ClassVar[int]
    src: ActionEnum
    tgt: ActionEnum
    def __init__(self, src: _Optional[_Union[ActionEnum, str]] = ..., tgt: _Optional[_Union[ActionEnum, str]] = ...) -> None: ...

class StepRequest(_message.Message):
    __slots__ = ("username", "action", "rqid", "teampreview")
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    RQID_FIELD_NUMBER: _ClassVar[int]
    TEAMPREVIEW_FIELD_NUMBER: _ClassVar[int]
    username: str
    action: Action
    rqid: int
    teampreview: bool
    def __init__(self, username: _Optional[str] = ..., action: _Optional[_Union[Action, _Mapping]] = ..., rqid: _Optional[int] = ..., teampreview: bool = ...) -> None: ...

class ResetRequest(_message.Message):
    __slots__ = ("username", "species_indices", "packed_set_indices", "smogon_format", "current_ckpt", "opponent_ckpt")
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    SPECIES_INDICES_FIELD_NUMBER: _ClassVar[int]
    PACKED_SET_INDICES_FIELD_NUMBER: _ClassVar[int]
    SMOGON_FORMAT_FIELD_NUMBER: _ClassVar[int]
    CURRENT_CKPT_FIELD_NUMBER: _ClassVar[int]
    OPPONENT_CKPT_FIELD_NUMBER: _ClassVar[int]
    username: str
    species_indices: _containers.RepeatedScalarFieldContainer[int]
    packed_set_indices: _containers.RepeatedScalarFieldContainer[int]
    smogon_format: str
    current_ckpt: int
    opponent_ckpt: int
    def __init__(self, username: _Optional[str] = ..., species_indices: _Optional[_Iterable[int]] = ..., packed_set_indices: _Optional[_Iterable[int]] = ..., smogon_format: _Optional[str] = ..., current_ckpt: _Optional[int] = ..., opponent_ckpt: _Optional[int] = ...) -> None: ...

class EnvironmentState(_message.Message):
    __slots__ = ("info", "action_mask", "history_entity_public", "history_entity_revealed", "history_entity_edges", "history_field", "history_length", "moveset", "public_team", "revealed_team", "private_team", "field", "rqid", "history_packed_length")
    INFO_FIELD_NUMBER: _ClassVar[int]
    ACTION_MASK_FIELD_NUMBER: _ClassVar[int]
    HISTORY_ENTITY_PUBLIC_FIELD_NUMBER: _ClassVar[int]
    HISTORY_ENTITY_REVEALED_FIELD_NUMBER: _ClassVar[int]
    HISTORY_ENTITY_EDGES_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_FIELD_NUMBER: _ClassVar[int]
    HISTORY_LENGTH_FIELD_NUMBER: _ClassVar[int]
    MOVESET_FIELD_NUMBER: _ClassVar[int]
    PUBLIC_TEAM_FIELD_NUMBER: _ClassVar[int]
    REVEALED_TEAM_FIELD_NUMBER: _ClassVar[int]
    PRIVATE_TEAM_FIELD_NUMBER: _ClassVar[int]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    RQID_FIELD_NUMBER: _ClassVar[int]
    HISTORY_PACKED_LENGTH_FIELD_NUMBER: _ClassVar[int]
    info: bytes
    action_mask: bytes
    history_entity_public: bytes
    history_entity_revealed: bytes
    history_entity_edges: bytes
    history_field: bytes
    history_length: int
    moveset: bytes
    public_team: bytes
    revealed_team: bytes
    private_team: bytes
    field: bytes
    rqid: int
    history_packed_length: int
    def __init__(self, info: _Optional[bytes] = ..., action_mask: _Optional[bytes] = ..., history_entity_public: _Optional[bytes] = ..., history_entity_revealed: _Optional[bytes] = ..., history_entity_edges: _Optional[bytes] = ..., history_field: _Optional[bytes] = ..., history_length: _Optional[int] = ..., moveset: _Optional[bytes] = ..., public_team: _Optional[bytes] = ..., revealed_team: _Optional[bytes] = ..., private_team: _Optional[bytes] = ..., field: _Optional[bytes] = ..., rqid: _Optional[int] = ..., history_packed_length: _Optional[int] = ...) -> None: ...

class EnvironmentTrajectory(_message.Message):
    __slots__ = ("states",)
    STATES_FIELD_NUMBER: _ClassVar[int]
    states: _containers.RepeatedCompositeFieldContainer[EnvironmentState]
    def __init__(self, states: _Optional[_Iterable[_Union[EnvironmentState, _Mapping]]] = ...) -> None: ...

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
