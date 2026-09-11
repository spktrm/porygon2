from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ModalityEnum(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODALITY_ENUM___UNSPECIFIED: _ClassVar[ModalityEnum]
    MODALITY_ENUM__MOVE: _ClassVar[ModalityEnum]
    MODALITY_ENUM__SWITCH: _ClassVar[ModalityEnum]
    MODALITY_ENUM__WILDCARD: _ClassVar[ModalityEnum]
    MODALITY_ENUM__OTHER: _ClassVar[ModalityEnum]

class MoveSlot(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MOVE_SLOT___UNSPECIFIED: _ClassVar[MoveSlot]
    MOVE_SLOT__ALLY_1_MOVE_1: _ClassVar[MoveSlot]
    MOVE_SLOT__ALLY_1_MOVE_2: _ClassVar[MoveSlot]
    MOVE_SLOT__ALLY_1_MOVE_3: _ClassVar[MoveSlot]
    MOVE_SLOT__ALLY_1_MOVE_4: _ClassVar[MoveSlot]
    MOVE_SLOT__ALLY_1_MOVE_1_WILDCARD: _ClassVar[MoveSlot]
    MOVE_SLOT__ALLY_1_MOVE_2_WILDCARD: _ClassVar[MoveSlot]
    MOVE_SLOT__ALLY_1_MOVE_3_WILDCARD: _ClassVar[MoveSlot]
    MOVE_SLOT__ALLY_1_MOVE_4_WILDCARD: _ClassVar[MoveSlot]
    MOVE_SLOT__ALLY_2_MOVE_1: _ClassVar[MoveSlot]
    MOVE_SLOT__ALLY_2_MOVE_2: _ClassVar[MoveSlot]
    MOVE_SLOT__ALLY_2_MOVE_3: _ClassVar[MoveSlot]
    MOVE_SLOT__ALLY_2_MOVE_4: _ClassVar[MoveSlot]
    MOVE_SLOT__ALLY_2_MOVE_1_WILDCARD: _ClassVar[MoveSlot]
    MOVE_SLOT__ALLY_2_MOVE_2_WILDCARD: _ClassVar[MoveSlot]
    MOVE_SLOT__ALLY_2_MOVE_3_WILDCARD: _ClassVar[MoveSlot]
    MOVE_SLOT__ALLY_2_MOVE_4_WILDCARD: _ClassVar[MoveSlot]

class ReserveSlot(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RESERVE_SLOT___UNSPECIFIED: _ClassVar[ReserveSlot]
    RESERVE_SLOT__RESERVE_1: _ClassVar[ReserveSlot]
    RESERVE_SLOT__RESERVE_2: _ClassVar[ReserveSlot]
    RESERVE_SLOT__RESERVE_3: _ClassVar[ReserveSlot]
    RESERVE_SLOT__RESERVE_4: _ClassVar[ReserveSlot]
    RESERVE_SLOT__RESERVE_5: _ClassVar[ReserveSlot]
    RESERVE_SLOT__RESERVE_6: _ClassVar[ReserveSlot]

class TargetSlot(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TARGET_SLOT___UNSPECIFIED: _ClassVar[TargetSlot]
    TARGET_SLOT__DEFAULT: _ClassVar[TargetSlot]
    TARGET_SLOT__ALLY_1: _ClassVar[TargetSlot]
    TARGET_SLOT__ALLY_1_PASS: _ClassVar[TargetSlot]
    TARGET_SLOT__ALLY_2: _ClassVar[TargetSlot]
    TARGET_SLOT__ALLY_2_PASS: _ClassVar[TargetSlot]
    TARGET_SLOT__ENEMY_1: _ClassVar[TargetSlot]
    TARGET_SLOT__ENEMY_2: _ClassVar[TargetSlot]
    TARGET_SLOT__AUTO: _ClassVar[TargetSlot]
    TARGET_SLOT__ALL: _ClassVar[TargetSlot]
    TARGET_SLOT__ALLY_SIDE: _ClassVar[TargetSlot]
    TARGET_SLOT__FOE_SIDE: _ClassVar[TargetSlot]
    TARGET_SLOT__ALLY_TEAM: _ClassVar[TargetSlot]
    TARGET_SLOT__RANDOM_NORMAL: _ClassVar[TargetSlot]
    TARGET_SLOT__ALL_ADJACENT: _ClassVar[TargetSlot]
    TARGET_SLOT__ALL_ADJACENT_FOES: _ClassVar[TargetSlot]
    TARGET_SLOT__ALLIES: _ClassVar[TargetSlot]

class ActionRequestKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTION_REQUEST_KIND___UNSPECIFIED: _ClassVar[ActionRequestKind]
    ACTION_REQUEST_KIND__MOVE: _ClassVar[ActionRequestKind]
    ACTION_REQUEST_KIND__FORCE_SWITCH: _ClassVar[ActionRequestKind]
    ACTION_REQUEST_KIND__TEAM_PREVIEW: _ClassVar[ActionRequestKind]
    ACTION_REQUEST_KIND__WAIT: _ClassVar[ActionRequestKind]
MODALITY_ENUM___UNSPECIFIED: ModalityEnum
MODALITY_ENUM__MOVE: ModalityEnum
MODALITY_ENUM__SWITCH: ModalityEnum
MODALITY_ENUM__WILDCARD: ModalityEnum
MODALITY_ENUM__OTHER: ModalityEnum
MOVE_SLOT___UNSPECIFIED: MoveSlot
MOVE_SLOT__ALLY_1_MOVE_1: MoveSlot
MOVE_SLOT__ALLY_1_MOVE_2: MoveSlot
MOVE_SLOT__ALLY_1_MOVE_3: MoveSlot
MOVE_SLOT__ALLY_1_MOVE_4: MoveSlot
MOVE_SLOT__ALLY_1_MOVE_1_WILDCARD: MoveSlot
MOVE_SLOT__ALLY_1_MOVE_2_WILDCARD: MoveSlot
MOVE_SLOT__ALLY_1_MOVE_3_WILDCARD: MoveSlot
MOVE_SLOT__ALLY_1_MOVE_4_WILDCARD: MoveSlot
MOVE_SLOT__ALLY_2_MOVE_1: MoveSlot
MOVE_SLOT__ALLY_2_MOVE_2: MoveSlot
MOVE_SLOT__ALLY_2_MOVE_3: MoveSlot
MOVE_SLOT__ALLY_2_MOVE_4: MoveSlot
MOVE_SLOT__ALLY_2_MOVE_1_WILDCARD: MoveSlot
MOVE_SLOT__ALLY_2_MOVE_2_WILDCARD: MoveSlot
MOVE_SLOT__ALLY_2_MOVE_3_WILDCARD: MoveSlot
MOVE_SLOT__ALLY_2_MOVE_4_WILDCARD: MoveSlot
RESERVE_SLOT___UNSPECIFIED: ReserveSlot
RESERVE_SLOT__RESERVE_1: ReserveSlot
RESERVE_SLOT__RESERVE_2: ReserveSlot
RESERVE_SLOT__RESERVE_3: ReserveSlot
RESERVE_SLOT__RESERVE_4: ReserveSlot
RESERVE_SLOT__RESERVE_5: ReserveSlot
RESERVE_SLOT__RESERVE_6: ReserveSlot
TARGET_SLOT___UNSPECIFIED: TargetSlot
TARGET_SLOT__DEFAULT: TargetSlot
TARGET_SLOT__ALLY_1: TargetSlot
TARGET_SLOT__ALLY_1_PASS: TargetSlot
TARGET_SLOT__ALLY_2: TargetSlot
TARGET_SLOT__ALLY_2_PASS: TargetSlot
TARGET_SLOT__ENEMY_1: TargetSlot
TARGET_SLOT__ENEMY_2: TargetSlot
TARGET_SLOT__AUTO: TargetSlot
TARGET_SLOT__ALL: TargetSlot
TARGET_SLOT__ALLY_SIDE: TargetSlot
TARGET_SLOT__FOE_SIDE: TargetSlot
TARGET_SLOT__ALLY_TEAM: TargetSlot
TARGET_SLOT__RANDOM_NORMAL: TargetSlot
TARGET_SLOT__ALL_ADJACENT: TargetSlot
TARGET_SLOT__ALL_ADJACENT_FOES: TargetSlot
TARGET_SLOT__ALLIES: TargetSlot
ACTION_REQUEST_KIND___UNSPECIFIED: ActionRequestKind
ACTION_REQUEST_KIND__MOVE: ActionRequestKind
ACTION_REQUEST_KIND__FORCE_SWITCH: ActionRequestKind
ACTION_REQUEST_KIND__TEAM_PREVIEW: ActionRequestKind
ACTION_REQUEST_KIND__WAIT: ActionRequestKind

class ClientRequest(_message.Message):
    __slots__ = ("step", "reset")
    STEP_FIELD_NUMBER: _ClassVar[int]
    RESET_FIELD_NUMBER: _ClassVar[int]
    step: StepRequest
    reset: ResetRequest
    def __init__(self, step: _Optional[_Union[StepRequest, _Mapping]] = ..., reset: _Optional[_Union[ResetRequest, _Mapping]] = ...) -> None: ...

class ActionMask(_message.Message):
    __slots__ = ("kind", "switch_slots", "move_targets", "other_srcs", "active_slot")
    KIND_FIELD_NUMBER: _ClassVar[int]
    SWITCH_SLOTS_FIELD_NUMBER: _ClassVar[int]
    MOVE_TARGETS_FIELD_NUMBER: _ClassVar[int]
    OTHER_SRCS_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_SLOT_FIELD_NUMBER: _ClassVar[int]
    kind: ActionRequestKind
    switch_slots: int
    move_targets: _containers.RepeatedScalarFieldContainer[int]
    other_srcs: int
    active_slot: int
    def __init__(self, kind: _Optional[_Union[ActionRequestKind, str]] = ..., switch_slots: _Optional[int] = ..., move_targets: _Optional[_Iterable[int]] = ..., other_srcs: _Optional[int] = ..., active_slot: _Optional[int] = ...) -> None: ...

class Action(_message.Message):
    __slots__ = ("cell",)
    CELL_FIELD_NUMBER: _ClassVar[int]
    cell: int
    def __init__(self, cell: _Optional[int] = ...) -> None: ...

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
    __slots__ = ("username", "smogon_format", "game_id", "packed_teams")
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    SMOGON_FORMAT_FIELD_NUMBER: _ClassVar[int]
    GAME_ID_FIELD_NUMBER: _ClassVar[int]
    PACKED_TEAMS_FIELD_NUMBER: _ClassVar[int]
    username: str
    smogon_format: str
    game_id: str
    packed_teams: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, username: _Optional[str] = ..., smogon_format: _Optional[str] = ..., game_id: _Optional[str] = ..., packed_teams: _Optional[_Iterable[int]] = ...) -> None: ...

class EnvironmentState(_message.Message):
    __slots__ = ("info", "history_entity_public_cache", "history_entity_revealed_cache", "history_entity_edge_cache", "history_field", "history_length", "my_moveset", "opp_moveset", "public_team", "revealed_team", "private_team", "field", "rqid", "history_packed_length", "structured_action_mask", "opp_private_team", "history_rewrite_count")
    INFO_FIELD_NUMBER: _ClassVar[int]
    HISTORY_ENTITY_PUBLIC_CACHE_FIELD_NUMBER: _ClassVar[int]
    HISTORY_ENTITY_REVEALED_CACHE_FIELD_NUMBER: _ClassVar[int]
    HISTORY_ENTITY_EDGE_CACHE_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_FIELD_NUMBER: _ClassVar[int]
    HISTORY_LENGTH_FIELD_NUMBER: _ClassVar[int]
    MY_MOVESET_FIELD_NUMBER: _ClassVar[int]
    OPP_MOVESET_FIELD_NUMBER: _ClassVar[int]
    PUBLIC_TEAM_FIELD_NUMBER: _ClassVar[int]
    REVEALED_TEAM_FIELD_NUMBER: _ClassVar[int]
    PRIVATE_TEAM_FIELD_NUMBER: _ClassVar[int]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    RQID_FIELD_NUMBER: _ClassVar[int]
    HISTORY_PACKED_LENGTH_FIELD_NUMBER: _ClassVar[int]
    STRUCTURED_ACTION_MASK_FIELD_NUMBER: _ClassVar[int]
    OPP_PRIVATE_TEAM_FIELD_NUMBER: _ClassVar[int]
    HISTORY_REWRITE_COUNT_FIELD_NUMBER: _ClassVar[int]
    info: bytes
    history_entity_public_cache: bytes
    history_entity_revealed_cache: bytes
    history_entity_edge_cache: bytes
    history_field: bytes
    history_length: int
    my_moveset: bytes
    opp_moveset: bytes
    public_team: bytes
    revealed_team: bytes
    private_team: bytes
    field: bytes
    rqid: int
    history_packed_length: int
    structured_action_mask: ActionMask
    opp_private_team: bytes
    history_rewrite_count: int
    def __init__(self, info: _Optional[bytes] = ..., history_entity_public_cache: _Optional[bytes] = ..., history_entity_revealed_cache: _Optional[bytes] = ..., history_entity_edge_cache: _Optional[bytes] = ..., history_field: _Optional[bytes] = ..., history_length: _Optional[int] = ..., my_moveset: _Optional[bytes] = ..., opp_moveset: _Optional[bytes] = ..., public_team: _Optional[bytes] = ..., revealed_team: _Optional[bytes] = ..., private_team: _Optional[bytes] = ..., field: _Optional[bytes] = ..., rqid: _Optional[int] = ..., history_packed_length: _Optional[int] = ..., structured_action_mask: _Optional[_Union[ActionMask, _Mapping]] = ..., opp_private_team: _Optional[bytes] = ..., history_rewrite_count: _Optional[int] = ...) -> None: ...

class EnvironmentTrajectory(_message.Message):
    __slots__ = ("states",)
    STATES_FIELD_NUMBER: _ClassVar[int]
    states: _containers.RepeatedCompositeFieldContainer[EnvironmentState]
    def __init__(self, states: _Optional[_Iterable[_Union[EnvironmentState, _Mapping]]] = ...) -> None: ...

class EnvironmentBatch(_message.Message):
    __slots__ = ("trajectories", "max_trajectory_length")
    TRAJECTORIES_FIELD_NUMBER: _ClassVar[int]
    MAX_TRAJECTORY_LENGTH_FIELD_NUMBER: _ClassVar[int]
    trajectories: _containers.RepeatedCompositeFieldContainer[EnvironmentTrajectory]
    max_trajectory_length: int
    def __init__(self, trajectories: _Optional[_Iterable[_Union[EnvironmentTrajectory, _Mapping]]] = ..., max_trajectory_length: _Optional[int] = ...) -> None: ...

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
