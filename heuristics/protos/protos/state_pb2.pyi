import history_pb2 as _history_pb2
import enums_pb2 as _enums_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Rewards(_message.Message):
    __slots__ = ("winReward", "hpReward", "faintedReward", "switchReward", "longevityReward", "faintedFibReward")
    WINREWARD_FIELD_NUMBER: _ClassVar[int]
    HPREWARD_FIELD_NUMBER: _ClassVar[int]
    FAINTEDREWARD_FIELD_NUMBER: _ClassVar[int]
    SWITCHREWARD_FIELD_NUMBER: _ClassVar[int]
    LONGEVITYREWARD_FIELD_NUMBER: _ClassVar[int]
    FAINTEDFIBREWARD_FIELD_NUMBER: _ClassVar[int]
    winReward: float
    hpReward: float
    faintedReward: float
    switchReward: float
    longevityReward: float
    faintedFibReward: float
    def __init__(self, winReward: _Optional[float] = ..., hpReward: _Optional[float] = ..., faintedReward: _Optional[float] = ..., switchReward: _Optional[float] = ..., longevityReward: _Optional[float] = ..., faintedFibReward: _Optional[float] = ...) -> None: ...

class Heuristics(_message.Message):
    __slots__ = ("heuristicAction",)
    HEURISTICACTION_FIELD_NUMBER: _ClassVar[int]
    heuristicAction: int
    def __init__(self, heuristicAction: _Optional[int] = ...) -> None: ...

class Info(_message.Message):
    __slots__ = ("gameId", "done", "playerIndex", "turn", "ts", "drawRatio", "workerIndex", "rewards", "seed", "draw", "heuristics", "requestCount")
    GAMEID_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    PLAYERINDEX_FIELD_NUMBER: _ClassVar[int]
    TURN_FIELD_NUMBER: _ClassVar[int]
    TS_FIELD_NUMBER: _ClassVar[int]
    DRAWRATIO_FIELD_NUMBER: _ClassVar[int]
    WORKERINDEX_FIELD_NUMBER: _ClassVar[int]
    REWARDS_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    DRAW_FIELD_NUMBER: _ClassVar[int]
    HEURISTICS_FIELD_NUMBER: _ClassVar[int]
    REQUESTCOUNT_FIELD_NUMBER: _ClassVar[int]
    gameId: int
    done: bool
    playerIndex: bool
    turn: int
    ts: float
    drawRatio: float
    workerIndex: int
    rewards: Rewards
    seed: int
    draw: bool
    heuristics: Heuristics
    requestCount: int
    def __init__(self, gameId: _Optional[int] = ..., done: bool = ..., playerIndex: bool = ..., turn: _Optional[int] = ..., ts: _Optional[float] = ..., drawRatio: _Optional[float] = ..., workerIndex: _Optional[int] = ..., rewards: _Optional[_Union[Rewards, _Mapping]] = ..., seed: _Optional[int] = ..., draw: bool = ..., heuristics: _Optional[_Union[Heuristics, _Mapping]] = ..., requestCount: _Optional[int] = ...) -> None: ...

class State(_message.Message):
    __slots__ = ("info", "legalActions", "history", "moveset", "team", "key")
    INFO_FIELD_NUMBER: _ClassVar[int]
    LEGALACTIONS_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    MOVESET_FIELD_NUMBER: _ClassVar[int]
    TEAM_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    info: Info
    legalActions: bytes
    history: _history_pb2.History
    moveset: bytes
    team: bytes
    key: str
    def __init__(self, info: _Optional[_Union[Info, _Mapping]] = ..., legalActions: _Optional[bytes] = ..., history: _Optional[_Union[_history_pb2.History, _Mapping]] = ..., moveset: _Optional[bytes] = ..., team: _Optional[bytes] = ..., key: _Optional[str] = ...) -> None: ...

class Trajectory(_message.Message):
    __slots__ = ("states", "actions", "rewards")
    STATES_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    REWARDS_FIELD_NUMBER: _ClassVar[int]
    states: _containers.RepeatedCompositeFieldContainer[State]
    actions: _containers.RepeatedScalarFieldContainer[int]
    rewards: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, states: _Optional[_Iterable[_Union[State, _Mapping]]] = ..., actions: _Optional[_Iterable[int]] = ..., rewards: _Optional[_Iterable[int]] = ...) -> None: ...

class Dataset(_message.Message):
    __slots__ = ("trajectories",)
    TRAJECTORIES_FIELD_NUMBER: _ClassVar[int]
    trajectories: _containers.RepeatedCompositeFieldContainer[Trajectory]
    def __init__(self, trajectories: _Optional[_Iterable[_Union[Trajectory, _Mapping]]] = ...) -> None: ...
