import history_pb2 as _history_pb2
import enums_pb2 as _enums_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Rewards(_message.Message):
    __slots__ = ("winReward", "hpReward", "faintedReward", "switchReward", "longevityReward")
    WINREWARD_FIELD_NUMBER: _ClassVar[int]
    HPREWARD_FIELD_NUMBER: _ClassVar[int]
    FAINTEDREWARD_FIELD_NUMBER: _ClassVar[int]
    SWITCHREWARD_FIELD_NUMBER: _ClassVar[int]
    LONGEVITYREWARD_FIELD_NUMBER: _ClassVar[int]
    winReward: float
    hpReward: float
    faintedReward: float
    switchReward: float
    longevityReward: float
    def __init__(self, winReward: _Optional[float] = ..., hpReward: _Optional[float] = ..., faintedReward: _Optional[float] = ..., switchReward: _Optional[float] = ..., longevityReward: _Optional[float] = ...) -> None: ...

class Info(_message.Message):
    __slots__ = ("gameId", "done", "playerIndex", "turn", "ts", "drawRatio", "workerIndex", "rewards", "seed", "draw")
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
    def __init__(self, gameId: _Optional[int] = ..., done: bool = ..., playerIndex: bool = ..., turn: _Optional[int] = ..., ts: _Optional[float] = ..., drawRatio: _Optional[float] = ..., workerIndex: _Optional[int] = ..., rewards: _Optional[_Union[Rewards, _Mapping]] = ..., seed: _Optional[int] = ..., draw: bool = ...) -> None: ...

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
