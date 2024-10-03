import history_pb2 as _history_pb2
import enums_pb2 as _enums_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LegalActions(_message.Message):
    __slots__ = ("move1", "move2", "move3", "move4", "switch1", "switch2", "switch3", "switch4", "switch5", "switch6")
    MOVE1_FIELD_NUMBER: _ClassVar[int]
    MOVE2_FIELD_NUMBER: _ClassVar[int]
    MOVE3_FIELD_NUMBER: _ClassVar[int]
    MOVE4_FIELD_NUMBER: _ClassVar[int]
    SWITCH1_FIELD_NUMBER: _ClassVar[int]
    SWITCH2_FIELD_NUMBER: _ClassVar[int]
    SWITCH3_FIELD_NUMBER: _ClassVar[int]
    SWITCH4_FIELD_NUMBER: _ClassVar[int]
    SWITCH5_FIELD_NUMBER: _ClassVar[int]
    SWITCH6_FIELD_NUMBER: _ClassVar[int]
    move1: bool
    move2: bool
    move3: bool
    move4: bool
    switch1: bool
    switch2: bool
    switch3: bool
    switch4: bool
    switch5: bool
    switch6: bool
    def __init__(self, move1: bool = ..., move2: bool = ..., move3: bool = ..., move4: bool = ..., switch1: bool = ..., switch2: bool = ..., switch3: bool = ..., switch4: bool = ..., switch5: bool = ..., switch6: bool = ...) -> None: ...

class Info(_message.Message):
    __slots__ = ("gameId", "done", "winReward", "hpReward", "playerIndex", "turn", "turnsSinceSwitch", "heuristicAction", "lastAction", "lastMove", "faintedReward", "heuristicDist", "switchReward")
    GAMEID_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    WINREWARD_FIELD_NUMBER: _ClassVar[int]
    HPREWARD_FIELD_NUMBER: _ClassVar[int]
    PLAYERINDEX_FIELD_NUMBER: _ClassVar[int]
    TURN_FIELD_NUMBER: _ClassVar[int]
    TURNSSINCESWITCH_FIELD_NUMBER: _ClassVar[int]
    HEURISTICACTION_FIELD_NUMBER: _ClassVar[int]
    LASTACTION_FIELD_NUMBER: _ClassVar[int]
    LASTMOVE_FIELD_NUMBER: _ClassVar[int]
    FAINTEDREWARD_FIELD_NUMBER: _ClassVar[int]
    HEURISTICDIST_FIELD_NUMBER: _ClassVar[int]
    SWITCHREWARD_FIELD_NUMBER: _ClassVar[int]
    gameId: int
    done: bool
    winReward: float
    hpReward: float
    playerIndex: bool
    turn: int
    turnsSinceSwitch: int
    heuristicAction: int
    lastAction: int
    lastMove: int
    faintedReward: float
    heuristicDist: bytes
    switchReward: float
    def __init__(self, gameId: _Optional[int] = ..., done: bool = ..., winReward: _Optional[float] = ..., hpReward: _Optional[float] = ..., playerIndex: bool = ..., turn: _Optional[int] = ..., turnsSinceSwitch: _Optional[int] = ..., heuristicAction: _Optional[int] = ..., lastAction: _Optional[int] = ..., lastMove: _Optional[int] = ..., faintedReward: _Optional[float] = ..., heuristicDist: _Optional[bytes] = ..., switchReward: _Optional[float] = ...) -> None: ...

class State(_message.Message):
    __slots__ = ("info", "legalActions", "history", "moveset", "team", "key")
    INFO_FIELD_NUMBER: _ClassVar[int]
    LEGALACTIONS_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    MOVESET_FIELD_NUMBER: _ClassVar[int]
    TEAM_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    info: Info
    legalActions: LegalActions
    history: _history_pb2.History
    moveset: bytes
    team: bytes
    key: str
    def __init__(self, info: _Optional[_Union[Info, _Mapping]] = ..., legalActions: _Optional[_Union[LegalActions, _Mapping]] = ..., history: _Optional[_Union[_history_pb2.History, _Mapping]] = ..., moveset: _Optional[bytes] = ..., team: _Optional[bytes] = ..., key: _Optional[str] = ...) -> None: ...