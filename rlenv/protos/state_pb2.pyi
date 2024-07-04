import history_pb2 as _history_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

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
    __slots__ = ("gameId", "done", "playerOneReward", "playerTwoReward", "playerIndex", "turn")
    GAMEID_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    PLAYERONEREWARD_FIELD_NUMBER: _ClassVar[int]
    PLAYERTWOREWARD_FIELD_NUMBER: _ClassVar[int]
    PLAYERINDEX_FIELD_NUMBER: _ClassVar[int]
    TURN_FIELD_NUMBER: _ClassVar[int]
    gameId: int
    done: bool
    playerOneReward: float
    playerTwoReward: float
    playerIndex: bool
    turn: int
    def __init__(self, gameId: _Optional[int] = ..., done: bool = ..., playerOneReward: _Optional[float] = ..., playerTwoReward: _Optional[float] = ..., playerIndex: bool = ..., turn: _Optional[int] = ...) -> None: ...

class State(_message.Message):
    __slots__ = ("info", "legalActions", "history")
    INFO_FIELD_NUMBER: _ClassVar[int]
    LEGALACTIONS_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    info: Info
    legalActions: LegalActions
    history: _containers.RepeatedCompositeFieldContainer[_history_pb2.HistoryStep]
    def __init__(self, info: _Optional[_Union[Info, _Mapping]] = ..., legalActions: _Optional[_Union[LegalActions, _Mapping]] = ..., history: _Optional[_Iterable[_Union[_history_pb2.HistoryStep, _Mapping]]] = ...) -> None: ...
