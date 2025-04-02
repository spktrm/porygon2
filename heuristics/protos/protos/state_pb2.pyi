import enums_pb2 as _enums_pb2
import history_pb2 as _history_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Rewards(_message.Message):
    __slots__ = ("win_reward", "hp_reward", "fainted_reward", "scaled_fainted_reward", "scaled_hp_reward", "terminal_hp_reward", "terminal_fainted_reward")
    WIN_REWARD_FIELD_NUMBER: _ClassVar[int]
    HP_REWARD_FIELD_NUMBER: _ClassVar[int]
    FAINTED_REWARD_FIELD_NUMBER: _ClassVar[int]
    SCALED_FAINTED_REWARD_FIELD_NUMBER: _ClassVar[int]
    SCALED_HP_REWARD_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_HP_REWARD_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_FAINTED_REWARD_FIELD_NUMBER: _ClassVar[int]
    win_reward: float
    hp_reward: float
    fainted_reward: float
    scaled_fainted_reward: float
    scaled_hp_reward: float
    terminal_hp_reward: float
    terminal_fainted_reward: float
    def __init__(self, win_reward: _Optional[float] = ..., hp_reward: _Optional[float] = ..., fainted_reward: _Optional[float] = ..., scaled_fainted_reward: _Optional[float] = ..., scaled_hp_reward: _Optional[float] = ..., terminal_hp_reward: _Optional[float] = ..., terminal_fainted_reward: _Optional[float] = ...) -> None: ...

class Heuristics(_message.Message):
    __slots__ = ("heuristic_action",)
    HEURISTIC_ACTION_FIELD_NUMBER: _ClassVar[int]
    heuristic_action: int
    def __init__(self, heuristic_action: _Optional[int] = ...) -> None: ...

class Info(_message.Message):
    __slots__ = ("game_id", "done", "player_index", "turn", "ts", "draw_ratio", "worker_index", "rewards", "seed", "draw", "heuristics", "request_count", "timestamp")
    GAME_ID_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    PLAYER_INDEX_FIELD_NUMBER: _ClassVar[int]
    TURN_FIELD_NUMBER: _ClassVar[int]
    TS_FIELD_NUMBER: _ClassVar[int]
    DRAW_RATIO_FIELD_NUMBER: _ClassVar[int]
    WORKER_INDEX_FIELD_NUMBER: _ClassVar[int]
    REWARDS_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    DRAW_FIELD_NUMBER: _ClassVar[int]
    HEURISTICS_FIELD_NUMBER: _ClassVar[int]
    REQUEST_COUNT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    game_id: int
    done: bool
    player_index: bool
    turn: int
    ts: float
    draw_ratio: float
    worker_index: int
    rewards: Rewards
    seed: int
    draw: bool
    heuristics: Heuristics
    request_count: int
    timestamp: int
    def __init__(self, game_id: _Optional[int] = ..., done: bool = ..., player_index: bool = ..., turn: _Optional[int] = ..., ts: _Optional[float] = ..., draw_ratio: _Optional[float] = ..., worker_index: _Optional[int] = ..., rewards: _Optional[_Union[Rewards, _Mapping]] = ..., seed: _Optional[int] = ..., draw: bool = ..., heuristics: _Optional[_Union[Heuristics, _Mapping]] = ..., request_count: _Optional[int] = ..., timestamp: _Optional[int] = ...) -> None: ...

class State(_message.Message):
    __slots__ = ("info", "legal_actions", "history", "moveset", "public_team", "private_team", "key", "all_my_moves", "all_opp_moves", "all_my_moves_mask", "all_opp_moves_mask")
    INFO_FIELD_NUMBER: _ClassVar[int]
    LEGAL_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    MOVESET_FIELD_NUMBER: _ClassVar[int]
    PUBLIC_TEAM_FIELD_NUMBER: _ClassVar[int]
    PRIVATE_TEAM_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    ALL_MY_MOVES_FIELD_NUMBER: _ClassVar[int]
    ALL_OPP_MOVES_FIELD_NUMBER: _ClassVar[int]
    ALL_MY_MOVES_MASK_FIELD_NUMBER: _ClassVar[int]
    ALL_OPP_MOVES_MASK_FIELD_NUMBER: _ClassVar[int]
    info: Info
    legal_actions: bytes
    history: _history_pb2.History
    moveset: bytes
    public_team: bytes
    private_team: bytes
    key: str
    all_my_moves: bytes
    all_opp_moves: bytes
    all_my_moves_mask: bytes
    all_opp_moves_mask: bytes
    def __init__(self, info: _Optional[_Union[Info, _Mapping]] = ..., legal_actions: _Optional[bytes] = ..., history: _Optional[_Union[_history_pb2.History, _Mapping]] = ..., moveset: _Optional[bytes] = ..., public_team: _Optional[bytes] = ..., private_team: _Optional[bytes] = ..., key: _Optional[str] = ..., all_my_moves: _Optional[bytes] = ..., all_opp_moves: _Optional[bytes] = ..., all_my_moves_mask: _Optional[bytes] = ..., all_opp_moves_mask: _Optional[bytes] = ...) -> None: ...

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
