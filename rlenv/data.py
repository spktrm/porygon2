import os
from rlenv.protos.enums_pb2 import (
    BoostsEnum,
    HyphenargsEnum,
    PseudoweatherEnum,
    SideconditionsEnum,
    TerrainEnum,
    VolatilestatusEnum,
)
from rlenv.protos.state_pb2 import Move, State


with open(os.path.join(os.path.dirname(__file__), "ex"), "rb") as f:
    EX_BUFFER = f.read()

EX_STATE = State.FromString(EX_BUFFER)

NUM_POKEMON_FIELDS = 19
NUM_VOLATILE_STATUS_FIELDS = len(VolatilestatusEnum.keys())
NUM_SIDE_CONDITION_FIELDS = len(SideconditionsEnum.keys())
NUM_BOOSTS_FIELDS = len(BoostsEnum.keys())
NUM_HYPHEN_ARGS_FIELDS = len(HyphenargsEnum.keys())
NUM_PSEUDOWEATHER_FIELDS = len(PseudoweatherEnum.keys())
NUM_TERRAIN_FIELDS = len(TerrainEnum.keys())

NUM_MOVE_FIELDS = len(Move.DESCRIPTOR.fields)
