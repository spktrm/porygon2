import os

from rlenv.protos.enums_pb2 import (
    BoostsEnum,
    GendersEnum,
    HyphenargsEnum,
    PseudoweatherEnum,
    SideconditionsEnum,
    StatusesEnum,
    TerrainEnum,
    TypesEnum,
    VolatilestatusEnum,
    WeathersEnum,
)
from rlenv.protos.state_pb2 import Move, State


with open(os.path.join(os.path.dirname(__file__), "ex"), "rb") as f:
    EX_BUFFER = f.read()

EX_STATE = State.FromString(EX_BUFFER)

NUM_POKEMON = 19
NUM_GENDERS = len(GendersEnum.keys())
NUM_STATUS = len(StatusesEnum.keys())
NUM_TYPES = len(TypesEnum.keys())
NUM_VOLATILE_STATUS = len(VolatilestatusEnum.keys())
NUM_SIDE_CONDITION = len(SideconditionsEnum.keys())
NUM_BOOSTS = len(BoostsEnum.keys())
NUM_HYPHEN_ARGS = len(HyphenargsEnum.keys())
NUM_PSEUDOWEATHER = len(PseudoweatherEnum.keys())
NUM_WEATHER = len(WeathersEnum.keys())
NUM_TERRAIN = len(TerrainEnum.keys())

NUM_MOVE_FIELDS = len(Move.DESCRIPTOR.fields)

NUM_HISTORY = 8
NUM_PLAYERS = 2
SOCKET_PATH = "/tmp/pokemon.sock"

SPIKES_TOKEN = SideconditionsEnum.sideConditions_spikes
TOXIC_SPIKES_TOKEN = SideconditionsEnum.sideConditions_toxicspikes
