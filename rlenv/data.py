import os

from rlenv.protos.enums_pb2 import (
    AbilitiesEnum,
    BoostsEnum,
    GendersEnum,
    HyphenargsEnum,
    ItemeffectEnum,
    ItemsEnum,
    MovesEnum,
    PseudoweatherEnum,
    SideconditionsEnum,
    SpeciesEnum,
    StatusesEnum,
    TerrainEnum,
    TypesEnum,
    VolatilestatusEnum,
    WeathersEnum,
)
from rlenv.protos.features_pb2 import FeatureMoveset
from rlenv.protos.state_pb2 import State


with open(os.path.join(os.path.dirname(__file__), "ex"), "rb") as f:
    EX_BUFFER = f.read()

EX_STATE = State.FromString(EX_BUFFER)

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
NUM_SPECIES = len(SpeciesEnum.keys())
NUM_MOVES = len(MovesEnum.keys())
NUM_ABILITIES = len(AbilitiesEnum.keys())
NUM_ITEMS = len(ItemsEnum.keys())
NUM_ITEM_EFFECTS = len(ItemeffectEnum.keys())

NUM_MOVE_FIELDS = len(FeatureMoveset.DESCRIPTOR.values)

NUM_HISTORY = 8
NUM_PLAYERS = 2

BASE_SOCKET_PATH = "/tmp/pokemon-{}.sock"
TRAINING_SOCKET_PATH = BASE_SOCKET_PATH.format("training")
EVALUATION_SOCKET_PATH = BASE_SOCKET_PATH.format("evaluation")
SocketPath = str

SPIKES_TOKEN = SideconditionsEnum.sideConditions_spikes
TOXIC_SPIKES_TOKEN = SideconditionsEnum.sideConditions_toxicspikes
