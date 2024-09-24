import os

from rlenv.protos.enums_pb2 import (
    AbilitiesEnum,
    BoostsEnum,
    GendernameEnum,
    ItemeffecttypesEnum,
    ItemsEnum,
    LastitemeffecttypesEnum,
    MovesEnum,
    PseudoweatherEnum,
    SideconditionEnum,
    SpeciesEnum,
    StatusEnum,
    TerrainEnum,
    VolatilestatusEnum,
    WeatherEnum,
)
from rlenv.protos.features_pb2 import FeatureMoveset
from rlenv.protos.state_pb2 import State


with open(os.path.join(os.path.dirname(__file__), "ex"), "rb") as f:
    EX_BUFFER = f.read()

EX_STATE = State.FromString(EX_BUFFER)

NUM_GENDERS = len(GendernameEnum.keys())
NUM_STATUS = len(StatusEnum.keys())
# NUM_TYPES = len(TypesEnum.keys())
NUM_VOLATILE_STATUS = len(VolatilestatusEnum.keys())
NUM_SIDE_CONDITION = len(SideconditionEnum.keys())
NUM_BOOSTS = len(BoostsEnum.keys())
NUM_PSEUDOWEATHER = len(PseudoweatherEnum.keys())
NUM_WEATHER = len(WeatherEnum.keys())
NUM_TERRAIN = len(TerrainEnum.keys())
NUM_SPECIES = len(SpeciesEnum.keys())
NUM_MOVES = len(MovesEnum.keys())
NUM_ABILITIES = len(AbilitiesEnum.keys())
NUM_ITEMS = len(ItemsEnum.keys())
NUM_ITEM_EFFECTS = len(ItemeffecttypesEnum.keys())
NUM_LAST_ITEM_EFFECTS = len(LastitemeffecttypesEnum.keys())

NUM_MOVE_FIELDS = len(FeatureMoveset.DESCRIPTOR.values)

NUM_HISTORY = 8
NUM_PLAYERS = 2

BASE_SOCKET_PATH = "/tmp/pokemon-{}.sock"
TRAINING_SOCKET_PATH = BASE_SOCKET_PATH.format("training")
EVALUATION_SOCKET_PATH = BASE_SOCKET_PATH.format("evaluation")
SocketPath = str

SPIKES_TOKEN = SideconditionEnum.SIDECONDITION_SPIKES
TOXIC_SPIKES_TOKEN = SideconditionEnum.SIDECONDITION_TOXICSPIKES
