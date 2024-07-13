import os

from enum import Enum, auto
from typing import Union
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
from rlenv.protos.state_pb2 import Move, State


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

NUM_MOVE_FIELDS = len(Move.DESCRIPTOR.fields)

NUM_HISTORY = 8
NUM_PLAYERS = 2

BASE_SOCKET_PATH = "/tmp/pokemon-{}.sock"
TRAINING_SOCKET_PATH = BASE_SOCKET_PATH.format("training")
EVALUATION_SOCKET_PATH = BASE_SOCKET_PATH.format("evaluation")
SocketPath = str

SPIKES_TOKEN = SideconditionsEnum.sideConditions_spikes
TOXIC_SPIKES_TOKEN = SideconditionsEnum.sideConditions_toxicspikes


class FeatureEntity(Enum):
    SPECIES = 0
    ITEM = auto()
    ITEM_EFFECT = auto()
    ABILITY = auto()
    GENDER = auto()
    ACTIVE = auto()
    FAINTED = auto()
    HP = auto()
    MAXHP = auto()
    STATUS = auto()
    TOXIC_TURNS = auto()
    SLEEP_TURNS = auto()
    BEING_CALLED_BACK = auto()
    TRAPPED = auto()
    NEWLY_SWITCHED = auto()
    LEVEL = auto()
    MOVEID0 = auto()
    MOVEID1 = auto()
    MOVEID2 = auto()
    MOVEID3 = auto()
    MOVEPP0 = auto()
    MOVEPP1 = auto()
    MOVEPP2 = auto()
    MOVEPP3 = auto()


class FeatureMoveset(Enum):
    MOVEID = 0
    PPLEFT = auto()
    PPMAX = auto()


class FeatureTurnContext(Enum):
    VALID = 0
    IS_MY_TURN = auto()
    ACTION = auto()
    MOVE = auto()
    SWITCH_COUNTER = auto()
    MOVE_COUNTER = auto()
    TURN = auto()


class FeatureWeather(Enum):
    WEATHER_ID = 0
    MIN_DURATION = auto()
    MAX_DURATION = auto()
