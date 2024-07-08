from rlenv.protos.pokemon_pb2 import Pokemon
from rlenv.protos.messages_pb2 import (
    BoostsMessage,
    PseudoweatherMessage,
    SideconditionsMessage,
    TerrainMessage,
    VolatilestatusMessage,
)


NUM_POKEMON_FIELDS = len(Pokemon.DESCRIPTOR.fields)
NUM_VOLATILE_STATUS_FIELDS = len(VolatilestatusMessage.DESCRIPTOR.fields)
NUM_SIDE_CONDITION_FIELDS = len(SideconditionsMessage.DESCRIPTOR.fields)
NUM_BOOSTS_FIELDS = len(BoostsMessage.DESCRIPTOR.fields)
NUM_PSEUDOWEATHER_FIELDS = len(PseudoweatherMessage.DESCRIPTOR.fields)
NUM_TERRAIN_FIELDS = len(TerrainMessage.DESCRIPTOR.fields)
