import json
import os
import time
import traceback

import jax.numpy as jnp
import numpy as np

from constants import MAX_RATIO_TOKEN
from rl.environment.protos.enums_pb2 import (
    AbilitiesEnum,
    BattlemajorargsEnum,
    BattleminorargsEnum,
    BoostsEnum,
    EffectEnum,
    GendernameEnum,
    ItemeffecttypesEnum,
    ItemsEnum,
    LastitemeffecttypesEnum,
    MovesEnum,
    NaturesEnum,
    PseudoweatherEnum,
    SideconditionEnum,
    SpeciesEnum,
    StatusEnum,
    TerrainEnum,
    TypechartEnum,
    VolatilestatusEnum,
    WeatherEnum,
)
from rl.environment.protos.features_pb2 import (
    ActionType,
    EntityEdgeFeature,
    EntityPrivateNodeFeature,
    EntityPublicNodeFeature,
    EntityRevealedNodeFeature,
    FieldFeature,
    InfoFeature,
    MovesetFeature,
    MovesetHasPP,
    PackedSetFeature,
    RequestType,
)
from rl.environment.protos.service_pb2 import ActionEnum, EnvironmentBatch, ModalityEnum
from rl.model.modules import PretrainedEmbedding, ZeroEmbedding

NUM_GENDERS = len(GendernameEnum.keys())
NUM_STATUS = len(StatusEnum.keys())
# NUM_TYPES = len(TypesEnum.keys())
NUM_VOLATILE_STATUS = len(VolatilestatusEnum.keys())
NUM_TYPECHART = len(TypechartEnum.keys())
NUM_SIDE_CONDITION = len(SideconditionEnum.keys())
NUM_BOOSTS = len(BoostsEnum.keys())
NUM_PSEUDOWEATHER = len(PseudoweatherEnum.keys())
NUM_WEATHER = len(WeatherEnum.keys())
NUM_TERRAIN = len(TerrainEnum.keys())
NUM_SPECIES = len(SpeciesEnum.keys())
NUM_MOVES = len(MovesEnum.keys())
NUM_FROM_SOURCE_EFFECTS = len(EffectEnum.keys())
NUM_ACTION_TYPES = len(ActionType.keys())
NUM_HAS_PP = len(MovesetHasPP.keys())
NUM_ABILITIES = len(AbilitiesEnum.keys())
NUM_ITEMS = len(ItemsEnum.keys())
NUM_MINOR_ARGS = len(BattleminorargsEnum.keys())
NUM_MAJOR_ARGS = len(BattlemajorargsEnum.keys())
NUM_ITEM_EFFECTS = len(ItemeffecttypesEnum.keys())
NUM_NATURES = len(NaturesEnum.keys())
NUM_LAST_ITEM_EFFECTS = len(LastitemeffecttypesEnum.keys())
NUM_EFFECTS = len(EffectEnum.keys())
NUM_MOVE_FEATURES = len(MovesetFeature.keys())
NUM_ENTITY_EDGE_FEATURES = len(EntityEdgeFeature.keys())
NUM_FIELD_FEATURES = len(FieldFeature.keys())
NUM_ENTITY_PRIVATE_FEATURES = len(EntityPrivateNodeFeature.keys())
NUM_ENTITY_PUBLIC_FEATURES = len(EntityPublicNodeFeature.keys())
NUM_ENTITY_REVEALED_FEATURES = len(EntityRevealedNodeFeature.keys())
NUM_ACTION_FEATURES = len(ActionEnum.keys())
NUM_MODALITY_FEATURES = len(ModalityEnum.keys())

SPIKES_TOKEN = SideconditionEnum.SIDECONDITION_ENUM__SPIKES
TOXIC_SPIKES_TOKEN = SideconditionEnum.SIDECONDITION_ENUM__TOXICSPIKES


ENTITY_PUBLIC_MAX_VALUES = {
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__LEVEL: 100,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO: MAX_RATIO_TOKEN,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__GENDER: NUM_GENDERS,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__STATUS: NUM_STATUS,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ITEM_EFFECT: NUM_ITEM_EFFECTS,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BEING_CALLED_BACK: 2,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TRAPPED: 2,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__NEWLY_SWITCHED: 2,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TOXIC_TURNS: 8,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SLEEP_TURNS: 4,
    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED: 2,
}

ENTITY_PRIVATE_MAX_VALUES = {
    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__TERA_TYPE: NUM_TYPECHART,
    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__HP_RATIO: MAX_RATIO_TOKEN,
    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__STATUS: NUM_STATUS,
    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__HAS_STATUS: 2,
    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__TOXIC_TURNS: 8,
    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SLEEP_TURNS: 4,
    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__FAINTED: 2,
    # Turn-delta staleness of the source request, clipped 0..8 by the
    # service; identically 0 on the own channel.
    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__REQUEST_LAG: 9,
}


ENTITY_EDGE_MAX_VALUES = {
    EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG: NUM_MAJOR_ARGS,
    EntityEdgeFeature.ENTITY_EDGE_FEATURE__DAMAGE_RATIO: MAX_RATIO_TOKEN,
    EntityEdgeFeature.ENTITY_EDGE_FEATURE__HEAL_RATIO: MAX_RATIO_TOKEN,
    EntityEdgeFeature.ENTITY_EDGE_FEATURE__STATUS_TOKEN: NUM_STATUS,
    # 1 on a damage event, the parsed count on |-hitcount|.
    EntityEdgeFeature.ENTITY_EDGE_FEATURE__HIT_COUNT: 6,
}


FIELD_MAX_VALUES = {
    FieldFeature.FIELD_FEATURE__WEATHER_ID: NUM_WEATHER,
    FieldFeature.FIELD_FEATURE__WEATHER_MAX_DURATION: 9,
    FieldFeature.FIELD_FEATURE__WEATHER_MIN_DURATION: 9,
    FieldFeature.FIELD_FEATURE__TERRAIN_ID: NUM_TERRAIN,
    FieldFeature.FIELD_FEATURE__TERRAIN_MAX_DURATION: 9,
    FieldFeature.FIELD_FEATURE__TERRAIN_MIN_DURATION: 9,
    FieldFeature.FIELD_FEATURE__PSEUDOWEATHER_ID: NUM_PSEUDOWEATHER,
    FieldFeature.FIELD_FEATURE__PSEUDOWEATHER_MAX_DURATION: 9,
    FieldFeature.FIELD_FEATURE__PSEUDOWEATHER_MIN_DURATION: 9,
    FieldFeature.FIELD_FEATURE__MY_SPIKES: 4,
    FieldFeature.FIELD_FEATURE__OPP_SPIKES: 4,
    FieldFeature.FIELD_FEATURE__MY_TOXIC_SPIKES: 2,
    FieldFeature.FIELD_FEATURE__OPP_TOXIC_SPIKES: 2,
    # Within-turn edge sequence index (Edge.addEdge increments it, the turn
    # boundary resets it). The ONLY observable of relative speed: the model
    # sees a turn's edges through segment_sum, which destroys their order,
    # and getField() never writes this column so the current-state row
    # contributes a constant bucket. Cap clips (encode_one_hot), so it is a
    # free tunable.
    FieldFeature.FIELD_FEATURE__TURN_ORDER_VALUE: 16,
}

# The scalars the read's info token carries. REQUEST_TYPE is not
# derivable from the action mask alone (a forced switch and a move turn on
# which every move is disabled mask alike), and NUM_ACTIVE is the doubles
# handle; neither is a FieldFeature, so neither can ride _embed_field, which
# is shared with history rows that have no info array.
INFO_MAX_VALUES = {
    InfoFeature.INFO_FEATURE__REQUEST_TYPE: len(RequestType.keys()) - 1,
    InfoFeature.INFO_FEATURE__NUM_ACTIVE: 2,
}

ACTION_MAX_VALUES = {
    MovesetFeature.MOVESET_FEATURE__ACTION_TYPE: NUM_ACTION_TYPES,
    MovesetFeature.MOVESET_FEATURE__HAS_PP: NUM_HAS_PP,
    MovesetFeature.MOVESET_FEATURE__PP: 64,
    MovesetFeature.MOVESET_FEATURE__MAXPP: 64,
    MovesetFeature.MOVESET_FEATURE__DISABLED: 2,
    MovesetFeature.MOVESET_FEATURE__IS_WILDCARD: 2,
}

with open("data/data/data.json", "r") as f:
    token_data = json.load(f)


PACKED_SET_MAX_VALUES = {
    PackedSetFeature.PACKED_SET_FEATURE__GENDER: NUM_GENDERS,
    PackedSetFeature.PACKED_SET_FEATURE__NATURE: NUM_NATURES,
    PackedSetFeature.PACKED_SET_FEATURE__HIDDENPOWERTYPE: NUM_TYPECHART,
    PackedSetFeature.PACKED_SET_FEATURE__TERATYPE: NUM_TYPECHART,
}

ITOS = {key.lower(): {v: k for k, v in token_data[key].items()} for key in token_data}
STOI = {key.lower(): {k: v for k, v in token_data[key].items()} for key in token_data}


def toid(string: str) -> str:
    return "".join(c for c in string if c.isalnum() or c == "_").lower()


VALID_GENERATIONS = [1, 9]


NUM_PACKED_SET_FEATURES = len(PackedSetFeature.keys())


ONEHOT_DTYPE = jnp.bfloat16


CAT_VF_SUPPORT = np.array([-1, 0, 1], dtype=np.float32)


def add_pretrained_embedding(generation):
    tables = {}
    for name in ["species", "abilities", "items", "moves", "learnset"]:
        try:
            tables[name] = PretrainedEmbedding(
                fpath=f"data/data/gen{generation}/{name}.npy",
                dtype=ONEHOT_DTYPE,
            )
        except Exception:
            traceback.print_exc()
            tables[name] = ZeroEmbedding(dtype=ONEHOT_DTYPE)
    return tables


ONEHOT_ENCODERS = {
    generation: add_pretrained_embedding(generation) for generation in VALID_GENERATIONS
}

_EX_BIN_PATH = os.path.join(os.path.dirname(__file__), "ex.bin")


def _read_ex_buffer(timeout_s: float = 180.0) -> bytes:
    """ex.bin (the example-batch shape fixture) is produced by the game
    service — only when missing, on service start, or explicitly via
    `npm run generate-ex` (service/src/tests/ex.ts). On a fresh checkout
    start.sh launches the service and this process concurrently, so wait
    for the file instead of losing that race and dying at import time.
    The service writes it atomically (write-then-rename), so existence
    implies the content is complete."""
    deadline = time.monotonic() + timeout_s
    announced = False
    while not os.path.exists(_EX_BIN_PATH):
        if time.monotonic() > deadline:
            raise FileNotFoundError(
                f"{_EX_BIN_PATH} still missing after {timeout_s:.0f}s — it is "
                "generated by the game service on first start. Start the "
                "service, or run `npm run generate-ex` in service/."
            )
        if not announced:
            print(
                f"Waiting for {_EX_BIN_PATH} — generated by the game "
                "service on its first start..."
            )
            announced = True
        time.sleep(2.0)
    with open(_EX_BIN_PATH, "rb") as f:
        return f.read()


EX_BUFFER = _read_ex_buffer()


EX_BATCH = EnvironmentBatch.FromString(EX_BUFFER)


MOVE_INDICES = np.array(
    [
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_1,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_2,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_3,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_4,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_1_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_2_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_3_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_4_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_1,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_2,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_3,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_4,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_1_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_2_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_3_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_4_WILDCARD,
    ]
)
WILDCARD_MOVE_INDICES = np.array(
    [
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_1_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_2_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_3_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_1_MOVE_4_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_1_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_2_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_3_WILDCARD,
        ActionEnum.ACTION_ENUM__ALLY_2_MOVE_4_WILDCARD,
    ]
)
RESERVE_ENTITY_INDICES = np.array(
    [
        ActionEnum.ACTION_ENUM__RESERVE_1_SWITCH_IN,
        ActionEnum.ACTION_ENUM__RESERVE_2_SWITCH_IN,
        ActionEnum.ACTION_ENUM__RESERVE_3_SWITCH_IN,
        ActionEnum.ACTION_ENUM__RESERVE_4_SWITCH_IN,
        ActionEnum.ACTION_ENUM__RESERVE_5_SWITCH_IN,
        ActionEnum.ACTION_ENUM__RESERVE_6_SWITCH_IN,
    ]
)
ALLY_SWITCH_INDICES = np.array(
    [
        ActionEnum.ACTION_ENUM__ALLY_1_SWITCH,
        ActionEnum.ACTION_ENUM__ALLY_2_SWITCH,
    ]
)
ALLY_TARGET_INDICES = np.array(
    [
        ActionEnum.ACTION_ENUM__ALLY_1_TARGET,
        ActionEnum.ACTION_ENUM__ALLY_2_TARGET,
    ]
)
ENEMY_TARGET_INDICES = np.array(
    [
        ActionEnum.ACTION_ENUM__ENEMY_1_TARGET,
        ActionEnum.ACTION_ENUM__ENEMY_2_TARGET,
    ]
)


for indices in [
    MOVE_INDICES,
    RESERVE_ENTITY_INDICES,
    ALLY_SWITCH_INDICES,
    ALLY_TARGET_INDICES,
    ENEMY_TARGET_INDICES,
]:
    assert len(indices) == len(set(indices)), "Duplicate indices found"
    indices.sort()


# ---- the block action space (2026-08-31) -----------------------------------
# The policy's action space is ActionMask's own fields flattened in field
# order -- see proto/service.proto `Action`. Offsets derive from the three
# slot-list lengths, which are the shared contract with the service
# (service/src/server/data.ts derives the same three and both suites assert
# the 295 total). The 41x41 (src, tgt) grid this replaced kept ~82% dead
# cells purely so the readout's scatter had somewhere to land.
NUM_SWITCH_CELLS = len(RESERVE_ENTITY_INDICES)
NUM_MOVE_SLOTS = len(MOVE_INDICES)
MOVE_SLOT_INDICES = MOVE_INDICES
TARGET_SLOT_INDICES = np.setdiff1d(
    np.arange(NUM_ACTION_FEATURES),
    np.concatenate([MOVE_INDICES, RESERVE_ENTITY_INDICES, ALLY_SWITCH_INDICES]),
)
NUM_TARGET_SLOTS = len(TARGET_SLOT_INDICES)
MOVE_CELL_OFFSET = NUM_SWITCH_CELLS
OTHER_CELL_OFFSET = MOVE_CELL_OFFSET + NUM_MOVE_SLOTS * NUM_TARGET_SLOTS
NUM_ACTION_CELLS = OTHER_CELL_OFFSET + NUM_TARGET_SLOTS
assert NUM_ACTION_CELLS == 295, "Block layout drifted from the proto contract"


def calculate_cell_modality_mask():
    """Per-cell modality over the block space.

    The same per-cell values the grid form carried -- switch cells (battle
    switch AND team preview lead) are SWITCH, a move cell inherits its move
    slot's regular/wildcard split, standalone cells are OTHER -- so
    `player_entropy_macro` and every modality-marginal consumer keeps its
    meaning across the 2026-08-31 grid retirement.
    """
    switch_block = np.full(
        NUM_SWITCH_CELLS, ModalityEnum.MODALITY_ENUM__SWITCH, dtype=np.int32
    )
    is_wildcard_slot = np.isin(MOVE_INDICES, WILDCARD_MOVE_INDICES)
    move_slot_modality = np.where(
        is_wildcard_slot,
        ModalityEnum.MODALITY_ENUM__WILDCARD,
        ModalityEnum.MODALITY_ENUM__MOVE,
    ).astype(np.int32)
    move_block = np.repeat(move_slot_modality, NUM_TARGET_SLOTS)
    other_block = np.full(
        NUM_TARGET_SLOTS, ModalityEnum.MODALITY_ENUM__OTHER, dtype=np.int32
    )
    return np.concatenate([switch_block, move_block, other_block])


CELL_MODALITY_MASK = calculate_cell_modality_mask()
