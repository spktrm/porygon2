import pandas as pd

from typing import Callable, List, TypedDict

from embeddings.encoders import (
    binary_encode,
    multihot_encode,
    z_score_scale,
    onehot_encode,
    text_encoding,
)


class Protocol(TypedDict):
    feature: str
    feature_fn: Callable[[str], bool]
    func: Callable[[pd.Series], pd.DataFrame]


SPECIES_PROTOCOLS: List[Protocol] = [
    *[
        {
            "feature": stat_feature,
            "func": binary_encode,
        }
        for stat_feature in [
            "baseStats.hp",
            "baseStats.atk",
            "baseStats.def",
            "baseStats.spa",
            "baseStats.spd",
            "baseStats.spe",
            "bst",
        ]
    ],
    {
        "feature": "weightkg",
        "func": lambda x: binary_encode(10 * x),
    },
    *[
        {
            "feature": stat_feature,
            "func": z_score_scale,
        }
        for stat_feature in [
            "baseStats.hp",
            "baseStats.atk",
            "baseStats.def",
            "baseStats.spa",
            "baseStats.spd",
            "baseStats.spe",
            "bst",
            "weightkg",
            "genderRatio.M",
            "genderRatio.F",
        ]
    ],
    *[
        {"feature": stat_feature, "func": onehot_encode}
        for stat_feature in [
            "id",
            "nfe",
            "tier",
            "maxHP",
            "baseForme",
            "gender",
            "requireMove",
            "requiredAbility",
            "requiredItem",
        ]
    ],
    *[
        {"feature": stat_feature, "func": multihot_encode}
        for stat_feature in [
            "abilities",
            "types",
            "tags",
            "otherFormes",
        ]
    ],
    {"feature_fn": lambda x: x.startswith("damageTaken."), "func": lambda x: x},
]

MOVES_PROTOCOLS: List[Protocol] = [
    *[
        {"feature": stat_feature, "func": onehot_encode}
        for stat_feature in [
            "id",
            "category",
            "priority",
            "type",
            "target",
            "volatileStatus",
            "status",
            "breaksProtect",
            "weather",
            "stallingMove",
            "sleepUsable",
            "selfdestruct",
            "struggleRecoil",
            "smartTarget",
            "slotCondition",
            "stealsBoosts",
            "terrain",
            "forceSwitch",
            "hasCrashDamage",
            "hasSheerForce",
            "mindBlownRecoil",
            "onDamagePriority",
            "onTry",
            "recoil",
            "heal",
            "ohko",
        ]
    ],
    *[{"feature": stat_feature, "func": multihot_encode} for stat_feature in []],
    {"feature_fn": lambda x: x.startswith("flags."), "func": lambda x: x.fillna(0)},
    {"feature_fn": lambda x: x.startswith("condition."), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("boosts."), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("secondary."), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("self."), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("selfBoost."), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("ignore"), "func": onehot_encode},
    {"feature": "basePower", "func": z_score_scale},
    {
        "feature": "basePower",
        "func": binary_encode,
    },
    {
        "feature": "desc",
        "func": text_encoding,
    },
    {
        "feature": "accuracy",
        "func": lambda x: binary_encode(
            x.map(lambda v: 100 if isinstance(v, bool) else v)
        ),
    },
    {
        "feature": "pp",
        "func": lambda x: binary_encode(x.map(lambda v: int(v * 8 / 5))),
    },
    {
        "feature": "accuracy",
        "func": lambda x: x.map(lambda v: 1 if isinstance(v, bool) else 0),
    },
    {"feature_fn": lambda x: x.startswith("damageTaken."), "func": lambda x: x},
]

ITEMS_PROTOCOLS: List[Protocol] = [
    *[{"feature": stat_feature, "func": onehot_encode} for stat_feature in ["id"]],
    {"feature_fn": lambda x: x.startswith("fling."), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("on"), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("is"), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("naturalGift."), "func": onehot_encode},
    *[{"feature": stat_feature, "func": multihot_encode} for stat_feature in []],
    {
        "feature": "desc",
        "func": text_encoding,
    },
]

ABILITIES_PROTOCOLS: List[Protocol] = [
    *[
        {"feature": stat_feature, "func": onehot_encode}
        for stat_feature in [
            "id",
            "suppressWeather",
        ]
    ],
    {"feature_fn": lambda x: x.startswith("condition."), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("on"), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("is"), "func": onehot_encode},
    {
        "feature": "desc",
        "func": text_encoding,
    },
]
