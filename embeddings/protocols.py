from enum import Enum, auto
from typing import Callable, List, TypedDict

import pandas as pd

from embeddings.encoders import multihot_encode, onehot_encode, sqrt_onehot_encode


class FeatureType(Enum):
    CATEGORICAL = auto()
    SCALAR = auto()


class Protocol(TypedDict):
    feature: str
    feature_fn: Callable[[str], bool]
    feature_type: FeatureType
    func: Callable[[pd.Series], pd.DataFrame]


SPECIES_PROTOCOLS: List[Protocol] = [
    *[
        {
            "feature": stat_feature,
            "func": sqrt_onehot_encode,
            "feature_type": FeatureType.CATEGORICAL,
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
    # *[
    #     {
    #         "feature": stat_feature,
    #         "feature_type": FeatureType.SCALAR,
    #     }
    #     for stat_feature in [
    #         "genderRatio.M",
    #         "genderRatio.F",
    #     ]
    # ],
    {
        "feature": "weightkg",
        "func": sqrt_onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    *[
        {
            "feature": stat_feature,
            "func": onehot_encode,
            "feature_type": FeatureType.CATEGORICAL,
        }
        for stat_feature in [
            "id",
            "nfe",
            # "tier",
            "maxHP",
            "baseForme",
            "gender",
            "requireMove",
            "requiredAbility",
            "requiredItem",
        ]
    ],
    *[
        {
            "feature": stat_feature,
            "func": multihot_encode,
            "feature_type": FeatureType.CATEGORICAL,
        }
        for stat_feature in [
            "abilities",
            "types",
            # "tags",
            # "otherFormes",
        ]
    ],
    {
        "feature_fn": lambda x: x.startswith("effectiveness."),
        "feature_type": FeatureType.SCALAR,
    },
    {
        "feature_fn": lambda x: x.startswith("effectiveness."),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
]

MOVES_PROTOCOLS: List[Protocol] = [
    *[
        {
            "feature": stat_feature,
            "func": onehot_encode,
            "feature_type": FeatureType.CATEGORICAL,
        }
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
            "willCrit",
            "tracksTarget",
            "thawsTarget",
        ]
    ],
    *[
        {
            "feature": stat_feature,
            "func": multihot_encode,
            "feature_type": FeatureType.CATEGORICAL,
        }
        for stat_feature in []
    ],
    {
        "feature_fn": lambda x: x.startswith("flags."),
        "func": lambda x: x.fillna(0),
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature_fn": lambda x: x.startswith("condition."),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature_fn": lambda x: x.startswith("boosts."),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature_fn": lambda x: x.startswith("on"),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature_fn": lambda x: x.startswith("override"),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature_fn": lambda x: x.startswith("secondary."),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature_fn": lambda x: x.startswith("self."),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature_fn": lambda x: x.startswith("selfBoost."),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature_fn": lambda x: x.startswith("effectiveness."),
        "feature_type": FeatureType.SCALAR,
    },
    {
        "feature_fn": lambda x: x.startswith("ignore"),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature": "basePower",
        "func": sqrt_onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature": "accuracy",
        "func": lambda x: sqrt_onehot_encode(
            x.map(lambda v: 100 if isinstance(v, bool) else v)
        ),
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature": "accuracy",
        "func": lambda x: x.map(lambda v: (100 if isinstance(v, bool) else v) / 100),
        "feature_type": FeatureType.SCALAR,
    },
    {
        "feature": "accuracy",
        "func": lambda x: x.map(lambda v: 1 if isinstance(v, bool) else 0),
    },
    {
        "feature": "pp",
        "func": lambda x: sqrt_onehot_encode(x.map(lambda v: int(v * 8 / 5))),
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature": "pp",
        "func": lambda x: x.map(lambda v: int(v * 8 / 5) / 64),
        "feature_type": FeatureType.SCALAR,
    },
]

ITEMS_PROTOCOLS: List[Protocol] = [
    *[
        {
            "feature": stat_feature,
            "func": onehot_encode,
            "feature_type": FeatureType.CATEGORICAL,
        }
        for stat_feature in [
            "id",
            "affectsFainted",
            "itemUser",
        ]
    ],
    {
        "feature_fn": lambda x: x.startswith("condition."),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature_fn": lambda x: x.startswith("boosts."),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature_fn": lambda x: x.startswith("fling."),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature_fn": lambda x: x.startswith("on"),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature_fn": lambda x: x.startswith("is"),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature_fn": lambda x: x.startswith("naturalGift."),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    *[
        {
            "feature": stat_feature,
            "func": multihot_encode,
            "feature_type": FeatureType.CATEGORICAL,
        }
        for stat_feature in []
    ],
    # {
    #     "feature": "desc",
    #     "func": text_encoding,
    #     "feature_type": FeatureType.SCALAR,
    # },
]

ABILITIES_PROTOCOLS: List[Protocol] = [
    *[
        {
            "feature": stat_feature,
            "func": onehot_encode,
            "feature_type": FeatureType.CATEGORICAL,
        }
        for stat_feature in [
            "id",
            "suppressWeather",
            "affectsFainted",
            "rating",
            "sourceEffect",
            "supressWeather",
        ]
    ],
    {
        "feature_fn": lambda x: x.startswith("flags."),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature_fn": lambda x: x.startswith("condition."),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature_fn": lambda x: x.startswith("on"),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
    {
        "feature_fn": lambda x: x.startswith("is"),
        "func": onehot_encode,
        "feature_type": FeatureType.CATEGORICAL,
    },
]
