"""The inputs of the three state linears, and the column blocks of the
kernels that read them (2026-09-04).

`public_persistent_linear`, `public_transient_linear` and
`private_state_linear` each read ONE concatenated feature vector; the
encoder used to assemble those inline, so the layout existed only as the
order of a concat. It is written here once, as a list of NAMED parts, and
`state_kernel_blocks` derives each kernel's column blocks from the same
list -- the telemetry that reads a kernel's hp columns against its status
columns (the normaliser-gaming instrument of the delta dynamics head)
cannot drift from the encoder's layout because there is no second copy
of it.

Structure-only against the inline form: the same arrays concatenated in
the same order, so the encoder's tokens are bit-identical.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from constants import MAX_RATIO_TOKEN
from rl.environment.data import (
    NUM_ENTITY_PRIVATE_FEATURES,
    NUM_ENTITY_PUBLIC_FEATURES,
    NUM_ENTITY_REVEALED_FEATURES,
    NUM_MOVES,
    NUM_TYPECHART,
)
from rl.environment.protos.enums_pb2 import MovesEnum
from rl.environment.protos.features_pb2 import (
    EntityPrivateNodeFeature,
    EntityPublicNodeFeature,
    EntityRevealedNodeFeature,
)
from rl.model.features import (
    binary_scale_encoding,
    encode_divided_one_hot_public_entity,
    encode_one_hot_private_entity,
    encode_one_hot_public_entity,
    encode_reg_boosts,
    encode_spe_boosts,
    encode_sqrt_one_hot_public_entity,
)
from rl.model.modules import one_hot_concat_jax

PUBLIC_MOVE_INDICES = np.array(
    [
        EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID0,
        EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID1,
        EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID2,
        EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID3,
    ]
)
STATE_KERNELS = (
    "public_persistent_linear",
    "public_transient_linear",
    "private_state_linear",
)
# The coarse reading of the fine-grained part names below; a part named
# in no group is "other". hp is the scalar + 32-bin one-hot (and the
# public path's coarser 32-bin divided one-hot beside it); status is the
# condition family; boosts the two stage-multiplier encodings.
STATE_KERNEL_GROUPS = {
    "hp": ("hp", "hp_bins"),
    "status": ("status", "has_status", "sleep_turns", "toxic_turns", "fainted"),
    "boosts": ("reg_boosts", "spe_boosts"),
}


def _one_hot_code(parts, dtype):
    """`one_hot_concat_jax` over named (index, width) pairs, plus each
    part's column slice within the code."""
    code = one_hot_concat_jax([pair for _, pair in parts], dtype=dtype)
    widths = [pair[1] for _, pair in parts]
    offsets = np.cumsum([0] + widths)
    blocks = {
        name: slice(int(start), int(start + width))
        for (name, _), start, width in zip(parts, offsets[:-1], widths)
    }
    return code, blocks


def _named_concat(parts):
    """Concatenate named arrays on the last axis, returning the array and
    each part's slice. A part may itself be a (array, blocks) pair from
    `_one_hot_code`, whose inner blocks are re-based onto the whole."""
    arrays = []
    blocks = {}
    offset = 0
    for name, part in parts:
        inner = None
        if isinstance(part, tuple):
            part, inner = part
        arrays.append(part)
        width = int(part.shape[-1])
        if inner is None:
            blocks[name] = slice(offset, offset + width)
        else:
            for inner_name, inner_slice in inner.items():
                blocks[inner_name] = slice(
                    offset + inner_slice.start, offset + inner_slice.stop
                )
        offset += width
    return jnp.concatenate(arrays, axis=-1), blocks


def _hp_features(hp_ratio_token, dtype):
    """The hp scalar beside a 32-bin one-hot of it, the same encoding on
    the public and private paths."""
    hp_ratio = (hp_ratio_token / MAX_RATIO_TOKEN).astype(dtype)
    return jnp.concatenate(
        [
            hp_ratio[..., None],
            jax.nn.one_hot(jnp.floor(32 * hp_ratio), 32, dtype=dtype),
        ],
        axis=-1,
    ).reshape(-1)


def public_persistent_features(public: jax.Array, revealed: jax.Array, dtype):
    """Persistent condition: survives switching out, meaningful on the
    bench. Returns (features, blocks)."""
    persistent_code = _one_hot_code(
        [
            (
                "level",
                encode_sqrt_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__LEVEL,
                    dtype=dtype,
                ),
            ),
            (
                "hp_bins",
                encode_divided_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO,
                    MAX_RATIO_TOKEN / 32,
                ),
            ),
            (
                "gender",
                encode_one_hot_public_entity(
                    public, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__GENDER
                ),
            ),
            (
                "status",
                encode_one_hot_public_entity(
                    public, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__STATUS
                ),
            ),
            (
                "item_effect",
                encode_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ITEM_EFFECT,
                ),
            ),
            (
                "sleep_turns",
                encode_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SLEEP_TURNS,
                ),
            ),
            (
                "fainted",
                encode_one_hot_public_entity(
                    public, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED
                ),
            ),
        ],
        dtype=dtype,
    )
    hp_features = _hp_features(
        public[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO], dtype
    )

    move_pp_indices = np.array(
        [
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP0,
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP1,
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP2,
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP3,
        ]
    )
    move_tokens = revealed[PUBLIC_MOVE_INDICES]
    move_pp_tokens = public[move_pp_indices]
    is_valid_move = (move_tokens != MovesEnum.MOVES_ENUM___NULL) & (
        move_tokens != MovesEnum.MOVES_ENUM___UNSPECIFIED
    )
    move_pp_ratios = is_valid_move * (move_pp_tokens / 31).astype(dtype)
    move_pp_onehot = (
        jnp.zeros(NUM_MOVES, dtype=move_pp_ratios.dtype)
        .at[move_tokens]
        .set(move_pp_ratios)
        .clip(min=0, max=1)
    )
    teratype_token = revealed[
        EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__TERA_TYPE
    ]
    return _named_concat(
        [
            ("persistent_code", persistent_code),
            ("hp", hp_features),
            ("move_pp", move_pp_onehot),
            ("tera_type", jax.nn.one_hot(teratype_token, NUM_TYPECHART, dtype=dtype)),
        ]
    )


def public_transient_features(public: jax.Array, dtype):
    """The active-only overlay (volatiles, boosts, typechange,
    trapped / called-back / newly-switched, toxic counter): all of it
    resets on switch, so it is its own token, masked by the ACTIVE flag.
    Returns (features, blocks)."""
    encode_hex = jax.vmap(
        functools.partial(binary_scale_encoding, dtype=dtype, world_dim=65535)
    )
    volatiles_indices = public[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES0 : EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES8
        + 1
    ]
    typechange_indices = public[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE0 : EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE1
        + 1
    ]
    reg_boost_features = public[
        np.array(
            [
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_ATK_VALUE,
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_DEF_VALUE,
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_SPA_VALUE,
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_SPD_VALUE,
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_SPE_VALUE,
            ]
        )
    ]
    spe_boost_features = public[
        np.array(
            [
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_ACCURACY_VALUE,
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_EVASION_VALUE,
            ]
        )
    ]
    transient_code = _one_hot_code(
        [
            (
                "being_called_back",
                encode_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BEING_CALLED_BACK,
                ),
            ),
            (
                "trapped",
                encode_one_hot_public_entity(
                    public, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TRAPPED
                ),
            ),
            (
                "newly_switched",
                encode_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__NEWLY_SWITCHED,
                ),
            ),
            (
                "toxic_turns",
                encode_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TOXIC_TURNS,
                ),
            ),
        ],
        dtype=dtype,
    )
    return _named_concat(
        [
            ("transient_code", transient_code),
            ("volatiles", encode_hex(volatiles_indices).reshape(-1)),
            ("typechange", encode_hex(typechange_indices).reshape(-1)),
            ("reg_boosts", encode_reg_boosts(reg_boost_features).astype(dtype)),
            ("spe_boosts", encode_spe_boosts(spe_boost_features).astype(dtype)),
        ]
    )


def private_state_features(private: jax.Array, dtype, num_stat_bands: int = 8):
    """Tera type, the request-side condition block (the truth channel,
    2026-08-31: same encodings as the public path), hp and the stats'
    Fourier bands. Returns (features, blocks)."""
    boolean_code = _one_hot_code(
        [
            (
                "tera_type",
                encode_one_hot_private_entity(
                    private,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__TERA_TYPE,
                ),
            ),
        ],
        dtype=dtype,
    )
    stat_features = private[
        np.array(
            [
                EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__HP_STAT,
                EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ATK_STAT,
                EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__DEF_STAT,
                EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SPA_STAT,
                EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SPD_STAT,
                EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SPE_STAT,
            ]
        )
    ].astype(dtype)
    stat_encoding = stat_features.astype(jnp.float32) / np.array(
        [714, 526, 658, 535, 658, 548], dtype=np.float32
    )
    freqs = (2.0 ** np.arange(num_stat_bands) * np.pi).astype(np.float32)
    # Phases reach 2^7.pi ~ 400 rad, where bf16 spacing is ~1 rad: cast
    # before sin/cos and the top bands are quantisation noise. Bands in
    # f32, cast after.
    phase = stat_encoding[..., None] * freqs[None]
    stat_encoding = (
        jnp.concatenate((jnp.sin(phase), jnp.cos(phase)), axis=-1)
        .reshape(-1)
        .astype(dtype)
    )
    condition_code = _one_hot_code(
        [
            (
                "status",
                encode_one_hot_private_entity(
                    private,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__STATUS,
                ),
            ),
            (
                "has_status",
                encode_one_hot_private_entity(
                    private,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__HAS_STATUS,
                ),
            ),
            (
                "toxic_turns",
                encode_one_hot_private_entity(
                    private,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__TOXIC_TURNS,
                ),
            ),
            (
                "sleep_turns",
                encode_one_hot_private_entity(
                    private,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SLEEP_TURNS,
                ),
            ),
            (
                "fainted",
                encode_one_hot_private_entity(
                    private,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__FAINTED,
                ),
            ),
            # Turn-delta staleness of the source request (identically 0
            # on the own channel; >0 on the opponent channel when their
            # client lags the observer's build).
            (
                "request_lag",
                encode_one_hot_private_entity(
                    private,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__REQUEST_LAG,
                ),
            ),
        ],
        dtype=dtype,
    )
    hp_features = _hp_features(
        private[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__HP_RATIO], dtype
    )
    return _named_concat(
        [
            ("boolean_code", boolean_code),
            ("condition_code", condition_code),
            ("hp", hp_features),
            ("stats", stat_encoding),
        ]
    )


@functools.cache
def state_kernel_blocks() -> dict[str, dict[str, list[slice]]]:
    """Per state kernel, the input-column slices of each STATE_KERNEL_GROUPS
    group plus "other" (every part in no group). Derived by running the
    feature builders above on one zero row, so it IS the encoder's layout."""
    public = jnp.zeros((NUM_ENTITY_PUBLIC_FEATURES,), jnp.int32)
    revealed = jnp.zeros((NUM_ENTITY_REVEALED_FEATURES,), jnp.int32)
    private = jnp.zeros((NUM_ENTITY_PRIVATE_FEATURES,), jnp.int32)
    layouts = {
        "public_persistent_linear": public_persistent_features(
            public, revealed, jnp.float32
        ),
        "public_transient_linear": public_transient_features(public, jnp.float32),
        "private_state_linear": private_state_features(private, jnp.float32),
    }
    grouped = {}
    for kernel, (features, blocks) in layouts.items():
        assert sum(sl.stop - sl.start for sl in blocks.values()) == features.shape[-1]
        named = {name for parts in STATE_KERNEL_GROUPS.values() for name in parts}
        by_group = {group: [] for group in STATE_KERNEL_GROUPS}
        by_group["other"] = []
        for part, block in blocks.items():
            if part in named:
                for group, parts in STATE_KERNEL_GROUPS.items():
                    if part in parts:
                        by_group[group].append(block)
            else:
                by_group["other"].append(block)
        grouped[kernel] = by_group
    return grouped


def hp_input_rows(kernel: str) -> np.ndarray:
    """The input-row indices of a state kernel's hp block(s), for the delta
    dynamics head's hp-subspace instrument."""
    blocks = state_kernel_blocks()[kernel]["hp"]
    rows = [np.arange(block.start, block.stop) for block in blocks]
    return np.concatenate(rows + [np.zeros((0,), np.int64)])
