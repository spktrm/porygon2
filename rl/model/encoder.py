import functools
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from constants import MAX_RATIO_TOKEN
from rl.environment.data import (
    MOVE_CELL_OFFSET,
    MOVE_INDICES,
    NUM_FROM_SOURCE_EFFECTS,
    NUM_TYPECHART,
    ONEHOT_ENCODERS,
    OTHER_CELL_OFFSET,
    TARGET_SLOT_INDICES,
)
from rl.environment.interfaces import (
    EventStates,
    HistoryCarry,
    PlayerEnvOutput,
    PlayerHistoryOutput,
    PlayerPackedHistoryOutput,
)
from rl.environment.protos.enums_pb2 import (
    AbilitiesEnum,
    BattlemajorargsEnum,
    EffectEnum,
    ItemsEnum,
    MovesEnum,
    SpeciesEnum,
)
from rl.environment.protos.features_pb2 import (
    EntityEdgeFeature,
    EntityPublicNodeFeature,
    EntityRevealedNodeFeature,
    FieldFeature,
    InfoFeature,
    MovesetFeature,
    RequestType,
)
from rl.model.constants import (
    ALLY_TARGET_ROWS,
    IS_WILDCARD_MOVE_SLOT,
    MY_ACTIVE_PUBLIC_ROWS,
    NUM_FIELD_ROWS,
    NUM_HISTORY_REGISTERS,
    NUM_PUBLIC_SLOTS,
    NUM_SEQUENCE_GROUPS,
    NUM_SEQUENCE_ROWS,
    NUM_TOKEN_TYPES,
    NUM_TRUNK_REGISTERS_PER_TIER,
    OPP_ACTIVE_PUBLIC_ROWS,
    POLICY_READABLE_ROWS,
    PRIVATE_TOKEN_TYPES,
    PUBLIC_SEQUENCE_ROWS,
    PUBLIC_TOKEN_TYPES,
    SEQUENCE_GROUP_IDS,
    SEQUENCE_LAYOUT,
    SEQUENCE_READ_MASK,
    SequenceGroup,
    TokenType,
)
from rl.model.features import (
    binary_scale_encoding,
    encode_divided_one_hot_edge,
    encode_one_hot_action,
    encode_one_hot_edge,
    encode_one_hot_field,
    encode_one_hot_info,
    encode_reg_boosts,
    encode_spe_boosts,
    encode_sqrt_one_hot_action,
    get_private_entity_mask,
    get_public_entity_mask,
)
from rl.model.heads import chosen_bank_rows
from rl.model.history_encoder import (
    STEP_KEY_MASK,
    PerSlotHistoryEncoder,
    history_carry_from,
    history_step_stats,
)
from rl.model.identity import (
    BENCH_POSITION,
    FIRST_ACTIVE_POSITION,
    SECOND_ACTIVE_POSITION,
    SIDE_MINE,
    SIDE_OPPONENT,
    active_slot_identities,
    active_slot_index,
    field_identities,
    public_identities,
    sequence_identities,
)
from rl.model.modules import (
    COLLECT_INTERMEDIATES,
    EntitySumPool,
    SequenceNormalisation,
    SumEmbeddings,
    one_hot_concat_jax,
)
from rl.model.state_features import (
    PUBLIC_MOVE_INDICES,
    private_state_features,
    public_persistent_features,
    public_transient_features,
)
from rl.model.trunk import Trunk, group_row_cosine, group_row_l2

# Typed action-slot groups (canonical partition lives in
# rl/environment/data.py next to the modality masks): move slots are
# move-feature-derived, switch slots entity-derived, target/structural
# slots key-only.


def _forward_vmap():
    """vmap of `Encoder._batched_forward` over the leading (time) axis.

    Normally a plain `jax.vmap` over the bound method. Under
    COLLECT_INTERMEDIATES the attention modules `sow` into the
    "intermediates" collection, and a sow inside a RAW jax.vmap escapes
    the transform's functional boundary (UnexpectedTracerError on a
    BatchTracer) — so in that mode only, use `nn.vmap`, which lifts the
    collection properly (params broadcast, intermediates mapped), exactly
    as the round trunk's nn.scan already does. Training never takes this
    branch, so its HLO is unchanged.
    """
    if not COLLECT_INTERMEDIATES:
        return lambda self, *a: jax.vmap(self._batched_forward)(*a)
    return nn.vmap(
        Encoder._batched_forward,
        variable_axes={"params": None, "intermediates": 0},
        split_rngs={"params": False},
        in_axes=0,
        out_axes=0,
    )


def _lifted_entity_vmap(method):
    """Lifted (flax.linen) replacement for the previous plain
    `jax.vmap(self._embed_*)` call-site pattern: `nn.vmap` maps the data
    axis while broadcasting params (`variable_axes={"params": None}` —
    the embedders only APPLY setup-defined submodules, never create
    variables), and the surrounding lifted `nn.jit` makes each embedder
    its own XLA subcomputation instead of being inlined wholesale into
    the caller's graph — smaller HLO and cheaper compiles (retained
    executables cost host RAM), plus trace reuse whenever two
    call sites agree on shapes. Composing lifted transforms (rather than
    plain jax ones) is what keeps this legal to nest under flax's other
    lifted transforms (nn.scan/nn.checkpoint) elsewhere in the model."""
    # NOT lifting "intermediates" here is deliberate: one pool instance is
    # applied at several sites with different entity counts, and mapping
    # the collection would demand one batch size across them. The pools'
    # attention sows are therefore dropped; scripts/attn_probe.py reads
    # the trunk's (rl/model/trunk.py lifts them).
    return nn.jit(
        nn.vmap(
            method,
            in_axes=0,
            out_axes=0,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )
    )


def active_slot_rows(
    slot_valid: jax.Array, sides: jax.Array, positions: jax.Array, side: int
) -> tuple[jax.Array, jax.Array]:
    """(..., 2) the slot holding each active position of `side` (first,
    second) and (..., 2) whether one exists -- the rows the target slots
    add, read off the slots' own side/position features."""
    found = []
    rows = []
    for position in (FIRST_ACTIVE_POSITION, SECOND_ACTIVE_POSITION):
        match = slot_valid & (sides == side) & (positions == position)
        found.append(match.any(axis=-1))
        rows.append(match.argmax(axis=-1))
    return jnp.stack(rows, axis=-1), jnp.stack(found, axis=-1)


class PublicRowInputs(NamedTuple):
    """Everything the public tier's rows are assembled from, AFTER the
    entity/field embedders: the request path fills it from the env step
    and the history pathway, the per-event path from the history scan's
    own per-step products. One assembly, two sources."""

    public_rows: jax.Array  # (12, D)
    public_valid: jax.Array  # (12,)
    public_sides: jax.Array  # (12,)
    public_positions: jax.Array  # (12,)
    field_rows: jax.Array  # (3, D)
    history_entity_rows: jax.Array  # (12, D)
    history_row_valid: jax.Array  # (12,)
    history_field_rows: jax.Array  # (3, D)
    history_register_rows: jax.Array  # (4, D)
    # Per active slot, mine then theirs, first position then second.
    active_state_rows: jax.Array  # (4, D)
    active_state_valid: jax.Array  # (4,)
    history_active_rows: jax.Array  # (4, D)
    info: jax.Array  # the request's info vector (REQUEST_TYPE, NUM_ACTIVE read)
    target_slot_valid: jax.Array  # (17,)


class Encoder(nn.Module):
    cfg: ConfigDict

    def setup(self):
        entity_size = self.cfg.entity_size
        self.entity_size = entity_size

        embed_kwargs = dict(features=entity_size, dtype=self.cfg.dtype)
        dense_kwargs = dict(features=entity_size, dtype=self.cfg.dtype)

        self.effect_from_source_embedding = nn.Embed(
            num_embeddings=NUM_FROM_SOURCE_EFFECTS,
            name="effect_from_source_embedding",
            **embed_kwargs,
        )

        embedding_init = nn.initializers.variance_scaling(
            1.0, "fan_in", "normal", out_axis=0
        )
        bias_init = nn.initializers.zeros_init()

        self.side_bias = nn.Embed(2, name="side_bias", **embed_kwargs)
        self.pos_bias = nn.Embed(3, name="position_bias", **embed_kwargs)

        # One learned identity per target slot. Pass, the structural slots
        # and the four active-slot targets are all ways of saying "a
        # thing a move can be aimed at", and the readout wants them as one
        # contiguous block it can score against.
        self.target_slot_embeddings = self.param(
            "target_slot_embeddings",
            embedding_init,
            (len(TARGET_SLOT_INDICES), entity_size),
        )
        self.value_cls_embedding = self.param(
            "value_cls_embedding", embedding_init, (1, entity_size)
        )
        self.public_cls_embedding = self.param(
            "public_cls_embedding", embedding_init, (1, entity_size)
        )
        register_shape = (NUM_TRUNK_REGISTERS_PER_TIER, entity_size)
        self.public_register_embeddings = self.param(
            "public_register_embeddings", embedding_init, register_shape
        )
        self.private_register_embeddings = self.param(
            "private_register_embeddings", embedding_init, register_shape
        )
        self.privileged_register_embeddings = self.param(
            "privileged_register_embeddings", embedding_init, register_shape
        )
        self.prev_action_src_bias = self.param(
            "prev_action_src_bias", embedding_init, (1, entity_size)
        )
        self.prev_action_tgt_bias = self.param(
            "prev_action_tgt_bias", embedding_init, (1, entity_size)
        )

        self.regular_move_bias = self.param(
            "regular_move_bias", bias_init, (1, entity_size)
        )
        self.wildcard_move_bias = self.param(
            "wildcard_move_bias", bias_init, (1, entity_size)
        )
        # The CLS row. The value head reads THIS ROW AND ONLY THIS ROW, so
        # loss_v_win's gradient reaches the trunk through it and it is the row
        # that has to aggregate the board. It is also unconditionally valid,
        # which is what stops a terminal step -- every action row masked off --
        # from attending over an empty key set and returning NaN.
        self.cls_embedding = self.param(
            "cls_embedding", embedding_init, (1, entity_size)
        )
        self.sequence_group_bias = self.param(
            "sequence_group_bias", embedding_init, (NUM_SEQUENCE_GROUPS, entity_size)
        )

        self.species_linear = nn.Dense(
            name="species_linear", use_bias=False, **dense_kwargs
        )
        self.items_linear = nn.Dense(
            name="items_linear", use_bias=False, **dense_kwargs
        )
        self.abilities_linear = nn.Dense(
            name="abilities_linear", use_bias=False, **dense_kwargs
        )
        self.moves_linear = nn.Dense(
            name="moves_linear", use_bias=False, **dense_kwargs
        )
        self.learnset_linear = nn.Dense(
            name="learnset_linear", use_bias=False, **dense_kwargs
        )

        # Entity pool, shared between private and public entities: a masked
        # sum of the attribute tokens plus the token-type bias -- an
        # intra-entity attention block here would erode type legibility
        # (`EntitySumPool`). Token provenance is carried by the
        # token-type bias table; per-provenance input norms downstream keep
        # the two entity kinds separable.
        self.entity_pool = EntitySumPool(
            num_token_types=NUM_TOKEN_TYPES, features=entity_size, name="entity_pool"
        )
        self.public_persistent_linear = nn.Dense(
            name="public_persistent_linear", use_bias=False, **dense_kwargs
        )
        self.public_transient_linear = nn.Dense(
            name="public_transient_linear", use_bias=False, **dense_kwargs
        )
        self.private_state_linear = nn.Dense(
            name="private_state_linear", use_bias=False, **dense_kwargs
        )

        self.action_sum = SumEmbeddings(
            output_size=entity_size,
            dtype=self.cfg.dtype,
            names=("move", "flags"),
            name="action_sum",
        )
        self.entity_edge_sum = SumEmbeddings(
            output_size=entity_size,
            dtype=self.cfg.dtype,
            names=(
                "minor_args",
                "flags",
                "stats",
                "ability",
                "item",
                "move",
                "effect_source",
                "from_type",
            ),
            name="entity_edge_sum",
        )
        self.field_linear = nn.Dense(
            name="field_linear", use_bias=False, **dense_kwargs
        )
        self.side_condition_linear = nn.Dense(
            name="side_condition_linear", use_bias=False, **dense_kwargs
        )
        # "What am I being asked": request type and active count. Both are
        # InfoFeatures, so neither can ride _embed_field (shared with history
        # rows, which carry no info array). REQUEST_TYPE is not derivable
        # from the action mask alone — a forced switch and a move turn whose
        # every move is disabled mask alike.
        self.info_linear = nn.Dense(name="info_linear", use_bias=False, **dense_kwargs)

        self.history_encoder = PerSlotHistoryEncoder(self.cfg, name="history_encoder")

        # The trunk. One sequence, `num_blocks` standard pre-RMSNorm blocks,
        # no gates and no block masks -- see rl/model/trunk.py.
        # Every row's CONTENT enters the trunk at RMS 1, a fresh embedding
        # table's magnitude, normalised per row and rescaled per group, with
        # the additive group/row identity on top -- the registers the trunk
        # appends pass through the same module, so nothing enters at a
        # magnitude of its own.
        self.trunk = Trunk(self.cfg.trunk, name="trunk")
        self.input_normalisation = SequenceNormalisation(
            num_groups=NUM_SEQUENCE_GROUPS, name="input_normalisation"
        )
        # The same module on the way OUT (2026-09-11): every row leaves the
        # trunk at RMS 1, rescaled per group, so the heads read every row
        # at one magnitude rather than at whatever the blocks' writes left
        # it.
        self.output_normalisation = SequenceNormalisation(
            num_groups=NUM_SEQUENCE_GROUPS, name="output_normalisation"
        )

    def _embed_species(self, token: jax.Array):
        mask = ~(
            (token == SpeciesEnum.SPECIES_ENUM___UNSPECIFIED)
            | (token == SpeciesEnum.SPECIES_ENUM___PAD)
            | (token == SpeciesEnum.SPECIES_ENUM___NULL)
        )
        _ohe_encoder = ONEHOT_ENCODERS[self.cfg.generation]["species"]
        return mask * self.species_linear(_ohe_encoder(token))

    def _embed_learnset(self, token: jax.Array):
        mask = ~(
            (token == SpeciesEnum.SPECIES_ENUM___UNSPECIFIED)
            | (token == SpeciesEnum.SPECIES_ENUM___PAD)
            | (token == SpeciesEnum.SPECIES_ENUM___NULL)
        )
        _ohe_encoder = ONEHOT_ENCODERS[self.cfg.generation]["learnset"]
        return mask * self.learnset_linear(_ohe_encoder(token))

    def _embed_item(self, token: jax.Array):
        mask = ~(
            (token == ItemsEnum.ITEMS_ENUM___UNSPECIFIED)
            | (token == ItemsEnum.ITEMS_ENUM___PAD)
            | (token == ItemsEnum.ITEMS_ENUM___NULL)
        )
        _ohe_encoder = ONEHOT_ENCODERS[self.cfg.generation]["items"]
        return mask * self.items_linear(_ohe_encoder(token))

    def _embed_ability(self, token: jax.Array):
        mask = ~(
            (token == AbilitiesEnum.ABILITIES_ENUM___UNSPECIFIED)
            | (token == AbilitiesEnum.ABILITIES_ENUM___PAD)
            | (token == AbilitiesEnum.ABILITIES_ENUM___NULL)
        )
        _ohe_encoder = ONEHOT_ENCODERS[self.cfg.generation]["abilities"]
        return mask * self.abilities_linear(_ohe_encoder(token))

    def _embed_move(self, token: jax.Array):
        mask = ~(
            (token == MovesEnum.MOVES_ENUM___UNSPECIFIED)
            | (token == MovesEnum.MOVES_ENUM___PAD)
            | (token == MovesEnum.MOVES_ENUM___NULL)
        )
        _ohe_encoder = ONEHOT_ENCODERS[self.cfg.generation]["moves"]
        return mask * self.moves_linear(_ohe_encoder(token))

    def _public_entity_tokens(self, public: jax.Array, revealed: jax.Array):
        """Public attribute content shared by current entities and history."""
        # The three state linears' inputs (and their column layout) live in
        # `rl.model.state_features`, once, beside the telemetry that reads
        # the kernels by block.
        persistent_features, _ = public_persistent_features(
            public, revealed, self.cfg.dtype
        )

        move_tokens = revealed[PUBLIC_MOVE_INDICES]
        is_valid_move = (move_tokens != MovesEnum.MOVES_ENUM___NULL) & (
            move_tokens != MovesEnum.MOVES_ENUM___UNSPECIFIED
        )
        move_embeddings = jax.vmap(self._embed_move)(move_tokens)

        species_token = revealed[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
        ]
        ability_token = revealed[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ABILITY
        ]
        item_token = revealed[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ITEM
        ]

        tokens = jnp.concatenate(
            (
                jnp.stack(
                    (
                        self._embed_species(species_token),
                        self._embed_ability(ability_token),
                        self._embed_item(item_token),
                    )
                ),
                move_embeddings,
                jnp.stack(
                    (
                        self._embed_learnset(species_token),
                        self.public_persistent_linear(persistent_features),
                    )
                ),
            ),
            axis=0,
        )

        mask = get_public_entity_mask(revealed)
        ability_valid = ~(
            (ability_token == AbilitiesEnum.ABILITIES_ENUM___UNSPECIFIED)
            | (ability_token == AbilitiesEnum.ABILITIES_ENUM___PAD)
            | (ability_token == AbilitiesEnum.ABILITIES_ENUM___NULL)
        )
        item_valid = ~(
            (item_token == ItemsEnum.ITEMS_ENUM___UNSPECIFIED)
            | (item_token == ItemsEnum.ITEMS_ENUM___PAD)
            | (item_token == ItemsEnum.ITEMS_ENUM___NULL)
        )
        token_mask = mask & jnp.concatenate(
            (
                jnp.stack((jnp.ones_like(mask), ability_valid, item_valid)),
                is_valid_move & (move_tokens != MovesEnum.MOVES_ENUM___PAD),
                jnp.stack((jnp.ones_like(mask), jnp.ones_like(mask))),
            ),
            axis=0,
        )

        return tokens, token_mask, mask

    def _embed_public_entity(self, public: jax.Array, revealed: jax.Array):
        """Pool each snapshot independently: history cache rows span turns."""
        tokens, token_mask, mask = self._public_entity_tokens(public, revealed)
        revealed_embedding = self.entity_pool(tokens, token_mask, PUBLIC_TOKEN_TYPES)
        return revealed_embedding, mask

    def _embed_active_state(self, public: jax.Array):
        """What a switch clears (volatiles, boosts, type change, trapped, ...)
        as a row of its own: the same token the entity pool used to sum, with
        the same field identity, and whether the entity holds an active slot.
        One embedder for the current board and the packed history cache."""
        transient_features, _ = public_transient_features(public, self.cfg.dtype)
        is_active = (
            public[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE]
            != BENCH_POSITION
        )
        token = self.entity_pool.typed(
            self.public_transient_linear(transient_features), TokenType.ACTIVE_STATE
        )
        return token, is_active

    def _private_entity_tokens(self, private: jax.Array, num_stat_bands: int = 8):
        """The attribute-token half of a private entity -- see
        `_public_entity_tokens`. NOTE the index constants below are the
        REVEALED enum applied to a PRIVATE row: legal only because
        SPECIES/ITEM/ABILITY/MOVEID0-3 are 1..7 in both enums."""
        move_tokens = private[PUBLIC_MOVE_INDICES]

        move_embeddings = jax.vmap(self._embed_move)(move_tokens)

        species_token = private[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
        ]
        ability_token = private[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ABILITY
        ]
        item_token = private[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ITEM
        ]

        # Tera type, the request-side condition block (the truth channel,
        # 2026-08-31), hp and the stats' Fourier bands -- built in
        # `rl.model.state_features` beside the public path's.
        state_features, _ = private_state_features(
            private, self.cfg.dtype, num_stat_bands
        )

        tokens = jnp.concatenate(
            (
                jnp.stack(
                    (
                        self._embed_species(species_token),
                        self._embed_ability(ability_token),
                        self._embed_item(item_token),
                    )
                ),
                move_embeddings,
                self.private_state_linear(state_features)[None],
            ),
            axis=0,
        )

        mask = get_private_entity_mask(private)
        ability_valid = ~(
            (ability_token == AbilitiesEnum.ABILITIES_ENUM___UNSPECIFIED)
            | (ability_token == AbilitiesEnum.ABILITIES_ENUM___PAD)
            | (ability_token == AbilitiesEnum.ABILITIES_ENUM___NULL)
        )
        item_valid = ~(
            (item_token == ItemsEnum.ITEMS_ENUM___UNSPECIFIED)
            | (item_token == ItemsEnum.ITEMS_ENUM___PAD)
            | (item_token == ItemsEnum.ITEMS_ENUM___NULL)
        )
        move_valid = ~(
            (move_tokens == MovesEnum.MOVES_ENUM___UNSPECIFIED)
            | (move_tokens == MovesEnum.MOVES_ENUM___PAD)
            | (move_tokens == MovesEnum.MOVES_ENUM___NULL)
        )
        token_mask = mask & jnp.concatenate(
            (
                jnp.stack((jnp.ones_like(mask), ability_valid, item_valid)),
                move_valid,
                jnp.ones_like(mask)[None],
            ),
            axis=0,
        )

        return tokens, token_mask, mask

    def _embed_private_entity(self, private: jax.Array, num_stat_bands: int = 8):
        """Entity-LOCAL pooling -- see `_embed_public_entity`. The path for
        the private sheet rows."""
        tokens, token_mask, mask = self._private_entity_tokens(private, num_stat_bands)
        private_embedding = self.entity_pool(tokens, token_mask, PRIVATE_TOKEN_TYPES)
        return private_embedding, mask

    def _embed_edge(self, edge: jax.Array):
        encode_hex = jax.vmap(
            functools.partial(
                binary_scale_encoding, world_dim=65535, dtype=self.cfg.dtype
            )
        )

        minor_args_indices = edge[
            EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG0 : EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG3
            + 1
        ]
        minor_args_encoding = encode_hex(minor_args_indices).reshape(-1)

        boolean_code = one_hot_concat_jax(
            [
                encode_one_hot_edge(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG,
                ),
                encode_divided_one_hot_edge(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__DAMAGE_RATIO,
                    MAX_RATIO_TOKEN / 32,
                ),
                encode_divided_one_hot_edge(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__HEAL_RATIO,
                    MAX_RATIO_TOKEN / 32,
                ),
                encode_one_hot_edge(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__STATUS_TOKEN,
                ),
                encode_one_hot_edge(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__HIT_COUNT,
                ),
            ],
            dtype=self.cfg.dtype,
        )

        effect_from_source_indices = np.array(
            [
                EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN0,
                EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN1,
                EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN2,
                EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN3,
                EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN4,
            ]
        )
        effect_from_source_tokens = edge[effect_from_source_indices]
        effect_from_source_mask = ~(
            (effect_from_source_tokens == EffectEnum.EFFECT_ENUM___UNSPECIFIED)
            | (effect_from_source_tokens == EffectEnum.EFFECT_ENUM___PAD)
            | (effect_from_source_tokens == EffectEnum.EFFECT_ENUM___NULL)
        )
        effect_from_source_embeddings = self.effect_from_source_embedding(
            effect_from_source_tokens
        )
        effect_from_source_embedding = effect_from_source_embeddings.sum(
            axis=0, where=effect_from_source_mask[..., None]
        )

        # FROM_TYPE tokens (2026-09-01): on the wire since the beginning,
        # never read. Summed masked one-hots over the typechart vocabulary
        # -- the cause channel ("hit by a Fire move") the type-matchup
        # reasoning needs.
        from_type_indices = np.array(
            [
                EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_TYPE_TOKEN0,
                EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_TYPE_TOKEN1,
                EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_TYPE_TOKEN2,
                EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_TYPE_TOKEN3,
                EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_TYPE_TOKEN4,
            ]
        )
        from_type_tokens = edge[from_type_indices]
        num_from_types = edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_TYPES]
        from_type_mask = np.arange(len(from_type_indices)) < num_from_types
        from_type_code = (
            jax.nn.one_hot(from_type_tokens, NUM_TYPECHART, dtype=self.cfg.dtype)
            * from_type_mask[..., None].astype(self.cfg.dtype)
        ).sum(axis=0)

        ability_token = edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__ABILITY_TOKEN]
        item_token = edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__ITEM_TOKEN]
        move_token = edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN]

        reg_boost_features = edge[
            np.array(
                [
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_ATK_VALUE,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_DEF_VALUE,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_SPA_VALUE,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_SPD_VALUE,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_SPE_VALUE,
                ]
            )
        ]
        spe_boost_features = edge[
            np.array(
                [
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_ACCURACY_VALUE,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_EVASION_VALUE,
                ]
            )
        ]
        stat_features = jnp.concatenate(
            (
                edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__DAMAGE_RATIO, None]
                / MAX_RATIO_TOKEN,
                edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__HEAL_RATIO, None]
                / MAX_RATIO_TOKEN,
                encode_reg_boosts(reg_boost_features),
                encode_spe_boosts(spe_boost_features),
            ),
            axis=-1,
        )

        embedding = self.entity_edge_sum(
            minor_args_encoding,
            boolean_code,
            stat_features.astype(self.cfg.dtype),
            self._embed_ability(ability_token),
            self._embed_item(item_token),
            self._embed_move(move_token),
            effect_from_source_embedding,
            from_type_code,
        )

        mask = (
            edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG]
            != BattlemajorargsEnum.BATTLEMAJORARGS_ENUM___UNSPECIFIED
        ) | (minor_args_indices.sum(axis=-1) > 0)

        embedding = mask * embedding

        return embedding, mask

    def _embed_field(self, field: jax.Array):
        turn_order_value = field[FieldFeature.FIELD_FEATURE__TURN_ORDER_VALUE]
        request_count = field[FieldFeature.FIELD_FEATURE__REQUEST_COUNT]

        encode_hex = jax.vmap(
            functools.partial(
                binary_scale_encoding, world_dim=65535, dtype=self.cfg.dtype
            )
        )

        my_side_condition_indices = field[
            FieldFeature.FIELD_FEATURE__MY_SIDECONDITIONS0 : FieldFeature.FIELD_FEATURE__MY_SIDECONDITIONS1
            + 1
        ]
        opp_side_condition_indices = field[
            FieldFeature.FIELD_FEATURE__OPP_SIDECONDITIONS0 : FieldFeature.FIELD_FEATURE__OPP_SIDECONDITIONS1
            + 1
        ]
        my_side_condition_encoding = encode_hex(my_side_condition_indices).reshape(-1)
        opp_side_condition_encoding = encode_hex(opp_side_condition_indices).reshape(-1)

        field_encoding = one_hot_concat_jax(
            [
                encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__WEATHER_ID,
                ),
                encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__WEATHER_MAX_DURATION,
                ),
                encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__WEATHER_MIN_DURATION,
                ),
                encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__TERRAIN_ID,
                ),
                encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__TERRAIN_MAX_DURATION,
                ),
                encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__TERRAIN_MIN_DURATION,
                ),
                encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__PSEUDOWEATHER_ID,
                ),
                encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__PSEUDOWEATHER_MAX_DURATION,
                ),
                encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__PSEUDOWEATHER_MIN_DURATION,
                ),
                encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__TURN_ORDER_VALUE,
                ),
            ],
            dtype=self.cfg.dtype,
        )

        my_side_condition_encoding = jnp.concatenate(
            (
                my_side_condition_encoding,
                one_hot_concat_jax(
                    [
                        encode_one_hot_field(
                            field,
                            FieldFeature.FIELD_FEATURE__MY_SPIKES,
                        ),
                        encode_one_hot_field(
                            field,
                            FieldFeature.FIELD_FEATURE__MY_TOXIC_SPIKES,
                        ),
                    ],
                    dtype=self.cfg.dtype,
                ),
            )
        )

        opp_side_condition_encoding = jnp.concatenate(
            (
                opp_side_condition_encoding,
                one_hot_concat_jax(
                    [
                        encode_one_hot_field(
                            field,
                            FieldFeature.FIELD_FEATURE__OPP_SPIKES,
                        ),
                        encode_one_hot_field(
                            field,
                            FieldFeature.FIELD_FEATURE__OPP_TOXIC_SPIKES,
                        ),
                    ],
                    dtype=self.cfg.dtype,
                ),
            )
        )

        mask = field[FieldFeature.FIELD_FEATURE__VALID].astype(jnp.bool)[..., None]

        field_embedding = self.field_linear(field_encoding)
        my_field_embedding = self.side_condition_linear(my_side_condition_encoding)
        opp_field_embedding = self.side_condition_linear(opp_side_condition_encoding)

        field_embeddings = jnp.stack(
            (
                field_embedding,
                my_field_embedding,
                opp_field_embedding,
            )
        )
        return field_embeddings, mask, request_count, turn_order_value

    def _embed_private_entities(self, private_team: jax.Array):
        return _lifted_entity_vmap(Encoder._embed_private_entity)(self, private_team)

    def _embed_action(self, action: jax.Array) -> jax.Array:
        boolean_code = one_hot_concat_jax(
            [
                encode_sqrt_one_hot_action(
                    action, MovesetFeature.MOVESET_FEATURE__PP, dtype=self.cfg.dtype
                ),
                encode_sqrt_one_hot_action(
                    action, MovesetFeature.MOVESET_FEATURE__MAXPP, dtype=self.cfg.dtype
                ),
                encode_one_hot_action(action, MovesetFeature.MOVESET_FEATURE__HAS_PP),
                encode_one_hot_action(action, MovesetFeature.MOVESET_FEATURE__DISABLED),
                encode_one_hot_action(
                    action, MovesetFeature.MOVESET_FEATURE__IS_WILDCARD
                ),
            ],
            dtype=self.cfg.dtype,
        )
        embedding = self.action_sum(
            self._embed_move(action[MovesetFeature.MOVESET_FEATURE__MOVE_ID]),
            boolean_code,
        )

        mask = (
            action[MovesetFeature.MOVESET_FEATURE__MOVE_ID]
            != MovesEnum.MOVES_ENUM___NULL
        ) & (
            action[MovesetFeature.MOVESET_FEATURE__MOVE_ID]
            != MovesEnum.MOVES_ENUM___PAD
        )

        return embedding, mask

    def _embed_moves(self, moveset: jax.Array) -> jax.Array:
        return _lifted_entity_vmap(Encoder._embed_action)(self, moveset)

    def _assemble_sequence(
        self,
        env_step: PlayerEnvOutput,
        history_row_states: jax.Array,
        history_row_valid: jax.Array,
        history_field_state: jax.Array,
        history_register_states: jax.Array,
        history_active_states: jax.Array,
    ):
        """One row per thing -> (sequence, row_valid), BEFORE the trunk.

        A CLS row, 12 public entities, my 6 sheet rows, my 16
        candidate move slots, the 17 target slots, the field triple, the
        recurrent field triple, the two previous-action rows, the request
        info row, the learner-only partition's 6 opponent sheet rows and
        VALUE_CLS row, 12 HISTORY_ENTITY rows, four recurrent registers, the
        learner-only PUBLIC_CLS row, and two trunk registers per tier (the
        privileged pair learner-only).
        Every identity a row
        carries is additive, and the layout itself lives in
        `rl/model/constants.py` so the offsets exist once.
        """
        dtype = self.cfg.dtype

        not_done = jnp.logical_not(env_step.done)
        move_cells = env_step.action_mask[MOVE_CELL_OFFSET:OTHER_CELL_OFFSET].reshape(
            len(MOVE_INDICES), len(TARGET_SLOT_INDICES)
        )
        other_cells = env_step.action_mask[OTHER_CELL_OFFSET:]
        move_slot_valid = move_cells.any(axis=-1) & not_done
        target_slot_valid = (move_cells.any(axis=0) | other_cells) & not_done
        switch_legal = env_step.action_mask[:MOVE_CELL_OFFSET].any() & not_done
        ally_slot_active = (
            jnp.arange(len(ALLY_TARGET_ROWS))
            < env_step.info[InfoFeature.INFO_FEATURE__NUM_ACTIVE]
        )
        ally_rows = jnp.asarray(ALLY_TARGET_ROWS)
        target_slot_valid = target_slot_valid.at[ally_rows].set(
            target_slot_valid[ally_rows] | (switch_legal & ally_slot_active)
        )

        public_rows, public_valid = _lifted_entity_vmap(Encoder._embed_public_entity)(
            self, env_step.public_team, env_step.revealed_team
        )
        field_rows, *_ = self._embed_field(env_step.field)
        # Public rows are per side, actives first, so the active slots'
        # occupants sit at fixed indices.
        active_public_rows = np.concatenate(
            (MY_ACTIVE_PUBLIC_ROWS, OPP_ACTIVE_PUBLIC_ROWS)
        )
        active_state_rows, is_active = _lifted_entity_vmap(Encoder._embed_active_state)(
            self, env_step.public_team[active_public_rows]
        )
        parts = self._public_parts(
            PublicRowInputs(
                public_rows=public_rows,
                public_valid=public_valid,
                public_sides=env_step.public_team[
                    :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE
                ],
                public_positions=env_step.public_team[
                    :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE
                ],
                field_rows=field_rows,
                history_entity_rows=history_row_states,
                history_row_valid=history_row_valid,
                history_field_rows=history_field_state,
                history_register_rows=history_register_states,
                active_state_rows=active_state_rows,
                active_state_valid=is_active & public_valid[active_public_rows],
                history_active_rows=history_active_states,
                info=env_step.info,
                target_slot_valid=target_slot_valid,
            )
        )
        target_rows = parts[SequenceGroup.TARGET_SLOT][0]

        private_rows, private_valid = self._embed_private_entities(
            env_step.private_team
        )
        move_rows, move_revealed = self._embed_moves(env_step.my_moveset)
        move_rows = move_rows + jnp.where(
            jnp.asarray(IS_WILDCARD_MOVE_SLOT)[:, None],
            self.wildcard_move_bias.astype(dtype),
            self.regular_move_bias.astype(dtype),
        )

        prev_source, prev_target = chosen_bank_rows(
            private_rows,
            move_rows,
            target_rows,
            env_step.info[InfoFeature.INFO_FEATURE__PREV_ACTION_CELL],
        )
        prev_action_rows = jnp.stack((prev_source, prev_target)) + jnp.concatenate(
            (
                self.prev_action_src_bias.astype(dtype),
                self.prev_action_tgt_bias.astype(dtype),
            ),
            axis=0,
        )
        has_prev_action = env_step.info[
            InfoFeature.INFO_FEATURE__HAS_PREV_ACTION
        ].astype(jnp.bool_)

        register_valid = jnp.ones(NUM_TRUNK_REGISTERS_PER_TIER, dtype=jnp.bool_)
        parts[SequenceGroup.CLS] = (
            self.cls_embedding.astype(dtype),
            jnp.ones(1, dtype=jnp.bool_),
        )
        parts.update(
            {
                SequenceGroup.PRIVATE_ENTITY: (
                    private_rows.astype(dtype),
                    private_valid,
                ),
                SequenceGroup.MOVE_SLOT: (
                    move_rows.astype(dtype),
                    move_revealed & move_slot_valid,
                ),
                SequenceGroup.PREV_ACTION: (
                    prev_action_rows,
                    jnp.full(2, has_prev_action),
                ),
                SequenceGroup.PRIVATE_REGISTER: (
                    self.private_register_embeddings.astype(dtype),
                    register_valid,
                ),
            }
        )
        if self.cfg.train:
            opp_private_rows, opp_private_valid = self._embed_private_entities(
                env_step.opp_private_team
            )
            parts.update(
                {
                    SequenceGroup.OPP_PRIVATE_ENTITY: (
                        opp_private_rows.astype(dtype),
                        opp_private_valid,
                    ),
                    SequenceGroup.PRIVILEGED_REGISTER: (
                        self.privileged_register_embeddings.astype(dtype),
                        register_valid,
                    ),
                    SequenceGroup.PUBLIC_CLS: self._public_cls_part(),
                    SequenceGroup.VALUE_CLS: (
                        self.value_cls_embedding.astype(dtype),
                        jnp.ones(1, dtype=jnp.bool_),
                    ),
                }
            )
        identities = sequence_identities(
            env_step,
            self.side_bias(jnp.arange(2)),
            self.pos_bias(jnp.arange(3)),
            self.target_slot_embeddings.astype(dtype),
            include_opponent=self.cfg.train,
        )
        return self._finish_sequence(parts, self.kept_rows(), identities)

    def _public_parts(
        self, inputs: PublicRowInputs
    ) -> dict[SequenceGroup, tuple[jax.Array, jax.Array]]:
        """The ten public-tier groups as (rows, valid), keyed by group."""
        dtype = self.cfg.dtype
        # No pokemon content: a target row is the slot's identity alone. Its
        # occupant may not be who the move lands on, so who stands opposite
        # reaches a move's logit through the readout's belief over their team.
        target_rows = jnp.zeros_like(self.target_slot_embeddings, dtype=dtype)
        info_row = self.info_linear(
            one_hot_concat_jax(
                [
                    encode_one_hot_info(
                        inputs.info, InfoFeature.INFO_FEATURE__REQUEST_TYPE
                    ),
                    encode_one_hot_info(
                        inputs.info, InfoFeature.INFO_FEATURE__NUM_ACTIVE
                    ),
                ],
                dtype=dtype,
            )
        )[None]
        return {
            SequenceGroup.PUBLIC_ENTITY: (
                inputs.public_rows.astype(dtype),
                inputs.public_valid,
            ),
            SequenceGroup.TARGET_SLOT: (target_rows, inputs.target_slot_valid),
            SequenceGroup.FIELD: (
                inputs.field_rows.astype(dtype),
                jnp.ones(NUM_FIELD_ROWS, dtype=jnp.bool_),
            ),
            SequenceGroup.HISTORY_FIELD: (
                inputs.history_field_rows.astype(dtype),
                jnp.ones(NUM_FIELD_ROWS, dtype=jnp.bool_),
            ),
            SequenceGroup.INFO: (info_row.astype(dtype), jnp.ones(1, dtype=jnp.bool_)),
            SequenceGroup.HISTORY_ENTITY: (
                inputs.history_entity_rows.astype(dtype),
                inputs.history_row_valid,
            ),
            SequenceGroup.HISTORY_REGISTER: (
                inputs.history_register_rows.astype(dtype),
                jnp.ones(NUM_HISTORY_REGISTERS, jnp.bool_),
            ),
            SequenceGroup.ACTIVE_STATE: (
                inputs.active_state_rows.astype(dtype),
                inputs.active_state_valid,
            ),
            SequenceGroup.HISTORY_ACTIVE_STATE: (
                inputs.history_active_rows.astype(dtype),
                inputs.active_state_valid,
            ),
            SequenceGroup.PUBLIC_REGISTER: (
                self.public_register_embeddings.astype(dtype),
                jnp.ones(NUM_TRUNK_REGISTERS_PER_TIER, dtype=jnp.bool_),
            ),
        }

    def _public_cls_part(self) -> tuple[jax.Array, jax.Array]:
        return (
            self.public_cls_embedding.astype(self.cfg.dtype),
            jnp.ones(1, dtype=jnp.bool_),
        )

    def _finish_sequence(
        self,
        parts: dict[SequenceGroup, tuple[jax.Array, jax.Array]],
        kept_rows: np.ndarray,
        identities: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Concatenate the parts in LAYOUT order (the order exists once, in
        constants.py; checked against the kept rows' group ids), normalise
        each row's content, then add the identities and group bias the
        content RMS must not rescale."""
        dtype = self.cfg.dtype
        kept_groups = [group for group, _ in SEQUENCE_LAYOUT if group in parts]
        assert [int(group) for group in kept_groups] == sorted(
            set(SEQUENCE_GROUP_IDS[kept_rows].tolist()),
            key=[int(group) for group, _ in SEQUENCE_LAYOUT].index,
        ), kept_groups
        sequence = jnp.concatenate([parts[group][0] for group in kept_groups], axis=0)
        row_valid = jnp.concatenate([parts[group][1] for group in kept_groups])
        assert sequence.shape[0] == len(kept_rows), sequence.shape
        group_ids = jnp.asarray(SEQUENCE_GROUP_IDS[kept_rows])
        sequence = self.input_normalisation(sequence, row_valid, group_ids)
        sequence = sequence + identities[kept_rows]
        sequence = sequence + self.sequence_group_bias.astype(dtype)[group_ids]
        sequence = jnp.where(row_valid[:, None], sequence, 0)
        return sequence, row_valid

    def _batched_forward(
        self,
        env_step: PlayerEnvOutput,
        history_row_states: jax.Array,
        history_row_valid: jax.Array,
        history_field_state: jax.Array,
        history_register_states: jax.Array,
        history_active_states: jax.Array,
    ):
        """The whole per-timestep forward: assemble, then run the trunk.

        Split from `_assemble_sequence` so a test can read the rows as they
        go IN. Every identity a row carries is additive and applied there, so
        that is where an identity bug is visible; after the trunk every row
        has mixed with every other and the reading is behavioural rather than
        structural.
        """
        sequence, row_valid = self._assemble_sequence(
            env_step,
            history_row_states,
            history_row_valid,
            history_field_state,
            history_register_states,
            history_active_states,
        )
        kept_rows = self.kept_rows()
        read_mask = SEQUENCE_READ_MASK[np.ix_(kept_rows, kept_rows)]
        trunk_in = sequence
        trunk_out = self.trunk(sequence, row_valid, read_mask)
        # The panels' read of which rows the blocks WRITE (L2, taken before
        # the output norm puts every row back at RMS 1) and by how much they
        # TURN each row (input-output cosine). Learner-only.
        if self.cfg.train:
            group_l2_sum, group_rows = group_row_l2(
                trunk_out, row_valid, self.group_ids(), NUM_SEQUENCE_GROUPS
            )
            in_out_cosine_sum = group_row_cosine(
                trunk_in, trunk_out, row_valid, self.group_ids(), NUM_SEQUENCE_GROUPS
            )
            trunk_group_stats = (group_l2_sum, group_rows, in_out_cosine_sum)
        else:
            trunk_group_stats = None
        sequence = self.output_normalisation(trunk_out, row_valid, self.group_ids())
        return sequence, row_valid, trunk_group_stats

    def _event_inputs(
        self,
        history_output,
        step_field_embeddings: jax.Array,
        public_cache: jax.Array,
        active_state_cache: jax.Array,
        num_active: int,
    ) -> tuple[PublicRowInputs, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Per-step PublicRowInputs from the history scan's own products
        (leading axis H): the slot's latest snapshot as its public row, its
        side/position/fainted read off the cache row that snapshot came
        from, each active slot's state token off its occupant's cache row,
        the step's field rows, the scan states as the history rows.
        The request row is a MOVE request with `num_active` actives and
        every target legal -- the replay convention."""
        row_index = history_output.node_row_index
        slot_valid = row_index >= 0
        cached = public_cache[row_index.clip(0)]
        sides = jnp.where(
            slot_valid,
            cached[..., EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE],
            0,
        )
        positions = jnp.where(
            slot_valid,
            cached[..., EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE],
            BENCH_POSITION,
        )
        fainted = jnp.where(
            slot_valid,
            cached[..., EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED],
            0,
        )

        ally_rows, ally_found = active_slot_rows(
            slot_valid, sides, positions, SIDE_MINE
        )
        enemy_rows, enemy_found = active_slot_rows(
            slot_valid, sides, positions, SIDE_OPPONENT
        )
        active_slots = jnp.concatenate((ally_rows, enemy_rows), axis=-1)
        active_cache_rows = jnp.take_along_axis(row_index, active_slots, axis=-1)
        num_steps = row_index.shape[0]
        info = jnp.zeros((num_steps, len(InfoFeature.keys())), jnp.int32)
        info = info.at[:, InfoFeature.INFO_FEATURE__REQUEST_TYPE].set(
            RequestType.REQUEST_TYPE__MOVE
        )
        info = info.at[:, InfoFeature.INFO_FEATURE__NUM_ACTIVE].set(num_active)
        inputs = PublicRowInputs(
            public_rows=history_output.node_snapshots,
            public_valid=slot_valid,
            public_sides=sides,
            public_positions=positions,
            field_rows=step_field_embeddings,
            history_entity_rows=history_output.slot_snapshots,
            history_row_valid=slot_valid,
            history_field_rows=history_output.field_snapshots,
            history_register_rows=history_output.register_snapshots,
            active_state_rows=active_state_cache[active_cache_rows.clip(0)],
            active_state_valid=jnp.concatenate((ally_found, enemy_found), axis=-1),
            history_active_rows=history_output.active_snapshots,
            info=info,
            target_slot_valid=jnp.ones(
                (num_steps, len(TARGET_SLOT_INDICES)), jnp.bool_
            ),
        )
        return inputs, slot_valid, sides, positions, fainted

    def _event_state(
        self, inputs: PublicRowInputs
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """One event's public sequence through the trunk: (post-trunk rows,
        pre-trunk rows, row_valid) over PUBLIC_SEQUENCE_ROWS."""
        parts = self._public_parts(inputs)
        parts[SequenceGroup.PUBLIC_CLS] = self._public_cls_part()
        identities = public_identities(
            inputs.public_sides,
            inputs.public_positions,
            self.side_bias(jnp.arange(2)),
            self.pos_bias(jnp.arange(3)),
            self.target_slot_embeddings.astype(self.cfg.dtype),
        )
        sequence, row_valid = self._finish_sequence(
            parts, PUBLIC_SEQUENCE_ROWS, identities
        )
        read_mask = SEQUENCE_READ_MASK[
            np.ix_(PUBLIC_SEQUENCE_ROWS, PUBLIC_SEQUENCE_ROWS)
        ]
        group_ids = jnp.asarray(SEQUENCE_GROUP_IDS[PUBLIC_SEQUENCE_ROWS])
        trunk_out = self.trunk(sequence, row_valid, read_mask)
        return (
            self.output_normalisation(trunk_out, row_valid, group_ids),
            sequence,
            row_valid,
        )

    def encode_events(
        self,
        packed_history_step: PlayerPackedHistoryOutput,
        history_step: PlayerHistoryOutput,
        carry: HistoryCarry = HistoryCarry(),
        num_active: int = 1,
    ) -> EventStates:
        """The public sequence through the trunk after EVERY history step:
        the event world model's states. Reads the packed caches and the
        field history only -- nothing a replay does not carry."""
        history_output, step_field_embeddings, active_state_cache = (
            self._run_history_encoder(packed_history_step, history_step, carry)
        )
        inputs, slot_valid, sides, positions, fainted = self._event_inputs(
            history_output,
            step_field_embeddings,
            packed_history_step.public_cache,
            active_state_cache,
            num_active,
        )
        states, pre_trunk, row_valid = jax.vmap(self._event_state)(inputs)
        return EventStates(
            states=states,
            inputs=pre_trunk,
            row_valid=row_valid,
            step_valid=history_output.step_valid,
            step_request_count=history_output.step_request_count,
            slot_valid=slot_valid,
            public_sides=sides,
            public_positions=positions,
            public_fainted=fainted,
            node_row_index=history_output.node_row_index,
        )

    def kept_rows(self) -> np.ndarray:
        """Which rows of SEQUENCE_LAYOUT this forward assembles: all of them
        for the learner, the policy-readable ones for the actor. Every head
        index is below the first dropped row (asserted in constants.py), so
        a head's absolute index names the same row in either sequence."""
        if self.cfg.get("public_only", False):
            return PUBLIC_SEQUENCE_ROWS
        if self.cfg.train:
            return np.arange(NUM_SEQUENCE_ROWS)
        return POLICY_READABLE_ROWS

    def local_row(self, row: int) -> int:
        """Where a layout row sits in this forward's kept sequence."""
        return int(np.flatnonzero(self.kept_rows() == row)[0])

    def group_ids(self) -> jax.Array:
        """The SequenceGroup of every kept row: the index into both norms'
        group scale banks, which are sized from the FULL layout on either
        path (one checkpoint; the actor indexes the rows it kept)."""
        return jnp.asarray(SEQUENCE_GROUP_IDS[self.kept_rows()])

    def _run_history_encoder(
        self,
        packed_history_step: PlayerPackedHistoryOutput,
        history_step: PlayerHistoryOutput,
        carry: HistoryCarry = HistoryCarry(),
    ):
        """Shared front half of the history pathway: embeds the packed
        caches and field rows once and runs the recurrent scan from
        `carry` (the learned h0 by default). Returns (scan output, per-step
        field vectors, the packed rows' active-state tokens)."""
        # Embed the packed (entity snapshot, edge) cache once; both are shared
        # across every request of the trajectory.
        node_embedding_cache, _ = _lifted_entity_vmap(Encoder._embed_public_entity)(
            self, packed_history_step.public_cache, packed_history_step.revealed_cache
        )
        edge_embedding_cache, _ = _lifted_entity_vmap(Encoder._embed_edge)(
            self, packed_history_step.edge_cache
        )
        edge_slot_ids = packed_history_step.edge_cache[
            :, EntityEdgeFeature.ENTITY_EDGE_FEATURE__ENTITY_IDX
        ]
        edge_major_args = packed_history_step.edge_cache[
            :, EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG
        ]
        node_sides = packed_history_step.public_cache[
            :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE
        ]
        (
            step_field_embeddings,
            step_valid,
            step_request_count,
            _,
        ) = _lifted_entity_vmap(Encoder._embed_field)(self, history_step.field)

        node_positions = packed_history_step.public_cache[
            :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE
        ]
        node_identity_cache = self.side_bias(node_sides) + self.pos_bias(node_positions)
        active_state_cache, _ = _lifted_entity_vmap(Encoder._embed_active_state)(
            self, packed_history_step.public_cache
        )
        history_output = self.history_encoder(
            history_field=history_step.field,
            node_embedding_cache=node_embedding_cache,
            edge_embedding_cache=edge_embedding_cache,
            edge_slot_ids=edge_slot_ids,
            edge_major_args=edge_major_args,
            node_identity_cache=node_identity_cache,
            active_state_cache=active_state_cache,
            active_slot_ids=active_slot_index(node_sides, node_positions),
            active_identities=active_slot_identities(
                self.side_bias(jnp.arange(2)), self.pos_bias(jnp.arange(3))
            ),
            field_identities=field_identities(self.side_bias(jnp.arange(2))),
            field_row_embeddings=step_field_embeddings,
            step_request_count=step_request_count,
            step_valid=step_valid.squeeze(-1),
            carry=carry,
        )
        return history_output, step_field_embeddings, active_state_cache

    def encode_history(
        self,
        env_step: PlayerEnvOutput,
        packed_history_step: PlayerPackedHistoryOutput,
        history_step: PlayerHistoryOutput,
        carry: HistoryCarry = HistoryCarry(),
    ):
        """Recurrent history pathway over the shared trajectory history.

        Consumes ONLY the public event stream — packed public entity/edge
        caches, the field history, and INFO_FEATURE__REQUEST_COUNT — no
        private observation fields, movesets, or action masks. This makes
        it safe to train against replay exports (which contain exactly the
        same inputs) and reuse live without any distribution projection.

        Returns, per request: ((T, NUM_PUBLIC_SLOTS, D) slot states,
        (T, D) field state, (T, NUM_PUBLIC_SLOTS, D) latest raw node
        snapshot per slot, (T, NUM_HISTORY_REGISTERS, D) global history
        registers, (T, NUM_ACTIVE_SLOTS, D) active-slot states), and the whole
        per-step PerSlotHistoryOutput for the telemetry that reads it.
        """
        history_output, *_ = self._run_history_encoder(
            packed_history_step, history_step, carry
        )

        # Read the recurrent state as of each request: the snapshot after the
        # last history step whose request_count <= the request's.
        request_count = env_step.info[..., InfoFeature.INFO_FEATURE__REQUEST_COUNT]
        return (
            *self.history_encoder.state_at_requests(
                history_output, request_count, carry
            ),
            history_output,
        )

    def _history_inputs(
        self,
        env_step: PlayerEnvOutput,
        packed_history_step: PlayerPackedHistoryOutput,
        history_step: PlayerHistoryOutput,
        carry: HistoryCarry = HistoryCarry(),
    ):
        """The history pathway's inputs to the sequence, in PUBLIC-ROW
        order: (row_states, order_valid, field_state, register_states,
        active_states), plus
        the per-step PerSlotHistoryOutput they were read from. The one
        place the slot-to-row alignment is written; offline reads call it
        directly.
        """
        (
            slot_states,
            field_state,
            _,
            register_states,
            active_states,
            history_output,
        ) = self.encode_history(env_step, packed_history_step, history_step, carry)

        # History-encoder slots are keyed by the stable entity index that
        # edges carry (revelation order across both sides), while public team
        # rows are per-side and re-sorted actives-first every state.
        # PUBLIC_ORDER is the server-provided permutation between the two:
        # row i of the public team holds the pokemon in slot public_order[i],
        # or -1 for unrevealed fillers (masked out of the cross-attention).
        public_order = env_step.info[
            ...,
            InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0 : InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11
            + 1,
        ]
        order_valid = (public_order >= 0) & (public_order < NUM_PUBLIC_SLOTS)
        aligned_order = public_order.clip(0, NUM_PUBLIC_SLOTS - 1)[..., None]
        row_states = jnp.take_along_axis(slot_states, aligned_order, axis=1)
        return (
            row_states,
            order_valid,
            field_state,
            register_states,
            active_states,
            history_output,
        )

    def assembled_sequence(
        self,
        env_step: PlayerEnvOutput,
        packed_history_step: PlayerPackedHistoryOutput,
        history_step: PlayerHistoryOutput,
    ):
        """The trunk's INPUT over time, (T, rows, width) with its row_valid --
        `_assemble_sequence` with the real history inputs and no trunk.
        Offline reads only (rl/probes/{trunk_homogeneity,separation_probe});
        nothing in training calls it."""
        assemble = nn.vmap(
            Encoder._assemble_sequence,
            variable_axes={"params": None, "intermediates": 0},
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )
        *history_inputs, _ = self._history_inputs(
            env_step, packed_history_step, history_step
        )
        sequence, row_valid = assemble(self, env_step, *history_inputs)
        return sequence, row_valid

    def __call__(
        self,
        env_step: PlayerEnvOutput,
        packed_history_step: PlayerPackedHistoryOutput,
        history_step: PlayerHistoryOutput,
        carry: HistoryCarry = HistoryCarry(),
    ):
        *history_inputs, history_output = self._history_inputs(
            env_step, packed_history_step, history_step, carry
        )
        sequence, row_valid, trunk_group_stats = _forward_vmap()(
            self, env_step, *history_inputs
        )
        return (
            sequence,
            row_valid,
            trunk_group_stats,
            history_step_stats(history_output, STEP_KEY_MASK),
            history_carry_from(history_output),
        )
