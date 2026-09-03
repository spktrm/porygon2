import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from constants import MAX_RATIO_TOKEN
from rl.environment.data import (
    MOVE_CELL_OFFSET,
    MOVE_INDICES,
    NUM_ACTION_FEATURES,
    NUM_FROM_SOURCE_EFFECTS,
    NUM_MOVES,
    NUM_TYPECHART,
    ONEHOT_ENCODERS,
    OTHER_CELL_OFFSET,
    TARGET_SLOT_INDICES,
)
from rl.environment.interfaces import (
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
    EntityPrivateNodeFeature,
    EntityPublicNodeFeature,
    EntityRevealedNodeFeature,
    FieldFeature,
    InfoFeature,
    MovesetFeature,
)
from rl.model.constants import (
    ALLY_TARGET_ROWS,
    ENEMY_TARGET_ROWS,
    IS_WILDCARD_MOVE_SLOT,
    MY_ACTIVE_PUBLIC_ROWS,
    NUM_FIELD_ROWS,
    NUM_PUBLIC_SLOTS,
    NUM_SEQUENCE_GROUPS,
    NUM_SEQUENCE_ROWS,
    NUM_TOKEN_TYPES,
    OPP_ACTIVE_PUBLIC_ROWS,
    PRIVATE_TOKEN_TYPES,
    PUBLIC_TOKEN_TYPES,
    SEQUENCE_GROUP_IDS,
)
from rl.model.features import (
    binary_scale_encoding,
    encode_divided_one_hot_edge,
    encode_divided_one_hot_public_entity,
    encode_one_hot_action,
    encode_one_hot_edge,
    encode_one_hot_field,
    encode_one_hot_info,
    encode_one_hot_private_entity,
    encode_one_hot_public_entity,
    encode_reg_boosts,
    encode_spe_boosts,
    encode_sqrt_one_hot_action,
    encode_sqrt_one_hot_public_entity,
    get_private_entity_mask,
    get_public_entity_mask,
)
from rl.model.history_encoder import (
    HistoryAttentionPool,
    NodeHistoryRead,
    PerSlotHistoryEncoder,
    history_carry_from,
    history_step_stats,
)
from rl.model.modules import (
    COLLECT_INTERMEDIATES,
    SumEmbeddings,
    one_hot_concat_jax,
)
from rl.model.trunk import Trunk

# Typed action-slot groups (canonical partition lives in
# rl/environment/data.py next to the modality masks): move slots are
# move-feature-derived, switch slots entity-derived, target/structural
# slots key-only. Since 2026-08-17 the groups are not just decoder
# bookkeeping — each is its own residual stream through the round trunk.


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
    the caller's graph — smaller HLO and cheaper compiles (the retained-
    executable RAM lesson from run 1326), plus trace reuse whenever two
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


class EntitySumPool(nn.Module):
    """Pool one entity's attribute tokens into a single entity vector by a
    masked SUM.

    Each token carries its token-type bias (the field identity), invalid
    attributes (unrevealed, or the active-only state token on a benched
    entity) contribute nothing, and a fully masked set pools to zeros. The
    divisor is the STATIC token count, as in `simple_sum_embeddings`, so the
    entity vector stays LINEAR in its attribute multi-hots: a matchup the
    readout's bilinear turns on (my move's type x their species' types) is
    then a fixed subspace of the row, not a function the pool has to route.
    Measured 2026-09-03 (`rl/offline/type_probe.py` and the supervised
    ceiling beside it): the same readout form reached held-out 0.60 on the
    attention-pooled rows and 0.79 on summed ones, against 0.80 from the raw
    multi-hots — the attention pool was eroding type legibility, not adding
    within-entity interactions the trunk could use.
    """

    @nn.compact
    def __call__(
        self, tokens: jax.Array, token_mask: jax.Array, token_types: jax.Array
    ) -> jax.Array:
        token_bias = self.param(
            "token_bias",
            nn.initializers.zeros_init(),
            (NUM_TOKEN_TYPES, tokens.shape[-1]),
        )
        tokens = tokens + token_bias[token_types].astype(tokens.dtype)
        weights = token_mask.astype(tokens.dtype)[..., None]
        num_tokens = tokens.shape[-2]
        return jnp.sum(tokens * weights, axis=-2) / jnp.sqrt(num_tokens).astype(
            tokens.dtype
        )


class Encoder(nn.Module):
    """
    Encoder model for processing environment steps and history to generate embeddings.
    """

    cfg: ConfigDict

    def setup(self):
        # Extract configuration parameters for embedding sizes.
        entity_size = self.cfg.entity_size
        self.entity_size = entity_size

        embed_kwargs = dict(features=entity_size, dtype=self.cfg.dtype)
        dense_kwargs = dict(features=entity_size, dtype=self.cfg.dtype)

        # Initialize embeddings for various entities and features.
        self.effect_from_source_embedding = nn.Embed(
            num_embeddings=NUM_FROM_SOURCE_EFFECTS,
            name="effect_from_source_embedding",
            **embed_kwargs,
        )

        # Positional / Modality Embeddings
        embedding_init = nn.initializers.variance_scaling(
            1.0, "fan_in", "normal", out_axis=0
        )
        bias_init = nn.initializers.zeros_init()

        self.side_bias = nn.Embed(2, name="side_bias", **embed_kwargs)
        self.pos_bias = nn.Embed(3, name="pos_bias", **embed_kwargs)

        # One learned identity per target slot. Replaces the separate
        # pass_embeddings / target_embeddings tables (2026-08-29): those two
        # plus the four entity-derived targets were three ways of saying "a
        # thing a move can be aimed at", and the readout wants them as one
        # contiguous block it can score against.
        self.target_slot_embeddings = self.param(
            "target_slot_embeddings",
            embedding_init,
            (len(TARGET_SLOT_INDICES), entity_size),
        )
        # The previous action's two slot ids, looked up directly. Before the
        # rewrite these were GATHERED out of the built action stream, which
        # made the token set circular and forced it to be assembled twice.
        self.prev_action_embeddings = nn.Embed(
            num_embeddings=NUM_ACTION_FEATURES,
            embedding_init=embedding_init,
            name="prev_action_embeddings",
            **embed_kwargs,
        )
        # My own private sheet's identity tag. NOT side_bias(0): the service
        # writes ENTITY_PUBLIC_NODE_FEATURE__SIDE = isMySide(...), so row 1 of
        # side_bias is MINE and row 0 is the opponent's -- the sheet was
        # carrying the opponent's tag (fixed 2026-08-28). It gets its own
        # param rather than side_bias(1) because the sheet's provenance is
        # already the read's group_bias; what it needs here is only "mine".
        self.private_side_bias = self.param(
            "private_side_bias", embedding_init, (1, entity_size)
        )
        # The learner-only partition (2026-09-01): the opponent's request
        # truth enters as 6 OPP_PRIVATE_ENTITY rows, and ONE privileged
        # value query row (VALUE_CLS) reads them -- SEQUENCE_READ_MASK is
        # what keeps every policy-readable row blind to both. The rows carry
        # a Dreamer-style discrete code, not the raw latent: per mon,
        # `opp_code_logits` maps the pooled private embedding to
        # (num_groups, num_classes) categoricals, straight-through argmax
        # picks one class per group, and the row content is the concat of
        # the groups' code-table vectors. The privileged value loss is the
        # gradient that GROUNDS the code (through the straight-through
        # estimator); the belief head later predicts it from public rows.
        self.opp_private_side_bias = self.param(
            "opp_private_side_bias", embedding_init, (1, entity_size)
        )
        self.value_cls_embedding = self.param(
            "value_cls_embedding", embedding_init, (1, entity_size)
        )
        code_groups = self.cfg.opp_code.num_groups
        code_classes = self.cfg.opp_code.num_classes
        assert entity_size % code_groups == 0
        self.opp_code_logits = nn.Dense(
            name="opp_code_logits",
            features=code_groups * code_classes,
            dtype=self.cfg.dtype,
        )
        self.opp_code_embedding = self.param(
            "opp_code_embedding",
            embedding_init,
            (code_groups, code_classes, entity_size // code_groups),
        )
        # Whose side a field token describes. Row 1 = mine, row 0 = theirs —
        # the SIDE convention, written once. Until 2026-08-28 these two
        # tokens borrowed pos_bias rows 1/0, but pos_bias is indexed by
        # ENTITY_PUBLIC_NODE_FEATURE__ACTIVE (= scoreOrder, {0, 2} in
        # singles), so row 0 meant "benched pokemon" AND "opponent side
        # conditions" — one vector, two meanings, coupled gradients. It is
        # also the only thing separating my hazards from theirs, since both
        # go through side_condition_linear.
        self.field_side_bias = self.param(
            "field_side_bias", embedding_init, (2, entity_size)
        )
        self.prev_action_src_bias = self.param(
            "prev_action_src_bias", embedding_init, (1, entity_size)
        )
        self.prev_action_tgt_bias = self.param(
            "prev_action_tgt_bias", embedding_init, (1, entity_size)
        )

        # Action biases
        self.regular_move_bias = self.param(
            "regular_move_bias", bias_init, (1, entity_size)
        )
        self.wildcard_move_bias = self.param(
            "wildcard_move_bias", bias_init, (1, entity_size)
        )
        self.ally_target_bias = self.param(
            "ally_target_bias", bias_init, (1, entity_size)
        )
        self.enemy_target_bias = self.param(
            "enemy_target_bias", bias_init, (1, entity_size)
        )

        # The CLS row. The value head reads THIS ROW AND ONLY THIS ROW, so
        # loss_v_win's gradient reaches the trunk through it and it is the row
        # that has to aggregate the board. It is also unconditionally valid,
        # which is what stops a terminal step -- every action row masked off --
        # from attending over an empty key set and returning NaN. Replaces the
        # 4-row value_embeddings_table and its (4 * entity_size,) concat.
        self.cls_embedding = self.param(
            "cls_embedding", embedding_init, (1, entity_size)
        )
        # The sequence's own identity: one bias per row group, one per row.
        # Group says WHAT KIND of thing a row is, row says WHICH -- and the row
        # bias is what separates two slots naming the same mon (ALLY_i_TARGET
        # and reserve i) before anything else has trained.
        self.sequence_group_bias = self.param(
            "sequence_group_bias", bias_init, (NUM_SEQUENCE_GROUPS, entity_size)
        )
        self.sequence_row_bias = self.param(
            "sequence_row_bias", embedding_init, (NUM_SEQUENCE_ROWS, entity_size)
        )

        # Initialize linear layers for encoding various entity features.
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
        # sum of the attribute tokens plus the token-type bias (see
        # `EntitySumPool` for the measurement that retired the intra-entity
        # attention block it replaces). Token provenance is carried by the
        # token-type bias table; per-provenance input norms downstream keep
        # the two entity kinds separable.
        self.entity_pool = EntitySumPool(name="entity_pool")
        self.public_persistent_linear = nn.Dense(
            name="public_persistent_linear", use_bias=False, **dense_kwargs
        )
        self.public_transient_linear = nn.Dense(
            name="public_transient_linear", use_bias=False, **dense_kwargs
        )
        self.private_state_linear = nn.Dense(
            name="private_state_linear", use_bias=False, **dense_kwargs
        )

        # Initialize aggregation modules for combining feature embeddings.
        self.action_sum = SumEmbeddings(
            output_size=entity_size, dtype=self.cfg.dtype, name="action_sum"
        )
        self.entity_edge_sum = SumEmbeddings(
            output_size=entity_size, dtype=self.cfg.dtype, name="entity_edge_sum"
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

        # Recurrent history encoder over history edges. Twelve GRU states
        # (one per public slot) scanned along the history axis; per request we
        # read the state as of that request and hand it to the latent
        # input read as 13 raw key tokens (12 slots + field); the trunk
        # rounds see it only through the state latents.
        self.history_encoder = PerSlotHistoryEncoder(self.cfg, name="history_encoder")
        self.history_pool = HistoryAttentionPool(self.cfg, name="history_pool")
        self.history_node_read = NodeHistoryRead(self.cfg, name="history_node_read")
        self.history_field_step_linear = nn.Dense(
            name="history_field_step_linear", use_bias=False, **dense_kwargs
        )

        # The trunk. One sequence, `num_blocks` standard pre-RMSNorm blocks,
        # no gates and no block masks -- see rl/model/trunk.py for why the
        # three gated streams and their two feeding cross-attention reads all
        # collapse into this at 61 rows.
        self.trunk = Trunk(self.cfg.trunk, name="trunk")

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
        """The attribute-token half of a public entity: the tokens, their
        validity mask, the entity mask and the (pos + side) bias that the
        pooled vector carries. Split out from `_embed_public_entity` so the
        current-state path can pool the tokens ACROSS entities while the
        history cache keeps pooling them entity-locally."""
        # Encode volatile and type-change indices using the binary encoder.
        encode_hex = jax.vmap(
            functools.partial(
                binary_scale_encoding, dtype=self.cfg.dtype, world_dim=65535
            )
        )
        volatiles_indices = public[
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES0 : EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES8
            + 1
        ]
        volatiles_encoding = encode_hex(volatiles_indices).reshape(-1)

        typechange_indices = public[
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE0 : EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE1
            + 1
        ]
        typechange_encoding = encode_hex(typechange_indices).reshape(-1)

        hp_ratio = (
            public[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO]
            / MAX_RATIO_TOKEN
        ).astype(self.cfg.dtype)
        hp_features = jnp.concatenate(
            [
                hp_ratio[..., None],
                jax.nn.one_hot(jnp.floor(32 * hp_ratio), 32, dtype=self.cfg.dtype),
            ],
            axis=-1,
        ).reshape(-1)

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

        # Persistent condition: survives switching out, meaningful on the
        # bench. The active-only overlay (volatiles, boosts, typechange,
        # trapped/called-back/newly-switched, toxic counter) all resets on
        # switch, so it becomes its own token, masked by the ACTIVE flag.
        persistent_code = one_hot_concat_jax(
            [
                encode_sqrt_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__LEVEL,
                    dtype=self.cfg.dtype,
                ),
                encode_divided_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO,
                    MAX_RATIO_TOKEN / 32,
                ),
                encode_one_hot_public_entity(
                    public, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__GENDER
                ),
                encode_one_hot_public_entity(
                    public, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__STATUS
                ),
                encode_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ITEM_EFFECT,
                ),
                encode_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SLEEP_TURNS,
                ),
                encode_one_hot_public_entity(
                    public, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED
                ),
            ],
            dtype=self.cfg.dtype,
        )
        transient_code = one_hot_concat_jax(
            [
                encode_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BEING_CALLED_BACK,
                ),
                encode_one_hot_public_entity(
                    public, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TRAPPED
                ),
                encode_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__NEWLY_SWITCHED,
                ),
                encode_one_hot_public_entity(
                    public,
                    EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TOXIC_TURNS,
                ),
            ],
            dtype=self.cfg.dtype,
        )

        move_indices = np.array(
            [
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID0,
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID1,
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID2,
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID3,
            ]
        )
        move_pp_indices = np.array(
            [
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP0,
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP1,
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP2,
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP3,
            ]
        )
        move_tokens = revealed[move_indices]
        move_pp_tokens = public[move_pp_indices]

        is_valid_move = (move_tokens != MovesEnum.MOVES_ENUM___NULL) & (
            move_tokens != MovesEnum.MOVES_ENUM___UNSPECIFIED
        )
        move_pp_ratios = is_valid_move * (move_pp_tokens / 31).astype(self.cfg.dtype)
        move_pp_onehot = (
            jnp.zeros(NUM_MOVES, dtype=move_pp_ratios.dtype)
            .at[move_tokens]
            .set(move_pp_ratios)
            .clip(min=0, max=1)
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
        teratype_token = revealed[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__TERA_TYPE
        ]

        persistent_features = jnp.concatenate(
            [
                persistent_code,
                hp_features,
                move_pp_onehot,
                jax.nn.one_hot(
                    teratype_token, NUM_TYPECHART, dtype=move_embeddings.dtype
                ),
            ],
            axis=-1,
        )
        transient_features = jnp.concatenate(
            [
                transient_code,
                volatiles_encoding,
                typechange_encoding,
                encode_reg_boosts(reg_boost_features).astype(self.cfg.dtype),
                encode_spe_boosts(spe_boost_features).astype(self.cfg.dtype),
            ],
            axis=-1,
        )

        pos_bias = self.pos_bias(
            public[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE]
        )
        side_bias = self.side_bias(
            public[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE]
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
                jnp.stack(
                    (
                        self._embed_learnset(species_token),
                        self.public_persistent_linear(persistent_features),
                        self.public_transient_linear(transient_features),
                    )
                ),
            ),
            axis=0,
        )

        mask = get_public_entity_mask(revealed)
        is_active = (
            public[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE] > 0
        )
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
                jnp.stack((jnp.ones_like(mask), jnp.ones_like(mask), is_active)),
            ),
            axis=0,
        )

        return tokens, token_mask, mask, pos_bias + side_bias

    def _embed_public_entity(self, public: jax.Array, revealed: jax.Array):
        """Entity-LOCAL pooling: one entity's attribute tokens in, one entity
        vector out. Kept bitwise as-is for the packed history cache (2 *
        NUM_HISTORY rows, where a global mix would be neither affordable nor
        meaningful -- those rows are different turns, not a shared board)."""
        tokens, token_mask, mask, bias = self._public_entity_tokens(public, revealed)
        revealed_embedding = (
            self.entity_pool(tokens, token_mask, PUBLIC_TOKEN_TYPES) + bias
        )
        return revealed_embedding, mask

    def _private_entity_tokens(self, private: jax.Array, num_stat_bands: int = 8):
        """The attribute-token half of a private entity -- see
        `_public_entity_tokens`. NOTE the index constants below are the
        REVEALED enum applied to a PRIVATE row: legal only because
        SPECIES/ITEM/ABILITY/MOVEID0-3 are 1..7 in both enums."""
        move_indices = np.array(
            [
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID0,
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID1,
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID2,
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID3,
            ]
        )
        move_tokens = private[move_indices]

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

        boolean_code = one_hot_concat_jax(
            [
                encode_one_hot_private_entity(
                    private,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__TERA_TYPE,
                ),
            ],
            dtype=self.cfg.dtype,
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
        ].astype(self.cfg.dtype)

        stat_encoding = stat_features.astype(jnp.float32) / np.array(
            [714, 526, 658, 535, 658, 548], dtype=np.float32
        )
        freqs = (2.0 ** np.arange(num_stat_bands) * np.pi).astype(np.float32)
        # Phases reach 2^7.pi ~ 400 rad, where bf16 spacing is ~1 rad: cast
        # before sin/cos and the top bands are quantisation noise. Bands
        # in f32, cast after.
        phase = stat_encoding[..., None] * freqs[None]
        stat_encoding = (
            jnp.concatenate((jnp.sin(phase), jnp.cos(phase)), axis=-1)
            .reshape(-1)
            .astype(self.cfg.dtype)
        )

        # The truth channel (2026-08-31): current condition, written from
        # the request side, so a switch candidate's own row finally says
        # whether it is hurt, statused or fainted -- probe C measured the
        # trunk's public-row workaround at the floor. Same encodings as the
        # public path: hp scalar + 32-bin one-hot, StatusEnum one-hot,
        # toxic/sleep counters, fainted.
        private_hp_ratio = (
            private[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__HP_RATIO]
            / MAX_RATIO_TOKEN
        ).astype(self.cfg.dtype)
        private_hp_features = jnp.concatenate(
            [
                private_hp_ratio[..., None],
                jax.nn.one_hot(
                    jnp.floor(32 * private_hp_ratio), 32, dtype=self.cfg.dtype
                ),
            ],
            axis=-1,
        ).reshape(-1)
        condition_code = one_hot_concat_jax(
            [
                encode_one_hot_private_entity(
                    private,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__STATUS,
                ),
                encode_one_hot_private_entity(
                    private,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__HAS_STATUS,
                ),
                encode_one_hot_private_entity(
                    private,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__TOXIC_TURNS,
                ),
                encode_one_hot_private_entity(
                    private,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SLEEP_TURNS,
                ),
                encode_one_hot_private_entity(
                    private,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__FAINTED,
                ),
                # Turn-delta staleness of the source request (identically 0
                # on the own channel; >0 on the opponent channel when their
                # client lags the observer's build).
                encode_one_hot_private_entity(
                    private,
                    EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__REQUEST_LAG,
                ),
            ],
            dtype=self.cfg.dtype,
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
                self.private_state_linear(
                    jnp.concatenate(
                        (
                            boolean_code,
                            condition_code,
                            private_hp_features,
                            stat_encoding,
                        ),
                        axis=-1,
                    )
                )[None],
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
        MY OWN private team rows (the opponent's sheet, which this once also
        served, was deleted 2026-08-25)."""
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
        """
        Embed features of the field
        """
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

        side_biases = self.field_side_bias.astype(field_embedding.dtype)
        field_embeddings = jnp.stack(
            (
                field_embedding,
                my_field_embedding + side_biases[1],
                opp_field_embedding + side_biases[0],
            )
        )
        return field_embeddings, mask, request_count, turn_order_value

    def _embed_private_entities(self, private_team: jax.Array):
        return _lifted_entity_vmap(Encoder._embed_private_entity)(self, private_team)

    def _opp_code_rows(self, opp_private_team: jax.Array):
        """The opponent sheet as discrete-code rows (2026-09-01).

        Same embedder as my own sheet, then per mon a (G, K) multi-softmax
        with a 1% unimix floor (keeps classes reachable) and a
        straight-through argmax: forward sees one hard class per group, the
        backward flows through the probabilities, so the privileged value
        loss trains the embedder THROUGH the code and grounds it. Returns
        (rows, row_valid, code_one_hot); the one-hot is the belief head's
        label. All-zero deploy/old-shard buffers give row_valid all-False
        and the trunk mask makes the rows inert.
        """
        opp_latents, opp_valid = self._embed_private_entities(opp_private_team)
        code_groups = self.cfg.opp_code.num_groups
        code_classes = self.cfg.opp_code.num_classes
        code_logits = self.opp_code_logits(opp_latents).reshape(
            opp_latents.shape[0], code_groups, code_classes
        )
        code_probs = jax.nn.softmax(code_logits.astype(jnp.float32), axis=-1)
        code_probs = 0.99 * code_probs + 0.01 / code_classes
        hard_one_hot = jax.nn.one_hot(
            jnp.argmax(code_probs, axis=-1), code_classes, dtype=code_probs.dtype
        )
        code_one_hot = hard_one_hot + code_probs - jax.lax.stop_gradient(code_probs)
        rows = jnp.einsum(
            "egk,gkd->egd",
            code_one_hot.astype(self.cfg.dtype),
            self.opp_code_embedding.astype(self.cfg.dtype),
        ).reshape(opp_latents.shape[0], -1)
        return rows, opp_valid, code_one_hot

    def _embed_action(self, action: jax.Array) -> jax.Array:
        """
        Encode features of a move, including its type, species, and action ID.
        """
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
        history_node_snapshots: jax.Array,
    ):
        """One row per thing -> (sequence, row_valid), BEFORE the trunk.

        61 rows: a CLS row, 12 public entities, my 6 sheet rows, my 16
        candidate move slots, the 17 target slots, the field triple, the
        recurrent field triple, the two previous-action rows, and the request
        info row. Every identity a row carries is additive, and the layout
        itself lives in `rl/model/constants.py` so the offsets exist once.
        """
        dtype = self.cfg.dtype

        # ---- the entities, one pooled vector each -------------------------
        # `_embed_public_entity` / `_embed_private_entity` are the SAME
        # entity-local pools the packed history cache runs on; before
        # 2026-08-29 the current board took a second path that emitted 10-11
        # raw attribute tokens per entity instead.
        public_rows, public_valid = _lifted_entity_vmap(Encoder._embed_public_entity)(
            self, env_step.public_team, env_step.revealed_team
        )
        # Until 2026-09-01 the history states were SUMMED into the public
        # rows here ("entity i's 11th attribute token"). They are their own
        # HISTORY_ENTITY rows now -- built below, once public_tag_index
        # exists -- so attention routes board-now vs diary instead of one
        # vector carrying their sum.

        private_rows, private_valid = self._embed_private_entities(
            env_step.private_team
        )
        private_rows = private_rows + self.private_side_bias.astype(private_rows.dtype)
        # No learned join key between a sheet row and its public row
        # (entity_index_tag, 2026-08-31 -> 2026-09-02): it never trained
        # (rms 0.0634 -> 0.0661 over 182k steps, ~3% of the row's norm) and
        # a public-only read from the sheet row scored no higher after the
        # trunk than before it, so the tag joined nothing. What relates the
        # two rows is their shared content -- one species/ability/item/move
        # embedder feeds both -- and the wire's ENTITY_IDX survives ONLY as
        # the belief head's alignment (player_model.belief_alignment).

        # ---- history as its own rows (2026-09-01) --------------------------
        # Entity i's diary: GRU slot state + the latest raw node snapshot
        # (the TGN embedding module's memory + raw-features pair), aligned
        # to public row i by Encoder.__call__'s PUBLIC_ORDER gather; the
        # group/row biases below are its identity, and row i's bias is what
        # pairs it with public row i.
        history_entity_rows = history_row_states.astype(
            dtype
        ) + history_node_snapshots.astype(dtype)

        # ---- the learner-only partition -----------------------------------
        # The opponent's request truth as discrete-code rows (see
        # _opp_code_rows) with their OWN side bias. SEQUENCE_READ_MASK keeps
        # every policy-readable row blind to these and to VALUE_CLS itself.
        opp_private_rows, opp_private_valid, opp_code_one_hot = self._opp_code_rows(
            env_step.opp_private_team
        )
        opp_private_rows = opp_private_rows + self.opp_private_side_bias.astype(
            opp_private_rows.dtype
        )

        # ---- my candidate moves, one row per action slot ------------------
        # Row k IS action slot MOVE_INDICES[k], carrying that slot's pp,
        # disabled and wildcard-availability state -- the only route those
        # features have into the model.
        move_rows, move_revealed = self._embed_moves(env_step.my_moveset)
        move_rows = move_rows + jnp.where(
            jnp.asarray(IS_WILDCARD_MOVE_SLOT)[:, None],
            self.wildcard_move_bias.astype(dtype),
            self.regular_move_bias.astype(dtype),
        )

        # ---- the target slots ---------------------------------------------
        # Every target slot has a learned identity; the four that NAME a mon
        # add that mon's entity row, so a move is scored against the actual
        # pokemon it would hit rather than against a bare positional slot.
        target_rows = self.target_slot_embeddings.astype(dtype)
        target_rows = target_rows.at[jnp.asarray(ALLY_TARGET_ROWS)].add(
            public_rows[jnp.asarray(MY_ACTIVE_PUBLIC_ROWS)]
            + self.ally_target_bias.astype(dtype)
        )
        target_rows = target_rows.at[jnp.asarray(ENEMY_TARGET_ROWS)].add(
            public_rows[jnp.asarray(OPP_ACTIVE_PUBLIC_ROWS)]
            + self.enemy_target_bias.astype(dtype)
        )

        # ---- field, now and remembered ------------------------------------
        field_rows, *_ = self._embed_field(env_step.field)
        # (global, mine, theirs), matching _embed_field's own triple and
        # tagged with the same field_side_bias, so "whose side" reads the same
        # way on a current field row and on its recurrent memory.
        field_side = self.field_side_bias.astype(dtype)
        history_field_rows = (
            history_field_state.astype(dtype)
            .at[1]
            .add(field_side[1])
            .at[2]
            .add(field_side[0])
        )

        # ---- the previous action ------------------------------------------
        # An embedding lookup on the two slot ids, NOT a gather out of a built
        # action stream. That gather is why `InputTokenSet.assemble` had to run
        # twice before 2026-08-29: the previous action's rows were read off a
        # sequence that was itself built from a read over those rows.
        prev_action_rows = self.prev_action_embeddings(
            jnp.stack(
                (
                    env_step.info[InfoFeature.INFO_FEATURE__PREV_ACTION_SRC],
                    env_step.info[InfoFeature.INFO_FEATURE__PREV_ACTION_TGT],
                )
            )
        ) + jnp.concatenate(
            (
                self.prev_action_src_bias.astype(dtype),
                self.prev_action_tgt_bias.astype(dtype),
            ),
            axis=0,
        )
        has_prev_action = env_step.info[
            InfoFeature.INFO_FEATURE__HAS_PREV_ACTION
        ].astype(jnp.bool_)

        info_row = self.info_linear(
            one_hot_concat_jax(
                [
                    encode_one_hot_info(
                        env_step.info, InfoFeature.INFO_FEATURE__REQUEST_TYPE
                    ),
                    encode_one_hot_info(
                        env_step.info, InfoFeature.INFO_FEATURE__NUM_ACTIVE
                    ),
                ],
                dtype=dtype,
            )
        )[None]

        # ---- the sequence --------------------------------------------------
        # Row validity from the block mask: a move row is live if any of its
        # target cells is, a target row if any move can reach it or it stands
        # alone -- the same content the old grid's any-over-both-axes gave.
        not_done = jnp.logical_not(env_step.done)
        move_cells = env_step.action_mask[MOVE_CELL_OFFSET:OTHER_CELL_OFFSET].reshape(
            len(MOVE_INDICES), len(TARGET_SLOT_INDICES)
        )
        other_cells = env_step.action_mask[OTHER_CELL_OFFSET:]
        move_slot_valid = move_cells.any(axis=-1) & not_done
        target_slot_valid = (move_cells.any(axis=0) | other_cells) & not_done

        sequence = jnp.concatenate(
            (
                self.cls_embedding.astype(dtype),
                public_rows.astype(dtype),
                private_rows.astype(dtype),
                move_rows.astype(dtype),
                target_rows,
                field_rows.astype(dtype),
                history_field_rows,
                prev_action_rows,
                info_row.astype(dtype),
                opp_private_rows.astype(dtype),
                self.value_cls_embedding.astype(dtype),
                history_entity_rows,
            ),
            axis=0,
        )
        row_valid = jnp.concatenate(
            (
                # The CLS row is ALWAYS valid. The value head reads it, and it
                # is also what guarantees every query row has a non-empty key
                # set -- a terminal step, where every action row is masked off,
                # would otherwise attend over nothing and return NaN.
                jnp.ones(1, dtype=jnp.bool_),
                public_valid,
                private_valid,
                move_revealed & move_slot_valid,
                target_slot_valid,
                jnp.ones(NUM_FIELD_ROWS, dtype=jnp.bool_),
                jnp.ones(NUM_FIELD_ROWS, dtype=jnp.bool_),
                jnp.full(2, has_prev_action),
                jnp.ones(1, dtype=jnp.bool_),
                # Secret rows: valid only where the wire carried a real mon
                # (all-zero deploy/old-shard buffers embed as invalid).
                opp_private_valid,
                # VALUE_CLS is always valid, like CLS: the privileged head
                # reads it every step, terminal or not.
                jnp.ones(1, dtype=jnp.bool_),
                history_row_valid,
            )
        )

        sequence = (
            sequence
            + self.sequence_group_bias.astype(dtype)[jnp.asarray(SEQUENCE_GROUP_IDS)]
            + self.sequence_row_bias.astype(dtype)
        )
        sequence = jnp.where(row_valid[:, None], sequence, 0)
        return sequence, row_valid, opp_code_one_hot

    def _batched_forward(
        self,
        env_step: PlayerEnvOutput,
        history_row_states: jax.Array,
        history_row_valid: jax.Array,
        history_field_state: jax.Array,
        history_node_snapshots: jax.Array,
    ):
        """The whole per-timestep forward: assemble, then run the trunk.

        Split from `_assemble_sequence` so a test can read the rows as they
        go IN. Every identity a row carries is additive and applied there, so
        that is where an identity bug is visible; after the trunk every row
        has mixed with every other and the reading is behavioural rather than
        structural.
        """
        sequence, row_valid, opp_code_one_hot = self._assemble_sequence(
            env_step,
            history_row_states,
            history_row_valid,
            history_field_state,
            history_node_snapshots,
        )
        return self.trunk(sequence, row_valid), opp_code_one_hot

    def _run_history_encoder(
        self,
        packed_history_step: PlayerPackedHistoryOutput,
        history_step: PlayerHistoryOutput,
        carry: HistoryCarry = HistoryCarry(),
    ):
        """Shared front half of the history pathway: embeds the packed
        caches and field rows once and runs the recurrent scan from
        `carry` (the learned h0 by default). Returns (scan output,
        edge_slot_ids, node_sides, per-step field vectors)."""
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
        # One pooled field vector per history step from the (field, my-side,
        # opp-side) token triple.
        (
            step_field_embeddings,
            step_valid,
            step_request_count,
            _,
        ) = _lifted_entity_vmap(Encoder._embed_field)(self, history_step.field)
        step_field_vec = self.history_field_step_linear(
            step_field_embeddings.reshape(step_field_embeddings.shape[0], -1)
        )

        history_output = self.history_encoder(
            history_field=history_step.field,
            node_embedding_cache=node_embedding_cache,
            edge_embedding_cache=edge_embedding_cache,
            edge_slot_ids=edge_slot_ids,
            edge_major_args=edge_major_args,
            node_sides=node_sides,
            field_step_embeddings=step_field_vec,
            field_row_embeddings=step_field_embeddings,
            step_request_count=step_request_count,
            step_valid=step_valid.squeeze(-1),
            carry=carry,
        )
        return (
            history_output,
            edge_slot_ids,
            node_sides,
            step_field_vec,
            step_field_embeddings,
        )

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
        same inputs) and reuse live without any distribution projection;
        the offline outcome critic (rl/offline/model.py) builds on it.

        Returns, per request: ((T, NUM_PUBLIC_SLOTS, D) slot states,
        (T, D) field state, (T, NUM_PUBLIC_SLOTS, D) latest raw node
        snapshot per slot — the entity's current state unmixed by the
        recurrence, which outcome readouts need verbatim), and the whole
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

    def read_history_into_nodes(
        self,
        node_states: jax.Array,
        slot_states: jax.Array,
        field_state: jax.Array,
    ) -> jax.Array:
        """Per request, enrich each slot's current snapshot with a gated
        cross-read of the recurrent states: (T, 12, D) x (T, 12, D) x
        (T, D) -> (T, 12, D)."""
        return jax.vmap(self.history_node_read)(node_states, slot_states, field_state)

    def pool_history(
        self,
        slot_states: jax.Array,
        field_state: jax.Array,
        token_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Pools the per-request history states into learned latent
        summaries: (T, 12, D), (T, 3, D) -> (T, num_latents, D). token_mask
        (15,) optionally restricts which tokens are readable (constant
        across T)."""
        tokens = jnp.concatenate((slot_states, field_state), axis=-2)
        if token_mask is None:
            return jax.vmap(self.history_pool)(tokens)
        return jax.vmap(self.history_pool, in_axes=(0, None))(tokens, token_mask)

    def history_slot_sides(
        self, packed_history_step: PlayerPackedHistoryOutput
    ) -> jax.Array:
        """Relative side of the entity occupying each history slot
        (1 = mine, 0 = opponent's). Slots with no cache rows resolve to the
        int minimum and match neither side's mask."""
        slot_ids = packed_history_step.edge_cache[
            :, EntityEdgeFeature.ENTITY_EDGE_FEATURE__ENTITY_IDX
        ].clip(0, NUM_PUBLIC_SLOTS - 1)
        sides = packed_history_step.public_cache[
            :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE
        ]
        return jax.ops.segment_max(sides, slot_ids, num_segments=NUM_PUBLIC_SLOTS)

    def _history_inputs(
        self,
        env_step: PlayerEnvOutput,
        packed_history_step: PlayerPackedHistoryOutput,
        history_step: PlayerHistoryOutput,
        carry: HistoryCarry = HistoryCarry(),
    ):
        """The history pathway's four inputs to the sequence, in PUBLIC-ROW
        order: (row_states, order_valid, field_state, snapshot_rows), plus
        the per-step PerSlotHistoryOutput they were read from. The one
        place the slot-to-row alignment is written; offline reads call it
        directly.
        """
        slot_states, field_state, node_snapshots, history_output = self.encode_history(
            env_step, packed_history_step, history_step, carry
        )

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
        # The latest raw node snapshot per entity, same alignment -- the
        # TGN staleness fix the RL path used to discard (only the offline
        # critic read it; "the GRU-only readout loses the latest node").
        snapshot_rows = jnp.take_along_axis(node_snapshots, aligned_order, axis=1)
        return row_states, order_valid, field_state, snapshot_rows, history_output

    def assembled_sequence(
        self,
        env_step: PlayerEnvOutput,
        packed_history_step: PlayerPackedHistoryOutput,
        history_step: PlayerHistoryOutput,
    ):
        """The trunk's INPUT over time, (T, rows, width) with its row_valid --
        `_assemble_sequence` with the real history inputs and no trunk.
        Offline reads only (rl/offline/{trunk_homogeneity,separation_probe});
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
        sequence, row_valid, _ = assemble(self, env_step, *history_inputs)
        return sequence, row_valid

    def __call__(
        self,
        env_step: PlayerEnvOutput,
        packed_history_step: PlayerPackedHistoryOutput,
        history_step: PlayerHistoryOutput,
        carry: HistoryCarry = HistoryCarry(),
    ):
        # ((T, NUM_SEQUENCE_ROWS, entity_size), (T, 6, G, K), history
        # stats, history carry). The heads slice the rows they own by name
        # (rl/model/constants.py), so no offset is ever written twice; the
        # second element is the opponent code one-hot -- the belief head's
        # label -- riding out beside the sequence because it is computed
        # where the secret rows are built. The third is the per-trajectory
        # History-panel scalars (history_step_stats); the actor path drops
        # them, and XLA drops the computation with them. The fourth is the
        # post-window history state (history_carry_from), the actor's next
        # carry; the learner drops that one the same way.
        *history_inputs, history_output = self._history_inputs(
            env_step, packed_history_step, history_step, carry
        )
        sequence, opp_code_one_hot = _forward_vmap()(self, env_step, *history_inputs)
        return (
            sequence,
            opp_code_one_hot,
            history_step_stats(history_output),
            history_carry_from(history_output),
        )
