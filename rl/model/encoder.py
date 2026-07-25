import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from constants import MAX_RATIO_TOKEN
from rl.environment.data import (
    ALLY_SWITCH_INDICES,
    ALLY_TARGET_INDICES,
    ENEMY_TARGET_INDICES,
    MOVE_INDICES,
    NUM_ACTION_FEATURES,
    NUM_FROM_SOURCE_EFFECTS,
    NUM_MOVES,
    NUM_TYPECHART,
    ONEHOT_ENCODERS,
    PASS_INDICES,
    REGULAR_MOVE_INDICES,
    RESERVE_ENTITY_INDICES,
    TARGET_INDICES,
    WILDCARD_MOVE_INDICES,
)
from rl.environment.interfaces import (
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
from rl.model.features import (
    binary_scale_encoding,
    encode_divided_one_hot_edge,
    encode_divided_one_hot_public_entity,
    encode_one_hot_action,
    encode_one_hot_edge,
    encode_one_hot_field,
    encode_one_hot_private_entity,
    encode_one_hot_public_entity,
    encode_reg_boosts,
    encode_spe_boosts,
    encode_sqrt_one_hot_action,
    encode_sqrt_one_hot_public_entity,
    get_private_entity_mask,
    get_public_entity_mask,
)
from rl.model.history_encoder import NUM_PUBLIC_SLOTS, PerSlotHistoryEncoder
from rl.model.modules import (
    COLLECT_INTERMEDIATES,
    FFWMLP,
    MLP,
    MultiHeadAttention,
    SumEmbeddings,
    TransformerDecoder,
    TransformerEncoder,
    create_attention_mask,
    layer_norm,
    one_hot_concat_jax,
)

# Action-decoder slot groups, segregated by input provenance rather than by
# behavioural modality. Move slots (regular + wildcard) are move-feature-derived
# and share a decoder; switch slots are entity-derived — the reserve candidates
# (battle-switch tgt keys / preview srcs) plus the two ALLY_i_SWITCH srcs that
# carry the outgoing active entity; the remaining target and structural slots
# (ally/enemy targets, TARGET_*, pass, default) only ever act as bilinear keys.
# These three groups partition all NUM_ACTION_FEATURES slots.
_MOVE_SLOTS = np.asarray(MOVE_INDICES)
_SWITCH_SLOTS = np.concatenate(
    [np.asarray(RESERVE_ENTITY_INDICES), np.asarray(ALLY_SWITCH_INDICES)]
)
_TARGET_STATIC_SLOTS = np.setdiff1d(
    np.arange(NUM_ACTION_FEATURES),
    np.concatenate([_MOVE_SLOTS, _SWITCH_SLOTS]),
)

# (name, static slot indices) per decoder, used to gather/scatter action embeddings.
ACTION_DECODER_SLOT_GROUPS = (
    ("move", _MOVE_SLOTS),
    ("switch", _SWITCH_SLOTS),
    ("target", _TARGET_STATIC_SLOTS),
)

# Intra-entity attribute-token types: rows of the shared token-type bias
# table, giving the (otherwise permutation-invariant) intra-entity attention
# a field identity per token. The four moves share one type — movesets are
# unordered. Public entities carry two state tokens: a persistent one
# (survives switching: hp, status, level, ...) and an active-only one
# (volatiles, boosts, typechange, trapped, ...) that is masked out for
# benched entities, so "not applicable" is an absent token rather than a
# default-valued vector.
_TOKEN_SPECIES = 0
_TOKEN_ABILITY = 1
_TOKEN_ITEM = 2
_TOKEN_MOVE = 3
_TOKEN_LEARNSET = 4
_TOKEN_PUBLIC_STATE = 5
_TOKEN_ACTIVE_STATE = 6
_TOKEN_PRIVATE_STATE = 7
_NUM_TOKEN_TYPES = 8

_PUBLIC_TOKEN_TYPES = np.array(
    [_TOKEN_SPECIES, _TOKEN_ABILITY, _TOKEN_ITEM]
    + 4 * [_TOKEN_MOVE]
    + [_TOKEN_LEARNSET, _TOKEN_PUBLIC_STATE, _TOKEN_ACTIVE_STATE]
)
_PRIVATE_TOKEN_TYPES = np.array(
    [_TOKEN_SPECIES, _TOKEN_ABILITY, _TOKEN_ITEM]
    + 4 * [_TOKEN_MOVE]
    + [_TOKEN_PRIVATE_STATE]
)


class EntityAttentionPool(nn.Module):
    """Pool one entity's attribute tokens into a single entity vector.

    Self-attends the (num_tokens, entity_size) set, then reads it out with a
    single learned query. Invalid attributes (unrevealed, or the active-only
    state token on a benched entity) are masked tokens; a fully masked set
    pools to zeros. The token-type bias gives the permutation-invariant
    attention a field identity per token.
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(
        self, tokens: jax.Array, token_mask: jax.Array, token_types: jax.Array
    ) -> jax.Array:
        embedding_init = nn.initializers.variance_scaling(
            1.0, "fan_in", "normal", out_axis=0
        )
        token_bias = self.param(
            "token_bias",
            nn.initializers.zeros_init(),
            (_NUM_TOKEN_TYPES, tokens.shape[-1]),
        )
        pool_query = self.param("pool_query", embedding_init, (1, tokens.shape[-1]))

        tokens = tokens + token_bias[token_types].astype(tokens.dtype)
        tokens = TransformerEncoder(
            name="attention", **self.cfg.intra_entity_encoder.to_dict()
        )(qkv=tokens, qkv_mask=token_mask)
        pooled = TransformerDecoder(
            name="pool", **self.cfg.intra_entity_pool.to_dict()
        )(
            q=pool_query.astype(tokens.dtype),
            kv=tokens,
            kv_mask=token_mask,
        )
        return jnp.squeeze(pooled, axis=0)


class RoundBlock(nn.Module):
    """One trunk round over the unified [state | action | value] sequence.

    Canonical decoder-layer shape: masked self-attention, a gated
    cross-read of the world-model history states, then one wide FFW — the
    attention sublayers are attention-only, so a round carries a single
    FFW instead of one per block. Information-flow rules live in the
    attention mask rather than in module topology: state tokens attend
    state only (never the action or value streams), action tokens attend
    state + actions (options can compare with each other), value tokens
    attend state only. Every residual write is behind a zero-init gate so
    a round starts as a no-op; nn.scan-ned num_rounds times with stacked
    params so every round has its own weights.
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(
        self,
        seq: jax.Array,
        attn_mask: jax.Array,
        seq_valid: jax.Array,
        history_context: jax.Array,
        history_mask: jax.Array,
    ):
        rcfg = self.cfg.round
        mha_kwargs = dict(
            num_heads=rcfg.num_heads,
            qk_size=rcfg.qk_size,
            v_size=rcfg.v_size,
            model_size=rcfg.model_size,
            qk_layer_norm=rcfg.qk_layer_norm,
            use_bias=rcfg.use_bias,
            dtype=seq.dtype,
            collect_intermediates=COLLECT_INTERMEDIATES,
        )

        def gate(name: str) -> jax.Array:
            return self.param(name, nn.initializers.zeros_init(), (1,)).astype(
                seq.dtype
            )

        seq_ln = layer_norm(seq)
        self_attn = MultiHeadAttention(name="self_attn", **mha_kwargs)(
            q=seq_ln, kv=seq_ln, mask=attn_mask
        )
        seq = seq + gate("self_gate") * self_attn

        # Every row (state, action and value tokens alike) reads the
        # per-entity recurrent history states directly.
        history_attn = MultiHeadAttention(name="history_cross", **mha_kwargs)(
            q=layer_norm(seq),
            kv=layer_norm(history_context),
            mask=create_attention_mask(seq_valid, history_mask),
        )
        seq = seq + gate("history_gate") * history_attn

        ffw = FFWMLP(hidden_size=rcfg.hidden_size, use_bias=rcfg.use_bias)(
            layer_norm(seq)
        )
        seq = seq + gate("ffw_gate") * ffw

        # Hard-zero invalid rows so padded tokens never accumulate content.
        seq = jnp.where(seq_valid[..., None], seq, 0)
        return seq, None


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

        self.side_bias = nn.Embed(2, name="side_bias", **embed_kwargs)
        self.pos_bias = nn.Embed(3, name="pos_bias", **embed_kwargs)

        self.pass_embeddings = self.param(
            "pass_embeddings", embedding_init, (2, entity_size)
        )
        self.target_embeddings = self.param(
            "target_embeddings", embedding_init, (len(TARGET_INDICES), entity_size)
        )
        self.prev_action_src_bias = self.param(
            "prev_action_src_bias", embedding_init, (1, entity_size)
        )
        self.prev_action_tgt_bias = self.param(
            "prev_action_tgt_bias", embedding_init, (1, entity_size)
        )

        # Action biases
        bias_init = nn.initializers.zeros_init()
        self.regular_move_bias = self.param(
            "regular_move_bias", bias_init, (1, entity_size)
        )
        self.wildcard_move_bias = self.param(
            "wildcard_move_bias", bias_init, (1, entity_size)
        )
        self.switch_src_bias = self.param(
            "switch_src_bias", bias_init, (1, entity_size)
        )
        self.switch_tgt_bias = self.param(
            "switch_tgt_bias", bias_init, (1, entity_size)
        )
        self.ally_target_bias = self.param(
            "ally_target_bias", bias_init, (1, entity_size)
        )
        self.enemy_target_bias = self.param(
            "enemy_target_bias", bias_init, (1, entity_size)
        )

        self.value_embeddings = self.param(
            "value_embeddings", embedding_init, (4, entity_size)
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

        # Intra-entity attention, shared between private and public entities:
        # each entity is a short set of attribute tokens, a small
        # self-attention block forms within-entity interactions (species x
        # item x moveset, boosts x stats, ...) that a linear sum cannot
        # express, and a single learned query pools the set back to one
        # entity vector. Token provenance is carried by the token-type bias
        # table; per-provenance input norms downstream keep the two entity
        # kinds separable. Rematted with nothing_saveable (not the house
        # checkpoint_dots, which saves the very matmul outputs that blow up):
        # the block runs per entity token-set, including the 2 * NUM_HISTORY
        # rows of the packed history cache, so storing its internals for the
        # backward pass OOMs the train step, while recomputing a ~10-token
        # block is cheap.
        self.entity_attention_pool = nn.checkpoint(
            EntityAttentionPool,
            policy=jax.checkpoint_policies.nothing_saveable,
        )(self.cfg, name="entity_attention_pool")
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

        # Recurrent history encoder over history edges. Twelve GRU states
        # (one per public slot) scanned along the history axis; per request we
        # read the state as of that request and let every trunk round
        # cross-attend to it.
        self.history_encoder = PerSlotHistoryEncoder(self.cfg, name="history_encoder")
        self.history_field_step_linear = nn.Dense(
            name="history_field_step_linear", use_bias=False, **dense_kwargs
        )

        # Per-modality input projections: each input-token modality comes
        # from a different generative process (its own SumEmbeddings /
        # linears upstream), so each gets its own norm+MLP into the shared
        # trunk space. The prev-action tokens especially need this — they
        # are borrowed mixed-provenance action-slot embeddings with only an
        # additive bias.
        input_mlp_shape = (4 * self.entity_size, self.entity_size)
        self.input_norm_private = MLP(input_mlp_shape, name="input_norm_private")
        self.input_norm_public = MLP(input_mlp_shape, name="input_norm_public")
        self.input_norm_field = MLP(input_mlp_shape, name="input_norm_field")
        self.input_norm_prev_action = MLP(
            input_mlp_shape, name="input_norm_prev_action"
        )
        # Moveset tokens carry the per-move battle state (pp, disabled,
        # wildcard availability) that the entity move tokens (move-ID only)
        # lack; without them the trunk — and therefore the value estimate —
        # is blind to pp, locks and spent tera.
        self.input_norm_my_moves = MLP(input_mlp_shape, name="input_norm_my_moves")
        self.input_norm_opp_moves = MLP(input_mlp_shape, name="input_norm_opp_moves")

        # Round trunk: one RoundBlock over the unified
        # [state | action | value] sequence, scanned num_rounds times with
        # stacked params, so every round has its own weights and rounds can
        # specialize instead of iterating one shared refinement operator.
        # All residual gates are zero-init, so each round starts as a no-op.
        # Rematted with nothing_saveable — checkpoint_dots would save the
        # very matmul outputs (the wide FFW hiddens) that dominate trunk
        # activation memory, while recomputing a round on the backward pass
        # is cheap.
        self.num_rounds = self.cfg.num_rounds
        round_block = nn.checkpoint(
            RoundBlock,
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        variable_axes = {"params": 0}
        if COLLECT_INTERMEDIATES:
            variable_axes["intermediates"] = 0
        self.round_trunk = nn.scan(
            round_block,
            variable_axes=variable_axes,
            variable_broadcast=False,
            split_rngs={"params": True},
            in_axes=nn.broadcast,
            length=self.num_rounds,
        )(self.cfg, name="round_trunk")
        # Per-provenance action-query warm starts (applied once, before the
        # round scan): each group's borrowed slot embeddings get their own
        # norm+MLP into the query space.
        self.action_norms = [
            MLP(
                (4 * self.entity_size, self.entity_size),
                name=f"action_norm_{group_name}",
            )
            for group_name, _ in ACTION_DECODER_SLOT_GROUPS
        ]
        # Head-facing output norms, hoisted out of the trunk so it carries
        # raw residual streams; applied once to the final round's action
        # tokens, keeping the move/switch/target slots in their own spaces
        # for the pointer and macro heads.
        self.action_out_norms = [
            MLP(name=f"action_out_norm_{group_name}")
            for group_name, _ in ACTION_DECODER_SLOT_GROUPS
        ]

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

    def _embed_public_entity(self, public: jax.Array, revealed: jax.Array):
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

        revealed_embedding = (
            self.entity_attention_pool(tokens, token_mask, _PUBLIC_TOKEN_TYPES)
            + pos_bias
            + side_bias
        )

        return revealed_embedding, mask

    def _embed_private_entity(self, private: jax.Array, num_stat_bands: int = 8):
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

        stat_encoding = stat_features / np.array([714, 526, 658, 535, 658, 548])
        freqs = 2.0 ** np.arange(num_stat_bands) * np.pi
        stat_encoding = (stat_encoding[..., None] * freqs[None]).astype(self.cfg.dtype)
        stat_encoding = jnp.concatenate(
            (jnp.sin(stat_encoding), jnp.cos(stat_encoding)),
            axis=-1,
        ).reshape(-1)

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
                    jnp.concatenate((boolean_code, stat_encoding), axis=-1)
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

        private_embedding = self.entity_attention_pool(
            tokens, token_mask, _PRIVATE_TOKEN_TYPES
        )

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

        # Aggregate embeddings for the relative edge.
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
        # Compute turn and request count differences for encoding.

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

        # Aggregate embeddings for the absolute edge.
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

        pos_biases = self.pos_bias.embedding.astype(field_embedding.dtype)
        field_embeddings = jnp.stack(
            (
                field_embedding,
                my_field_embedding + pos_biases[1],
                opp_field_embedding + pos_biases[0],
            )
        )
        return field_embeddings, mask, request_count, turn_order_value

    def _embed_public_entities(
        self, env_step: PlayerEnvOutput
    ) -> tuple[jax.Array, jax.Array]:
        revealed_entity_embedding, mask = jax.vmap(self._embed_public_entity)(
            env_step.public_team, env_step.revealed_team
        )
        field_embeddings, *_ = self._embed_field(env_step.field)

        return (revealed_entity_embedding, field_embeddings, mask)

    def _embed_private_entities(self, private_team: jax.Array):
        return jax.vmap(self._embed_private_entity)(private_team)

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
        return jax.vmap(self._embed_action)(moveset)

    def _batched_forward(
        self,
        env_step: PlayerEnvOutput,
        history_row_states: jax.Array,
        history_row_valid: jax.Array,
        history_field_state: jax.Array,
    ):
        (
            revealed_entity_embeddings,
            field_embeddings,
            revealed_entity_mask,
        ) = self._embed_public_entities(env_step)

        # Existence masks (move revealed / set), not action legality: a
        # disabled or pp-locked move is still state the trunk should see.
        my_move_embeddings, my_move_mask = self._embed_moves(env_step.my_moveset)
        opp_move_embeddings, opp_move_mask = self._embed_moves(env_step.opp_moveset)

        private_entity_embeddings, private_entity_mask = self._embed_private_entities(
            env_step.private_team
        )

        input_mask = jnp.concatenate(
            (
                private_entity_mask,
                revealed_entity_mask,
                jnp.ones_like(field_embeddings[..., 0], dtype=jnp.bool),
            ),
            axis=-1,
        )

        output_state_sequence = jnp.zeros(
            (NUM_ACTION_FEATURES, self.entity_size), dtype=self.cfg.dtype
        )

        # define positional biases
        pass_embeddings = self.pass_embeddings.astype(self.cfg.dtype)
        target_embeddings = self.target_embeddings.astype(self.cfg.dtype)

        # set/accumulate ally embeddings and positional biases
        for indices, accumulator in [
            (MOVE_INDICES, my_move_embeddings),
            (RESERVE_ENTITY_INDICES, private_entity_embeddings[:6]),
            (ALLY_SWITCH_INDICES, revealed_entity_embeddings[:2]),
            (ALLY_TARGET_INDICES, revealed_entity_embeddings[:2]),
            (ENEMY_TARGET_INDICES, revealed_entity_embeddings[6:8]),
            (PASS_INDICES, pass_embeddings),
            (TARGET_INDICES, target_embeddings),
        ]:
            output_state_sequence = output_state_sequence.at[indices].add(accumulator)

        # Add modality biases. Battle switches read (ALLY_i_SWITCH src,
        # RESERVE_j tgt): the src carries the outgoing active entity, the
        # reserve slots carry the incoming candidates.
        for indices, accumulator in [
            (REGULAR_MOVE_INDICES, self.regular_move_bias.astype(self.cfg.dtype)),
            (WILDCARD_MOVE_INDICES, self.wildcard_move_bias.astype(self.cfg.dtype)),
            (ALLY_SWITCH_INDICES, self.switch_src_bias.astype(self.cfg.dtype)),
            (RESERVE_ENTITY_INDICES, self.switch_tgt_bias.astype(self.cfg.dtype)),
            (ALLY_TARGET_INDICES, self.ally_target_bias.astype(self.cfg.dtype)),
            (ENEMY_TARGET_INDICES, self.enemy_target_bias.astype(self.cfg.dtype)),
        ]:
            output_state_sequence = output_state_sequence.at[indices].add(accumulator)

        prev_action_src = jnp.take(
            output_state_sequence,
            env_step.info[InfoFeature.INFO_FEATURE__PREV_ACTION_SRC],
            axis=0,
        )
        prev_action_tgt = jnp.take(
            output_state_sequence,
            env_step.info[InfoFeature.INFO_FEATURE__PREV_ACTION_TGT],
            axis=0,
        )

        prev_action_tokens = jnp.concatenate(
            (
                prev_action_src + self.prev_action_src_bias.astype(self.cfg.dtype),
                prev_action_tgt + self.prev_action_tgt_bias.astype(self.cfg.dtype),
            ),
            axis=0,
        )

        # Project each input modality into the shared trunk space with its
        # own norm+MLP before concatenating into one sequence. The moveset
        # tokens (mine and the opponent's revealed set) are state content,
        # distinct from the move action slots below: they let the state
        # stream — and through it the value estimate — see pp, disabled
        # moves and wildcard availability.
        state_sequence = jnp.concatenate(
            (
                self.input_norm_private(private_entity_embeddings),
                self.input_norm_public(revealed_entity_embeddings),
                self.input_norm_field(field_embeddings),
                self.input_norm_prev_action(prev_action_tokens),
                self.input_norm_my_moves(my_move_embeddings),
                self.input_norm_opp_moves(opp_move_embeddings),
            ),
            axis=0,
        )

        prev_action_doubles_mask = jnp.array(
            [
                env_step.info[InfoFeature.INFO_FEATURE__HAS_PREV_ACTION],
                env_step.info[InfoFeature.INFO_FEATURE__HAS_PREV_ACTION],
            ],
            dtype=jnp.bool,
        )

        state_mask = jnp.concatenate(
            (input_mask, prev_action_doubles_mask, my_move_mask, opp_move_mask),
            axis=0,
        )

        output_state_mask = env_step.action_mask.any(axis=0) | env_step.action_mask.any(
            axis=1
        )
        output_state_mask = output_state_mask & jnp.logical_not(env_step.done)

        # Per-entity recurrent history (12 rows, PUBLIC_ORDER-aligned with
        # the public team, masked to mapped rows) plus the field history
        # state; every trunk round cross-reads it.
        history_context = jnp.concatenate(
            (history_row_states, history_field_state[None]), axis=0
        )
        history_mask = jnp.concatenate(
            (history_row_valid, jnp.ones(1, dtype=jnp.bool_)), axis=0
        )

        # Warm-start the action tokens with their per-provenance input norms.
        action_tokens = jnp.zeros_like(output_state_sequence)
        for q_norm, (_, slot_indices) in zip(
            self.action_norms, ACTION_DECODER_SLOT_GROUPS
        ):
            action_tokens = action_tokens.at[slot_indices].set(
                q_norm(output_state_sequence[slot_indices])
            )
        value_tokens = self.value_embeddings.astype(self.cfg.dtype)

        # Unified trunk sequence [state | action | value]. Information-flow
        # rules are declarative — one static block mask instead of
        # per-stream decoder topology: state rows attend state only (they
        # never read the action or value streams), action rows attend
        # state + actions (options can compare with each other), value rows
        # attend state only. Combined with per-token validity for keys.
        n_state = state_sequence.shape[0]
        n_action = action_tokens.shape[0]
        n_value = value_tokens.shape[0]

        seq = jnp.concatenate((state_sequence, action_tokens, value_tokens), axis=0)
        seq_valid = jnp.concatenate(
            (state_mask, output_state_mask, jnp.ones(n_value, dtype=jnp.bool_)),
            axis=0,
        )

        allowed = np.zeros((n_state + n_action + n_value,) * 2, dtype=bool)
        allowed[:n_state, :n_state] = True
        allowed[n_state : n_state + n_action, : n_state + n_action] = True
        allowed[n_state + n_action :, :n_state] = True
        attn_mask = allowed[None] & create_attention_mask(seq_valid, seq_valid)

        # Bulk of computation: the round trunk, scanned num_rounds times
        # with per-round (stacked) weights.
        seq, _ = self.round_trunk(
            seq, attn_mask, seq_valid, history_context, history_mask
        )

        action_queries = seq[n_state : n_state + n_action]
        value_queries = seq[n_state + n_action :]

        # Head-facing embeddings from the final round's raw residual
        # streams: the per-group out-norms (hoisted out of the trunk) keep
        # the move/switch/target slots in their own spaces for the pointer
        # and macro heads.
        action_embeddings = jnp.zeros_like(output_state_sequence)
        for out_norm, (_, slot_indices) in zip(
            self.action_out_norms, ACTION_DECODER_SLOT_GROUPS
        ):
            action_embeddings = action_embeddings.at[slot_indices].set(
                out_norm(action_queries[slot_indices])
            )
        value_embeddings = value_queries.reshape(-1)

        # (NUM_ACTION_FEATURES, entity_size) and (4 * entity_size,); the
        # final round drives the acting policy and value estimate.
        return action_embeddings, value_embeddings

    def __call__(
        self,
        env_step: PlayerEnvOutput,
        packed_history_step: PlayerPackedHistoryOutput,
        history_step: PlayerHistoryOutput,
    ):
        # --- Recurrent world model over the shared trajectory history ---
        # Embed the packed (entity snapshot, edge) cache once; both are shared
        # across every request of the trajectory.
        node_embedding_cache, _ = jax.vmap(self._embed_public_entity)(
            packed_history_step.public_cache, packed_history_step.revealed_cache
        )
        edge_embedding_cache, _ = jax.vmap(self._embed_edge)(
            packed_history_step.edge_cache
        )
        edge_slot_ids = packed_history_step.edge_cache[
            :, EntityEdgeFeature.ENTITY_EDGE_FEATURE__ENTITY_IDX
        ]

        # One pooled field vector per history step from the (field, my-side,
        # opp-side) token triple.
        (
            step_field_embeddings,
            step_valid,
            step_request_count,
            _,
        ) = jax.vmap(
            self._embed_field
        )(history_step.field)
        step_field_vec = self.history_field_step_linear(
            step_field_embeddings.reshape(step_field_embeddings.shape[0], -1)
        )

        history_output = self.history_encoder(
            history_field=history_step.field,
            node_embedding_cache=node_embedding_cache,
            edge_embedding_cache=edge_embedding_cache,
            edge_slot_ids=edge_slot_ids,
            field_step_embeddings=step_field_vec,
            step_request_count=step_request_count,
            step_valid=step_valid.squeeze(-1),
        )

        # Read the recurrent state as of each request: the snapshot after the
        # last history step whose request_count <= the request's.
        request_count = env_step.info[..., InfoFeature.INFO_FEATURE__REQUEST_COUNT]
        slot_states, field_state = self.history_encoder.state_at_requests(
            history_output, request_count
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
        row_states = jnp.take_along_axis(
            slot_states,
            public_order.clip(0, NUM_PUBLIC_SLOTS - 1)[..., None],
            axis=1,
        )

        action_embeddings, value_embeddings = jax.vmap(self._batched_forward)(
            env_step, row_states, order_valid, field_state
        )

        return action_embeddings, value_embeddings
