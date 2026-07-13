import functools
import math
from functools import partial

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from constants import MAX_RATIO_TOKEN
from rl.environment.data import (
    ACTION_MAX_VALUES,
    ALLY_SWITCH_INDICES,
    ALLY_TARGET_INDICES,
    ENEMY_TARGET_INDICES,
    ENTITY_EDGE_MAX_VALUES,
    ENTITY_PRIVATE_MAX_VALUES,
    ENTITY_PUBLIC_MAX_VALUES,
    FIELD_MAX_VALUES,
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
from rl.model.modules import (
    COLLECT_INTERMEDIATES,
    DO_CHECKPOINT,
    MLP,
    SumEmbeddings,
    TransformerDecoder,
    TransformerEncoder,
    one_hot_concat_jax,
)
from rl.model.world_model import NUM_PUBLIC_SLOTS, PerSlotWorldModel

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


def _binary_scale_encoding(
    to_encode: jax.Array, world_dim: int, dtype: jnp.dtype = jnp.float32
) -> jax.Array:
    """Encode the feature using its binary representation."""
    chex.assert_rank(to_encode, 0)
    chex.assert_type(to_encode, jnp.int32)
    num_bits = (world_dim - 1).bit_length()
    bit_mask = 1 << np.arange(num_bits)
    pos = jnp.broadcast_to(to_encode[jnp.newaxis], num_bits)
    result = jnp.not_equal(jnp.bitwise_and(pos, bit_mask), 0)
    return result.astype(dtype)


def _encode_one_hot(
    entity: jax.Array,
    feature_idx: int,
    max_values: dict[int, int],
    value_offset: int = 0,
) -> tuple[int, int]:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    return entity[feature_idx] + value_offset, max_values[feature_idx] + 1


def _encode_capped_one_hot(
    entity: jax.Array, feature_idx: int, max_values: dict[int, int]
) -> tuple[int, int]:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    max_value = max_values[feature_idx]
    return jnp.minimum(entity[feature_idx], max_value), max_value + 1


def _encode_sqrt_one_hot(
    entity: jax.Array,
    feature_idx: int,
    max_values: dict[int, int],
    dtype: jnp.dtype = jnp.int32,
) -> tuple[int, int]:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    max_value = max_values[feature_idx]
    max_sqrt_value = int(math.floor(math.sqrt(max_value)))
    x = jnp.floor(jnp.sqrt(entity[feature_idx].astype(dtype)))
    x = jnp.minimum(x.astype(jnp.int32), max_sqrt_value)
    return x, max_sqrt_value + 1


def _encode_divided_one_hot(
    entity: jax.Array, feature_idx: int, divisor: int, max_values: dict[int, int]
) -> tuple[int, int]:
    chex.assert_rank(entity, 1)
    chex.assert_type(entity, jnp.int32)
    max_value = max_values[feature_idx]
    max_divided_value = max_value // divisor
    x = jnp.floor_divide(entity[feature_idx], divisor)
    x = jnp.minimum(x, max_divided_value)
    return x, max_divided_value + 1


_encode_one_hot_public_entity = partial(
    _encode_one_hot, max_values=ENTITY_PUBLIC_MAX_VALUES
)
_encode_one_hot_private_entity = partial(
    _encode_one_hot, max_values=ENTITY_PRIVATE_MAX_VALUES
)
_encode_one_hot_action = partial(_encode_one_hot, max_values=ACTION_MAX_VALUES)
_encode_one_hot_edge = partial(_encode_one_hot, max_values=ENTITY_EDGE_MAX_VALUES)
_encode_one_hot_field = partial(_encode_one_hot, max_values=FIELD_MAX_VALUES)
_encode_sqrt_one_hot_public_entity = partial(
    _encode_sqrt_one_hot, max_values=ENTITY_PUBLIC_MAX_VALUES
)
_encode_sqrt_one_hot_action = partial(
    _encode_sqrt_one_hot, max_values=ACTION_MAX_VALUES
)
_encode_divided_one_hot_public_entity = partial(
    _encode_divided_one_hot, max_values=ENTITY_PUBLIC_MAX_VALUES
)
_encode_divided_one_hot_edge = partial(
    _encode_divided_one_hot, max_values=ENTITY_EDGE_MAX_VALUES
)


def get_public_entity_mask(revealed: jax.Array) -> jax.Array:
    """
    Generate a mask to identify valid entities based on species tokens.
    """
    species_token = revealed[
        EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
    ]
    return ~(
        (species_token == SpeciesEnum.SPECIES_ENUM___NULL)
        | (species_token == SpeciesEnum.SPECIES_ENUM___PAD)
        | (species_token == SpeciesEnum.SPECIES_ENUM___UNSPECIFIED)
    )


def get_private_entity_mask(private: jax.Array) -> jax.Array:
    """
    Generate a mask to identify valid entities based on species tokens.
    """
    species_token = private[
        EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SPECIES
    ]
    return ~(
        (species_token == SpeciesEnum.SPECIES_ENUM___NULL)
        | (species_token == SpeciesEnum.SPECIES_ENUM___PAD)
        | (species_token == SpeciesEnum.SPECIES_ENUM___UNSPECIFIED)
    )


def encode_boosts(boosts: jax.Array, offset: int):
    return jnp.where(
        boosts > 0,
        (offset + boosts) / offset,
        offset / (offset - boosts),
    )


def encode_reg_boosts(boosts: jax.Array):
    """Encodes according to https://bulbapedia.bulbagarden.net/wiki/Stat_modifier#Stage_multipliers"""
    return (1 / math.log(2)) * jnp.log(encode_boosts(boosts, 2))


def encode_spe_boosts(boosts: jax.Array):
    """Encodes according to https://bulbapedia.bulbagarden.net/wiki/Stat_modifier#Stage_multipliers"""
    return (2 / math.log(3)) * jnp.log(encode_boosts(boosts, 3))


def _mean_pool(x: jax.Array, m: jax.Array) -> jax.Array:
    """Mean pool over axis=0 with mask m (same shape as x)."""
    denom = m.sum(axis=0, keepdims=True).clip(min=1)
    return jnp.where(m, x, 0).sum(axis=0, keepdims=True) / denom


def _max_pool(x: jax.Array, m: jax.Array) -> jax.Array:
    """Masked max pool over axis=0; returns minimum dtype value where mask is False."""
    dtype_finfo = jnp.finfo(x.dtype)
    return m.any(axis=0, keepdims=True) * jnp.where(m, x, dtype_finfo.min).max(
        axis=0, keepdims=True
    )


class RoundBlock(nn.Module):
    """One trunk round, nn.scan-ned num_rounds times with stacked params.

    Self-attends the latents, cross-attends them to the per-entity history
    states, then lets each provenance group's action queries and the value
    queries cross-read the freshly updated latents. The carry holds the raw
    residual streams (latents, per-group action queries, value queries);
    masks and history context are broadcast across rounds. Scanned with
    variable_axes={"params": 0}, so every round has its own weights.
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(
        self,
        carry: tuple[jax.Array, tuple[jax.Array, ...], jax.Array],
        input_state_mask: jax.Array,
        history_context: jax.Array,
        history_mask: jax.Array,
        group_masks: tuple[jax.Array, ...],
    ):
        latent_queries, action_queries, value_queries = carry

        latent_queries = TransformerEncoder(
            name="latent_self", **self.cfg.latent_encoder.to_dict()
        )(qkv=latent_queries, qkv_mask=input_state_mask)
        latent_queries = TransformerDecoder(
            name="history_cross", **self.cfg.history_cross_decoder.to_dict()
        )(
            q=latent_queries,
            kv=history_context,
            q_mask=input_state_mask,
            kv_mask=history_mask,
        )

        # Information flow is one-way (latents never attend to action
        # queries), and each group reads with its own weights, so the
        # per-group decode-in-isolation property holds round by round.
        action_queries = tuple(
            TransformerDecoder(
                name=f"action_decoder_{group_name}",
                **self.cfg.action_decoder.to_dict(),
            )(
                q=queries,
                kv=latent_queries,
                q_mask=group_mask,
                kv_mask=input_state_mask,
            )
            for queries, group_mask, (group_name, _) in zip(
                action_queries, group_masks, ACTION_DECODER_SLOT_GROUPS
            )
        )

        value_queries = TransformerDecoder(
            name="value_decoder", **self.cfg.value_decoder.to_dict()
        )(q=value_queries, kv=latent_queries, kv_mask=input_state_mask)

        return (latent_queries, action_queries, value_queries), None


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

        # Initialize aggregation modules for combining feature embeddings.
        self.private_entity_sum = SumEmbeddings(
            output_size=entity_size, dtype=self.cfg.dtype, name="private_entity_sum"
        )
        self.public_entity_sum = SumEmbeddings(
            output_size=entity_size, dtype=self.cfg.dtype, name="public_entity_sum"
        )
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
        # read the state as of that request and let the latent sequence
        # cross-attend to it.
        self.world_model = PerSlotWorldModel(self.cfg, name="world_model")
        self.wm_field_step_linear = nn.Dense(
            name="wm_field_step_linear", use_bias=False, **dense_kwargs
        )

        # Per-modality input projections (replaces the single shared
        # input_seq_mlp): each input-token modality comes from a different
        # generative process (its own SumEmbeddings / linears upstream), so
        # each gets its own norm+MLP into the shared latent space. The
        # prev-action tokens especially need this — they are borrowed
        # mixed-provenance action-slot embeddings with only an additive bias.
        input_mlp_shape = (4 * self.entity_size, self.entity_size)
        self.input_norm_private = MLP(input_mlp_shape, name="input_norm_private")
        self.input_norm_public = MLP(input_mlp_shape, name="input_norm_public")
        self.input_norm_field = MLP(input_mlp_shape, name="input_norm_field")
        self.input_norm_prev_action = MLP(
            input_mlp_shape, name="input_norm_prev_action"
        )

        # Round trunk: one RoundBlock — (latent self-attention, history
        # cross-attention, per-group action cross-attention, value
        # cross-attention) — scanned num_rounds times with stacked params,
        # so every round has its own weights and rounds can specialize
        # instead of iterating one shared refinement operator. The cross
        # blocks' residual gates are zero-init, so every round's history
        # integration starts as a no-op.
        self.num_rounds = self.cfg.num_rounds
        round_block = RoundBlock
        if DO_CHECKPOINT or self.num_rounds > 1:
            round_block = nn.checkpoint(
                RoundBlock,
                policy=jax.checkpoint_policies.checkpoint_dots,
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
        # Head-facing output norms, hoisted out of the decoders (norm_output
        # is off) so the trunk carries raw residual streams; applied once to
        # the final round's action queries for the policy head.
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
        _encode_hex = jax.vmap(
            functools.partial(
                _binary_scale_encoding, dtype=self.cfg.dtype, world_dim=65535
            )
        )
        volatiles_indices = public[
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES0 : EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES8
            + 1
        ]
        volatiles_encoding = _encode_hex(volatiles_indices).reshape(-1)

        typechange_indices = public[
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE0 : EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE1
            + 1
        ]
        typechange_encoding = _encode_hex(typechange_indices).reshape(-1)

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

        encodings_list = [
            _encode_sqrt_one_hot_public_entity(
                public,
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__LEVEL,
                dtype=self.cfg.dtype,
            ),
            _encode_divided_one_hot_public_entity(
                public,
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO,
                MAX_RATIO_TOKEN / 32,
            ),
            _encode_one_hot_public_entity(
                public, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__GENDER
            ),
            _encode_one_hot_public_entity(
                public, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__STATUS
            ),
            _encode_one_hot_public_entity(
                public,
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ITEM_EFFECT,
            ),
            _encode_one_hot_public_entity(
                public,
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BEING_CALLED_BACK,
            ),
            _encode_one_hot_public_entity(
                public, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TRAPPED
            ),
            _encode_one_hot_public_entity(
                public,
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__NEWLY_SWITCHED,
            ),
            _encode_one_hot_public_entity(
                public,
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TOXIC_TURNS,
            ),
            _encode_one_hot_public_entity(
                public,
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SLEEP_TURNS,
            ),
            _encode_one_hot_public_entity(
                public, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED
            ),
        ]

        boolean_code = one_hot_concat_jax(encodings_list, dtype=self.cfg.dtype)

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

        public_encoding = jnp.concatenate(
            [
                jnp.concatenate(
                    [
                        boolean_code,
                        volatiles_encoding,
                        typechange_encoding,
                        hp_features,
                    ],
                    axis=-1,
                ),
                move_pp_onehot,
                self._embed_learnset(species_token),
                encode_reg_boosts(reg_boost_features),
                encode_spe_boosts(spe_boost_features),
                jax.nn.one_hot(
                    teratype_token, NUM_TYPECHART, dtype=move_embeddings.dtype
                ),
            ],
            axis=-1,
        )

        pos_bias = self.pos_bias(
            public[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE]
        )
        side_bias = self.side_bias(
            public[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE]
        )

        revealed_embedding = self.public_entity_sum(
            self._embed_species(species_token),
            self._embed_ability(ability_token),
            self._embed_item(item_token),
            move_embeddings.mean(axis=0),
            public_encoding,
            pos_bias,
            side_bias,
        )

        # Apply mask to filter out invalid entities.
        mask = get_public_entity_mask(revealed)

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
                _encode_one_hot_private_entity(
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

        private_embedding = self.private_entity_sum(
            self._embed_species(species_token),
            self._embed_ability(ability_token),
            self._embed_item(item_token),
            move_embeddings.mean(axis=0),
            boolean_code,
            stat_encoding,
        )

        # Apply mask to filter out invalid entities.
        mask = get_private_entity_mask(private)

        return private_embedding, mask

    def _embed_edge(self, edge: jax.Array):
        _encode_hex = jax.vmap(
            functools.partial(
                _binary_scale_encoding, world_dim=65535, dtype=self.cfg.dtype
            )
        )

        minor_args_indices = edge[
            EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG0 : EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG3
            + 1
        ]
        minor_args_encoding = _encode_hex(minor_args_indices).reshape(-1)

        # Aggregate embeddings for the relative edge.
        boolean_code = one_hot_concat_jax(
            [
                _encode_one_hot_edge(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG,
                ),
                _encode_divided_one_hot_edge(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__DAMAGE_RATIO,
                    MAX_RATIO_TOKEN / 32,
                ),
                _encode_divided_one_hot_edge(
                    edge,
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__HEAL_RATIO,
                    MAX_RATIO_TOKEN / 32,
                ),
                _encode_one_hot_edge(
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

        _encode_hex = jax.vmap(
            functools.partial(
                _binary_scale_encoding, world_dim=65535, dtype=self.cfg.dtype
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
        my_side_condition_encoding = _encode_hex(my_side_condition_indices).reshape(-1)
        opp_side_condition_encoding = _encode_hex(opp_side_condition_indices).reshape(
            -1
        )

        # Aggregate embeddings for the absolute edge.
        field_encoding = one_hot_concat_jax(
            [
                _encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__WEATHER_ID,
                ),
                _encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__WEATHER_MAX_DURATION,
                ),
                _encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__WEATHER_MIN_DURATION,
                ),
                _encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__TERRAIN_ID,
                ),
                _encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__TERRAIN_MAX_DURATION,
                ),
                _encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__TERRAIN_MIN_DURATION,
                ),
                _encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__PSEUDOWEATHER_ID,
                ),
                _encode_one_hot_field(
                    field,
                    FieldFeature.FIELD_FEATURE__PSEUDOWEATHER_MAX_DURATION,
                ),
                _encode_one_hot_field(
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
                        _encode_one_hot_field(
                            field,
                            FieldFeature.FIELD_FEATURE__MY_SPIKES,
                        ),
                        _encode_one_hot_field(
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
                        _encode_one_hot_field(
                            field,
                            FieldFeature.FIELD_FEATURE__OPP_SPIKES,
                        ),
                        _encode_one_hot_field(
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
                _encode_sqrt_one_hot_action(
                    action, MovesetFeature.MOVESET_FEATURE__PP, dtype=self.cfg.dtype
                ),
                _encode_sqrt_one_hot_action(
                    action, MovesetFeature.MOVESET_FEATURE__MAXPP, dtype=self.cfg.dtype
                ),
                _encode_one_hot_action(action, MovesetFeature.MOVESET_FEATURE__HAS_PP),
                _encode_one_hot_action(
                    action, MovesetFeature.MOVESET_FEATURE__DISABLED
                ),
                _encode_one_hot_action(
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
        wm_row_states: jax.Array,
        wm_row_valid: jax.Array,
        wm_field_state: jax.Array,
    ):
        (
            revealed_entity_embeddings,
            field_embeddings,
            revealed_entity_mask,
        ) = self._embed_public_entities(env_step)

        my_move_embeddings, my_move_mask = self._embed_moves(env_step.my_moveset)
        opp_move_embeddings, opp_move_mask = self._embed_moves(env_step.opp_moveset)

        my_move_mask = my_move_mask & jnp.take(
            env_step.action_mask, MOVE_INDICES, axis=0
        ).any(axis=-1)

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

        # Project each modality into the shared latent space with its own
        # norm+MLP before concatenating into one sequence.
        input_state_sequence = jnp.concatenate(
            (
                self.input_norm_private(private_entity_embeddings),
                self.input_norm_public(revealed_entity_embeddings),
                self.input_norm_field(field_embeddings),
                self.input_norm_prev_action(prev_action_tokens),
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

        input_state_mask = jnp.concatenate(
            (input_mask, prev_action_doubles_mask), axis=0
        )

        output_state_mask = env_step.action_mask.any(axis=0) | env_step.action_mask.any(
            axis=1
        )
        output_state_mask = output_state_mask & jnp.logical_not(env_step.done)

        # bulk of computation: the round trunk, scanned num_rounds times
        # with per-round (stacked) weights. Each round self-attends the
        # latent sequence, cross-attends it to the per-entity history states
        # (12 rows, PUBLIC_ORDER-aligned with the public team, masked to
        # mapped rows) plus the field history state, then lets each
        # provenance group's action queries and the value queries cross-read
        # the freshly updated latents.
        history_context = jnp.concatenate((wm_row_states, wm_field_state[None]), axis=0)
        history_mask = jnp.concatenate(
            (wm_row_valid, jnp.ones(1, dtype=jnp.bool_)), axis=0
        )

        latent_queries = input_state_sequence

        # Warm-start the action queries with their per-provenance input norms.
        action_queries = tuple(
            q_norm(output_state_sequence[slot_indices])
            for q_norm, (_, slot_indices) in zip(
                self.action_norms, ACTION_DECODER_SLOT_GROUPS
            )
        )
        group_masks = tuple(
            output_state_mask[slot_indices]
            for _, slot_indices in ACTION_DECODER_SLOT_GROUPS
        )

        # Value queries join the same scan: each round's value-decoder block
        # reads that round's latents and the query stream carries across
        # rounds, mirroring the action queries.
        value_queries = self.value_embeddings.astype(self.cfg.dtype)

        (_, action_queries, value_queries), _ = self.round_trunk(
            (latent_queries, action_queries, value_queries),
            input_state_mask,
            history_context,
            history_mask,
            group_masks,
        )

        # Head-facing embeddings from the final round's raw residual streams:
        # the per-group out-norms (hoisted out of the decoders) norm the
        # action queries and scatter them into the full action-slot layout.
        action_embeddings = jnp.zeros_like(output_state_sequence)
        for out_norm, group_queries, (_, slot_indices) in zip(
            self.action_out_norms, action_queries, ACTION_DECODER_SLOT_GROUPS
        ):
            action_embeddings = action_embeddings.at[slot_indices].set(
                out_norm(group_queries)
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
        step_field_vec = self.wm_field_step_linear(
            step_field_embeddings.reshape(step_field_embeddings.shape[0], -1)
        )

        wm_output = self.world_model(
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
        wm_slot_states, wm_field_state = self.world_model.state_at_requests(
            wm_output, request_count
        )

        # World-model slots are keyed by the stable entity index that edges
        # carry (revelation order across both sides), while public team rows
        # are per-side and re-sorted actives-first every state. PUBLIC_ORDER
        # is the server-provided permutation between the two: row i of the
        # public team holds the pokemon in world-model slot public_order[i],
        # or -1 for unrevealed fillers (masked out of the cross-attention).
        public_order = env_step.info[
            ...,
            InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0 : InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11
            + 1,
        ]
        order_valid = (public_order >= 0) & (public_order < NUM_PUBLIC_SLOTS)
        wm_row_states = jnp.take_along_axis(
            wm_slot_states,
            public_order.clip(0, NUM_PUBLIC_SLOTS - 1)[..., None],
            axis=1,
        )

        action_embeddings, value_embeddings = jax.vmap(self._batched_forward)(
            env_step, wm_row_states, order_valid, wm_field_state
        )

        # wm_field_state (T, D) rides along for the latent opponent model:
        # consecutive snapshots bracket the transition each turn caused.
        return action_embeddings, value_embeddings, wm_field_state
