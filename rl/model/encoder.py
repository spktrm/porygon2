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
    MOVE_SLOT_INDICES,
    NUM_ACTION_FEATURES,
    NUM_FROM_SOURCE_EFFECTS,
    NUM_MOVES,
    NUM_TYPECHART,
    ONEHOT_ENCODERS,
    PASS_INDICES,
    REGULAR_MOVE_INDICES,
    RESERVE_ENTITY_INDICES,
    SWITCH_SLOT_INDICES,
    TARGET_INDICES,
    TARGET_SLOT_INDICES,
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
from rl.model.history_encoder import (
    NUM_PUBLIC_SLOTS,
    HistoryAttentionPool,
    NodeHistoryRead,
    PerSlotHistoryEncoder,
    mask_outcome_features,
)
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

# Typed action-slot groups (canonical partition lives in
# rl/environment/data.py next to the modality masks): move slots are
# move-feature-derived, switch slots entity-derived, target/structural
# slots key-only. Since 2026-08-17 the groups are not just decoder
# bookkeeping — each is its own residual stream through the round trunk.
_MOVE_SLOTS = np.asarray(MOVE_SLOT_INDICES)
_SWITCH_SLOTS = np.asarray(SWITCH_SLOT_INDICES)
_TARGET_STATIC_SLOTS = np.asarray(TARGET_SLOT_INDICES)


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
    return nn.jit(
        nn.vmap(
            method,
            in_axes=0,
            out_axes=0,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )
    )


# (name, static slot indices) per decoder, used to gather/scatter action embeddings.
ACTION_DECODER_SLOT_GROUPS = (
    ("move", _MOVE_SLOTS),
    ("switch", _SWITCH_SLOTS),
    ("target", _TARGET_STATIC_SLOTS),
)
# Slot-aligned indices of the concatenated action stream's rows
# ([move | switch | target] order) and the split points between groups —
# the gather on the way into the trunk and the scatter back out are the
# same permutation.
ACTION_GROUP_SLOTS = np.concatenate(
    [slot_indices for _, slot_indices in ACTION_DECODER_SLOT_GROUPS]
)
ACTION_GROUP_SPLITS = np.cumsum(
    [len(slot_indices) for _, slot_indices in ACTION_DECODER_SLOT_GROUPS]
)[:-1]

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
# Non-entity tokens of the flat input set the latent read consumes
# (2026-08-21): field / side conditions, the two prev-action slots, the
# per-slot recurrent history states and the field history state, and the
# public latents themselves when the privileged read re-reads them.
_TOKEN_FIELD = 8
_TOKEN_PREV_ACTION = 9
_TOKEN_HISTORY_SLOT = 10
_TOKEN_HISTORY_FIELD = 11
_TOKEN_LATENT = 12
_NUM_TOKEN_TYPES = 13

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
_FIELD_TOKEN_TYPES = np.array(3 * [_TOKEN_FIELD])
_PREV_ACTION_TOKEN_TYPES = np.array(2 * [_TOKEN_PREV_ACTION])
_HISTORY_TOKEN_TYPES = np.array(
    NUM_PUBLIC_SLOTS * [_TOKEN_HISTORY_SLOT] + [_TOKEN_HISTORY_FIELD]
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


class LatentInputRead(nn.Module):
    """Perceiver-style input read (2026-08-21): a learned latent array
    cross-attends ONE flat set of input tokens and becomes the trunk's state
    rows. Replaces the per-entity pooling + per-substream input MLPs: the
    current board used to collapse to one vector per entity (then one
    projection per substream) before the trunk ever saw it, so a matchup was
    something the trunk reconstructed from pooled summaries; here every
    attribute token (species / ability / item / moves / state), the field,
    the prev-action slots and the raw recurrent history states are keys of
    the same read, and the latents decide what to keep.

    Identity is purely additive -- there is no substream slicing left to
    carry it. Every token gets, before the read: the caller's per-entity
    bias (the public embedder's pos + side, the "mine"/"theirs" side bias for
    the sheets), a per-token-TYPE bias, a zero-init per-ROW bias (public rows
    are side-partitioned actives-first, so a row index is a rank identity
    worth learning) and a non-zero-init per-GROUP bias so e.g. a sheet row
    and a public row of the same mon are told apart from step 0.

    Groups arrive as (num_entities, num_tokens, dim) blocks with their own
    token-type ids; non-entity inputs are passed as ONE entity with N tokens
    (field: 1 x 3, prev-action: 1 x 2, history: 1 x 13). Masked tokens are
    absent keys, so a masked entity is inert. The read's residual starts at
    1.0 (cfg.latent_read.init_residual_scale): token content can only reach
    the latents through it, so it must not start as a no-op -- which also
    means the value-ladder leak tests exercise this path without gate
    opening. Privileged routing is STRUCTURAL: the caller builds two
    instances -- the public read over the player's own information set, and
    a small privileged read over [sheet tokens | public latents] whose output
    only the value-`all` rung ever reads.
    """

    cfg: ConfigDict
    num_latents: int

    @nn.compact
    def __call__(
        self,
        groups: tuple[jax.Array, ...],
        masks: tuple[jax.Array, ...],
        types: tuple[np.ndarray, ...],
        biases: tuple[jax.Array | None, ...] | None = None,
    ) -> jax.Array:
        embedding_init = nn.initializers.variance_scaling(
            1.0, "fan_in", "normal", out_axis=0
        )
        dtype = groups[0].dtype
        model_size = groups[0].shape[-1]
        num_entities = sum(group.shape[0] for group in groups)

        token_bias = self.param(
            "token_bias",
            nn.initializers.zeros_init(),
            (_NUM_TOKEN_TYPES, model_size),
        )
        entity_bias = self.param(
            "entity_bias", nn.initializers.zeros_init(), (num_entities, model_size)
        )
        group_bias = self.param("group_bias", embedding_init, (len(groups), model_size))
        latents = self.param("latents", embedding_init, (self.num_latents, model_size))
        if biases is None:
            biases = (None,) * len(groups)

        flat_tokens = []
        flat_masks = []
        offset = 0
        for group_index, (group, mask, token_types, bias) in enumerate(
            zip(groups, masks, types, biases, strict=True)
        ):
            num, num_tokens = group.shape[:2]
            entity_ids = np.arange(offset, offset + num)
            # token_types may arrive traced (nn.checkpoint traces its
            # args), so index with it directly rather than via numpy.
            group = (
                group.astype(dtype)
                + token_bias[token_types].astype(dtype)[None]
                + entity_bias[entity_ids].astype(dtype)[:, None]
                + group_bias[group_index].astype(dtype)
            )
            if bias is not None:
                group = group + bias.astype(dtype)[:, None]
            flat_tokens.append(group.reshape(num * num_tokens, model_size))
            flat_masks.append(mask.reshape(num * num_tokens))
            offset += num

        tokens = jnp.concatenate(flat_tokens, axis=0)
        token_mask = jnp.concatenate(flat_masks, axis=0)

        return TransformerDecoder(name="read", **self.cfg.latent_read.to_dict())(
            q=latents.astype(dtype), kv=tokens, kv_mask=token_mask
        )


class RoundBlock(nn.Module):
    """One trunk round over the public LATENTS (the state stream: K rows
    produced by LatentInputRead over the flat input token set, 2026-08-21
    -- before that, the concat [private | public | field | prev_action |
    history] of pooled per-entity vectors with per-substream gates), the
    privileged latents (opp), the concatenated ACTION stream
    [move | switch | target] and the value ladder. Action substream
    identity survives via per-substream input norms, static slice
    boundaries (derived from the typed valid masks) and PER-SUBSTREAM GATE
    VECTORS — each write to the action stream is scaled by its type's own
    zero-init scalar, broadcast over that type's rows; the state stream is
    one group with one gate per write. Per round:

        1. state self-attention: one module over the state concat,
           per-substream gate vector. Within-type pairs are a subset of
           this all-pairs attention, so there are no intra self-attn
           modules
        2. opp cross-read: the privileged latents read [state | opp]
           (one module) so the opponent sheet is CONTEXTUALISED against
           the live game each round. Nothing that feeds the policy ever reads opp (the one
           leak-critical rule — opp is consumed only by the value-`all`
           read in step 4)
        3. action self-attention: one module over the action concat,
           per-type gate vector — option comparison, within- and
           cross-type, in one all-pairs attention; then the EXCHANGE:
           state -> action decode (q = action, kv = state rows only,
           never opp), per-type gate vector, followed by action -> state
           decode (q = state, kv = the updated action), per-substream
           gate vector
        4. value-ladder reads. `all` and `private` share one fused read
           module with per-rung key masks (`all` sees [state | opp],
           `private` sees state only) but are otherwise INDEPENDENT
           estimators — separate query inits and separate residual gates
           (user decision 2026-08-16). `public` reads the RAW history
           inputs (raw_history / raw_history_valid, the pre-trunk
           recurrent embeddings) — NOT the state stream's history slice,
           which mixes with private tokens in the state self-attention;
           reading raw keeps the public rung's information set purely
           public-historical
        5. group-level FFWs: one state FFW, one action FFW (per-token,
           applied to the group stream under its per-substream gate
           vector); opp and the value rungs sit outside both groups and
           keep their own FFWs

    Every residual write stays behind a zero-init gate so a round starts
    as a no-op; nn.scan-ned num_rounds times with stacked params so every
    round has its own weights. A query row whose key set is entirely
    invalid (e.g. a terminal row, where no action slot is legal) receives
    a ZERO attention output, not NaN: MultiHeadAttention masks the
    attention probs back to 0 after the -1e9-masked softmax.
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(
        self,
        streams: tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
        state_valid: jax.Array,
        opp_valid: jax.Array,
        move_valid: jax.Array,
        switch_valid: jax.Array,
        target_valid: jax.Array,
        raw_history: jax.Array,
        raw_history_valid: jax.Array,
    ):
        state, opp, action, value_all, value_private, value_public = streams
        rcfg = self.cfg.round
        mha_kwargs = dict(
            num_heads=rcfg.num_heads,
            qk_size=rcfg.qk_size,
            v_size=rcfg.v_size,
            model_size=rcfg.model_size,
            qk_layer_norm=rcfg.qk_layer_norm,
            use_bias=rcfg.use_bias,
            dtype=state.dtype,
            collect_intermediates=COLLECT_INTERMEDIATES,
        )
        n_state = state.shape[0]
        n_opp = opp.shape[0]
        n_value = value_all.shape[0]
        value_valid = jnp.ones(n_value, dtype=jnp.bool_)

        # Substream boundaries come from the (static) valid-mask shapes;
        # the concat order is fixed by the encoder. The state stream is
        # one group (the latents).
        state_parts = (("state", state_valid.shape[0]),)
        action_parts = (
            ("move", move_valid.shape[0]),
            ("switch", switch_valid.shape[0]),
            ("target", target_valid.shape[0]),
        )
        if COLLECT_INTERMEDIATES:
            # Self-describing capture: the probe needs the concat layout to
            # attribute attention mass to substreams.
            self.sow(
                "intermediates",
                "state_part_sizes",
                jnp.asarray([n for _, n in state_parts], dtype=jnp.int32),
            )
            self.sow(
                "intermediates",
                "action_part_sizes",
                jnp.asarray([n for _, n in action_parts], dtype=jnp.int32),
            )

        action_valid = jnp.concatenate((move_valid, switch_valid, target_valid), axis=0)

        def gate(name: str) -> jax.Array:
            return self.param(name, nn.initializers.zeros_init(), (1,)).astype(
                state.dtype
            )

        def group_gate(parts, pattern: str) -> jax.Array:
            """Per-substream zero-init scalars broadcast to a (rows, 1)
            gate vector over the group concat."""
            return jnp.concatenate(
                [
                    jnp.broadcast_to(gate(pattern.format(name)), (n,))
                    for name, n in parts
                ]
            )[:, None]

        def attend(name: str, q, q_valid, kv, kv_valid, allowed=None):
            mask = create_attention_mask(q_valid, kv_valid)
            if allowed is not None:
                mask = mask & allowed[None]
            return MultiHeadAttention(name=name, **mha_kwargs)(
                q=layer_norm(q), kv=layer_norm(kv), mask=mask
            )

        # 1. State self-attention over the group concat, per-substream
        # gate vector.
        state = state + group_gate(state_parts, "{}_global_gate") * attend(
            "state_global_attn", state, state_valid, state, state_valid
        )

        # 2. Opp cross-read over [state | opp]; opp is never read by
        # anything policy-facing.
        opp = opp + gate("opp_read_gate") * attend(
            "opp_cross_attn",
            opp,
            opp_valid,
            jnp.concatenate((state, opp), axis=0),
            jnp.concatenate((state_valid, opp_valid), axis=0),
        )

        # 3. Action self-attention over the group concat (per-type gate
        # vector), then the state<->action exchange: state -> action
        # decode (kv = state rows only, never opp), action -> state
        # decode reading the updated options back.
        action = action + group_gate(action_parts, "{}_global_gate") * attend(
            "action_global_attn", action, action_valid, action, action_valid
        )
        action = action + group_gate(action_parts, "state_to_{}_gate") * attend(
            "state_to_action", action, action_valid, state, state_valid
        )
        state = state + group_gate(state_parts, "action_to_{}_gate") * attend(
            "action_to_state", state, state_valid, action, action_valid
        )

        # 4. Fused value-ladder read: one shared module, per-rung key
        # masks — `all` rows may read [state | opp]; `private` rows read
        # the state slice only. Separate residual gates per rung
        # (independent estimators, matching their separate query inits).
        state_opp = jnp.concatenate((state, opp), axis=0)
        state_opp_valid = jnp.concatenate((state_valid, opp_valid), axis=0)
        value_allowed = np.zeros((2 * n_value, n_state + n_opp), dtype=bool)
        value_allowed[:n_value, :] = True
        value_allowed[n_value:, :n_state] = True
        value_read = attend(
            "value_read",
            jnp.concatenate((value_all, value_private), axis=0),
            jnp.ones(2 * n_value, dtype=jnp.bool_),
            state_opp,
            state_opp_valid,
            allowed=value_allowed,
        )
        value_all = value_all + gate("value_all_read_gate") * value_read[:n_value]
        value_private = (
            value_private + gate("value_private_read_gate") * value_read[n_value:]
        )
        # `public` rung reads the RAW pre-trunk history embeddings, not
        # the state stream's history slice (which mixes with private
        # tokens in the state self-attention) — its information set must
        # stay purely public-historical.
        value_public = value_public + gate("value_public_read_gate") * attend(
            "history_to_value_public",
            value_public,
            value_valid,
            raw_history,
            raw_history_valid,
        )

        # 5. Group-level FFWs with per-substream gate vectors; opp and
        # the value rungs keep their own FFWs and scalar gates.
        def ffw(name: str):
            return FFWMLP(
                hidden_size=rcfg.hidden_size, use_bias=rcfg.use_bias, name=name
            )

        state = state + group_gate(state_parts, "{}_ffw_gate") * ffw("state_ffw")(
            layer_norm(state)
        )
        action = action + group_gate(action_parts, "{}_ffw_gate") * ffw("action_ffw")(
            layer_norm(action)
        )
        opp = opp + gate("opp_ffw_gate") * ffw("opp_ffw")(layer_norm(opp))
        value_all = value_all + gate("value_all_ffw_gate") * ffw("value_all_ffw")(
            layer_norm(value_all)
        )
        value_private = value_private + gate("value_private_ffw_gate") * ffw(
            "value_private_ffw"
        )(layer_norm(value_private))
        value_public = value_public + gate("value_public_ffw_gate") * ffw(
            "value_public_ffw"
        )(layer_norm(value_public))

        # Hard-zero invalid rows so padded tokens never accumulate content.
        state = jnp.where(state_valid[..., None], state, 0)
        opp = jnp.where(opp_valid[..., None], opp, 0)
        action = jnp.where(action_valid[..., None], action, 0)
        return (
            state,
            opp,
            action,
            value_all,
            value_private,
            value_public,
        ), None


class GroupNorm(nn.Module):
    """Per-substream norm+MLP projections at a trunk-group boundary:
    each named substream gets its own MLP, and the results are
    concatenated in order into (or back out of) the group stream
    RoundBlock carries. One class serves every boundary — the state and
    action INPUT norms (each substream comes from a different generative
    process, so each needs its own projection into trunk space) and the
    action OUTPUT norms (the head-facing per-group spaces over the final
    round's slices) — so all group-boundary projections are built
    identically. The substream order fixes the slice boundaries
    RoundBlock derives from the valid masks."""

    substream_names: tuple[str, ...]
    layer_sizes: tuple[int, ...] | None = None

    @nn.compact
    def __call__(self, substreams: tuple[jax.Array, ...]) -> jax.Array:
        return jnp.concatenate(
            [
                MLP(self.layer_sizes, name=f"{name}_norm")(tokens)
                for name, tokens in zip(self.substream_names, substreams, strict=True)
            ],
            axis=0,
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

        # Value-query groups, an information ladder (2026-08-16): `all`
        # reads state + the opponent's privileged team sheet, `private`
        # reads state only (the deployable baseline), `public` reads the
        # history context only. Each rung has its OWN query init (user
        # decision 2026-08-16, replacing the earlier shared-init variant):
        # rungs are independent estimators specialised to their
        # information route. The trade: the all-vs-private gap now
        # includes an estimator component and the empty-sheet all==private
        # equality no longer holds — read the ladder telemetry
        # accordingly.
        self.all_value_embeddings = self.param(
            "all_value_embeddings", embedding_init, (4, entity_size)
        )
        self.private_value_embeddings = self.param(
            "private_value_embeddings", embedding_init, (4, entity_size)
        )
        self.public_value_embeddings = self.param(
            "public_value_embeddings", embedding_init, (4, entity_size)
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
        # Perceiver-style input reads (2026-08-21). The PUBLIC read: K
        # learned latents cross-attend the flat set of every token in the
        # player's own information set -- both sides' public attribute
        # tokens, my private sheet's, the field, the prev-action slots and
        # the raw recurrent history states -- and become the trunk's state
        # rows; no per-entity pooling and no per-substream input MLPs on
        # this path any more. The PRIVILEGED read: a few latents over
        # [opp sheet tokens | public latents], consumed ONLY by the
        # value-`all` rung (RoundBlock's opp stream) -- the sheet never
        # enters the public read, so the policy's invariance to it is
        # structural. The entity-local pool above survives for the
        # history cache (~2 orders of magnitude more rows) and for the
        # per-entity vectors that warm-start the typed action slots.
        # Rematted like its neighbours; the read's probability matrix is
        # K x ~186 at the trunk's head count -- a sixth of the 168^2
        # cross-entity mix it replaces.
        latent_read = nn.checkpoint(
            LatentInputRead, policy=jax.checkpoint_policies.nothing_saveable
        )
        self.latent_input_read = latent_read(
            self.cfg, self.cfg.num_latents, name="latent_input_read"
        )
        self.priv_latent_read = latent_read(
            self.cfg, self.cfg.num_priv_latents, name="priv_latent_read"
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
        self.history_pool = HistoryAttentionPool(self.cfg, name="history_pool")
        self.history_node_read = NodeHistoryRead(self.cfg, name="history_node_read")
        self.history_field_step_linear = nn.Dense(
            name="history_field_step_linear", use_bias=False, **dense_kwargs
        )

        # Per-type input projections for the action stream (the state
        # stream is the latent read's output and needs none). The
        # prev-action tokens used to need theirs too -- they are borrowed
        # mixed-provenance action-slot embeddings -- and now get identity
        # from the read's token-type/group biases instead.
        input_mlp_shape = (4 * self.entity_size, self.entity_size)
        self.action_input_norm = GroupNorm(
            substream_names=tuple(
                group_name for group_name, _ in ACTION_DECODER_SLOT_GROUPS
            ),
            layer_sizes=input_mlp_shape,
            name="action_input_norm",
        )

        # Round trunk: one RoundBlock over the public latents (state), the
        # privileged latents (opp), the concatenated action stream
        # [move | switch | target], and the
        # value ladder, scanned num_rounds times with stacked params, so
        # every round has its own weights and rounds can specialize
        # instead of iterating one shared refinement operator.
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
        # Head-facing output norms, hoisted out of the trunk so it carries
        # raw residual streams; applied once to the final round's action
        # stream, keeping the move/switch/target slices in their own
        # spaces for the readout and macro heads. Same GroupNorm class as
        # the input side (identity-size MLPs).
        self.action_out_norm = GroupNorm(
            substream_names=tuple(
                group_name for group_name, _ in ACTION_DECODER_SLOT_GROUPS
            ),
            name="action_out_norm",
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
            self.entity_attention_pool(tokens, token_mask, _PUBLIC_TOKEN_TYPES) + bias
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

        return tokens, token_mask, mask

    def _embed_private_entity(self, private: jax.Array, num_stat_bands: int = 8):
        """Entity-LOCAL pooling -- see `_embed_public_entity`. Still the path
        for the opponent's privileged sheet, which must stay out of any
        policy-facing token set."""
        tokens, token_mask, mask = self._private_entity_tokens(private, num_stat_bands)
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

    def _pool_public_tokens(self, tokens: jax.Array, token_mask: jax.Array):
        return self.entity_attention_pool(tokens, token_mask, _PUBLIC_TOKEN_TYPES)

    def _pool_private_tokens(self, tokens: jax.Array, token_mask: jax.Array):
        return self.entity_attention_pool(tokens, token_mask, _PRIVATE_TOKEN_TYPES)

    def _current_entity_tokens(self, env_step: PlayerEnvOutput):
        """Attribute tokens of every entity of the CURRENT board -- both
        sides' public rows and my own private sheet -- plus the per-entity
        identity biases they carry into the latent read, plus the cheap
        entity-local pooled vectors that warm-start the typed action slots
        (a switch option must still be "this mon" as a query; the latents
        are not per-entity). The token set is exactly the player's own
        legal information set; `opp_private_team` is deliberately absent."""
        public_tokens, public_token_mask, public_mask, public_bias = (
            _lifted_entity_vmap(Encoder._public_entity_tokens)(
                self, env_step.public_team, env_step.revealed_team
            )
        )
        private_tokens, private_token_mask, private_mask = _lifted_entity_vmap(
            Encoder._private_entity_tokens
        )(self, env_step.private_team)
        private_bias = jnp.broadcast_to(
            self.side_bias(jnp.zeros((), dtype=jnp.int32)).astype(private_tokens.dtype),
            private_tokens.shape[:1] + private_tokens.shape[-1:],
        )
        public_vectors = (
            _lifted_entity_vmap(Encoder._pool_public_tokens)(
                self, public_tokens, public_token_mask
            )
            + public_bias
        )
        private_vectors = (
            _lifted_entity_vmap(Encoder._pool_private_tokens)(
                self, private_tokens, private_token_mask
            )
            + private_bias
        )
        return (
            (public_tokens, public_token_mask, public_mask, public_bias),
            (private_tokens, private_token_mask, private_mask, private_bias),
            public_vectors,
            private_vectors,
        )

    def _embed_private_entities(self, private_team: jax.Array):
        return _lifted_entity_vmap(Encoder._embed_private_entity)(self, private_team)

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

    def _batched_forward(
        self,
        env_step: PlayerEnvOutput,
        history_row_states: jax.Array,
        history_row_valid: jax.Array,
        history_field_state: jax.Array,
    ):
        # Attribute tokens of the current board (the latent read's keys)
        # and the entity-local pooled vectors that warm-start the typed
        # action slots.
        (
            (public_tokens, public_token_mask, revealed_entity_mask, public_bias),
            (private_tokens, private_token_mask, private_entity_mask, private_bias),
            revealed_entity_embeddings,
            private_entity_embeddings,
        ) = self._current_entity_tokens(env_step)
        field_embeddings, *_ = self._embed_field(env_step.field)

        # My moveset embeddings carry per-move battle state (pp, disabled,
        # wildcard availability); they warm-start the move action stream
        # below and reach the value ladder via the action->state readbacks.
        my_move_embeddings, _ = self._embed_moves(env_step.my_moveset)

        # Opponent's privileged match-start team sheet (training self-play
        # only; all-unspecified rows at deploy, masked out here) as
        # attribute TOKENS with the "theirs" side bias -- keys of the
        # privileged latent read only; they never join the public read.
        opp_private_tokens, opp_private_token_mask, _ = _lifted_entity_vmap(
            Encoder._private_entity_tokens
        )(self, env_step.opp_private_team)
        opp_private_bias = jnp.broadcast_to(
            self.side_bias(jnp.ones((), dtype=jnp.int32)).astype(
                opp_private_tokens.dtype
            ),
            opp_private_tokens.shape[:1] + opp_private_tokens.shape[-1:],
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

        field_valid = jnp.ones_like(field_embeddings[..., 0], dtype=jnp.bool)
        prev_action_doubles_mask = jnp.array(
            [
                env_step.info[InfoFeature.INFO_FEATURE__HAS_PREV_ACTION],
                env_step.info[InfoFeature.INFO_FEATURE__HAS_PREV_ACTION],
            ],
            dtype=jnp.bool,
        )

        output_state_mask = env_step.action_mask.any(axis=0) | env_step.action_mask.any(
            axis=1
        )
        output_state_mask = output_state_mask & jnp.logical_not(env_step.done)

        # Per-entity recurrent history (12 rows, PUBLIC_ORDER-aligned with
        # the public team, masked to mapped rows) and the field history
        # state: RAW keys of the latent read, and the SAME raw rows the
        # public value rung reads (public-information-set purity -- see
        # RoundBlock). The attention-pooled history latents are no longer
        # an RL input (they summarised exactly these 13 tokens; the read
        # sees them directly) -- pool_history survives for the offline
        # critic only, and is not called on the RL path so history_pool
        # holds no RL params (265k dead leaves + their Adam state before
        # 2026-08-24; merge_params drops them from older checkpoints).
        history_tokens = jnp.concatenate(
            (history_row_states, history_field_state[None]), axis=0
        ).astype(self.cfg.dtype)
        history_valid = jnp.concatenate(
            (history_row_valid, jnp.ones(1, dtype=jnp.bool_)), axis=0
        )

        typed_action_valids = tuple(
            output_state_mask[slot_indices]
            for _, slot_indices in ACTION_DECODER_SLOT_GROUPS
        )
        value_all_tokens = self.all_value_embeddings.astype(self.cfg.dtype)
        value_private_tokens = self.private_value_embeddings.astype(self.cfg.dtype)
        value_public_tokens = self.public_value_embeddings.astype(self.cfg.dtype)

        # The public read: K latents over the flat token set of my own
        # information set -- 168 entity attribute tokens + field 3 +
        # prev-action 2 + history 13 -- become the trunk's state rows.
        public_latents = self.latent_input_read(
            (
                public_tokens,
                private_tokens,
                field_embeddings[None],
                prev_action_tokens[None],
                history_tokens[None],
            ),
            (
                public_token_mask,
                private_token_mask,
                field_valid[None],
                prev_action_doubles_mask[None],
                history_valid[None],
            ),
            (
                _PUBLIC_TOKEN_TYPES,
                _PRIVATE_TOKEN_TYPES,
                _FIELD_TOKEN_TYPES,
                _PREV_ACTION_TOKEN_TYPES,
                _HISTORY_TOKEN_TYPES,
            ),
            (public_bias, private_bias, None, None, None),
        )
        # The privileged read: a few latents over [sheet tokens | public
        # latents]; the trunk's opp stream, read by value-`all` only.
        priv_latents = self.priv_latent_read(
            (opp_private_tokens, public_latents[None]),
            (
                opp_private_token_mask,
                jnp.ones(public_latents.shape[:1], dtype=jnp.bool_)[None],
            ),
            (
                _PRIVATE_TOKEN_TYPES,
                np.full(public_latents.shape[0], _TOKEN_LATENT),
            ),
            (opp_private_bias, None),
        )
        state_valid = jnp.ones(public_latents.shape[0], dtype=jnp.bool_)
        opp_valid = jnp.ones(priv_latents.shape[0], dtype=jnp.bool_)

        # The action stream is built as before: per-type norm+MLP
        # projections over the typed slices gathered from the warm-started
        # slot-aligned sequence, carried through the scan as-is.
        action_tokens = self.action_input_norm(
            tuple(
                jnp.split(
                    output_state_sequence[ACTION_GROUP_SLOTS],
                    ACTION_GROUP_SPLITS,
                    axis=0,
                )
            )
        )
        (
            (
                _,
                _,
                action_queries,
                value_all_queries,
                value_private_queries,
                value_public_queries,
            ),
            _,
        ) = self.round_trunk(
            (
                public_latents,
                priv_latents,
                action_tokens,
                value_all_tokens,
                value_private_tokens,
                value_public_tokens,
            ),
            state_valid,
            opp_valid,
            *typed_action_valids,
            history_tokens,
            history_valid,
        )

        # Head-facing embeddings from the final round's raw action
        # stream: the GroupNorm out-norms keep the move/switch/target
        # slices in their own spaces, scattered back slot-aligned so the
        # flat src x tgt grid contract (action indexing, Q head, learner
        # metrics) is untouched.
        action_embeddings = (
            jnp.zeros_like(output_state_sequence)
            .at[ACTION_GROUP_SLOTS]
            .set(
                self.action_out_norm(
                    tuple(jnp.split(action_queries, ACTION_GROUP_SPLITS, axis=0))
                )
            )
        )
        value_embeddings = value_all_queries.reshape(-1)
        private_value_embeddings = value_private_queries.reshape(-1)
        public_value_embeddings = value_public_queries.reshape(-1)

        # (NUM_ACTION_FEATURES, entity_size) and three (4 * entity_size,)
        # value readouts; the final round drives the acting policy and
        # value estimates. `value_embeddings` (the `all` ladder rung) feeds
        # the main critic; own/public feed the counterfactual aux heads.
        return (
            action_embeddings,
            value_embeddings,
            private_value_embeddings,
            public_value_embeddings,
        )

    def _run_history_encoder(
        self,
        packed_history_step: PlayerPackedHistoryOutput,
        history_step: PlayerHistoryOutput,
    ):
        """Shared front half of the history pathway: embeds the packed
        caches and field rows once and runs the recurrent scan. Returns
        (scan output, edge_slot_ids, node_sides, per-step field vectors)."""
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
            node_sides=node_sides,
            field_step_embeddings=step_field_vec,
            step_request_count=step_request_count,
            step_valid=step_valid.squeeze(-1),
        )
        return history_output, edge_slot_ids, node_sides, step_field_vec

    def encode_history(
        self,
        env_step: PlayerEnvOutput,
        packed_history_step: PlayerPackedHistoryOutput,
        history_step: PlayerHistoryOutput,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Recurrent history pathway over the shared trajectory history.

        Consumes ONLY the public event stream — packed public entity/edge
        caches, the field history, and INFO_FEATURE__REQUEST_COUNT — no
        private observation fields, movesets, or action masks. This makes
        it safe to train against replay exports (which contain exactly the
        same inputs) and reuse live without any distribution projection;
        the offline outcome critic (rl/offline/model.py) builds on it.

        Returns, per request: ((T, NUM_PUBLIC_SLOTS, D) GRU slot states,
        (T, D) field state, (T, NUM_PUBLIC_SLOTS, D) latest raw node
        snapshot per slot — the entity's current state unmixed by GRU
        gating, which outcome readouts need verbatim).
        """
        history_output, *_ = self._run_history_encoder(
            packed_history_step, history_step
        )

        # Read the recurrent state as of each request: the snapshot after the
        # last history step whose request_count <= the request's.
        request_count = env_step.info[..., InfoFeature.INFO_FEATURE__REQUEST_COUNT]
        return self.history_encoder.state_at_requests(history_output, request_count)

    def encode_history_with_announced(
        self,
        env_step: PlayerEnvOutput,
        packed_history_step: PlayerPackedHistoryOutput,
        history_step: PlayerHistoryOutput,
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array],
        tuple[jax.Array, jax.Array, jax.Array],
    ]:
        """encode_history plus, per request, the ANNOUNCED state: the
        previous request's recurrent state advanced one extra step with
        outcome-masked messages of the request's own turn (both players'
        revealed choices, chance unresolved). The scan and both embedding
        caches run once; only the masked edge cache is embedded extra.

        Returns ((slot states, field state, node snapshots) as
        encode_history, (announced slot states, announced field state,
        pre-turn node snapshots)), each per request.
        """
        history_output, edge_slot_ids, node_sides, step_field_vec = (
            self._run_history_encoder(packed_history_step, history_step)
        )
        request_count = env_step.info[..., InfoFeature.INFO_FEATURE__REQUEST_COUNT]
        states = self.history_encoder.state_at_requests(history_output, request_count)

        masked_cache, row_is_announcement = mask_outcome_features(
            packed_history_step.edge_cache
        )
        announced_edge_embedding_cache, _ = jax.vmap(self._embed_edge)(masked_cache)
        announced = self.history_encoder.announced_states_at_requests(
            history_output=history_output,
            history_field=history_step.field,
            announced_edge_embedding_cache=announced_edge_embedding_cache,
            edge_slot_ids=edge_slot_ids,
            node_sides=node_sides,
            row_is_announcement=row_is_announcement,
            field_step_embeddings=step_field_vec,
            request_counts=request_count,
        )
        return states, announced

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
        summaries: (T, 12, D), (T, D) -> (T, num_latents, D). token_mask
        (13,) optionally restricts which tokens are readable (constant
        across T)."""
        tokens = jnp.concatenate((slot_states, field_state[..., None, :]), axis=-2)
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

    def __call__(
        self,
        env_step: PlayerEnvOutput,
        packed_history_step: PlayerPackedHistoryOutput,
        history_step: PlayerHistoryOutput,
    ):
        slot_states, field_state, _ = self.encode_history(
            env_step, packed_history_step, history_step
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

        (
            action_embeddings,
            value_embeddings,
            private_value_embeddings,
            public_value_embeddings,
        ) = _forward_vmap()(self, env_step, row_states, order_valid, field_state)

        return (
            action_embeddings,
            value_embeddings,
            private_value_embeddings,
            public_value_embeddings,
        )
