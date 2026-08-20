import pprint

import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.data import CAT_VF_SUPPORT


def set_attributes(config_dict: ConfigDict, **kwargs) -> None:
    """
    Sets multiple attributes on a ConfigDict object using keyword arguments.
    Args:
        config_dict (ConfigDict): The configuration object to update.
        **kwargs: Arbitrary keyword arguments representing attribute names and their corresponding values to set on the config_dict.
    Example:
        set_attributes(config, learning_rate=0.01, batch_size=32)
    """
    for key, value in kwargs.items():
        setattr(config_dict, key, value)


DEFAULT_DTYPE = jnp.bfloat16


def get_player_model_config(generation: int = 3, train: bool = False) -> ConfigDict:
    cfg = ConfigDict()

    base_size = 64
    num_heads = 4
    width_scale = 1

    entity_size = int(width_scale * base_size * num_heads)

    cfg.generation = generation
    cfg.entity_size = entity_size
    cfg.dtype = DEFAULT_DTYPE
    cfg.train = train
    # 1 = singles (one flat categorical per request — the historical path,
    # bit-identical). 2 = doubles: two head-level decision stages per turn,
    # slot 2 conditioned on slot 1's choice via SlotConditioning, ONE trunk
    # pass. Setting 2 additionally requires the service to send per-slot
    # action masks in a single request and accept two actions back (the
    # remaining doubles workstream) — the model side is complete.
    cfg.num_decision_slots = 1

    cfg.encoder = ConfigDict()
    cfg.encoder.generation = generation
    cfg.encoder.entity_size = entity_size
    cfg.encoder.dtype = DEFAULT_DTYPE

    encoder_num_layers = 1
    encoder_num_heads = num_heads
    encoder_hidden_size_scale = 4
    encoder_hidden_size = int(encoder_hidden_size_scale * entity_size)
    encoder_qkv_scale = 1 / encoder_num_heads
    encoder_qkv_size = int(encoder_qkv_scale * entity_size)
    encoder_use_bias = True
    encoder_qk_layer_norm = True
    encoder_init_residual_scale = 0.0

    decoder_num_layers = 1
    decoder_num_heads = num_heads
    decoder_hidden_size_scale = 4
    decoder_hidden_size = int(decoder_hidden_size_scale * entity_size)
    decoder_qkv_scale = 1 / decoder_num_heads
    decoder_qkv_size = int(decoder_qkv_scale * entity_size)
    decoder_use_bias = True
    decoder_qk_layer_norm = True
    decoder_init_residual_scale = 0.0

    transformer_encoder_kwargs = dict(
        num_layers=encoder_num_layers,
        num_heads=encoder_num_heads,
        qk_size=encoder_qkv_size,
        v_size=encoder_qkv_size,
        model_size=entity_size,
        use_bias=encoder_use_bias,
        resblocks_hidden_size=encoder_hidden_size,
        qk_layer_norm=encoder_qk_layer_norm,
        init_residual_scale=encoder_init_residual_scale,
    )

    transformer_decoder_kwargs = dict(
        num_layers=decoder_num_layers,
        num_heads=decoder_num_heads,
        qk_size=decoder_qkv_size,
        v_size=decoder_qkv_size,
        model_size=entity_size,
        use_bias=decoder_use_bias,
        resblocks_hidden_size=decoder_hidden_size,
        qk_layer_norm=decoder_qk_layer_norm,
        init_residual_scale=decoder_init_residual_scale,
    )

    cfg.encoder.intra_entity_encoder = ConfigDict()
    cfg.encoder.intra_entity_pool = ConfigDict()
    set_attributes(cfg.encoder.intra_entity_encoder, **transformer_encoder_kwargs)
    set_attributes(cfg.encoder.intra_entity_pool, **transformer_decoder_kwargs)

    # Intra-entity attention: each pokemon is a short set of attribute tokens
    # (species / ability / item / moves / state) mixed by a small
    # self-attention block, then pooled back to one entity vector by a single
    # learned query. Runs per-entity over ~10 tokens, so its cost is
    # negligible next to the trunk. The pool's residual gate starts at 1.0 —
    # unlike the trunk blocks, token content can only reach the entity vector
    # through this read, so it must not start as a no-op.
    cfg.encoder.intra_entity_encoder.need_pos = False
    cfg.encoder.intra_entity_encoder.num_layers = 1
    cfg.encoder.intra_entity_pool.need_pos = False
    cfg.encoder.intra_entity_pool.num_layers = 1
    cfg.encoder.intra_entity_pool.init_residual_scale = 1.0

    # Cross-entity attribute attention: the SAME shape as the intra-entity
    # block above -- one attention layer over attribute tokens, then a single
    # learned query per entity -- but the attention is GLOBAL over every
    # current-state entity's tokens instead of being confined to one entity,
    # and it REPLACES the intra-entity layer on that path (own-entity pairs
    # are a subset of the global mask, so nothing is lost). Matchup reasoning
    # is a species-token x move-token comparison ACROSS two mons; with
    # entity-local pooling there is no layer in the model where those two
    # tokens coexist, so it has to be reconstructed downstream from two lossy
    # pooled summaries. Cost is flat in the terms that dominate -- same token
    # count, same layer count, same FFW width -- and only the attention
    # probability matrix grows (168^2 vs 12*10^2 + 6*8^2 per timestep).
    # History and the opponent's privileged sheet keep the intra_entity_*
    # path unchanged: history because its packed cache is ~2 orders of
    # magnitude more rows, the opp sheet because it must stay out of any
    # policy-facing token set (see RoundBlock's leak contract).
    cfg.encoder.cross_entity_encoder = ConfigDict()
    cfg.encoder.cross_entity_pool = ConfigDict()
    set_attributes(cfg.encoder.cross_entity_encoder, **transformer_encoder_kwargs)
    set_attributes(cfg.encoder.cross_entity_pool, **transformer_decoder_kwargs)
    cfg.encoder.cross_entity_encoder.need_pos = False
    cfg.encoder.cross_entity_encoder.num_layers = 1
    # Head count is the memory dial here and the only one that bites: the
    # attention probability matrix is the whole cost of widening the mask,
    # and it scales linearly with heads while nothing else about the layer
    # does. Measured on the player model fwd+bwd at T=64 (compiled temp
    # size, 2026-08-20): entity-local baseline 182.5MB, 2 heads 202.8MB
    # (+11%), 4 heads 240.9MB (+32%). Two heads keeps the full 168-token
    # set -- no entity or attribute is dropped from the mix -- and buys the
    # capability at a third of the cost of matching the trunk's 4.
    cfg.encoder.cross_entity_encoder.num_heads = 2
    cfg.encoder.cross_entity_pool.need_pos = False
    cfg.encoder.cross_entity_pool.num_layers = 1
    cfg.encoder.cross_entity_pool.init_residual_scale = 1.0

    # Round trunk over the unified [state | action | value] sequence: each
    # round is masked self-attention, a gated cross-read of the world-model
    # history states, then one wide FFW (attention sublayers are
    # attention-only, canonical decoder-layer shape). nn.scan-ned
    # num_rounds times with stacked params, so every round has its own
    # weights and rounds can specialize instead of iterating one shared
    # refinement operator. All residual gates are zero-init, so extra
    # rounds are stable to add; each round adds parameters as well as
    # compute. 4 rounds matches the Nov 2025 everything-transformer depth.
    # Attention pooling of the recurrent history states into a fixed bank of
    # learned latents. Shared between the RL trunk (extra history-context
    # tokens) and the offline outcome critic (flattened latents -> linear
    # probe), so outcome-readout capacity lands in warm-startable params.
    cfg.encoder.history_pool = ConfigDict()
    cfg.encoder.history_pool.num_latents = 4
    cfg.encoder.history_pool.num_heads = num_heads
    cfg.encoder.history_pool.qk_size = encoder_qkv_size
    cfg.encoder.history_pool.use_bias = encoder_use_bias

    cfg.encoder.num_rounds = 4
    cfg.encoder.round = ConfigDict()
    cfg.encoder.round.num_heads = num_heads
    cfg.encoder.round.qk_size = encoder_qkv_size
    cfg.encoder.round.v_size = encoder_qkv_size
    cfg.encoder.round.model_size = entity_size
    cfg.encoder.round.hidden_size = encoder_hidden_size
    cfg.encoder.round.use_bias = encoder_use_bias
    cfg.encoder.round.qk_layer_norm = encoder_qk_layer_norm

    # Within-modality (micro) readout: NO config block — the head is a
    # parameter-less dot grid over the typed trunk streams (2026-08-17)
    # plus three zero-init per-group scales. The modality depth the
    # November experiments proved necessary lives in the round trunk
    # (move/switch/target residual streams with per-type gates), not in
    # per-modality head stacks.

    # Dedicated modality-level head: per-modality attention pooling over
    # src-slot embeddings, shared MLP, zero-init output layer (keeps the
    # init policy anchored to calculate_hierarchical_prior).
    # Policy instantiation of the shared MacroMicroHead (2026-08-20; the
    # Q critic instantiates the same module under cfg.q_head): 'dot'
    # micro = the parameter-free scaled grid over the typed trunk
    # streams; num_logits 1 = scalar logits per cell/modality for the
    # policy's log-space hierarchy.
    cfg.macro_micro = ConfigDict()
    cfg.macro_micro.micro_kind = "dot"
    cfg.macro_micro.num_logits = 1
    cfg.macro_micro.macro = ConfigDict()
    cfg.macro_micro.macro.qk_logits = ConfigDict()
    cfg.macro_micro.macro.qk_logits.num_heads = 1
    cfg.macro_micro.macro.qk_logits.use_bias = True
    cfg.macro_micro.macro.qk_logits.qk_layer_norm = True
    cfg.macro_micro.macro.mlp = ConfigDict()
    cfg.macro_micro.macro.mlp.layer_sizes = entity_size

    # Policy-owned residual adapter between the trunk's action embeddings
    # and its MacroMicroHead (2026-08-20): zero-init out layer = exact
    # identity at init and at a params-mode fresh reload, so adding it is
    # policy-preserving; it exists so the Q head's CE gradient stops
    # reshaping the parameter-free micro dot grid's geometry directly.
    cfg.policy_adapter = ConfigDict()
    cfg.policy_adapter.mlp = ConfigDict()
    cfg.policy_adapter.mlp.layer_sizes = entity_size

    # Deep value readout (Aug 2026): the previous single linear layer made
    # the value head the thinnest module in the model while the action
    # decoder kept the depth the November experiments proved necessary —
    # forcing the trunk itself to linearise win probability, in direct
    # competition with policy features. Two hidden layers on the pooled
    # 4*entity_size value embedding mirror the pi_head's per-modality
    # block depth.
    cfg.v_head = ConfigDict()
    cfg.v_head.mlp = ConfigDict()
    cfg.v_head.mlp.layer_sizes = (2 * entity_size, entity_size, len(CAT_VF_SUPPORT))
    cfg.v_head.category_values = jnp.asarray(CAT_VF_SUPPORT, dtype=cfg.dtype)

    # Multi-lambda auxiliary value head (learner-only): K categorical
    # rows over the same win/draw/loss support as v_head, one per
    # auxiliary lambda. num_heads must match the learner config's
    # player_aux_lambdas length (shape mismatch fails loudly otherwise).
    cfg.aux_v_head = ConfigDict()
    cfg.aux_v_head.num_heads = 4
    cfg.aux_v_head.mlp = ConfigDict()
    # Same depth as v_head (see comment there); final width = one
    # categorical row per aux lambda.
    cfg.aux_v_head.mlp.layer_sizes = (
        2 * entity_size,
        entity_size,
        cfg.aux_v_head.num_heads * len(CAT_VF_SUPPORT),
    )

    # Privileged two-rung all-action Q head (learner-only;
    # docs/q-critic-plan.md): categorical logits over CAT_VF_SUPPORT per
    # src x tgt cell, read off the same action embeddings as the policy
    # heads and conditioned on a pooled value embedding — the one module
    # is called twice, with the privileged value_all embedding (Q_all,
    # drives Retrace) and the private value embedding (Q_private, the
    # policy's information set), sharing every param across rungs.
    # STRUCTURAL since 2026-08-20 (no enable flag) and the policy's exact
    # module stack: an owned ActionAdapter with the cond concatenated in
    # (the rung's information set reaches every cell) into the shared
    # MacroMicroHead at num_logits = the categorical bin count, composed
    # additively via compose_action_grid — the modality-centred micro grid
    # plus a per-modality per-bin macro readout, the explicit
    # low-dimensional parameter path for "is switching better here" that
    # the flat grid made the head express cell-by-cell.
    cfg.q_head = ConfigDict()
    cfg.q_head.adapter = ConfigDict()
    cfg.q_head.adapter.mlp = ConfigDict()
    cfg.q_head.adapter.mlp.layer_sizes = entity_size
    cfg.q_head.macro_micro = ConfigDict()
    cfg.q_head.macro_micro.micro_kind = "pointer"
    cfg.q_head.macro_micro.num_logits = len(CAT_VF_SUPPORT)
    cfg.q_head.macro_micro.micro_qk = ConfigDict()
    cfg.q_head.macro_micro.micro_qk.use_bias = True
    cfg.q_head.macro_micro.micro_qk.qk_layer_norm = True
    cfg.q_head.macro_micro.macro = ConfigDict()
    cfg.q_head.macro_micro.macro.qk_logits = ConfigDict()
    cfg.q_head.macro_micro.macro.qk_logits.num_heads = 1
    cfg.q_head.macro_micro.macro.qk_logits.use_bias = True
    cfg.q_head.macro_micro.macro.qk_logits.qk_layer_norm = True
    cfg.q_head.macro_micro.macro.mlp = ConfigDict()
    cfg.q_head.macro_micro.macro.mlp.layer_sizes = entity_size

    if cfg.num_decision_slots != 1:
        # The Q critic is structural and singles-only: the doubles path
        # stacks per-stage log_policy/action_index, which the Retrace
        # target code does not yet consume. Fail loudly rather than train
        # a silently-wrong Q.
        raise ValueError("q_head requires num_decision_slots == 1 (singles)")

    return cfg


def get_builder_model_config(generation: int = 3, train: bool = False) -> ConfigDict:
    cfg = ConfigDict()

    base_size = 64
    num_heads = 4
    scale = 1

    entity_size = int(scale * base_size * num_heads)

    cfg.entity_size = entity_size
    cfg.generation = generation
    cfg.dtype = DEFAULT_DTYPE

    num_layers = 4
    num_heads = num_heads
    hidden_size_scale = 4
    hidden_size = int(hidden_size_scale * entity_size)
    qkv_scale = 1 / num_heads
    qkv_size = int(qkv_scale * entity_size)
    use_bias = False
    qk_layer_norm = True
    init_residual_scale = 0.0

    transformer_kwargs = dict(
        num_layers=num_layers,
        num_heads=num_heads,
        qk_size=qkv_size,
        v_size=qkv_size,
        model_size=entity_size,
        use_bias=use_bias,
        resblocks_hidden_size=hidden_size,
        qk_layer_norm=qk_layer_norm,
        init_residual_scale=init_residual_scale,
    )

    cfg.encoder = ConfigDict()
    set_attributes(cfg.encoder, **transformer_kwargs)

    if generation < 4:
        cfg.encoder.need_pos = True

    for name in [
        "value_head",
        "entropy_head",
        "species_head",
        "item_head",
        "ability_head",
        "move_head",
        "ev_head",
        "nature_head",
        "gender_head",
        "hiddenpower_head",
        "teratype_head",
    ]:
        head_cfg = ConfigDict()
        setattr(cfg, name, head_cfg)

    cfg.entropy_head = ConfigDict()
    cfg.entropy_head.mlp = ConfigDict()
    cfg.entropy_head.mlp.layer_sizes = 1
    cfg.entropy_head.mlp.use_bias = True

    cfg.value_head.mlp = ConfigDict()
    cfg.value_head.mlp.layer_sizes = 3
    cfg.value_head.mlp.use_bias = True
    cfg.value_head.category_values = jnp.asarray(CAT_VF_SUPPORT, dtype=cfg.dtype)

    for head in [
        cfg.species_head,
        cfg.item_head,
        cfg.ability_head,
        cfg.move_head,
        cfg.ev_head,
        cfg.nature_head,
        cfg.gender_head,
        cfg.hiddenpower_head,
        cfg.teratype_head,
    ]:
        head.qk_logits = ConfigDict()
        head.train = train

    return cfg


def main():
    cfg = get_player_model_config()
    pprint.pprint(cfg)


if __name__ == "__main__":
    main()
