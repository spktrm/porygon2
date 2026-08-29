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
    # 0.05, not 0.0 (2026-08-24): the entity pool's self-attention sat at
    # mha_a -0.026 / ffn_a -0.003 after 74.6k steps -- a zero-init scalar
    # times a random block is a product whose gate gradient has no
    # consistent sign under RL noise and whose block gradient is exactly
    # 0 until the gate moves. See cfg.encoder.round.init_gate.
    encoder_init_residual_scale = 0.05

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

    # Perceiver-style latent input read (2026-08-21): K learned latents
    # cross-attend ONE flat token set -- 12 public x 11 (10 attributes plus
    # that slot's recurrent history state, folded in 2026-08-28) + 6 private
    # x 8 + field 3 + prev-action 2 + history field 3 + info 1 = 189 keys --
    # and become the trunk's state rows. It
    # replaces the cross-entity pool + per-entity pooling + per-substream
    # input MLPs on that path: the board no longer collapses to one
    # vector per entity before the trunk sees it, and substream identity
    # is carried by additive token-type / row / group / side biases on the
    # tokens instead of by slice boundaries. Cost: probability matrix
    # K x 186 at the trunk's head count (~9k) once per timestep vs the
    # 168^2 x 2 heads (~56k) cross-entity mix it replaces, so this should
    # land BELOW the pre-pool entity-local baseline (182.5MB at T=64);
    # measure before merge. The read's residual starts at 1.0: token
    # content reaches the latents only through it.
    cfg.encoder.history_pool = ConfigDict()
    cfg.encoder.history_pool.num_latents = 4
    cfg.encoder.history_pool.num_heads = num_heads
    cfg.encoder.history_pool.qk_size = encoder_qkv_size
    cfg.encoder.history_pool.use_bias = encoder_use_bias

    # The trunk: `num_blocks` standard pre-RMSNorm blocks over ONE sequence
    # of NUM_SEQUENCE_ROWS (61) rows -- see rl/model/trunk.py. Replaces the
    # four-round, three-stream, five-masked-attention RoundBlock on
    # 2026-08-29: at 61 rows an all-pairs attention is 3.7k cells, so the
    # block masks that encoded the routing were buying nothing but their own
    # complexity, and the 48 latents they fed were a bottleneck between rows
    # the trunk can now simply carry. Depth is the knob: a block costs ~1.05M
    # params and almost no attention at this sequence length.
    cfg.encoder.trunk = ConfigDict()
    cfg.encoder.trunk.num_blocks = 6
    cfg.encoder.trunk.num_heads = num_heads
    cfg.encoder.trunk.qk_size = encoder_qkv_size
    cfg.encoder.trunk.v_size = encoder_qkv_size
    cfg.encoder.trunk.model_size = entity_size
    cfg.encoder.trunk.hidden_size = encoder_hidden_size
    cfg.encoder.trunk.use_bias = encoder_use_bias
    cfg.encoder.trunk.qk_layer_norm = encoder_qk_layer_norm

    # Within-modality (micro) readout: NO config block — the head is a
    # parameter-less dot grid over the typed trunk streams (2026-08-17)
    # plus three zero-init per-group scales. The modality depth the
    # November experiments proved necessary lives in the round trunk
    # (move/switch/target residual streams with per-type gates), not in
    # per-modality head stacks.

    # The action readout (2026-08-29). Three small heads over named trunk
    # rows -- a scalar per sheet row for switching, ONE bilinear for
    # moves x targets, a scalar per target row for pass/default -- replacing
    # the hierarchical macro/micro stack that was instantiated twice, for a
    # policy and for an advantage head the policy did not read. 2.65M
    # parameters became 0.13M.
    #
    # qk_size is the bilinear's projection width. It is the ONLY dimension
    # here: there is no adapter (the head reads the trunk's rows directly),
    # no per-group projection (a move row and a target row are already
    # different kinds of thing), and no per-modality block (modality is a
    # function of the src half, so `local_src` carries it).
    cfg.action_head = ConfigDict()
    cfg.action_head.qk_size = entity_size

    # Deep value readout (Aug 2026): the previous single linear layer made
    # the value head the thinnest module in the model while the action
    # decoder kept the depth the November experiments proved necessary —
    # forcing the trunk itself to linearise win probability, in direct
    # competition with policy features. Two hidden layers on the pooled
    # 4*entity_size value embedding mirror the policy head's per-modality
    # block depth.
    cfg.v_head = ConfigDict()
    cfg.v_head.mlp = ConfigDict()
    cfg.v_head.mlp.layer_sizes = (2 * entity_size, entity_size, len(CAT_VF_SUPPORT))
    cfg.v_head.category_values = jnp.asarray(CAT_VF_SUPPORT, dtype=cfg.dtype)
    if cfg.num_decision_slots != 1:
        # The Q critic is structural and singles-only: the doubles path
        # stacks per-stage log_policy/action_index, which the one-step
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
    # 0.05 for the same reason as cfg.encoder.round.init_gate (dormant
    # under randombattle; no live effect today).
    init_residual_scale = 0.05

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
