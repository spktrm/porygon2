import pprint

import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.data import CAT_VF_SUPPORT


def set_attributes(config_dict: ConfigDict, **kwargs) -> None:
    for key, value in kwargs.items():
        setattr(config_dict, key, value)


DEFAULT_DTYPE = jnp.bfloat16


def get_player_model_config(
    generation: int = 3, train: bool = False, dtype: jnp.dtype = DEFAULT_DTYPE
) -> ConfigDict:
    """``dtype`` is the forward's COMPUTE dtype (params are stored f32
    regardless). bf16 on the GPU; the CPU actor path passes f32, since XLA:CPU
    only emulates bf16."""
    cfg = ConfigDict()

    base_size = 64
    num_heads = 4
    width_scale = 1

    entity_size = int(width_scale * base_size * num_heads)

    cfg.generation = generation
    cfg.entity_size = entity_size
    cfg.dtype = dtype
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
    cfg.encoder.dtype = dtype
    # The actor's encoder assembles the policy-readable rows only
    # (Encoder.kept_rows); the learner's every row.
    cfg.encoder.train = train
    # The public-only sequence (PUBLIC_SEQUENCE_ROWS): the offline world-model
    # trainer's rows. Nothing private is assembled; encode_events is the
    # entry point.
    cfg.encoder.public_only = False

    encoder_num_heads = num_heads
    encoder_hidden_size_scale = 4
    encoder_hidden_size = int(encoder_hidden_size_scale * entity_size)
    encoder_qkv_scale = 1 / encoder_num_heads
    encoder_qkv_size = int(encoder_qkv_scale * entity_size)
    encoder_use_bias = True
    encoder_qk_layer_norm = True

    cfg.encoder.history_step = ConfigDict()
    cfg.encoder.history_step.num_heads = 2
    cfg.encoder.history_step.qk_size = encoder_qkv_size // 2

    # The trunk: `num_blocks` standard pre-RMSNorm blocks over ONE sequence
    # of NUM_SEQUENCE_ROWS rows (84; the actor assembles the 77
    # policy-readable ones, encoder.kept_rows) -- see rl/model/trunk.py.
    # Depth is the knob: at this sequence length a block is almost all
    # feed-forward parameters and almost no attention.
    cfg.encoder.trunk = ConfigDict()
    cfg.encoder.trunk.num_blocks = 6
    cfg.encoder.trunk.num_heads = num_heads
    cfg.encoder.trunk.qk_size = encoder_qkv_size
    cfg.encoder.trunk.v_size = encoder_qkv_size
    cfg.encoder.trunk.model_size = entity_size
    cfg.encoder.trunk.hidden_size = encoder_hidden_size
    cfg.encoder.trunk.use_bias = encoder_use_bias
    cfg.encoder.trunk.qk_layer_norm = encoder_qk_layer_norm
    # nGPT-style normalised residual (trunk.TrunkBlock): each sub-layer step
    # is a per-channel-alpha move on the RMS-1 sphere rather than a plain
    # add, and train_step projects the block kernels back onto the unit
    # sphere after every update. Off = today's forward, bit for bit. The
    # alpha init follows nGPT's stated rule, "of order 1/n_layers" (their
    # 0.05 at 24-36 layers); at six blocks that is 1/6. Set from the learner
    # config (artifact.player_model_config_for), never here.
    cfg.encoder.trunk.normalised_residual = False
    cfg.encoder.trunk.residual_alpha_init = 1 / cfg.encoder.trunk.num_blocks

    # The action readout. Three small heads over named trunk rows -- a
    # scalar per sheet row for switching, ONE bilinear for moves x targets,
    # a scalar per target row for pass/default.
    #
    # qk_size is the bilinear's projection width. It is the ONLY dimension
    # here: there is no adapter (the head reads the trunk's rows directly),
    # no per-group projection (a move row and a target row are already
    # different kinds of thing), and no per-modality block (modality is a
    # function of the source row, so `move_score` carries it).
    cfg.action_head = ConfigDict()
    cfg.action_head.qk_size = entity_size

    cfg.value_head = ConfigDict()
    cfg.value_head.mlp = ConfigDict()
    cfg.value_head.mlp.layer_sizes = (2 * entity_size, entity_size, len(CAT_VF_SUPPORT))
    cfg.value_head.category_values = jnp.asarray(CAT_VF_SUPPORT, dtype=cfg.dtype)
    # The privileged critic (2026-09-01): same shape as value_head, reading the
    # VALUE_CLS row of the leak-masked partition. Only instantiated under
    # cfg.train (see Porygon2PlayerModel.setup).
    cfg.privileged_value_head = ConfigDict()
    cfg.privileged_value_head.mlp = ConfigDict()
    cfg.privileged_value_head.mlp.layer_sizes = (
        2 * entity_size,
        entity_size,
        len(CAT_VF_SUPPORT),
    )
    cfg.privileged_value_head.category_values = jnp.asarray(
        CAT_VF_SUPPORT, dtype=cfg.dtype
    )
    # The public critic (2026-09-15): same shape again, reading PUBLIC_CLS,
    # the row that attends over the public tier alone. Learner-only.
    cfg.public_value_head = ConfigDict()
    cfg.public_value_head.mlp = ConfigDict()
    cfg.public_value_head.mlp.layer_sizes = (
        2 * entity_size,
        entity_size,
        len(CAT_VF_SUPPORT),
    )
    cfg.public_value_head.category_values = jnp.asarray(CAT_VF_SUPPORT, dtype=cfg.dtype)
    # The state value head (2026-09-20): same shape, reading STATE_VALUE_CLS,
    # the row that attends over the 15 public state rows alone. Learner-only.
    cfg.state_value_head = ConfigDict()
    cfg.state_value_head.mlp = ConfigDict()
    cfg.state_value_head.mlp.layer_sizes = (
        2 * entity_size,
        entity_size,
        len(CAT_VF_SUPPORT),
    )
    cfg.state_value_head.category_values = jnp.asarray(CAT_VF_SUPPORT, dtype=cfg.dtype)
    # The consequence model (rl/model/consequence.py, 2026-09-20): learner-
    # only. `rank` is the pair form's bilinear width; the sampler is a small
    # trunk of its OWN -- plain residual, never the nGPT form, because
    # project_trunk_kernels only re-projects encoder/trunk/blocks and a
    # normalised block left unprojected is a different function every step.
    cfg.consequence = ConfigDict()
    cfg.consequence.rank = entity_size
    cfg.consequence.trunk = ConfigDict()
    cfg.consequence.trunk.num_blocks = 2
    cfg.consequence.trunk.num_heads = num_heads
    cfg.consequence.trunk.qk_size = encoder_qkv_size
    cfg.consequence.trunk.v_size = encoder_qkv_size
    cfg.consequence.trunk.model_size = entity_size
    cfg.consequence.trunk.hidden_size = encoder_hidden_size
    cfg.consequence.trunk.use_bias = encoder_use_bias
    cfg.consequence.trunk.qk_layer_norm = encoder_qk_layer_norm
    cfg.consequence.trunk.normalised_residual = False
    cfg.consequence.trunk.residual_alpha_init = 1 / cfg.consequence.trunk.num_blocks
    # The PBRS potential channel's value head (2026-09-11): learner-only and
    # built only when the learner's player_potential_strength > 0 (main.py
    # sets `enabled`). One scalar in unit potential units, zero at init so a
    # merge starts the channel at W = 0 in params, target and reg alike.
    cfg.potential_head = ConfigDict()
    cfg.potential_head.enabled = False
    cfg.potential_head.zero_init_output = True
    cfg.potential_head.mlp = ConfigDict()
    cfg.potential_head.mlp.layer_sizes = (2 * entity_size, entity_size, 1)
    if cfg.num_decision_slots != 1:
        raise ValueError("q_head requires num_decision_slots == 1 (singles)")

    return cfg


def get_builder_model_config(
    generation: int = 3, train: bool = False, dtype: jnp.dtype = DEFAULT_DTYPE
) -> ConfigDict:
    cfg = ConfigDict()

    base_size = 64
    num_heads = 4
    scale = 1

    entity_size = int(scale * base_size * num_heads)

    cfg.entity_size = entity_size
    cfg.generation = generation
    cfg.dtype = dtype

    num_layers = 4
    num_heads = num_heads
    hidden_size_scale = 4
    hidden_size = int(hidden_size_scale * entity_size)
    qkv_scale = 1 / num_heads
    qkv_size = int(qkv_scale * entity_size)
    use_bias = False
    qk_layer_norm = True
    # Dormant under randombattle; no live effect today.
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
