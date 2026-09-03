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

    encoder_num_heads = num_heads
    encoder_hidden_size_scale = 4
    encoder_hidden_size = int(encoder_hidden_size_scale * entity_size)
    encoder_qkv_scale = 1 / encoder_num_heads
    encoder_qkv_size = int(encoder_qkv_scale * entity_size)
    encoder_use_bias = True
    encoder_qk_layer_norm = True

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

    # The step GAT (2026-09-02): one attention layer over the rows of a
    # single history step (up to 8 mons touched by one major log line, self
    # included), replacing the masked mean over the step's source rows
    # that every message used to carry. Each row's message now reads WHO
    # else was in the step and what happened to them, weighted by content
    # rather than averaged -- a tidy-up in singles (2-source steps are
    # 2-row steps, so the mean was invertible), load-bearing for doubles
    # spread moves. Zero-init output projection: identity at step 0.
    cfg.encoder.history_step = ConfigDict()
    cfg.encoder.history_step.num_heads = 2
    cfg.encoder.history_step.qk_size = encoder_qkv_size // 2

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
    # The privileged critic (2026-09-01): same shape as v_head, reading the
    # VALUE_CLS row of the leak-masked partition. Only instantiated under
    # cfg.train (see Porygon2PlayerModel.setup).
    cfg.priv_v_head = ConfigDict()
    cfg.priv_v_head.mlp = ConfigDict()
    cfg.priv_v_head.mlp.layer_sizes = (
        2 * entity_size,
        entity_size,
        len(CAT_VF_SUPPORT),
    )
    cfg.priv_v_head.category_values = jnp.asarray(CAT_VF_SUPPORT, dtype=cfg.dtype)
    # The opponent discrete code: per mon, num_groups categoricals of
    # num_classes -- 16x16 = a 64-bit-ish joint space over a randbats build
    # pool, with entity_size divisible by num_groups so the code embedding
    # concatenates to one row.
    cfg.encoder.opp_code = ConfigDict()
    cfg.encoder.opp_code.num_groups = 16
    cfg.encoder.opp_code.num_classes = 16
    # The belief head: matched public row -> the mon's (G, K) code logits.
    cfg.belief_head = ConfigDict()
    cfg.belief_head.mlp = ConfigDict()
    cfg.belief_head.mlp.layer_sizes = (
        2 * entity_size,
        entity_size,
        cfg.encoder.opp_code.num_groups * cfg.encoder.opp_code.num_classes,
    )
    # The revealed-row control (2026-09-04): the belief head's (G, K)
    # logits again, from NOTHING but the matched mon's own PRE-trunk
    # public row (stop-gradient) -- no history, no other rows. Same
    # widths as the belief head, so the two differ only in what they may
    # read; their accuracy gap is inference from CONTEXT.
    cfg.revealed_belief = ConfigDict()
    cfg.revealed_belief.mlp = ConfigDict()
    cfg.revealed_belief.mlp.layer_sizes = cfg.belief_head.mlp.layer_sizes
    # The dynamics head (2026-09-03): per target row, [post-trunk row ; the
    # taken cell's source row ; its target row ; row * source row] -> the
    # row's NEXT-step pre-trunk content (rl/model/constants.DYNAMICS_TARGET_ROWS).
    cfg.dynamics_head = ConfigDict()
    cfg.dynamics_head.mlp = ConfigDict()
    cfg.dynamics_head.mlp.layer_sizes = (2 * entity_size, entity_size)
    if cfg.num_decision_slots != 1:
        # The Q critic is structural and singles-only: the doubles path
        # stacks per-stage log_policy/action_index, which the one-step
        # target code does not yet consume. Fail loudly rather than train
        # a silently-wrong Q.
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
