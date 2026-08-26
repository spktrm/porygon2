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
    # cross-attend ONE flat token set -- 168 current-board attribute
    # tokens (12 public x 10 + 6 private x 8) + field 3 + prev-action 2 +
    # raw history 13 = 186 keys -- and become the trunk's state rows. It
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
    cfg.encoder.num_latents = 48
    cfg.encoder.latent_read = ConfigDict()
    set_attributes(cfg.encoder.latent_read, **transformer_decoder_kwargs)
    cfg.encoder.latent_read.need_pos = False
    cfg.encoder.latent_read.num_layers = 1
    cfg.encoder.latent_read.init_residual_scale = 1.0

    # Per-group action decoders (2026-08-27): the Nov-2025 capacity
    # restored — one PRIVATE 2-layer decoder per action slot group
    # (move/switch/target), q = that group's trunk-output slice, kv = the
    # trunk's final public latents. Motivated by the separation probe
    # (rl/offline/separation_probe.py): the shared action stream
    # generalises identity-keyed within-modality labels at held-out
    # r = 0.58 for MOVES and r = 0.096 for SWITCHES — the live critic's
    # within-row asymmetry reproduced under control, so the deficit is
    # representational, not label supply. FFW width is the OLD design's 1x
    # (8492b54^: resblocks_hidden = entity_size), not the 4x decoder
    # default — evidence-backed and ~0.93M params per group.
    # init_residual_scale 0.05 EXPLICITLY: the shared decoder default
    # above is 0.0, the measured-dead product-of-zeros case (see
    # round.init_gate below).
    cfg.encoder.action_decoder = ConfigDict()
    set_attributes(cfg.encoder.action_decoder, **transformer_decoder_kwargs)
    cfg.encoder.action_decoder.need_pos = False
    cfg.encoder.action_decoder.num_layers = 2
    cfg.encoder.action_decoder.resblocks_hidden_size = entity_size
    cfg.encoder.action_decoder.init_residual_scale = 0.05

    # Round trunk over the unified [state | action | value] sequence: each
    # round is masked self-attention over the state latents, the action
    # self-attention plus the state<->action exchange, the value rows'
    # read of the state stream, then group-level FFWs (attention sublayers
    # are attention-only, canonical decoder-layer shape). History reaches
    # the trunk only through the latent input read, not through a
    # per-round cross-read. nn.scan-ned
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
    # Init of every RoundBlock residual gate (2026-08-24; was a hard-coded
    # zeros_init). Offline gate-contribution read on ckpt_00074597
    # (rl/offline/gate_contribution.py): all six FFWs (48% of the params)
    # and the state self-attention contributed <= 5e-4 of their stream in
    # every round -- gates at |g| ~ 1e-3, a random walk -- while every
    # ones-init scale and every cross-stream READ gate trained. ReZero's
    # alpha = 0 needs <f_random(x), delta> to carry a consistent sign; it
    # does not here (RL noise, and the reads open first and make the FFW's
    # random features redundant), and the block's own gradient is exactly
    # 0 until alpha moves. Any non-zero constant hands the block a
    # direction-consistent gradient from step 0 under Adam; 0.05 keeps the
    # init contribution at ~1% of the state/action streams (~5% of the
    # tiny value streams). The head's flat-at-init contract is untouched
    # (type_scale, the zero-init micro_local_src/tgt routes and the
    # zero-init macro out layers all live in the head).
    # Acceptance: FFW contribution >= 0.01 at the 20k read.
    cfg.encoder.round.init_gate = 0.05

    # Within-modality (micro) readout: NO config block — the head is a
    # parameter-less dot grid over the typed trunk streams (2026-08-17)
    # plus three zero-init per-group scales. The modality depth the
    # November experiments proved necessary lives in the round trunk
    # (move/switch/target residual streams with per-type gates), not in
    # per-modality head stacks.

    # The two action-axis readouts (2026-08-25). Both are ActionScoreHead
    # over the same src x tgt grid — adapter -> macro/micro -> composition —
    # and differ only in `reduce`, set at the call site in player_model:
    # the policy multiplies softmaxes in log space, the advantage adds macro
    # onto the legality-centred micro. Same shape of config for both, so the
    # asymmetries between them are visible as config diffs rather than buried
    # in two hand-written forward passes.
    #
    # Each owns a zero-init residual ADAPTER onto the trunk's action
    # embeddings: exact identity at init and at a params-mode fresh reload,
    # so adding one is policy-preserving, and the advantage head's loss
    # cannot reshape the policy's micro geometry directly.
    #
    # num_logits 1 on both: a SCALAR per cell. For the advantage that is
    # A(s, a), with Q = sg(V) + A centred under pi (heads.compose_q) — the
    # categorical per-cell readout it replaced (2026-08-23, Step 3 of
    # docs/critic-weakness-analysis.md) let the head fit taken-cell labels
    # through a state-only route (Step 6 probe: label floor reached with
    # within-state action variance collapsing 5x).
    def _action_score_head() -> ConfigDict:
        head = ConfigDict()
        head.adapter = ConfigDict()
        head.adapter.mlp = ConfigDict()
        head.adapter.mlp.layer_sizes = entity_size
        head.macro_micro = ConfigDict()
        head.macro_micro.num_logits = 1
        # Per-slot-group micro projections (2026-08-25): qk_size is the
        # width EACH group gets, so the Dense is entity_size x (3 * qk_size)
        # and the three groups own disjoint coordinates. Set explicitly —
        # the num_heads default would have silently split entity_size three
        # ways and shrunk every group to 85.
        head.macro_micro.micro_qk = ConfigDict()
        head.macro_micro.micro_qk.qk_size = entity_size
        head.macro_micro.micro_qk.use_bias = True
        head.macro_micro.micro_qk.qk_layer_norm = True
        # Modality-level head: per-modality attention pooling over src-slot
        # embeddings, MLP, zero-init output layer (keeps the init policy
        # anchored to calculate_hierarchical_prior).
        head.macro_micro.macro = ConfigDict()
        head.macro_micro.macro.qk_logits = ConfigDict()
        head.macro_micro.macro.qk_logits.num_heads = 1
        head.macro_micro.macro.qk_logits.use_bias = True
        head.macro_micro.macro.qk_logits.qk_layer_norm = True
        head.macro_micro.macro.mlp = ConfigDict()
        head.macro_micro.macro.mlp.layer_sizes = entity_size
        return head

    cfg.policy_head = _action_score_head()
    cfg.advantage_head = _action_score_head()

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
