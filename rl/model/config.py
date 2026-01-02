import pprint

import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.data import NUM_WILDCARD_FEATURES


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

    base_size = 96
    num_heads = 4
    scale = 1

    entity_size = int(scale * base_size * num_heads)

    cfg.generation = generation
    cfg.entity_size = entity_size
    cfg.dtype = DEFAULT_DTYPE

    cfg.encoder = ConfigDict()
    cfg.encoder.generation = generation
    cfg.encoder.entity_size = entity_size
    cfg.encoder.dtype = DEFAULT_DTYPE
    cfg.encoder.num_latent_embeddings = 16
    cfg.encoder.num_perceiver_steps = 2
    cfg.encoder.num_thinking_steps = 4

    encoder_num_layers = 1
    encoder_num_heads = num_heads
    encoder_hidden_size_scale = 4
    encoder_hidden_size = int(encoder_hidden_size_scale * entity_size)
    encoder_qkv_scale = 1 / encoder_num_heads
    encoder_qkv_size = int(encoder_qkv_scale * entity_size)
    encoder_qk_layer_norm = True
    encoder_use_bias = True
    encoder_use_post_attn_norm = False
    encoder_use_post_ffw_norm = False

    decoder_num_layers = 1
    decoder_num_heads = num_heads
    decoder_hidden_size_scale = 4
    decoder_hidden_size = int(decoder_hidden_size_scale * entity_size)
    decoder_qkv_scale = 1 / decoder_num_heads
    decoder_qkv_size = int(decoder_qkv_scale * entity_size)
    decoder_qk_layer_norm = True
    decoder_use_bias = True
    decoder_use_post_attn_norm = False
    decoder_use_post_ffw_norm = False

    transformer_encoder_kwargs = dict(
        num_layers=encoder_num_layers,
        num_heads=encoder_num_heads,
        qk_size=encoder_qkv_size,
        v_size=encoder_qkv_size,
        model_size=entity_size,
        use_bias=encoder_use_bias,
        resblocks_hidden_size=encoder_hidden_size,
        qk_layer_norm=encoder_qk_layer_norm,
        use_post_attn_norm=encoder_use_post_attn_norm,
        use_post_ffw_norm=encoder_use_post_ffw_norm,
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
        use_post_attn_norm=decoder_use_post_attn_norm,
        use_post_ffw_norm=decoder_use_post_ffw_norm,
    )

    cfg.encoder.timestep_encoder = ConfigDict()
    cfg.encoder.timestep_encoder.need_pos = True

    cfg.encoder.input_decoder = ConfigDict()
    cfg.encoder.input_decoder.need_pos = True

    cfg.encoder.state_encoder = ConfigDict()
    cfg.encoder.state_encoder.need_pos = False

    cfg.encoder.output_decoder = ConfigDict()
    cfg.encoder.output_decoder.need_pos = False

    for encoder in [
        cfg.encoder.timestep_encoder,
        cfg.encoder.state_encoder,
    ]:
        set_attributes(encoder, **transformer_encoder_kwargs)

    for decoder in [
        cfg.encoder.input_decoder,
        cfg.encoder.output_decoder,
    ]:
        set_attributes(decoder, **transformer_decoder_kwargs)

    # Policy Head Configuration
    cfg.wildcard_head = ConfigDict()
    cfg.value_head = ConfigDict()

    for head, output_size in [
        (cfg.value_head, 1),
        (cfg.wildcard_head, NUM_WILDCARD_FEATURES),
    ]:
        head.logits = ConfigDict()
        head.logits.layer_sizes = output_size
        head.logits.use_layer_norm = False
        head.logits.input_activation = False

    cfg.train = train
    cfg.action_head = ConfigDict()
    cfg.action_head.qk_logits = ConfigDict()
    cfg.action_head.qk_logits.qk_layer_norm = False

    # for head in [cfg.value_head, cfg.wildcard_head]:
    #     head.resnet = ConfigDict()
    #     head.resnet.num_resblocks = 1

    for head in [cfg.action_head, cfg.wildcard_head]:
        head.train = train

    return cfg


def get_builder_model_config(generation: int = 3, train: bool = False) -> ConfigDict:
    cfg = ConfigDict()

    base_size = 96
    num_heads = 4
    scale = 1

    entity_size = int(scale * base_size * num_heads)

    cfg.entity_size = entity_size
    cfg.generation = generation
    cfg.dtype = DEFAULT_DTYPE

    num_layers = 3
    num_heads = num_heads
    hidden_size_scale = 4
    hidden_size = int(hidden_size_scale * entity_size)
    qkv_scale = 1 / num_heads
    qkv_size = int(qkv_scale * entity_size)
    qk_layer_norm = True
    use_bias = True
    use_post_attn_norm = False
    use_post_ffw_norm = False

    transformer_kwargs = dict(
        num_layers=num_layers,
        num_heads=num_heads,
        qk_size=qkv_size,
        v_size=qkv_size,
        model_size=entity_size,
        use_bias=use_bias,
        resblocks_hidden_size=hidden_size,
        qk_layer_norm=qk_layer_norm,
        use_post_attn_norm=use_post_attn_norm,
        use_post_ffw_norm=use_post_ffw_norm,
    )

    cfg.encoder = ConfigDict()
    set_attributes(cfg.encoder, **transformer_kwargs)
    if generation < 4:
        cfg.encoder.need_pos = True

    for name in ["species_head", "packed_set_head", "value_head"]:
        head_cfg = ConfigDict()
        setattr(cfg, name, head_cfg)

    cfg.value_head.logits = ConfigDict()
    cfg.value_head.logits.use_layer_norm = False
    cfg.value_head.logits.input_activation = False
    cfg.value_head.logits.layer_sizes = 1

    for head in [
        cfg.species_head,
        cfg.packed_set_head,
    ]:
        head.qk_logits = ConfigDict()
        # head.qk_logits.qk_layer_norm = True
        head.train = train

    return cfg


def main():
    cfg = get_player_model_config()
    pprint.pprint(cfg)


if __name__ == "__main__":
    main()
