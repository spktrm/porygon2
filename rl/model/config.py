import pprint

import jax.numpy as jnp
from ml_collections import ConfigDict


def camel_case_path(path_list):
    """Converts a list of path components to a CamelCase string."""
    return "".join(
        [
            (
                word.capitalize()
                if len(word.split("_")) <= 1
                else camel_case_path(word.split("_"))
            )
            for word in path_list
        ]
    )


def add_name_recursive(cfg, path=None):
    """Recursively creates a new ConfigDict with `name` attributes based on the path."""
    if path is None:
        path = []

    new_cfg = ConfigDict()

    if isinstance(cfg, ConfigDict):
        new_cfg.name = camel_case_path(path)  # Set the name based on the current path

        for key, value in cfg.items():
            # Recursively add names to each nested ConfigDict
            new_cfg[key] = add_name_recursive(value, path + [key])
    else:
        # For non-ConfigDict items, simply copy them over
        new_cfg = cfg

    return new_cfg


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


def get_player_model_config(generation: int = 3, **head_params: dict) -> ConfigDict:
    cfg = ConfigDict()

    base_size = 128
    num_heads = 2
    scale = 1

    entity_size = int(scale * base_size * num_heads)

    num_state_latents = 6
    num_history_latents = 64
    cfg.num_state_latents = num_state_latents
    cfg.num_history_latents = num_history_latents

    cfg.generation = generation
    cfg.entity_size = entity_size
    cfg.dtype = DEFAULT_DTYPE

    cfg.encoder = ConfigDict()
    cfg.encoder.generation = generation
    cfg.encoder.entity_size = entity_size
    cfg.encoder.dtype = DEFAULT_DTYPE
    cfg.encoder.num_state_latents = num_state_latents
    cfg.encoder.num_history_latents = num_history_latents

    encoder_num_layers = 1
    encoder_num_heads = num_heads
    encoder_hidden_size_scale = 1
    encoder_hidden_size = int(encoder_hidden_size_scale * entity_size)
    encoder_qkv_scale = 1 / encoder_num_heads
    encoder_qkv_size = int(encoder_qkv_scale * entity_size)
    encoder_qk_layer_norm = True
    encoder_use_bias = True

    decoder_num_layers = 1
    decoder_num_heads = num_heads
    decoder_hidden_size_scale = 1
    decoder_hidden_size = int(decoder_hidden_size_scale * entity_size)
    decoder_qkv_scale = 1 / decoder_num_heads
    decoder_qkv_size = int(decoder_qkv_scale * entity_size)
    decoder_qk_layer_norm = True
    decoder_use_bias = True

    transformer_encoder_kwargs = dict(
        num_layers=encoder_num_layers,
        num_heads=encoder_num_heads,
        qk_size=encoder_qkv_size,
        v_size=encoder_qkv_size,
        model_size=entity_size,
        use_bias=encoder_use_bias,
        resblocks_hidden_size=encoder_hidden_size,
        qk_layer_norm=encoder_qk_layer_norm,
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
    )

    cfg.encoder.timestep_gat = ConfigDict()
    cfg.encoder.timestep_gat.out_dim = entity_size
    cfg.encoder.timestep_gat.num_layers = 1
    cfg.encoder.timestep_gat.num_heads = num_heads
    cfg.encoder.timestep_gat.max_edges = 4

    cfg.encoder.timestep_encoder = ConfigDict()
    set_attributes(cfg.encoder.timestep_encoder, **transformer_encoder_kwargs)
    cfg.encoder.timestep_encoder.need_pos = True

    cfg.encoder.timestep_decoder = ConfigDict()
    set_attributes(cfg.encoder.timestep_decoder, **transformer_decoder_kwargs)
    cfg.encoder.timestep_decoder.need_pos = True

    cfg.encoder.entity_timestep_transformer = ConfigDict()
    set_attributes(
        cfg.encoder.entity_timestep_transformer, **transformer_decoder_kwargs
    )
    cfg.encoder.entity_timestep_transformer.num_layers = 4

    cfg.encoder.query_pool = ConfigDict()
    set_attributes(cfg.encoder.query_pool, **transformer_decoder_kwargs)
    cfg.encoder.query_pool.num_layers = 1
    cfg.encoder.query_pool.need_pos = False

    cfg.encoder.action_entity_decoder = ConfigDict()
    set_attributes(cfg.encoder.action_entity_decoder, **transformer_decoder_kwargs)
    cfg.encoder.action_entity_decoder.num_layers = 1
    cfg.encoder.action_entity_decoder.need_pos = False

    # Policy Head Configuration
    cfg.action_type_head = ConfigDict()
    cfg.wildcard_head = ConfigDict()
    cfg.value_head = ConfigDict()

    for head, output_size in [
        (cfg.value_head, 1),
        (cfg.action_type_head, 3),
        (cfg.wildcard_head, 5),
    ]:
        head.logits = ConfigDict()
        head.logits.layer_sizes = output_size
        head.logits.use_layer_norm = True

    cfg.move_head = ConfigDict()
    cfg.switch_head = ConfigDict()

    for head in [cfg.move_head, cfg.switch_head]:
        head.qk_logits = ConfigDict()
        head.qk_logits.num_layers_query = 1
        head.qk_logits.num_layers_keys = 3
        head.qk_logits.use_layer_norm = True

    for head in [
        cfg.value_head,
        cfg.action_type_head,
        cfg.move_head,
        cfg.switch_head,
        cfg.wildcard_head,
    ]:
        head.resnet = ConfigDict()
        head.resnet.num_resblocks = 1

    if head_params is not None:
        for head in [
            cfg.action_type_head,
            cfg.move_head,
            cfg.switch_head,
            cfg.wildcard_head,
        ]:
            for param_name, param_value in head_params.items():
                setattr(head, param_name, param_value)

    return cfg


def get_builder_model_config(generation: int = 3, **head_params: dict) -> ConfigDict:
    cfg = ConfigDict()

    base_size = 128
    num_heads = 2
    scale = 1

    num_latents = 3
    entity_size = int(scale * base_size * num_heads)

    cfg.num_latents = num_latents

    cfg.entity_size = entity_size
    cfg.generation = generation
    cfg.dtype = DEFAULT_DTYPE
    cfg.temp = 1.0

    num_layers = 2
    num_heads = num_heads
    hidden_size_scale = 1
    hidden_size = int(hidden_size_scale * entity_size)
    qkv_scale = 1 / num_heads
    qkv_size = int(qkv_scale * entity_size)
    qk_layer_norm = True
    use_bias = True

    transformer_kwargs = dict(
        num_layers=num_layers,
        num_heads=num_heads,
        qk_size=qkv_size,
        v_size=qkv_size,
        model_size=entity_size,
        use_bias=use_bias,
        resblocks_hidden_size=hidden_size,
        qk_layer_norm=qk_layer_norm,
    )

    cfg.transformer = ConfigDict()
    set_attributes(cfg.transformer, **transformer_kwargs)
    cfg.transformer.need_pos = True

    for param_name, param_value in head_params.items():
        setattr(cfg, param_name, param_value)

    for name in [
        "continue_head",
        "selection_head",
        "species_head",
        "packed_set_head",
        "value_head",
    ]:
        head_cfg = ConfigDict()
        head_cfg.resnet = ConfigDict()
        head_cfg.resnet.num_resblocks = 1
        setattr(cfg, name, head_cfg)

    for head, output_size in [
        (cfg.continue_head, 2),
        (cfg.value_head, 1),
    ]:
        head.logits = ConfigDict()
        head.logits.layer_sizes = output_size
        head.logits.use_layer_norm = True

    for head in [
        cfg.selection_head,
        cfg.species_head,
        cfg.packed_set_head,
    ]:
        head.qk_logits = ConfigDict()
        head.qk_logits.num_layers_query = 1
        head.qk_logits.num_layers_keys = 3
        head.qk_logits.use_layer_norm = True

    if head_params is not None:
        for head in [
            cfg.continue_head,
            cfg.selection_head,
            cfg.species_head,
            cfg.packed_set_head,
        ]:
            for param_name, param_value in head_params.items():
                setattr(head, param_name, param_value)

    return cfg


def main():
    cfg = get_player_model_config()
    pprint.pprint(cfg)


if __name__ == "__main__":
    main()
