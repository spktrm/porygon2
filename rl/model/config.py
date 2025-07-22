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


def get_model_config():
    cfg = ConfigDict()

    num_heads = 6
    entity_size = 64 * num_heads
    num_latents = 8 * num_heads
    dtype = jnp.bfloat16

    cfg.entity_size = entity_size
    cfg.num_latents = num_latents

    use_layer_norm = True

    cfg.encoder = ConfigDict()
    cfg.encoder.entity_size = entity_size
    cfg.encoder.num_latents = num_latents
    cfg.encoder.dtype = dtype

    cfg.encoder.entity_encoder = ConfigDict()
    cfg.encoder.timestep_encoder = ConfigDict()
    cfg.encoder.action_encoder = ConfigDict()
    cfg.encoder.latent_timestep_decoder = ConfigDict()
    cfg.encoder.latent_entity_decoder = ConfigDict()
    cfg.encoder.latent_action_decoder = ConfigDict()
    cfg.encoder.latent_encoder = ConfigDict()

    encoder_num_layers = 1
    encoder_num_heads = num_heads
    encoder_hidden_size_scale = 1
    encoder_hidden_size = int(encoder_hidden_size_scale * entity_size)
    encoder_key_value_scale = 1 / encoder_num_heads
    encoder_key_value_size = int(encoder_key_value_scale * entity_size)
    encoder_qk_layer_norm = True

    decoder_num_layers = 1
    decoder_num_heads = num_heads
    decoder_hidden_size_scale = 1
    decoder_hidden_size = int(decoder_hidden_size_scale * entity_size)
    decoder_key_value_scale = 1 / decoder_num_heads
    decoder_key_value_size = int(decoder_key_value_scale * entity_size)
    decoder_qk_layer_norm = True

    transformer_encoder_kwargs = dict(
        num_layers=encoder_num_layers,
        num_heads=encoder_num_heads,
        key_size=encoder_key_value_size,
        value_size=encoder_key_value_size,
        model_size=entity_size,
        use_layer_norm=use_layer_norm,
        resblocks_hidden_size=encoder_hidden_size,
        qk_layer_norm=encoder_qk_layer_norm,
        dtype=dtype,
    )

    transformer_decoder_kwargs = dict(
        num_layers=decoder_num_layers,
        num_heads=decoder_num_heads,
        key_size=decoder_key_value_size,
        value_size=decoder_key_value_size,
        model_size=entity_size,
        use_layer_norm=use_layer_norm,
        resblocks_hidden_size=decoder_hidden_size,
        qk_layer_norm=decoder_qk_layer_norm,
        dtype=dtype,
    )

    set_attributes(cfg.encoder.entity_encoder, **transformer_encoder_kwargs)
    cfg.encoder.entity_encoder.need_pos = False

    set_attributes(cfg.encoder.timestep_encoder, **transformer_encoder_kwargs)
    cfg.encoder.timestep_encoder.need_pos = True

    set_attributes(cfg.encoder.action_encoder, **transformer_encoder_kwargs)
    cfg.encoder.action_encoder.need_pos = False

    set_attributes(cfg.encoder.latent_timestep_decoder, **transformer_decoder_kwargs)
    cfg.encoder.latent_timestep_decoder.need_pos = True

    set_attributes(cfg.encoder.latent_entity_decoder, **transformer_decoder_kwargs)

    set_attributes(cfg.encoder.latent_action_decoder, **transformer_decoder_kwargs)

    set_attributes(cfg.encoder.latent_encoder, **transformer_encoder_kwargs)
    cfg.encoder.latent_encoder.num_layers = 3
    cfg.encoder.latent_encoder.need_pos = False

    # Policy Head Configuration
    cfg.policy_head = ConfigDict()
    cfg.policy_head.transformer = ConfigDict()
    cfg.policy_head.dtype = dtype

    set_attributes(cfg.policy_head.transformer, **transformer_encoder_kwargs)

    # Value Head Configuration
    cfg.value_head = ConfigDict()
    cfg.value_head.transformer = ConfigDict()
    cfg.value_head.entity_size = entity_size
    cfg.value_head.dtype = dtype

    set_attributes(cfg.value_head.transformer, **transformer_encoder_kwargs)

    return cfg


def main():
    cfg = get_model_config()
    pprint.pprint(cfg)


if __name__ == "__main__":
    main()
