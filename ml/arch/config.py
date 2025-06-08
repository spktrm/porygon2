import pprint

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


def get_model_cfg():
    cfg = ConfigDict()

    entity_size = 256
    vector_size = 1024
    num_latents = 32

    use_layer_norm = True

    cfg.encoder = ConfigDict()
    cfg.encoder.entity_size = entity_size
    cfg.encoder.vector_size = vector_size
    cfg.encoder.num_latents = num_latents

    cfg.encoder.entity_encoder = ConfigDict()
    cfg.encoder.timestep_encoder1 = ConfigDict()
    cfg.encoder.timestep_encoder2 = ConfigDict()
    cfg.encoder.action_encoder = ConfigDict()
    cfg.encoder.latent_timestep_decoder = ConfigDict()
    cfg.encoder.latent_entity_decoder = ConfigDict()
    cfg.encoder.latent_action_decoder = ConfigDict()
    cfg.encoder.latent_encoder = ConfigDict()
    cfg.encoder.action_latent_decoder = ConfigDict()

    encoder_num_layers = 2
    encoder_num_heads = 4
    encoder_hidden_size_scale = 4
    encoder_hidden_size = int(encoder_hidden_size_scale * entity_size)
    encoder_key_value_scale = 1 / encoder_num_heads
    encoder_key_value_size = int(encoder_key_value_scale * entity_size)

    decoder_num_layers = 1
    decoder_num_heads = 1
    decoder_hidden_size_scale = 1
    decoder_hidden_size = int(decoder_hidden_size_scale * entity_size)
    decoder_key_value_scale = 1 / encoder_num_heads
    decoder_key_value_size = int(decoder_key_value_scale * entity_size)

    cfg.encoder.entity_encoder.num_layers = encoder_num_layers
    cfg.encoder.entity_encoder.key_size = encoder_key_value_size
    cfg.encoder.entity_encoder.value_size = encoder_key_value_size
    cfg.encoder.entity_encoder.model_size = entity_size
    cfg.encoder.entity_encoder.num_heads = encoder_num_heads
    cfg.encoder.entity_encoder.use_layer_norm = use_layer_norm
    cfg.encoder.entity_encoder.resblocks_hidden_size = encoder_hidden_size
    cfg.encoder.entity_encoder.need_pos = False

    cfg.encoder.timestep_encoder1.num_layers = encoder_num_layers
    cfg.encoder.timestep_encoder1.key_size = encoder_key_value_size
    cfg.encoder.timestep_encoder1.value_size = encoder_key_value_size
    cfg.encoder.timestep_encoder1.model_size = entity_size
    cfg.encoder.timestep_encoder1.num_heads = encoder_num_heads
    cfg.encoder.timestep_encoder1.resblocks_hidden_size = use_layer_norm
    cfg.encoder.timestep_encoder1.use_layer_norm = encoder_hidden_size
    cfg.encoder.timestep_encoder1.need_pos = True

    cfg.encoder.timestep_encoder2.num_layers = encoder_num_layers
    cfg.encoder.timestep_encoder2.key_size = encoder_key_value_size
    cfg.encoder.timestep_encoder2.value_size = encoder_key_value_size
    cfg.encoder.timestep_encoder2.model_size = entity_size
    cfg.encoder.timestep_encoder2.num_heads = encoder_num_heads
    cfg.encoder.timestep_encoder2.resblocks_hidden_size = use_layer_norm
    cfg.encoder.timestep_encoder2.use_layer_norm = encoder_hidden_size
    cfg.encoder.timestep_encoder2.need_pos = True

    cfg.encoder.action_encoder.num_layers = encoder_num_layers
    cfg.encoder.action_encoder.key_size = encoder_key_value_size
    cfg.encoder.action_encoder.value_size = encoder_key_value_size
    cfg.encoder.action_encoder.model_size = entity_size
    cfg.encoder.action_encoder.num_heads = encoder_num_heads
    cfg.encoder.action_encoder.resblocks_hidden_size = use_layer_norm
    cfg.encoder.action_encoder.use_layer_norm = encoder_hidden_size
    cfg.encoder.action_encoder.need_pos = False

    cfg.encoder.latent_timestep_decoder.num_layers = decoder_num_layers
    cfg.encoder.latent_timestep_decoder.key_size = decoder_key_value_size
    cfg.encoder.latent_timestep_decoder.value_size = decoder_key_value_size
    cfg.encoder.latent_timestep_decoder.model_size = entity_size
    cfg.encoder.latent_timestep_decoder.num_heads = decoder_num_heads
    cfg.encoder.latent_timestep_decoder.resblocks_hidden_size = use_layer_norm
    cfg.encoder.latent_timestep_decoder.use_layer_norm = decoder_hidden_size
    cfg.encoder.latent_timestep_decoder.y_need_pos = True

    cfg.encoder.latent_entity_decoder.num_layers = decoder_num_layers
    cfg.encoder.latent_entity_decoder.key_size = decoder_key_value_size
    cfg.encoder.latent_entity_decoder.value_size = decoder_key_value_size
    cfg.encoder.latent_entity_decoder.model_size = entity_size
    cfg.encoder.latent_entity_decoder.num_heads = decoder_num_heads
    cfg.encoder.latent_entity_decoder.resblocks_hidden_size = use_layer_norm
    cfg.encoder.latent_entity_decoder.use_layer_norm = decoder_hidden_size

    cfg.encoder.latent_action_decoder.num_layers = decoder_num_layers
    cfg.encoder.latent_action_decoder.key_size = decoder_key_value_size
    cfg.encoder.latent_action_decoder.value_size = decoder_key_value_size
    cfg.encoder.latent_action_decoder.model_size = entity_size
    cfg.encoder.latent_action_decoder.num_heads = decoder_num_heads
    cfg.encoder.latent_action_decoder.use_layer_norm = use_layer_norm
    cfg.encoder.latent_action_decoder.resblocks_hidden_size = decoder_hidden_size

    cfg.encoder.latent_encoder.num_layers = 4
    cfg.encoder.latent_encoder.key_size = encoder_key_value_size
    cfg.encoder.latent_encoder.value_size = encoder_key_value_size
    cfg.encoder.latent_encoder.model_size = entity_size
    cfg.encoder.latent_encoder.num_heads = encoder_num_heads
    cfg.encoder.latent_encoder.resblocks_hidden_size = use_layer_norm
    cfg.encoder.latent_encoder.use_layer_norm = encoder_hidden_size
    cfg.encoder.latent_encoder.need_pos = False

    cfg.encoder.action_latent_decoder.num_layers = decoder_num_layers
    cfg.encoder.action_latent_decoder.key_size = decoder_key_value_size
    cfg.encoder.action_latent_decoder.value_size = decoder_key_value_size
    cfg.encoder.action_latent_decoder.model_size = entity_size
    cfg.encoder.action_latent_decoder.num_heads = decoder_num_heads
    cfg.encoder.action_latent_decoder.use_layer_norm = use_layer_norm
    cfg.encoder.action_latent_decoder.resblocks_hidden_size = decoder_hidden_size

    # Policy Head Configuration
    cfg.policy_head = ConfigDict()
    cfg.policy_head.transformer = ConfigDict()
    cfg.policy_head.logits = ConfigDict()

    cfg.policy_head.transformer.num_layers = encoder_num_layers
    cfg.policy_head.transformer.key_size = encoder_key_value_size
    cfg.policy_head.transformer.value_size = encoder_key_value_size
    cfg.policy_head.transformer.model_size = entity_size
    cfg.policy_head.transformer.num_heads = encoder_num_heads
    cfg.policy_head.transformer.resblocks_hidden_size = use_layer_norm
    cfg.policy_head.transformer.use_layer_norm = encoder_hidden_size

    cfg.policy_head.logits.num_logits = 1
    cfg.policy_head.logits.num_linear_layers = 2
    cfg.policy_head.logits.use_layer_norm = use_layer_norm

    # Value Head Configuration
    cfg.value_head = ConfigDict()
    cfg.value_head.transformer = ConfigDict()
    cfg.value_head.logits = ConfigDict()
    cfg.value_head.num_latents = num_latents
    cfg.value_head.num_heads = 8

    cfg.value_head.transformer.num_layers = encoder_num_layers
    cfg.value_head.transformer.key_size = encoder_key_value_size
    cfg.value_head.transformer.value_size = encoder_key_value_size
    cfg.value_head.transformer.model_size = entity_size
    cfg.value_head.transformer.num_heads = encoder_num_heads
    cfg.value_head.transformer.resblocks_hidden_size = use_layer_norm
    cfg.value_head.transformer.use_layer_norm = encoder_hidden_size

    cfg.value_head.logits.num_logits = 1
    cfg.value_head.logits.num_linear_layers = 2
    cfg.value_head.logits.use_layer_norm = use_layer_norm

    return cfg


def main():
    cfg = get_model_cfg()
    pprint.pprint(cfg)


if __name__ == "__main__":
    main()
