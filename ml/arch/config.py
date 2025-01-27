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

    entity_size = 512
    vector_size = 1024

    use_layer_norm = True
    use_spectral_linear = False

    cfg.encoder = ConfigDict()
    cfg.encoder.entity_size = entity_size
    cfg.encoder.vector_size = vector_size

    cfg.encoder.entity_encoder = ConfigDict()
    cfg.encoder.timestep_encoder = ConfigDict()
    cfg.encoder.entity_timestep_decoder = ConfigDict()
    cfg.encoder.action_entity_decoder = ConfigDict()

    num_transformer_layers = 1
    num_transformer_heads = 4
    transformer_hidden_size_scale = 4
    transformer_hidden_size = int(transformer_hidden_size_scale * entity_size)
    transformer_key_value_scale = 1 / num_transformer_heads
    transformer_key_value_size = int(transformer_key_value_scale * entity_size)

    cfg.encoder.timestep_encoder.num_layers = num_transformer_layers
    cfg.encoder.timestep_encoder.key_size = transformer_key_value_size
    cfg.encoder.timestep_encoder.value_size = transformer_key_value_size
    cfg.encoder.timestep_encoder.model_size = entity_size
    cfg.encoder.timestep_encoder.num_heads = num_transformer_heads
    cfg.encoder.timestep_encoder.use_layer_norm = use_layer_norm
    cfg.encoder.timestep_encoder.use_spectral_linear = use_spectral_linear
    cfg.encoder.timestep_encoder.resblocks_hidden_size = transformer_hidden_size

    cfg.encoder.entity_encoder.num_layers = num_transformer_layers
    cfg.encoder.entity_encoder.key_size = transformer_key_value_size
    cfg.encoder.entity_encoder.value_size = transformer_key_value_size
    cfg.encoder.entity_encoder.model_size = entity_size
    cfg.encoder.entity_encoder.num_heads = num_transformer_heads
    cfg.encoder.entity_encoder.use_layer_norm = use_layer_norm
    cfg.encoder.entity_encoder.use_spectral_linear = use_spectral_linear
    cfg.encoder.entity_encoder.resblocks_hidden_size = transformer_hidden_size

    cfg.encoder.entity_timestep_decoder.num_layers = num_transformer_layers
    cfg.encoder.entity_timestep_decoder.key_size = transformer_key_value_size
    cfg.encoder.entity_timestep_decoder.value_size = transformer_key_value_size
    cfg.encoder.entity_timestep_decoder.model_size = entity_size
    cfg.encoder.entity_timestep_decoder.num_heads = num_transformer_heads
    cfg.encoder.entity_timestep_decoder.use_layer_norm = use_layer_norm
    cfg.encoder.entity_timestep_decoder.use_spectral_linear = use_spectral_linear
    cfg.encoder.entity_timestep_decoder.resblocks_hidden_size = transformer_hidden_size

    cfg.encoder.action_entity_decoder.num_layers = num_transformer_layers
    cfg.encoder.action_entity_decoder.key_size = transformer_key_value_size
    cfg.encoder.action_entity_decoder.value_size = transformer_key_value_size
    cfg.encoder.action_entity_decoder.model_size = entity_size
    cfg.encoder.action_entity_decoder.num_heads = num_transformer_heads
    cfg.encoder.action_entity_decoder.use_layer_norm = use_layer_norm
    cfg.encoder.action_entity_decoder.use_spectral_linear = use_spectral_linear
    cfg.encoder.action_entity_decoder.resblocks_hidden_size = transformer_hidden_size

    # Policy Head Configuration
    cfg.policy_head = ConfigDict()
    cfg.policy_head.transformer = ConfigDict()
    cfg.policy_head.logits = ConfigDict()

    cfg.policy_head.transformer.num_layers = num_transformer_layers
    cfg.policy_head.transformer.key_size = transformer_key_value_size
    cfg.policy_head.transformer.value_size = transformer_key_value_size
    cfg.policy_head.transformer.model_size = entity_size
    cfg.policy_head.transformer.num_heads = num_transformer_heads
    cfg.policy_head.transformer.use_layer_norm = use_layer_norm
    cfg.policy_head.transformer.use_spectral_linear = use_spectral_linear
    cfg.policy_head.transformer.resblocks_hidden_size = transformer_hidden_size

    cfg.policy_head.logits.num_logits = 1
    cfg.policy_head.logits.num_linear_layers = 3
    cfg.policy_head.logits.use_layer_norm = use_layer_norm

    # Value Head Configuration
    cfg.value_head = ConfigDict()
    cfg.value_head.transformer = ConfigDict()
    cfg.value_head.logits = ConfigDict()

    cfg.value_head.transformer.num_layers = num_transformer_layers
    cfg.value_head.transformer.key_size = transformer_key_value_size
    cfg.value_head.transformer.value_size = transformer_key_value_size
    cfg.value_head.transformer.model_size = entity_size
    cfg.value_head.transformer.num_heads = num_transformer_heads
    cfg.value_head.transformer.use_layer_norm = use_layer_norm
    cfg.value_head.transformer.use_spectral_linear = use_spectral_linear
    cfg.value_head.transformer.resblocks_hidden_size = transformer_hidden_size

    cfg.value_head.logits.num_logits = 1
    cfg.value_head.logits.num_linear_layers = 2
    cfg.value_head.logits.use_layer_norm = use_layer_norm

    return cfg


def main():
    cfg = get_model_cfg()
    pprint.pprint(cfg)


if __name__ == "__main__":
    main()
