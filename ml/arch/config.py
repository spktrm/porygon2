import pprint

from ml_collections import ConfigDict

from ml.arch.modules import GatingType, PoolMethod


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

    entity_size = 128
    vector_size = 512

    use_layer_norm = True
    use_spectral_linear = False

    cfg.encoder = ConfigDict()
    cfg.encoder.entity_size = entity_size
    cfg.encoder.vector_size = vector_size
    cfg.encoder.entity_encoder = ConfigDict()
    cfg.encoder.move_encoder = ConfigDict()
    cfg.encoder.edge_encoder = ConfigDict()
    cfg.encoder.side_condition_encoder = ConfigDict()
    cfg.encoder.field_encoder = ConfigDict()
    cfg.encoder.timestep_merge = ConfigDict()
    cfg.encoder.entity_transformer = ConfigDict()
    cfg.encoder.timestep_transformer = ConfigDict()
    cfg.encoder.entity_timestep_transformer = ConfigDict()
    cfg.encoder.action_transformer = ConfigDict()
    cfg.encoder.action_entity_transformer = ConfigDict()
    cfg.encoder.contextual_entity_agg = ConfigDict()
    cfg.encoder.contextual_timestep_agg = ConfigDict()
    cfg.encoder.contextual_action_agg = ConfigDict()
    cfg.encoder.average_contextual_entity_resnet = ConfigDict()
    cfg.encoder.average_contextual_timestep_resnet = ConfigDict()
    cfg.encoder.average_contextual_action_resnet = ConfigDict()
    cfg.encoder.state_merge = ConfigDict()
    cfg.encoder.state_resnet = ConfigDict()

    cfg.encoder.entity_encoder.entity_size = entity_size
    cfg.encoder.edge_encoder.entity_size = entity_size
    cfg.encoder.move_encoder.entity_size = entity_size
    cfg.encoder.side_condition_encoder.entity_size = entity_size // 4
    cfg.encoder.field_encoder.entity_size = entity_size // 4

    cfg.encoder.timestep_merge.output_size = entity_size
    cfg.encoder.timestep_merge.gating_type = GatingType.POINTWISE
    cfg.encoder.timestep_merge.use_layer_norm = use_layer_norm

    num_transformer_layers = 1
    num_transformer_heads = 2
    transformer_hidden_size_scale = 64
    transformer_hidden_size = int(transformer_hidden_size_scale * entity_size)

    cfg.encoder.entity_transformer.num_layers = num_transformer_layers
    cfg.encoder.entity_transformer.key_size = entity_size
    cfg.encoder.entity_transformer.value_size = entity_size
    cfg.encoder.entity_transformer.model_size = entity_size
    cfg.encoder.entity_transformer.num_heads = num_transformer_heads
    cfg.encoder.entity_transformer.use_layer_norm = use_layer_norm
    cfg.encoder.entity_transformer.x_need_pos = True
    cfg.encoder.entity_transformer.y_need_pos = True
    cfg.encoder.entity_transformer.use_spectral_linear = use_spectral_linear
    cfg.encoder.entity_transformer.resblocks_hidden_size = transformer_hidden_size

    cfg.encoder.timestep_transformer.num_layers = num_transformer_layers
    cfg.encoder.timestep_transformer.key_size = entity_size
    cfg.encoder.timestep_transformer.value_size = entity_size
    cfg.encoder.timestep_transformer.model_size = entity_size
    cfg.encoder.timestep_transformer.num_heads = num_transformer_heads
    cfg.encoder.timestep_transformer.use_layer_norm = use_layer_norm
    cfg.encoder.timestep_transformer.x_need_pos = True
    cfg.encoder.timestep_transformer.y_need_pos = True
    cfg.encoder.timestep_transformer.use_spectral_linear = use_spectral_linear
    cfg.encoder.timestep_transformer.resblocks_hidden_size = transformer_hidden_size

    cfg.encoder.entity_timestep_transformer.num_layers = num_transformer_layers
    cfg.encoder.entity_timestep_transformer.key_size = entity_size
    cfg.encoder.entity_timestep_transformer.value_size = entity_size
    cfg.encoder.entity_timestep_transformer.model_size = entity_size
    cfg.encoder.entity_timestep_transformer.num_heads = num_transformer_heads
    cfg.encoder.entity_timestep_transformer.use_layer_norm = use_layer_norm
    cfg.encoder.entity_timestep_transformer.x_need_pos = False
    cfg.encoder.entity_timestep_transformer.y_need_pos = True
    cfg.encoder.entity_timestep_transformer.use_spectral_linear = use_spectral_linear
    cfg.encoder.entity_timestep_transformer.resblocks_hidden_size = (
        transformer_hidden_size
    )

    cfg.encoder.action_entity_transformer.num_layers = num_transformer_layers
    cfg.encoder.action_entity_transformer.key_size = entity_size
    cfg.encoder.action_entity_transformer.value_size = entity_size
    cfg.encoder.action_entity_transformer.model_size = entity_size
    cfg.encoder.action_entity_transformer.num_heads = num_transformer_heads
    cfg.encoder.action_entity_transformer.use_layer_norm = use_layer_norm
    cfg.encoder.action_entity_transformer.use_spectral_linear = use_spectral_linear
    cfg.encoder.action_entity_transformer.resblocks_hidden_size = (
        transformer_hidden_size
    )

    cfg.encoder.contextual_entity_agg.units_hidden_sizes = (entity_size, vector_size)
    cfg.encoder.contextual_entity_agg.use_layer_norm = use_layer_norm
    cfg.encoder.contextual_entity_agg.pool_method = PoolMethod.MEAN

    cfg.encoder.contextual_timestep_agg.units_hidden_sizes = (entity_size, vector_size)
    cfg.encoder.contextual_timestep_agg.use_layer_norm = use_layer_norm
    cfg.encoder.contextual_timestep_agg.pool_method = PoolMethod.MEAN

    cfg.encoder.contextual_action_agg.units_hidden_sizes = (entity_size, vector_size)
    cfg.encoder.contextual_action_agg.use_layer_norm = use_layer_norm
    cfg.encoder.contextual_action_agg.pool_method = PoolMethod.MEAN

    cfg.encoder.average_contextual_entity_resnet.num_resblocks = 0
    cfg.encoder.average_contextual_entity_resnet.use_layer_norm = use_layer_norm

    cfg.encoder.average_contextual_timestep_resnet.num_resblocks = 0
    cfg.encoder.average_contextual_timestep_resnet.use_layer_norm = use_layer_norm

    cfg.encoder.average_contextual_action_resnet.num_resblocks = 0
    cfg.encoder.average_contextual_action_resnet.use_layer_norm = use_layer_norm

    cfg.encoder.state_merge.output_size = vector_size
    cfg.encoder.state_merge.gating_type = GatingType.POINTWISE
    cfg.encoder.state_merge.use_layer_norm = use_layer_norm

    cfg.encoder.state_resnet.num_resblocks = 2
    cfg.encoder.state_resnet.use_layer_norm = use_layer_norm

    # Policy Head Configuration
    cfg.policy_head = ConfigDict()
    cfg.policy_head.transformer = ConfigDict()
    cfg.policy_head.logits = ConfigDict()

    cfg.policy_head.transformer.num_layers = num_transformer_layers
    cfg.policy_head.transformer.key_size = entity_size
    cfg.policy_head.transformer.value_size = entity_size
    cfg.policy_head.transformer.model_size = entity_size
    cfg.policy_head.transformer.num_heads = num_transformer_heads
    cfg.policy_head.transformer.use_layer_norm = use_layer_norm
    cfg.policy_head.transformer.use_spectral_linear = use_spectral_linear
    cfg.policy_head.transformer.resblocks_hidden_size = transformer_hidden_size

    cfg.policy_head.logits.num_logits = 1
    cfg.policy_head.logits.num_linear_layers = 1
    cfg.policy_head.logits.use_layer_norm = use_layer_norm

    # Value Head Configuration
    cfg.value_head = ConfigDict()
    cfg.value_head.transformer = ConfigDict()
    cfg.value_head.logits = ConfigDict()

    cfg.value_head.transformer.num_layers = num_transformer_layers
    cfg.value_head.transformer.key_size = entity_size
    cfg.value_head.transformer.value_size = entity_size
    cfg.value_head.transformer.model_size = entity_size
    cfg.value_head.transformer.num_heads = num_transformer_heads
    cfg.value_head.transformer.use_layer_norm = use_layer_norm
    cfg.value_head.transformer.use_spectral_linear = use_spectral_linear
    cfg.value_head.transformer.resblocks_hidden_size = transformer_hidden_size

    cfg.value_head.logits.num_logits = 1
    cfg.value_head.logits.num_linear_layers = 1
    cfg.value_head.logits.use_layer_norm = use_layer_norm

    return add_name_recursive(cfg)


def main():
    cfg = get_model_cfg()
    pprint.pprint(cfg)


if __name__ == "__main__":
    main()
