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

    entity_size = 256
    vector_size = 1024
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
    cfg.encoder.entity_edge_cross_transformer = ConfigDict()
    cfg.encoder.aggregate_nodes_resnet = ConfigDict()
    cfg.encoder.aggregate_edges_resnet = ConfigDict()
    cfg.encoder.side_field_resnet = ConfigDict()
    cfg.encoder.timestep_merge = ConfigDict()
    cfg.encoder.timestep_resnet = ConfigDict()
    cfg.encoder.entity_aggregator = ConfigDict()
    cfg.encoder.edge_aggregator = ConfigDict()
    cfg.encoder.timestep_transformer = ConfigDict()
    cfg.encoder.entity_transformer = ConfigDict()
    cfg.encoder.entity_timestep_cross_transformer = ConfigDict()
    cfg.encoder.action_transformer = ConfigDict()
    cfg.encoder.action_entity_cross_transformer = ConfigDict()
    cfg.encoder.contextual_entity_aggregator = ConfigDict()
    cfg.encoder.contextual_timestep_aggregator = ConfigDict()
    cfg.encoder.average_contextual_entity_resnet = ConfigDict()
    cfg.encoder.average_contextual_timestep_resnet = ConfigDict()
    cfg.encoder.state_merge = ConfigDict()
    cfg.encoder.state_resnet = ConfigDict()

    cfg.encoder.entity_encoder.entity_size = entity_size
    cfg.encoder.edge_encoder.entity_size = entity_size
    cfg.encoder.move_encoder.entity_size = entity_size
    cfg.encoder.side_condition_encoder.entity_size = entity_size // 4
    cfg.encoder.field_encoder.entity_size = entity_size // 4

    cfg.encoder.entity_edge_cross_transformer.num_layers = 1
    cfg.encoder.entity_edge_cross_transformer.key_size = entity_size // 2
    cfg.encoder.entity_edge_cross_transformer.value_size = entity_size // 2
    cfg.encoder.entity_edge_cross_transformer.model_size = entity_size
    cfg.encoder.entity_edge_cross_transformer.num_heads = 2
    cfg.encoder.entity_edge_cross_transformer.use_layer_norm = use_layer_norm
    cfg.encoder.entity_edge_cross_transformer.x_need_pos = False
    cfg.encoder.entity_edge_cross_transformer.y_need_pos = True
    cfg.encoder.entity_edge_cross_transformer.use_spectral_linear = use_spectral_linear
    cfg.encoder.entity_edge_cross_transformer.resblocks_hidden_size = entity_size // 2

    cfg.encoder.entity_aggregator.units_hidden_sizes = (entity_size, entity_size)
    cfg.encoder.entity_aggregator.use_layer_norm = use_layer_norm
    cfg.encoder.entity_aggregator.pool_method = PoolMethod.MEAN

    cfg.encoder.edge_aggregator.units_hidden_sizes = (entity_size, entity_size)
    cfg.encoder.edge_aggregator.use_layer_norm = use_layer_norm
    cfg.encoder.edge_aggregator.pool_method = PoolMethod.MEAN

    cfg.encoder.aggregate_nodes_resnet.num_resblocks = 0
    cfg.encoder.aggregate_nodes_resnet.use_layer_norm = use_layer_norm

    cfg.encoder.aggregate_edges_resnet.num_resblocks = 0
    cfg.encoder.aggregate_edges_resnet.use_layer_norm = use_layer_norm

    cfg.encoder.side_field_resnet.num_resblocks = 2
    cfg.encoder.side_field_resnet.use_layer_norm = use_layer_norm

    cfg.encoder.timestep_merge.output_size = entity_size
    cfg.encoder.timestep_merge.gating_type = GatingType.POINTWISE
    cfg.encoder.timestep_merge.use_layer_norm = use_layer_norm

    cfg.encoder.timestep_resnet.num_resblocks = 2
    cfg.encoder.timestep_resnet.use_layer_norm = use_layer_norm

    num_transformer_layers = 1
    num_transformer_heads = 2

    cfg.encoder.timestep_transformer.num_layers = num_transformer_layers
    cfg.encoder.timestep_transformer.key_size = entity_size // 2
    cfg.encoder.timestep_transformer.value_size = entity_size // 2
    cfg.encoder.timestep_transformer.model_size = entity_size
    cfg.encoder.timestep_transformer.num_heads = num_transformer_heads
    cfg.encoder.timestep_transformer.use_layer_norm = use_layer_norm
    cfg.encoder.timestep_transformer.need_pos = True
    cfg.encoder.timestep_transformer.use_spectral_linear = use_spectral_linear
    cfg.encoder.timestep_transformer.resblocks_hidden_size = entity_size // 2

    cfg.encoder.entity_transformer.num_layers = num_transformer_layers
    cfg.encoder.entity_transformer.key_size = entity_size // 2
    cfg.encoder.entity_transformer.value_size = entity_size // 2
    cfg.encoder.entity_transformer.model_size = entity_size
    cfg.encoder.entity_transformer.num_heads = num_transformer_heads
    cfg.encoder.entity_transformer.use_layer_norm = use_layer_norm
    cfg.encoder.entity_transformer.need_pos = False
    cfg.encoder.entity_transformer.use_spectral_linear = use_spectral_linear
    cfg.encoder.entity_transformer.resblocks_hidden_size = entity_size // 2

    cfg.encoder.entity_timestep_cross_transformer.num_layers = num_transformer_layers
    cfg.encoder.entity_timestep_cross_transformer.key_size = entity_size // 2
    cfg.encoder.entity_timestep_cross_transformer.value_size = entity_size // 2
    cfg.encoder.entity_timestep_cross_transformer.model_size = entity_size
    cfg.encoder.entity_timestep_cross_transformer.num_heads = num_transformer_heads
    cfg.encoder.entity_timestep_cross_transformer.use_layer_norm = use_layer_norm
    cfg.encoder.entity_timestep_cross_transformer.x_need_pos = False
    cfg.encoder.entity_timestep_cross_transformer.y_need_pos = True
    cfg.encoder.entity_timestep_cross_transformer.use_spectral_linear = (
        use_spectral_linear
    )
    cfg.encoder.entity_timestep_cross_transformer.resblocks_hidden_size = (
        entity_size // 2
    )

    cfg.encoder.action_transformer.num_layers = num_transformer_layers
    cfg.encoder.action_transformer.key_size = entity_size // 2
    cfg.encoder.action_transformer.value_size = entity_size // 2
    cfg.encoder.action_transformer.model_size = entity_size
    cfg.encoder.action_transformer.num_heads = num_transformer_heads
    cfg.encoder.action_transformer.use_layer_norm = use_layer_norm
    cfg.encoder.action_transformer.need_pos = False
    cfg.encoder.action_transformer.use_spectral_linear = use_spectral_linear
    cfg.encoder.action_transformer.resblocks_hidden_size = entity_size // 2

    cfg.encoder.action_entity_cross_transformer.num_layers = num_transformer_layers
    cfg.encoder.action_entity_cross_transformer.key_size = entity_size // 2
    cfg.encoder.action_entity_cross_transformer.value_size = entity_size // 2
    cfg.encoder.action_entity_cross_transformer.model_size = entity_size
    cfg.encoder.action_entity_cross_transformer.num_heads = num_transformer_heads
    cfg.encoder.action_entity_cross_transformer.use_layer_norm = use_layer_norm
    cfg.encoder.action_entity_cross_transformer.x_need_pos = False
    cfg.encoder.action_entity_cross_transformer.y_need_pos = False
    cfg.encoder.action_entity_cross_transformer.use_spectral_linear = (
        use_spectral_linear
    )
    cfg.encoder.action_entity_cross_transformer.resblocks_hidden_size = entity_size // 2

    cfg.encoder.contextual_entity_aggregator.units_hidden_sizes = (
        entity_size,
        vector_size,
    )
    cfg.encoder.contextual_entity_aggregator.use_layer_norm = use_layer_norm
    cfg.encoder.contextual_entity_aggregator.pool_method = PoolMethod.MEAN

    cfg.encoder.contextual_timestep_aggregator.units_hidden_sizes = (
        entity_size,
        vector_size,
    )
    cfg.encoder.contextual_timestep_aggregator.use_layer_norm = use_layer_norm
    cfg.encoder.contextual_timestep_aggregator.pool_method = PoolMethod.MEAN

    cfg.encoder.average_contextual_entity_resnet.num_resblocks = 0
    cfg.encoder.average_contextual_entity_resnet.use_layer_norm = use_layer_norm

    cfg.encoder.average_contextual_timestep_resnet.num_resblocks = 0
    cfg.encoder.average_contextual_timestep_resnet.use_layer_norm = use_layer_norm

    cfg.encoder.state_merge.output_size = vector_size
    cfg.encoder.state_merge.gating_type = GatingType.POINTWISE
    cfg.encoder.state_merge.use_layer_norm = use_layer_norm

    cfg.encoder.state_resnet.num_resblocks = 2
    cfg.encoder.state_resnet.use_layer_norm = use_layer_norm

    # Policy Head Configuration
    cfg.policy_head = ConfigDict()
    cfg.policy_head.key_size = entity_size

    cfg.policy_head.query = ConfigDict()
    cfg.policy_head.query.num_resblocks = 2
    cfg.policy_head.query.use_layer_norm = use_layer_norm

    cfg.policy_head.pointer_logits = ConfigDict()
    cfg.policy_head.pointer_logits.num_layers_query = 1
    cfg.policy_head.pointer_logits.num_layers_keys = 2
    cfg.policy_head.pointer_logits.key_size = 64
    cfg.policy_head.pointer_logits.use_layer_norm = use_layer_norm
    cfg.policy_head.entity_size = entity_size
    cfg.policy_head.vector_size = vector_size

    # Value Head Configuration
    cfg.value_head = ConfigDict()
    cfg.value_head.resnet = ConfigDict()
    cfg.value_head.resnet.num_resblocks = 2
    cfg.value_head.resnet.use_layer_norm = use_layer_norm

    cfg.value_head.logits = ConfigDict()
    cfg.value_head.logits.num_logits = 1
    cfg.value_head.logits.use_layer_norm = use_layer_norm

    return add_name_recursive(cfg)


def main():
    cfg = get_model_cfg()
    pprint.pprint(cfg)


if __name__ == "__main__":
    main()
